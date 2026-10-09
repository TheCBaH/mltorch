(* Whole bundles on the CPU: the model tests' graphs with every invocation
   compiled to an image of its own, each call against the reference
   evaluator. *)

module F = Native_test.Graph_fixtures
module T = Machine_model_test.Model_test
module Rt = Machine_rivet_aarch64.Rivet_a64_route

let route = Rt.route ()

let%expect_test "conv, batch norm, relu natively" =
  T.check ~route ~constant:T.positive ~calls:3 "chain" F.chain;
  [%expect {| chain (3 invocations): bitwise, bitwise, bitwise |}]

let%expect_test "wider chain, and both pipelines" =
  List.iter
    (fun pipeline ->
      T.check ~route ~constant:T.positive ~calls:3
        ("chain " ^ Ssa_backends.Pipeline.name pipeline)
        ~pipeline F.chain)
    [ Ssa_backends.Pipeline.Representation; Ssa_backends.Pipeline.Exact ];
  T.check ~route ~constant:T.positive "wide chain exact" T.wide_chain;
  [%expect
    {|
    chain representation (3 invocations): bitwise, bitwise, bitwise
    chain exact (3 invocations): bitwise, bitwise, bitwise
    wide chain exact (3 invocations): bitwise, bitwise |}]

let%expect_test "Region nodes: locals, scans and the meter in a context" =
  T.check ~route "softmax over C" (fun () ->
      F.build "softmax"
        Graph_builder.(
          let* x = input ~shape:(F.s 1 1 2 3 4 5) () in
          softmax { Reduce.Softmax.axis = Axis.C } x));
  T.check ~route "layer_norm over W, C" (fun () ->
      F.build "layer_norm"
        Graph_builder.(
          let* x = input ~shape:(F.s 1 1 1 2 4 5) () in
          layer_norm
            { Norm.LayerNorm.dims = [ Axis.W; Axis.C ]; eps = 1e-5 }
            ~x ()));
  T.check ~route "rms_norm over C" (fun () ->
      F.build "rms_norm"
        Graph_builder.(
          let* x = input ~shape:(F.s 1 1 1 3 4 5) () in
          rms_norm { Norm.RmsNorm.dims = [ Axis.C ]; eps = 1e-5 } ~x ()));
  T.check ~route "sdpa, masked rows"
    (fun () ->
      F.build "sdpa_mask"
        Graph_builder.(
          let* q = input ~shape:(F.s 1 1 2 3 4 5) () in
          let* k = input ~shape:(F.s 1 1 2 3 6 5) () in
          let* v = input ~shape:(F.s 1 1 2 3 6 5) () in
          let* m = input ~shape:(F.s 1 1 2 3 4 6) () in
          sdpa
            { Attention.Sdpa.scale = Attention.Sdpa.Scale.Default }
            ~query:q ~key:k ~value:v ~mask:m ()))
    ~input:(fun ~salt i sg ->
      if i = 3 then
        Tensor.materialize sg.Tensor_sig.shape (fun c ->
            if
              (Vec6.get c Axis.W :> int) = 1
              || (salt mod 2 = 1 && (Vec6.get c Axis.C :> int) = 0)
            then neg_infinity
            else 0.)
      else T.values ~salt sg);
  [%expect
    {|
    softmax over C (1 invocations): bitwise, bitwise
    layer_norm over W, C (1 invocations): bitwise, bitwise
    rms_norm over C (1 invocations): bitwise, bitwise
    sdpa, masked rows (1 invocations): bitwise, bitwise |}]

let%expect_test "bmm and a padded, strided convolution" =
  T.check ~route "bmm" (fun () ->
      F.build "bmm"
        Graph_builder.(
          let* a = input ~shape:(F.s 1 1 1 2 5 7) () in
          let* b = input ~shape:(F.s 1 1 1 2 7 3) () in
          bmm a b));
  T.check ~route ~constant:T.positive "conv 3x3 stride 2 pad 1" (fun () ->
      F.build "conv"
        Graph_builder.(
          let* x = input ~shape:(F.s 2 1 1 7 7 3) () in
          let* w = constant ~shape:(F.s 4 1 1 3 3 3) () in
          let* bias = constant ~shape:(F.s1c 4) () in
          let axis = F.conv_axis ~kernel:3 ~stride:2 ~pad:1 in
          conv2d
            {
              Conv.Conv2d.h = axis;
              w = axis;
              in_channels = Dim.extent 3;
              groups = Op_config.Pos.of_int 1;
            }
            ~x ~weight:w ~bias ()));
  [%expect
    {|
    bmm (1 invocations): bitwise, bitwise
    conv 3x3 stride 2 pad 1 (1 invocations): bitwise, bitwise |}]

(* The first failing invocation names the record the CPU stored, and the
   context goes on to the next call. *)
let%expect_test
    "the first failing invocation, natively, then the context goes on" =
  let g = T.gather () in
  let b = T.bundle g in
  let m =
    Result.get_ok (T.M.prepare ~route ~pipeline:Ssa_backends.Pipeline.Exact b)
  in
  let cx = Result.get_ok (T.M.Context.create m ~constants:(fun _ -> None)) in
  let self_id, index_id =
    match b.Loop_ir.Loop_bundle.inputs with
    | [ s; i ] -> (s, i)
    | _ -> assert false
  in
  let call idx =
    let inputs =
      [
        (self_id, T.values ~salt:1 (T.sig_of g self_id));
        ( index_id,
          Tensor.materialize_i64 (F.s 1 1 1 1 1 2) (fun c ->
              List.nth idx (Vec6.get c Axis.C :> int)) );
      ]
    in
    Fmt.pr "index [%s]: %s@."
      (String.concat "; " (List.map Int64.to_string idx))
      (match
         T.M.Context.run cx ~inputs:(fun id -> List.assoc_opt id inputs)
       with
      | Ok _ -> "succeeds"
      | Error s -> Fmt.str "%a" T.M.Stop.pp s)
  in
  call [ 2L; -3L ];
  call [ 0L; 3L ];
  call [ -4L; 1L ];
  call [ 1L; 1L ];
  [%expect
    {|
    index [2; -3]: succeeds
    index [0; 3]: invocation 1 (n1): failure gather_index_out_of_range(3:i64, 3:i64)
    index [-4; 1]: invocation 1 (n1): failure gather_index_out_of_range(-4:i64, 3:i64)
    index [1; 1]: succeeds |}]

(* The same bundles with the split linear-scan allocator behind sink
   scheduling: another physical program, the same answers. *)
let%expect_test "scanned allocation natively" =
  let route = Rt.route ~allocation:Rt.Allocation.Scanned () in
  T.check ~route ~constant:T.positive ~calls:3 "chain" F.chain;
  T.check ~route ~constant:T.positive "wide chain exact" T.wide_chain;
  T.check ~route "bmm" (fun () ->
      F.build "bmm"
        Graph_builder.(
          let* a = input ~shape:(F.s 1 1 1 2 5 7) () in
          let* b = input ~shape:(F.s 1 1 1 2 7 3) () in
          bmm a b));
  T.check ~route "softmax over C" (fun () ->
      F.build "softmax"
        Graph_builder.(
          let* x = input ~shape:(F.s 1 1 2 3 4 5) () in
          softmax { Reduce.Softmax.axis = Axis.C } x));
  [%expect
    {|
    chain (3 invocations): bitwise, bitwise, bitwise
    wide chain exact (3 invocations): bitwise, bitwise
    bmm (1 invocations): bitwise, bitwise
    softmax over C (1 invocations): bitwise, bitwise |}]

(* A mapping defect leaves a bundle that compiles, loads and runs. *)
let%expect_test "mapping mutations change a bundle's answer" =
  let module Mu = Machine_rivet_aarch64.Rivet_a64_form.Mutation in
  List.iter
    (fun (name, mutation) ->
      T.check ~route:(Rt.route ~mutation ()) ~constant:T.positive ~calls:1 name
        F.chain)
    [ ("dropped lo12", Mu.Dropped_lo12); ("commuted sub", Mu.Commuted_sub) ];
  [%expect
    {|
    dropped lo12 (3 invocations): DIFFERS
    commuted sub (3 invocations): DIFFERS |}]
