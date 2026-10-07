open Graph_ir
module F = Native_test.Graph_fixtures
module M = Machine_model.Mir_model

(* Values that vary in sign and size, so a max, a denominator and a mask all
   matter; [salt] makes a different call's inputs differ. *)
let values ~salt (sg : Tensor_sig.t) =
  Tensor.materialize sg.Tensor_sig.shape (fun c ->
      let k = (Vec6.offset sg.Tensor_sig.shape c :> int) + salt in
      float_of_int ((k mod 9) - 4) /. 2.)

let positive (sg : Tensor_sig.t) =
  Tensor.materialize sg.Tensor_sig.shape (fun c ->
      0.25
      +. 0.05
         *. float_of_int ((Vec6.offset sg.Tensor_sig.shape c :> int) mod 11))

let bits (Tensor.Tensor t as packed) =
  let acc = ref [] in
  Vec6.iter t.Tensor.shape (fun c ->
      acc := Int64.bits_of_float (Tensor.read packed c) :: !acc);
  List.rev !acc

let bundle g =
  Err.or_raise ~pp_error:Loop_ir.Loop_bundle.pp_error
    (Loop_ir.Loop_bundle.build g)

let sig_of (g : graph) id = Tensor_id.Map.find id g.Graph.tensors

(* One context, [calls] calls with different inputs, each against the
   reference evaluator. *)
let check ?(pipeline = Ssa_backends.Pipeline.Exact) ?(calls = 2) ?constant
    ?(input = fun ~salt _ -> values ~salt) name g =
  let g = g () in
  let b = bundle g in
  let constant = Option.value constant ~default:(values ~salt:3) in
  let constants =
    List.map
      (fun id -> (id, constant (sig_of g id)))
      b.Loop_ir.Loop_bundle.constants
  in
  match M.prepare ~pipeline b with
  | Error rs ->
      Fmt.pr "%s: refused: %a@." name Fmt.(list ~sep:(any "; ") M.Refusal.pp) rs
  | Ok m -> (
      match
        M.Context.create m ~constants:(fun id -> List.assoc_opt id constants)
      with
      | Error s -> Fmt.pr "%s: %a@." name M.Stop.pp s
      | Ok cx ->
          let verdicts =
            List.init calls (fun call ->
                let inputs =
                  List.mapi
                    (fun i id -> (id, input ~salt:(call + i) i (sig_of g id)))
                    b.Loop_ir.Loop_bundle.inputs
                in
                let reference =
                  Err.or_raise ~pp_error:Eval_direct.pp_error
                    (Eval_direct.run g ~constants ~inputs)
                in
                match
                  M.Context.run cx ~inputs:(fun id -> List.assoc_opt id inputs)
                with
                | Error s -> Fmt.str "%a" M.Stop.pp s
                | Ok outs ->
                    if
                      List.for_all2
                        (fun id t ->
                          bits t = bits (Tensor_id.Map.find id reference))
                        g.Graph.outputs outs
                    then "bitwise"
                    else "DIFFERS")
          in
          Fmt.pr "%s (%d invocations): %s@." name (M.invocations m)
            (String.concat ", " verdicts))

let wide_chain () =
  F.build "wide_chain"
    Graph_builder.(
      let* x = input ~shape:(F.nhwc ~h:6 ~w:6 ~c:4) () in
      let* w =
        constant ~shape:(F.weight_shape ~out_channels:8 ~in_channels:4) ()
      in
      let* bias = constant ~shape:(F.s1c 8) () in
      let* gamma = constant ~shape:(F.s1c 8) () in
      let* beta = constant ~shape:(F.s1c 8) () in
      let* mean = constant ~shape:(F.s1c 8) () in
      let* var = constant ~shape:(F.s1c 8) () in
      let* y = conv2d (F.conv_params ~in_channels:4) ~x ~weight:w ~bias () in
      let* n =
        batch_norm F.bn_params ~x:y ~weight:gamma ~bias:beta ~running_mean:mean
          ~running_var:var ()
      in
      relu n)

let%expect_test "conv, batch norm, relu: bitwise, call after call" =
  List.iter
    (fun pipeline ->
      check ~constant:positive ~calls:3
        ("chain " ^ Ssa_backends.Pipeline.name pipeline)
        ~pipeline F.chain)
    [ Ssa_backends.Pipeline.Representation; Ssa_backends.Pipeline.Exact ];
  check ~constant:positive "wide chain exact" wide_chain;
  [%expect
    {|
    chain representation (3 invocations): bitwise, bitwise, bitwise
    chain exact (3 invocations): bitwise, bitwise, bitwise
    wide chain exact (3 invocations): bitwise, bitwise |}]

let%expect_test "Region nodes: locals, scans and the meter in a context" =
  check "softmax over C" (fun () ->
      F.build "softmax"
        Graph_builder.(
          let* x = input ~shape:(F.s 1 1 2 3 4 5) () in
          softmax { Reduce.Softmax.axis = Axis.C } x));
  check "layer_norm over W, C" (fun () ->
      F.build "layer_norm"
        Graph_builder.(
          let* x = input ~shape:(F.s 1 1 1 2 4 5) () in
          layer_norm
            { Norm.LayerNorm.dims = [ Axis.W; Axis.C ]; eps = 1e-5 }
            ~x ()));
  check "rms_norm over C" (fun () ->
      F.build "rms_norm"
        Graph_builder.(
          let* x = input ~shape:(F.s 1 1 1 3 4 5) () in
          rms_norm { Norm.RmsNorm.dims = [ Axis.C ]; eps = 1e-5 } ~x ()));
  check "sdpa, masked rows"
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
      else values ~salt sg);
  [%expect
    {|
    softmax over C (1 invocations): bitwise, bitwise
    layer_norm over W, C (1 invocations): bitwise, bitwise
    rms_norm over C (1 invocations): bitwise, bitwise
    sdpa, masked rows (1 invocations): bitwise, bitwise |}]

(* relu, then a gather by a runtime index, then relu: the gather is the
   second invocation, and an index outside [-3, 3) fails there. *)
let gather () =
  Graph_builder.(
    build ~name:"gather" ~outputs:(fun r -> [ r ])
    @@
    let* self = input ~shape:(F.s 1 1 1 2 3 2) () in
    let* index =
      input ~shape:(F.s 1 1 1 1 1 2) ~fmt:(Payload.Fmt Payload.I64) ()
    in
    let* r = relu self in
    let* g =
      index_tensor
        { Index_tensor.Index_tensor.axis = Axis.W; index_rank = Rank.of_int 1 }
        ~self:r ~index
    in
    relu g)
  |> Result.get_ok

let%expect_test "the first failing invocation, then the context goes on" =
  let g = gather () in
  let b = bundle g in
  let m = Result.get_ok (M.prepare ~pipeline:Ssa_backends.Pipeline.Exact b) in
  let cx = Result.get_ok (M.Context.create m ~constants:(fun _ -> None)) in
  let self_id, index_id =
    match b.Loop_ir.Loop_bundle.inputs with
    | [ s; i ] -> (s, i)
    | _ -> assert false
  in
  let call idx =
    let inputs =
      [
        (self_id, values ~salt:1 (sig_of g self_id));
        ( index_id,
          Tensor.materialize_i64 (F.s 1 1 1 1 1 2) (fun c ->
              List.nth idx (Vec6.get c Axis.C :> int)) );
      ]
    in
    let reference =
      match Eval_direct.run g ~constants:[] ~inputs with
      | r -> (
          match Err.payload r with
          | Ok env -> Ok (Tensor_id.Map.find (List.hd g.Graph.outputs) env)
          | Error e -> Error (Fmt.str "%a" Eval_direct.pp_error e))
      | exception Err.Exn.E e -> Error (Fmt.str "%a" Err.Exn.pp_kind e)
    in
    let mir = M.Context.run cx ~inputs:(fun id -> List.assoc_opt id inputs) in
    Fmt.pr "index [%s]: %s | reference %s@."
      (String.concat "; " (List.map Int64.to_string idx))
      (match mir with
      | Ok [ t ] -> (
          match reference with
          | Ok r -> if bits t = bits r then "bitwise" else "DIFFERS"
          | Error _ -> "succeeds")
      | Ok _ -> "outputs?"
      | Error s -> Fmt.str "%a" M.Stop.pp s)
      (match reference with Ok _ -> "succeeds" | Error e -> "fails: " ^ e)
  in
  call [ 2L; -3L ];
  call [ 0L; 3L ];
  call [ -4L; 1L ];
  call [ 1L; 1L ];
  [%expect
    {|
    index [2; -3]: bitwise | reference succeeds
    index [0; 3]: invocation 1 (n1): failure gather_index_out_of_range(3:i64, 3:i64) | reference fails: gather index 3 out of range [-3, 2]
    index [-4; 1]: invocation 1 (n1): failure gather_index_out_of_range(-4:i64, 3:i64) | reference fails: gather index -4 out of range [-3, 2]
    index [1; 1]: bitwise | reference succeeds |}]

let%expect_test "independent contexts, and a comparison that can fail" =
  let g = F.chain () in
  let b = bundle g in
  let m = Result.get_ok (M.prepare ~pipeline:Ssa_backends.Pipeline.Exact b) in
  let constants salt =
    List.map
      (fun id -> (id, values ~salt (sig_of g id)))
      b.Loop_ir.Loop_bundle.constants
  in
  let inputs salt =
    List.map
      (fun id -> (id, values ~salt (sig_of g id)))
      b.Loop_ir.Loop_bundle.inputs
  in
  let context salt =
    Result.get_ok
      (M.Context.create m ~constants:(fun id ->
           List.assoc_opt id (constants salt)))
  in
  let a = context 3 and c = context 5 in
  let run cx ~cs ~is =
    let reference =
      Err.or_raise ~pp_error:Eval_direct.pp_error
        (Eval_direct.run g ~constants:(constants cs) ~inputs:(inputs is))
    in
    match
      M.Context.run cx ~inputs:(fun id -> List.assoc_opt id (inputs is))
    with
    | Ok [ t ] ->
        if
          bits t = bits (Tensor_id.Map.find (List.hd g.Graph.outputs) reference)
        then "bitwise"
        else "DIFFERS"
    | Ok _ -> "outputs?"
    | Error s -> Fmt.str "%a" M.Stop.pp s
  in
  Fmt.pr "a: %s; c: %s; a: %s@." (run a ~cs:3 ~is:0) (run c ~cs:5 ~is:1)
    (run a ~cs:3 ~is:2);
  (* against the other context's constants, the same comparison fails *)
  Fmt.pr "a against c's constants: %s@." (run a ~cs:5 ~is:0);
  (* an input the call lacks *)
  (match M.Context.run a ~inputs:(fun _ -> None) with
  | Error s -> Fmt.pr "%a@." M.Stop.pp s
  | Ok _ -> Fmt.pr "ran@.");
  [%expect
    {|
    a: bitwise; c: bitwise; a: bitwise
    a against c's constants: DIFFERS
    no tensor for t0 |}]

let%expect_test "every refusal is reported" =
  let b = bundle (F.chain ()) in
  (match
     M.prepare
       ~pipeline:
         (Ssa_backends.Pipeline.Planned
            {
              numerics = Ssa_ir.Ssa_numerics.Reference_f64;
              target = Ssa_ir.Ssa_target.neon128;
            })
       b
   with
  | Error rs -> List.iter (fun r -> Fmt.pr "%a@." M.Refusal.pp r) rs
  | Ok _ -> Fmt.pr "prepared@.");
  [%expect
    {|
    invocation 0 (n0): a planned pipeline is not admitted
    invocation 1 (n1): a planned pipeline is not admitted
    invocation 2 (n2): a planned pipeline is not admitted |}]
