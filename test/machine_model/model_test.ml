open Graph_ir
module F = Native_test.Graph_fixtures
module M = Machine_model.Mir_model
module Attention_site = Machine_model.Attention_site
module Sdpa_select = Machine_model.Sdpa_select

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

(* A NaN's sign is the host's: [0/0] is negative on x86-64 and positive on
   AArch64, while the RV32 targets produce the canonical positive NaN. Compare
   every NaN as one value. *)
let bits (Tensor.Tensor t as packed) =
  let acc = ref [] in
  Vec6.iter t.Tensor.shape (fun c ->
      let x = Tensor.read packed c in
      acc :=
        Int64.bits_of_float (if Float.is_nan x then Float.nan else x) :: !acc);
  List.rev !acc

let bundle g =
  Err.or_raise ~pp_error:Loop_ir.Loop_bundle.pp_error
    (Loop_ir.Loop_bundle.build g)

let sig_of (g : graph) id = Tensor_id.Map.find id g.Graph.tensors

(* One context, [calls] calls with different inputs, each against the
   reference evaluator. *)
(* What an engine must equal: the binary64 per-node reference, or, for a
   binary32 policy, the Loop interpreter at binary32 over the bundle, its
   single-product sums fused for [Fp32_contracted]. *)
type oracle =
  | Custom of
      (graph ->
      (Tensor_id.t * Tensor.packed) list ->
      (Tensor_id.t * Tensor.packed) list ->
      Tensor.packed Tensor_id.Map.t)
      (** the answer from the constants and the inputs, by a caller's own rule:
          one the Loop interpreter has no accumulation for *)
  | Fp32
  | Fp32_contracted
  | Reference_f64

let expected oracle (g : graph) (b : Loop_ir.Loop_bundle.t) ~constants ~inputs =
  let loop accumulate =
    let outs =
      Err.or_raise ~pp_error:Loop_ir.Loop_bundle_oracle.pp_error
        (Loop_ir.Loop_bundle_oracle.run ~accumulate b
           ~constants:(fun id -> List.assoc_opt id constants)
           ~inputs:(fun id -> List.assoc_opt id inputs))
    in
    List.fold_left2
      (fun m id t -> Tensor_id.Map.add id t m)
      Tensor_id.Map.empty g.Graph.outputs outs
  in
  match oracle with
  | Custom f -> f g constants inputs
  | Fp32 -> loop Loop_ir.Loop_sum.Accumulate.Add
  | Fp32_contracted -> loop Loop_ir.Loop_sum.Accumulate.Fuse_products
  | Reference_f64 ->
      Err.or_raise ~pp_error:Eval_direct.pp_error
        (Eval_direct.run g ~constants ~inputs)

(* Two tensors cell for cell, a NaN equal to any NaN: for a host whose
   default NaN differs in sign from the oracle's. *)
let same_portable a b =
  List.for_all2
    (fun x y ->
      Core.Float_bits.equal_portable (Int64.float_of_bits x)
        (Int64.float_of_bits y))
    (bits a) (bits b)

let check ?route ?contractions ?bf16_pairs ?attention ?prepack
    ?(pipeline = Ssa_backends.Pipeline.Exact) ?(oracle = Reference_f64)
    ?(same = fun a b -> bits a = bits b) ?(calls = 2) ?constant
    ?(input = fun ~salt _ -> values ~salt) name g =
  let g = g () in
  let b = bundle g in
  let constant = Option.value constant ~default:(values ~salt:3) in
  let constants =
    List.map
      (fun id -> (id, constant (sig_of g id)))
      b.Loop_ir.Loop_bundle.constants
  in
  match
    M.prepare ?route ?contractions ?bf16_pairs ?attention ?prepack ~pipeline b
  with
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
                let reference = expected oracle g b ~constants ~inputs in
                match
                  M.Context.run cx ~inputs:(fun id -> List.assoc_opt id inputs)
                with
                | Error s -> Fmt.str "%a" M.Stop.pp s
                | Ok outs ->
                    if
                      List.for_all2
                        (fun id t -> same t (Tensor_id.Map.find id reference))
                        g.Graph.outputs outs
                    then "bitwise"
                    else "DIFFERS")
          in
          Fmt.pr "%s (%d invocations%s): %s@." name (M.invocations m)
            (match prepack with
            | Some true -> Fmt.str ", %d packed" (M.packed m)
            | Some false | None -> "")
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
  let planned numerics =
    Ssa_backends.Pipeline.Planned
      { numerics; target = Ssa_ir.Ssa_target.neon128 }
  in
  let show ?blocking pipeline =
    match M.prepare ?blocking ~pipeline b with
    | Error rs -> List.iter (fun r -> Fmt.pr "%a@." M.Refusal.pp r) rs
    | Ok _ -> Fmt.pr "prepared@."
  in
  (* a planned pipeline is lowered with the plan's own summary *)
  show (planned Ssa_ir.Ssa_numerics.Reference_f64);
  show (planned Ssa_ir.Ssa_numerics.Simd_fp32_ordered);
  (* blocking is the exact pipeline's *)
  show ~blocking:Machine_model.Mir_blocking.Policy.Feedback
    (planned Ssa_ir.Ssa_numerics.Simd_fp32_ordered);
  [%expect
    {|
    prepared
    prepared
    invocation 0 (n0): blocking needs the exact pipeline
    invocation 1 (n1): blocking needs the exact pipeline
    invocation 2 (n2): blocking needs the exact pipeline |}]

let fp32_ordered =
  Ssa_backends.Pipeline.Planned
    {
      numerics = Ssa_ir.Ssa_numerics.Fp32_ordered;
      target = Ssa_ir.Ssa_target.scalar;
    }

let%expect_test "Fp32_ordered: generic Machine IR is the binary32 oracle" =
  let run name g =
    check ~pipeline:fp32_ordered ~oracle:Fp32 ~constant:positive ~calls:3 name g
  in
  run "chain" F.chain;
  run "wide chain" wide_chain;
  run "layer_norm over W, C" (fun () ->
      F.build "layer_norm"
        Graph_builder.(
          let* x = input ~shape:(F.s 1 1 1 2 4 5) () in
          layer_norm
            { Norm.LayerNorm.dims = [ Axis.W; Axis.C ]; eps = 1e-5 }
            ~x ()));
  [%expect
    {|
    chain (3 invocations): bitwise, bitwise, bitwise
    wide chain (3 invocations): bitwise, bitwise, bitwise
    layer_norm over W, C (1 invocations): bitwise, bitwise, bitwise |}]

(* The failing invocation and its row under [Fp32_ordered], beside the binary32
   oracle's: both name invocation 1 and the offending index. *)
let%expect_test "Fp32_ordered: a failing gather reports the oracle's row" =
  let g = gather () in
  let b = bundle g in
  let m = Result.get_ok (M.prepare ~pipeline:fp32_ordered b) in
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
    let lookup id = List.assoc_opt id inputs in
    let oracle =
      match
        Err.payload
          (Loop_ir.Loop_bundle_oracle.run b
             ~constants:(fun _ -> None)
             ~inputs:lookup)
      with
      | Ok [ t ] -> Ok t
      | Ok _ -> Error "outputs?"
      | Error e -> Error (Fmt.str "%a" Loop_ir.Loop_bundle_oracle.pp_error e)
    in
    let mir = M.Context.run cx ~inputs:lookup in
    Fmt.pr "index [%s]: %s | oracle %s@."
      (String.concat "; " (List.map Int64.to_string idx))
      (match (mir, oracle) with
      | Ok [ t ], Ok o -> if bits t = bits o then "bitwise" else "DIFFERS"
      | Ok _, _ -> "outputs?"
      | Error s, _ -> Fmt.str "%a" M.Stop.pp s)
      (match oracle with Ok _ -> "succeeds" | Error e -> "fails: " ^ e)
  in
  call [ 2L; -3L ];
  call [ 0L; 3L ];
  call [ -4L; 1L ];
  [%expect
    {|
    index [2; -3]: bitwise | oracle succeeds
    index [0; 3]: invocation 1 (n1): failure gather_index_out_of_range(3:i64, 3:i64) | oracle fails: invocation 1: gather index 3 out of range [-3, 2]
    index [-4; 1]: invocation 1 (n1): failure gather_index_out_of_range(-4:i64, 3:i64) | oracle fails: invocation 1: gather index -4 out of range [-3, 2] |}]

(* ---- Fp32_ordered with vectors ------------------------------------------------

   The same policy planned for neon128: loops vectorize on the exact binary32
   form. Bundles wide enough for sixteen binary32 lanes, on the generic
   interpreter and on both targets' scanned programs in the physical
   interpreter, each bitwise the binary32 oracle; the vector loops the plans
   hold, and the binary64 census of the generic programs, which must be zero
   in every role. The softmax and layer_norm kernels hold a local array per
   output, which has no vector form: they stay scalar, under every policy. *)

let fp32_vector =
  Ssa_backends.Pipeline.Planned
    {
      numerics = Ssa_ir.Ssa_numerics.Fp32_ordered;
      target = Ssa_ir.Ssa_target.neon128;
    }

let vector_fixtures =
  [
    ( "relu then add over C=32",
      fun () ->
        F.build "pointwise"
          Graph_builder.(
            let* x = input ~shape:(F.s 1 1 1 2 3 32) () in
            let* y = relu x in
            add y x) );
    ( "softmax over H, W=32",
      fun () ->
        F.build "softmax"
          Graph_builder.(
            let* x = input ~shape:(F.s 1 1 1 5 32 3) () in
            softmax { Reduce.Softmax.axis = Axis.H } x) );
    ( "layer_norm over H, W=32",
      fun () ->
        F.build "layer_norm"
          Graph_builder.(
            let* x = input ~shape:(F.s 1 1 1 5 3 32) () in
            layer_norm { Norm.LayerNorm.dims = [ Axis.H ]; eps = 1e-5 } ~x ())
    );
    ( "conv, batch norm, relu to C=32",
      fun () ->
        F.build "wide_conv"
          Graph_builder.(
            let* x = input ~shape:(F.nhwc ~h:4 ~w:4 ~c:4) () in
            let* w =
              constant
                ~shape:(F.weight_shape ~out_channels:32 ~in_channels:4)
                ()
            in
            let* bias = constant ~shape:(F.s1c 32) () in
            let* gamma = constant ~shape:(F.s1c 32) () in
            let* beta = constant ~shape:(F.s1c 32) () in
            let* mean = constant ~shape:(F.s1c 32) () in
            let* var = constant ~shape:(F.s1c 32) () in
            let* y =
              conv2d (F.conv_params ~in_channels:4) ~x ~weight:w ~bias ()
            in
            let* n =
              batch_norm F.bn_params ~x:y ~weight:gamma ~bias:beta
                ~running_mean:mean ~running_var:var ()
            in
            relu n) );
  ]

let vector_loops b =
  List.fold_left
    (fun n inv ->
      match
        Ssa_backends.plan ~numerics:Ssa_ir.Ssa_numerics.Fp32_ordered
          ~target:Ssa_ir.Ssa_target.neon128 inv
      with
      | Ok p ->
          n
          + List.length
              (List.filter
                 (fun (d : Ssa_ir.Ssa_vectorize.Decision.t) ->
                   d.Ssa_ir.Ssa_vectorize.Decision.outcome
                   = Ssa_ir.Ssa_vectorize.Decision.Vectorized)
                 p.Ssa_ir.Ssa_plan.vectorized)
      | Error _ -> n)
    0 b.Loop_ir.Loop_bundle.invocations

let%expect_test "Fp32_ordered vectors: Machine IR is the binary32 oracle" =
  List.iter
    (fun (name, g) ->
      let b = bundle (g ()) in
      let census =
        match M.prepare ~pipeline:fp32_vector b with
        | Error _ -> "refused"
        | Ok m ->
            let cs =
              List.map Machine_ir.Mir_f64_census.of_program (M.generic m)
            in
            String.concat ", "
              (List.map
                 (fun r ->
                   Fmt.str "%s %d"
                     (Machine_ir.Mir_f64_census.Role.name r)
                     (List.fold_left
                        (fun n c -> n + Machine_ir.Mir_f64_census.count c r)
                        0 cs))
                 Machine_ir.Mir_f64_census.Role.all)
      in
      Fmt.pr "%s: %d vector loops; binary64 %s@." name (vector_loops b) census;
      List.iter
        (fun (rname, route) ->
          check ?route ~pipeline:fp32_vector ~oracle:Fp32 ~constant:positive
            ("  " ^ rname) g)
        [
          ("generic", None);
          ("aarch64 scanned", Some (M.Route.Aarch64 M.Stage.Scanned));
          ("x86_64 scanned", Some (M.Route.X86_64 M.Stage.Scanned));
        ])
    vector_fixtures;
  [%expect
    {|
    relu then add over C=32: 2 vector loops; binary64 arithmetic 0, conversion 0, helper_call 0, memory_move 0
      generic (2 invocations): bitwise, bitwise
      aarch64 scanned (2 invocations): bitwise, bitwise
      x86_64 scanned (2 invocations): bitwise, bitwise
    softmax over H, W=32: 0 vector loops; binary64 arithmetic 0, conversion 0, helper_call 0, memory_move 0
      generic (1 invocations): bitwise, bitwise
      aarch64 scanned (1 invocations): bitwise, bitwise
      x86_64 scanned (1 invocations): bitwise, bitwise
    layer_norm over H, W=32: 0 vector loops; binary64 arithmetic 0, conversion 0, helper_call 0, memory_move 0
      generic (1 invocations): bitwise, bitwise
      aarch64 scanned (1 invocations): bitwise, bitwise
      x86_64 scanned (1 invocations): bitwise, bitwise
    conv, batch norm, relu to C=32: 3 vector loops; binary64 arithmetic 0, conversion 0, helper_call 0, memory_move 0
      generic (3 invocations): bitwise, bitwise
      aarch64 scanned (3 invocations): bitwise, bitwise
      x86_64 scanned (3 invocations): bitwise, bitwise |}]

(* ---- matrix products ---------------------------------------------------------

   [Linear] and [Bmm] at the Coral peak RVV FP32 kernel's shape (C[9,32],
   K=64) and with M/N/K tails (M=7, N=29, K=63), with and without a bias, at
   batch 1 and 3. An empty K has no fixture: an extent is at least 1
   ([Dim.extent]), so no graph names one. *)

let linear_fixture ~m ~n ~k ~bias () =
  F.build "linear"
    Graph_builder.(
      let* x = input ~shape:(F.s 1 1 1 1 m k) () in
      let* weight = constant ~shape:(F.s n 1 1 1 1 k) () in
      let params = { Linear.Linear.in_features = Dim.extent k } in
      if bias then
        let* b = constant ~shape:(F.s1c n) () in
        linear params ~x ~weight ~bias:b ()
      else linear params ~x ~weight ())

(* [linear_fixture] with its input and weight bfloat16, [bmm_fixture] with
   both inputs. *)
let linear_bf16_fixture ~m ~n ~k ~bias () =
  let bf16 = Payload.Fmt Payload.BF16 in
  F.build "linear_bf16"
    Graph_builder.(
      let* x = input ~shape:(F.s 1 1 1 1 m k) ~fmt:bf16 () in
      let* weight = constant ~shape:(F.s n 1 1 1 1 k) ~fmt:bf16 () in
      let params = { Linear.Linear.in_features = Dim.extent k } in
      if bias then
        let* b = constant ~shape:(F.s1c n) () in
        linear params ~x ~weight ~bias:b ()
      else linear params ~x ~weight ())

(* [t]'s values stored in [sg]'s format: bfloat16 (each value rounded to
   it) for a bfloat16 tensor, else [t] itself. *)
let in_format (sg : Tensor_sig.t) t =
  match sg.Tensor_sig.fmt with
  | Payload.Fmt Payload.BF16 ->
      let shape = sg.Tensor_sig.shape in
      let data =
        Bigarray.(
          Array1.create int16_unsigned c_layout (Vec6.numel shape :> int))
      in
      Vec6.iter shape (fun c ->
          data.{(Vec6.offset shape c :> int)} <-
            Half.Bf16.of_float (Tensor.read t c));
      Tensor.Tensor
        {
          shape;
          payload =
            { Payload.fmt = Payload.BF16; quant = Payload.No_quant; data };
        }
  | Payload.Fmt _ -> t

let bmm_bf16_fixture ~batch ~m ~n ~k () =
  let bf16 = Payload.Fmt Payload.BF16 in
  F.build "bmm_bf16"
    Graph_builder.(
      let* a = input ~shape:(F.s 1 1 1 batch m k) ~fmt:bf16 () in
      let* b = input ~shape:(F.s 1 1 1 batch k n) ~fmt:bf16 () in
      bmm a b)

(* [Sdpa] over bfloat16 Q, K, V (default scale), with a binary32 mask of
   [mask] = [(rows, cols)] when given: [rows] and [cols] are each [tq] / [tk]
   or 1 (a broadcast axis). *)
let sdpa_bf16_fixture ~tq ~tk ~d ?mask () =
  let bf16 = Payload.Fmt Payload.BF16 in
  F.build "sdpa_bf16"
    Graph_builder.(
      let* q = input ~shape:(F.s 1 1 1 1 tq d) ~fmt:bf16 () in
      let* k = input ~shape:(F.s 1 1 1 1 tk d) ~fmt:bf16 () in
      let* v = input ~shape:(F.s 1 1 1 1 tk d) ~fmt:bf16 () in
      let params = { Attention.Sdpa.scale = Attention.Sdpa.Scale.Default } in
      match mask with
      | None -> sdpa params ~query:q ~key:k ~value:v ()
      | Some (rows, cols) ->
          let* m = input ~shape:(F.s 1 1 1 1 rows cols) () in
          sdpa params ~query:q ~key:k ~value:v ~mask:m ())

let bf16_fixtures =
  [
    ("bmm bf16 3x7x29 k63", bmm_bf16_fixture ~batch:3 ~m:7 ~n:29 ~k:63);
    ("linear bf16 9x32 k64", linear_bf16_fixture ~m:9 ~n:32 ~k:64 ~bias:false);
    ( "linear bf16 7x29 k63 bias",
      linear_bf16_fixture ~m:7 ~n:29 ~k:63 ~bias:true );
  ]

let bmm_fixture ~batch ~m ~n ~k () =
  F.build "bmm"
    Graph_builder.(
      let* a = input ~shape:(F.s 1 1 1 batch m k) () in
      let* b = input ~shape:(F.s 1 1 1 batch k n) () in
      bmm a b)

let contraction_fixtures =
  [
    ("linear 9x32 k64", linear_fixture ~m:9 ~n:32 ~k:64 ~bias:false);
    ("linear 9x32 k64 bias", linear_fixture ~m:9 ~n:32 ~k:64 ~bias:true);
    ("linear 7x29 k63 bias", linear_fixture ~m:7 ~n:29 ~k:63 ~bias:true);
    ("bmm 1x9x32 k64", bmm_fixture ~batch:1 ~m:9 ~n:32 ~k:64);
    ("bmm 3x7x29 k63", bmm_fixture ~batch:3 ~m:7 ~n:29 ~k:63);
  ]

(* A 1x1 convolution over an [h] x [w] image of [cin] channels to [cout]: a
   matrix product whose rows are two loops (with [cin] 1, a sum over a
   single index); a [pad] or [stride] other than 0
   or 1 makes windows the recognizer must refuse. *)
let pointwise_fixture ~pad ~stride ~h ~w ~cin ~cout () =
  let axis = F.conv_axis ~kernel:1 ~stride ~pad in
  F.build "pointwise"
    Graph_builder.(
      let* x = input ~shape:(F.nhwc ~h ~w ~c:cin) () in
      let* weight = constant ~shape:(F.s cout 1 1 1 1 cin) () in
      let* bias = constant ~shape:(F.s1c cout) () in
      conv2d
        {
          Conv.Conv2d.h = axis;
          w = axis;
          in_channels = Dim.extent cin;
          groups = Op_config.Pos.of_int 1;
        }
        ~x ~weight ~bias ())

let pointwise_fixtures =
  [
    ( "conv1x1 3x5 c19 to 29",
      pointwise_fixture ~pad:0 ~stride:1 ~h:3 ~w:5 ~cin:19 ~cout:29 );
    ( "conv1x1 3x5 c1 to 4",
      pointwise_fixture ~pad:0 ~stride:1 ~h:3 ~w:5 ~cin:1 ~cout:4 );
    ( "conv1x1 2x2 c8 to 4 pad 1",
      pointwise_fixture ~pad:1 ~stride:1 ~h:2 ~w:2 ~cin:8 ~cout:4 );
    ( "conv1x1 4x4 c8 to 4 stride 2",
      pointwise_fixture ~pad:0 ~stride:2 ~h:4 ~w:4 ~cin:8 ~cout:4 );
  ]

(* A matrix-product fixture's multiply-adds: its output's cells times the
   contraction extent, its first input's C. *)
let contraction_macs g =
  let g = g () in
  let b = bundle g in
  let shape id = (sig_of g id).Tensor_sig.shape in
  (Vec6.numel (shape (List.hd g.Graph.outputs)) :> int)
  * (Vec6.get (shape (List.hd b.Loop_ir.Loop_bundle.inputs)) Axis.C :> int)

(* Binary32 values whose sums round, cancel, overflow and underflow: mostly
   inexact fractions of varied magnitude, with cancelling pairs, values whose
   products are near the binary32 maximum, subnormals and, with [nonfinite],
   infinities and NaN. Deterministic in the cell and [salt]. *)
let adversarial ?(nonfinite = true) ~salt (sg : Tensor_sig.t) =
  Tensor.materialize sg.Tensor_sig.shape (fun c ->
      let k = (Vec6.offset sg.Tensor_sig.shape c :> int) + (salt * 7919) in
      let h = k * 2654435761 land 0x3fffffff in
      F32_math.round32
        (match (h mod 1021, h mod 61) with
        | 0, _ when nonfinite -> Float.infinity
        | 1, _ when nonfinite -> Float.nan
        | _, 0 -> 1.5e19
        | _, 1 -> -1.7e19
        | _, 2 -> 1e8
        | _, 3 -> -1e8
        | _, 4 -> 1e-40
        | _, 5 -> -3e-39
        | _, r -> float_of_int ((h / 61 mod 2001) - 1000) /. float_of_int (r + 3)))

let adversarial_input ~salt _ sg = adversarial ~salt sg

(* ---- binary32 transcendentals as vectors -------------------------------------

   [exp], [log], [sin], [cos] and [erf] (through the exact GELU) over the
   {!F32_math} sample set: the special values, then a stratified sample of the
   binary32 line. Planned for the forced neon128 target, so every loop takes
   its vector form and the sequence runs as one vector instance; each lane must
   equal the binary32 oracle (the Loop interpreter at binary32, whose functions
   are {!F32_math.Value}). *)

let fp32_vector_forced =
  Ssa_backends.Pipeline.Planned
    {
      numerics = Ssa_ir.Ssa_numerics.Fp32_ordered;
      target = Ssa_ir.Ssa_target.forced Ssa_ir.Ssa_target.neon128;
    }

let f32_math_samples =
  let specials =
    [|
      0.;
      -0.;
      1.;
      -1.;
      Float.infinity;
      Float.neg_infinity;
      Float.nan;
      1e-45;
      -1e-45;
      3e38;
      -3e38;
      88.7;
      89.;
      -103.9;
      -104.5;
      0.5;
      -0.5;
      2.5;
      1e-38;
      100.;
      -100.;
      8191.;
      3.14159265;
      -1.5707963;
    |]
  in
  Array.init 256 (fun k ->
      if k < Array.length specials then F32_math.round32 specials.(k)
      else
        (* every (2^32 / 232)-th bit pattern, the finite ones *)
        let bits =
          Int64.mul (Int64.of_int (k - Array.length specials)) 18512790L
        in
        let x = Int32.float_of_bits (Int64.to_int32 bits) in
        if Float.is_finite x then x else 0.75)

let sample_input ~salt:_ _ (sg : Tensor_sig.t) =
  Tensor.materialize sg.Tensor_sig.shape (fun c ->
      f32_math_samples.((Vec6.offset sg.Tensor_sig.shape c :> int) mod 256))

let transcendental_fixtures =
  let unary name f () =
    F.build name
      Graph_builder.(
        let* x = input ~shape:(F.s 1 1 1 1 4 64) () in
        f x)
  in
  [
    ("exp", unary "exp" (fun x -> Graph_builder.exp x));
    ("log", unary "log" (fun x -> Graph_builder.log x));
    ("sin", unary "sin" (fun x -> Graph_builder.sin x));
    ("cos", unary "cos" (fun x -> Graph_builder.cos x));
    ( "gelu (erf)",
      unary "gelu" (fun x -> Graph_builder.gelu Pointwise.Gelu.Exact x) );
  ]

let%expect_test "binary32 transcendentals as Machine IR vectors" =
  List.iter
    (fun (name, g) ->
      List.iter
        (fun (rname, route) ->
          check ?route ~pipeline:fp32_vector_forced ~oracle:Fp32 ~calls:1
            ~input:sample_input
            (Fmt.str "%s %s" name rname)
            g)
        [
          ("generic", None);
          ("aarch64 scanned", Some (M.Route.Aarch64 M.Stage.Scanned));
          ("x86_64 scanned", Some (M.Route.X86_64 M.Stage.Scanned));
        ])
    transcendental_fixtures;
  [%expect
    {|
    exp generic (1 invocations): bitwise
    exp aarch64 scanned (1 invocations): bitwise
    exp x86_64 scanned (1 invocations): bitwise
    log generic (1 invocations): bitwise
    log aarch64 scanned (1 invocations): bitwise
    log x86_64 scanned (1 invocations): bitwise
    sin generic (1 invocations): bitwise
    sin aarch64 scanned (1 invocations): bitwise
    sin x86_64 scanned (1 invocations): bitwise
    cos generic (1 invocations): bitwise
    cos aarch64 scanned (1 invocations): bitwise
    cos x86_64 scanned (1 invocations): bitwise
    gelu (erf) generic (1 invocations): bitwise
    gelu (erf) aarch64 scanned (1 invocations): bitwise
    gelu (erf) x86_64 scanned (1 invocations): bitwise |}]

let sdpa_bf16_fixtures =
  [
    ("sdpa bf16 32x32 d32", sdpa_bf16_fixture ~tq:32 ~tk:32 ~d:32 ?mask:None);
    ("sdpa bf16 64x64 d64", sdpa_bf16_fixture ~tq:64 ~tk:64 ~d:64 ?mask:None);
    ( "sdpa bf16 9x29 d64 masked",
      sdpa_bf16_fixture ~tq:9 ~tk:29 ~d:64 ~mask:(9, 29) );
  ]

(* The per-op composition of an [Sdpa]: scores [bmm (Q, K^T)] over bfloat16
   operands (the first product, [K^T] given as an input), the scale, a
   binary32 softmax down the keys, and [bmm (P, V)] with [V] widened exactly
   (the second product reads binary32 operands: the engine multiplies no mixed
   pair). It prices today's route at sizes where every invocation fits the
   core; it is not the streaming kernel's arithmetic. *)
let sdpa_composition_bf16_fixture ~tq ~tk ~d () =
  let bf16 = Payload.Fmt Payload.BF16 in
  F.build "sdpa_composition_bf16"
    Graph_builder.(
      let* q = input ~shape:(F.s 1 1 1 1 tq d) ~fmt:bf16 () in
      let* kt = input ~shape:(F.s 1 1 1 1 d tk) ~fmt:bf16 () in
      let* v = input ~shape:(F.s 1 1 1 1 tk d) ~fmt:bf16 () in
      let* s = bmm q kt in
      let* z = mul_scalar (1. /. Stdlib.sqrt (float_of_int d)) s in
      let* p = softmax { Reduce.Softmax.axis = Axis.C } z in
      let* vf = to_copy Pointwise.To_copy.Float v in
      bmm p vf)

(* The composition's second half alone: binary32 scores in, the context out. *)
let sdpa_composition_tail_fixture ~tq ~tk ~d () =
  let bf16 = Payload.Fmt Payload.BF16 in
  F.build "sdpa_composition_tail"
    Graph_builder.(
      let* s = input ~shape:(F.s 1 1 1 1 tq tk) () in
      let* v = input ~shape:(F.s 1 1 1 1 tk d) ~fmt:bf16 () in
      let* z = mul_scalar (1. /. Stdlib.sqrt (float_of_int d)) s in
      let* p = softmax { Reduce.Softmax.axis = Axis.C } z in
      let* vf = to_copy Pointwise.To_copy.Float v in
      bmm p vf)

(* bfloat16 Q, K, V widen on load: the binary64 reference and the binary32
   Loop oracle both run on the fixtures the streaming kernel will take, and
   agree to binary32 accuracy. *)
let%expect_test "bfloat16 Sdpa: the reference and the binary32 oracle run" =
  List.iter
    (fun (name, g) ->
      let g = g () in
      let b = bundle g in
      let inputs =
        List.mapi
          (fun i id ->
            let sg = sig_of g id in
            ( id,
              in_format sg
                ( adversarial_input ~salt:i i sg |> fun t ->
                  if i = 3 then t else values ~salt:i sg ) ))
          b.Loop_ir.Loop_bundle.inputs
      in
      let run oracle =
        expected oracle g b ~constants:[] ~inputs |> fun m ->
        List.concat_map
          (fun id ->
            List.map Int64.float_of_bits (bits (Tensor_id.Map.find id m)))
          g.Graph.outputs
      in
      let r = run Reference_f64 and f = run Fp32 in
      let worst =
        List.fold_left2 (fun w x y -> Float.max w (Float.abs (x -. y))) 0. r f
      in
      Fmt.pr "%s: %d cells, reference vs binary32 oracle within 1e-5: %b@." name
        (List.length r) (worst < 1e-5))
    sdpa_bf16_fixtures;
  [%expect
    {|
    sdpa bf16 32x32 d32: 1024 cells, reference vs binary32 oracle within 1e-5: true
    sdpa bf16 64x64 d64: 4096 cells, reference vs binary32 oracle within 1e-5: true
    sdpa bf16 9x29 d64 masked: 576 cells, reference vs binary32 oracle within 1e-5: true |}]
