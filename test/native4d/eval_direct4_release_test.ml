(* [Eval_direct4.run ~retain], the twin of test/native's
   eval_direct_release_test.ml: under [Only], outputs and errors are those of
   an [All] run, and the result holds exactly the outputs and the retained
   edges. The GC check is native-only and lives in test/native_gc. See .ai/
   (tensor release). *)

open Native4d

let only_empty = Release_schedule.Retain.Only Tensor_id.Set.empty

let run ~retain (g : Graph.graph) ~inputs ~constants =
  Eval_direct4.run ~retain ~constants g ~inputs

(* One line per graph: outputs bit-identical, or the same error. *)
let equivalent name (g : Graph.graph) ~inputs ~constants =
  let pp_err ppf e = Eval_direct4.pp_error ppf (Err.Error.kind e) in
  match
    ( run ~retain:All g ~inputs ~constants,
      run ~retain:only_empty g ~inputs ~constants )
  with
  | Ok all, Ok only ->
      let same id =
        Tensor.equal_bits
          (Tensor_id.Map.find id all)
          (Tensor_id.Map.find id only)
      in
      if List.for_all same g.Graph.Graph.outputs then
        Fmt.pr "%s: %d outputs equal@." name (List.length g.Graph.Graph.outputs)
      else Fmt.pr "%s: OUTPUTS DIFFER@." name
  | Error a, Error b ->
      let a = Fmt.str "%a" pp_err a and b = Fmt.str "%a" pp_err b in
      if String.equal a b then Fmt.pr "%s: same error: %s@." name a
      else Fmt.pr "%s: ERRORS DIFFER: %s / %s@." name a b
  | Ok _, Error e -> Fmt.pr "%s: ONLY FAILED: %a@." name pp_err e
  | Error e, Ok _ -> Fmt.pr "%s: ALL FAILED: %a@." name pp_err e

let%expect_test "Only empty: every per-op graph's outputs equal an All run's" =
  List.iter
    (fun (name, g, inputs, constants) -> equivalent name g ~inputs ~constants)
    (Fixtures4.per_op ());
  [%expect
    {|
    add: 1 outputs equal
    sub: 1 outputs equal
    mul: 1 outputs equal
    div: 1 outputs equal
    add_scalar: 1 outputs equal
    div_scalar: 1 outputs equal
    floor_div_scalar: 1 outputs equal
    eq_scalar: 1 outputs equal
    eq_tensor: 1 outputs equal
    gt_scalar: 1 outputs equal
    ne_scalar: 1 outputs equal
    ne_tensor: 1 outputs equal
    bitwise_not: 1 outputs equal
    expand4: 1 outputs equal
    mul_scalar: 1 outputs equal
    pow: 1 outputs equal
    rpow_scalar: 1 outputs equal
    cos: 1 outputs equal
    sin: 1 outputs equal
    rsub_scalar: 1 outputs equal
    clamp: 1 outputs equal
    col2im: 1 outputs equal
    hardtanh: 1 outputs equal
    im2col: 1 outputs equal
    leaky_relu: 1 outputs equal
    zeros4: 1 outputs equal
    arange4: 1 outputs equal
    eye4: 1 outputs equal
    batch_norm: 1 outputs equal
    batch_norm_no_stats: 3 outputs equal
    batched_matmul: 1 outputs equal
    addcmul: 1 outputs equal
    relu: 1 outputs equal
    repeat4: 1 outputs equal
    repeat_interleave4: 1 outputs equal
    gelu: 1 outputs equal
    sigmoid: 1 outputs equal
    silu: 1 outputs equal
    hardsigmoid: 1 outputs equal
    hardswish: 1 outputs equal
    sqrt: 1 outputs equal
    to_copy: 1 outputs equal
    max_pool2d: 1 outputs equal
    max_pool2d_with_indices: 2 outputs equal
    adaptive_avg_pool2d: 1 outputs equal
    adaptive_max_pool2d: 1 outputs equal
    adaptive_max_pool2d_with_indices: 2 outputs equal
    avg_pool2d: 1 outputs equal
    mean_keepdims: 1 outputs equal
    max_dim4: 2 outputs equal
    max_keepdims: 1 outputs equal
    sum_keepdims: 1 outputs equal
    pad4: 1 outputs equal
    slice4: 1 outputs equal
    softmax4: 1 outputs equal
    cumsum4: 1 outputs equal
    select4: 1 outputs equal
    select_scatter4: 1 outputs equal
    concat4: 1 outputs equal
    meshgrid: 2 outputs equal
    stack4: 1 outputs equal
    permute4: 1 outputs equal
    reshape4: 1 outputs equal
    rms_norm: 1 outputs equal
    layer_norm: 1 outputs equal
    sdpa: 1 outputs equal
    lstm: 3 outputs equal
    group_norm4: 1 outputs equal
    conv2d: 1 outputs equal
    depthwise_conv2d: 1 outputs equal
    grouped_conv2d: 1 outputs equal
    transposed_conv2d: 1 outputs equal
    unbind: 2 outputs equal
    split_with_sizes4: 2 outputs equal
    upsample_bicubic2d: 1 outputs equal
    upsample_bilinear2d: 1 outputs equal
    upsample_nearest2d: 1 outputs equal
    vector_norm_keepdims: 1 outputs equal
    index_tensor4: 1 outputs equal |}]

(* Deterministic, sign-varying, small, from the signature alone. *)
let tensor_of_sig (sg : Tensor_sig.t) =
  let v c =
    float_of_int (((Vec6.offset sg.Tensor_sig.shape c :> int) mod 7) - 3) /. 4.
  in
  match sg.Tensor_sig.fmt with
  | Payload.Fmt Payload.I64 ->
      Tensor.materialize_i64 sg.Tensor_sig.shape (fun c ->
          Int64.of_int ((Vec6.offset sg.Tensor_sig.shape c :> int) mod 2))
  | _ -> Tensor.materialize sg.Tensor_sig.shape v

(* Multi-node graphs: Native fixtures lowered to Native4D, including live index
   outputs and a multi-output Region. A discarded index does not lower (the
   dialect has no [Discard]); the next test builds its Native4D shape. *)
let lowered name native =
  let line =
    Lower_fixtures.described native ~render:(fun dst ->
        let bound kind =
          List.filter_map
            (fun id ->
              if Graph.input_kind dst id = kind then
                Some
                  ( id,
                    tensor_of_sig
                      (Tensor_id.Map.find id dst.Graph_common.Graph.tensors) )
              else None)
            dst.Graph_common.Graph.inputs
        in
        Fmt.str "%t" (fun _ ->
            equivalent name dst
              ~inputs:(bound Graph_ir.Input.Input)
              ~constants:(bound Graph_ir.Input.Constant)))
  in
  if not (String.equal line "") then Fmt.pr "%s: %s@." name line

let%expect_test "Only empty: lowered multi-node graphs equal an All run's" =
  List.iter
    (fun (name, g) -> lowered name g)
    Fixtures.
      [
        ("adaptive_maxpool_indices_live", adaptive_maxpool_indices_live ());
        ("bmm_batch", bmm_batch 1 ());
        ("expand", expand ());
        ("group_norm_tiny", group_norm_tiny ());
        ("layer_norm_tiny", layer_norm_tiny ());
        ("linear_layer", linear_layer ());
        ("lstm_states_live", lstm_states_live ());
        ("maxpool_indices_live", maxpool_indices_live ());
        ("repeat", repeat ());
        ("sdpa", sdpa 1 ());
        ("split_with_sizes_w_batch2", split_with_sizes_w_batch2 ());
        ("unbind_n", unbind_n ());
      ];
  [%expect
    {|
    adaptive_maxpool_indices_live: 1 outputs equal
    bmm_batch: 1 outputs equal
    expand: 1 outputs equal
    group_norm_tiny: 1 outputs equal
    layer_norm_tiny: 1 outputs equal
    linear_layer: 1 outputs equal
    lstm_states_live: 1 outputs equal
    maxpool_indices_live: 1 outputs equal
    repeat: 1 outputs equal
    sdpa: 1 outputs equal
    split_with_sizes_w_batch2: 2 outputs equal
    unbind_n: 2 outputs equal |}]

(* An index output nothing reads: never allocated, in either setting. *)
let%expect_test "Only empty: an unread index output" =
  let g =
    Fixtures4.build
      ~outputs:(fun o -> [ o ])
      (let open Builder in
       let* x = input ~shape:Fixtures4.nhwc () in
       let* outs = max_pool2d_with_indices Fixtures4.pool_params x in
       relu (List.hd outs))
  in
  equivalent "maxpool_index_unread" g
    ~inputs:[ (List.hd g.Graph.Graph.inputs, Fixtures4.seq Fixtures4.nhwc) ]
    ~constants:[];
  [%expect {| maxpool_index_unread: 1 outputs equal |}]

(* x -> relu -> sqrt -> relu. *)
let chain () =
  Fixtures4.build
    ~outputs:(fun o -> [ o ])
    (let open Builder in
     let* x = input ~shape:Fixtures4.flat () in
     let* a = relu x in
     let* b = sqrt a in
     relu b)

let%expect_test "the result holds exactly the outputs and the retained edges" =
  let g = chain () in
  let inputs =
    [ (List.hd g.Graph.Graph.inputs, Fixtures4.seq Fixtures4.flat) ]
  in
  let first = List.hd (List.hd g.Graph.Graph.nodes).Graph.Node.outputs in
  List.iter
    (fun retain ->
      match Err.payload (run ~retain g ~inputs ~constants:[]) with
      | Ok env ->
          Fmt.pr "%a@."
            Fmt.(brackets (list ~sep:sp Tensor_id.pp))
            (List.map fst (Tensor_id.Map.bindings env))
      | Error e -> Fmt.pr "error: %a@." Eval_direct4.pp_error e)
    [
      Release_schedule.Retain.All;
      only_empty;
      Only (Tensor_id.Set.singleton first);
      Only (Tensor_id.Set.singleton (List.hd g.Graph.Graph.inputs));
    ];
  [%expect {|
    [t0 t1 t2 t3]
    [t3]
    [t1 t3]
    [t0 t3] |}]
