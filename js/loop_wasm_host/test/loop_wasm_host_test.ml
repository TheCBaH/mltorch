open Graph_ir
open Loop_ir
module H = Loop_wasm_host

let tensor_of ~salt (sg : Tensor_sig.t) =
  let v c =
    float_of_int
      ((((Vec6.offset sg.Tensor_sig.shape c :> int) + salt) mod 7) - 3)
    /. 4.
  in
  Tensor.materialize sg.Tensor_sig.shape v

let bits t =
  let (Tensor.Tensor tt) = t in
  let acc = ref [] in
  Vec6.iter tt.Tensor.shape (fun c ->
      acc := Int32.bits_of_float (Tensor.read_at t (Vec6.get c)) :: !acc);
  List.rev !acc

let sig_of_edge (b : Loop_bundle.t) id =
  match Tensor_id.Map.find_opt id b.Loop_bundle.graph.Graph.tensors with
  | Some sg -> sg
  | None -> Fmt.failwith "no signature for %a" Tensor_id.pp id

let map_of l id = List.assoc_opt id l

let build g =
  Err.or_raise ~pp_error:Loop_bundle.pp_error
    (Loop_bundle.build ~config:Loop_bundle_wasm.default_config g)

let prepare g =
  let b = build g in
  let constants =
    List.map
      (fun id -> (id, tensor_of ~salt:3 (sig_of_edge b id)))
      b.Loop_bundle.constants
  in
  let m =
    Err.or_raise ~pp_error:H.pp_error
      (H.prepare b ~constants:(map_of constants))
  in
  (b, constants, m)

let check g m b constants ~salt =
  let inputs =
    List.map
      (fun id -> (id, tensor_of ~salt (sig_of_edge b id)))
      b.Loop_bundle.inputs
  in
  let reference =
    Err.or_raise ~pp_error:Eval_direct.pp_error
      (Eval_direct.run g ~constants ~inputs)
  in
  match Err.payload (H.run m ~bind:(map_of inputs)) with
  | Error e -> Fmt.pr "run failed: %a@." H.pp_error e
  | Ok outs ->
      List.iter2
        (fun id t ->
          Fmt.pr "output t%d identical to the reference: %b@."
            (Tensor_id.to_int id)
            (bits t = bits (Tensor_id.Map.find id reference)))
        g.Graph.outputs outs

let%expect_test "chain: run repeatedly on one instance, inputs changing" =
  let g = Native_test.Graph_fixtures.chain () in
  let b, constants, m = prepare g in
  List.iter (fun salt -> check g m b constants ~salt) [ 0; 2; 5; 0 ];
  [%expect
    {|
    output t9 identical to the reference: true
    output t9 identical to the reference: true
    output t9 identical to the reference: true
    output t9 identical to the reference: true |}]

let%expect_test "shared kernels, forwarded and repeated outputs, Region nodes" =
  let forwarded =
    Graph_builder.build ~name:"forwarded_input"
      ~outputs:(fun (x, o) -> [ x; o ])
      Graph_builder.(
        let* x = input ~shape:(Native_test.Graph_fixtures.s1c 4) () in
        let+ o = relu x in
        (x, o))
    |> Err.or_raise ~pp_error:Graph_builder.pp_error
  in
  let repeated =
    Graph_builder.build ~name:"repeated_output"
      ~outputs:(fun o -> [ o; o ])
      Graph_builder.(
        let* x = input ~shape:(Native_test.Graph_fixtures.s1c 4) () in
        relu x)
    |> Err.or_raise ~pp_error:Graph_builder.pp_error
  in
  List.iter
    (fun (name, g) ->
      Fmt.pr "%s@." name;
      let b, constants, m = prepare g in
      check g m b constants ~salt:1;
      check g m b constants ~salt:4)
    [
      ("residual", Native_test.Graph_fixtures.residual ());
      ("forwarded", forwarded);
      ("repeated", repeated);
      ("layer_norm", Native_test.Graph_fixtures.sink_permute_layer_norm ());
      ("sdpa", Native_test.Graph_fixtures.sink_permute_sdpa ());
    ];
  [%expect
    {|
    residual
    output t3 identical to the reference: true
    output t3 identical to the reference: true
    forwarded
    output t0 identical to the reference: true
    output t1 identical to the reference: true
    output t0 identical to the reference: true
    output t1 identical to the reference: true
    repeated
    output t1 identical to the reference: true
    output t1 identical to the reference: true
    output t1 identical to the reference: true
    output t1 identical to the reference: true
    layer_norm
    output t2 identical to the reference: true
    output t2 identical to the reference: true
    sdpa
    output t4 identical to the reference: true
    output t4 identical to the reference: true |}]

let%expect_test "a failing invocation is reported by position" =
  let g = Native_test.Graph_fixtures.chain () in
  let b0 = build g in
  let constants =
    map_of
      (List.map
         (fun id -> (id, tensor_of ~salt:3 (sig_of_edge b0 id)))
         b0.Loop_bundle.constants)
  in
  let bind =
    map_of
      (List.map
         (fun id -> (id, tensor_of ~salt:0 (sig_of_edge b0 id)))
         b0.Loop_bundle.inputs)
  in
  let always = Loop_bool.Index_eq (Loop_index.Const 0, Loop_index.Const 0) in
  let poisoned =
    {
      b0 with
      Loop_bundle.invocations =
        List.mapi
          (fun i (inv : Loop_bundle.invocation) ->
            if i <> 1 then inv
            else
              let p = inv.Loop_bundle.program in
              {
                inv with
                Loop_bundle.program =
                  {
                    p with
                    Loop_program.body =
                      Loop_stmt.Fail_if
                        (always, Loop_failure.I64_division_by_zero)
                      :: p.Loop_program.body;
                  };
              })
          b0.Loop_bundle.invocations;
    }
  in
  let m = Err.or_raise ~pp_error:H.pp_error (H.prepare poisoned ~constants) in
  (match Err.payload (H.run m ~bind) with
  | Error (`Inference_failed (i, e)) ->
      Fmt.pr "invocation %d, %a@." i Loop_interp.pp_error e
  | Error e -> Fmt.pr "%a@." H.pp_error e
  | Ok _ -> Fmt.pr "no failure@.");
  [%expect {| invocation 1, I64 division by zero |}]

let%expect_test "typed errors: missing input, disposed model" =
  let g = Native_test.Graph_fixtures.chain () in
  let b, constants, m = prepare g in
  ignore constants;
  (match Err.payload (H.run m ~bind:(fun _ -> None)) with
  | Error e -> Fmt.pr "%a@." H.pp_error e
  | Ok _ -> Fmt.pr "ran@.");
  H.dispose m;
  H.dispose m;
  let bind =
    map_of
      (List.map
         (fun id -> (id, tensor_of ~salt:0 (sig_of_edge b id)))
         b.Loop_bundle.inputs)
  in
  (match Err.payload (H.run m ~bind) with
  | Error e -> Fmt.pr "%a@." H.pp_error e
  | Ok _ -> Fmt.pr "ran@.");
  Fmt.pr "memory after dispose: %d@." (H.memory_bytes m);
  [%expect
    {|
    no value for input t0
    the model was disposed
    memory after dispose: 0 |}]

let%expect_test
    "an input of the wrong shape or format is refused before running" =
  let g = Native_test.Graph_fixtures.chain () in
  let b, constants, m = prepare g in
  ignore constants;
  let input_id = List.hd b.Loop_bundle.inputs in
  let expected = sig_of_edge b input_id in
  let wrong_shape =
    Vec6.shape ~n:1 ~t:1 ~d:1 ~h:1 ~w:1
      ~c:(Dim.to_int (Vec6.get expected.Tensor_sig.shape Expr.Axis.C) + 1)
  in
  let bad = Tensor.materialize wrong_shape (fun _ -> 0.) in
  (match
     Err.payload
       (H.run m ~bind:(fun id ->
            if Tensor_id.equal id input_id then Some bad else None))
   with
  | Error e -> Fmt.pr "%a@." H.pp_error e
  | Ok _ -> Fmt.pr "ran@.");
  (* The refusal leaves the instance usable. *)
  check g m b constants ~salt:0;
  [%expect
    {|
    t0: bound tensor has the wrong shape
    output t9 identical to the reference: true |}]

let%expect_test "feature detection validates the probe, and SIMD models run" =
  List.iter
    (fun f -> Fmt.pr "%s: %b@." (Wasm_features.name f) (H.supports f))
    Wasm_features.all;
  let g = Native_test.Graph_fixtures.chain () in
  let b = build g in
  let constants =
    List.map
      (fun id -> (id, tensor_of ~salt:3 (sig_of_edge b id)))
      b.Loop_bundle.constants
  in
  let m =
    Err.or_raise ~pp_error:H.pp_error
      (H.prepare
         ~vector:(Loop_target.forced Loop_target.wasm128)
         b ~constants:(map_of constants))
  in
  check g m b constants ~salt:1;
  Fmt.pr "needs simd128: %b@."
    (List.mem Wasm_features.Simd128
       (Wasm_features.of_module (H.bundle_wasm m).Loop_bundle_wasm.module_));
  [%expect
    {|
    bulk-memory: true
    nontrapping-float-to-int: true
    relaxed-simd: false
    sign-extension: true
    simd128: true
    output t9 identical to the reference: true
    needs simd128: true |}]
