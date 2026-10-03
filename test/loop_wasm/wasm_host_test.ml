open Graph_ir
open Loop_ir

(* The whole-model Wasm backend against the per-node reference evaluator: the
   same graph, constants and inputs, outputs compared bitwise, under node. *)

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

let with_dir f =
  let dir = Loop_c_exec.Proc.temp_dir "wasm_host_test" in
  Fun.protect
    ~finally:(fun () -> Loop_c_exec.Proc.remove_tree dir)
    (fun () -> f dir)

module H = Loop_wasm_exec.Host

let build g =
  Err.or_raise ~pp_error:Loop_bundle.pp_error
    (Loop_bundle.build ~config:Loop_bundle_wasm.default_config g)

let prepare ?(bundle = build) g dir =
  let b = bundle g in
  let constants =
    List.map
      (fun id -> (id, tensor_of ~salt:3 (sig_of_edge b id)))
      b.Loop_bundle.constants
  in
  let p =
    Err.or_raise ~pp_error:H.pp_error
      (H.prepare ~dir b ~constants:(map_of constants))
  in
  (b, constants, p)

let check ?(repeat = 0) g p b constants ~salt ~poison =
  let inputs =
    List.map
      (fun id -> (id, tensor_of ~salt (sig_of_edge b id)))
      b.Loop_bundle.inputs
  in
  let reference =
    Err.or_raise ~pp_error:Eval_direct.pp_error
      (Eval_direct.run g ~constants ~inputs)
  in
  match Err.payload (H.run ~poison ~repeat p ~bind:(map_of inputs)) with
  | Error e -> Fmt.pr "run failed: %a@." H.pp_error e
  | Ok outs ->
      List.iter2
        (fun id t ->
          Fmt.pr "output t%d identical to the reference: %b@."
            (Tensor_id.to_int id)
            (bits t = bits (Tensor_id.Map.find id reference)))
        g.Graph.outputs outs

let stats p =
  let st = (H.bundle_wasm p).Loop_bundle_wasm.stats in
  Fmt.pr "%d invocations, %d kernels@." st.Loop_bundle_wasm.invocations
    st.Loop_bundle_wasm.distinct_kernels

let%expect_test "chain: generated model matches the reference, repeatedly" =
  with_dir (fun dir ->
      let g = Native_test.Graph_fixtures.chain () in
      let b, constants, p = prepare g dir in
      check g p b constants ~salt:0 ~poison:false;
      check g p b constants ~salt:2 ~poison:true;
      check ~repeat:3 g p b constants ~salt:5 ~poison:true;
      stats p);
  [%expect
    {|
    output t9 identical to the reference: true
    output t9 identical to the reference: true
    output t9 identical to the reference: true
    3 invocations, 3 kernels |}]

let%expect_test "residual: two same-shape invocations share one kernel" =
  with_dir (fun dir ->
      let g = Native_test.Graph_fixtures.residual () in
      let b, constants, p = prepare g dir in
      check g p b constants ~salt:1 ~poison:true;
      stats p);
  [%expect
    {|
    output t3 identical to the reference: true
    3 invocations, 2 kernels |}]

let%expect_test "a graph output that is a graph input, and a repeated output" =
  with_dir (fun dir ->
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
      List.iteri
        (fun i g ->
          let sub = Filename.concat dir (string_of_int i) in
          let b, constants, p = prepare g sub in
          check g p b constants ~salt:4 ~poison:true)
        [ forwarded; repeated ]);
  [%expect
    {|
    output t0 identical to the reference: true
    output t1 identical to the reference: true
    output t1 identical to the reference: true
    output t1 identical to the reference: true |}]

let%expect_test "Region-authored nodes with synthetic defaults" =
  with_dir (fun dir ->
      List.iteri
        (fun i (name, g) ->
          Fmt.pr "%s@." name;
          let sub = Filename.concat dir (string_of_int i) in
          let b, constants, p = prepare g sub in
          check g p b constants ~salt:2 ~poison:true)
        [
          ("layer_norm", Native_test.Graph_fixtures.sink_permute_layer_norm ());
          ("sdpa", Native_test.Graph_fixtures.sink_permute_sdpa ());
        ]);
  [%expect
    {|
    layer_norm
    output t2 identical to the reference: true
    sdpa
    output t4 identical to the reference: true |}]

(* ---- inference failures carry the invocation, not the kernel --------------- *)

let poisoned_bundle b ~at failure =
  let invocations =
    List.mapi
      (fun i (inv : Loop_bundle.invocation) ->
        if i <> at then inv
        else
          let p = inv.Loop_bundle.program in
          let always =
            Loop_bool.Index_eq (Loop_index.Const 0, Loop_index.Const 0)
          in
          {
            inv with
            Loop_bundle.program =
              {
                p with
                Loop_program.body =
                  Loop_stmt.Fail_if (always, failure) :: p.Loop_program.body;
              };
          })
      b.Loop_bundle.invocations
  in
  { b with Loop_bundle.invocations }

let%expect_test
    "a failing invocation is reported by position, decoded like the interpreter"
    =
  with_dir (fun dir ->
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
      let buffer =
        List.hd
          (List.nth b0.Loop_bundle.invocations 2).Loop_bundle.program
            .Loop_program.buffers
      in
      let cases =
        [
          ("division by zero", 1, Loop_failure.I64_division_by_zero);
          ( "coordinate out of range",
            2,
            Loop_failure.Load_out_of_range
              {
                buffer;
                coord =
                  Expr.Coord.make ~n:(Loop_index.Const 0)
                    ~t:(Loop_index.Const 0) ~d:(Loop_index.Const 0)
                    ~h:(Loop_index.Const 0) ~w:(Loop_index.Const 99)
                    ~c:(Loop_index.Const 0);
              } );
        ]
      in
      List.iter
        (fun (name, at, failure) ->
          let b = poisoned_bundle b0 ~at failure in
          let sub = Filename.concat dir (string_of_int at) in
          let p =
            Err.or_raise ~pp_error:H.pp_error (H.prepare ~dir:sub b ~constants)
          in
          match Err.payload (H.run p ~bind) with
          | Error (`Inference_failed (i, e)) ->
              Fmt.pr "%s: invocation %d, %a@." name i Loop_interp.pp_error e
          | Error e -> Fmt.pr "%s: %a@." name H.pp_error e
          | Ok _ -> Fmt.pr "%s: no failure@." name)
        cases);
  [%expect
    {|
    division by zero: invocation 1, I64 division by zero
    coordinate out of range: invocation 2, t8[0,0,0,0,99,0] out of range on axis W: 99 |}]

let%expect_test "an unavailable node is a typed error, not a skip" =
  with_dir (fun dir ->
      let g = Native_test.Graph_fixtures.chain () in
      let b, constants, p = prepare g dir in
      ignore constants;
      let saved = !H.node in
      H.node := [ "node-that-does-not-exist" ];
      let inputs =
        map_of
          (List.map
             (fun id -> (id, tensor_of ~salt:0 (sig_of_edge b id)))
             b.Loop_bundle.inputs)
      in
      (match Err.payload (H.run p ~bind:inputs) with
      | Error (`Node_unavailable _) -> Fmt.pr "node unavailable@."
      | Error e -> Fmt.pr "other: %a@." H.pp_error e
      | Ok _ -> Fmt.pr "ran@.");
      H.node := saved);
  [%expect {| node unavailable |}]
