open Graph_ir
open Loop_ir

(* The baseline route: the C backend's whole-model unit compiled to Wasm with
   Clang against a wasm32 libc, run under node, against the per-node
   reference. A missing toolchain is a failure naming it, never a skip. *)

module V = Loop_wasm_exec.Via_c

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
  let dir = Loop_c_exec.Proc.temp_dir "wasm_c_host_test" in
  Fun.protect
    ~finally:(fun () -> Loop_c_exec.Proc.remove_tree dir)
    (fun () -> f dir)

let toolchain () =
  match V.toolchain_from_env () with
  | Ok t -> t
  | Error e -> Fmt.failwith "%a" V.pp_error e

let build g =
  Err.or_raise ~pp_error:Loop_bundle.pp_error
    (Loop_bundle.build ~config:Loop_bundle_c.default_config g)

let prepare g dir =
  let b = build g in
  let constants =
    List.map
      (fun id -> (id, tensor_of ~salt:3 (sig_of_edge b id)))
      b.Loop_bundle.constants
  in
  let p =
    Err.or_raise ~pp_error:V.pp_error
      (V.prepare ~toolchain:(toolchain ()) ~dir b ~constants:(map_of constants))
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
  match Err.payload (V.run ~poison ~repeat p ~bind:(map_of inputs)) with
  | Error e -> Fmt.pr "run failed: %a@." V.pp_error e
  | Ok outs ->
      List.iter2
        (fun id t ->
          Fmt.pr "output t%d identical to the reference: %b@."
            (Tensor_id.to_int id)
            (bits t = bits (Tensor_id.Map.find id reference)))
        g.Graph.outputs outs

let%expect_test "chain: C compiled to Wasm matches the reference, repeatedly" =
  with_dir (fun dir ->
      let g = Native_test.Graph_fixtures.chain () in
      let b, constants, p = prepare g dir in
      check g p b constants ~salt:0 ~poison:false;
      check ~repeat:3 g p b constants ~salt:2 ~poison:true;
      let z = V.sizes p in
      Fmt.pr "vector instructions in the generated code: %d@."
        z.V.vector_instructions);
  [%expect
    {|
    output t9 identical to the reference: true
    output t9 identical to the reference: true
    vector instructions in the generated code: 0 |}]

let%expect_test "forwarded and repeated outputs, Region nodes" =
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
        (fun i (name, g) ->
          Fmt.pr "%s@." name;
          let sub = Filename.concat dir (string_of_int i) in
          let b, constants, p = prepare g sub in
          check g p b constants ~salt:4 ~poison:true)
        [
          ("forwarded", forwarded);
          ("repeated", repeated);
          ("layer_norm", Native_test.Graph_fixtures.sink_permute_layer_norm ());
          ("sdpa", Native_test.Graph_fixtures.sink_permute_sdpa ());
        ]);
  [%expect
    {|
    forwarded
    output t0 identical to the reference: true
    output t1 identical to the reference: true
    repeated
    output t1 identical to the reference: true
    output t1 identical to the reference: true
    layer_norm
    output t2 identical to the reference: true
    sdpa
    output t4 identical to the reference: true |}]

let%expect_test "a missing toolchain is a typed error naming what is missing" =
  let saved = Sys.getenv_opt "MLTORCH_WASI_SYSROOT" in
  Unix.putenv "MLTORCH_WASI_SYSROOT" "/nonexistent-sysroot";
  (match V.toolchain_from_env () with
  | Ok _ -> Fmt.pr "found@."
  | Error e -> Fmt.pr "%a@." V.pp_error e);
  Unix.putenv "MLTORCH_WASI_SYSROOT" "";
  (match V.toolchain_from_env () with
  | Ok _ -> Fmt.pr "found@."
  | Error e -> Fmt.pr "%a@." V.pp_error e);
  Option.iter (Unix.putenv "MLTORCH_WASI_SYSROOT") saved;
  [%expect
    {|
    wasm toolchain missing: no wasm32 libc headers under /nonexistent-sysroot
    wasm toolchain missing: MLTORCH_WASI_SYSROOT is unset (run scripts/wasi-sysroot-userland.py and pass its SYSROOT) |}]
