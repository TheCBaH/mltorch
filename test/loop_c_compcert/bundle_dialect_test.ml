open Graph_ir
open Loop_ir

(* The whole-model C backend in the Compcert_scalar dialect against the
   per-node reference evaluator: [model_infer.c] has no preprocessor directive
   and no header macro, and compiled beside the host main it is bitwise equal. *)

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
  List.find_map
    (fun (inv : Loop_bundle.invocation) ->
      List.find_map
        (fun ((buf : Loop_buffer.t), edge) ->
          if Tensor_id.equal edge id then Some buf.Loop_buffer.sg else None)
        (List.combine inv.Loop_bundle.program.Loop_program.buffers
           inv.Loop_bundle.edges))
    b.Loop_bundle.invocations
  |> Option.get

let map_of l id = List.assoc_opt id l

let with_dir f =
  let dir = Loop_c_exec.Proc.temp_dir "c_host_test" in
  Fun.protect
    ~finally:(fun () -> Loop_c_exec.Proc.remove_tree dir)
    (fun () -> f dir)

let prepare ?vector ?numerics ?(constant = tensor_of ~salt:3) g dir =
  let b =
    Err.or_raise ~pp_error:Loop_bundle.pp_error
      (Loop_bundle.build ~config:Loop_bundle_c.default_config g)
  in
  let constants =
    List.map
      (fun id -> (id, constant (sig_of_edge b id)))
      b.Loop_bundle.constants
  in
  let p =
    Err.or_raise ~pp_error:Loop_c_exec.Host.pp_error
      (Loop_c_exec.Host.prepare ~dialect:Loop_c_dialect.Compcert_scalar ?vector
         ?numerics ~dir b ~constants:(map_of constants))
  in
  (b, constants, p)

let check g p b constants ~salt ~poison =
  let inputs =
    List.map
      (fun id -> (id, tensor_of ~salt (sig_of_edge b id)))
      b.Loop_bundle.inputs
  in
  let reference =
    Err.or_raise ~pp_error:Eval_direct.pp_error
      (Eval_direct.run g ~constants ~inputs)
  in
  match Err.payload (Loop_c_exec.Host.run ~poison p ~bind:(map_of inputs)) with
  | Error e -> Fmt.pr "run failed: %a@." Loop_c_exec.Host.pp_error e
  | Ok outs ->
      List.iter2
        (fun id t ->
          Fmt.pr "output t%d identical to the reference: %b@."
            (Tensor_id.to_int id)
            (bits t = bits (Tensor_id.Map.find id reference)))
        g.Graph.outputs outs

let%expect_test "chain: the dialect model matches the reference, repeatedly" =
  with_dir (fun dir ->
      let g = Native_test.Graph_fixtures.chain () in
      let b, constants, p = prepare g dir in
      check g p b constants ~salt:0 ~poison:false;
      check g p b constants ~salt:2 ~poison:true;
      check g p b constants ~salt:5 ~poison:true;
      let st = (Loop_c_exec.Host.bundle_c p).Loop_bundle_c.stats in
      Fmt.pr "%d invocations, %d kernels@." st.Loop_bundle_c.invocations
        st.Loop_bundle_c.distinct_kernels);
  [%expect
    {|
    output t9 identical to the reference: true
    output t9 identical to the reference: true
    output t9 identical to the reference: true
    3 invocations, 3 kernels |}]

(* ---- binary32 kernels under a numerical policy ------------------------------ *)

let%expect_test "the whole-model unit is free of directives and header macros" =
  with_dir (fun dir ->
      let g = Native_test.Graph_fixtures.chain () in
      let _, _, p = prepare g dir in
      let text = (Loop_c_exec.Host.bundle_c p).Loop_bundle_c.source in
      let leak =
        List.find_opt
          (fun l -> String.length l > 0 && l.[0] = '#')
          (String.split_on_char '\n' text)
      in
      Fmt.pr "%s@." (Option.value leak ~default:"no directive"));
  [%expect {| no directive |}]
