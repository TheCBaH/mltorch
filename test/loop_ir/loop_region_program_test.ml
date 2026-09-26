open Loop_ir
open Graph_ir

(* [Loop_region_program.lower] wraps a single [Region_program.t] -- exactly
   the program a [region_result]-style caller already built -- in a minimal
   Kernel.t and runs it through [Loop_lower]. Confirms end to end (lower +
   interpret) that generated Loop IR agrees with the reference evaluator, at
   both a toy shape and the real [fastvit_sa12] SDPA shape ([H=16 W=49
   C=32]). *)

let sdpa_graph ~query_shape ~key_shape =
  let mask_shape = Attention.Sdpa.score_shape ~query_shape ~key_shape in
  let materialize shape scale =
    Tensor.materialize shape (fun coord ->
        (scale *. float_of_int (Dim.to_int (Vec6.get coord Axis.W)))
        +. float_of_int (Dim.to_int (Vec6.get coord Axis.C))
        +. 1.)
  in
  let query = materialize query_shape 10. in
  let key = materialize key_shape 10. in
  let value = materialize key_shape 100. in
  let mask = Tensor.materialize mask_shape (fun _ -> 0.) in
  let g =
    Err.or_raise ~pp_error:Graph_builder.pp_error
      Graph_builder.(
        build ~name:"sdpa_loop_region" ~outputs:(fun output -> [ output ])
        @@
        let* qi = input ~shape:query_shape ~name:"query" () in
        let* ki = input ~shape:key_shape ~name:"key" () in
        let* vi = input ~shape:key_shape ~name:"value" () in
        let* mi = input ~shape:mask_shape ~name:"mask" () in
        sdpa
          { Attention.Sdpa.scale = Attention.Sdpa.Scale.Default }
          ~query:qi ~key:ki ~value:vi ~mask:mi ())
  in
  let inputs =
    match g.Graph.inputs with
    | [ qid; kid; vid; mid ] ->
        [ (qid, query); (kid, key); (vid, value); (mid, mask) ]
    | _ -> assert false
  in
  (g, inputs)

let check ~query_shape ~key_shape =
  let g, inputs = sdpa_graph ~query_shape ~key_shape in
  let output_id = List.hd g.Graph.outputs in
  let out_shape =
    (Tensor_id.Map.find output_id g.Graph.tensors).Tensor_sig.shape
  in
  let node = List.hd g.Graph.nodes in
  let filled = ref Tensor_id.Map.empty in
  let fill role value shape =
    let id =
      Tensor_id.of_int
        (1
        + Tensor_id.Map.fold
            (fun id _ acc -> max acc (Tensor_id.to_int id))
            g.Graph.tensors (-1))
    in
    ignore role;
    filled := Tensor_id.Map.add id (value, shape) !filled;
    Tensor_sig.create ~id ~name:"direct optional operand" ~shape
      ~fmt:(Payload.Fmt Payload.F32) ()
  in
  let reference =
    Err.or_raise ~pp_error:Eval_direct.pp_error (Eval_direct.run ~inputs g)
  in
  let reference_tensor = Tensor_id.Map.find output_id reference in
  let program =
    Err.or_raise ~pp_error:Region_computation.pp_error
      (Region_computation.program ~limits:Kernel.Limits.default
         ~op:node.Graph_ir.Node.op ~output:Output_ordinal.zero
         ~output_shape:out_shape
         ~operand:(fun id -> Tensor_id.Map.find_opt id g.Graph.tensors)
         ~fill)
  in
  let operand_env =
    List.fold_left
      (fun m (id, tensor) -> Tensor_id.Map.add id tensor m)
      Tensor_id.Map.empty inputs
  in
  let sources = Region_program.Fold.sources program in
  let synthetic_bindings =
    Tensor_id.Map.filter_map
      (fun id (value, shape) ->
        if Expr.Source.Set.mem (Expr_bridge.source_of_id id) sources then
          Some (Tensor.materialize shape (fun _ -> value))
        else None)
      !filled
  in
  let bindings =
    Tensor_id.Map.union
      (fun _ tensor _ -> Some tensor)
      operand_env synthetic_bindings
  in
  match
    Loop_region_program.lower ~limits:Kernel.Limits.default ~out_shape ~bindings
      program
  with
  | Error e ->
      Fmt.pr "lower: %a@." Loop_region_program.pp_error (Err.Error.kind e)
  | Ok loop_program -> (
      let bind id = List.assoc_opt id inputs in
      match Loop_interp.run loop_program ~bind with
      | Error e -> Fmt.pr "interp: %a@." Loop_interp.pp_error (Err.Error.kind e)
      | Ok env ->
          let outputs =
            List.filter_map
              (fun (b : Loop_buffer.t) ->
                if b.Loop_buffer.role = Loop_buffer.Output then
                  Some b.Loop_buffer.id
                else None)
              loop_program.Loop_program.buffers
          in
          let out_id = List.hd outputs in
          let loop_tensor = Tensor_id.Map.find out_id env in
          Fmt.pr "lower: ok, agree=%b@."
            (Tensor.equal_bits reference_tensor loop_tensor))

let%expect_test "toy SDPA shape lowers and agrees with the reference" =
  check
    ~query_shape:(Vec6.shape ~n:1 ~t:1 ~d:1 ~h:1 ~w:2 ~c:3)
    ~key_shape:(Vec6.shape ~n:1 ~t:1 ~d:1 ~h:1 ~w:3 ~c:3);
  [%expect {| lower: ok, agree=true |}]

let%expect_test
    "fastvit_sa12-scale SDPA shape (H=16 W=49 C=32) lowers and agrees with the \
     reference" =
  check
    ~query_shape:(Vec6.shape ~n:1 ~t:1 ~d:1 ~h:16 ~w:49 ~c:32)
    ~key_shape:(Vec6.shape ~n:1 ~t:1 ~d:1 ~h:16 ~w:49 ~c:32);
  [%expect {| lower: ok, agree=true |}]
