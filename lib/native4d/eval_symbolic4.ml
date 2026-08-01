(* Symbolic evaluation of a Native4D graph, into the SAME [Stage_program.t] a
   Native graph produces. The twin of [Eval_symbolic].

   That the result type is shared and unparameterised is what makes cross-dialect
   verification cheap: [Stage_program.t] is typed on [Tensor_id.t],
   [Tensor_sig.t] and [Expr.t], none of which are dialect-specific, so the whole
   grounding and comparison machinery below [Map_verify] works on a Native4D
   program with no change at all. Design §9.3 assumes this; it holds only because
   Native4D reuses [Tensor_sig.t] verbatim (correction C3). *)

let f32 = Payload.Fmt Payload.F32

let first_free_tid (g : Graph.graph) =
  Tensor_id.Map.fold
    (fun k _ acc -> max acc (Tensor_id.to_int k + 1))
    g.Graph.Graph.tensors 0

let run (g : Graph.graph) : Stage_program.t =
  (* A fresh [Symbolic] instance per graph, so its reduction-variable counter
     starts clean and the produced expressions are deterministic. *)
  let module S = Symbolic.Make () in
  let module E = Eval_op4.Make (S) in
  let consts = ref [] in
  let next_const = ref (first_free_tid g) in
  let fill v shape =
    let id = Tensor_id.of_int !next_const in
    incr next_const;
    let sg = Tensor_sig.create ~id ~name:"" ~shape ~fmt:f32 () in
    consts := (sg, v) :: !consts;
    sg
  in
  let process_node env stages (node : Graph.node) =
    let op = node.Graph.Node.op in
    let operand r = Tensor_id.Map.find r env in
    let shape_of r = (Tensor_id.Map.find r env).Tensor_sig.shape in
    List.fold_left
      (fun (env, stages) (output, oid) ->
        let body =
          E.pixel op ~output ~operand ~shape_of ~fill Symbolic.out_vec
        in
        let out_sig = Tensor_id.Map.find oid g.Graph.Graph.tensors in
        let st = { Stage_program.Stage.id = oid; sg = out_sig; body } in
        (Tensor_id.Map.add oid out_sig env, st :: stages))
      (env, stages)
      (List.mapi (fun i oid -> (i, oid)) node.Graph.Node.outputs)
  in
  let _env, rev_stages =
    List.fold_left
      (fun (env, stages) node -> process_node env stages node)
      (g.Graph.Graph.tensors, [])
      g.Graph.Graph.nodes
  in
  {
    Stage_program.inputs =
      List.map
        (fun id -> (id, Tensor_id.Map.find id g.Graph.Graph.tensors))
        g.Graph.Graph.inputs;
    input_kinds = g.Graph.Graph.input_kinds;
    consts = List.rev !consts;
    stages = List.rev rev_stages;
    outputs = g.Graph.Graph.outputs;
  }
