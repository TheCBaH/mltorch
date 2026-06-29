(* See eval_symbolic.mli. The env maps each edge id to its [Tensor_sig.t] (the
   [S.input] in symbolic mode); since the builder already created a signature for
   every edge, the env starts as [graph.tensors] and a node's operands resolve to
   their producers' signatures directly. A missing optional operand is filled with
   a fresh constant signature, recorded in [consts] for grounding. *)

open Graph_ir

(* An identity copy expression: read [sg] at the output coordinate, per axis. *)
let identity_load (sg : Tensor_sig.t) : Expr.t =
  Expr.Load (sg, Array.of_list (List.map (fun a -> Expr.Index_var a) Axis.all))

let f32 = Payload.Fmt Payload.F32

let first_free_tid (g : graph) =
  let rec walk acc gr =
    let acc =
      Tensor_id.Map.fold
        (fun k _ a -> max a (Tensor_id.to_int k + 1))
        gr.Graph.tensors acc
    in
    List.fold_left
      (fun acc (node : node) ->
        match node.Node.op with
        | Subgraph { graph = sub; _ } -> walk acc sub
        | _ -> acc)
      acc gr.Graph.nodes
  in
  walk 0 g

let run (g : graph) : Stage_program.t =
  (* a fresh Symbolic instance per graph: its reduction-var counter starts clean,
     so the produced expressions are deterministic. *)
  let module S = Symbolic.Make () in
  let module E = Eval_op.Make (S) in
  let consts = ref [] in
  let next_const = ref (first_free_tid g) in
  let fill v shape =
    let id = Tensor_id.of_int !next_const in
    incr next_const;
    let sg =
      Tensor_sig.create ~id
        ~name:(Printf.sprintf "const%g" v)
        ~shape ~fmt:f32 ()
    in
    consts := (sg, v) :: !consts;
    sg
  in
  let rec process (gr : graph) env stages =
    List.fold_left
      (fun (env, stages) node -> process_node gr env stages node)
      (env, stages) gr.Graph.nodes
  and process_node (gr : graph) env stages (node : node) =
    match node.Node.op with
    | Subgraph { graph = sub; args } ->
        (* inline: add the sub's edge sigs, alias its inputs to the call args *)
        let env =
          Tensor_id.Map.union (fun _ a _ -> Some a) env sub.Graph.tensors
        in
        let env =
          List.fold_left2
            (fun e iid aid ->
              Tensor_id.Map.add iid (Tensor_id.Map.find aid env) e)
            env sub.Graph.inputs args
        in
        let env, stages = process sub env stages in
        (* alias node outputs to sub outputs via identity-copy stages *)
        List.fold_left2
          (fun (env, stages) oid sub_oid ->
            let out_sig = Tensor_id.Map.find oid gr.Graph.tensors in
            let src_sig = Tensor_id.Map.find sub_oid env in
            let st =
              {
                Stage_program.Stage.id = oid;
                sg = out_sig;
                body = identity_load src_sig;
              }
            in
            (Tensor_id.Map.add oid out_sig env, st :: stages))
          (env, stages) node.Node.outputs sub.Graph.outputs
    | op ->
        let operand r = Tensor_id.Map.find r env in
        let shape_of r = (Tensor_id.Map.find r env).Tensor_sig.shape in
        let body = E.pixel op ~operand ~shape_of ~fill Symbolic.out_coord in
        let oid =
          match node.Node.outputs with
          | [ oid ] -> oid
          | _ -> invalid_arg "Eval_symbolic: expected a single output id"
        in
        let out_sig = Tensor_id.Map.find oid gr.Graph.tensors in
        let st = { Stage_program.Stage.id = oid; sg = out_sig; body } in
        (Tensor_id.Map.add oid out_sig env, st :: stages)
  in
  let _env, rev_stages = process g g.Graph.tensors [] in
  let inputs =
    List.map
      (fun id -> (id, Tensor_id.Map.find id g.Graph.tensors))
      g.Graph.inputs
  in
  {
    Stage_program.inputs;
    consts = List.rev !consts;
    stages = List.rev rev_stages;
    outputs = g.Graph.outputs;
  }
