(* Native-op verification: the two backends must agree. Evaluate the subject's
   graph with the Direct (concrete) backend and with the Symbolic backend
   (whole-graph stage DAG, then chain-grounded with the same inputs), and compare
   every graph output element-wise. Both share each op's [Compute] functor, so a
   divergence points at the staging/scheduling/shape machinery, not the pixel
   math. Pure OCaml — no libtorch. *)

let tensors_match shape a b =
  let ok = ref true in
  Vec6.iter shape (fun c ->
      if not (Float.equal (Tensor.read a c) (Tensor.read b c)) then ok := false);
  !ok

let run ppf (s : Native_subject.t) : bool =
  let g = s.Native_subject.graph in
  let shape_of oid =
    (Tensor_id.Map.find oid g.Graph_ir.Graph.tensors).Tensor_sig.shape
  in
  match Eval_direct.run g ~inputs:s.Native_subject.inputs with
  | Error e ->
      Fmt.pf ppf "[native] %s: eval error: %a@." s.Native_subject.target
        Eval_direct.pp_error (Err.Error.kind e);
      false
  | Ok direct -> (
      let prog = Eval_symbolic.run g in
      let bind id = List.assoc id s.Native_subject.inputs in
      match Stage_program.ground prog ~bind with
      | Error e ->
          Fmt.pf ppf "[native] %s: ground error: %a@." s.Native_subject.target
            Stage_program.pp_error (Err.Error.kind e);
          false
      | Ok grounded ->
          let agree oid =
            let d = Tensor_id.Map.find oid direct in
            let gr = Tensor_id.Map.find oid grounded in
            tensors_match (shape_of oid) d gr
          in
          if List.for_all agree g.Graph_ir.Graph.outputs then (
            Fmt.pf ppf "[native] %s: direct==symbolic@." s.Native_subject.target;
            true)
          else (
            Fmt.pf ppf "[native] %s: MISMATCH direct vs symbolic@."
              s.Native_subject.target;
            false))
