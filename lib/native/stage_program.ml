(* See stage_program.mli. Grounding keys the [Schedule.ground] binding by the
   signature's (globally unique) id, seeded with the graph inputs and the synthetic
   constants, then extended stage by stage in topo order. *)

open Graph_ir

module Stage = struct
  type t = {
    id : Tensor_id.t;
    sg : Tensor_sig.t;
    computation : Region_program.t;
  }

  let computation t = t.computation
  let sources t = Region_program.Fold.sources (computation t)

  let pixel_body ~max_size ~max_depth t =
    Region_program.specialize_pixel ~max_size ~max_depth t.computation

  let check ~max_size ~max_depth t =
    Region_program.check ~max_size ~max_depth (computation t)
end

type t = {
  inputs : (Tensor_id.t * Tensor_sig.t) list;
  input_kinds : Input.kind Tensor_id.Map.t;
  consts : (Tensor_sig.t * float) list;
  stages : Stage.t list;
  outputs : Tensor_id.t list;
}

let pp fmt (p : t) =
  let name_of id = Format.asprintf "%a" Tensor_id.pp id in
  let comma = String.concat ", " in
  Format.fprintf fmt "@[<v>inputs: %s@,"
    (comma (List.map (fun (id, _) -> name_of id) p.inputs));
  List.iter
    (fun (st : Stage.t) ->
      match Region_program.pixel_expression st.computation with
      | Some body ->
          Format.fprintf fmt "%a = %a@," Tensor_id.pp st.id Expr.Pp.value body
      | None ->
          Format.fprintf fmt "%a = %a@," Tensor_id.pp st.id Region_program.pp
            st.computation)
    p.stages;
  Format.fprintf fmt "outputs: %s@]" (comma (List.map name_of p.outputs))

let ground (p : t) ~(bind : Tensor_id.t -> Tensor.packed) :
    Tensor.packed Tensor_id.Map.t =
  let seed =
    List.fold_left
      (fun m (id, (s : Tensor_sig.t)) -> Tensor_id.Map.add s.id (bind id) m)
      Tensor_id.Map.empty p.inputs
  in
  let seed =
    List.fold_left
      (fun m ((s : Tensor_sig.t), v) ->
        Tensor_id.Map.add s.id (Tensor.materialize s.shape (fun _ -> v)) m)
      seed p.consts
  in
  (* Thread the sig->tensor binding through the stages in topo order, collecting
     each stage's grounded result keyed by its edge id. *)
  let _, result =
    List.fold_left
      (fun (binds, result) (st : Stage.t) ->
        (* [find_opt], not [find]: a missing binding must reach the evaluator as
           a value it can report, not a [Not_found] raised out of a map lookup
           before any error path exists. *)
        let binding id = Tensor_id.Map.find_opt id binds in
        let t =
          (* [?limits] does not reach here yet -- [Eval_symbolic.run] hardcodes
             [Kernel.Limits.default] the same way, and both are threaded a real
             configured value together later. *)
          match
            Err.or_raise ~pp_error:Region_program.pp_error
              (Region_execution.lower
                 ~max_size:Kernel.Limits.default.Kernel.Limits.max_size
                 ~max_depth:Kernel.Limits.default.Kernel.Limits.max_depth
                 (Stage.computation st))
          with
          | Region_execution.Pixel_loop body ->
              Schedule.ground st.sg.shape ~binding body
          | Region_execution.Region_loop lowered ->
              Err.or_raise ~pp_error:Region_eval.pp_error
                (Region_execution.materialize lowered ~output_shape:st.sg.shape
                   ~env:(Expr_bridge.env ~binding))
        in
        (Tensor_id.Map.add st.sg.id t binds, Tensor_id.Map.add st.id t result))
      (seed, Tensor_id.Map.empty)
      p.stages
  in
  result
