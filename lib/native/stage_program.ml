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

type error = [ Region_program.error | Region_eval.error ]

let pp_error fmt : [< error ] -> unit = function
  | #Region_program.error as e -> Region_program.pp_error fmt e
  | #Region_eval.error as e -> Region_eval.pp_error fmt e

let widen_program r =
  Err.map_error (fun (e : Region_program.error) -> (e :> error)) r

let lower ~(limits : Kernel.Limits.t) (st : Stage.t) =
  widen_program
    (Region_execution.lower ~max_size:limits.Kernel.Limits.max_size
       ~max_depth:limits.Kernel.Limits.max_depth
       ~max_local_slots:limits.Kernel.Limits.max_local_slots
       ~max_scan_state:limits.Kernel.Limits.max_scan_state
       ~max_scan_updates:limits.Kernel.Limits.max_scan_updates_per_key
       ~output_shape:st.Stage.sg.Tensor_sig.shape (Stage.computation st))

let ground ?(limits = Kernel.Limits.default) (p : t)
    ~(bind : Tensor_id.t -> Tensor.packed) :
    (Tensor.packed Tensor_id.Map.t, error) Err.t =
  Err.Escape.with_escape @@ fun esc ->
  (* Preflight every stage before materializing the first one: a program that
     fails halfway through an otherwise-successful ground would have already
     mutated no shared state (each stage's tensor is a fresh value), but
     reporting the failure only after doing part of the graph's real work is
     still the wrong contract for a validation step this cheap to run first. *)
  List.iter
    (fun st -> ignore (Err.Escape.or_throw esc (lower ~limits st)))
    p.stages;
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
          match Err.Escape.or_throw esc (lower ~limits st) with
          | Region_execution.Pixel_loop body ->
              Err.Escape.or_throw esc
                (Err.map_error
                   (fun (e : Expr.Eval.error) -> (e :> error))
                   (Schedule.ground st.sg.shape ~binding body))
          | Region_execution.Region_loop lowered ->
              Err.Escape.or_throw esc
                (Err.map_error
                   (fun (e : Region_eval.error) -> (e :> error))
                   (Region_execution.materialize lowered
                      ~env:(Expr_bridge.env ~binding)))
        in
        (Tensor_id.Map.add st.sg.id t binds, Tensor_id.Map.add st.id t result))
      (seed, Tensor_id.Map.empty)
      p.stages
  in
  result
