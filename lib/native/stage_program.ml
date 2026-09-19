(* See stage_program.mli. Grounding keys the [Schedule.ground] binding by the
   signature's (globally unique) id, seeded with the graph inputs and the synthetic
   constants, then extended stage by stage in topo order. *)

open Graph_ir

module Stage = struct
  type t = {
    id : Tensor_id.t;
    sg : Tensor_sig.t;
    computation : Region_group.Ref.t;
  }

  let computation t = t.computation
  let sources t = Region_group.Ref.sources (computation t)

  type pixel_body_error = [ Region_program.error | `Not_a_pixel_program ]

  let pp_pixel_body_error fmt : [< pixel_body_error ] -> unit = function
    | #Region_program.error as e -> Region_program.pp_error fmt e
    | `Not_a_pixel_program ->
        Fmt.string fmt "a group emitter is not a pixel program"

  let pixel_body ~max_size ~max_depth ~scan_limits t =
    match computation t with
    | Region_group.Ref.Solo p ->
        Err.map_error
          (fun e -> (e :> pixel_body_error))
          (Region_program.specialize_pixel ~max_size ~max_depth ~scan_limits p)
    | Region_group.Ref.Grouped _ -> Err.fail `Not_a_pixel_program

  let check ~max_size ~max_depth t =
    Region_group.Ref.check ~max_size ~max_depth (computation t)
end

module Stage_i64 = struct
  type t = { id : Tensor_id.t; sg : Tensor_sig.t; pixel : int64 Expr.Value.t }
end

type t = {
  inputs : (Tensor_id.t * Tensor_sig.t) list;
  input_kinds : Input.kind Tensor_id.Map.t;
  consts : (Tensor_sig.t * float) list;
  stages : Stage.t list;
  stages_i64 : Stage_i64.t list;
  outputs : Tensor_id.t list;
}

let pp fmt (p : t) =
  let name_of id = Format.asprintf "%a" Tensor_id.pp id in
  let comma = String.concat ", " in
  Format.fprintf fmt "@[<v>inputs: %s@,"
    (comma (List.map (fun (id, _) -> name_of id) p.inputs));
  List.iter
    (fun (st : Stage.t) ->
      match Region_group.Ref.pixel_expression st.computation with
      | Some body ->
          Format.fprintf fmt "%a = %a@," Tensor_id.pp st.id Expr.Pp.value body
      | None ->
          Format.fprintf fmt "%a = %a@," Tensor_id.pp st.id Region_group.Ref.pp
            st.computation)
    p.stages;
  List.iter
    (fun (st : Stage_i64.t) ->
      Format.fprintf fmt "%a = %a@," Tensor_id.pp st.Stage_i64.id
        Expr.Pp.value_i64 st.Stage_i64.pixel)
    p.stages_i64;
  Format.fprintf fmt "outputs: %s@]" (comma (List.map name_of p.outputs))

type error =
  [ Region_group.error
  | Region_program.error
  | Region_eval.error
  | `Duplicate_group_ordinal of int ]

let pp_error fmt : [< error ] -> unit = function
  | #Region_group.error as e -> Region_group.pp_error fmt e
  | #Region_program.error as e -> Region_program.pp_error fmt e
  | #Region_eval.error as e -> Region_eval.pp_error fmt e
  | `Duplicate_group_ordinal ordinal ->
      Fmt.pf fmt "group run repeats emitter ordinal %d" ordinal

let widen_group r =
  Err.map_error (fun (e : Region_group.error) -> (e :> error)) r

let widen_program r =
  Err.map_error (fun (e : Region_program.error) -> (e :> error)) r

(* [Stage.computation]'s [Region_group.Ref.t] is projected to an ordinary
   [Region_program.t] first (a [Grouped] stage rebuilds its emitter's
   projection independently every time [lower] runs -- project step 19's
   Section C milestone 2 removes this per-stage duplication for real), then
   lowered/preflighted exactly as before. *)
let lower ~(limits : Kernel.Limits.t) (st : Stage.t) =
  let open Err.Syntax in
  let* program =
    widen_group
      (Region_group.Ref.project ~max_size:limits.Kernel.Limits.max_size
         ~max_depth:limits.Kernel.Limits.max_depth (Stage.computation st))
  in
  widen_program
    (Region_execution.lower ~max_size:limits.Kernel.Limits.max_size
       ~max_depth:limits.Kernel.Limits.max_depth
       ~max_local_slots:limits.Kernel.Limits.max_local_slots
       ~scan_limits:(Kernel.Limits.scan_limits limits)
       ~output_shape:st.Stage.sg.Tensor_sig.shape program)

(* [Region_group.Run.t] specialized to [Stage.t]: a maximal run of consecutive
   stages sharing one physically-identical [Region_group.t], or a single
   ordinary stage. Both symbolic builders (project step 19's wrapper
   migration) always emit one multi-output node's sibling stages contiguously
   and in ascending emitter-ordinal order, so a [Group]'s [members] list is
   exactly that order; a caller-built or reordered [Stage_program.t] with the
   same physical group split across non-contiguous stages would instead
   surface as several separate runs over the same [g] -- correct (each still
   projects/preflights/executes soundly) but forgoes sharing across the gap,
   which no real producer here creates. *)
let runs_of_stages stages =
  Region_group.runs ~computation:Stage.computation stages

let preflight_run ~limits esc (run : Stage.t Region_group.Run.t) =
  match run with
  | Region_group.Run.Solo st ->
      ignore (Err.Escape.or_throw esc (lower ~limits st))
  | Region_group.Run.Group (g, _) ->
      (match Region_group.Run.duplicate_ordinal run with
      | Some ordinal -> Err.Escape.throw esc (`Duplicate_group_ordinal ordinal)
      | None -> ());
      ignore
        (Err.Escape.or_throw esc
           (widen_group
              (Region_execution.lower_group
                 ~max_size:limits.Kernel.Limits.max_size
                 ~max_depth:limits.Kernel.Limits.max_depth
                 ~max_local_slots:limits.Kernel.Limits.max_local_slots
                 ~scan_limits:(Kernel.Limits.scan_limits limits)
                 g)))

(* A Bool-declared stage is computed on the float path and stored as canonical
   Bool bytes; every other declared format keeps the tensor the float path
   produced. [Kernel_eval.stored] is the Kernel-side twin. *)
let stored = Output_spec.store

let execute_run ~limits ~region_counters ~lookup esc (binds, result) = function
  | Region_group.Run.Solo st ->
      (* [find_opt], not [find]: a missing binding must reach the evaluator as
         a value it can report, not a [Not_found] raised out of a map lookup
         before any error path exists. *)
      let binding id = lookup binds id in
      let t =
        match Err.Escape.or_throw esc (lower ~limits st) with
        | Region_execution.Pixel_loop body ->
            Err.Escape.or_throw esc
              (Err.map_error
                 (fun (e : Expr.Eval.error) -> (e :> error))
                 (Schedule.ground st.Stage.sg.shape ~binding
                    ~scan_limits:(Kernel.Limits.scan_limits limits)
                    body))
        | Region_execution.Region_loop lowered ->
            let counters =
              Option.bind region_counters (fun m ->
                  Tensor_id.Map.find_opt st.Stage.id m)
            in
            Err.Escape.or_throw esc
              (Err.map_error
                 (fun (e : Region_eval.error) -> (e :> error))
                 (Region_execution.materialize ?counters lowered
                    ~env:(Expr_bridge.env ~binding)))
      in
      let t = stored st.Stage.sg t in
      ( Tensor_id.Map.add st.Stage.sg.id t binds,
        Tensor_id.Map.add st.Stage.id t result )
  | Region_group.Run.Group (g, members) ->
      let binding id = lookup binds id in
      let lowered_group =
        Err.Escape.or_throw esc
          (widen_group
             (Region_execution.lower_group
                ~max_size:limits.Kernel.Limits.max_size
                ~max_depth:limits.Kernel.Limits.max_depth
                ~max_local_slots:limits.Kernel.Limits.max_local_slots
                ~scan_limits:(Kernel.Limits.scan_limits limits)
                g))
      in
      (* One counters record for the whole group, resolved from its FIRST
         member: [materialize_group] runs exactly once per group-run, so
         whichever single record the caller bound its members to (typically
         all three, per lstm_scale_test.ml's own convention) is charged
         exactly once, never once per sibling. *)
      let counters =
        match members with
        | (_, first_st) :: _ ->
            Option.bind region_counters (fun m ->
                Tensor_id.Map.find_opt first_st.Stage.id m)
        | [] -> None
      in
      let tensors =
        Err.Escape.or_throw esc
          (Err.map_error
             (fun (e : Region_eval.error) -> (e :> error))
             (Region_execution.materialize_group ?counters lowered_group
                ~env:(Expr_bridge.env ~binding) ~selected:(List.map fst members)))
      in
      List.fold_left
        (fun (binds, result) (ordinal, tensor) ->
          let st = List.assoc ordinal members in
          let tensor = stored st.Stage.sg tensor in
          ( Tensor_id.Map.add st.Stage.sg.id tensor binds,
            Tensor_id.Map.add st.Stage.id tensor result ))
        (binds, result) tensors

let ground ?(limits = Kernel.Limits.default) ?region_counters (p : t)
    ~(bind : Tensor_id.t -> Tensor.packed) :
    (Tensor.packed Tensor_id.Map.t, error) Err.t =
  Err.Escape.with_escape @@ fun esc ->
  let runs = runs_of_stages p.stages in
  (* Preflight every run before materializing the first one: a program that
     fails halfway through an otherwise-successful ground would have already
     mutated no shared state (each stage's tensor is a fresh value), but
     reporting the failure only after doing part of the graph's real work is
     still the wrong contract for a validation step this cheap to run first. *)
  List.iter (preflight_run ~limits esc) runs;
  let seed =
    List.fold_left
      (fun m (id, (s : Tensor_sig.t)) -> Tensor_id.Map.add s.id (bind id) m)
      Tensor_id.Map.empty p.inputs
  in
  let seed =
    List.fold_left
      (fun m ((s : Tensor_sig.t), v) ->
        let filled =
          match s.fmt with
          | Payload.Fmt Payload.Bool ->
              Tensor.materialize_bool s.shape (fun _ -> v <> 0.)
          | _ -> Tensor.materialize s.shape (fun _ -> v)
        in
        Tensor_id.Map.add s.id filled m)
      seed p.consts
  in
  (* Thread the sig->tensor binding through the runs in topo order, collecting
     each stage's grounded result keyed by its edge id. A [Group_run] installs
     every member's tensor in both maps atomically before the fold advances to
     a consumer -- [Region_execution.materialize_group] either returns every
     selected tensor or fails via [Err.Escape], so no partial group result is
     ever published (design record §5.2). *)
  (* An int64 stage is computed the first time anything reads it, memoised, from
     whatever is bound by then: an int64 stage may read a float stage (which is
     bound before any consumer of the int64 stage runs) and a float stage may
     read an int64 one. [in_progress] turns a malformed cycle into a missing
     binding instead of a loop. *)
  let i64_stages =
    List.fold_left
      (fun m (st : Stage_i64.t) -> Tensor_id.Map.add st.Stage_i64.id st m)
      Tensor_id.Map.empty p.stages_i64
  in
  let cache = ref Tensor_id.Map.empty in
  let in_progress = ref Tensor_id.Set.empty in
  let rec lookup binds id =
    match Tensor_id.Map.find_opt id binds with
    | Some _ as found -> found
    | None -> (
        match Tensor_id.Map.find_opt id !cache with
        | Some _ as found -> found
        | None -> (
            match Tensor_id.Map.find_opt id i64_stages with
            | Some st when not (Tensor_id.Set.mem id !in_progress) ->
                in_progress := Tensor_id.Set.add id !in_progress;
                let tensor =
                  Err.Escape.or_throw esc
                    (Err.map_error
                       (fun (e : Region_eval.error) -> (e :> error))
                       (Region_eval.materialize_i64
                          ~output_shape:st.Stage_i64.sg.Tensor_sig.shape
                          ~env:(Expr_bridge.env ~binding:(lookup binds))
                          st.Stage_i64.pixel))
                in
                in_progress := Tensor_id.Set.remove id !in_progress;
                cache := Tensor_id.Map.add id tensor !cache;
                Some tensor
            | _ -> None))
  in
  let binds, result =
    List.fold_left
      (execute_run ~limits ~region_counters ~lookup esc)
      (seed, Tensor_id.Map.empty)
      runs
  in
  List.fold_left
    (fun result (st : Stage_i64.t) ->
      match lookup binds st.Stage_i64.id with
      | Some tensor -> Tensor_id.Map.add st.Stage_i64.id tensor result
      | None -> result)
    result p.stages_i64
