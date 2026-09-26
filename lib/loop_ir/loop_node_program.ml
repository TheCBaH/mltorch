(* See loop_node_program.mli. *)

type error = [ `Adapt of Kernel_adapt.error | `Lower of Loop_lower.error ]

let pp_error fmt : [< error ] -> unit = function
  | `Adapt e -> Kernel_adapt.pp_error fmt e
  | `Lower e -> Loop_lower.pp_error fmt e

(* [stages_i64] entries [oid] actually needs, transitively -- everything else
   in [p.Stage_program.stages_i64] is an unrelated sibling (e.g. [Max_dim]'s
   own index stage, when [oid] names its VALUE ordinal instead). Needed
   because [Kernel_adapt.of_stage_program] copies [stages_i64] into
   [Kernel.t.values_i64] UNCONDITIONALLY of [~select] (every int64 value is
   always materialized, by design -- see [Kernel.Value_i64.t]'s own doc), and
   [Loop_lower.lower] in turn emits every [values_i64] entry as an output
   buffer unconditionally too: left unpruned, a value-ordinal-only kernel
   still carries its index sibling's stage along, so the lowered program ends
   up with TWO output buffers instead of one, and [Loop_node_executor]
   correctly refuses it ("expected exactly one output buffer") rather than
   silently picking one. Pruning here, before [Kernel_adapt] ever sees the
   unrelated entry, is the fix: an i64 stage [oid] itself depends on stays,
   everything else is dropped. *)
let reachable_i64 (p : Stage_program.t) (oid : Tensor_id.t) =
  let i64_of_id =
    List.fold_left
      (fun m (st : Stage_program.Stage_i64.t) ->
        Tensor_id.Map.add st.Stage_program.Stage_i64.id st m)
      Tensor_id.Map.empty p.Stage_program.stages_i64
  in
  let float_of_id =
    List.fold_left
      (fun m (st : Stage_program.Stage.t) ->
        Tensor_id.Map.add st.Stage_program.Stage.id st m)
      Tensor_id.Map.empty p.Stage_program.stages
  in
  let rec walk_i64 id acc =
    if Tensor_id.Set.mem id acc then acc
    else
      match Tensor_id.Map.find_opt id i64_of_id with
      | None -> acc
      | Some st ->
          let acc = Tensor_id.Set.add id acc in
          Expr.Source.Set.fold
            (fun src acc -> walk_i64 (Expr_bridge.id_of_source src) acc)
            (Expr.Fold.sources_i64 st.Stage_program.Stage_i64.pixel)
            acc
  in
  let from_sources sources =
    Expr.Source.Set.fold
      (fun src acc -> walk_i64 (Expr_bridge.id_of_source src) acc)
      sources Tensor_id.Set.empty
  in
  match Tensor_id.Map.find_opt oid i64_of_id with
  | Some _ -> walk_i64 oid Tensor_id.Set.empty
  | None -> (
      match Tensor_id.Map.find_opt oid float_of_id with
      | Some st -> from_sources (Stage_program.Stage.sources st)
      | None -> Tensor_id.Set.empty)

(* The [Kernel.t] one output ordinal adapts to -- exposed separately from
   [lower] so a consumer that needs the SAME scoped kernel a generated
   program was checked against (a reference oracle, e.g.) builds it through
   this one path rather than re-deriving the pruning above by hand and
   risking the two drifting apart. *)
let kernel ?(limits = Kernel.Limits.default) (g : Graph_ir.graph)
    (node : Graph_ir.node) ~(output : Output_ordinal.t) :
    (Kernel.t, Kernel_adapt.error) Err.t =
  let oid =
    List.assoc output (Output_ordinal.indexed node.Graph_ir.Node.outputs)
  in
  let program = Eval_symbolic.node_program ~limits g node in
  let needed = reachable_i64 program oid in
  let program =
    {
      program with
      Stage_program.stages_i64 =
        List.filter
          (fun (st : Stage_program.Stage_i64.t) ->
            Tensor_id.Set.mem st.Stage_program.Stage_i64.id needed)
          program.Stage_program.stages_i64;
      (* A sibling output ([Max_dim]'s other ordinal, say) that [analyse]
         would otherwise need to classify no longer names a [stages_i64]
         entry once pruned above -- narrowed to [oid] alone, matching
         [~select]/[~outputs] below, so [analyse] never has to classify a
         sibling this lowering was never asked to route. *)
      Stage_program.outputs = [ oid ];
    }
  in
  (* [~select] must narrow to just [oid] too, not only [~outputs]: with
     [~select] absent (whole-program), [Kernel_adapt.required]'s own
     [graph_outs] step includes EVERY entry of [Stage_program.outputs] that
     falls in the selection -- for a multi-output node (`Unbind`, `Lstm`,
     `Max_dim`'s two outputs, ...) that is every sibling output, not just
     [oid], so a single-element [~outputs] then fails to "begin with" the
     multi-element required list. Narrowing [~select] to [oid] alone makes
     the required list exactly [[oid]], matching [~outputs] again. *)
  Kernel_adapt.of_stage_program ~limits
    ~select:(Tensor_id.Set.singleton oid)
    ~outputs:[ oid ] program

let lower ?(limits = Kernel.Limits.default) (g : Graph_ir.graph)
    (node : Graph_ir.node) ~(output : Output_ordinal.t) :
    (Loop_program.t, error) Err.t =
  let open Err.Syntax in
  let* kernel =
    kernel ~limits g node ~output |> Err.map_error (fun e -> `Adapt e)
  in
  Loop_lower.lower (Fusion_plan.default kernel)
  |> Err.map_error (fun e -> `Lower e)
