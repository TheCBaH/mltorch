(* The per-node twin of [Loop_region_program.lower] (design §4.3, plan T3.1):
   builds [Eval_symbolic.node_program] for one node, adapts it to a [Kernel.t]
   scoped to a single output ordinal via [Kernel_adapt.of_stage_program], places
   it with [Fusion_plan.default] (a single-value kernel has nothing to fuse),
   and lowers to a [Loop_program.t]. Needs no tensor payload -- everything
   comes from signatures -- so it runs natively, is testable with the ordinary
   [Loop_check] harness, and can compile a node before any weight is loaded
   (design §4.4, "precompile"). [~limits] is the same value [Eval_direct]'s own
   [~limits] threads through [Region_execution.lower], so both paths admit
   against the same budget. *)

type error = [ `Adapt of Kernel_adapt.error | `Lower of Loop_lower.error ]

val pp_error : Format.formatter -> [< error ] -> unit

val kernel :
  ?limits:Kernel.Limits.t ->
  Graph_ir.graph ->
  Graph_ir.node ->
  output:Output_ordinal.t ->
  (Kernel.t, Kernel_adapt.error) Err.t
(** The scoped [Kernel.t] [lower] itself adapts to, one step earlier: a consumer
    that needs an oracle to check the generated [Loop_program.t] against (e.g.
    [Kernel_eval.run_plan]) should build it by calling THIS rather than
    [Kernel_adapt.of_stage_program] directly on [Eval_symbolic.node_program] --
    the raw call skips the sibling-i64-stage pruning [lower] applies (see the
    design comment on [reachable_i64] in the implementation), so it names an
    extra, unrelated int64 output [lower]'s own kernel does not, and the two
    would disagree over a value neither was ever asked to compute. *)

val lower :
  ?limits:Kernel.Limits.t ->
  ?passes:Loop_opt.pass list ->
  Graph_ir.graph ->
  Graph_ir.node ->
  output:Output_ordinal.t ->
  (Loop_program.t, error) Err.t
(** [passes] defaults to {!Loop_opt.passes} (the full pipeline, matching
    {!Loop_lower.lower}'s own choke point exactly). [~passes:[]] is the raw,
    unoptimized program ({!Loop_lower.lower_unoptimized}); {!Loop_opt.select}
    names an arbitrary subset. Selecting a subset here changes only what a
    READER sees -- every executed path ([Kernel_eval], the jsoo executor, the
    Loop interpreter) still goes through {!Loop_lower.lower}'s fixed pipeline,
    never this one. *)
