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

val lower :
  ?limits:Kernel.Limits.t ->
  Graph_ir.graph ->
  Graph_ir.node ->
  output:Output_ordinal.t ->
  (Loop_program.t, error) Err.t
