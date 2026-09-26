(** Stage 1 of the optimization pipeline ([.ai/loop_ir_optimization_design.md]
    pass 1): a [For] whose bounds are literal [Const]s exactly one apart always
    runs its body exactly once with [var = lo], so it is replaced by its own
    body with the loop variable substituted throughout -- including inside every
    [Load]/[Store] coordinate, [Fail_if] predicate and failure payload the body
    carries. Runs first, so later passes see constants where the raw lowering
    had [Var]-plus-[Var]-plus-[Var] offsets. *)

(* [Loop_program.t -> Loop_program.t] is written out, not [Loop_opt.pass]:
   [Loop_opt] itself depends on this module to build its default pipeline, so
   naming its type back here would be a module cycle. *)

val run : Loop_program.t -> Loop_program.t

val with_substitute :
  (Loop_index.t -> Loop_index.t) -> Loop_program.t -> Loop_program.t
(** [run] is [with_substitute Fun.id]: a proven unit loop's variable is replaced
    by its own [lo] bound, unchanged. Exposed so a mutation test installs a
    deliberately wrong replacement (e.g. [lo + 1]) through {!Loop_opt.run}'s
    [~passes] hook without duplicating the traversal. *)
