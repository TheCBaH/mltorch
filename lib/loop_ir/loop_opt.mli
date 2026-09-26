(** The optimization pipeline: pure, exactness-preserving rewrites of a lowered
    [Loop_program.t]. [Loop_lower.lower] is the single choke point that applies
    it, so the interpreter, [Loop_js] and the jsoo executor all consume the same
    optimized program. See [.ai/loop_ir_optimization_design.md] for the
    invariants every pass keeps and why the pass order is fixed. *)

type pass = Loop_program.t -> Loop_program.t

val passes : pass list
(** The fixed pipeline, in the design doc's order (not alphabetical: each pass
    exposes facts the next one uses). *)

val run : ?passes:pass list -> Loop_program.t -> Loop_program.t
(** Defaults to {!passes}. A stage's mutation proof installs a deliberately
    broken variant of one pass here, via [~passes], rather than editing the
    pipeline itself. *)
