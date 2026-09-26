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

(** Naming the pipeline, for a caller that wants a subset rather than a
    hand-rolled [pass list] -- e.g. the Model Explorer "generated JS" export,
    which lets a request name which passes ran. The type carries no function: a
    [Pass.t] is a stable, comparable label, never a closure a caller could
    substitute for the pipeline's own. *)
module Pass : sig
  (* Same order as {!passes} and for the same reason: the pipeline's order is
     behavior (§ above), so this list is that order restated as data, not
     alphabetized. *)
  type t = Unit_loops | Fold | Simplify | Guards | Cse | Hoist | Collapse

  val all : t list
  val name : t -> string
  val of_name : string -> t option
end

val select : Pass.t list -> pass list
(** [passes], filtered to exactly the named subset, still in pipeline order
    regardless of the argument's order or duplicates -- selecting is a
    sub-sequence of the fixed pipeline, never a reordering of it. *)
