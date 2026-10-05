(** The pass driver. A pass takes a verified revision and returns the next one
    with whether it changed anything; the driver verifies every revision it
    produces, so a pass that breaks the structure or states a proof the program
    cannot support is caught at the pass that did it, not by a later consumer. A
    pass is measured against the interpreter, not trusted: see the differential
    sweep in the tests. *)

type pass = { name : string; run : Ssa_program.t -> Ssa_program.t * bool }

val simplify : pass
(** Constant folding, pure common subexpressions and dead pure values. *)

val guards : pass
(** What the ranges prove cannot happen: checked operations that cannot fail,
    checks that cannot fire, loops whose trip count is fixed. *)

val hoist : alias:Ssa_effects.policy -> pass
(** Pure operations out of the loops they do not depend on, and loads proved in
    bounds out of loops that run and never write what they read. *)

val share : alias:Ssa_effects.policy -> pass
(** A load an earlier load already performed, with nothing that may write its
    buffer in between. *)

val block : alias:Ssa_effects.policy -> group:Ssa_opt_block.group -> pass
(** Independent outputs computed together: a loop whose iterations are separate
    outputs around one reduction becomes full groups that run the reduction once
    with an accumulator per output, plus the original loop for the remainder.
    Each output keeps its own accumulator and its terms in order, so its value
    and the logical work are unchanged; the loads the group shares are then
    merged by {!share}. A loop that does not meet every condition is left alone.
*)

val vectorize : alias:Ssa_effects.policy -> target:Ssa_target.t -> pass
(** {!Ssa_vectorize.program}: loops of independent outputs become vector loops
    where the target's cost model says they pay. *)

val schedule_sums : target:Ssa_target.t -> pass
(** {!Ssa_vector_sum.program}: sums no enclosing loop's lanes took are scheduled
    along their own axis. Reorders a sum, so only a plan whose numerical policy
    permits it runs this. *)

val contract : scalar:bool -> pass
(** {!Ssa_opt_contract.pass}: [a + x * y] becomes one fused operation. Only a
    plan whose numerical policy permits contraction, on a target with a fused
    operation, runs this. *)

val pipeline :
  ?target:Ssa_target.t -> alias:Ssa_effects.policy -> unit -> pass list
(** The default order. Each pass sees the result of the last. With a [target]
    the vectorizer runs after the guards and the hoisting it relies on and
    before the scalar blocking, which takes only what the vectorizer left. *)

type report = (string * int) list
(** For each pass of the pipeline, how many rounds it changed something in. *)

val run :
  ?alias:Ssa_effects.policy ->
  ?target:Ssa_target.t ->
  ?passes:pass list ->
  Ssa_program.t ->
  Ssa_program.t * report
(** Repeats the passes until none changes anything, a bounded number of times.
    [alias] (default conservative) says which buffers may share memory: a load
    is hoisted or shared past a write only under a policy that proves the write
    cannot reach it. [Invalid_argument] if a pass returns a revision that does
    not verify: that is a defect in the pass. *)
