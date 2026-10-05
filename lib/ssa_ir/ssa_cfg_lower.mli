(** Structured program to control-flow graph. A [for] becomes a header block
    that tests the induction value against the bound, a body that ends in the
    increment and a back edge, and an exit block whose parameters are the loop's
    results; an [if] becomes a branch to two blocks that meet in a join whose
    parameters are its results; an ordered sum is a loop whose accumulator is a
    header parameter and whose terms are added on the back edge, in order. Every
    carried value is a parameter and every transfer an edge, so the simultaneous
    rebinding of a loop is the edge's own semantics.

    A zero-trip loop goes from its entry straight to its exit with its
    initializers. A value captured by a body is simply a dominating definition:
    nothing is closure-converted. No critical edge is made: a branch's targets
    each have one predecessor. *)

type error =
  | Step_leaves_domain
      (** A loop whose induction value plus its step may leave the index domain
          on the last increment: the increment is a checked-free in-domain
          addition and must be provable from the loop's bounds. *)

val pp_error : Format.formatter -> error -> unit

val program : Ssa_program.t -> (Ssa_cfg.t, error) result
(** The program must verify; the result verifies by {!Ssa_cfg_verify}. *)
