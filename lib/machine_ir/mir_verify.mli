(** The generic verifier. A {!Generic.t} exists only as its result: a pass that
    takes one cannot be handed an unchecked program. *)

module Generic : sig
  type t

  val program : t -> Mir_program.generic
  val revision : t -> Mir_id.Revision.t
end

val generic : Mir_program.generic -> (Generic.t, Mir_diagnostic.t) Err.t
(** Structure (ids, reachability, dominance, edges, the order chain, objects,
    helpers), typing, and the generic rules: a shift count is a constant below
    the width; a [Idiv] or [Fto_sint] is dominated by guard evidence for its
    whole domain (a nonzero divisor and not [min / -1]; [-2^63 <= x < 2^63],
    which a NaN fails); a load or store statically derived from a view respects
    its permission; a failure payload matches its kind's schema; a return
    matches the function's results. *)
