(** A contraction called on views of its buffers: the descriptor of
    {!Contraction} evaluated with every operand's cells read from, and written
    to, past dynamic element offsets. A band driver calls the placed kernel once
    per block this way, each call naming the rows it covers by base values,
    where {!Ssa_op.Contract} names whole buffers.

    An operand is placed one of two ways. By one base ([Base]), element [x] of
    the descriptor lies at [base + x] and the descriptor's strides are
    unchanged. By one base per row ([Rows]), for an operand whose rows may sit
    anywhere (a rolling window that wraps), the operand's outer axis (A's and
    C's rows [i], B's columns [j]) takes the next base from the list, which must
    have as many bases as that axis has extent, and the descriptor's outer
    stride does not apply: the cell is [rows.(i) + batch + inner]. The bases are
    [index] values. The verifier proves, for every base, the whole footprint
    inside the buffer ({!Ssa_verify}), so a call never fails and checks nothing.
*)

(** Where an operand's cells start. Alphabetical. *)
module At : sig
  (* @alphabetical *)
  type t = Base of Ssa_value.t | Rows of Ssa_value.t list
end

module View : sig
  type t = { buffer : Ssa_id.Buffer.t; at : At.t }
end

type t = { desc : Contraction.t; a : View.t; b : View.t; c : View.t }
(** A and B of [desc]'s format, C binary32, C disjoint from A and B. *)

val bases : View.t -> Ssa_value.t list

val operands : t -> Ssa_value.t list
(** Every base of A, then B, then C, in order. *)

val map_operands : (Ssa_value.t -> Ssa_value.t) -> t -> t
val reads : t -> Ssa_id.Buffer.t list
val writes : t -> Ssa_id.Buffer.t

val rows_expected : t -> bool
(** Whether every [Rows] list has the extent of the axis it names. *)

module Reach : sig
  type t = { a : int64; b : int64; c : int64 }
end

val reach : t -> Reach.t
(** The last element offset the descriptor touches from each operand's base
    (from each of its row bases, for [Rows]). *)

val eval :
  t ->
  base:(Ssa_value.t -> int64) ->
  read:(Ssa_id.Buffer.t -> int64 -> float) ->
  write:(int64 -> float -> unit) ->
  unit
(** {!Contraction.eval} with each operand's cells placed as above. [read] takes
    an absolute element offset of the named buffer, [write] one of C. *)
