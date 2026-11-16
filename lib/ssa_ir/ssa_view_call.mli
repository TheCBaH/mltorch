(** A contraction called on views of its buffers: the descriptor of
    {!Contraction} evaluated with every operand's cells read from, and written
    to, one dynamic element offset past the buffer's first cell. A band driver
    calls the placed kernel once per block this way, each call naming the rows
    it covers by a base value, where {!Ssa_op.Contract} names whole buffers.

    A view's element [x] of the descriptor lies at [base + x]; the descriptor's
    strides are unchanged. The bases are [index] values. The verifier proves the
    whole footprint, base and last row, column and step together, inside the
    buffer ({!Ssa_verify}), so a call never fails and checks nothing. *)

module View : sig
  type t = { buffer : Ssa_id.Buffer.t; base : Ssa_value.t }
  (** [base] is an [index] value: the element offset of the view's first cell.
  *)
end

type t = { desc : Contraction.t; a : View.t; b : View.t; c : View.t }
(** A and B of [desc]'s format, C binary32, C disjoint from A and B. *)

val operands : t -> Ssa_value.t list
(** The three bases, in the order A, B, C. *)

val map_operands : (Ssa_value.t -> Ssa_value.t) -> t -> t
val reads : t -> Ssa_id.Buffer.t list
val writes : t -> Ssa_id.Buffer.t

(** The last element offset the descriptor touches, from each view's base. *)
module Reach : sig
  type t = { a : int64; b : int64; c : int64 }
end

val reach : Contraction.t -> Reach.t

val eval :
  t ->
  base:(Ssa_value.t -> int64) ->
  read:(Ssa_id.Buffer.t -> int64 -> float) ->
  write:(int64 -> float -> unit) ->
  unit
(** {!Contraction.eval} with each operand's offsets moved by its base. [read]
    takes an absolute element offset of the named buffer, [write] one of C. *)
