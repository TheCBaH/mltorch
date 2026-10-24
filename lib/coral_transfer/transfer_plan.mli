(** Where a generated contraction function reads A and B from: in place, copied
    whole into DTCM panels of its own by one DMA chain before the product, or (A
    only) streamed by blocks of rows through two DTCM halves, each block's chain
    started beside the previous block's product. A plan is data; only {!check}
    turns one into the {!Witness.t} an emitter takes, so an emitted function's
    panels fit the DTCM it was given and its blocks are ones the DMA can move.
*)

module A : sig
  type t = Copied | In_place | Streamed of { rows : int }
end

module B : sig
  type t =
    | Copied
    | In_place
    | Shared
        (** copied, three quarters by the core's RVV copy beside the DMA's
            quarter, as the DMA suite's [b-shared] prologue *)
end

(** Who stores C: the core's stores straight to SRAM, or (A streamed) into the
    ring half beside the block's A, the DMA carrying each block's C out beside
    the next block's product. *)
module Out : sig
  type t = Core | Dma
end

type t = { a : A.t; b : B.t; out : Out.t }

val in_place : t
val pp : t Fmt.t

(** Why a plan cannot be emitted. Alphabetical. *)
module Refusal : sig
  type t =
    | Block_rows of { rows : int; unit : int }
        (** a block of rows not a positive multiple of the fewest rows whose
            bytes are whole beats and whole k steps *)
    | C_not_beats  (** C by the DMA, its rows not whole 16-byte beats *)
    | C_not_streamed  (** C by the DMA, A not streamed *)
    | C_rows_not_contiguous  (** C by the DMA, C's rows not one after another *)
    | Over_budget of { needed : int; budget : int }
        (** panels and ring larger than the DTCM given *)
    | Over_transfer of { bytes : int }  (** a copy past one DMA transfer *)
    | Rows_not_contiguous
        (** a streamed A whose rows do not follow each other *)

  val pp : t Fmt.t
end

val span : Contraction.t -> [ `A | `B ] -> int
(** Bytes an operand spans from its first element, rounded up to a 16-byte beat:
    what a copy moves. *)

val group_rows : Contraction.t -> int
(** The rows a k step reads together: three fused or integer, two ordered. *)

val row_bytes : Contraction.t -> int
(** One row of A, [k] elements. *)

val unit_rows : ?out:Out.t -> Contraction.t -> int
(** The fewest rows a ring block may hold: a multiple of {!group_rows} whose
    bytes are whole beats, and, with C by the DMA, its rows of C too. *)

val c_row_bytes : Contraction.t -> int

val half_bytes : Contraction.t -> rows:int -> out:Out.t -> int
(** A ring half: a block of A's rows, then, with C by the DMA, its rows of C. *)

val rows_contiguous : Contraction.t -> bool
(** A's rows follow each other ([k] stride 1, row stride [k]). *)

val bytes : Contraction.t -> t -> int
(** The DTCM the plan's panels and ring take. *)

(** A checked plan, for the contraction it was checked against. *)
module Witness : sig
  type plan := t
  type t

  val plan : t -> plan
  val contraction : t -> Contraction.t
  val bytes : t -> int
end

val check : budget:int -> Contraction.t -> t -> (Witness.t, Refusal.t) result
(** The plan for [d] within [budget] bytes of DTCM. *)

val candidates : budget:int -> Contraction.t -> t list
(** Every plan worth ranking for [d]: B in place, copied or shared; A in place,
    copied, or streamed by the largest block the rest of the budget holds, with
    C by the core or the DMA; each one passes {!check}, and the in-place plan is
    always among them. *)
