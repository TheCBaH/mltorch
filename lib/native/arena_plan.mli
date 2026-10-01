(* A static placement of a run's eligible intermediates in per-kind pools,
   computed from a dry run's [Alloc_script] (see .ai/ on the tensor arena).

   One allocation problem per element kind: a pool holds cells of a single
   Bigarray kind. The allocator places each slot's exact bytes, its start at
   its script alignment ([Alignment_policy]); padding enters only the bounds.
   Slots and pools are reported in cells of their kind. Every
   placement is witnessed by [Interval_alloc.check] before it is kept, and the
   plan keeps the whole script it was built from, so a run can require that its
   own dry run equals it.

   Every figure here is for the node order the script was produced in: a plan
   cannot beat the lower bound that order sets. *)

open Core.Storage_units
module Kind = Alloc_script.Kind

(** Where an eligible edge lives: [bytes] ([numel] cells) from [offset] in its
    kind's pool. [offset] is a multiple of the edge's script alignment, and so a
    whole number of cells. *)
module Slot : sig
  type t = {
    id : Tensor_id.t;
    signature : Tensor_sig.t;
    kind : Kind.t;
    offset : Byte_offset.t;
    bytes : Byte_size.t;
    numel : Element_count.t;
  }
end

(** One kind's pool. [alignment] is the strictest of its slots': the base
    alignment the pool's storage needs for every slot's offset to be aligned in
    memory, not only relative to the pool. *)
module Pool : sig
  type t = {
    kind : Kind.t;
    bytes : Byte_size.t;
    numel : Element_count.t;
    alignment : Byte_alignment.t;
  }
end

module Stats : sig
  type t = {
    kinds : (Kind.t * Interval_alloc.Stats.t) list;
        (** In bytes, per kind that has any eligible edge. *)
    pool_bytes : Byte_size.t;  (** The pools' bytes, summed. *)
    per_kind_bound_bytes : Byte_size.t;
        (** The kinds' padded live bounds, summed: what separate pools of padded
            slots cannot beat. Slots are placed exact, so [pool_bytes] can sit
            below it by up to the padding. *)
    combined_bound_bytes : Byte_size.t;
        (** The padded live bound of one pool shared by every kind. The gap to
            [per_kind_bound_bytes] is the cross-kind loss. *)
    out_of_arena_bytes : Byte_size.t;
        (** The script's ineligible allocations (outputs, retained and quantized
            edges): they stay outside the arena. *)
  }

  val pp : Format.formatter -> t -> unit
end

type t

(** What crossed a ceiling. *)
module Over_limit : sig
  type limit = Bytes of Byte_size.t | Elements of Element_count.t

  type t = {
    kind : Kind.t;
    numel : Element_count.t;
    bytes : Byte_size.t;
    limit : limit;
  }
end

(** The allocator's own errors, keyed by edge. Reaching one means a placement
    the allocator built failed its own checker, or a size overflowed [int64]. *)
module Placement_error : sig
  type t =
    [ `Duplicate_placement of Tensor_id.t
    | `Live_overflow of Tensor_id.t
    | `Misaligned of Tensor_id.t Interval_alloc.Misaligned.t
    | `Offset_overflow of Tensor_id.t
    | `Out_of_pool of Tensor_id.t Interval_alloc.Out_of_pool.t
    | `Overlap of Tensor_id.t Interval_alloc.Overlap.t
    | `Pool_overflow of Tensor_id.t
    | `Unknown_key of Tensor_id.t
    | `Unplaced of Tensor_id.t ]
end

type error =
  [ `Arena_over_limit of Over_limit.t
  | `Arena_placement of Placement_error.t
  | `Arena_script of Tensor_id.t ]
(** [`Arena_script id]: the script allocates or frees [id] inconsistently. *)

val place :
  ?budget:Interval_alloc.Budget.t ->
  Arena_problem.Kind_problem.t ->
  ( Tensor_id.t Interval_alloc.Witness.t * Interval_alloc.Stats.t,
    [> `Arena_placement of Placement_error.t ] )
  Err.t
(** One kind's placement: the portfolio on the exact script and on the padded
    one, each checked against the exact script and cut to its actual length,
    whichever is shorter (exact on a tie). A padded placement is an exact one
    too, so a plan never holds more than padding would have. The stats are the
    chosen run's, with its actual pool and the exact script's live bound. *)

val create :
  ?limits:Kernel.Limits.t ->
  ?budget:Interval_alloc.Budget.t ->
  ?alignment:Alignment_policy.t ->
  Alloc_script.t ->
  (t, [> error ]) Err.t
(** Plans the eligible edges. [budget] is the order search's iteration budget
    per kind. A pool over [Kernel.Limits.Hard.numel] cells or [limits.max_bytes]
    bytes is [`Arena_over_limit]: a partial arena is never built. [alignment]
    (default [standard]) is the policy [script] was dry-run under; the plan
    keeps it, and a run of the plan dry-runs under it again. *)

val script : t -> Alloc_script.t
(** The witnessed script: what a run's own dry run must equal. *)

val slot : t -> Tensor_id.t -> Slot.t option
val slots : t -> Slot.t list
val pools : t -> Pool.t list
val stats : t -> Stats.t

val base_alignment : t -> Byte_alignment.t option
(** The strictest pool alignment; [None] when the plan has no pool. *)

val policy : t -> Alignment_policy.t
(** The alignment policy the plan's script, and so every slot, was made under.
    It is part of the plan's identity: a run compares scripts, alignments
    included. *)

val revalidate : t -> Alignment_policy.t -> (t, [> error ]) Err.t
(** The same placement under another policy, its script and pools realigned, or
    [`Arena_placement (`Misaligned _)] for the first slot whose offset the new
    policy's alignment does not divide. A plan is never reused under a stronger
    policy without this check. *)
