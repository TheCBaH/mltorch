(* A static placement of a run's eligible intermediates in per-kind pools,
   computed from a dry run's [Alloc_script] (see .ai/ on the tensor arena).

   One allocation problem per element kind: a pool holds cells of a single
   Bigarray kind, so its unit is an element and no alignment is needed. Every
   placement is witnessed by [Interval_alloc.check] before it is kept, and the
   plan keeps the whole script it was built from, so a run can require that its
   own dry run equals it.

   Every figure here is for the node order the script was produced in: a plan
   cannot beat the lower bound that order sets. *)

module Kind = Alloc_script.Kind

(** Where an eligible edge lives: [numel] cells from [offset] in its kind's
    pool. *)
module Slot : sig
  type t = {
    id : Tensor_id.t;
    signature : Tensor_sig.t;
    kind : Kind.t;
    offset : int64;
    numel : int64;
  }
end

(** One kind's pool. *)
module Pool : sig
  type t = { kind : Kind.t; numel : int64; bytes : int64 }
end

module Stats : sig
  type t = {
    kinds : (Kind.t * Interval_alloc.Stats.t) list;
        (** In elements, per kind that has any eligible edge. *)
    pool_bytes : int64;  (** The pools' bytes, summed. *)
    per_kind_bound_bytes : int64;
        (** The kinds' lower bounds in bytes, summed: what separate pools cannot
            beat. *)
    combined_bound_bytes : int64;
        (** The lower bound of one pool shared by every kind. The gap to
            [per_kind_bound_bytes] is the cross-kind loss. *)
    out_of_arena_bytes : int64;
        (** The script's ineligible allocations (outputs, retained and quantized
            edges): they stay outside the arena. *)
  }

  val pp : Format.formatter -> t -> unit
end

type t

(** What crossed a ceiling. *)
module Over_limit : sig
  type limit = Bytes of int64 | Elements of int64
  type t = { kind : Kind.t; numel : int64; bytes : int64; limit : limit }
end

(** The allocator's own errors, keyed by edge. Reaching one means a placement
    the allocator built failed its own checker, or a size overflowed [int64]. *)
module Placement_error : sig
  type t =
    [ `Duplicate_placement of Tensor_id.t
    | `Live_overflow of Tensor_id.t
    | `Negative_offset of Tensor_id.t
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

val create :
  ?limits:Kernel.Limits.t ->
  ?budget:Interval_alloc.Budget.t ->
  Alloc_script.t ->
  (t, [> error ]) Err.t
(** Plans the eligible edges. [budget] is the order search's iteration budget
    per kind. A pool over [Kernel.Limits.Hard.numel] cells or [limits.max_bytes]
    bytes is [`Arena_over_limit]: a partial arena is never built. *)

val script : t -> Alloc_script.t
(** The witnessed script: what a run's own dry run must equal. *)

val slot : t -> Tensor_id.t -> Slot.t option
val slots : t -> Slot.t list
val pools : t -> Pool.t list
val stats : t -> Stats.t
