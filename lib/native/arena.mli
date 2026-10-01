(* The memory behind an [Arena_plan.t]: one Bigarray pool per element kind, and
   typed slots taken out of it. See .ai/ on the tensor arena.

   OWNERSHIP. A view returned by [view] is valid for the run that acquired it, no
   longer: the plan reuses its cells once the edge is released, so nothing may
   hold one past its release. [Eval_direct.run] never returns one (outputs and
   retained edges are ineligible, so they are never arena-backed), and a node
   executor must not keep its operands or [dst] past the call.

   [create] allocates every pool up front, so a partial arena never exists and
   an out-of-memory is one error row, not a failure in the middle of a run. *)

open Core.Storage_units

module Poison : sig
  (** A value written into a slot each time it is acquired, so a cell the writer
      forgot to overwrite holds poison instead of the previous tenant's data.
      Two poisons differ in every kind, and both are representable in it, so two
      runs under different poisons must agree bit for bit. A diagnostic, not a
      proof: downstream compute can mask a missed cell. *)
  type t = A | B
end

module Admission : sig
  type t =
    | Best_effort
        (** A plan the arena cannot take (over a ceiling, or the pools cannot be
            allocated) falls back to a release-only run. *)
    | Required of Byte_size.t
        (** The scoped footprint (the pools plus the script's out-of-arena
            payload bytes, in bytes) must fit the budget, or the run is rejected
            before it starts. There is no release-only fallback. *)
end

module Over_budget : sig
  type t = { footprint : Byte_size.t; budget : Byte_size.t }
end

module Alloc_failed : sig
  type t = { kind : Alloc_script.Kind.t; bytes : Byte_size.t }
end

(** What a backend guarantees about a pool's base address. [Logical_only]: slot
    offsets are aligned relative to their pool, but the pool's storage is
    whatever the allocator returns (malloc's, or an ArrayBuffer's), so a slot is
    physically aligned only as far as that base happens to be. *)
module Physical_alignment : sig
  type t = Logical_only

  val pp : Format.formatter -> t -> unit

  val backend : t
  (** What this backend gives every pool. *)
end

(** Whether a run accepts slots aligned only relative to their pool.
    [Physical_required] refuses a plan whose base alignment the backend cannot
    give, rather than silently reducing it to a logical one. *)
module Physical_requirement : sig
  type t = Logical_accepted | Physical_required
end

module Physical_unsupported : sig
  type t = { required : Byte_alignment.t; provided : Physical_alignment.t }
end

type error =
  [ `Arena_alloc_failed of Alloc_failed.t
  | `Arena_busy
  | `Arena_script_mismatch of Alloc_script.Difference.t
  | `Arena_slot of Tensor_id.t
  | `Over_budget of Over_budget.t
  | `Physical_alignment_unsupported of Physical_unsupported.t
  | `Storage_script_mismatch of Alloc_script.Position.t ]
(** [`Arena_slot id]: the slot of [id] does not fit its edge's format.
    [`Storage_script_mismatch]: a storage plan's script differs from the run's
    at that event ([Storage_script.first_difference]). *)

val pp_error : Format.formatter -> [< error ] -> unit

type t

val create : ?poison:Poison.t -> Arena_plan.t -> (t, [> error ]) Err.t
(** Allocates every pool. Out of memory is [`Arena_alloc_failed]. *)

val plan : t -> Arena_plan.t

val physical_alignment : t -> Physical_alignment.t
(** Every backend today is [Logical_only]. *)

val footprint :
  Arena_plan.t -> (Byte_size.t, [> `Peak_bytes_overflow of Tensor_id.t ]) Err.t
(** The scoped footprint: the pools plus the script's out-of-arena payload
    bytes, saturating at [Int64.max_int]. It says nothing about the process's
    total memory. *)

val view : t -> Tensor_id.t -> (Tensor.packed option, [> error ]) Err.t
(** The slot of an eligible edge, poisoned first when the arena has a poison;
    [None] for an edge the plan does not place. Views of edges whose lifetimes
    overlap never share cells. *)

(** Mixed-mode copies: results a node executor returned in storage other than
    the destination it was given, copied into the slot. *)
module Copies : sig
  type t = { count : int64; bytes : Byte_size.t }
end

val copies : t -> Copies.t
val record_copy : t -> bytes:Byte_size.t -> unit
val busy : t -> bool

val acquire : t -> (unit, [> `Arena_busy ]) Err.t
(** Marks the arena in use until {!release}: for storage that outlives a run (a
    result lease). [`Arena_busy] when it already is. *)

val release : t -> unit

val with_run :
  t -> (unit -> ('a, ([> `Arena_busy ] as 'e)) Err.t) -> ('a, 'e) Err.t
(** Marks the arena in use for the duration of [f], on every exit path. Entering
    a busy arena is [`Arena_busy]: a run's views would otherwise be overwritten
    by another's. *)

val settle :
  t ->
  Tensor_id.t ->
  dst:Tensor.packed ->
  result:Tensor.packed ->
  (Tensor.packed, [> Tensor.dst_error ]) Err.t
(** The tensor to bind for an arena-backed edge: [dst] when [result] is exactly
    the destination the executor was handed, otherwise [result]'s cells copied
    into [dst] (and counted). The test is descriptor identity, which asks "did
    the arm return the object it was given", and says nothing about whether two
    tensors share memory. *)

val fill_for_test : t -> Poison.t -> unit
(** Overwrites every pool with a poison, to show that nothing a run returned
    reads from one. *)
