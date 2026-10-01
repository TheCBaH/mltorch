(* A run's results, leased from the arena they were computed in (see .ai/ on
   the tensor arena). A lease pins that whole arena (the run's [Outputs]
   arena, or its whole [Execution] arena under a shared layout) until
   [release]: returning from a run is not permission to overwrite its results,
   so no other run can acquire the arena while the lease is live. It also
   keeps the constant version the run read alive, since a forwarded constant
   is a result too.

   Access is scoped: [with_outputs] fails once the lease is released. A raw
   view that escaped the scope before then cannot be revoked; it is never
   overwritten while the lease is live, and after [release] it reads whatever
   the next run writes. Take a copy ([copy_out]) to keep results past the
   lease. *)

open Core.Storage_units

module Generation : sig
  type t

  val equal : t -> t -> bool
  val pp : Format.formatter -> t -> unit
end

type t

val make :
  arena:Arena.t option ->
  constants:Constant_arena.t ->
  generation:Generation.t ->
  signatures:Tensor_sig.t Tensor_id.Map.t ->
  Tensor.packed Tensor_id.Map.t ->
  t
(** For [Storage_run]: [arena], already acquired, is released by {!release}. *)

val first_generation : Generation.t
val next_generation : Generation.t -> Generation.t
val generation : t -> Generation.t
val constants : t -> Constant_arena.t
val released : t -> bool

val with_outputs :
  t -> (Tensor.packed Tensor_id.Map.t -> 'a) -> ('a, [> `Lease_released ]) Err.t
(** [f] over the results, while the lease is live. *)

val release : t -> unit
(** Unpins the arena. Idempotent. *)

val copy_out :
  t ->
  ( Tensor.packed Tensor_id.Map.t * Arena.Copies.t,
    [> `Lease_released | `Quant_missing of Tensor_id.t | Tensor.dst_error ] )
  Err.t
(** Every result copied into fresh storage, then the lease released: the copies
    alias no arena. Both sets exist at once during the copy, so its bytes count
    in a peak as well as in copy traffic. *)

val pp_lease_released : Format.formatter -> [< `Lease_released ] -> unit

val bytes : t -> Byte_size.t
(** The results' payload bytes. *)
