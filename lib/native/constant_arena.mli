(* A model's constants, prepared once and shared read-only by every run of it
   (see .ai/ on the tensor arena). Under [Copied] ownership each used constant
   is copied into the plan's [Constants] arena at [create], and that copy is
   what runs bind; under [Borrowed] the caller's own payloads are bound in
   place and nothing is copied. Either way nothing here is ever reclaimed or
   poisoned: constants are not scratch between runs.

   A prepared set is one version of the model's constants. Replacing a model
   prepares a new version in new storage, so it can never overwrite what an
   older run or a still-leased result reads: an older version stays alive for
   as long as anything references it. The payloads are read-only by contract:
   a view cannot be made immutable, so nothing here writes to one after
   [create], and no run may either. *)

open Core.Storage_units

module Generation : sig
  type t

  val equal : t -> t -> bool
  val pp : Format.formatter -> t -> unit
end

type t

val create :
  Storage_plan.t ->
  constants:(Tensor_id.t * Tensor.packed) list ->
  ( t,
    [> Arena.error | `Missing_constant of Tensor_id.t | Tensor.dst_error ] )
  Err.t
(** Prepares every constant the plan's script uses, from [constants]. Allocates
    the constant arena's pools (no poison) and copies each copied constant into
    its slot. *)

val script : t -> Storage_script.t
(** The script of the plan it was prepared from. *)

val generation : t -> Generation.t
(** Distinct for every [create]. *)

val bindings : t -> (Tensor_id.t * Tensor.packed) list
(** What a run binds for the constants: the slot views, or the borrowed
    payloads, in the script's order. *)

val loaded : t -> Arena.Copies.t
(** The copies [create] made into the arena: preparation cost, never a run's. *)

val arena_bytes : t -> Byte_size.t
(** The constant arena's pool bytes; zero when constants are borrowed. *)
