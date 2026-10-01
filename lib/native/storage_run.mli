(* Runs a graph in the arenas of a [Storage_plan]: constants from a prepared
   [Constant_arena], copied inputs into their slots, intermediates and outputs
   into theirs, and the results leased ([Result_lease]) rather than returned
   (see .ai/ on the tensor arena).

   Arenas by role. The arena results live in (the [Outputs] arena, or the
   whole [Execution] arena under [Shared_execution]) is pinned by the lease
   until it is released; up to [max_outstanding] of them exist, created on
   demand, and a run with none free is [`Arena_busy] -- never an unbounded pool,
   and never a silent overwrite. Every other execution arena ([Inputs],
   [Intermediates]) is held for the run only and is reusable as soon as it
   returns, whatever the lease does.

   On failure, or an exception, every arena the run acquired is released
   before the error is returned: nothing it wrote was published, and neither
   constants nor another run's lease are touched. Poison, when given, is
   written into each slot as a run acquires it, never into constants. *)

open Graph_ir
open Core.Storage_units

type error = Arena_run.error

val pp_error : Format.formatter -> [< error ] -> unit

type t

val create :
  ?poison:Arena.Poison.t ->
  ?max_outstanding:int64 ->
  Storage_plan.t ->
  Constant_arena.t ->
  (t, [> error ]) Err.t
(** Allocates the per-run arenas. [max_outstanding] (default 1, at least 1)
    bounds the result arenas, so it bounds the leases alive at once. The
    constants must have been prepared from the same plan
    ([`Storage_script_mismatch] otherwise). *)

val plan : t -> Storage_plan.t

val run :
  t ->
  ?hooks:Eval_direct.hooks ->
  ?region_executor:Region_executor.t ->
  ?region_group_executor:Region_executor.group ->
  ?node_executor:Node_executor.t ->
  ?limits:Kernel.Limits.t ->
  ?retain:Release_schedule.Retain.t ->
  graph ->
  inputs:(Tensor_id.t * Tensor.packed) list ->
  (Result_lease.t, error) Err.t
(** One run under the plan's script ([retain] must be the one it was made
    under). The lease holds [g.outputs] and the retained edges. *)

val outstanding : t -> int64
(** Result arenas pinned by a live lease. *)

(** What the runner holds, for a caller's report: the plan's footprint (one
    run's worth), the result arenas actually allocated and how many are pinned,
    and the copies each role made. *)
module Report : sig
  type t = {
    footprint : Storage_plan.Footprint.t;
    constant_generation : Constant_arena.Generation.t;
    result_arenas : int64;
    result_arena_bytes : Byte_size.t;  (** Each one's pools. *)
    outstanding : int64;
    constant_loads : Arena.Copies.t;
    input_copies : Arena.Copies.t;
    mixed_mode_copies : Arena.Copies.t;
  }

  val pp : Format.formatter -> t -> unit
end

val report : t -> (Report.t, [> `Peak_bytes_overflow of Tensor_id.t ]) Err.t
