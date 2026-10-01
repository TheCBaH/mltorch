(* A static placement of every logical arena a [Storage_script] uses: one
   [Arena_plan] per arena, each over that arena's projection of the script
   ([Storage_script.arena_script]), so every arena has its own per-kind pools
   and witnessed slots (see .ai/ on the tensor arena). The plan keeps the
   script it was built from; a run requires its own storage script to equal
   it. *)

open Core.Storage_units
module Arena_id = Storage_script.Arena_id

type t

val create :
  ?limits:Kernel.Limits.t ->
  ?budget:Interval_alloc.Budget.t ->
  Storage_script.t ->
  (t, [> Arena_plan.error ]) Err.t
(** Plans every arena some block of the script lives in, under the script's
    alignment policy. A partial plan is never built. *)

val script : t -> Storage_script.t

val arenas : t -> (Arena_id.t * Arena_plan.t) list
(** In [Arena_id.all] order; an arena no block lives in has no plan. *)

val arena : t -> Arena_id.t -> Arena_plan.t option

(** What a plan holds, kept apart: none of it is the process's total memory.
    [constants] is the constant arena's pools, held once per resident model;
    [execution] every other arena's pools, held once per run slot (and, for the
    arena results live in, once per outstanding result); [borrowed] the payloads
    the caller keeps and the run only reads (borrowed constants and inputs);
    [outside] the peak of fresh allocations outside every arena (quantized
    edges). An output is counted in its arena's pools, never again as a payload.
*)
module Footprint : sig
  type t = {
    constants : Byte_size.t;
    execution : Byte_size.t;
    borrowed : Byte_size.t;
    outside : Byte_size.t;
  }

  val total : t -> Byte_size.t
  (** Saturating at [Int64.max_int]: a total past it is over every budget. *)

  val pp : Format.formatter -> t -> unit
end

val footprint :
  t -> (Footprint.t, [> `Peak_bytes_overflow of Tensor_id.t ]) Err.t
