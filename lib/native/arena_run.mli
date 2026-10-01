(* Planning and creating an arena for one run of a graph: the step between "I want
   this graph run in an arena" and [Eval_direct.run ?arena]. See .ai/ on the
   tensor arena.

   Under [Best_effort] a plan the arena cannot take is not an error: the run
   proceeds without one and the reason is reported. Under [Required budget] it is
   an error before anything is evaluated, and there is no fallback. *)

open Graph_ir

type error =
  [ Arena.error
  | Arena_plan.error
  | Eval_direct.error
  | `Peak_bytes_overflow of Tensor_id.t ]

val pp_error : Format.formatter -> [< error ] -> unit

type outcome =
  | Arena of Arena.t
  | Release_only of error
      (** [Best_effort] declined, with the row [Required] would have failed
          with. *)

val acquire :
  ?limits:Kernel.Limits.t ->
  ?budget:Interval_alloc.Budget.t ->
  ?poison:Arena.Poison.t ->
  ?retain:Release_schedule.Retain.t ->
  admission:Arena.Admission.t ->
  graph ->
  (outcome, error) Err.t
(** Dry-runs [graph] under [retain] (default: keep everything, so nothing is
    eligible), plans it, checks the admission and allocates the pools. The arena
    must then be run on that same graph and [retain]. *)

(** What an arena run held, for a caller's stats. Pool bytes, the script's
    out-of-arena payload bytes, and mixed-mode copies are reported separately:
    none of them is the process's total memory. *)
module Report : sig
  type t = {
    pool_bytes : int64;
    out_of_arena_bytes : int64;
    copies : Arena.Copies.t;
  }
end

val report :
  Arena.t -> (Report.t, [> `Peak_bytes_overflow of Tensor_id.t ]) Err.t

(** How a run that asked for an arena went. *)
module Outcome : sig
  type t =
    | Declined of error  (** [Best_effort] ran without one, for this reason *)
    | Used of Report.t
end

val with_arena :
  ?limits:Kernel.Limits.t ->
  ?budget:Interval_alloc.Budget.t ->
  ?poison:Arena.Poison.t ->
  ?retain:Release_schedule.Retain.t ->
  admission:Arena.Admission.t ->
  graph ->
  (Arena.t option -> ('a, error) Err.t) ->
  ('a * Outcome.t, error) Err.t
(** [with_arena ~admission g f] acquires an arena and runs [f] with it ([None]
    when [Best_effort] declined). Under [Required] an arena that cannot be had
    is an error and [f] is never called, so nothing is evaluated. [f] must run
    [g] under the same [retain]. *)
