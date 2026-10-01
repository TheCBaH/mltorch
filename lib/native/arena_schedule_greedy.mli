(* The constructive schedules: Kahn's algorithm over the ready set, scoring each
   ready node by the exact transition it would cause. The incremental [State]
   is immutable (persistent maps), so a search can keep many of them. See
   .ai/ on the arena scheduling design. *)

open Core.Storage_units
module Problem = Arena_schedule_problem
module Position = Problem.Position

module Score : sig
  type t = {
    peak : Byte_size.t;  (** the target peak after the node's outputs exist *)
    live : Byte_size.t;  (** target bytes live once its releases are done *)
    all_peak : Byte_size.t;  (** the same peak over every allocation *)
  }
end

module State : sig
  type t

  val start : Problem.t -> (t, [> Problem.error ]) Err.t
  (** Nothing scheduled; the ready set is every node without predecessors, and
      the problem's fixed prefix is already allocated. *)

  val ready : t -> Position.t list
  (** Ascending original position. *)

  val is_done : t -> bool
  val score : t -> Position.t -> (Score.t, [> Problem.error ]) Err.t

  val step : t -> Position.t -> (t, [> Problem.error ]) Err.t
  (** Schedules a ready node. *)

  val order : t -> Position.t array
  val peak : t -> Byte_size.t
  val live : t -> Byte_size.t
end

module Policy : sig
  type t = Live_first | Peak_first
end

val schedule :
  Problem.t ->
  Policy.t ->
  (Position.t array * Arena_schedule.Stats.t, [> Problem.error ]) Err.t
(** A complete valid order; ties go to the original position. *)

module Candidate : sig
  type t = {
    strategy : Arena_schedule.Strategy.t;
    order : Position.t array;
    metrics : Arena_schedule.Metrics.t;
  }
end

val candidates :
  Problem.t ->
  (Candidate.t list * Arena_schedule.Stats.t, [> Problem.error ]) Err.t
(** The original order, then peak-first and live-first; an order equal to an
    earlier one is dropped. *)

val select :
  Problem.t ->
  Candidate.t list ->
  Arena_schedule.Stats.t ->
  (Arena_schedule.Result.t, [> Problem.error ]) Err.t
(** The candidate with the smallest target peak, then total peak; the earliest
    wins a tie, so the original is displaced only by a strict improvement. *)

val run :
  Arena_schedule.Config.t ->
  Graph_ir.graph ->
  (Arena_schedule.Result.t, [> Problem.error ]) Err.t
(** Validates, builds the problem, schedules both ways and selects. Under
    [Retain.All] in intermediate mode nothing is ever released, so the original
    is returned. *)
