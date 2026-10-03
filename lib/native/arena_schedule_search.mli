(* Bounded deterministic beam search over the scheduling states of
   [Arena_schedule_greedy.State], and the full engine that combines it with the
   constructive schedules. The beam only ever adds a candidate: the original
   and both greedy orders stay incumbents, so the answer is never worse. See
   .ai/ on the arena scheduling design. *)

module Problem = Arena_schedule_problem
module Position = Problem.Position

module Outcome : sig
  type t = {
    order : Position.t array option;
        (** The best complete order found, if the beam finished. A layer cut
            short by a limit is discarded, never completed heuristically. *)
    stop : Arena_schedule.Stop.t;
    stats : Arena_schedule.Stats.t;
  }
end

val beam :
  Problem.t -> Arena_schedule.Limits.t -> (Outcome.t, [> Problem.error ]) Err.t
(** Layer by layer: every ready choice of every retained state is one expansion,
    and the best [width] children survive, ranked by
    [(max (peak, bound), peak, live, all-allocation peak, prefix order)]. With
    no expansion budget the beam does not run. *)

val portfolio :
  Arena_schedule.Config.t ->
  Problem.t ->
  ( Arena_schedule_greedy.Candidate.t list
    * Arena_schedule.Stats.t
    * Arena_schedule.Stop.t,
    [> Problem.error ] )
  Err.t
(** The original order, both greedy orders and the beam's, without repeats, in
    that order; the stop reason is the beam's. *)

val stop_of :
  selected:Arena_schedule.Stop.t ->
  beam_stop:Arena_schedule.Stop.t ->
  Arena_schedule.Stop.t
(** The result's stop reason: the bound when selection proved it, otherwise the
    limit the beam ended on, otherwise completion. *)

val run :
  Arena_schedule.Config.t ->
  Graph_ir.graph ->
  (Arena_schedule.Result.t, [> Problem.error ]) Err.t
(** {!Arena_schedule_greedy.run} plus, when [limits.expansions > 0], the beam's
    order as a further candidate. The stop reason says why the search ended. *)
