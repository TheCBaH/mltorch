(* Choosing among candidate orders by what the arena actually allocates.

   The scheduling engine minimizes payload; the pools a placement needs depend
   on alignment and fragmentation too, and neither implies the other. So the
   candidate orders ([Arena_schedule_search.portfolio]) are each placed by the
   existing witnessed planners under identical limits, budget and alignment,
   and the order with the fewest allocated pool bytes wins. The original order
   is always a candidate and wins every tie, so a feasible baseline is never
   made larger. The chosen graph and the plan built for it are returned
   together: execute that graph with that plan, never the original. See .ai/ on
   the arena scheduling design. *)

open Core.Storage_units

type error =
  [ Arena_schedule_problem.error
  | Arena_run.error
  | `Predicted_metrics_differ of Arena_schedule.Strategy.t ]
(** [`Predicted_metrics_differ]: a candidate's replayed metrics disagree with a
    fresh script of its reordered graph. A correctness failure, never a refusal.
*)

val pp_error : Format.formatter -> [< error ] -> unit

val is_refusal : error -> bool
(** The planning and admission rows that only mean "this order cannot be taken":
    a pool over a ceiling, over the budget, or physically unsupported.
    Everything else is a defect. *)

(** The plan placed for the chosen graph. *)
module Plan : sig
  type t = Intermediate of Arena_plan.t | Roles of Storage_plan.t
end

module Verdict : sig
  type t =
    | Planned of Byte_size.t
        (** Placed and admitted, with this many allocated pool bytes. *)
    | Refused of error
    | Skipped  (** Its payload peak is above the original's: not placed. *)
end

module Report : sig
  type t = {
    strategy : Arena_schedule.Strategy.t;
    metrics : Arena_schedule.Metrics.t;
    pool_bytes : Byte_size.t option;
        (** Whenever a plan exists, admitted or not. *)
    verdict : Verdict.t;
  }
end

module Selection : sig
  type t = {
    graph : Graph_ir.graph;  (** The graph to execute. *)
    plan : Plan.t option;
        (** Built for [graph]. [None] only when [Best_effort] admission declined
            every candidate, in which case [graph] is the original. *)
    strategy : Arena_schedule.Strategy.t;
    metrics : Arena_schedule.Metrics.t;
    baseline : Arena_schedule.Metrics.t;
    payload_winner : Arena_schedule.Strategy.t;
        (** What payload alone would have chosen. *)
    pool_bytes : Byte_size.t option;
    baseline_pool_bytes : Byte_size.t option;
    reports : Report.t list;  (** The original first. *)
    declined : error option;
        (** The reason a [Best_effort] run goes ahead without an arena. *)
    stop : Arena_schedule.Stop.t;
    stats : Arena_schedule.Stats.t;
  }
end

val choose :
  ?limits:Kernel.Limits.t ->
  ?budget:Interval_alloc.Budget.t ->
  ?admission:Arena.Admission.t ->
  ?physical:Arena.Physical_requirement.t ->
  Arena_schedule.Config.t ->
  Graph_ir.graph ->
  (Selection.t, error) Err.t
(** [admission] (default [Best_effort]) applies to intermediate mode only; role
    mode keeps the storage planner's own errors. When every candidate is
    refused, intermediate [Required] fails with the original order's refusal,
    [Best_effort] declines to the original with no plan, and role mode fails
    with the original order's error. A candidate may rescue a refused original.
*)
