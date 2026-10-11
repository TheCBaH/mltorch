(** Typed selector evaluation over the endpoint graphs and the supplied cluster
    mapping.

    {!check} rejects a selector before anything is evaluated: syntax and size
    bounds, unknown references and port arity. {!eval} then decides, for one
    comparison site, whether a [Metric_selector.t] matches, and returns one
    deterministic witness per satisfying anchor. [All_of] over tensor selectors
    constrains the {e same} anchored tensor and, for [Produced_by] /
    [Consumed_by], the same use occurrence; chains use [Flows_to], matched by
    layered dynamic programming over indexed port transitions (no path
    enumeration). Paths follow data edges only. *)

open Inference_compare_schema

type ctx = {
  left : Inference_compare_plan.Graph_index.t;
  right : Inference_compare_plan.Graph_index.t;
  partition : Inference_compare_plan.Partition_check.t;
  limits : Resource_limits.t;
}

val check :
  ctx ->
  Comparison_site.t list ->
  rule:Id.Metric_rule.t ->
  Metric_selector.t ->
  (unit, Policy_error.t) Err.t
(** [Invalid_selector] for an unknown comparison, canonical value, member pair,
    unit, output occurrence, tensor, node or an out-of-arity port;
    [Resource_exhausted] for AST size, depth or path length over the limits. *)

val eval :
  ctx ->
  Comparison_site.t ->
  Metric_selector.t ->
  Topology_witness.t list option
(** [None] when the site does not match; otherwise the witnesses of the
    satisfying anchors, sorted by side and anchor (possibly empty for selectors
    that name no anchors). *)
