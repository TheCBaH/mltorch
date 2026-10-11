(** Deterministic resolution of a {!Policy_selection.t} into one concrete policy
    per comparison site.

    Base policy: an explicit binding is authoritative (a matching
    [Select_policy] rule must name the same policy); otherwise the matching
    [Select_policy] rules of highest priority, whose policies must be identical;
    otherwise the default. Every matching [Add_metrics] rule then contributes
    its metrics regardless of priority; identical metric ids merge only when the
    complete specifications are equal, and a reused id with a different
    specification is a conflict. List order has no precedence, and
    lower-priority matches stay visible without contributing a base. The result
    is validated against the site's spec before it is returned. *)

open Inference_compare_schema

type result = {
  bindings : Policy_binding.t list;
      (** Sorted by comparison id; [policy_digest] is filled by sealing. *)
  empty_matches : Id.Metric_rule.t list;
      (** Rules that matched no site under [empty_match = Allow]. *)
}

val resolve :
  ctx:Selector_eval.ctx ->
  codecs:Inference_compare_codecs.Element_codec.Registry.t ->
  env:Inference_compare_metrics.Tensor_metrics.env ->
  placeholder_digest:Digest.Comparison_policy.t ->
  Comparison_site.t list ->
  Policy_selection.t ->
  (result, Policy_error.t) Err.t
(** [placeholder_digest] fills [Policy_binding.policy_digest] until sealing
    replaces it. *)
