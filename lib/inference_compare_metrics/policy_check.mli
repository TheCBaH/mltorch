(** Configuration validation of a comparison policy against the spec of the
    values it will compare. Everything here is decidable before inference: plan
    validation calls it for every site, and {!Comparison.compare} calls it again
    so a policy never executes unchecked. *)

open Inference_compare_schema

val validate :
  codecs:Inference_compare_codecs.Element_codec.Registry.t ->
  Tensor_metrics.env ->
  Tensor_spec.t ->
  Comparison_policy.t ->
  (unit, Metric_config_error.t) Err.t
(** Checks, in order: the element rule fits the codec; integer and Boolean
    codecs carry an exact rule; a float rule's scale is the policy's relative
    scale; no metric id is reused with a different specification; every metric
    validates against the spec; and at least one required check exists. *)

val required_checks : Comparison_policy.t -> int
(** Element rule plus required metrics, identical duplicates counted once. *)

val distinct_metrics : Comparison_policy.t -> Tensor_metric.t list
(** Metrics with identical duplicates removed, in first-occurrence order. *)
