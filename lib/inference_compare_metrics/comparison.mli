(** One comparison: metadata, the element rule, every tensor metric, and the
    verdict that joins them.

    The verdict is the conjunction of all required checks, in this precedence:
    [Metadata_error] (specs differ; no element is ever counted as checked),
    [Unsupported] (an element rule that names a registered comparator),
    [Metric_unavailable] (a required metric that is undefined, unsupported,
    nonfinite, failed, or scored fewer groups than expected), [Mismatch] (an
    element-rule or required-threshold violation) and [Pass]. Diagnostic metrics
    and bitwise/statistics findings stay visible but never change it. *)

open Inference_compare_schema
module Codecs = Inference_compare_codecs

type error =
  | Config of Metric_config_error.t
  | Element of Element_compare.error
  | Metric of Tensor_metrics.error

val pp_error : Format.formatter -> error -> unit

val compare :
  ?max_samples:int ->
  Codecs.Logical_value.Context.t ->
  Tensor_metrics.env ->
  Comparison_policy.t ->
  comparison_id:Id.Comparison.t ->
  policy_digest:Digest.Comparison_policy.t ->
  left:Codecs.Logical_value.t ->
  right:Codecs.Logical_value.t ->
  (Comparison_result.t, error) Err.t
(** A configuration fault is an [Error]; a disagreement is a result. *)
