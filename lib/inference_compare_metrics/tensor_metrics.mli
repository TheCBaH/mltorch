(** Tensor metrics over axis groups.

    {!validate} catches every configuration fault from the tensor's spec alone
    (axes, threshold direction and range, dtype, registered implementation,
    support size), so planning rejects it before an engine is prepared.
    {!compute} then evaluates a metric on every group in canonical order and
    reports group coverage: a required threshold is tested on every group, a
    mean across groups is a separate explicit metric, and an unscored group
    makes the metric unavailable rather than silently narrowing the check. *)

open Inference_compare_schema
module Codecs = Inference_compare_codecs

type env = {
  registry : Metric_registry.t;
  supports : Metric_support.t -> float array option;
      (** The verified binary64 support payload of a histogram metric. *)
}

val empty_env : env

type error =
  | Config of Metric_config_error.t
  | Spec_mismatch
  | Unknown_codec of Id.Codec.t

val pp_error : Format.formatter -> error -> unit

val validate :
  env -> Tensor_spec.t -> Tensor_metric.t -> (unit, Metric_config_error.t) Err.t

val compute :
  ?max_samples:int ->
  Codecs.Logical_value.Context.t ->
  env ->
  Tensor_metric.t ->
  left:Codecs.Logical_value.t ->
  right:Codecs.Logical_value.t ->
  (Metric_result.t, error) Err.t
(** [max_samples] (default 8) bounds the retained finite group scores. *)

val satisfied : Tensor_metric.t -> Metric_result.t -> bool option
(** [None] for a diagnostic metric; otherwise [Some true] only when every group
    was scored and none violates the threshold. *)
