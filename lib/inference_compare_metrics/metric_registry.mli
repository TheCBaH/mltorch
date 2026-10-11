(** Registered metric implementations. A registered metric is pinned by
    comparator id, implementation digest and configuration digest, declares its
    legal threshold direction and a group-size budget, and receives finite
    operand groups only. Spatial optimal transport needs an explicit solver,
    geometry and budget registered here; nothing falls back to scalar sorting.
*)

open Inference_compare_schema

type impl = {
  comparator : Id.Comparator.t;
  implementation_digest : Digest.Implementation.t;
  configuration_digest : Digest.Comparison_policy.t;
  threshold_direction : Capability.threshold_direction;
  max_group_elements : int;
  compute : float array -> float array -> Metric_outcome.t;
}

type t

val empty : t

val register :
  t -> impl -> (t, [> `Duplicate_comparator of Id.Comparator.t ]) Err.t

val find :
  t ->
  comparator:Id.Comparator.t ->
  implementation_digest:Digest.Implementation.t ->
  configuration_digest:Digest.Comparison_policy.t ->
  impl option

val known : t -> Id.Comparator.t -> bool
val capabilities : t -> Capability.Metric.t list
