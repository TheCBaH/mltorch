(** Composition of mapping artifacts. [compose first second] relates the left
    graph of [first] to the right graph of [second] through their shared
    intermediate graph.

    Nodes are related by connected components over the three graphs: units of
    both maps that touch the same intermediate node merge into one composed
    unit, a component with nodes on only one end becomes a created or eliminated
    relation, and an intermediate-only component stays in the derivation
    digests. Tensor relations follow the same components; a route in one
    direction exists only where both maps supply one, and two different step
    lists between the same tensors are [Ambiguous_route]. The common frontier
    keeps [first]'s canonical values: the right side of every composed unit is
    bound by threading [first]'s left-side recipes through the composed routes,
    and a terminal constant that only [second] pins is carried over with its
    payload. Casts and rounding steps stay in order; nothing is cancelled.

    The result is a candidate. It is not checked against the endpoint graphs:
    partition legality, dependency order and recipe types are the planner's
    ([Inference_compare_plan.Structural.check]) to decide, and a composition
    that does not pass it is a mapping error, not something to repair here.
    [derivation_digest] records both inputs for audit and is excluded from
    meaning comparisons. *)

open Inference_compare_schema

val compose :
  hashers:Inference_compare_codecs.Payload_hasher.Registry.t ->
  algorithm:Id.Digest_algorithm.t ->
  Mapping_artifact.t ->
  Mapping_artifact.t ->
  (Mapping_artifact.t, Compose_error.t) Err.t
