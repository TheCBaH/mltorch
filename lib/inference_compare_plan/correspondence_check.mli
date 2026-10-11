(** Correspondence, common-frontier recipe and coverage validation.

    Given a complete partition, this checks that: every unit input has one
    lossless binding from the common frontier and every canonical read has an
    exact reconstruction check on both sides; every canonical value has exactly
    one initial binding or one writing unit; both sides' candidate recipes for a
    written value exist and read only that unit's outputs (or its declared
    passthroughs); member pairs resolve and convert both sides to their declared
    comparison spec; every required outgoing value is carried by a candidate
    recipe; and the ordered final outputs correspond position by position.
    Recipes are resolved through the codec layer's checker, so a provider's
    [lossless] flag is never taken on trust. *)

open Inference_compare_schema
module Codecs = Inference_compare_codecs

type options = {
  codecs : Codecs.Element_codec.Registry.t;
  operations : Codecs.Conversion_registry.t;
}

module Unit_plan : sig
  type t = {
    unit : Id.Unit.t;
    canonical_reads : Id.Canonical_value.t list;  (** Sorted. *)
    canonical_writes : Id.Canonical_value.t list;  (** Sorted. *)
    input_bindings : Input_binding.t list;
        (** Left inputs in cluster order, then right. *)
    input_checks : Input_check.t list;  (** By side, then canonical value. *)
    output_bindings : Output_binding.t list;  (** Sorted by canonical value. *)
    output_pairs : Member_pair.t list;  (** As the unit declares them. *)
  }
end

type t

val check :
  options:options ->
  left:Graph_index.t ->
  right:Graph_index.t ->
  partition:Partition_check.t ->
  Mapping_artifact.t ->
  (t, Plan_error.t) Err.t

val unit_plans : t -> Unit_plan.t list
(** Sorted by unit id. *)

val unit_plan : t -> Id.Unit.t -> Unit_plan.t option

val writer : t -> Id.Canonical_value.t -> Id.Unit.t option
(** [None] for an initial value or an unknown key. *)

val initial : t -> Id.Canonical_value.t -> bool
val canonical_spec : t -> Id.Canonical_value.t -> Tensor_spec.t option

val final_values : t -> Id.Canonical_value.t list
(** One per final output occurrence, in occurrence order. *)

val coverage : t -> Coverage.t
