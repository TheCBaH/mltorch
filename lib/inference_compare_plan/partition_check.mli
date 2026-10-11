(** Complete-partition and boundary validation.

    Every present node of both endpoints belongs to exactly one unit, helpers
    and assertions included; each unit's supplied inputs equal the values its
    nodes read from outside it, and its supplied outputs include every value
    consumed outside it or returned as a graph output. Extra outputs are allowed
    only as diagnostics produced inside the cluster or as declared passthroughs
    of its inputs. *)

open Inference_compare_schema

module Boundary : sig
  type t = {
    unit : Id.Unit.t;
    side : Side.t;
    nodes : Node_ref.t list;  (** In the endpoint's topological order. *)
    inputs : Tensor_ref.t list;  (** As supplied. *)
    outputs : Tensor_ref.t list;  (** As supplied. *)
    required_outgoing : Tensor_ref.t list;
        (** Consumed outside the cluster or returned as a final output. *)
    diagnostic_outputs : Tensor_ref.t list;
        (** Supplied outputs that nothing requires. *)
  }
end

type t

val check :
  left:Graph_index.t ->
  right:Graph_index.t ->
  Mapping_artifact.t ->
  (t, Plan_error.t) Err.t

val boundaries : t -> Boundary.t list
(** Sorted by unit id, left then right. *)

val boundary : t -> Id.Unit.t -> Side.t -> Boundary.t option
val owner : t -> Side.t -> Node_ref.t -> Id.Unit.t option
val units : t -> Id.Unit.t list
val matched_unit : t -> Id.Unit.t -> Matched_unit.t option
