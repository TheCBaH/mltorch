(** The structural plan for extracting one side of a matched unit as an ordinary
    graph.

    External operands become graph inputs (every occurrence of a repeated
    operand keeps its position), the cluster's nodes keep their operators,
    result order and in-cluster control edges, and the declared boundary results
    become the graph outputs in the supplied order -- declared passthroughs
    simply point at an input. No arithmetic, identity cast or guessed relayout
    is added. Identifiers are deterministic: the extracted graph lives under a
    version naming the unit and side, each tensor and node keeps its local id
    under a scope component naming the unit, and the maps back to the originals
    are part of the result and of its digest. A cluster need not be contiguous
    in the original node list. *)

open Inference_compare_schema

type t = {
  unit : Id.Unit.t;
  side : Side.t;
  body : Graph_desc.t;  (** The extracted description, final key included. *)
  descriptor : Extracted_graph.t;
  inputs : Tensor_ref.t list;  (** Extracted ids, in boundary input order. *)
  outputs : Tensor_ref.t list;  (** Extracted ids, in boundary output order. *)
  attributes : (Node_ref.t * string) list;
      (** Per extracted node, the text of everything the dataflow description
          does not carry (operation parameters, constants); part of the payload
          digest. *)
}

val plan :
  ?node_attributes:(Node_ref.t -> string) ->
  hashers:Inference_compare_codecs.Payload_hasher.Registry.t ->
  algorithm:Id.Digest_algorithm.t ->
  Inference_compare_plan.Graph_index.t ->
  Inference_compare_plan.Partition_check.Boundary.t ->
  (t, Extract_error.t) Err.t
(** [node_attributes] maps an {e original} node to its attribute text. The body
    is validated by the planner's own graph validator before it is returned. *)

val extract_all :
  ?node_attributes:(Side.t -> Node_ref.t -> string) ->
  hashers:Inference_compare_codecs.Payload_hasher.Registry.t ->
  algorithm:Id.Digest_algorithm.t ->
  Inference_compare_plan.Structural.t ->
  (t list, Extract_error.t) Err.t
(** Every unit and side of a structural plan, in execution order, left then
    right. *)

val body_preimage : t -> string
(** The text the payload digest is computed over: the canonical body plus both
    id maps. *)
