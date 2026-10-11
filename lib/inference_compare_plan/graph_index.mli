(** The producer, consumer, ownership and ordered-output indexes of one endpoint
    graph, built and validated before anything refers to the graph.

    Building rejects, with a structured error and before any runner exists:
    references into another graph version, duplicate node or result ids, an
    operand with no producer, a result or source without a spec, a final output
    list that disagrees with its output tree, a control predecessor that is not
    a node, an unsupported effect, and a cycle through data or control edges. *)

open Inference_compare_schema

module Producer : sig
  type t =
    | Node of { node : Node_ref.t; port : Output_port.t }
    | Source of Graph_source.t
end

module Use : sig
  type t = { node : Node_ref.t; port : Input_port.t }
  (** One operand occurrence: a tensor read twice by a node has two uses. *)
end

type t

val build :
  limits:Resource_limits.t ->
  side:Side.t ->
  Graph_desc.t ->
  (t, Plan_error.t) Err.t

val side : t -> Side.t
val graph : t -> Graph_desc.t
val key : t -> Graph_key.t

val nodes : t -> Node_desc.t list
(** In the graph's declared order. *)

val node : t -> Node_ref.t -> Node_desc.t option
val producer : t -> Tensor_ref.t -> Producer.t option

val consumers : t -> Tensor_ref.t -> Use.t list
(** Operand occurrences in node order, then port order. *)

val spec : t -> Tensor_ref.t -> Tensor_spec.t option
val final_outputs : t -> Tensor_ref.t list
val is_final_output : t -> Tensor_ref.t -> bool
val is_source : t -> Tensor_ref.t -> bool

val topological : t -> Node_ref.t list
(** A dependency order over data and control edges. *)
