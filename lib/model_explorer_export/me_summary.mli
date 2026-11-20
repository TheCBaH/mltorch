(* One metric's value for every node of the canonical graph: the data behind a
   whole-graph colouring, in the shape Model Explorer's node data providers
   take (node id string to optional number).

   A PROTOCOL-LEVEL record like {!Me_artifact}: it names no backend type, so the
   worker computes the numbers and the page reads them without the lowering
   libraries. The node ids are the canonical view's own ({!Me_ids.op_node}).

   The count is bounded by [max_node_data_results_per_graph], the ceiling the
   session applies to the same overlay installed at export time, so no new
   ceiling exists; the encoded document is bounded by the caller as an
   artifact's is.

   See the `.ai/` directory for the lowering inspector design. *)

module String_map : Map.S with type key = string

type t = { metric : Lowering_key.Metric.t; values : float option String_map.t }
type error = Me_limits.over_limit_error

val pp_error : Format.formatter -> [< error ] -> unit

val check_count : limits:Me_limits.Limits.t -> int -> (unit, [> error ]) Err.t
(** The count against [max_node_data_results_per_graph], so a caller can refuse
    before a walk that is linear in it. *)

val create :
  limits:Me_limits.Limits.t ->
  metric:Lowering_key.Metric.t ->
  (Graph_ir.Node_id.t * float option) list ->
  (t, [> error ]) Err.t
(** A non-finite value is [None]: JSON has no spelling for it, and "no number"
    is what the page would show. Fails when there are more nodes than
    [max_node_data_results_per_graph]. *)

val jsont : t Jsont.t
