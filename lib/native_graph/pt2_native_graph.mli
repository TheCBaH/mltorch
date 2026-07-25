(* PT2 provenance attached to a native graph. This stays outside [lib/native]:
   native execution uses only deterministic ids, while this wrapper retains the
   exporter-facing names, paths, metadata, and payload targets needed for PT2
   loading and diagnostics. *)

open Graph_ir

module Graph_path : sig
  type t = int list

  val root : t
  val child : t -> int -> t
  val pp : Format.formatter -> t -> unit
end

module Tensor_origin : sig
  type t = {
    graph_path : Graph_path.t;
    ssa_name : string;
    meta : Pytorch_types.TensorMeta.t option;
  }
end

module Node_origin : sig
  type t = {
    graph_path : Graph_path.t;
    index : int;
    target : string;
    name : string option;
    metadata : string Schema_runtime.String_map.t;
  }
end

type tensor_origin = Source of Tensor_origin.t | Derived

type t = {
  graph : graph;
  tensor_origins : tensor_origin Tensor_id.Map.t;
  node_origins : Node_origin.t list Node_id.Map.t;
  captured_targets : string Tensor_id.Map.t;
}

type error =
  [ `Unknown_tensor_id of Tensor_id.t
  | `Unknown_node_id of Node_id.t
  | `Captured_target_for_non_constant of Tensor_id.t ]

val pp_error : Format.formatter -> [< error ] -> unit

(* Validate and construct a PT2 provenance wrapper. [Derived] tensors and a
   one-to-many [node_origins] mapping represent importer-introduced relayout,
   decomposition, or future fusion nodes without inventing PT2 identities. *)
val make :
  graph:graph ->
  tensor_origins:tensor_origin Tensor_id.Map.t ->
  node_origins:Node_origin.t list Node_id.Map.t ->
  captured_targets:string Tensor_id.Map.t ->
  (t, error) Core.result
