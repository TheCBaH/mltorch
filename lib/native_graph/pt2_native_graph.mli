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

type tensor_origin = Derived | Source of Tensor_origin.t

type t = {
  graph : graph;
  tensor_origins : tensor_origin Tensor_id.Map.t;
  node_origins : Node_origin.t list Node_id.Map.t;
  captured_targets : string Tensor_id.Map.t;
}

type error =
  [ `Captured_target_for_non_constant of Tensor_id.t
  | `Unknown_node_id of Node_id.t
  | `Unknown_tensor_id of Tensor_id.t ]

val pp_error : Format.formatter -> [< error ] -> unit

(* Validate and construct a PT2 provenance wrapper. [Derived] tensors and a
   one-to-many [node_origins] mapping represent importer-introduced relayout,
   decomposition, or future fusion nodes without inventing PT2 identities. *)
val make :
  graph:graph ->
  tensor_origins:tensor_origin Tensor_id.Map.t ->
  node_origins:Node_origin.t list Node_id.Map.t ->
  captured_targets:string Tensor_id.Map.t ->
  (t, error) Err.t

(* Resolving provenance for a TRANSFORMED graph.

   The sidecar is never transformed. It stays anchored to the graph the importer
   built, and a destination id's origin is *recovered* by walking a composed
   source-to-destination map backwards. So there is no rebuild path, no
   per-step revalidation, and no drift — there is only ever one sidecar value.
   See .ai/native_transform_design.md §10. *)

type lens_error =
  [ `Sidecar_graph_mismatch
  | `Unknown_destination_node of Node_id.t
  | `Unknown_destination_tensor of Tensor_id.t
  | error
  | Graph_map.error ]

val pp_lens_error : Format.formatter -> [< lens_error ] -> unit

(* Parameterised by the DESTINATION version only; the source is existential,
   having no use beyond construction. *)
type 'dst lens

(* Both endpoint states are required. Correspondence is sparse, so an id in no
   cluster resolves to itself — without [dst] a bogus destination id would
   silently "resolve" to the same-numbered source. The lens therefore keeps both
   SNAPSHOTS and resolves every id through them: a query id is looked up in the
   destination, an implicit identity in the source.

   Validated once, here: the sidecar's graph against [src]'s by canonical
   [graph_jsont] bytes, plus [Graph_map.check_claim_closure]. The byte comparison
   ties the sidecar to the source. Endpoints no longer need re-checking —
   [Graph_map.create] does that, and the ids in a map are version-indexed — but
   closure does, because the map reaching a lens is composed and
   [Graph_map.compose] takes no snapshots. An unclosed map is what would let
   [captured_target] hand back source bytes for an edge whose value differs. *)
val lens :
  t ->
  src:'a Rewrite.t ->
  ('a, 'b) Graph_map.t ->
  dst:'b Rewrite.t ->
  ('b lens, lens_error) Err.t

(* A LIST because a cluster is many-to-one and picking one origin would be
   arbitrary. Yields [Tensor_origin.t] rather than the sidecar's [tensor_origin]
   variant, so "derived" has exactly one representation — the empty list — and is
   never ambiguous between [[]] and [[Derived]]. Sorted by source id and
   deduplicated. *)
val tensor_origins :
  'b lens -> Tensor_id.t -> (Tensor_origin.t list, lens_error) Err.t

(* Combined through the node clusters, sorted by [(graph_path, index)] and
   deduplicated. *)
val node_origins :
  'b lens -> Node_id.t -> (Node_origin.t list, lens_error) Err.t

(* Follows ONLY an [Identical] correspondence. Provenance is never a fallback
   here and neither is a weaker claim: for [archive w --Permute--> wp] the
   provenance edge says [w -> wp], but w's archive bytes are in the *unpermuted*
   layout and are not a valid payload for wp — conflating the two is data
   corruption, not imprecision. Where a many-to-one cluster offers several
   captured sources they are all [Identical] and so interchangeable; the lowest
   id is taken for determinism. *)
val captured_target :
  'b lens -> Tensor_id.t -> (string option, lens_error) Err.t

(* The derivation, exposed separately from value correspondence because it
   answers a different question: where a folded constant came from, not which
   bytes it holds. Diagnostics only — never a payload source. *)
val provenance_sources : 'b lens -> Tensor_id.t -> Tensor_id.t list
