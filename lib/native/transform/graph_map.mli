(* The mapping between two graph versions: what a pipeline composes and what a
   future numerical/symbolic verifier consumes. Three relations, because merging
   any two of them loses information (see .ai/native_transform_design.md §3):

   - [values]     tensor value clusters, labelled with the claim a checker may assert
   - [nodes]      node clusters, unlabelled
   - [provenance] directed "was computed from", no value claim at all

   There is deliberately no [invert]. Reversal is exposed per relation —
   [Correspondence.invert], [Node_map.invert], [Provenance.sources_of] — because
   inverting a "computed from" edge is not a meaningful claim. That is the honest
   reading of "the mapping should be reversible": the value and node
   correspondences reverse; the computation history is queried backwards. *)

open Graph_ir

type ('src, 'dst) t = {
  values : ('src, 'dst) Correspondence.t;
  nodes : ('src, 'dst) Node_map.t;
  provenance : ('src, 'dst) Provenance.t;
}

type error =
  [ `Node_endpoint of Node_id.t Cluster_relation.issue
  | `Provenance_endpoint of Tensor_id.t Cluster_relation.issue
  | `Value_endpoint of Tensor_id.t Cluster_relation.issue ]

val pp_error : Format.formatter -> [< error ] -> unit
val identity : ('v, 'v) t
val compose : ('a, 'b) t -> ('b, 'c) t -> ('a, 'c) t

(* Explicit clusters only. Ids in none of them are implicitly [Identical]; the
   map cannot enumerate them because it does not know either graph's id
   universe, which is what [clusters_over] is for. *)
val clusters : ('a, 'b) t -> Correspondence.Cluster.t list

(* Every value cluster including the untouched implicit identities, synthesised
   from the two graphs. This is the equivalence-cluster extraction a verifier
   walks. *)
val clusters_over :
  ('a, 'b) t -> src:graph -> dst:graph -> Correspondence.Cluster.t list

(* Phantoms tag a map but cannot tie it to two PARTICULAR graphs:
   [Correspondence.of_clusters] is polymorphic in them, so a caller can hand out
   a well-typed map full of ids that exist in neither endpoint, and [identity]
   type-checks between unrelated graphs. Maps built by [Rewrite] are validated on
   construction; any consumer taking a map from elsewhere must call this. *)
val validate : ('a, 'b) t -> src:graph -> dst:graph -> (unit, error) Core.result
val pp : Format.formatter -> ('a, 'b) t -> unit
