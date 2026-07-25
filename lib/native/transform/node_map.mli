(* Node clusters: which node in the source graph is which node in the
   destination. The same machinery as [Correspondence] with no label, because
   nodes have no values to claim anything about — an N:1 fusion is
   [{n1,n2,n3} <-> {n4}], a deletion [{n} <-> {}], a creation [{} <-> {n}].

   That empty sides are meaningful is the point: node creation, deletion,
   composition and reverse lookup all fall out of one relation, and it is
   symmetric, so [invert] is sound (unlike [Provenance]). See
   .ai/native_transform_design.md §3. *)

module Id :
  Cluster_relation.ID
    with type t = Graph_ir.Node_id.t
     and module Set = Graph_ir.Node_id.Set

module Label : Cluster_relation.LABEL with type t = unit
include module type of Cluster_relation.Make (Id) (Label)

val pair : Graph_ir.Node_id.t -> Graph_ir.Node_id.t -> Cluster.t

(* [from] may be empty, which records the node as created. *)
val fused : from:Graph_ir.Node_id.t list -> Graph_ir.Node_id.t -> Cluster.t
val delete : Graph_ir.Node_id.t -> Cluster.t
