(* Which correspondence cluster each of two graphs' edges belongs to: one
   [Cluster_var.t] per non-vacuous cluster, looked up per side by raw id.

   Membership, and nothing else. No graph definitions, no operator categories,
   no label, no raw-id equality — which is what makes it total, cheap, and
   impossible to talk into a false proof on its own. Deciding what a variable
   ENTITLES a comparison to assume is the driver's job; this only answers "same
   cluster?".

   Pairwise by construction, which is the property the crossed-inputs case
   turns on: {src t0 ↔ dst t1} and {src t1 ↔ dst t0} are two clusters, so they
   get two variables, and a comparison reading [v0; v1] against [v1; v0] does
   not match. A rule keyed on anything coarser — "both operands are some
   input" — proves sub(a,b) identical to sub(b,a).

   VACUOUS clusters have none. A creation or a deletion relates one side to
   nothing, so there is no shared value to name; those edges stay
   side-qualified and are expanded through instead.

   Well-defined because [Cluster_relation.normalise] merges overlapping
   clusters and [Graph_map.clusters_over] adds the untouched identities as
   disjoint singletons, so each side's ids partition across clusters and no id
   can land in two. *)

type t

(* [clusters] is what [Graph_map.clusters_over] yields — explicit clusters plus
   the implicit identities — since an edge in neither is not related at all. *)
val create : ('src, 'dst) Correspondence.Cluster.t list -> t
val dst : t -> Graph_ir.Tensor_id.t -> Cluster_var.t option
val src : t -> Graph_ir.Tensor_id.t -> Cluster_var.t option
