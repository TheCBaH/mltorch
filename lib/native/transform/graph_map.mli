(* The mapping between two graph versions: what a pipeline composes and what the
   verifier and the PT2 lens consume. Three relations, because merging any two of
   them loses information (see .ai/native_transform_design.md §3):

   - [values]     tensor value clusters, labelled with the claim a checker may assert
   - [nodes]      node clusters, unlabelled
   - [provenance] directed "was computed from", no value claim at all

   There is deliberately no [invert]. Reversal is exposed per relation —
   [Correspondence.invert], [Node_map.invert], [Provenance.sources_of] — because
   inverting a "computed from" edge is not a meaningful claim. That is the honest
   reading of "the mapping should be reversible": the value and node
   correspondences reverse; the computation history is queried backwards.

   ABSTRACT, with [create] the only way to build one. The record used to be
   public and [validate] a separate call consumers had to remember; a map is the
   trust boundary for everything downstream, so the checks belong on the way in.
   See .ai/native_transform_versioning.md §2. *)

type ('src, 'dst) t

type error =
  [ `Cluster_format of Graph_ir.Tensor_id.t * Graph_ir.Tensor_id.t
  | `Cluster_shape of Graph_ir.Tensor_id.t * Graph_ir.Tensor_id.t
  | `Node_endpoint of Graph_ir.Node_id.t Cluster_relation.issue
  | `Provenance_endpoint of Graph_ir.Tensor_id.t Cluster_relation.issue
  | `Graph_output_arity of int * int
  | `Graph_output_mismatch of Graph_ir.Tensor_id.t * Graph_ir.Tensor_id.t
  | `Unclosed_claim of Graph_ir.Tensor_id.t
  | `Value_endpoint of Graph_ir.Tensor_id.t Cluster_relation.issue ]
(* [`Graph_output_*], not [`Output_arity]: [Graph_view.error] already owns that
   tag with a different payload, and [Rewrite.error] unions both rows, so
   reusing the name would give one tag two payloads and fail to typecheck. *)

val pp_error : Format.formatter -> [< error ] -> unit

(* Builds and checks in one step. Beyond the endpoint checks the relations
   perform, this establishes two map-level invariants no consumer can recover on
   its own:

   - CLUSTER METADATA. Corresponding shapes agree, and an [Identical] cluster's
     endpoints agree on format and quantization — an [Identical] claim across F32
     and BF16 is a contradiction, it is [Approximate]. This is step 9 of
     .ai/native_transform_design.md §7, which specified it for [Rewrite.apply]
     and was never implemented anywhere. It matters because the PT2 lens hands
     back captured SOURCE bytes for any edge whose claim is [Identical]: a
     cluster spanning two formats makes that a data-corruption path, not an
     imprecision.

   - CLAIM CLOSURE, see [check_claim_closure].

   Both are about the map as a data structure, so both belong here rather than in
   the verifier — [Pt2_native_graph] never goes near a symbolic expression. *)
(* The snapshot-consuming half, over a PAIR of dialects. [t] itself stays
   outside, so [compose] works ACROSS the boundary with no existential
   packaging; only these need to know which dialects they are between. *)
module Make_pair (Src : Side.S) (Dst : Side.S) : sig
  val create :
    src:'src Src.Snapshot.t ->
    dst:'dst Dst.Snapshot.t ->
    values:('src, 'dst) Correspondence.Cluster.t list ->
    nodes:('src, 'dst) Node_map.Cluster.t list ->
    provenance:('src, 'dst) Provenance.t ->
    (('src, 'dst) t, error) Err.t

  val clusters_over :
    ('src, 'dst) t ->
    src:'src Src.Snapshot.t ->
    dst:'dst Dst.Snapshot.t ->
    ('src, 'dst) Correspondence.Cluster.t list

  val check_claim_closure :
    ('src, 'dst) t ->
    src:'src Src.Snapshot.t ->
    dst:'dst Dst.Snapshot.t ->
    (unit, error) Err.t

  (* Positional and two-sided: equal output arity, and source output [i] sharing
     a cluster with destination output [i]. Coverage alone is too weak — it
     admits crossed outputs and an output paired with an internal tensor.

     Called by [create], and deliberately NOT by [Map_verify.run]: unlike claim
     closure this property is preserved by composition, and [identity] has it
     trivially, so a check there could never fire. *)
  val check_output_correspondence :
    ('src, 'dst) t ->
    src:'src Src.Snapshot.t ->
    dst:'dst Dst.Snapshot.t ->
    (unit, error) Err.t
end

include module type of Make_pair (Native_side) (Native_side)

val nodes : ('src, 'dst) t -> ('src, 'dst) Node_map.t
val provenance : ('src, 'dst) t -> ('src, 'dst) Provenance.t
val values : ('src, 'dst) t -> ('src, 'dst) Correspondence.t
val identity : ('v, 'v) t
val compose : ('a, 'b) t -> ('b, 'c) t -> ('a, 'c) t

(* Explicit clusters only. Ids in none of them are implicitly [Identical]; the
   map cannot enumerate them because it does not know either graph's id
   universe, which is what [clusters_over] is for. *)
val clusters : ('src, 'dst) t -> ('src, 'dst) Correspondence.Cluster.t list

(* Every value cluster including the untouched implicit identities, synthesised
   from the two graphs. This is the equivalence-cluster extraction a verifier
   walks. *)

(* Recomputes claim propagation from the map's explicit clusters over the
   destination graph, and rejects when an edge left implicitly [Identical] would
   have received something weaker. Consider source [t2 = add(a,b); t3 = relu(t2)]
   against destination [t2 = sub(a,b); t3 = relu(t2)], with [t2 <-> t2] claimed
   [Unverifiable] and t3 unmentioned: every consumer is then wrong about t3, and
   the lens will fetch source bytes for an edge whose value differs.

   Called by [create], so no map is born unclosed. Exported as well because
   [compose] takes no snapshots and so cannot re-check, and both consumers
   receive composed maps.

   Named CLOSURE, not lineage: this proves the labels are closed over the
   destination graph. It is not evidence of a rewrite history, and it says
   nothing about a map with no explicit claims at all — an empty map between
   structurally unrelated graphs passes. What guards the verifier there is a
   different mechanism entirely. *)

val pp : Format.formatter -> ('src, 'dst) t -> unit
