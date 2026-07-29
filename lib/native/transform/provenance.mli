(* "Was computed from": directed hyperedges over tensors, recording that a
   destination edge was derived from a set of source edges. Constant folding is
   the motivating producer — folding [w -Permute-> wp] deletes w and records
   [[w] -> wp].

   Deliberately NOT part of the value lattice in [Correspondence]. Provenance
   asserts nothing about values: w's payload is in the unpermuted layout and is
   not a valid payload for wp. Putting a [Derived] label in that lattice would
   make [join Identical Derived = Derived] and destroy an identity claim on
   composition, and following provenance to fetch bytes would be data corruption
   rather than imprecision. It is therefore directional, has reverse LOOKUP but
   no [invert], and is skipped by any consumer asking about values. See
   .ai/native_transform_design.md §3 and §10.

   Because it is directional, its two sides were the easiest to confuse: the old
   [sources_of : ('a, 'b) t -> Tensor_id.t -> Tensor_id.Set.t] named neither.
   Both are version-indexed now. *)

type ('src, 'dst) t

val empty : ('src, 'dst) t

(* Record that [dst] was computed from [sources]. Repeated edges into the same
   destination accumulate. *)
val add :
  sources:'src Correspondence.set ->
  'dst Correspondence.id ->
  ('src, 'dst) t ->
  ('src, 'dst) t

val of_list :
  ('src Correspondence.set * 'dst Correspondence.id) list -> ('src, 'dst) t

val is_empty : ('src, 'dst) t -> bool

(* Substitutes middle ids through the two value correspondences and closes
   transitively, so provenance survives intervening renames. *)
val compose :
  ('a, 'b) t ->
  ('b, 'c) t ->
  values:('a, 'b) Correspondence.t * ('b, 'c) Correspondence.t ->
  ('a, 'c) t

(* Reverse lookup, not an inverse: "which source edges fed this one". *)
val sources_of :
  ('src, 'dst) t -> 'dst Correspondence.id -> 'src Correspondence.set

val targets_of :
  ('src, 'dst) t -> 'src Correspondence.id -> 'dst Correspondence.set

val edges :
  ('src, 'dst) t -> ('src Correspondence.set * 'dst Correspondence.id) list

(* Endpoints must resolve: sources in the source graph, targets in the
   destination. A tagged id witnesses membership of SOME universe at its version,
   not of the one given here, so this still has work to do — see
   [Cluster_relation.of_clusters]. *)
val validate :
  ('src, 'dst) t ->
  src:'src Correspondence.Universe.t ->
  dst:'dst Correspondence.Universe.t ->
  (unit, Graph_ir.Tensor_id.t Cluster_relation.issue) Stdlib.result

val pp : Format.formatter -> ('src, 'dst) t -> unit
