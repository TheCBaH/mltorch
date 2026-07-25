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
   .ai/native_transform_design.md §3 and §10. *)

open Graph_ir

type ('src, 'dst) t

val empty : ('a, 'b) t

(* Record that [dst] was computed from [sources]. Repeated edges into the same
   destination accumulate. *)
val add : sources:Tensor_id.t list -> Tensor_id.t -> ('a, 'b) t -> ('a, 'b) t
val of_list : (Tensor_id.t list * Tensor_id.t) list -> ('a, 'b) t
val is_empty : ('a, 'b) t -> bool

(* Substitutes middle ids through the two value correspondences and closes
   transitively, so provenance survives intervening renames. *)
val compose :
  ('a, 'b) t ->
  ('b, 'c) t ->
  values:('a, 'b) Correspondence.t * ('b, 'c) Correspondence.t ->
  ('a, 'c) t

(* Reverse lookup, not an inverse: "which source edges fed this one". *)
val sources_of : ('a, 'b) t -> Tensor_id.t -> Tensor_id.Set.t
val targets_of : ('a, 'b) t -> Tensor_id.t -> Tensor_id.Set.t
val edges : ('a, 'b) t -> (Tensor_id.Set.t * Tensor_id.t) list

(* Endpoints must resolve: sources in the source graph, targets in the
   destination. *)
val validate :
  ('a, 'b) t ->
  src:Tensor_id.Set.t ->
  dst:Tensor_id.Set.t ->
  (unit, Tensor_id.t Cluster_relation.issue) Stdlib.result

val pp : Format.formatter -> ('a, 'b) t -> unit
