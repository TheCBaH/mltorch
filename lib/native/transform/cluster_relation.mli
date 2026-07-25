(* A bipartite equivalence between the ids of two graph versions: a set of
   CLUSTERS, each grouping source ids with the destination ids they correspond
   to, under a label. Shared by [Correspondence] (tensors, labelled with a value
   claim) and [Node_map] (nodes, unlabelled).

   Clusters, not a partial matching: many-to-one is the normal case. Trimming an
   identity permute [t0 -permute-> t1] yields the single cluster
   [{t0,t1} <-> {t0}], since both the untouched input and the trimmed output
   correspond to the surviving edge; trimming a chain widens the same cluster.

   An empty side carries meaning — [{a} <-> {}] is a deletion, [{} <-> {b}] a
   creation — so no separate created/deleted sets are needed. Ids in no cluster
   are implicitly identity-related, which keeps a relation proportional to what
   actually changed; that is sound only because of the id-identity rule (a
   changed value always means a new id), see .ai/native_transform_design.md §3-4. *)

(* Endpoint problems found by [validate], parametrised over the id type so both
   instantiations share one variant. *)
type 'id issue =
  | Dangling_dst of 'id (* a cluster names a dst id the destination lacks *)
  | Dangling_src of 'id
  | Uncovered_dst of 'id
    (* unmentioned, so implicitly identity, but absent from the other side *)
  | Uncovered_src of 'id
  | Unpaired_dst of 'id
    (* mentioned on one side only, while present in both graphs *)
  | Unpaired_src of 'id

val pp_issue : 'id Fmt.t -> Format.formatter -> 'id issue -> unit

module type ID = sig
  type t

  val compare : t -> t -> int
  val pp : Format.formatter -> t -> unit

  module Set : Set.S with type elt = t
end

(* The label a cluster carries. [identity] is what an unmentioned id implicitly
   claims, and is therefore also what normalisation may drop. *)
module type LABEL = sig
  type t

  val identity : t
  val join : t -> t -> t
  val equal : t -> t -> bool
  val pp : Format.formatter -> t -> unit
end

module Make (Id : ID) (Label : LABEL) : sig
  module Cluster : sig
    type t = { src : Id.Set.t; dst : Id.Set.t; label : Label.t }

    val pp : Format.formatter -> t -> unit
  end

  (* ['src] and ['dst] are phantom graph-version tags. They stop a map being
     composed in the wrong order or against the wrong middle, but they cannot tie
     a map to two PARTICULAR graphs — [of_clusters] is polymorphic in them — which
     is why [validate] exists and why consumers taking a relation from outside the
     rewrite path must call it. *)
  type ('src, 'dst) t

  (* Everything corresponds to itself: no clusters at all. *)
  val identity : ('v, 'v) t

  (* Normalises: clusters sharing an id on either side are merged (their labels
     joined), [{x} <-> {x}] at the identity label is dropped as implicit, and the
     result is ordered deterministically. *)
  val of_clusters : Cluster.t list -> ('a, 'b) t
  val clusters : ('a, 'b) t -> Cluster.t list
  val is_empty : ('a, 'b) t -> bool

  (* Where an id went / came from. An unmentioned id maps to itself. *)
  val forward : ('a, 'b) t -> Id.t -> Id.Set.t
  val backward : ('a, 'b) t -> Id.t -> Id.Set.t

  (* Ids with no counterpart: [created] have an empty src side, [deleted] an
     empty dst side. *)
  val created : ('a, 'b) t -> Id.Set.t
  val deleted : ('a, 'b) t -> Id.Set.t

  (* Joins over the middle version, labelling each resulting cluster with the
     [Label.join] of everything that contributed. Identity-extension of a middle
     id is skipped when the partner declares that id created or deleted, so a
     dead id can never be resurrected and fused with a later cluster that reuses
     its numeric value (see .ai/native_transform_design.md §3, §9). *)
  val compose : ('a, 'b) t -> ('b, 'c) t -> ('a, 'c) t

  (* Sound because every label is a symmetric claim about the two sides. *)
  val invert : ('a, 'b) t -> ('b, 'a) t

  (* Checks that this relation actually describes the two given id universes:
     every endpoint resolves on its own side, and implicit identity is covered —
     an unmentioned id must exist in both, and an id mentioned on one side while
     present in both graphs must be mentioned on the other too (which is what
     rejects a "creation" of an id the source already has). *)
  val validate :
    ('a, 'b) t ->
    src:Id.Set.t ->
    dst:Id.Set.t ->
    (unit, Id.t issue) Stdlib.result

  val pp : Format.formatter -> ('a, 'b) t -> unit
end
