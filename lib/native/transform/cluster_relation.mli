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

  (* The same relation over VERSION-INDEXED ids: the raw API above erased to
     [Id.t], this one indexed by the graph version an id belongs to, so looking
     a source id up on the destination side stops compiling. Erased at runtime —
     ['v id] is [Id.t] and ['v set] is [Id.Set.t] inside this functor.

     Defined here rather than beside [Snapshot] because the implicit-identity
     rule needs to retag: [forward] answers an UNMENTIONED source id with itself
     on the destination side. Only the body that owns the erasure can do that,
     so keeping it inside [Make] is what lets the retag exist without a
     [retag : 'a id -> 'b id] appearing in any signature, where it would be as
     forgeable as the raw ids it replaces.

     Stage 1 of .ai/native_transform_versioning.md: additive, so the raw API
     above is still the one in use. *)
  module Tagged : sig
    type 'v id
    type 'v set

    (* One-way. Erasing a tag discards evidence and is always sound; nothing
       here lets a raw id acquire one. Consumers that must reach raw ids —
       printers, the PT2 lens, [Ground_eval] — go through these. *)
    val raw : 'v id -> Id.t
    val raws : 'v set -> Id.Set.t

    (* The id universe of one graph version: which ids exist, at which tag. The
       only way to obtain a ['v id], and the reason a tagged id cannot be
       conjured from a raw one. *)
    module Universe : sig
      type 'v t

      val create : 'v Brand.t -> Id.Set.t -> 'v t
      val ids : 'v t -> Id.Set.t
      val find : 'v t -> Id.t -> 'v id option
    end

    module Set : sig
      val add : 'v id -> 'v set -> 'v set
      val cardinal : 'v set -> int
      val disjoint : 'v set -> 'v set -> bool
      val elements : 'v set -> 'v id list
      val empty : 'v set
      val equal : 'v set -> 'v set -> bool
      val fold : ('v id -> 'a -> 'a) -> 'v set -> 'a -> 'a
      val is_empty : 'v set -> bool
      val mem : 'v id -> 'v set -> bool
      val min_elt_opt : 'v set -> 'v id option
      val of_list : 'v id list -> 'v set
      val singleton : 'v id -> 'v set
      val union : 'v set -> 'v set -> 'v set
    end

    module Cluster : sig
      type ('src, 'dst) t = { src : 'src set; dst : 'dst set; label : Label.t }

      val pp : Format.formatter -> ('src, 'dst) t -> unit
    end

    type ('src, 'dst) t

    val identity : ('v, 'v) t

    (* Validation and tagging fused, and the ONLY constructor. Taking both
       universes is what pins ['src] and ['dst] to particular graphs: a
       tag-polymorphic constructor leaves them free, and free tags unify with
       whatever they meet first, so [forward] on a destination id would still
       compile. That is the hole the raw [of_clusters] + [validate] pair leaves
       open, closed here by construction rather than by asking callers to
       remember. *)
    val of_clusters :
      src:'src Universe.t ->
      dst:'dst Universe.t ->
      ('src, 'dst) Cluster.t list ->
      (('src, 'dst) t, Id.t issue) Stdlib.result

    val clusters : ('src, 'dst) t -> ('src, 'dst) Cluster.t list
    val is_empty : ('src, 'dst) t -> bool
    val forward : ('src, 'dst) t -> 'src id -> 'dst set
    val backward : ('src, 'dst) t -> 'dst id -> 'src set
    val created : ('src, 'dst) t -> 'dst set
    val deleted : ('src, 'dst) t -> 'src set
    val compose : ('a, 'b) t -> ('b, 'c) t -> ('a, 'c) t
    val invert : ('a, 'b) t -> ('b, 'a) t
    val pp : Format.formatter -> ('a, 'b) t -> unit
  end
end
