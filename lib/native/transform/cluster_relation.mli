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
   changed value always means a new id), see .ai/native_transform_design.md §3-4.

   Ids are VERSION-INDEXED: a ['v id] names an edge of graph version ['v], so
   looking a source id up on the destination side does not compile. See
   .ai/native_transform_versioning.md. *)

(* Endpoint problems found on construction, parametrised over the id type so both
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
  (* An id of graph version ['v]. Erased at runtime — inside this functor ['v id]
     IS [Id.t] — so the indexing costs nothing and the raw id stays the
     serialization and execution identity. *)
  type 'v id
  type 'v set

  (* One-way. Erasing a tag discards evidence and is always sound; nothing here
     lets a raw id acquire one. Consumers that must reach raw ids — printers, the
     PT2 lens, [Ground_eval] — go through these. *)
  val raw : 'v id -> Id.t
  val raws : 'v set -> Id.Set.t

  (* The id universe of one graph version: which ids exist, at which tag. The
     only way to obtain a ['v id], and so the reason a tagged id cannot be
     conjured from a raw one. *)
  module Universe : sig
    type 'v t

    val create : 'v Brand.t -> Id.Set.t -> 'v t

    (* [find] is the lift: a raw id enters the typed world only by being found
       in a version that has it. *)
    val find : 'v t -> Id.t -> 'v id option
    val elements : 'v t -> 'v id list
    val ids : 'v t -> Id.Set.t
  end

  (* Keyed by an id of one version, holding anything — [Provenance] is a map from
     destination edge to the source edges that fed it, so the key and the payload
     sit at different versions and neither may be confused for the other. *)
  module Map : sig
    type ('v, 'a) t

    val bindings : ('v, 'a) t -> ('v id * 'a) list
    val empty : ('v, 'a) t
    val find_opt : 'v id -> ('v, 'a) t -> 'a option
    val fold : ('v id -> 'a -> 'b -> 'b) -> ('v, 'a) t -> 'b -> 'b
    val is_empty : ('v, 'a) t -> bool
    val update : 'v id -> 'a -> ('v, 'a) t -> ('v, 'a) t
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

    (* The same cluster with its version indices dropped, for a DIAGNOSTIC that
       cannot carry them: [Map_verify.Report] escapes into [Pass.outcome] and the
       interpreter's result record, and parameterising that hierarchy would push
       ['src] and ['dst] through both for no gain — a report is read, not
       computed with. Checking stays typed; what comes out is erased. *)
    module Erased : sig
      type t = { src : Id.Set.t; dst : Id.Set.t; label : Label.t }

      val pp : Format.formatter -> t -> unit
    end

    val erase : ('src, 'dst) t -> Erased.t
  end

  type ('src, 'dst) t

  (* Everything corresponds to itself: no clusters at all. *)
  val identity : ('v, 'v) t

  (* The only constructor, and the only place a relation acquires its two
     versions. Taking both universes is what pins ['src] and ['dst] to particular
     graphs: a tag-polymorphic constructor leaves them free, and free tags unify
     with whatever they meet first, so [forward] on a destination id would still
     compile — the hole a separate [validate] left open, closed here by
     construction rather than by asking callers to remember.

     Also normalises: clusters sharing an id on either side are merged (their
     labels joined), [{x} <-> {x}] at the identity label is dropped as implicit,
     and the result is ordered deterministically.

     Validation is the same check that pinning cannot make: every endpoint
     resolves in the universe passed here (a ['v id] proves membership of SOME
     universe at ['v], not of this one), and implicit identity is covered — an
     unmentioned id must exist in both, and an id mentioned on one side while
     present in both graphs must be mentioned on the other too, which is what
     rejects a "creation" of an id the source already has. *)
  val of_clusters :
    src:'src Universe.t ->
    dst:'dst Universe.t ->
    ('src, 'dst) Cluster.t list ->
    (('src, 'dst) t, Id.t issue) Stdlib.result

  (* The merge step of [of_clusters] on its own, for a caller assembling a
     relation in pieces and needing "which cluster is this id in" before the
     whole is known — [Rewrite.apply] decides whether a definition really changed
     by asking exactly that, before it knows the created and deleted sets.

     Tag-polymorphic, and safe to be: it yields clusters, not a relation, so
     nothing reachable from here answers a question about an id's version. *)
  val normalise : ('src, 'dst) Cluster.t list -> ('src, 'dst) Cluster.t list
  val clusters : ('src, 'dst) t -> ('src, 'dst) Cluster.t list
  val is_empty : ('src, 'dst) t -> bool

  (* Where an id went / came from. An unmentioned id maps to itself, which is a
     retag across versions — the reason this layer lives inside the functor that
     owns the erasure, rather than beside [Snapshot] where it would need a
     [retag : 'a id -> 'b id] as forgeable as the raw ids it replaces. *)
  val forward : ('src, 'dst) t -> 'src id -> 'dst set
  val backward : ('src, 'dst) t -> 'dst id -> 'src set

  (* Ids with no counterpart: [created] have an empty src side, [deleted] an
     empty dst side. *)
  val created : ('src, 'dst) t -> 'dst set
  val deleted : ('src, 'dst) t -> 'src set

  (* Joins over the middle version, labelling each resulting cluster with the
     [Label.join] of everything that contributed. Identity-extension of a middle
     id is skipped when the partner declares that id created or deleted, so a
     dead id can never be resurrected and fused with a later cluster that reuses
     its numeric value (see .ai/native_transform_design.md §3, §9). *)
  val compose : ('a, 'b) t -> ('b, 'c) t -> ('a, 'c) t

  (* Sound because every label is a symmetric claim about the two sides. *)
  val invert : ('a, 'b) t -> ('b, 'a) t
  val pp : Format.formatter -> ('a, 'b) t -> unit
end
