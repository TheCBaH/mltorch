(* A graph pinned to one version: the validated view, plus the id universes that
   make that version's ids typed. The public minter of a version tag, and the
   only way to turn a raw [Tensor_id.t] into a tagged one.

   [create] returns a PACKED snapshot, which is what keeps it honest.
   .ai/native_transform_design.md rejected a versioned [Graph_view] twice (§5,
   §11) because [of_graph : graph -> ('v t, error) result] lets the CALLER pick
   ['v], and so lets them present graph B under graph A's tag. Handing back an
   existential removes the choice: unpacking [Pack s] binds a rigid tag that
   unifies with nothing else, exactly as [Rewrite.origin] already does. That is
   the whole difference, and it is why this module can be public.

   One [Brand.t] covers both id kinds, so a snapshot's tensors and nodes share a
   version and a [Graph_map] can be indexed by a single tag rather than one per
   id space.

   Note what this does NOT establish: that two snapshots are a meaningful pair,
   or that a map between them is semantically valid. Snapshots make ids
   attributable; [Graph_map] is where a relation between two of them is
   checked. *)

open Graph_ir

type 'v t
type packed = Pack : 'v t -> packed
type 'v edge = 'v Correspondence.Tagged.id
type 'v node = 'v Node_map.Tagged.id

(* Validates the graph, so every accessor below is unambiguous for the same
   reason [Graph_view]'s are. *)
val create : graph -> (packed, Graph_view.error) Core.result

(* [None] when the id is not in this version, which is the check that stops a
   raw id from being tagged into a graph it does not belong to. *)
val edge : 'v t -> Tensor_id.t -> 'v edge option
val node : 'v t -> Node_id.t -> 'v node option

(* Needed by [Graph_map]: building a relation demands both universes, and they
   cannot be recovered from an abstract snapshot otherwise. *)
val edges : 'v t -> 'v Correspondence.Tagged.Universe.t
val nodes : 'v t -> 'v Node_map.Tagged.Universe.t
val graph : 'v t -> graph
val view : 'v t -> Graph_view.t
