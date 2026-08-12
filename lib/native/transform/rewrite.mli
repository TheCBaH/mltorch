(* The transform state and the single rewriter. Every version-bound abstraction
   lives here — allocator, recipe, and the versioned state itself — so no other
   module needs a public constructor that could forge one. See
   .ai/native_transform_design.md §5 and §7. *)

open Graph_ir

module Make (S : Side.S) : sig
  type graph = S.op Graph_common.Graph.t
  type node = S.op Graph_common.Node.t

  type error =
    [ Graph_map.error
    | Graph_view.Make(S.Dialect).error
    | Recipe.Make(S).error
    | `Bad_constant_payload of Tensor_id.t
    | `Constant_payload_overwrite of Tensor_id.t
    | `Cycle of Node_id.t
    | `Discontiguous_allocation
    | `Id_reuse_with_changed_value of Tensor_id.t
    | `Not_a_constant of Tensor_id.t
    | `Overlapping_replacements of Node_id.t
    | `Stale_allocator
    | `Substitution_conflict of Tensor_id.t
    | `Substitution_cycle of Tensor_id.t
    | `Unclaimed_redefinition of Tensor_id.t
    | `Unclaimed_substitution of Tensor_id.t
    | `Unknown_node of Node_id.t ]

  val pp_error : Format.formatter -> [< error ] -> unit

  (* Graph + constant payloads + the monotone id supply. The state IS the
     versioned value, so payloads and the supply are cumulative by construction —
     a step returning only a delta made fixed-point constant folding impossible to
     write, since iteration two would have no payload for iteration one's output. *)
  type 'v t
  type origin = Origin : 'v t -> origin

  (* The only minter of a version. Validates the graph, and validates the
     payloads: no duplicate ids, every id an effective [Constant] input, and each
     payload's shape and format matching its signature. *)
  val origin :
    ?constants:(Tensor_id.t * Tensor.packed) list ->
    graph ->
    (origin, error) Err.t

  val graph : 'v t -> graph
  val constants : 'v t -> Tensor.packed Tensor_id.Map.t
  val view : 'v t -> Graph_view.Make(S.Dialect).t

  (* The version itself: the id universes a map into or out of this state is
     indexed by, which is what a consumer needs to look an id up at the right
     version. [graph] and [view] are projections of it. *)
  val snapshot : 'v t -> 'v S.Snapshot.t

  (* Fresh ids come only from here, and only as a version-bound, watermarked
     allocator: an unversioned supply would let a recipe allocate below the
     persistent watermark, where an id deleted from the current graph passes the
     "unused" check despite having denoted a different edge earlier. *)
  type 'v allocator
  type 'v recipe

  val allocator : 'v t -> 'v allocator

  val plan :
    'v t ->
    'v allocator ->
    ('v, unit) Recipe.Make(S).t ->
    ('v recipe * 'v allocator, error) Err.t

  (* Allocators are immutable and so copyable, which a watermark check alone does
     not catch: two recipes planned from the same allocator both start at its
     watermark and allocate the same ids. [merge] therefore requires CONTIGUOUS
     allocation intervals in argument order, which rejects a branched history
     outright; an allocation-free recipe has equal start and end and merges
     freely. Regions must also be disjoint. *)
  val merge : 'v recipe -> 'v recipe -> ('v recipe, error) Err.t
  val pp_recipe : Format.formatter -> 'v recipe -> unit

  type 'v step = Step : 'w t * ('v, 'w) Graph_map.t -> 'v step

  val apply : 'v t -> 'v recipe -> ('v step, error) Err.t

  (* Terminal compaction, run once when no further rewriting is planned. Monotone
     allocation makes ids creep, so the ids introduced *after* the origin are
     renumbered densely upward from the origin's per-space watermark, in canonical
     order: graph inputs in signature order then node outputs in topological order
     for tensors, [Graph.nodes] order for nodes, tree pre-order for groups.

     ORIGIN IDS ARE NEVER TOUCHED. Renumbering one would be exactly the reuse §4
     forbids, and it would turn every untouched tensor into an explicit rename,
     destroying the implicit-identity bulk that keeps a map proportional to what
     changed. The resulting map is all-[Identical] and mentions only the ids that
     moved, so it composes like any other step and the PT2 lens still resolves a
     packed id. Constant payloads are renumbered with their edges.

     Not one of the four transformation kinds: nothing in stages 1-9 needs or
     assumes it, and it is only well-defined once "which ids are worth compacting"
     has stopped changing. See .ai/native_transform_design.md §9. *)
  val pack : 'v t -> ('v step, error) Err.t
end

include module type of Make (Native_side)
