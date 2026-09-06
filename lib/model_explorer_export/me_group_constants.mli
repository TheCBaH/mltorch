(* A presentation-only transform on an already-built graph: where a constant
   boundary node sits in the namespace tree. It changes no node id, no edge,
   and no attribute — only [GraphNode.namespace] on nodes classified as
   constants — so it is safe to apply after [Me_source]/[Me_build] have run,
   with no re-validation of the graph it produces.

   Both projectors emit a constant boundary at namespace [""], which places
   every constant on the graph's root rank. For a graph with hundreds of
   independent parameters, a layered layout gives that rank the whole canvas
   and the actual compute path becomes a thin, unreadable line. [Grouped]
   moves each constant into the namespace its consumers already share, so
   Model Explorer's own collapsed-group layout absorbs it into the module
   that uses it. *)

type mode =
  | Explicit  (** identity: every constant stays at namespace [""] *)
  | Grouped
      (** each constant moves to the longest common namespace of its consumers
      *)

val is_constant_id : string -> bool
(** Both exporters spell a constant boundary id [const:...]
    ([Me_ids.pt2_boundary], [Me_ids.boundary]). Classifying by that prefix, in
    this one function, is what the module comment above promises: nothing else
    in this file, or in a caller, tests node labels or metadata to decide what a
    constant is. *)

val apply : mode -> Model_explorer.Graph.t -> Model_explorer.Graph.t
(** [apply Explicit] returns its argument unchanged (by value; not guaranteed to
    be physically the same graph). [apply Grouped g] reassigns the namespace of
    every node [is_constant_id] accepts:

    - one distinct consumer namespace: the constant takes it directly;
    - more than one distinct consumer namespace: the constant takes their
      longest common ['/']-separated prefix, or the dedicated root namespace
      ["parameters"] when that prefix is empty — a real disagreement about
      ownership, not the trivial case of a single root-level consumer;
    - no consumer at all: also ["parameters"]. An unconsumed constant (a
      declared buffer no traced op reads, e.g. a batch-norm
      [num_batches_tracked] that eval-mode folding never touches) has no owner
      to inherit from, and MobileNet alone carries 52 of them: left at namespace
      [""] as scattered singletons, they dominate the root rank as badly as the
      ungrouped baseline and defeat the whole transform, which a render against
      the real pinned element is what caught. *)
