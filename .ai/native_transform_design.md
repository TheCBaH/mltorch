# Native graph transformation — matching, rewriting, mapping

`native_graph_design.md` builds the graph IR and says its long-term purpose is
**transformation** (decompose, fuse, relayout); its §7 sketched an API and left it
unimplemented. This doc is that layer: `lib/native/transform/`, an **applicative**
matcher and rewriter where every transformation is a pure step producing a new
graph plus a mapping, and mappings compose across a whole pipeline.

The mapping is not a by-product. A future harness must be able to take
`(A, B, A→B)`, extract **equivalence clusters**, and check numerically or
symbolically that both sides compute the same values. That verifier is out of
scope here; everything below is shaped so it can exist.

Status: **complete** — all eleven stages have landed. The framework, the
transformations built on it (permute simplification, constant folding, batch-norm
folding), terminal id packing, and PT2 provenance resolved through the composed
map. `native_graph transform` runs the pipeline end to end on a real model.
Each section below carries its own status marker, flipped by the commit that
implements it; `## 12. Staging and the transformations` tracks the whole sequence.

## 1. What it is for

Three pieces of already-documented work are blocked on this layer:

- pruning the `Discard` sinks that `native_multi_output_design.md` introduces;
- cancelling the adjacent inverse `Permute`s that the relayout lowering in
  `native_aten_bridge_layout.md` emits at every op boundary;
- **folding a `Permute` of a constant weight** — today every OIHW conv weight is
  re-permuted on every single inference instead of once at load time. This is the
  motivating case and the one the constant-folding pass is built around.

## 2. What the IR gives us

The framework leans on four properties of `graph_ir.ml`, all current as of
`5ca988c`:

- **One flat SSA node list.** `Graph.nodes` is globally topo-ordered and there is
  no `Subgraph` op, so matching is transparent across the whole graph for free.
  `Graph.root : Group.t` is a *structural* hierarchy over node ids that "never
  introduces a call boundary"; the framework maintains it but never matches on it.
- **Constants are classified in the IR.** `Input.kind = Input | Constant` with
  `Graph.input_kinds` and `Graph_ir.input_kind`. The map is **sparse by design**:
  `dec_graph` fills it only from the optional `input_constants` JSON field, and
  `input_kind` defaults everything else to `Input`. Payloads live outside the IR
  (`Eval_direct.run ?constants`), so the IR stays data-free.
- **Stable ids and no names.** Identity is the id; human-facing annotation is
  `Graph_ir.Printer` and the `Pt2_native_graph` sidecar.
- **A deterministic canonical encoding.** `Graph_ir.graph_jsont` emits tensor sigs
  from key-ordered `Map.bindings`, derives `input_constants` from the *effective*
  `input_kind`, and writes the group tree with labels and item order — which makes
  it usable as a graph identity check (§10).

## 3. The mapping — three relations, not one

Status: **implemented** — `cluster_relation.ml` (the shared functor), `correspondence.ml`,
`node_map.ml`, `provenance.ml`, `graph_map.ml`; tests in `test/native/graph_map_test.ml`.

The single most consequential decision here. A mapping between two graph versions
carries three genuinely different relations, and merging any two of them loses
information:

| Relation | Shape | Symmetric? | Answers |
|---|---|---|---|
| `Correspondence` | tensor clusters, labelled | yes, `invert` | which edge here is which edge there, and what may be asserted about their values |
| `Node_map` | node clusters, unlabelled | yes, `invert` | which node here is which node there |
| `Provenance` | directed tensor hyperedges | no, reverse *lookup* only | what a value was computed from |

Both cluster relations are one functor, `Cluster_relation.Make (Id) (Label)`, over
an id space and a label with a `join` and an `identity`:

```ocaml
module Cluster : sig
  type t = { src : Id.Set.t; dst : Id.Set.t; label : Label.t }
end

type ('src, 'dst) t
val identity    : ('v, 'v) t
val of_clusters : Cluster.t list -> ('a, 'b) t   (* normalises *)
val compose     : ('a, 'b) t -> ('b, 'c) t -> ('a, 'c) t
val invert      : ('a, 'b) t -> ('b, 'a) t
val forward     : ('a, 'b) t -> Id.t -> Id.Set.t
val backward    : ('a, 'b) t -> Id.t -> Id.Set.t
val created     : ('a, 'b) t -> Id.Set.t         (* clusters with an empty src *)
val deleted     : ('a, 'b) t -> Id.Set.t
val validate    : ('a, 'b) t -> src:Id.Set.t -> dst:Id.Set.t ->
                  (unit, Id.t issue) Stdlib.result
```

`of_clusters` is what makes the representation canonical: clusters sharing an id
on either side are merged and their labels joined, `{x} ↔ {x}` at the identity
label is dropped as implicit, and the survivors are ordered by lowest src then
lowest dst. So a relation is always a list of pairwise-disjoint clusters, "which
cluster is id x in" is unambiguous, and two relations are equal iff their cluster
lists are.

`Correspondence` instantiates it at `Tensor_id` with the claim lattice;
`Node_map` at `Node_id` with `unit`, whose printer emits nothing (and `Cluster.pp`
omits an empty label rather than leaving a dangling separator).

```ocaml
module Precision : sig
  type t = { fmt : Payload.packed_fmt; quant : Quant.t option }
  module Set : Set.S with type elt = t
end

(* Identical ⊑ Equivalent ⊑ Approximate ⊑ Unverifiable. *)
type relation =
  | Approximate of Precision.Set.t  (* every lossy representation traversed *)
  | Equivalent                      (* equal in exact arithmetic; rounding may differ *)
  | Identical                       (* bit-for-bit *)
  | Unverifiable                    (* corresponds structurally; asserts nothing *)

val join : relation -> relation -> relation
```

`Precision` is compared by format *name* — `packed_fmt` is an existential over a
GADT, so it has no usable structural order — and by the quantization parameters,
which are first-order enough for `Stdlib.compare` to be exact. Two independent
passes through BF16 therefore collapse to one set entry.

**Clusters, not a matching.** Correspondence is a bipartite equivalence over
source ⊎ destination ids whose connected components are the clusters. Many-to-one
is the *normal* case: trimming an identity permute `t0 →permute→ t1` yields the
single cluster `A.{t0,t1} ↔ B.{t0}`, because both the untouched input and the
trimmed output correspond to the surviving edge. Trimming a chain widens the same
cluster. A strict partial matching cannot express this and was rejected for it.

**Empty sides carry meaning.** `{a} ↔ {}` is a deletion, `{} ↔ {b}` a creation, so
no separate created/deleted sets are needed.

**Implicit identity.** Ids in no cluster are implicitly `Identical`, and a
`{x} ↔ {x} Identical` cluster is normalised away. This keeps maps proportional to
what actually changed. It is safe only because of the id-identity rule (§4): a
changed value always means a new id, so a change can never hide in the implicit
bulk.

**`Approximate` carries a set, not an ordering.** F16 and BF16 are incomparable
(range versus mantissa) and quantization error depends on scale and saturation, so
a `coarser : t -> t -> t` would be a fiction. The label records *which* lossy
representations the value passed through; `join` is set union; choosing a
tolerance is the checker's policy. F32→BF16→F32 therefore composes to
`Approximate {bf16}` rather than to something reading as lossless.

**`compose`** identity-extends both relations over the ids they mention, takes
connected components of the join over middle ids, and labels each component with
the `join` of its contributing labels. Extension is what makes a rename survive a
map that does not mention it: without it, `{a} ↔ {b}` composed with the empty map
would read as "a was deleted" instead of "a is still b".

One side condition: **never identity-extend an id the partner map declares created
or deleted**. The motivating hazard is `{t11} ↔ {}` (deleted) composed with a later
`{t12} ↔ {t11}` (a packed rename, §9) — resurrecting the dead `t11` would fuse the
two clusters and claim the dead edge corresponds to the packed one.

> **Found while implementing.** For maps that are internally consistent the guard
> turns out to be vacuous: a deleted id does not exist in the middle graph, so a
> valid next map cannot mention it as a source, and the join never reaches it. The
> guard therefore protects hand-built and partially-validated maps rather than
> anything `Rewrite` produces. It is kept because it is two set lookups and it
> makes `compose` total on maps that have not been through `validate`. Since it is
> still a side condition on extension, associativity is demonstrated by test over
> create/delete chains rather than assumed.

**Provenance is not in the lattice.** "Was computed from" is directional and
asserts nothing about values; putting `Derived` in the claim lattice would make
`join Identical Derived = Derived` and destroy an identity claim on composition.
It gets `sources_of`/`targets_of` reverse lookup, never `invert`. Consequently
there is deliberately **no `Graph_map.invert`** — reversal is exposed per relation.
That is the honest reading of "the mapping should be reversible": the value and
node correspondences reverse; the computation history is queried backwards.

It is keyed by destination id, since every query is "where did this come from" and
a destination has one derivation. Its `compose` takes both value correspondences,
pulling sources back through the first and pushing targets forward through the
second, and closes transitively so a chain of derivations does not decay into a
one-step one:

```ocaml
val add : sources:Tensor_id.t list -> Tensor_id.t -> ('a,'b) t -> ('a,'b) t
val compose : ('a,'b) t -> ('b,'c) t ->
              values:('a,'b) Correspondence.t * ('b,'c) Correspondence.t -> ('a,'c) t
val sources_of : ('a,'b) t -> Tensor_id.t -> Tensor_id.Set.t
```

**Endpoint validation.** Phantoms tag a map but cannot tie it to two particular
graphs — `Correspondence.of_clusters` is polymorphic in its tags, so a caller can
hand out a well-typed map full of ids that exist in neither endpoint, and
`identity` type-checks between unrelated graphs. Maps built by `Rewrite` are
validated at construction; any consumer taking a map from elsewhere must call:

```ocaml
val validate : ('a,'b) t -> src:graph -> dst:graph -> (unit, error) Core.result
```

which checks explicit endpoints, that creations exist only in `dst` and deletions
only in `src`, node and provenance endpoints, and — the check that actually pins a
map to a *pair* of graphs — **implicit-identity coverage**: every `src` id neither
mentioned nor deleted must exist in `dst`, and symmetrically. That last check is
the one with teeth: an `identity` map between two *differing* graphs passes every
endpoint check trivially, because it names no endpoints at all, and is caught only
by coverage.

Failures are one `issue` variant shared by both instantiations —
`Dangling_src`/`Dangling_dst` for an endpoint absent from its own side,
`Uncovered_src`/`Uncovered_dst` for an unmentioned id missing from the other graph,
`Unpaired_src`/`Unpaired_dst` for an id mentioned on one side while present in
both — which `Graph_map` wraps per relation into `` `Value_endpoint ``,
`` `Node_endpoint `` and `` `Provenance_endpoint ``.

**The composite.** What a pipeline actually threads is the record of all three:

```ocaml
type ('src, 'dst) t = {
  values     : ('src, 'dst) Correspondence.t;
  nodes      : ('src, 'dst) Node_map.t;
  provenance : ('src, 'dst) Provenance.t;
}

val clusters      : ('a,'b) t -> Correspondence.Cluster.t list
val clusters_over : ('a,'b) t -> src:graph -> dst:graph ->
                    Correspondence.Cluster.t list
```

`clusters` returns only what is explicit; `clusters_over` synthesises the
untouched implicit identities from the two graphs, which the relation cannot do
alone because it does not know either id universe. The verifier wants the latter;
a golden or a diagnostic wants the former.

## 4. Id identity — the rule everything else leans on

Status: **implemented** — `id_supply.ml` holds the monotone supply and the frozen
origin watermark; `rewrite.ml` enforces the rule (`check_signatures`, then
`check_preserved` for redefinition).

> **An id may be kept only when the tensor is exactly the same one** — same value,
> shape, format and quantization. Any modification mints a new id, including a
> *lossless* widening such as F16→F32. A fresh id differs from every id the origin
> graph used.

Consequences: a preserved id can only ever carry an `Identical` claim;
`Equivalent` and `Approximate` always relate an old id to a new one; and the
implicit-identity convention in §3 is sound rather than optimistic.

This is enforced, not documented. `apply` rejects a preserved id whose
`Tensor_sig` differs at all (`` `Id_reuse_with_changed_value ``) — no claim can
excuse it. A preserved id whose *definition* changed but whose value did not
(`fold_const` moving a node-defined edge to a `Constant` input) is legal but must
be declared, or it is `` `Unclaimed_redefinition ``.

The supply is therefore **monotone forever**: seeded once from the origin and
never re-derived from the current graph, because re-deriving would let a deleted
high id be re-issued and an id would stop denoting one edge for the life of the
pipeline. Terminal packing (§9) is the single, explicitly bounded exception.

```ocaml
val of_graph : graph -> t          (* one past the highest id in each space *)
val origin   : t -> t              (* the frozen origin watermarks *)
val tensor   : t -> Tensor_id.t * t
val node     : t -> Node_id.t * t
val group    : t -> Group_id.t * t
val is_post  : t -> Tensor_id.t -> bool   (* introduced after the origin? *)
val equal    : t -> t -> bool             (* watermark equality, for §5's checks *)
```

Three independent counters, one per id space, plus the origin copy. `of_graph`
takes the maximum over everything a graph mentions — tensor-signature keys,
inputs, outputs, every node's outputs, every node id, and every group id in the
tree — rather than over live edges only, so a fresh id differs even from the ids
of edges a later step deletes. The origin watermarks never move, which is exactly
what lets §9 compact post-origin ids into a range disjoint from every origin id,
and `is_post` is how it tells the two apart.

## 5. The transform state

Status: **implemented** — `rewrite.ml` and `pass.ml`; tests in
`test/native/rewrite_test.ml` and `pass_test.ml`.

The driver sits on top: `Pass.per_node` and `Pass.of_pattern` collect *builders*
from a sweep, and the driver plans them one after another so their allocations
are contiguous, merges them, and applies once — so a sweep is a single step with
a single mapping rather than N steps whose maps the caller would compose.
`Pass.fixpoint` re-sweeps until nothing changes, composing each iteration into
the accumulated mapping; exhausting `max_iters` is an error rather than a silent
stop, because the answer would otherwise depend on the bound.

> **Found while implementing.** `fixpoint`'s accumulator cannot be a
> `('v,'w) Graph_map.t` — its destination changes every iteration, so `'w` would
> escape. `Rewrite.step` already packages a state with the map reaching it,
> existentially, so the accumulator *is* a step and composing into it keeps the
> caller's view as one mapping from the state handed in to the final graph.

```ocaml
type 'v t                                   (* graph + constants + supply + origin watermark *)
type origin = Origin : 'v t -> origin
val origin : ?constants:(Tensor_id.t * Tensor.packed) list -> graph ->
             (origin, error) Core.result    (* the ONLY minter of a version *)
val graph     : 'v t -> graph
val constants : 'v t -> Tensor.packed Tensor_id.Map.t
val view      : 'v t -> Graph_view.t        (* validated index of the current graph *)

type 'v allocator                           (* abstract; watermarked *)
type 'v recipe                              (* abstract; retains start and end watermark *)
val allocator : 'v t -> 'v allocator
val plan  : 'v t -> 'v allocator -> Recipe.builder ->
            ('v recipe * 'v allocator, error) Core.result
val merge : 'v recipe -> 'v recipe -> ('v recipe, error) Core.result

type 'v step = Step : 'w t * ('v,'w) Graph_map.t -> 'v step
val apply : 'v t -> 'v recipe -> ('v step, error) Core.result
val pack  : 'v t -> 'v step
```

**The state is the versioned value.** Constant payloads and the id supply are
carried *inside* it, so they are cumulative by construction — a design where a
step returned a payload delta made fixpoint constant folding impossible to write
(iteration 2 has no payload for iteration 1's output) and that bug is now
unexpressible.

**Every version-bound abstraction lives in this one module.** `Recipe` holds only
unversioned data and builder combinators. That is not tidiness: with
`wrapped false` there is no internal visibility, so a type defined in `Recipe` and
constructed by `Rewrite` would need a public constructor that anything could use to
forge an allocator or a versioned recipe.

**Version safety.** `'v` is minted only by `origin`; validation is unversioned
(`Graph_view.validate : graph -> validated`) and the only route to a versioned view
is `Rewrite.view`. A `val of_graph : graph -> ('v t, …)` would let the caller pick
`'v` and present graph B under graph A's tag.

**Watermarks.** Allocators are immutable and therefore copyable, so a watermark
check alone permits a branched history (`a0 → plan r1 → a1` and `a0 → plan r2 → a2`
both start at `a0` and allocate the same ids). `merge` requires **contiguous
allocation intervals in argument order**; `apply` requires the recipe's start
watermark to equal the state's current one.

**Constant payloads are a trust boundary.** `origin ~constants` rejects duplicate
ids, ids that are not effective `Constant` graph inputs, and payloads whose shape
or format contradicts their `Tensor_sig`. A recipe may bind only a newly created id
or one moving from node-defined to `Constant`; rebinding an existing constant is
`` `Constant_payload_overwrite ``, because payloads live in the state rather than
the graph and such an overwrite would change a value with no structural trace at
all.

## 6. The recipe

Status: **implemented** — `recipe.ml`.

```ocaml
type insertion = { op : op; outputs : Tensor_id.t list; from : Node_id.t list }
type placement = Inherit | New_group of string option

type replacement = {
  remove       : Node_id.Set.t;
  insert       : insertion list;
  placement    : placement;
  tensors      : Tensor_sig.t list;
  subst        : Tensor_id.t Tensor_id.Map.t;      (* REWIRING only *)
  value_claims : (Tensor_id.t * Tensor_id.t * Correspondence.relation) list;
  constants    : (Tensor_id.t * Tensor.packed) list;
  provenance   : (Tensor_id.t list * Tensor_id.t) list;
}
```

**A list of independently placed replacements**, not one global
remove/insert/placement record: two independent rewrites in different groups must
keep their own placements instead of both migrating to the root when merged.

**`insertion` carries no id.** `apply` stamps inserted nodes, so every `Node_id.t`
in a recipe is source-side by construction and the fresh-node namespace does not
exist to be confused with the source one. `placement` likewise names no fresh
group. Tensors cannot get the same treatment — an inserted op must name the fresh
edge the next one consumes, and payloads hardcode `Tensor_ref.t` — so there the
guarantees are the recipe's version tag plus validation.

**`subst` is rewiring; `value_claims` is correspondence.** They are separate
because a claim can involve no rewiring at all: a self-claim for a redefined edge
would otherwise need a `t ↦ t` substitution, which normalisation either rejects as
a cycle or erases along with its claim.

**`apply` never infers a claim.** Every constructor emits its claims explicitly:

| Constructor | `subst` | `value_claims` |
|---|---|---|
| 1:N decomposition | fragment output ↦ source output | `(old_out, old_out, Identical)` |
| N:1 / M:N fusion (convex region) | replacement output ↦ boundary output | `(old_out, old_out, Identical)` |
| trivial trimming | removed output ↦ surviving upstream edge | `(removed_out, surviving, Identical)` |
| `fold_const` | — | `(out, out, Identical)` |
| value-changing rewrite (bn fold) | old output ↦ **fresh** id | `(old_out, fresh, Equivalent)` |
| precision change | old edge ↦ **fresh** id | `(old, fresh, Approximate …)` |

The last two rows are forced by §4. Self-claims are required because
`` `Unclaimed_substitution `` fires only when a substitution's *source* survives,
and boundary renaming `fresh_frag_out ↦ old_out` has a source that never existed —
so without them a decomposition that replaced `Relu` with `Sqrt` behind a preserved
output id would be handed an implicit `Identical`.

## 7. `apply`

Status: **implemented** — `rewrite.ml`. In order:

1. **Validate** against the source view: known nodes, disjoint replacement regions,
   convexity where a constructor requires it, start watermark equal to the state's.
2. **Normalise the substitution.** Union every `subst`, rejecting duplicate keys
   with differing targets; compute the transitive normal form, rejecting cycles;
   require each terminal target to be defined after the rewrite; require every
   substitution whose source survives to be covered by a claim. Only the normalised
   map is applied, once. This is what makes two independently matched no-ops merge:
   `t2 ↦ t1` and `t1 ↦ t0` normalise to both pointing at `t0`.
3. Per replacement: drop `remove`, stamp fresh `Node_id`s, splice.
4. Apply the normalised substitution to definitions, operands, `Graph.inputs`,
   `Graph.outputs`.
5. Register `constants` as `Constant` inputs; drop sigs and input entries for edges
   with neither definition nor use.
6. Stable topological re-sort; rebuild the group tree; re-verify shapes via
   `Graph_shape.output_shape`; re-run `Graph_view.validate`.
7. **Enforce the id-identity rule** on every preserved id (§4).
8. **Propagate claims** in topological order (§8).
9. **Validate the mapping metadata**: claim and provenance endpoints resolve on
   their own sides, corresponding shapes agree, `Identical` endpoints have
   compatible `fmt`/`quant` — an `Identical` claim across F32 and BF16 is a
   contradiction, it is `Approximate`.
10. **Build the map**, closing each cluster over identity: for every terminal
    target surviving in both graphs, merge in its implicit `A.t ↔ B.t` identity.
    Trimming `t0 →noop→ t1 →noop→ t2` yields `A.{t0,t1,t2} ↔ B.{t0}`, not
    `{t1,t2} ↔ {t0}` — without the closure the explicit and implicit clusters would
    overlap and the relation would not be an equivalence.

> **Two ordering constraints found while implementing.** Both are load-bearing,
> and both are invisible until something fails.
>
> The clusters have to be built **before** step 8, not after, because "did this
> definition change" is a question about clusters rather than raw claim pairs.
> Deriving a representative pairwise gives `t1 ↦ t0` from one claim and
> `t1 ↦ t1` from the next, so after trimming a chain a consumer rewired from
> `t2` to `t0` looks redefined and is rejected. Taking the representative from
> the normalised cluster answers correctly. A claim's destination also has to be
> read through the normalised substitution first, or a claim naming an edge that
> was itself substituted away points at an id the destination graph no longer has.
>
> The **signature** half of the id-identity check has to run before the result is
> validated as a graph. Otherwise a recipe that takes over an id with a different
> shape is reported as "this node's output signature disagrees with its op",
> which is true but describes the symptom; the cause is that it reused an id it
> was not entitled to.

**Group maintenance.** Removed nodes drop out of their items; a group whose subtree
empties is pruned; insertions are appended to the nearest common ancestor of the
replacement's touched nodes, or to a fresh child group when
`placement = New_group`. Sibling items are ordered by the topological position of
their first node.

> **Limitation — group order is structural, not executable.** Sorting siblings by
> topological position cannot in general reproduce the execution order: a group
> owning nodes at positions 1 and 3 with a parent-owned node at 2 has no item
> ordering expressing group-node, parent-node, group-node. So `Graph.nodes` is the
> execution order and always a valid topological order, while the group tree's
> traversal order may differ. The alternative — require every group subtree to be a
> contiguous topological interval, splitting groups when a rewrite breaks it — is
> what to adopt if a consumer ever needs interval semantics, e.g. a scheduler
> fusing a group as a unit.

## 8. Claim propagation

Status: **implemented** — `rewrite.ml`'s `propagate`, over `output_transfer.ml`.

A node whose op is unchanged *modulo cluster representatives* (so a consumer whose
operand was merely rewired to an equal edge counts as unchanged) gives its outputs
the `join` of its operands' claims, to a fixed point. This is a correctness
requirement, not an optimisation: after a bn fold declares `bn_out ↔ fresh` is
`Equivalent`, every downstream edge is `Equivalent` too, and leaving them
implicitly `Identical` would have a verifier assert bit-equality on the model's
final output and fail.

Propagation is per output, driven by an exhaustive classifier with **no default** —
a defaulting classifier would silently mis-transfer the next op someone adds:

```ocaml
val output_transfer : op -> output:int -> [ `Continuous | `Discontinuous | `Reindexing ]
```

| Incoming claim | `Reindexing` | `Continuous` | `Discontinuous` |
|---|---|---|---|
| `Identical` | `Identical` | `Identical` | `Identical` |
| `Equivalent` | `Equivalent` | `Equivalent` | `Unverifiable` |
| `Approximate s` | `Approximate s` | `Unverifiable` | `Unverifiable` |
| `Unverifiable` | `Unverifiable` | `Unverifiable` | `Unverifiable` |

`Identical` survives everything because evaluation is deterministic. `Equivalent`
is a claim about exact arithmetic, so it survives any continuous output but not a
discontinuous one — a rounding difference does not perturb an argmax slightly, it
selects a categorically different element, and the checker compares computed
values.

> **Limitation — `Approximate` transfer.** It dies at any actual computation
> because continuity gives no error *bound*: multiplication amplifies by the
> magnitude of its other operand, reductions accumulate, `Sqrt` has unbounded
> sensitivity near zero, and quantized saturation is only piecewise continuous.
> Only a proven reindexing carries it, the value multiset being unchanged. A real
> op-specific error-transfer policy is future work; nothing in the staged passes
> emits `Approximate`, so the conservative table costs nothing today.

Current classification: `Reindexing` = `Permute`, `Reshape`; `Discontinuous` = the
index output of `Max_pool2d_with_indices`; everything else `Continuous`. `Relu` and
the pooled *value* output are branch-selecting but continuous, so they propagate
normally.

## 9. Terminal packing

Status: **implemented** — `rewrite.ml`'s `pack`, over `Id_supply.origin_marks`
and `repack`; tests in `test/native/pack_test.ml`.

Monotone allocation makes ids creep. `pack` compacts them once at the very end.
Origin ids keep their values — renumbering them would be exactly the reuse §4
forbids, and it would turn every untouched tensor into an explicit rename,
destroying the implicit-identity bulk. Only ids introduced after the origin are
compacted, assigned densely upward from the origin's per-space high-water mark in
canonical order: graph inputs first in signature order, then nodes in topological
order with each node's outputs in order, groups in tree pre-order.

> **Its contract is "never reuses an origin id", not "never reuses an id".**
> Compacting upward from the origin watermark can still land a live post-origin id
> on a value some *dead* post-origin id held — create `t11`, delete it, create
> `t12`, pack `t12 → t11`. Nor is "only the origin and final graph remain"
> enforceable: graphs are immutable and callers may retain intermediates. The
> accurate statement is that the reuse is confined to dead post-origin values, is
> always recorded in the map, and is why `compose` needs the guard in §3.

`pack` is a distinct terminal operation, not one of the four transformation kinds.
Its map is all-`Identical` and mentions only repacked ids, so it composes like any
other step and the PT2 lens still resolves packed ids. Constant payloads are
renumbered with them.

It is **idempotent**, which is the check that the canonical order is genuinely
canonical: a second pack re-enumerates the same structure and can only differ from
the first if the enumeration depends on something other than the graph.

> **Changed while implementing: `pack` returns a `result`.** The sketch had
> `val pack : 'v t -> 'v step`, total. But the state holds a *validated* view, and
> the only way to one is `Graph_view.of_graph`, which is fallible — there is no
> honest way to fabricate the view a total signature would need. Running the
> result back through the trust boundary is also the right thing on its own
> merits: a bulk renaming of every id in a graph is exactly the edit that can
> silently produce something nobody would accept.

> **Found while implementing: the dead-id reuse is not hypothetical, and the
> pipeline already produces it.** Batch-norm folding mints `t10`–`t19`, constant
> folding kills `t10`–`t14`, and packing then lands the surviving `t15` **on
> `t10`**. So the §3 identity-extension guard is exercised by the ordinary
> pipeline rather than by a constructed case, and `pack_test.ml` pins that the
> composed map keeps `{} → {t10}` (a creation) separate from the dead `t10`,
> under both association orders. Provenance survives it too: the folded weight
> resolves back to the origin's weight, gamma and var with a packed destination
> id, which is precisely what the PT2 lens will walk.

## 10. PT2 provenance — recovered, never carried

Status: **implemented** — `pt2_native_graph.ml`'s `lens` and its queries,
`native_interp.ml`'s `transform`/`evaluate`, and the `native_graph transform`
subcommand; tests in `test/native/lens_test.ml`, `test/native_transform_cram.t`
(structure) and `make native-transform-verify` (numbers).

**The sidecar is not transformed.** `Pt2_native_graph.t` stays anchored to the
original graph exactly as the importer built it; a destination id's origin is
recovered by walking the composed map backwards. There is no `remap`, no
`materialized` field, no per-step revalidation and no drift, because there is only
ever one sidecar value.

```ocaml
type 'dst lens
val lens : t -> src:'a Rewrite.t -> ('a,'b) Graph_map.t -> dst:'b Rewrite.t ->
           ('b lens, error) Core.result
val tensor_origins     : 'b lens -> Tensor_id.t -> (Tensor_origin.t list, error) Core.result
val node_origins       : 'b lens -> Node_id.t -> (Node_origin.t list, error) Core.result
val captured_target    : 'b lens -> Tensor_id.t -> (string option, error) Core.result
val provenance_sources : 'b lens -> Tensor_id.t -> Tensor_id.t list  (* diagnostics only *)
```

**Both endpoint states are required.** Sparse correspondence resolves an
unmentioned id to itself, so without `dst` a bogus destination id would silently
"resolve" to the same-numbered source. The lens stores destination membership sets
and rejects unknown ids. It validates once at construction: the sidecar's graph
against `Rewrite.graph src` by comparing canonical `graph_jsont` encoded bytes
(§2), plus `Graph_map.validate ~src ~dst` (§3) — the byte comparison ties the
sidecar to the source, the map validation ties the *map* to both graphs.

An id-set fingerprint would be worthless: builders start at zero, so unrelated
graphs routinely share ids while differing in ops, params, sigs, constant
classification or group contents. Direct value comparison is not the alternative
either — `graph` contains `Tensor_id.Map`, whose tree shape depends on insertion
order, so `=` can report two identical maps unequal.

**Provenance is never a fallback for identity or payload.** For
`archive w --Permute--> wp`, provenance says `w → wp`, but w's archive bytes are in
the *unpermuted* layout and are not a valid payload for wp. Conflating the two is
data corruption, not imprecision. So `captured_target` follows **only an
`Identical` correspondence** to a captured source (picking the lowest id when a
many-to-one cluster offers several — all members are `Identical` and therefore
interchangeable), `tensor_origins` inherits through value correspondence only and
returns `Tensor_origin.t` payloads so that "derived" has exactly one
representation (the empty list, never `[Derived]`), and `provenance_sources` is a
separate diagnostic API. Results are sorted and deduplicated: tensor origins by
source id, node origins by `(graph_path, index)`.

**Constant payloads split the same way.** Archive-captured data is reached by
resolving a destination constant back to its source `captured_target`;
pass-computed data lives in the final `Rewrite.constants`. `Native_interp.evaluate`
consults the state's payload map first and falls back to the archive through the
lens, replacing "load every constant via `captured_targets`", which cannot see a
folded constant at all. `run` itself is untouched: it transforms nothing, so it has
no folded constants and no reason to take the new path.

`transform` and `evaluate` are separate because seeing what a pipeline produced
should not require running inference over it — which is also what lets the cram
test pin structure without paying for, or depending on, arithmetic.

> **Found while implementing: the lazy path is the interesting one, and it
> decides what a pass can do.** `transform` seeds the origin state with *no*
> payloads, so a structural pipeline never materialises a weight — but `Fold_const`
> then declines every node, because it refuses a constant whose payload is not
> bound (§12b). `~preload:true` binds every captured payload a node reads first,
> which is what lets folding hoist a permuted weight. Two genuinely different
> modes, and the flag is the honest way to say which: on ResNet-18 the structural
> pipeline removes 41 nodes reading nothing from disk, and the preloaded one
> removes 62 and hoists 21 weights.

> **Found while implementing: only load what a node reads.** Preloading every
> `captured_target` fails on ResNet-18 — the archive carries an int64
> `num_batches_tracked` per batch-norm module that no node evaluates, and the
> engine has no reason to support the dtype. Filtering to operands is both the fix
> and the cheaper thing to do.

**On a real model** (`test/native_transform_cram.t`), the ResNet-18 import goes
from 174 nodes to 112 with identical outputs. The 41 nodes the permute passes
remove are the inverse pairs the relayout lowering emits at each op boundary —
§1's second case — and the 21 the fold removes are the constant weight permutes,
§1's motivating case. Each hoisted weight reports `captured_target = None` and
names its origin through provenance (`t124 <- [p_conv1_weight]`), which is exactly
the split this section exists to enforce.

**Pass order is load-bearing, and the reason is a layout.** The importer emits
every conv weight behind a relayout `Permute`, so the weight is a *node output*
until folding materialises it — and batch-norm folding requires constant
parameters (§12c). So the pipeline runs constant folding first to make the
weights constant, then the batch-norm fold, then constant folding again to
collapse the parameter arithmetic that fold emits. With `Fold_batch_norm` placed
before the first fold it matches nothing at all, which is easy to mistake for the
pattern being wrong.

With that order, ResNet-18 goes from 174 nodes to **92**: all 20 batch norms
disappear into their convolutions, and 41 constants are computed at load time
where the imported graph recomputed them on every inference.

**And the claim lattice predicts the numbers.** The structural pipeline claims
only `Identical`, and `--verify` reports `max_abs=0` — bit-identical over all
1000 logits. Add the fold, whose one `Equivalent` claim says "equal in exact
arithmetic, rounds differently", and the same comparison reports
`max_abs=1.9e-06`. A verifier reading the map would assert bit-equality in the
first case and a tolerance in the second, which is exactly what §3 exists to let
it decide.

**Structure is pinned; numbers are checked.** `test/native_transform_cram.t`
prints the transformed graph — exact, fast, and annotated with provenance
recovered through the lens — while `make native-transform-verify` executes it
and compares against the untransformed run. The split is deliberate: a full
inference is far too slow for the test suite, and the residual above is floating
point, so pinning it as golden text would make the suite depend on the
platform's arithmetic.

## 11. Matching

Status: **implemented** — `graph_view.ml`, `region.ml` and `pattern.ml`, with
tests in `test/native/graph_view_test.ml` and `pattern_test.ml`, over the shared
fixtures in `test/native/graph_fixtures.ml`.

> **Changed while implementing: the view is unversioned.** The plan had
> `'v Graph_view.t`, so that a recipe built from a match would inherit the view's
> version tag. Writing the signature showed the tag would be forgeable anyway —
> `of_graph : graph -> ('v t, error) result` lets the caller pick `'v`, exactly the
> hole this design already closed twice elsewhere — while forcing every `Pass`
> callback taking a view to be a rank-2 record. It buys nothing and costs real
> complexity, so `Graph_view.t` is plain.
>
> Version safety does not depend on it. A recipe's tag comes from
> `Rewrite.plan`, which takes the *state*, and that cannot be faked. What remains
> is matching against a view of one version and planning against another; the
> id-identity rule (§4) makes that safe rather than merely unlikely, because an id
> is never reused within a pipeline, so an id present in two versions denotes the
> same tensor. A stale match is therefore either caught by `apply` (the id is gone)
> or harmless (it still means what it meant).

A state+error monad over `{ view; claimed; shared }`. **No cursor** — every
primitive names the edge it operates on, which is the honest formulation for a
DAG and reads better than a hidden position. Matching walks *up* operand
edges, which is deterministic because operands are typed fields; `uses` covers
fan-out.

```ocaml
val def   : Tensor_id.t -> (op -> 'a option) -> ('a * node) t  (* producer; claims it *)
val peek  : Tensor_id.t -> (op -> 'a option) -> ('a * node) t
val uses  : Tensor_id.t -> node list t
val interior : Tensor_id.t -> unit t     (* exactly one use and not a graph output *)
val constant : Tensor_id.t -> unit t
val claim : node -> unit t               (* exclusive: this match removes or rewrites it *)
val claim_shared : node -> unit t        (* read-only: this match only reads it *)
val chain : (Tensor_id.t -> ('a * Tensor_id.t) t) -> Tensor_id.t -> 'a list t
val run  : 'a t -> Graph_view.t -> ('a * Region.t, failure) result
val scan : (Tensor_id.t -> 'a t) -> Graph_view.t -> ('a * Region.t) list
```

The projector is a plain `op -> 'a option`, so patterns destructure the existing
typed payloads directly (`function Relu r -> Some r | _ -> None`) and there is no
per-op pattern DSL to keep in sync as ops are added. Sequence is `let*`; fan-out is
`uses`/`interior`; a union is two `def` calls on two operand fields of one payload;
alternation is `choice`; repetition is `chain`.

Specified semantics, each pinned by a test:

- **Anchors.** `scan` anchors on every node output edge in `Graph.nodes` order,
  each output of a multi-output node separately. `Discard` has no outputs and is
  reached only through `uses`.
- **Overlap is read/write, not a flat node-set intersection.** Greedy: first
  match in anchor order wins; a later match is discarded, not re-attempted, if
  it *conflicts* with an accepted one — where two matches conflict iff one's
  `claim` (exclusive) lands on a node the other touches at all, `claim`ed or
  `claim_shared`ed. Two matches that only `claim_shared` the same node do
  **not** conflict, since neither removes or rewrites it. So `scan`'s results
  are pairwise *safe to apply together*, which for an all-`claim` pattern (every
  pass but §12e's `Reuse_permute`) is exactly node-disjoint, same as before
  `claim_shared` existed.
- **Rollback.** State is immutable, so a failed `choice`/`optional` branch discards
  its state wholesale, including claims. Only `Mismatch` backtracks; `Invalid`
  aborts the scan.
- **Re-claiming.** Claiming an already-claimed node inside the same match is
  idempotent success — that is what makes diamond patterns work.
- **Progress.** `chain` rejects a step whose returned edge is not strictly earlier
  in topo order, so a non-advancing parser cannot loop.

`Region` derives its boundary from the claimed node set (`claim` and
`claim_shared` together — a region is what the match *touches*, exclusively or
not) — inputs are operands whose producer is unclaimed, outputs are claimed
outputs used outside or in `Graph.outputs` — and reports `convex` rather than
enforcing it: non-convexity
forces a cycle only when the replacement collapses the region to one scheduling
point, so the fusing constructors require it while other recipes need not, and
`apply`'s cycle check is the general backstop. `Region.extract` builds a standalone
graph so a match prints with plain `Graph_ir.pp`.

`Graph_view.of_graph` is the framework's trust boundary and checks: unique node and
group ids, single assignment, every operand defined-or-declared-input, every graph
output present, `tensors` key equal to `Tensor_sig.id`, inputs unique and not
node-defined, **every `input_kinds` key a member of `Graph.inputs`** (totality is
*not* required — the map is sparse by design, §2), node output arity matching
`Graph_shape.output_shape`, `nodes` genuinely topologically ordered, and `root` a
partition of `nodes`. Without it, `def = None` is ambiguous between "graph input"
and "dangling operand" and every matcher silently misbehaves.

Two details the tests pin, both easy to get subtly wrong:

- **`uses` deduplicates consumers.** A node reading the same edge twice (`mul x x`)
  is *one* consumer. Otherwise `interior`'s "exactly one use" would refuse to fuse
  a perfectly fusable edge.
- **`topo_sort` takes no view** and resolves producers from the node list it is
  given. Its caller is `apply`, re-sorting a list containing freshly inserted
  nodes; those outputs are absent from any view of the pre-rewrite graph, so
  resolving through a view would make dependencies *between new nodes* invisible
  and emit a plausible but wrong order.
- **An output's signature must be the shape its op produces.** Arity alone lets a
  rewrite install an edge whose declared shape contradicts its operands, which
  then surfaces only at evaluation. Added while building `apply`, whose whole job
  is producing graphs that were not written by the builder.

Convexity is subtler than "the region skips a node": in the diamond fixture,
claiming `{relu, add}` skips the `mul`, but the `mul` reads the graph input rather
than the `relu`, so no path leaves the region and returns — it is convex. The
non-convex case needs a claimed node feeding an unclaimed one that feeds back in.

## 12. Staging and the transformations

Each stage is one commit carrying code, its expect tests, and this doc's status
delta. Stages 1–5 are the framework, 6–9 the transformations, 10–11 integration.

| # | Commit | Lands | Status |
|---|---|---|---|
| 0 | `docs: design the native graph transformation framework` | this doc; `native_graph_design.md` §7 → pointer | done |
| 1 | `native: add id_supply and graph mapping` | `Tensor_id.Set`/`Node_id.Set`, `Group_id` ordering, `Id_supply`, `Cluster_relation`, `Correspondence`, `Node_map`, `Provenance`, `Graph_map` | done |
| 2 | `native: add graph_view and region` | validation, the index, `common_group`, topo sort, `Region`, `test/native/graph_fixtures.ml` | done |
| 3 | `native: add recipe and rewrite` | the state, `Recipe`, `apply`, `output_transfer` (+ its entry in `native_add_op.md`) | done |
| 4 | `native: add graph match combinators` | `Pattern`, `run`, `scan` | done |
| 5 | `native: add pass driver` | `Pass`, `fixpoint`, `run_all` | done |
| 6 | `native: add permute simplification passes` | `Trim_permute`, `Chain_permute`, `Reshape_to_permute`, perm algebra on `Permute.Permute` | done |
| 7 | `native: add constant folding` | `Fold_const` — the motivating permute-of-constant-weight case; `Pass.env` | done |
| 8 | `native: add Sub, Div and Sqrt ops` | prerequisite for bn folding; two stale steps fixed in `native_add_op.md` | done |
| 9 | `native: add batch-norm folding pass` | `Fold_batch_norm`; recipe-payload validation in `apply` | done |
| 10 | `native: add terminal id packing` | `Rewrite.pack`, `Id_supply.origin_marks`/`repack` | done |
| 11 | `native: resolve PT2 provenance through a transformation map` | `Pt2_native_graph.lens`; `Native_interp.transform`/`evaluate`; `native_graph transform` | done |
| 12 | `native: sink permutes through elementwise ops` | `Sink_permute` (§12d) — cancels a permute pair flanking `Relu`/`Add`/... | done |
| 13 | `native: reuse existing layouts and bypass inverse permutes` | `Reuse_permute`, `Bypass_permute` (§12e), `Permute.Permute.are_inverse`, `Pass.sequence`, `Pattern.is_graph_output`, `relayout_pass` | done |
| 14 | `native: transport permutes through keepdim=true Mean` | `Sink_permute_mean` (§12f), `Reduce.Mean.map_dims` | done |

### 12a. The permute simplifications

Status: **implemented** — `lib/native/transform/passes/`, tests in
`test/native/permute_passes_test.ml`.

Perm algebra (`compose`, `identity`, `is_identity`, `lookup`, `of_fn`) lives on
`Permute.Permute`, not in the transform layer: composing two permutations is a
property of the op, and stage 9 needs to *build* one (the C→N permute of the
batch-norm scale). `compose ~before ~after` reads in dataflow order —
`y = before x`, `z = after y` — and `of_fn` emits pairs in `Axis.all` order so
composing twice yields the same list, which keeps goldens stable.

| Pass | Matches | Emits |
|---|---|---|
| `Trim_permute` | a maximal run of permutes ending at the anchor whose composition is the identity | `Recipe.trim`, tying each edge whose *prefix* already composes to the identity onto the run's input |
| `Chain_permute` | two adjacent permutes with an interior edge between them | one `Permute` taking over the outer output id, `from` naming both removed nodes |
| `Reshape_to_permute` | a contiguous `Reshape` that only relabels axes | one `Permute` taking over the reshape's output id |

Three things they establish for every pass that follows.

**Only the anchor may escape.** `Trim_permute` walks upstream with
`Pattern.chain`, and every edge but the anchor must be `interior`. An
intermediate value of a cancelling run is a genuine rearrangement of the run's
input, *not* equal to it, so a second consumer would lose its producer.
`Chain_permute` needs the same precondition for a different reason: with a shared
intermediate the inner node has to stay, so "fusing" would duplicate work rather
than remove it. Failing `interior` merely ends a chain — a shorter run may still
cancel, which is exactly how the partially-cancelling fixture is handled.

**A preserved id needs a claim even when the value is untouched.** `Chain_permute`
and `Reshape_to_permute` both keep the anchor's id: same tensor, same shape, so §4
is satisfied. Their *definition* changes, though, and `apply` never infers — so
each emits `(out, out, Identical)` explicitly. `Recipe.replace` adds that
automatically for ids taken over via `subst`; these two take the id over through
the insertion's `outputs` instead, where nothing can infer it for them.

**`Reshape_to_permute`'s legality condition.** A contiguous reshape sends element
*k* of the row-major input to element *k* of the row-major output, and unit axes
contribute nothing to a row-major offset. So when the **non-unit extents of the
two shapes agree in axis order**, the reshape is exactly the bijection carrying
the i-th non-unit input axis onto the i-th non-unit output axis; leftover unit
axes pair up in canonical order, since a unit axis computes the same tensor
whichever way it is matched. Anything else — a genuine flatten or split — mixes
extents and no permutation of six axes expresses it. The conversion is worth doing
because a permute *composes*: it fuses with its neighbours and cancels against
them, where a reshape is opaque to both. This is the one pass here whose
correctness rests on an argument about offsets rather than a syntactic identity,
so it is pinned by a numeric test (`Eval_direct` over both graphs) on a case that
moves two non-unit axes, where a plausible-but-wrong perm still yields the right
shape.

> **Found while implementing: mapping precision depends on the route taken.**
> Collapsing an all-identity run with `fixpoint Trim_permute` yields
> `{t0,t1,t2,t3} → {t0} Identical`; reaching the *same destination graph* via
> `fixpoint Chain_permute` then `Trim_permute` yields `{t0,t3} → {t0}` plus
> `{t1} → {}` and `{t2} → {}`. Both are sound — a deletion asserts nothing — and
> the difference is honest rather than a defect: fusing two permutes genuinely
> does not know that the intermediate equalled the input, and in general it does
> not. A verifier gets fewer clusters to check on the second route, never a false
> one. Worth stating because "the map is precise" is not a property a pipeline
> preserves; "the map is sound" is.

> **Found while implementing: greedy disjoint matching favours the shortest run.**
> `Pattern.scan` anchors in node order and drops later matches that overlap an
> accepted one, so on a three-permute chain the one-node run anchored at `t1` wins
> and the two- and three-node runs are discarded. One sweep removes one node;
> `Pass.fixpoint` is what collapses the chain. That follows from §11's specified
> overlap rule rather than contradicting it, but it does mean a pass author cannot
> assume a sweep takes the *largest* match, and any pass with nested matches wants
> a fixed point.

### 12b. Constant folding

Status: **implemented** — `lib/native/transform/passes/fold_const.ml`, tests in
`test/native/fold_const_test.ml`.

`Fold_const` is what lets every other pass stay simple. Constant arithmetic is
expressed as **ordinary graph ops** — a relayout emits a `Permute` of a weight,
batch-norm folding (§12c) will emit its parameter arithmetic as nodes — and this
one pass turns the resulting all-constant sub-DAG into data. No pass computes on
payloads itself, so no pass needs an evaluator, and there is exactly one place
where a value can be got wrong.

It folds a node with **exactly one output and at least one operand**, every
operand a `Constant` with a bound payload. It refuses:

- **zero outputs** (`Discard`) — removing those is dead-code elimination's job;
- **several outputs** — binding them all needs one recipe that says so, which
  `Recipe.fold_to_constant` does not express. Atomic multi-output folding is
  future work, recorded rather than half-done;
- **a constant with no bound payload** — payloads live outside the IR (§2), so
  "declared constant, not yet loaded" is a legitimate state, not an error. The
  node simply stays.

Evaluation goes through `Region.extract` + `Eval_direct.run`: the matched node
becomes a one-node standalone graph (boundary inputs keep their `Input.kind`,
which is what makes the operands resolve as constants) and runs down the same
path inference takes. There is no second evaluator to disagree with the first.
`Eval_direct` returns every edge, so the value is available whether or not the
folded output escaped the region.

The recipe keeps the node's output id — same tensor, computed earlier — so it is
a self-claim plus a provenance edge, exactly §3's worked example: for
`w --Permute--> wp`, `wp` stays implicit, `{w} ↔ {}` is a deletion, and
`[w] → wp` is provenance. Under `Pass.fixpoint` a multi-node sub-DAG collapses
one layer per sweep, since a node only becomes foldable once the sweep before it
bound its operands, and the intermediate payloads drop out of the state as they
stop being referenced.

**Evaluation failure is unreachable, and skipping is the response.** Each way
`evaluate` could fail is ruled out by something already checked: the region is a
known singleton, shapes and arities were verified by `Graph_view.of_graph`, the
extracted graph has no `Input`-kind inputs because every operand is constant, and
every operand has a payload. Rather than thread a new error row through `Pass`
for a case that cannot arise, the pass declines to fold — which leaves the graph
computing the node itself, i.e. correct.

> **Changed while implementing: a pass sees `Pass.env`, not a bare view.**
> `per_node`'s callback took a `Graph_view.t`, and constant folding cannot be
> written against one — it has to *evaluate*, which needs the payloads. So the
> callback now takes `{ constants; view }`, both read off the state by the
> driver. The payloads must be the state's cumulative map rather than a
> per-step delta, or the second iteration of a `fixpoint` fold would have no
> payload for the first iteration's output; §5 already required that of the
> state, and this is where it pays. `of_pattern` still passes only the view,
> because `Pattern` is defined over one and no pattern-based pass has needed
> payloads; one that does should grow the env there too rather than reach
> around the driver.
> `Pattern.scan` anchors in node order and drops later matches that overlap an
> accepted one, so on a three-permute chain the one-node run anchored at `t1` wins
> and the two- and three-node runs are discarded. One sweep removes one node;
> `Pass.fixpoint` is what collapses the chain. That follows from §11's specified
> overlap rule rather than contradicting it, but it does mean a pass author cannot
> assume a sweep takes the *largest* match, and any pass with nested matches wants
> a fixed point.

### 12c. Batch-norm folding

Status: **implemented** — `lib/native/transform/passes/fold_batch_norm.ml`, tests
in `test/native/fold_batch_norm_test.ml`.

With `s = gamma / sqrt(var + eps)`, inference batch norm is `y = (z - mean) * s +
beta` over `z = conv(x, W) + b`, so

```
y = conv(x, W * s) + (b - mean) * s + beta
```

and the normalisation disappears into the conv's parameters. **Layouts decide the
shape of the rewrite.** `beta`, `mean`, `var` and the conv bias all live on C, so
the new bias is plain pointwise; the weight is `[Cout,1,1,Kh,Kw,Cin]` with Cout on
**N**, so the scale is permuted C → N before it multiplies the weight. The
arithmetic is emitted as ordinary graph nodes and §12b collapses it, leaving a
plain conv with two computed constants.

It folds into `Conv2d` **and** a forward `Convolution`, which is the op the PT2
importer actually emits — resnet18 lowers every convolution through
`aten.convolution.default`. A non-transposed `Convolution` delegates its shape
and compute straight to `Conv2d`, so one rewrite serves both and only the
reconstruction differs; both arms get the same eight-way numeric check rather
than one being a spot check.

Preconditions, each checked by the pattern:

- `bn.channel = Axis.C` — the rewrite moves the scale onto the weight's N axis,
  which is the channel axis only for a C-norm.
- the convolution is **not transposed**. `Conv.Convolution.bias_shape` takes a
  transposed conv's channel count from the weight's C rather than its N, so the
  C → N permuted scale would broadcast against the wrong extent. Nothing about
  the arithmetic would look wrong — only the axis — which is why the test shows
  the forward and transposed forms of one weight side by side.
- the conv output is `interior` — folding removes the node computing it, and any
  other consumer still wants the *unnormalised* value.
- every parameter is `Constant`. Not a correctness condition but an economic one:
  on a runtime parameter the fold moves a per-channel scale of the *output* onto
  the whole weight tensor, which is more work per inference, not less.
- `weight.N = param.C` — otherwise the permuted scale broadcasts against the
  wrong extent.

**The folded conv gets a fresh id.** It is equal in exact arithmetic but rounds
differently, and §4 permits keeping an id only for the very same tensor. So the
recipe substitutes the old bn output onto a fresh id and claims
`(old, fresh, Equivalent)` — the first rewrite here to claim anything but
`Identical`. §8's propagation then carries `Equivalent` to everything downstream,
which the test pins on the relu below the fold: leaving that implicitly
`Identical` would have a verifier assert bit-equality on the model's own output.

**One arithmetic path, not eight.** BN weight, BN bias and conv bias are
independently optional. Rather than branch the emitted graph eight ways, an
absent operand becomes an explicit scalar constant holding its identity value
(1, 0, 0). That costs nothing: `eps` is a float *parameter* rather than an edge,
so the pass has to bind a literal payload regardless, and `Fold_const` removes
all of them again. The eight combinations are then the same code, which is why
the numeric test runs all eight rather than a representative few.

> **Found while implementing: two framework gaps, both exposed by being the first
> value-changing pass.**
>
> `Recipe.replace` decided whether to invent an `Identical` self-claim by asking
> whether the substitution's target was already a claim *source*. Every earlier
> constructor substitutes a fresh output *onto* an existing id, so that was
> enough. A value-changing rewrite substitutes the other way — old output onto a
> fresh id — and the invented self-claim then named an edge absent from the
> source graph, which `Graph_map.validate` rightly rejects. The test is now
> "does the recipe already speak about this edge", on either side of a claim.
>
> `apply` never validated a recipe-bound payload against the signature the recipe
> declares; only `origin` checked its own, and only the shape at that. Fine while
> `Fold_const` was the only producer, since its values come from `Eval_direct` and
> are right by construction — but a pass that computes a parameter itself can bind
> data whose shape or format contradicts its edge, and the mismatch would surface
> only at evaluation. Both paths now check shape *and* format.

### 12d. Sinking a permute through an elementwise op

Status: **implemented** — `lib/native/transform/passes/sink_permute.ml`, tests
in `test/native/permute_passes_test.ml`.

§12a's permute passes only cancel *adjacent* permutes. The relayout lowering
also produces runs like `permute → relu → permute` and `permute a, permute b →
add → relu → permute`, where an elementwise op sits between the two halves of
what would otherwise cancel — most visibly at every residual add, since both
branches get relaid-out on the way into it. Neither `Chain_permute` nor
`Trim_permute` ever sees these as a run.

An elementwise op reads and writes each output slot independently of every
other slot, with no axis-specific semantics, so it commutes exactly — bit for
bit, not just numerically — with a permutation applied uniformly to every
operand:

```
op(permute_p(a), permute_p(b), ...) = permute_p(op(a, b, ...))
```

For a unary op (`Relu`, `Sqrt`) this is immediate: same shape in and out, one
slot at a time. It holds for the broadcasting binary ops too (`Add`, `Sub`,
`Mul`, `Div`): broadcast comparison is per axis, and `p` maps corresponding
axes of every operand identically, so the broadcast pattern permutes right
along with the data. It does **not** hold for `Mean` (reduces over *named*
axes — permuting the input first would reduce over the wrong ones unless the
reduced axes were also carried through `p`, which this pass does not attempt;
§12f transports them instead, for the `keepdim=true` case), or for anything
else with axis-specific semantics: `Conv2d`/`Pool`/`Bmm`/
`Linear`/`Batch_norm`/`Rms_norm`/`Reshape`/`Permute` itself. So `Sink_permute`
matches an explicit allowlist — `{Add, Div, Mul, Relu, Sqrt, Sub}` — rather
than every op `Graph_ir.operands` can see.

**Matching.** The anchor is one of the six ops. Every one of its operands
(`Graph_ir.operands`, so this covers unary and n-ary arity uniformly) must be
`interior` (single consumer, not a graph output — deleting its producer must
not strand anyone else) and produced by a `Permute`, and every one of those
permutes must carry the *same* perm — compared via `Permute.Permute.lookup`
over `Axis.all` rather than list equality, since two perms built by different
call sites need not enumerate their axis pairs in the same order.

**Building.** The op is rebuilt reading each permute's own input instead of
the permuted edge (`Graph_ir.map_operands`), landing on a fresh id; a new
`Permute` with the common perm, reading that fresh id, takes over the
anchor's original id — same tensor, so `claims:[(out, out, Identical)]`,
`Chain_permute`'s self-claim pattern. The pre-permute shape is the anchor's
own shape with `perm` inverted against it via `Vec6.copy`, mirroring
`Permute.Permute.output_shape`'s fold without its validation (the perm is
already known valid, coming from a real node in a well-formed graph).

**One sweep moves a permute past one op.** A permute trapped behind a whole
run of elementwise ops (`permute → add → relu → permute` at a residual add)
needs the SAME pass to fire again after the first rewrite exposes the next
match — sinking through `add` redefines what `relu` reads as a fresh
`Permute` node, which is then itself sinkable. `Pass.fixpoint Sink_permute.pass`
handles this the same way `Trim_permute`'s own fixpoint handles a longer
permute run: each sweep re-scans the current graph, so a chain of *N*
elementwise ops needs *N* sweeps, subject to `Pass.fixpoint`'s iteration bound
(default `max_iters = 16`). The pass creates cancellation opportunities but
never cancels anything itself, so the pipeline (`bin/native_graph.ml`) runs
`Chain_permute`/`Trim_permute` a second time after it.

Not every permute-flanked elementwise op is reachable this way: resnet18's
identity-skip residual adds mix one permuted branch with one branch that was
never permuted to begin with (the skip connection already matches the add's
layout at the PT2 level), so the "every operand shares one perm" guard
correctly declines there — there is no uniform `p` to factor out. §12e is the
follow-up pass that handles exactly this mixed case, by reusing an
alternate-layout edge the graph already computes rather than declining.

### 12e. Reusing an existing layout, and bypassing individual inverse consumers

Status: **implemented** — `lib/native/transform/passes/reuse_permute.ml` and
`bypass_permute.ml`, tests in `test/native/permute_passes_test.ml`. Design record:
`.ai/native_layout_reuse_plan.md`.

`Sink_permute` declines a mixed elementwise op outright: `add(P(a), b)` has no
uniform `p` to factor out. But `b` is not necessarily relayout-free — the
relayout lowering that produces `P(a)` in the first place also, at every op
boundary, tends to leave an existing `Permute(Q)` consumer on `b` for some
*other* native consumer, where `Q = P⁻¹`. If so,

```
add(P(a), b) = P(add(a, Q(b)))
```

by the same per-slot commutation argument as §12d, and `Q(b)` is an edge the
graph **already computes** — reusing it costs nothing, where inserting a fresh
`Permute(Q)` for `b` would only move the boundary, never remove one.
`Reuse_permute` is the pass that recognises this: scope is the same
broadcasting binary allowlist as `Sink_permute`'s binary half (`Add, Div, Mul,
Sub`; a unary op cannot have a mixed operand layout at all).

**Two independent perm-algebra questions, one helper each.** `Sink_permute`
asks whether two perms are the *same* (`perm_equal`, an association-list-order-
independent comparison already local to that file). `Reuse_permute` and
`Bypass_permute` both ask whether two perms are *inverse*, which is genuinely
permutation algebra rather than pass-specific logic, so it lives on the op:
`Permute.Permute.are_inverse ~before ~after = is_identity (compose ~before
~after)`.

**Resolving every operand into the candidate's domain.** For a candidate `P`
— tried once per operand actually produced by a `Permute`, in operand order,
first complete resolution wins — each operand resolves one of two ways:

- **Unwrap.** Produced by `Permute(P)` and `interior` (single consumer, not a
  graph output): claim and remove the producer, read its input directly.
- **Reuse.** Some other consumer of the operand is `Permute(Q)` with `Q = P⁻¹`
  and a format/quantization compatible with the target: read that consumer's
  output directly. Its node is `claim_shared`ed (§11) — reserved against a
  DIFFERENT match unwrap-and-removing that very node in the same sweep, but
  never exclusive against ANOTHER match's own `claim_shared` of it, so
  independent matches all reading one preserved `Permute(Q)` can all be
  accepted together rather than one per sweep. If the first inverse consumer
  found is not format/quantization-compatible, later ones are tried in turn
  (graph order, since `Graph_view.uses` already returns consumers
  execution-ordered) rather than declining the whole operand outright.

At least one of each is required. All-unwrap is `Sink_permute`'s job, kept
disjoint on purpose; all-reuse would rebuild the op without removing or moving
any boundary, which is never worth doing speculatively — the pass never
inserts a new relayout, it only reuses one the graph already pays for.

> **Found in review: an unreserved reused node is a same-sweep conflict
> waiting to happen.** Two disjoint matches — `add(P(a), b)` reusing `Q(b)`,
> and `add(Q(b), c)` elsewhere unwrapping (and removing) that very `Q` node,
> when `P(c)` also exists — can both be accepted by one `Pattern.scan` sweep,
> since nothing marked `Q` as touched. The merged recipe then removes `Q`
> while the first match's insertion still reads its output, and
> `Rewrite.apply` fails on a dangling reference. Reserving the reused node —
> first as an ordinary `claim`, later refined to `claim_shared` below — is
> what makes `scan` catch this instead: the two matches now both touch `Q`
> and the later one is dropped for this sweep, retried — soundly, against
> whatever the graph looks like — on the next.
>
> **Found in review: exclusive reservation over-serializes safe read/read
> sharing.** Plain `claim` on the reused node fixed the conflict above, but it
> also made every OTHER match reusing the SAME `Q` mutually exclusive, even
> though none of them removes it. Sixteen independent `add(P(x_i), b)` nodes
> all reusing one `Q(b)` therefore resolved one per `Pattern.scan` sweep —
> sixteen sweeps needed, which exactly exhausted `Pass.fixpoint`'s default
> `max_iters = 16` fuel with `` `Not_converged `` on a graph that was never
> actually stuck; it just needed more iterations than the bound allowed for
> what should have been one sweep. This is what `claim_shared` (§11) and
> `scan`'s read/write conflict rule are for: several matches that only
> `claim_shared` the same node no longer conflict with EACH OTHER, only with
> a `claim` on it, so the sixteen-way fixture now resolves in a single,
> non-`fixpoint`ed sweep (`test/native/permute_passes_test.ml`,
> `reuse_permute_wide_fanout`).

> **Found in review: `claim` alone does not stop ONE match from conflicting
> with itself.** `claim` is idempotent, so it only rules out conflicts BETWEEN
> separate matches — not a single match resolving two of its OWN operands
> through the same node in opposite roles. `add(swap_hw(b), b)` with a square
> (`H = W`) shape is the reproduction: `swap_hw` is its own inverse, so the one
> `Permute` node in the graph is simultaneously the unwrap target for operand 1
> (it is that operand's own producer) and the reuse source for operand 2 (it
> is also its own inverse consumer). Resolving both independently accepted
> both roles for the same node, and the rebuilt op ended up reading an output
> whose producer the SAME recipe had just removed — `Rewrite.apply` failing
> with "operand t1 has no definition". Every `Resolution.t` now carries the
> node id regardless of role, and `resolve_operands` threads a `Committed.t`
> (`removed`/`kept`, its own module per the repository's record convention)
> through the operands IN ORDER, rejecting a candidate whose node is already
> committed the OTHER way. This is entirely local to `Reuse_permute` — a
> WITHIN-match concern, orthogonal to §11's `claim`/`claim_shared`, which is
> about conflicts BETWEEN separate matches. Rejection has to happen
> inside the SAME `choice` that offers each operand's remaining alternatives —
> written as "pick a candidate, THEN resolve everything after it," combined
> with `choice` at EVERY level — so that a conflict discovered several
> operands later can fall back to an earlier operand's next candidate, not
> merely fail the match outright. (For `Reuse_permute`'s binary-only scope
> this degenerate case has no other resolution to fall back to, so the
> correct outcome is simply no match — but the machinery generalises.)

**The rewrite mirrors `Sink_permute`'s shape exactly:** rebuild the op over
the resolved operands on a fresh id, insert `Permute(P)` over that id taking
over the anchor's original output id (`claims:[(out, out, Identical)]`, the
`Chain_permute` self-claim pattern), remove the anchor and every UNWRAPPED
producer — reused nodes are never touched. Operand order in the rebuilt op is
preserved exactly (`Graph_ir.map_operands`), which matters for `Sub`/`Div`:
swapping which operand unwrapped and which reused would silently negate or
invert the result rather than fail to type-check, so the fixtures pin both
orders and a numeric test on `Sub` specifically.

**`Bypass_permute` complements `Trim_permute` the way `Sink_permute` motivated
it to exist.** `Trim_permute` cancels a *complete* identity run and requires
every intermediate edge to be `interior`, so a shared producer or a fan-out of
several inverse consumers is invisible to it. After `Reuse_permute` and
`Sink_permute` move a `P` downstream, it often lands right in front of an
existing `Permute(Q)` that is no longer the only thing reading its input:

```
x -> P -> y -> Q -> z         becomes         x ----------------> z
       \-> (other uses of y)                   \-> P -> y (if y still needed)
```

`Bypass_permute` anchors on `y` (a `Permute(P)` output) and, among its
consumers, FILTERS for `Permute(Q)` with `Q = P⁻¹` AND a format/quantization
compatible with `P`'s input `x` — filters, not guards: one incompatible
inverse consumer must not sink a match where other consumers are perfectly
bypassable, it simply counts against `all_covered` below like any other
non-inverse consumer, keeping `P` live for its sake. The survivors are
`claim`ed atomically, in one match, so greedy scanning cannot split the
fan-out across overlapping recipes — and requires at least one. Every
collected `z` is tied straight to `x` (`Recipe.trim`, so `{z} -> {x}
Identical`, `y` itself simply vanishes and gets picked up as an implicit
deletion by `Rewrite.apply`'s own bookkeeping — no explicit claim needed for
it). `P` is removed too, but only when **every** consumer of `y` was one of
the collected (compatible) inverse permutes and `y` is not a graph output;
otherwise `P` stays, servicing whichever consumer is not being bypassed.

This is the one place matching needed a primitive `Pattern` didn't have:
"is this edge a graph output", *without* also requiring exactly one consumer
the way `interior` does (`Bypass_permute` explicitly wants fan-out). Added as
`Pattern.is_graph_output`, decomposed out of `interior`'s existing check.

**Why an outer fixed point over all five passes, not five separate rounds.**
`Reuse_permute` inserts a `P` right after the rebuilt op, taking over the
anchor's *old* id — so downstream, nothing looks different by id, but there
genuinely is a new `Permute` node sitting between the rebuilt (native-layout)
op and whatever used to read the old id. `Sink_permute` has to move that `P`
through the elementwise chain below it before `Chain_permute`, `Trim_permute`
or `Bypass_permute` can do anything with it. And bypassing one block's
now-fan-out permute can turn a previously-shared edge `interior`, exposing the
next block's sink/trim opportunity — a single non-interleaved pass over each
pass once is not enough for a chain of residual blocks, exactly as it was not
enough for `Sink_permute` alone (§12d). The pipeline (`bin/native_graph.ml`)
therefore wraps the whole relayout family in one outer `Pass.fixpoint`:

```ocaml
let relayout_pass =
  Pass.fixpoint
    (Pass.sequence ~name:"relayout"
       [
         Pass.fixpoint Chain_permute.pass;
         Pass.fixpoint Trim_permute.pass;
         Pass.fixpoint Sink_permute_mean.pass;
         Pass.fixpoint Sink_permute.pass;
         Pass.fixpoint Reuse_permute.pass;
         Pass.fixpoint Sink_permute.pass;
         Pass.fixpoint Bypass_permute.pass;
         Pass.fixpoint Chain_permute.pass;
         Pass.fixpoint Trim_permute.pass;
       ])
```

`Pass.sequence` is the small combinator this needed: a fixed list of passes as
one named pass, delegating to `Pass.run_all` and composing the resulting step,
so the whole group can be handed to `fixpoint` as a unit. (`Sink_permute_mean`
— §12f — was added after this pipeline shape was established; it slots in
right after the initial chain/trim cleanup, since transporting a permutation
through `Mean` is a local rewrite with no fan-out interaction of its own, and
its output feeds the same sink/reuse/bypass machinery either way.)

**On ResNet-18** (`test/native_transform_cram.t`), the structural pipeline
goes from 174 nodes to **91** (down from 112 with `Sink_permute` alone, 93
before `Sink_permute_mean`) — the identity-skip residual blocks' `Add`/`Relu`
now run in native layout without a speculative relayout, and `Bypass_permute`
removes the duplicate downstream rotations that leaves behind; the two
remaining nodes come from `Sink_permute_mean` collapsing the global-average-pool
`Mean`'s flanking permute pair (§12f). `--verify` still reports `max_abs=0`:
reusing an existing edge, bypassing a redundant consumer, and transporting a
permutation through `Mean` are all `Identical` rewrites, same as
`Sink_permute`. The folded pipeline (`--fold`) goes from 71 to **50** nodes,
with the same `~1.9e-06` batch-norm-fold residual as before — the relayout
family's own claims are all `Identical`, so the tolerance is unchanged by this
stage. Not every permute disappears: initial input relayouts, genuine
pool/linear layout boundaries, and any operand pair that never had a reusable
alternate-layout edge to begin with are expected to remain.

### 12f. Transporting a permutation through a keepdim=true Mean

Status: **implemented** — `lib/native/transform/passes/sink_permute_mean.ml`,
`Reduce.Mean.map_dims`, tests in `test/native/permute_passes_test.ml`.

`Mean` reduces over *named* axes, so `Sink_permute` (§12d) deliberately excludes
it: naively sinking a permute through `Mean` and reducing the same axis names
afterward would reduce the wrong dimensions. But the reduction's dimension
list is itself expressible in terms of either layout, so it can be
*transported* through the permutation instead of left behind. For `P` an
output-to-input permutation and `D` the ordered list of reduction axes:

```
Mean_D(P(x)) = P(Mean_{P(D)}(x))          (keepdim=true only)
```

`P(D)` is `List.map (Permute.Permute.lookup P) D`. This holds because
`keepdim=true` collapses each reduced axis to extent 1 *in place* — the
induced output permutation is exactly `P` again, with no separate survivor
remapping the way `keepdim=false` (`Mean.kept_map`) would need. That remapping
is out of scope for this pass; so are multi-operand ops whose axis parameters
interact with weights, masks, or windows (`Batch_norm`, `Rms_norm`, `Conv2d`,
`Pool`, `Linear`, `Bmm`).

**Parameter transport.** `Reduce.Mean.map_dims : (Axis.t -> Axis.t) -> params
-> params` maps every entry of `params.dims` through a caller-supplied
function, in list order, keeping `keepdim` untouched. `Sink_permute_mean`
supplies `Permute.Permute.lookup P`. Keeping this on `Reduce.Mean` rather than
the pass mirrors `Sink_permute`'s own split: permutation algebra stays on
`Permute.Permute`, op-specific parameter semantics stay on the op.

**Matching.** The anchor is a `Mean` node with `params.keepdim = true`. Its
input must be `interior` (single consumer, not a graph output) and produced by
a `Permute(P)`. The permute's own input and output signatures must agree in
format and quantization (`precision_equal`, `Bypass_permute`'s helper: a
`packed_fmt` has no usable structural order, so it compares by format name).
Finally, the shapes must round-trip exactly: `Mean.output_shape` on the
permute's pre-image with the mapped dims, then `Permute.output_shape` with `P`
again, must equal the untouched `Mean` node's own output shape — this is the
identity's own claim, checked rather than assumed. The `Mean` output itself
carries no interior or graph-output requirement: the rewrite preserves its
tensor id, so any downstream consumer or graph output stays valid regardless.

**Building.** Mirrors `Sink_permute`'s shape, with the roles reversed: a fresh
`Mean` node reads the permute's own input directly (`Graph_ir`'s `x`, not the
permuted edge) and lands on a fresh id; a new `Permute(P)` node, reading that
fresh id, takes over the *original* `Mean` node's output id — same tensor, so
`claims:[(out, out, Identical)]`, the same self-claim `Chain_permute` and
`Sink_permute` use. The fresh tensor's format and quantization are copied from
the original `Mean` output's own signature (permute preserves them, so the
pre-permute tensor and the post-permute one agree). The old input permutation
and the old `Mean` node are both removed; the rebuilt `Mean` maps from the old
`Mean` node, and the new wrapping permute maps from the old input permutation
— the same node-provenance shape `Sink_permute` uses for its rebuilt op and
inserted permute.

One match transports one permutation across one `Mean`; there is no reverse
direction, so under `Pass.fixpoint` the rewrite makes monotonic progress. The
pass creates an adjacency (the new wrapping `Permute(P)` next to whatever the
old `Mean` output already fed, or next to a downstream inverse) without
cancelling anything itself — `Chain_permute`/`Trim_permute`, which run again
later in `relayout_pass`, do that. On ResNet-18's global-average-pool `Mean`
(the motivating case: a permute, a `keepdim=true` `Mean` over two axes, and its
exact inverse permute), the three passes together collapse the whole
three-node run to one `Mean` reducing the correctly-mapped dimensions, with no
permute left around it at all — see the `Sink_permute_mean.pass` result quoted
in `test/native_transform_cram.t` under `torch.ops.aten.mean.dim`.

## 13. Tests

`test/native/` is `ppx_expect` inline tests in the `native_test` library. New tests
follow the house idiom (`graph_test.ml`): a local `type error = [ … ]` row,
`pp_error`, `let pp_result pp_ok = Core.Pretty.core_result ~ok:pp_ok
~error:pp_error`, `lift_*` adapters, one `Core.Syntax` chain, one `Format.printf`,
one `[%expect]`.

`test/native/graph_fixtures.ml` is the shared graph library this layer needs —
there is none today, and `s`/`s1c`/`conv_axis` are copy-pasted across five test
files. New tests use the fixtures; existing files are left alone. Fixtures:
`chain` (conv→bn→relu), `residual`, `diamond`, `const_permute` (the motivating
fold), `const_arith`, `multi_output` (`Max_pool2d_with_indices` + `Discard`),
`grouped` (two `Graph_builder.group` blocks with a rewrite spanning both), and one
per permute/reshape shape the passes have to tell apart — `permute_noop`,
`permute_sequence` (cancels, neither node a no-op alone),
`permute_identity_chain`, `permute_pair` (fuses, does not cancel),
`permute_partial_cancel` (only a prefix cancels), `permute_shared` (a second
consumer of the intermediate), `reshape_relabel`, `reshape_flatten`, and
(§12d) `sink_permute_unary`/`sink_permute_binary` (one/two sweeps to make the
flanking permutes adjacent), `sink_permute_fuse` (the flanking permutes are
NOT inverses, so the sweep it exposes feeds `Chain_permute`'s fusion rather
than `Trim_permute`'s cancellation — the end-to-end case sinking exists for
but that the unary/binary fixtures, both landing on an identity, don't cover),
`sink_permute_broadcast` (differently-shaped operands), `sink_permute_mismatch`
(different perms on each operand), `sink_permute_shared`/`sink_permute_output`
(the two `interior` failure modes), `sink_permute_allowlist` (one of each
accepted op); and (§12e) `reuse_permute_basic` (the motivating mixed-add
shape), `reuse_permute_missing_alternate`/`reuse_permute_wrong_alternate` (no
reusable edge; a reusable edge that is not the inverse), `reuse_permute_sub_order`/
`reuse_permute_div_order` (non-commutative ops, unwrap and reuse on opposite
sides), `reuse_permute_backtrack_candidate` (the first inverse consumer found
is format-incompatible, so the pass must fall through to the second),
`reuse_permute_competing_matches` (two disjoint matches over one producer —
one reuses it, the other would unwrap-and-remove it; only one is accepted per
sweep), `reuse_permute_wide_fanout` (sixteen independent matches all reusing
the SAME producer — none removes it, so all sixteen resolve in one sweep,
not one per sweep as an all-exclusive claim would force), `reuse_permute_self_inverse`
(a self-inverse permutation resolving both operands through the SAME node
in opposite roles — correctly no match, the reproduction for the "operand
has no definition" corruption above), `bypass_permute_pair` (the minimal
one-`P`-one-`Q` case),
`bypass_permute_fanout` (one `P`, several inverse `Q` consumers, none other —
`P` and every `Q` go), `bypass_permute_mixed_compatibility` (a compatible and
an incompatible inverse consumer together — only the compatible one is
bypassed, keeping `P` alive for the other), `bypass_permute_shared`/
`bypass_permute_output` (the two reasons `P` has to stay: another consumer,
or being a graph output itself), and `bypass_unlocks_sink` — `y`'s producer `P`
is shared between an inverse
`Q` fed straight to `Discard` (so `Sink_permute` can never reach it directly;
that pass only ever anchors on an elementwise op's own output) and a plain
`Relu` (`P` is not yet `interior`, so `Sink_permute` cannot move it through
the `Relu` either). Bypassing `Q` is `Bypass_permute`'s job alone here, and it
runs after both `Sink_permute` stages in the sequence, so one application
leaves the `Relu` still reading the permuted edge; only wrapping the whole
sequence in a second `Pass.fixpoint` sinks it — the fixture pinning §12e's
own "why an outer fixed point" claim the same way `sink_permute_binary`
pinned it for `Sink_permute` alone. (§12f) `sink_permute_mean_basic` (transport
with no downstream inverse — the rewrite leaves `Mean(mapped dims) ->
Permute(P)` rather than cancelling anything), `sink_permute_mean_cycle` (the
motivating permute/`Mean`/inverse-permute run, collapsing to one `Mean` under
`Sink_permute_mean` + `Chain_permute` + `Trim_permute`), `sink_permute_mean_shared`
(the input permutation has a second consumer, so it is not `interior`),
`sink_permute_mean_not_keepdim` (`keepdim=false` is out of scope for this
pass and is declined outright).
A fixture is named after the *shape* it exercises, never after the pass that
consumes it, so one graph can pin several passes.

Each test prints the source graph, then whichever applies: the matched portion
(`Region.extract` → `Graph_ir.pp`, plus `Region.pp`), the recipe
(`Rewrite.pp_recipe`), the result graph, and the mapping (`Graph_map.pp`, printing
value clusters, node clusters and provenance separately). Goldens are id-based
since `Tensor_sig` has no names; a test wanting labels supplies a
`Graph_ir.Printer`.

Beyond per-feature cases, three families are load-bearing:

- **Algebraic laws** for the mapping: `compose identity`, associativity *including
  chains containing creations and deletions* (where §3's guard applies, under both
  association orders), `invert ∘ invert`, and `validate` rejecting forged maps.
  Landed in `graph_map_test.ml`, along with the two cases the design turns on:
  trimming's many-to-one cluster, and delete/create/repack staying two clusters.
- **Cumulative constants** across a `fixpoint` fold over a multi-node constant
  sub-DAG.
- **Numeric equivalence** — `Eval_direct` over the source and destination graphs
  on the same input — wherever a pass's legality rests on an argument rather than
  a syntactic identity. That is `reshape_to_permute` (row-major offsets), bn
  folding (the latter across all eight operand combinations — conv bias × BN
  weight × BN bias, independently optional), and `sink_permute` (the
  permutation/broadcasting commutation argument in §12d, checked on
  `sink_permute_broadcast`, whose two operands have DIFFERENT pre-permute
  shapes — the case where the argument has to hold for broadcasting, not just
  equal shapes), and `reuse_permute` (§12e), checked on `reuse_permute_sub_order`
  and `reuse_permute_div_order` specifically — `Sub` and `Div` are not
  commutative, so a rebuilt op with its operands transposed would silently
  negate or invert the result rather than fail to type-check.

## 14. Non-goals

- The numerical/symbolic verifier itself. This doc defines the mapping it will
  consume and nothing more.
- Re-nesting or interval-preserving groups (§7 limitation).
- An op-specific `Approximate` error-transfer policy (§8 limitation).
- Precision and quantization passes. Nothing here emits `Approximate`; the label
  and its tests exist so the machinery is not dead when those land. Two gaps they
  will hit: `Graph_builder`'s op-output edges are hardcoded F32, so `fragment` will
  need a target format, and `op` has no cast constructor, so mixed-precision
  boundaries will need `Convert`.
- Batching disjoint matches into one sweep. `Rewrite.merge` exists for it; the
  driver applies one recipe per sweep and rescans, which is clearly correct.
