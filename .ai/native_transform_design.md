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

Status: **in progress** — stages 1 to 4 have landed.
Each section below carries its own status marker, flipped by the commit that
implements it; `## 12. Staging` tracks the whole sequence.

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

Status: **implemented** — `rewrite.ml`; tests in `test/native/rewrite_test.ml`.

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

Status: **design only**.

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

## 10. PT2 provenance — recovered, never carried

Status: **design only**.

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
pass-computed data lives in the final `Rewrite.constants`. `native_interp` consults
the state's payload map first and falls back to the archive through the lens,
replacing today's "load every constant via `captured_targets`", which cannot see a
folded constant at all.

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

A state+error monad over `{ view; claimed }`. **No cursor** — every primitive names
the edge it operates on, which is the honest formulation for a DAG and reads better
than a hidden position. Matching walks *up* operand edges, which is deterministic
because operands are typed fields; `uses` covers fan-out.

```ocaml
val def   : Tensor_id.t -> (op -> 'a option) -> ('a * node) t  (* producer; claims it *)
val peek  : Tensor_id.t -> (op -> 'a option) -> ('a * node) t
val uses  : Tensor_id.t -> node list t
val interior : Tensor_id.t -> unit t     (* exactly one use and not a graph output *)
val constant : Tensor_id.t -> unit t
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
- **Overlap.** Greedy: first match in anchor order wins; a later match intersecting
  an accepted region is discarded, not re-attempted.
- **Rollback.** State is immutable, so a failed `choice`/`optional` branch discards
  its state wholesale, including claims. Only `Mismatch` backtracks; `Invalid`
  aborts the scan.
- **Re-claiming.** Claiming an already-claimed node inside the same match is
  idempotent success — that is what makes diamond patterns work.
- **Progress.** `chain` rejects a step whose returned edge is not strictly earlier
  in topo order, so a non-advancing parser cannot loop.

`Region` derives its boundary from the claimed node set — inputs are operands whose
producer is unclaimed, outputs are claimed outputs used outside or in
`Graph.outputs` — and reports `convex` rather than enforcing it: non-convexity
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

## 12. Staging

Each stage is one commit carrying code, its expect tests, and this doc's status
delta. Stages 1–5 are the framework, 6–9 the transformations, 10–11 integration.

| # | Commit | Lands | Status |
|---|---|---|---|
| 0 | `docs: design the native graph transformation framework` | this doc; `native_graph_design.md` §7 → pointer | done |
| 1 | `native: add id_supply and graph mapping` | `Tensor_id.Set`/`Node_id.Set`, `Group_id` ordering, `Id_supply`, `Cluster_relation`, `Correspondence`, `Node_map`, `Provenance`, `Graph_map` | done |
| 2 | `native: add graph_view and region` | validation, the index, `common_group`, topo sort, `Region`, `test/native/graph_fixtures.ml` | done |
| 3 | `native: add recipe and rewrite` | the state, `Recipe`, `apply`, `output_transfer` (+ its entry in `native_add_op.md`) | done |
| 4 | `native: add graph match combinators` | `Pattern`, `run`, `scan` | done |
| 5 | `native: add pass driver` | `Pass`, `fixpoint`, `run_all` |
| 6 | `native: add permute simplification passes` | `trim_permute`, `chain_permute`, `reshape_to_permute` |
| 7 | `native: add constant folding` | `fold_const` — the motivating permute-of-constant-weight case |
| 8 | `native: add Sub, Div and Sqrt ops` | prerequisite for bn folding |
| 9 | `native: add batch-norm folding pass` | `fold_batch_norm` |
| 10 | `native: add terminal id packing` | `Rewrite.pack` |
| 11 | `native: resolve PT2 provenance through a transformation map` | the lens; `native_interp` payload order |

## 13. Tests

`test/native/` is `ppx_expect` inline tests in the `native_test` library. New tests
follow the house idiom (`graph_test.ml`): a local `type error = [ … ]` row,
`pp_error`, `let pp_result pp_ok = Core.Pretty.core_result ~ok:pp_ok
~error:pp_error`, `lift_*` adapters, one `Core.Syntax` chain, one `Format.printf`,
one `[%expect]`.

`test/native/graph_fixtures.ml` is the shared graph library this layer needs —
there is none today, and `s`/`s1c`/`conv_axis` are copy-pasted across five test
files. New tests use the fixtures; existing files are left alone. Fixtures:
`chain` (conv→bn→relu), `residual`, `diamond`, `permute_noop`, `permute_sequence`,
`reshape_relabel`, `const_permute` (the motivating fold), `const_arith`,
`multi_output` (`Max_pool2d_with_indices` + `Discard`), `grouped` (two
`Graph_builder.group` blocks with a rewrite spanning both).

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
- **Numeric equivalence** for bn folding across all eight operand combinations
  (conv bias × BN weight × BN bias, independently optional).

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
