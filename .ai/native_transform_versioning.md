# Version-indexed ids — making a graph version part of an id's type

Deepens `native_transform_design.md` §3-§5, §7-§8 and `native_transform_verify.md`
§7. Covers `lib/native/transform/brand.ml`, `snapshot.ml`, the indexed layer
inside `cluster_relation.ml`, and `graph_map.ml`'s `create`.

Status: **all five stages implemented** — `Brand`, `Snapshot`, version-indexed ids
throughout the relations, `Graph_map.create` as the only constructor, the
verifier's `Member` as a GADT, and the negative-compile harness
(`test/native/version_safety.t`). Stages 4-5 (`Input_var`, typed recipes) are
designed in §6 and not yet built.

## 1. The problem the phantoms did not solve

`native_transform_design.md` §3 already tags a map `('src, 'dst)`. The tags
order a composition, but the *contents* erase them: `Cluster.t` holds plain
`Id.Set.t`, and `forward`/`backward`/`sources_of` took and returned raw ids. The
`.mli` said so itself: they "cannot tie a map to two PARTICULAR graphs —
`of_clusters` is polymorphic in them — which is why `validate` exists".

The cost is on record three times. `map_verify.ml:316` fixed a source id looked up
in the destination producer map. `symbol-impl-2.md` records a **false proof**: a
map swapping a graph's two inputs was reported structurally proved, output
included, because src and dst share one numeric namespace. And the third was found
by the types themselves, in stage 2 — see §3a.

Goal: an id carries which graph version it belongs to, so those are compile
errors. Raw `Tensor_id.t` stays the serialization and execution identity — the
tag is erased at runtime.

## 2. Why unforgeability has to be structural

`lib/native/dune` is `(wrapped false)`. There are no library-private modules, so
**every constructor is public API** and visibility cannot be part of the
argument. `native_transform_design.md:318-323` drew the conclusion that every
version-bound abstraction must therefore live in `Rewrite`.

The escape is the existential. `Brand.fresh : unit -> packed` is public, yet a
caller cannot *choose* a version: unpacking `Pack b` binds a rigid variable that
unifies with nothing else. `Universe.create` can then take a brand as an ordinary
argument without becoming a forgery — building a universe at some *existing*
snapshot's version would need that snapshot's brand, and nothing hands it out.
Minting your own is harmless, because its tag corresponds to no other value in
the program.

One consequence worth stating: a `'v id` witnesses membership of *some* universe
at `'v`, not of the one a given call was passed — two universes can be minted from
one unpacked brand. So the endpoint checks inside `of_clusters` still have work to
do, and are not vacuous.

> **This reverses two recorded decisions, deliberately.** `native_transform_design.md`
> §5 (`:326-328`) and §11 (`:675-689`) rejected a versioned `Graph_view` because
> `of_graph : graph -> ('v t, error) result` lets the caller pick `'v`.
> `Snapshot.create` returns a `packed` and does not. The companion objection —
> "would force every `Pass` callback to be a rank-2 record" — is already paid:
> `pass.mli:51` is one today.

## 3. Where the tagged layer lives, and why

The indexing lives *inside* `Cluster_relation.Make`, not in a module beside
`Snapshot`. The reason is the implicit-identity rule: `forward` answers an
*unmentioned* source id with itself on the destination side, which is a retag
across versions. Only a body that owns the erasure (`type 'v id = Id.t`) can do
that. Keeping it inside the functor is what lets the retag exist without a
`retag : 'a id -> 'b id` appearing in any signature — where it would be exactly
as forgeable as the raw ids it replaces.

Everything above the functor lifts through `Universe.find`, which is a membership
test: a raw id enters the typed world only by being found in a version that has
it. So the runtime guards that used to precede a raw pairing *become* the lift.
`rewrite.ml`'s identity closure is the clearest case — `Tensor_id.Map.mem dst
old_g.Graph.tensors` is now `Snapshot.edge old_snap dst`, and the answer is the
tagged id it needed anyway.

`Snapshot` binds one `Brand.t` to both id spaces plus the validated view, so a
snapshot's tensors and nodes share a version and a `Graph_map` can be indexed by
one tag rather than one per id space.

## 3a. What the types found: `Provenance.compose`

Typing the endpoints turned a latent bug into a compile error, which is the whole
argument for this work stated once, concretely.

`compose (a : A→B) (b : B→C)` walks each C-edge's middle ids and pulls them back
through the A→B correspondence. For a middle id that is *itself* derived, the code
took its own sources — already A-side ids — and pulled those back a second time:

```ocaml
let via = sources_of a mid in
if Set.is_empty via then pull_back (Set.singleton mid) else pull_back via
```

`pull_back` is `Correspondence.backward ab`, which reads its argument as a middle
id. Under raw ids both sides were `Tensor_id.Set.t` and it compiled; typed, `via`
is an `'a set` where a `'b set` is required, and unification collapsed the two
versions — the signature check in `provenance.mli` is what refused it.

It was silent because `backward` answers an unmentioned id with itself, so a
source id absent from `ab`'s destination side came back unchanged. It breaks when
a source id *also* occurs as a middle destination — a rename and a fold in one
step is enough — and the derivation then names the wrong origin. Fixed to `else
via`; only `mid` is a middle id.

Worth noting what the regression test is. Writing the old form again does not
produce a wrong answer, it produces a **build failure**: the two versions unify
and `provenance.mli` refuses the collapsed signature. The expect test next to it
pins the answer, which no type can state, but the guard is the signature.

## 4. Tagging alone is not enough: the constructor must consume both universes

Found by testing, not by reading. The first negative case *compiled*:

```ocaml
forward m dst_id   (* accepted! *)
```

A relation built by a tag-polymorphic `of_clusters` leaves `'src` and `'dst`
free, and free tags unify with whatever they meet first. This is precisely the
hole `cluster_relation.mli:60` names, and it survives tagging the ids.

It closes only when the constructor consumes both universes:

```ocaml
val of_clusters :
  src:'src Universe.t -> dst:'dst Universe.t ->
  ('src, 'dst) Cluster.t list -> (('src, 'dst) t, Id.t issue) Stdlib.result
```

which also fuses validation with tagging — the universes are the id sets
`validate` needed anyway. `Graph_map.validate` therefore no longer exists as a
separate call a consumer must remember: `Graph_map.t` is abstract and
`create ~src ~dst` is the only way to assemble one.

## 4a. What `create` establishes beyond the endpoints

Two map-level invariants, both from `native_transform_design.md` (§7 step 9 and
§8) and both previously unenforced. They belong on the map rather than in the
verifier because `Pt2_native_graph` is wrong in the same way without them and
never goes near a proof.

**Cluster metadata.** Corresponding shapes agree; an `Identical` cluster's
endpoints agree on format and quantization. Step 9 specified this for `apply` and
nothing implemented it — `check_signatures` compares an id against *itself* across
versions, which is §4's preserved-id rule, not a statement about two ids in one
cluster. Enforcing it found a real contradiction in the suite: a permute's output
is materialized as f32, so `Trim_permute` on a non-f32 input claimed an i32 edge
`Identical` to an f32 one. `verify_test.ml`'s own comment already said that claim
was false; it is now rejected rather than merely unproven.

**Claim closure.** The same propagation §8 uses to *label* a map, re-run to
*reject* one whose labels are not closed. Verified by mutation, and the harm is
concrete: with the check removed, the `add`/`sub` map in `graph_map_test.ml`
constructs, and `Map_verify` reports `t3` — `relu(add(a,b))` against
`relu(sub(a,b))` — **proved (structural) identical**, because both sides ground to
`relu(cell t2)` and t2 carries the same raw number on either side. That last
failure is stage 4's problem, not closure's; closure is what keeps it out of reach
today.

Neither subsumes the other, and neither subsumes stage 4. Closure guards the map
as a data structure and is what stops `pt2_native_graph.ml:170` resolving an
unmentioned id to `Identical` and fetching the wrong bytes — which never goes near
a cell. It also says nothing about a map with no explicit claims at all: an empty
map between structurally unrelated graphs passes closure by construction.

## 5. The negative-compile harness

`test/native/version_safety.t` compiles snippets that must **not** type-check and
asserts on acceptance. This is the type-level form of the rule in `CLAUDE.md`
that a check which has never failed is not evidence.

**Every negative sits next to a control that must still compile.** A harness with
a wrong include path rejects everything and would otherwise "pass" while proving
nothing — which is not hypothetical: it happened twice while building this, once
from an existential escaping through a callback and once from the cram sandbox
pruning the interfaces.

| case | must |
|---|---|
| `forward m d` / `backward m s` | be rejected |
| `forward m s` / `backward m d` | compile |
| src edge resolved in the dst universe | be rejected |
| universe minted under a self-minted brand, used against a snapshot's map | be rejected |
| `raw` off either side | compile |

Three plumbing facts worth keeping:

- **The snippets go to a dune-built toplevel**, `(toplevel (name version_probe)
  (libraries native))`. The alternative — `ocamlfind ocamlc -I
  ../../lib/native/.native.objs/byte` — reaches into dune's internals and needs
  every interface globbed into the cram sandbox, because the sandbox is a
  **pruned** view of the build tree and `wrapped false` means that is the whole
  library. The toplevel is self-contained and, being an ordinary dep, is what
  makes editing an `.mli` rerun the test.
- **Acceptance is read from `val check :` in the output, not from an exit code.**
  A toplevel reports a type error and carries on, exiting 0 either way.
- The outcome is asserted, not the diagnostic. Messages name types like
  `Cluster_relation.Make(Correspondence.Id)(Correspondence.Label).t` and
  existentials like `$Pack_'v1`, whose spelling is a compiler detail.

Verified non-vacuous by mutation: relaxing `forward` to `'a id -> 'dst set` flips
exactly one line to `COMPILES` and leaves the other six untouched.

## 5a. Where the evidence is mutation, not a test

Stage 3's protection is entirely inside `map_verify.ml`: `Member`'s constructors
are not public — only `Member.Erased` is — so the cram harness cannot reach them,
and there is nothing for a runtime test to observe either, since the printed form
is deliberately unchanged. The evidence is that the two crossings do not compile,
checked by making them:

| edit | result |
|---|---|
| `resolve` pairing `Dst` with `sides.src` | `Type d is not compatible with type s` |
| `Group_path.of_cluster` reading `c.src` | rejected — the constraint `'src = 'dst` propagates out to `Map_verify.step`'s call site |

The second is worth noting for its shape: the error surfaces at the caller, not at
the fallback, because forcing the two versions equal only becomes contradictory
where a genuinely two-version map is supplied. It still does not build, which is
the property.

## 6. Staging

| stage | content | status |
|---|---|---|
| 1 | `Brand`, `Snapshot`, `Cluster_relation.Tagged`, harness — additive | done |
| 2 | raw API deleted, indexing promoted; typed endpoints through `Rewrite`, `Map_verify`, the PT2 lens; `Graph_map` abstract with `create ~src ~dst`, claim closure, cluster metadata validation | done |
| 3 | `Map_verify.Member` as a GADT, tag erased at the report boundary | done |
| 4 | `Input_var`, and `Ground_expr.Cell.origin` with `Shared` earned structurally AND by induction | done, then SUPERSEDED — §6a |
| 5 | typed recipes: `'v source`, `'v target`, `'v fresh` | done |

Stage 2 could not be subdivided: `map_verify.ml` and `pt2_native_graph.ml`
consumed `Graph_map.validate`, `clusters_over` and `Correspondence.Cluster.t`
directly, so changing the relations without them in the same commit would have
left the tree red or forced temporary raw escape hatches.

Three things stage 2 needed that the plan did not anticipate, all small and all
additive to the functor:

- **`normalise`**, the merge step of `of_clusters` on its own. `Rewrite.apply`
  decides whether a definition really changed by asking which cluster an id is in,
  and it must do that *before* it knows the created and deleted sets — so it cannot
  go through the validating constructor. Tag-polymorphic and safe to be: it yields
  clusters, not a relation, so nothing reachable from it answers a question about
  an id's version.
- **`Map`**, an id-keyed map. `Provenance` keys by destination edge and stores
  source edges, so its key and payload sit at different versions.
- **`Cluster.Erased`**, the cluster with its indices dropped. `Map_verify.Report`
  escapes into `Pass.outcome` and the interpreter's result record; parameterising
  that hierarchy buys nothing a reader can use. This is stage 3's "reports erase
  the tag" rule, arriving one stage early because `Cluster.t` gained parameters
  here.

**Testing an existential.** `Snapshot.create` and `Brand.fresh` return packed
values, and an existential cannot be bound by a toplevel `let` — which would force
every test through a rank-2 callback. `test/native/version_fixture.ml` carries the
tag out in a first-class module instead (`Brand.Pack (type a) (b : a Brand.t)`),
so a test writes `module A = (val Version_fixture.of_graph g)` and then names
`A.v`. Nothing is weakened: `ids` mints its own brand, and `of_graph` re-exposes
the tag `Snapshot.create` already chose.

## 6a. Stage 4: what a raw id may mean, in two layers

> **Superseded.** The two layers below were both removed. The static one
> validated origin-to-endpoint equality rather than the local obligation the
> verifier is supposed to check, and the dynamic one needed the cluster DAG to be
> acyclic, which two graphs quotiented by a correspondence need not be. What
> replaced them is cluster MEMBERSHIP alone: `Cluster_var`, `Boundary_index` and
> `Ground_expr.project`, with `Input_var` generalised to `Cluster_var` and
> `Cell_origin` deleted. See `.ai/native_transform_local_verify_plan.md` — §4 for
> the diagnosis, §§5-6 for the replacement, §13 for the one-sided-variable rule
> that keeps `reuse_permute` provable. The section is kept because the false
> proof it diagnoses is real and the tests that pin it are still in the suite;
> only the cure changed.

The verifier's fast path stopped expansion at any cell whose raw id matched on
both sides. That is not an induction, it is an **unchecked fixed-point
assumption**: every same-numbered edge treated as one variable, all at once, with
no argument for well-foundedness and no regard for whether the cluster was
proved. It is the assumption `map_verify.ml`'s sigma note declines to make when
it refuses to rename both sides of an internal cluster to one representative —
made silently, by numbering.

**What it cost.** An empty map between `relu(add a b)` and `relu(sub a b)` passes
`Graph_map.create` (closure has no explicit claim to propagate) and the output
cluster comes back *proved (structural) identical*. Worse, and this is the part
that makes it a shipping defect rather than a curiosity: when the differing
upstream edge is merely BUDGET-LIMITED rather than refuted, nothing in the report
is refuted, and `Policy.Reject_refuted` — the release bar — accepts the whole
thing. Measured on `mean(a+b)` versus `mean(a−b)` at 8192 coords:

```
{t2} -> {t2} identical: unproved: too large (8192 coords)
{t3} -> {t3} identical: proved (structural) [exhaustive]
refuted=false   Reject_refuted accepts=true
```

**The replacement, and why it needs two layers.**

- **Static** (`Cell_origin`): compare the two graphs' DEFINITIONS — op constructor
  and non-tensor parameters, operand ORIGINS pairwise, output ordinal, signature.
  Coordinate-free, so it holds at any budget, and it is what keeps an untouched
  prefix short-circuiting. On its own it is too narrow: any edge downstream of any
  edit is side-tagged, and in a real pipeline every activation is downstream of
  some edit, so the frontier runs to the graph inputs.
- **Dynamic**: an edge also becomes shared once its OWN cluster has been proved.
  Clusters are checked in destination-topological order, so a proof leans only on
  conclusions established strictly earlier — a real induction. This stops the
  static rule's cascade one step past each edit. Cycles need no special handling:
  "already proved" is by construction a strictly-earlier property, so a cluster
  whose source-side upstream lands later in destination order simply finds it
  unshared and expands. Conservative, automatic, well-founded.

**Only `Proved Structural` with `Exhaustive` coverage licenses sharing**, and each
half of that is load-bearing. Anything weaker than proved is exactly the case that
yields the false proof above. `Constants` is a statement about the payloads this
model carries, while the driver's first attempt leaves them free. And a structural
proof is PER-COORDINATE: agreeing at eight sampled coordinates says nothing about
the rest, and `map_verify.mli` already refuses to count a sampled verdict as
proved — licensing further proofs from one would compound precisely that
overstatement.

**What that means in practice, measured on ResNet-18.** Every `Effort` preset
samples (`Quick` 4, `Standard` 8, `Thorough` 32) while `Budget.default` and
`cumulative` do not. So the dynamic layer fires during per-pass verification and
the composed check, and never on the `--verify-symbolic <effort>` CLI path:

| pipeline | before | static only | + induction |
|---|---|---|---|
| structural | 12.0s, 102 proved | 31.2s, 88 proved | 29.9s, 88 proved |
| `--fold` | 21.3s, 20 bn proved | 59.1s, 20 bn `max_rounds` | 57.0s, unchanged |

The cost at sampled efforts is therefore real and is the honest price: the old
speed there rested entirely on the unsound assumption, and no sound rule recovers
it, because at a sampled budget there is nothing exhaustive to induct from. Where
the budget does not sample, the induction is worth a great deal —
`verify_test.ml`'s "a proved cluster licenses the edges below it" closes a
four-node chain at `max_rounds = 1` that otherwise exhausts it.

**Coverage.** Each component of the static key has a soundness case paired with
the mutation that turns it red — output ordinal (swapped max-pool slots), operand
identity (crossed inputs over `relu (sub a b)`, the false proof this rule is named
after), input variables. The signature stays a conservative guard with no mutation
case, as predicted: `Shared` only decides whether an edge expands, and an expanded
edge with identical producers yields identical expressions either way. The
induction has its own, above.

**One refinement left on the table.** `Proved Constants` could license sharing in
the constants-bound attempt only, where its premise holds. Not implemented; it
would need two shared sets threaded through `compare_at`'s two attempts.

## 6b. Stage 5: what the recipe types do and do not buy

Three kinds of edge, and which one a field takes is the whole point: `'v source`
(exists in the graph being rewritten, and only `existing` makes one, by finding
the id there), `'v fresh` (allocated by this recipe), `'v target` (what an edit
defines — fresh, or a source output taken over).

**Only `subst` is a symmetric union**, and it earns it: `trim` substitutes
existing onto existing, `fold_batch_norm` existing onto fresh. Every other field
is directional. One union across all of them would let a claim be stated as
`(fresh, source)` and give up the guarantee the stage exists for — which is the
case `version_safety.t` now pins.

Two things this does NOT do, stated because the gap is easy to misread:

- **Edit construction is not side-safe.** `insertion.op` is a `Graph_ir.op` and
  takes raw `tensor_ref`s, so `fold_batch_norm` still writes fresh and existing
  ids side by side inside one payload, through `raw_fresh`/`raw_target`. Closing
  that means parameterising the whole IR, which §7 rules out.
- **`trim ~tie` stays unprotected.** Both ends are existing source edges, so the
  types cannot tell them apart and reversal there remains well typed.

`Pass`'s callbacks became rank-2 records to carry the version, joining `Pass.t`'s
`run`. That is the cost the design record predicted and had already paid once.

## 7. Non-goals

- **Parameterising the graph IR.** `Graph_ir.op` keeps raw `tensor_ref`
  operands. Typing them means touching every operation, node, builder,
  evaluator, importer and pass, for a boundary that recipes already guard.
- **Proving a map is semantically valid.** The compiler can show an id belongs
  to the right version. It cannot show that two separately loaded graphs are the
  intended pair; that is what stage 2's `create`-time checks are for.
- **A version tag on `Ground_expr.Cell`.** The verifier compares a source-side
  and destination-side expression structurally; tagged cells would give the two
  sides incompatible types. Stage 4's split is a *namespace* distinction with a
  runtime tag, not a phantom one.
