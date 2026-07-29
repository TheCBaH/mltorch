# Version-indexed ids — making a graph version part of an id's type

Deepens `native_transform_design.md` §3-§5 and `native_transform_verify.md` §7.
Covers `lib/native/transform/brand.ml`, `snapshot.ml`, and the `Tagged` layer
inside `cluster_relation.ml`.

Status: **stage 1 of 5 implemented** — `Brand`, `Snapshot`, `Cluster_relation.Tagged`,
and the negative-compile harness (`test/native/version_safety.t`). Stages 2-5
(typed endpoints, the verifier's `Member`, `Input_var`, typed recipes) are
designed in §6 and not yet built.

## 1. The problem the phantoms did not solve

`native_transform_design.md` §3 already tags a map `('src, 'dst)`. The tags
order a composition, but the *contents* erase them: `Cluster.t` holds plain
`Id.Set.t`, and `forward`/`backward`/`sources_of` take and return raw ids. The
`.mli` says so itself (`cluster_relation.mli:58-62`): they "cannot tie a map to
two PARTICULAR graphs — `of_clusters` is polymorphic in them — which is why
`validate` exists".

The cost is on record twice. `map_verify.ml:316` fixed a source id looked up in
the destination producer map. `symbol-impl-2.md` records a **false proof**: a map
swapping a graph's two inputs was reported structurally proved, output included,
because src and dst share one numeric namespace.

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

> **This reverses two recorded decisions, deliberately.** `native_transform_design.md`
> §5 (`:326-328`) and §11 (`:675-689`) rejected a versioned `Graph_view` because
> `of_graph : graph -> ('v t, error) result` lets the caller pick `'v`.
> `Snapshot.create` returns a `packed` and does not. The companion objection —
> "would force every `Pass` callback to be a rank-2 record" — is already paid:
> `pass.mli:51` is one today.

## 3. Where the tagged layer lives, and why

`Tagged` is a submodule of `Cluster_relation.Make`, not a module beside
`Snapshot`. The reason is the implicit-identity rule: `forward` answers an
*unmentioned* source id with itself on the destination side, which is a retag
across versions. Only a body that owns the erasure (`type 'v id = Id.t`) can do
that. Keeping it inside the functor is what lets the retag exist without a
`retag : 'a id -> 'b id` appearing in any signature — where it would be exactly
as forgeable as the raw ids it replaces.

`Snapshot` binds one `Brand.t` to both id spaces plus the validated view, so a
snapshot's tensors and nodes share a version and a `Graph_map` can be indexed by
one tag rather than one per id space.

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
`validate` needed anyway. From stage 2, `Graph_map.validate` therefore stops
being a separate call a consumer must remember, and `create ~src ~dst` becomes
the only way to assemble a map.

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
  `Cluster_relation.Make(Correspondence.Id)(Correspondence.Label).Tagged.t` and
  existentials like `$Pack_'v1`, whose spelling is a compiler detail.

Verified non-vacuous by mutation: relaxing `forward` to `'a id -> 'dst set` flips
exactly one line to `COMPILES` and leaves the other six untouched.

## 6. Staging

| stage | content | status |
|---|---|---|
| 1 | `Brand`, `Snapshot`, `Cluster_relation.Tagged`, harness — additive | done |
| 2 | delete the raw API; typed endpoints through `Rewrite`, `Map_verify`, the PT2 lens; `Graph_map` abstract with `create ~src ~dst`, claim closure, cluster metadata validation | planned |
| 3 | `Map_verify.Member` as a GADT, tag erased at the report boundary | planned |
| 4 | `Input_var`, and `Ground_expr.Cell.origin` with `Shared` earned structurally | planned |
| 5 | typed recipes: `'v source`, `'v target`, `'v fresh` | planned |

Stage 2 cannot be subdivided: `map_verify.ml:794-795` and
`pt2_native_graph.ml:145,166-167` consume `Graph_map.validate`,
`clusters_over` and `Correspondence.Cluster.t` directly, so changing the
relations without them in the same commit leaves the tree red or forces
temporary raw escape hatches.

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
