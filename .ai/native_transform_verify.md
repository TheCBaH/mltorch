# Symbolic verification of graph transformations

`native_transform_design.md` opens by saying its mapping exists so a harness can
take `(A, B, A→B)`, extract equivalence clusters, and check that both sides
compute the same values, and its §14 lists **"the numerical/symbolic verifier
itself"** as the one deliberate non-goal. This doc is that verifier:
`lib/native/transform/ground_expr.ml`, `ground_eval.ml`, `map_verify.ml`.

Status: **complete** — structural tier, constant payloads, the probe, cumulative
verification, the pipeline hook, the coefficient tier, sampling, and the CLI.

## 1. What it proves, and what it assumes

Given a source graph, a destination graph and the `Graph_map.t` between them, it
checks each value cluster's claim **without payload data for the graph inputs**.
A `proved` verdict is therefore a statement about *every* input, not about one
sample — which is the whole reason to do this symbolically rather than by
running both graphs on a random tensor.

The one hypothesis is that **corresponding graph inputs are fed the same data**.
That is not an obligation the verifier discharges; it is what "these two graphs
compute the same thing" means. Everything else is an obligation.

What this replaces: nine passes whose legality rested on prose plus a handful of
`Eval_direct` spot checks with one hand-made input (`fold_batch_norm_test.ml`,
`permute_passes_test.ml`). Those need a payload for every input, only inspect the
graph *output*, and never check the map's claim at all.

## 2. The ground form: concrete indices, symbolic payloads

`Eval_symbolic.run` gives one stage per output edge, each carrying its per-pixel
`Expr.t` whose `Load` leaves name the producing edge. `Ground_expr.t` is that
expression with every index expression evaluated at one concrete output
coordinate, every `Reduce` unrolled at its now-concrete bounds, and every
`Max_pool` window expanded. What is left is a binder-free term over unknown
tensor **cells** — `Cell of { id; coord }`, one scalar element of one edge.

**Grounding the indices is forced, not an optimisation.**
`Reshape.Compute.pixel` reaches its input through `flat_offset` then
`delinearize`, emitting `Index_floor_div_pos` and a mod-idiom per axis, while
`Permute.Compute.pixel` emits a bare `Index_var`. Those agree only under
`0 ≤ coordₐ < extentₐ` — **not an affine identity** — so no canonical
`Σ kᵢ·varᵢ + c` form could ever discharge `reshape_to_permute`. Evaluated at a
coordinate they are simply the same six integers.

Grounding also removes every binder, so alpha-renaming is unnecessary and the
reduction-variable capture problem that a symbolic-index design would have to
solve does not arise at all.

Because no binders survive, **structural equality is the whole `Identical`
test**: two ground terms denote the same value exactly when they are the same
term. There is no algebra in the structural tier.

## 3. Rounding is part of the semantics

`Schedule.evaluate`/`ground` call `Tensor.materialize`, which writes a **float32**
Bigarray. Every node output is rounded to f32. Naive inlining would turn
`f32(f32(a+b)·c)` into `f32((a+b)·c)` and let a future fusion be "proved"
`Identical` while changing bits.

So the ground language carries an explicit `Round` node. `Ground_eval.at` wraps a
stage's **own** body in `Round` (its result is materialized too), and `expand`
replaces a `Cell` by `Round (producer body at that coord)` — the boundary lands
exactly where the stored value was.

Three collapse rules, in `Ground_expr.normalise`:

| Rule | Condition |
|---|---|
| `Round (Cell c) → Cell c` | **only when `c`'s stored format round-trips through f32** |
| `Round (Round e) → Round e` | always — f32 rounding is idempotent |
| `Round (Const v) → Const (f32 v)` | always |

The first rule's side condition is load-bearing, not decoration.
`Payload.get_float` decodes `I32`/`I64` through `Int32.to_float`/`Int64.to_float`
— values above 2²⁴ are not f32-representable — and `I8`/`I16` through
`Quant.dequantize`, a scale multiply whose product need not be either. `F32`,
`F16` and `BF16` are safe (they carry no more mantissa). Dropping the condition
would make a permute stage's materialization vanish for a quantized input, where
it is observable.

This is why the permute family stays provable — a permute or reshape body grounds
to a bare `Cell`, so rule 1 collapses the removed stage — while a value-changing
fusion correctly does not. And `verify_test.ml` shows the condition earning its
keep: trimming an identity permute off an **i32** input reports
`format blocks collapse` rather than `proved`, and that is not conservatism, it
is correct — the transformation is genuinely false for a large enough i32 value.

## 4. Comparison is bitwise

`Identical` means bit-for-bit, so `Ground_expr.compare` compares `Const` leaves
by `Int64.bits_of_float`. `Float.equal`/`compare` equate `-0.` with `+0.` and
every NaN with every other, which are exactly the distinctions the max-pool
semantics turn on (see `Max_op` and `native_symbolic_language.md`). The same
mistake was live in the tests — `tensors_match` and `compare_symbolic` used
`Float.equal`, so they could not have observed the bug they existed to pin;
`Tensor.equal_bits` replaced them.

## 5. Clusters are sets, and every member is checked

`Correspondence.Cluster.t` carries `src : Set` and `dst : Set`. Comparing one
representative per side proves nothing about the others: for `{t0,t1} → {t0}`,
`t1` is precisely the trimmed edge the map claims is identical. So a canonical
member is chosen and **every other member is compared against it** —
`|src| + |dst| − 1` comparisons.

Members are side-tagged (`Member.t = Dst of _ | Src of _`) for two reasons:
`{src=t0} ↔ {dst=t0}` holds two distinct members sharing a raw id, and a
comparison can legitimately run **source against source**. `verify_test.ml`'s
non-canonical-member test reports exactly that: `src.t0 vs src.t1`.

## 6. Iterative deepening, and why it crosses corresponding edges

A cluster that does not close at round 0 has both sides expanded one level and is
retried. Expansion crosses **all** cells, not only those without a counterpart.

`reuse_permute_sub_order` is the counterexample to the narrower rule: its map
mentions only `{t3} → {}` and `{} → {t5}`, yet proving the untouched output
cluster requires expanding through `t2 = permute(t1)` — an edge present and
*corresponding* in both graphs. A frontier that stopped at corresponding edges
would leave it unproved.

Structural equality is tried **before** normalising and again after. Two
identical terms carrying the same uncollapsible `Round (Cell _)` are equal and
must be proved, not rejected for the blocked collapse.

Termination: each round strictly lowers the maximum stage depth of the remaining
cells, bounded by the stage count, and by `max_rounds`/`max_nodes` independently.

## 7. σ is restricted to graph inputs

Renaming both sides of an *internal* cluster to a representative assumes the very
claim under verification. That is sound only under an induction over a
topological order of the cluster DAG, which two graphs quotiented by a
correspondence can in principle make cyclic. Graph inputs are different in kind —
"corresponding inputs are fed the same data" is the hypothesis, not an obligation
— and renaming them is what makes `Rewrite.pack`'s input renumbering verifiable.

Relying on that keeps a cell's id usable both as a comparison key (renamed) and
as a stage key (original), because the two coincide off the inputs, which have no
stage.

Internal σ is a later, purely performance-motivated addition: it shortens
expansions, it never proves anything expansion could not.

## 8. Proof is sound under over-approximation; refutation is not

The asymmetry that shapes the verdict type. Two terms equal as functions of their
free cells are equal under **every** assignment, including the constrained ones a
truncated frontier admits — so a proof survives a frontier that never reached the
graph inputs. A *disagreement* has no such property: internal cells are
constrained by their producers, so assigning them independently can manufacture a
"counterexample" that no graph-input valuation can realise.

Consequently:

- a failed comparison is `Unproved` unless a witness was actually produced.
  Different normal forms only mean the incomplete prover failed;
- `Refuted` carries either a shape mismatch or a `Valuation` — a concrete
  assignment to the free cells that replays through `Ground_expr.eval` and
  separates the two terms;
- the probe is a **refutation engine**, not a weak proof: no number of agreeing
  draws yields `Proved`. It may only run once `Ground_eval.expandable` is false
  on both sides, because cells left at a truncated frontier are internal stage
  results constrained by their producers;
- it formally refutes only `Identical`. `Equivalent` explicitly permits rounding
  differences, so a rounded-term disagreement cannot refute it, and
  `Approximate` needs declared input ranges and an error model that
  `Precision.Set.t` alone cannot supply. Those get `Tested (Disagrees v)` —
  evidence loud enough to fail a strict policy, without claiming a proof.

Draw 0 is a coordinate ramp rather than noise: the errors this looks for are
permutation and indexing mistakes, and a ramp separates every cell of a tensor
where a constant would not. Later draws are pseudo-random and non-zero, and
every draw is a pure function of `(cell, n)` — no RNG state, so a printed
witness is reproducible and does not depend on the order cells were visited.

## 9. Constants narrow what a proof quantifies over

Binding the model's constant payloads turns those cells into `Const` leaves, and
`normalise` then folds the closed arithmetic over them — substitution alone is
not enough, since a multi-node constant sub-DAG has to collapse to the single
number the destination edge carries. Folding goes through `eval`, so it
reproduces the engine's arithmetic including `Round`'s f32 step. Because every
subtree is folded bottom-up, "closed" is exactly "came back a `Const`", and a
closed `Select` guard picks its branch even when the branches stay open. This
is what makes `fold_const` checkable at all — the destination edge *is* a
payload the pass computed — but it weakens the statement from "for every
payload" to "for every input, with these constants". So the driver keeps two
envs per graph and tries the unqualified one first, reporting
`Proved Structural` when that succeeds and `Proved Constants` when only the
bound one does.

**The unqualified attempt may only prove, never refute**, and that is a
soundness requirement rather than a preference. With constants left free the
probe could assign a known weight any value it liked and "refute" a fold that is
perfectly correct for the weight the model actually carries — the same
manufactured counterexample that probing a truncated frontier would produce.
Every verdict other than `Proved` therefore comes from the constant-bound
attempt, where the cells still free really are free inputs.

Constants are also **per-graph**, not shared: `Rewrite.apply` filters the
payload map to live destination ids, so one that a fold consumed and deleted
survives only in the before-state. `Map_verify.step` reads
`Rewrite.constants` from each side separately.

## 10. Cumulative verification

`Pass.run_all` already threads `Graph_map.compose`, and `Map_verify.step` takes
the before-state and a step, so verifying a **composed** origin-to-final map
needs no extra API — handing `run_all` more than one pass already does it. What
the cumulative stage adds is the other half: applying passes one at a time and
verifying each step against the state it started from, so a failure names the
pass that caused it.

The two answer different questions. The per-step chain is already a proof of the
end-to-end claim *provided composition is sound*; the composed check is what
tests that proviso, and it is what the PT2 provenance lens actually resolves
through.

**What composed verification catches, and what it cannot.** It catches a
composition error that makes a composed value claim **false at the endpoints**.
It does not validate `compose`'s algebraic contract in general: an
over-conservative (too weak) label is legal and therefore unverifiable,
provenance edges carry no value claim at all, and a cluster set that is wrong but
endpoint-consistent still passes. Associativity, identity-extension and the
created/deleted guard remain `graph_map_test.ml`'s job.

**One law, pinned in `verify_test.ml`:** if every step verifies `Proved`, the
composed verification must not be `Refuted`.

Its converse is **not** a law. A composed `Unproved` with every step `Proved` is
an acceptable outcome — the composed frontier spans the whole pipeline and can
exhaust its budget where a single step does not — which is why
`Budget.cumulative` exists and why the acceptance criterion is "composed never
refuted", not "composed always proved". Nor is verification strength monotone
under composition: two `Equivalent` steps whose roundings cancel can compose to a
bit-identical pair that the composed check proves outright.

The stress targets are `origin → passes → Rewrite.pack`, since packing is the
`{t11} ↔ {}` then `{t12} ↔ {t11}` hazard §9 of `native_transform_design.md`
warns about, and a `fixpoint` fold over a multi-node constant sub-DAG, where the
composed map is a chain of per-iteration maps.

## 11. The pipeline hook, and why two policies

`Pass.run_all ?verify` checks each step against the state it came from **as it
is applied**, so the first offending pass stops the pipeline and the error names
it, rather than a later composed map hiding which rewrite was wrong. Reports for
accepted steps are dropped; a caller wanting all of them calls
`Map_verify.step` itself.

`Pass.error` grows one variant carrying a `Verification.t`, which distinguishes
the two ways verification fails — the verifier itself erroring (a map that does
not describe its two graphs, a missing signature) versus succeeding and having
its report rejected — and names the pass in both.

`Policy.Reject_refuted` fails only on an actual counterexample;
`Policy.Require_proved` fails on anything short of a proof. They are not
redundant, and the i32 trim in `verify_test.ml` is the case that separates them:
it is genuinely unproven — false, in fact, for a value above 2²⁴ — but the
verifier has exhibited no counterexample, so the release bar tolerates it while
the development bar does not. Shipping only `Reject_refuted` would let an
unjustified rewrite land; shipping only `Require_proved` would fail a release
build on every budget exhaustion.

## 12. The coefficient tier

Batch-norm folding re-associates: `(Σ xₖ·Wₖ)·s` becomes `Σ xₖ·(Wₖ·s)`. No
structural comparison reaches that. `Coeff_form` compares the two as
polynomials in their free cells — which is exactly what distribution and
re-association look like — with `Round` erased, since that is the `Equivalent`
reading.

**It never yields a proof.** Coefficients `1` and `1 + ε` pass a tolerance while
being neither exactly equal nor boundedly close for an unbounded free cell. So
agreement is `Tested (Agrees tol)`, and for an `Identical` claim a probe still
gets to refute — bits are the question there, and coefficients do not answer it.

Two details that are not incidental:

- The tier runs on the **normalised** terms, not the raw ones. Folding is what
  turns `sqrt (Const _)` — batch norm's normaliser — into a coefficient rather
  than an opaque generator the polynomial view cannot see through. Running it on
  raw terms reports disagreement on a correct fold.
- `agree` recurses through **matching non-arithmetic heads** rather than
  reducing the whole term to one polynomial. Without that, a relu wrapping the
  fold — `select(E < 0, 0, E)` on both sides with `E` differing only by rounding
  — compares two unequal opaque generators and disagrees. The gap this leaves:
  an arithmetic *combination* of structurally-different non-polynomial subterms
  still compares those exactly. Closing it needs matching atoms up to the same
  relation, a matching problem no current pass poses.

The tolerance is `|a − b| ≤ tol · max(1, |a|, |b|)` per coefficient, defaulting
to `default_coefficient_tolerance = 1e-5`. That is the same number
`fold_batch_norm_test.ml`'s output-level check uses, shared for familiarity and
**not** because the two are the same bar: per-coefficient and whole-tensor
agreement are incomparable in general, since many small coefficient errors can
sum while one large error on a near-zero activation may never surface. That is
precisely why the verifier was added alongside that check rather than replacing
it, and why both verdicts print on the same line there.

No exact-rational tier is planned. It would close nothing: `fold_batch_norm`
re-derives its constants numerically, and `eps` arrives as a constant *edge*
with a payload against a source-side `Const`, so exact equality fails whatever
the arithmetic.

## 13. Budgets, and what a real model costs

Counted, never timed, so a verdict is deterministic and a golden is stable.
`max_coords` is checked against `Vec6.numel` **before any expansion** — O(1),
and it alone refuses every whole activation tensor. Then `max_nodes`, then
`max_rounds`. All three produce **verdicts**, not exceptions, so pointing the
verifier at a real model yields a report full of `too_large` lines in bounded
time rather than hanging.

`Ground_eval.at` does not meter itself: it builds a single op's body at a single
coordinate, bounded by the op (a conv's kernel × in-channels) rather than by the
graph. **`expand` does**, and that turned out to matter more than anything else
here. One substitution round is quadratic where a conv feeds a conv, so
measuring only afterwards lets a term reach tens of millions of nodes first.
Measured on resnet18: verification ran **over 25× the whole transform** before
`expand` took a budget, and **0.3×** after. Running out mid-round leaves cells
unexpanded, which is sound rather than approximate — an unexpanded cell keeps
`expandable` true, so the driver reports a budget verdict and no probe may run
against a frontier that never reached the inputs.

`Budget.release` is shaped by what deep expansion is worth on a real model:
nothing. The clusters a real graph makes checkable are the **constant-shaped**
ones — folded weights and biases, where `fold_const` and `fold_batch_norm` act —
and those close in one or two rounds. An activation-shaped cluster needs as many
rounds as the network is deep and is hopeless however much budget it gets.

Widening `max_coords` alone is a trap: a late-layer activation like `[1,512,7,7]`
is only 25088 elements, so it passes a generous coord budget and then expands the
entire network behind it. There is no coord threshold that separates the two —
`[1,512,1,1]` after the average pool is 512 elements and just as deep. **Depth is
the discriminator**, so `max_rounds = 2` is what actually bounds it.

Timings, resnet18 and mobilenet_v3_small, `transform --fold`:

| | baseline | `--verify-symbolic` |
|---|---|---|
| resnet18 | 4.7s | 6.4s |
| mobilenet_v3_small | 1.5s | 2.5s |

## 14. Reporting: groups, effort, and why a count needs a reason

A flat count over a real model's ~1600 clusters says nothing about which part
was covered, so a `Report` carries an `Entry` per cluster with the
**`Group_path`** it belongs to — the destination graph's structural hierarchy,
which for an imported model is the PT2 call sites the importer recorded
(`torch.ops.aten.convolution.default`, and so on). A cluster is placed by its
destination edges, since that is the graph the transformation produced; an edge
with no producer is a graph input and lands at the root.

`Tally` counts by **outcome and reason together**. "40 unproved" is not
actionable; "40 unproved (too large)" says the coordinate budget refused them
before any expansion, which is a very different statement from "40 unproved
(frontier exhausted)". A sampled verdict gets its own label rather than being
folded in with the exhaustive ones — counting a sampled proof as "proved" is
precisely the overstatement `Coverage` exists to prevent.

`Effort` (`Quick | Standard | Thorough`) is a named point on the cost/coverage
curve, so a caller picks how hard to look without knowing what a coord or a
round is. It drives the probe count as well as the budget, since a deeper
frontier is worth more draws.

Audits reach the caller by being **carried through the pass tree** —
`Pass.run_reporting` returns them alongside the step — rather than pushed to a
callback, so observing a run needs no mutable state. `run_all` drops them for
the callers that do not want them. A pass whose sweep matched nothing produces
an identity step, which has an empty map and so nothing to verify; skipping
those is also what keeps a no-op sweep from dominating the cost on a real graph.

`test/native_transform_verify_cram.t` and its `--fold` companion pin all of this
on ResNet-18. The contrast between them is the point of having both: without
folding everything provable is `proved (structural)`, and with it the
constant-shaped clusters become `proved (for these constants)` — a weaker claim,
labelled apart, that holds for every input but only for the weights this model
carries.

## 15. Verdicts

```
Proved of Strength.proof        Structural (every payload) | Constants (every input)
Refuted of Refutation.t         a shape mismatch, or a reproducible valuation
Tested of Strength.test         evidence, never a proof
Unproved of Unproved.t          the prover failed; nothing is asserted
Vacuous                         a creation or a deletion: the cluster claims nothing
```

carried in `Outcome.t = { coverage; verdict }`. Coverage is **orthogonal** to the
verdict rather than one more verdict constructor, which is what keeps a sampled
`Tested` as visibly partial as a sampled `Proved` — folding sampling into the
verdict would have weakened only `Proved` and let a sampled `Agrees` read as
exhaustive. `Report.proved` is strict — every outcome `Proved` and `Exhaustive`,
or `Vacuous` — while `Report.refuted` ignores coverage, because a counterexample
found at a sampled coordinate is still a counterexample.

Sampling picks every stride-th coordinate in `fold_coords` order: deterministic,
because a sampled verdict has to be reproducible from a golden, and strided
rather than random because the errors being looked for are index-shaped, so
spreading over the space beats clustering.

## 16. Coverage today

On the fixtures, exhaustively:

Proved structurally, over all payloads: `trim_permute`, `chain_permute`,
`bypass_permute`, `sink_permute`, `sink_permute_mean`, `reuse_permute` (including
the non-commutative `Sub`/`Div` orderings), `reshape_to_permute`.

Proved with the model's constants substituted, for every input:
`fold_const`.

Agreeing within tolerance, for every input, with these constants:
`fold_batch_norm` — all eight operand combinations, through both `Conv2d` and
`Convolution`. Not `Proved`, and deliberately so (§12).

## 17. Non-goals

- An exact-rational tier — see §12 for why it would close nothing.
- An `Approximate` error model. Nothing emits `Approximate` yet.
- Validating `compose`'s algebraic contract — see §10 for the boundary.
- Renaming already-proved INTERNAL clusters through σ. It was planned as a
  speed-up, and the measurement above retired it: the cost was never chain
  DEPTH, which σ would shorten, but the breadth of a single conv-into-conv
  round, which it would not touch. Bounding `expand` fixed that outright. It
  would also need a topological order over the cluster DAG with a cycle
  verdict, for a benefit nothing now demands.
