# Symbolic verification of graph transformations

`native_transform_design.md` opens by saying its mapping exists so a harness can
take `(A, B, A→B)`, extract equivalence clusters, and check that both sides
compute the same values, and its §14 lists **"the numerical/symbolic verifier
itself"** as the one deliberate non-goal. This doc is that verifier:
`lib/native/transform/ground_expr.ml`, `ground_eval.ml`, `map_verify.ml`.

Status: **stage 1** — the structural tier, per rewrite step. Constants, the
probe, cumulative verification, the pipeline hook, the coefficient tier and the
CLI are staged after it; each section below says what it does and does not yet
cover.

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

- a failed comparison is `Unproved`, **never** `Refuted`. Different normal forms
  only mean the incomplete prover failed;
- `Refuted` requires an actual witness. In stage 1 the only one available is a
  shape mismatch; the value case waits for the probe, which is a *refutation
  engine*, not a weak proof, and may only run once `Ground_eval.expandable` is
  false on both sides;
- a probe will formally refute only `Identical`. `Equivalent` explicitly permits
  rounding differences, so a rounded-term disagreement cannot refute it, and
  `Approximate` needs declared input ranges and an error model that
  `Precision.Set.t` alone cannot supply.

## 9. Budgets

Counted, never timed, so a verdict is deterministic and a golden is stable.
`max_coords` is checked against `Vec6.numel` **before any expansion** — O(1), and
it alone refuses every real activation tensor. Then `max_nodes` on the expanded
pair, then `max_rounds`. All three produce **verdicts**, not exceptions, so
pointing the verifier at a real model yields a report full of `too_large` lines
in bounded time rather than hanging.

Neither `Ground_eval.at` nor `expand` meters itself: one `at` builds a single
op's body at a single coordinate, bounded by the op (a conv's kernel × in
channels) rather than by the graph. The unbounded direction is repeated
expansion, and that loop belongs to the driver — so the driver sizes the result
between rounds instead of threading fuel through the recursion.

## 10. Verdicts

```
Proved of Strength.proof        structural equality of the ground terms
Refuted of Refutation.t         a shape mismatch (stage 1) or a valuation (stage 2)
Unproved of Unproved.t          the prover failed; nothing is asserted
Vacuous                         a creation or a deletion: the cluster claims nothing
```

carried in `Outcome.t = { coverage; verdict }`. Coverage is **orthogonal** to the
verdict rather than one more verdict constructor, so that when sampling arrives
(stage 6) a sampled `Tested` is as visibly partial as a sampled `Proved`.
`Report.proved` is strict — every outcome `Proved` and `Exhaustive`, or `Vacuous`
— while `Report.refuted` ignores coverage, because a counterexample found at a
sampled coordinate is still a counterexample.

## 11. Coverage today

Proved structurally, over all payloads: `trim_permute`, `chain_permute`,
`bypass_permute`, `sink_permute`, `sink_permute_mean`, `reuse_permute` (including
the non-commutative `Sub`/`Div` orderings), `reshape_to_permute`.

Pending: `fold_const` needs constant payloads (stage 2); `fold_batch_norm` needs
the coefficient tier (stage 5) and lands at `Tested (Agrees tol)`, not `Proved` —
tolerance is never a proof, since coefficients `1` and `1+ε` pass a tolerance
while being neither exactly equal nor boundedly close for unbounded free input.

## 12. Non-goals

- An exact-rational tier (`Proved Exact_algebra`). No current pass needs it:
  `fold_batch_norm` re-derives its constants numerically, so exact coefficient
  equality fails regardless.
- An `Approximate` error model. Nothing emits `Approximate` yet.
- Validating `compose`'s algebraic contract. Cumulative verification (stage 3)
  catches composition errors that make a composed claim **false at the
  endpoints**; it cannot see an over-conservative label, provenance carries no
  value claim, and a cluster set that is wrong but endpoint-consistent still
  passes. `graph_map_test.ml` keeps that job.
