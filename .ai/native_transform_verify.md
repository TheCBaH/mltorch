# Symbolic verification of graph transformations

`native_transform_design.md` opens by saying its mapping exists so a harness can
take `(A, B, A→B)`, extract equivalence clusters, and check that both sides
compute the same values, and its §14 lists **"the numerical/symbolic verifier
itself"** as the one deliberate non-goal. This doc is that verifier:
`lib/native/transform/ground_expr.ml`, `ground_eval.ml`, `map_verify.ml`.

Status: **stage 4** — structural tier, constant payloads, the probe, cumulative
verification, and the pipeline hook. The coefficient tier and the CLI are staged
after it.

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

## 12. Budgets

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

## 13. Verdicts

```
Proved of Strength.proof        Structural (every payload) | Constants (every input)
Refuted of Refutation.t         a shape mismatch, or a reproducible valuation
Tested of Strength.test         evidence, never a proof
Unproved of Unproved.t          the prover failed; nothing is asserted
Vacuous                         a creation or a deletion: the cluster claims nothing
```

carried in `Outcome.t = { coverage; verdict }`. Coverage is **orthogonal** to the
verdict rather than one more verdict constructor, so that when sampling arrives
(stage 6) a sampled `Tested` is as visibly partial as a sampled `Proved`.
`Report.proved` is strict — every outcome `Proved` and `Exhaustive`, or `Vacuous`
— while `Report.refuted` ignores coverage, because a counterexample found at a
sampled coordinate is still a counterexample.

## 14. Coverage today

Proved structurally, over all payloads: `trim_permute`, `chain_permute`,
`bypass_permute`, `sink_permute`, `sink_permute_mean`, `reuse_permute` (including
the non-commutative `Sub`/`Div` orderings), `reshape_to_permute`.

Proved with the model's constants substituted, for every input:
`fold_const`.

Pending: `fold_batch_norm` needs
the coefficient tier (stage 5) and lands at `Tested (Agrees tol)`, not `Proved` —
tolerance is never a proof, since coefficients `1` and `1+ε` pass a tolerance
while being neither exactly equal nor boundedly close for unbounded free input.

## 15. Non-goals

- An exact-rational tier (`Proved Exact_algebra`). No current pass needs it:
  `fold_batch_norm` re-derives its constants numerically, so exact coefficient
  equality fails regardless.
- An `Approximate` error model. Nothing emits `Approximate` yet.
- Validating `compose`'s algebraic contract — see §10 for the boundary.
