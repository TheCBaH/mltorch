# Region computation — landed design

## Status and scope

This record describes the landed Region-computation implementation.  It is the
authoritative tracked companion to `native_compute_design.md` and
`native4d_design.md`; the gitignored planning notes this branch worked from
are execution scaffolding, not required reading for a fresh checkout.

The implementation gives RMSNorm, LayerNorm, plain Softmax, and SDPA an
operation-authored `Region_program.t`.  Other operations remain Pixel-authored.
Every logical Kernel value has exactly one `Region_program.t`: a Pixel body is
embedded mechanically with `Region_program.pixel`, preserving the original
expression object.  There is no Kernel sum type and no optional Region
admission/fallback path.

SDPA's program caches both scalars (row max/sum/scale, Stage A) and, since
`0a0bc41`, two per-key VECTOR locals -- its score row and its normalized
softmax weight row (Stage B) -- see "Design headroom" below for what that
adds and what it still does not.

Out of scope: loop IR, tiling, multi-output programs, automatic Pixel-to-Region
discovery, graph fusion, GroupNorm and batch normalization Region forms,
safe/log softmax, blocked online-softmax state, relaxed rounding, tree/parallel
reductions, and multi-node Region fusion.

## Program contract

A program consists of a partition of output coordinates, an ordered list of
scalar locals, and one emitter.  The partition maps each output coordinate to
one key.  For each key, locals are evaluated once in declaration order; the
emitter is then evaluated for every coordinate owned by that key.  A Pixel
program is the degenerate one-output-per-key form with no locals.

`Region_program.check` is the construction trust boundary.  In particular, a
local may depend on inputs, earlier locals, and only output coordinates which
are invariant within its partition.  `Non_invariant_local` is the load-bearing
invariant: it prevents a value computed once for a key from silently depending
on an output coordinate that varies inside that key.  The checker also enforces
ordered/known locals, partition legality, and caller-supplied size/depth
limits.

`Region_partition.fold_outputs` enumerates a key's owned outputs directly from
the key and the `Whole`-axis extents.  Materialization therefore visits each
output once, rather than scanning the entire domain for every key.

There are three deliberately distinct observations of a program:

- `Region_execution.materialize` is production tensor execution and evaluates
  locals once per key.
- `Region_program.value_at` is a fresh scalar projection, available only at
  named compatibility/test boundaries; repeated use is intentionally not a
  materialization strategy.
- `Region_program.specialize_pixel` creates a symbolic scalar expression for
  transformation checks; it does not execute a tensor.

The `Region_trace` test API independently records keys, local evaluations,
emitter visits, and ownership.  It verifies complete coverage, no duplicate
outputs, and agreement between partition ownership and enumeration.  This is
why a dense result comparison alone is insufficient evidence.

## Design headroom for deferred work

Three decisions were fixed now specifically so a later phase extends this IR
rather than redesigns it.

`Region_local.t` carries a `shape : Region_local.Shape.t` field, reserved for
exactly this: `Shape.t` now has a second constructor, `Vector of { extent :
int; var : Expr.Reduce_var.t }` (the `K`-element cache SDPA needed), landed
without any change to `Shape.t`'s own field or to `Region_local.t`'s three
fields -- it is a new constructor and new `Region_execution`/`Region_eval`
cases, not a language change, exactly as this section anticipated.  `Expr`
needed no new binder-minting primitive either: `Builder.fresh_reduce`, public
since Stage A, already mints a `Reduce_var.t` outside a reduction, and a
vector local's body is free in that binder the same way a reduction's body is
bound in one -- `Expr.Value` gained one new leaf, `Local_at`, to read it back
at a computed index, and `Expr.Check.fragment` gained one exemption
(`~allowed_free`) so that binder's deliberate freedom is not mistaken for the
composition defect the free-reducer check otherwise exists to catch.  Still
out of scope: a `V`-element output accumulator and blocked online-softmax
state, neither of which SDPA's own Stage B needed (see below).

Region semantics are deliberately separate from schedule tiles.  An output
region states what may share one computed local; a schedule tile is a
backend's physical loop/memory block.  Required sharing comes from the
operation; locality blocking is schedule-selected either way.  This is why
`Region_partition` has only `Whole` and `Singleton` and no `Block`: a `Block`
partition is reserved for a program whose block itself owns an explicit local
or ordered phase, not for an arbitrary machine tile a schedule picks
independently of program meaning.

SDPA needed more than scalar sharing to reach an efficient form.  The
per-pixel implementation recomputes every attention score once per output
value feature, so its leading cost is `Theta(Q * K * E^2)` against an ideal
`Theta(Q * K * (E + V))`.  Stage A's scalar locals removed the repeated
softmax row max/sum (`sf`/`m`/`z` shared once per key, cutting the dominant
term from `3*Wk*E^2` to `Wk*E*(E+2)`) but not the repeated numerator; Stage B
closes most of the remaining gap with two `K`-element (`Wk`) vector locals --
`s`, the score row (`score_at`, cached instead of recomputed at each of
`m`/`z`/`numer`'s three uses), and `p`, the normalized softmax weight row
(`exp(s[k]-m)/z`, cached instead of recomputed at `numer`'s own use) -- taking
the per-key cost to `Wk*(2E+3)`, against Stage A's `Wk*E*(E+2)`: roughly
`1.5*E` faster, e.g. ~96x at `head_dim = 64`.  What remains open is a
`V`-element output accumulator or blocked online-softmax state, which would
close the rest of the gap to `Theta(Q*K*(E+V))` — still out of scope.

The numerical point worth stating precisely, since a shaped-local cache
sounds like it could open a tolerance question and does not: caching `s`/`p`
only SUBSTITUTES values that were already computed identically at every use
site.  No reduction changes order, grouping or bounds, so
`Region_program.specialize_pixel` still beta-reduces each cached read back to
exactly the expression `Legacy_pixel` computes, and the claim stays
`Identical` -- proved by `Region_program.reconstructs`, not asserted.  A
tolerance policy, a second oracle, or an `Equivalent` claim would only be
needed by the online-softmax strategy this section still defers, which is the
one strategy that reassociates rather than merely caches.

A lower operation count is not automatically a lower wall-clock time, and
SDPA is where that gap first became visible: `Region_execution` evaluates
through `Expr.Eval`'s general interpreter, and a nested reduction (SDPA's
per-feature score is a reduction inside the row max/sum reduction) costs more
per element there than a flat one. Stage A's own measurement (query/key
`[W=2,C=3]`/`[W=3,C=3]`, and the `report_sdpa` sweep below) found the scalar-
sharing win did NOT show up as a faster `Eval_direct` run against the
`Legacy_pixel` oracle it reconstructs -- only as fewer operations, because at
those sizes the interpreter's per-node constant dominated the raw op-count
cut.

**Stage B reverses that finding.** Re-measured with the fix below applied
(`opam exec -- dune exec bin/region_compute_bench.exe`, median of 20 samples,
stable across repeated runs), `kernel_region_ms`/`direct_ms` now BEAT
`legacy_pixel_ms` at every `report_sdpa` point except the smallest, and by a
wide, consistent margin against Stage A's own recorded numbers at the same
shapes:

| regions (Wq=Wk) | extent (E) | Stage A `kernel_region_ms` | Stage B `kernel_region_ms` | Stage B `legacy_pixel_ms` | Stage B speedup vs `Legacy_pixel` |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1  | 8  | 0.029 | 0.006 | 0.009 | 1.5x |
| 4  | 8  | 0.448 | 0.071 | 0.126 | 1.8x |
| 16 | 8  | 7.112 | 1.10  | 2.00  | 1.8x |
| 4  | 16 | 1.502 | 0.14  | 0.47  | 3.4x |

Two effects compound: caching `s`/`p` cuts the interpreter's own per-key work
(the `report_sdpa` sweep's `loads`/`reductions` counters drop the same way
Stage B's own tracker recorded), and doing so ALSO shrinks
`kernel_region_ms`/`direct_ms` by 4.8x-11x from Stage A's own recorded numbers
at identical shapes -- large enough that the interpreter's constant no longer
dominates at these sizes. This is a measured reversal of this section's
previous claim, not a re-derivation of the cost model above (which was never
wrong -- it always predicted fewer operations, just not by how much wall
clock would follow). A specialized/compiled execution form remains unbuilt
and would still help further, but Stage B does not need it to already beat
the uncached oracle here.

## Numerical policy

Programs retain the existing operation order and `f32` materialization policy.
`Identical` means the original expression and reconstructed/specialized Region
expression are structurally the same under the stated limits.  `Equivalent`
is reserved for transformations whose declared arithmetic or materialization
boundaries change.  `Region_program.reconstructs` remains a general
transformation tool and the tests compare authored RMSNorm, LayerNorm,
Softmax, and SDPA programs against their test-only legacy scalar oracles.  A
malformed
authoritative program is a typed construction/limit error, never a request to
run a second handwritten Pixel algorithm.

## Ownership and routing

`Region_computation` is the operation-facing construction boundary.  It takes
the output ordinal and shape plus role-resolved operands, applies actual
`Kernel.Limits`, and returns a program or a typed error.  It contains the
generic construction helpers and the normalization divisor's bounded shared
calculation; dispatchers contain no operation arithmetic.

Native `Eval_direct` routes RMSNorm, LayerNorm, Softmax, and SDPA through that
program and materializes it once per key; Native4D `Eval_direct4` routes all
four the same way -- SDPA only at `D = 1`, its one real batch axis
(`native4d_design.md` §7.9); `D > 1` stays outside Native4D's domain.  Their
Symbolic counterparts put the exact program in
`Stage_program.Stage.computation`.  `Stage_program`, grounding, transform
verification, Kernel adaptation, and the Model Explorer consume this
structural program.  Pixel-authored operations remain on their existing
specialized Pixel schedule.

Native4D owns Axis4 legality and parameter adaptation, but delegates numeric
Region construction to the shared Native operation definition.  Its Direct and
Symbolic routes therefore have the same operation form and Region sharing as
Native, without a second numeric kernel, for the three forms it admits.

`Regionizer`, candidate maps, optional `regionized` results, candidate
reconstruction admission, synthetic-value lookup, and production scalar
materialization are removed.  `Legacy_pixel` exists only as the test oracle for
the four migrated operations.

The routing audit is deliberately narrow and exhaustive.  `Region_computation`
recognizes Native `Rms_norm`, `Layer_norm`, `Sdpa`, and `Softmax`; its Native4D
adapter recognizes only `Rms_norm`, `Layer_norm`, and `Softmax4` before mapping
to that same Native dispatcher.  The Native Direct and Symbolic drivers select
the Region route for all four forms; Native4D's select its three.
`Eval_op.pixel` and `Eval_op4.pixel` retain explicit impossible arms for their
respective sets, so a future caller cannot silently revive a
scalar production algorithm.  Repository-wide references to the four
`Legacy_pixel` implementations are their definitions, explanatory comments,
and test invocations only; production dispatch does not invoke them.

## Cost model and evidence

For `R` region keys and reduction extent `K`, a Region materialization owns
each output once and computes its locals once per key.  The benchmark sweeps
both variables at `(R, K) = (1, 32), (4, 32), (16, 32), (64, 32), (4, 8),
(4, 64)`; `(64, 32)` is the required `R > K` case.  It reports median elapsed
milliseconds and allocated GC words for Kernel Region execution, Native Direct,
and Native4D Direct, alongside `keys`, `locals`, `emitters`, `loads`, and
`reductions`.

At `(64, 32)`, all three routes report the same operation counters:

| Operation | keys | locals | emitters | loads | reductions |
| --- | ---: | ---: | ---: | ---: | ---: |
| RMSNorm | 64 | 128 | 2048 | 8192 | 2048 |
| LayerNorm | 64 | 256 | 2048 | 12288 | 4096 |
| Softmax | 64 | 128 | 2048 | 6144 | 4096 |

Those counters observe only Region keys, local evaluations, emitter visits,
expression loads, and reductions.  They do not observe partition-membership
tests, scalar-projection calls, allocations, or elapsed time; the last two are
reported separately by the benchmark.  They are evidence of what they count,
not a substitute for the ownership trace or wall-clock measurement.

SDPA runs the same benchmark under a separate, smaller `report_sdpa` sweep
(query/key sequence length and head dim, not `(R, K)`) in the same
executable, and it additionally times `Legacy_pixel(Direct)` directly. Under
Stage A this was the evidence for the previous section's operation-count-vs-
wall-clock distinction: `Legacy_pixel` was faster in absolute terms at every
size measured. Under Stage B, re-measured, the direction reverses -- see the
table in the previous section. SDPA's row now carries a `direct4_ms` column
too: since the `D = 1` admission (`native4d_design.md` §7.9) Native4D has a
real Sdpa route, delegating to the same Region program Native does, and this
column measures Native4D's translation/dispatch overhead over that identical
program rather than a second numeric kernel.

**A vector local of extent 1 was silently mis-evaluated until this
measurement surfaced it.** Both evaluators' `evaluate_locals` loops must
dispatch on a local's declared `Region_local.Shape.t`, rather than its numeric
SLOT COUNT (`(offset, 1)` for "scalar" versus `(offset, count)` for
"vector"). A `Vector` local whose extent happens to be 1 (SDPA's `s`/`p` at
`Wk = 1`, e.g. `report_sdpa`'s own `(regions=1, extent=8)` point, the first
one this benchmark sweeps) has the identical `(offset, 1)` range as a
`Scalar`. Selecting the scalar branch leaves its per-element reducer unbound.
`s`'s body mentions that binder free by construction
(`Region_program.Builder.vector`), so this raised `Unbound_reducer`. The
production materializer and reference `Region_eval` projection/materializer
are separate loops, so the regression constructs the same Wk=1 SDPA Region
directly as well as through `Eval_direct.run`; it compares both materialized
tensors and a `value_at` projection bitwise with `Legacy_pixel`. Restoring the
faulty dispatch makes that test fail at the reference materializer, proving it
reaches the previously missing path.

Reproduce elapsed-time and GC-word medians with
`opam exec -- dune exec bin/region_compute_bench.exe`; both are
runtime-sensitive, so this record keeps the counter table above rather than a
dated run of that command, except for the Stage-A-vs-Stage-B table above,
kept because the comparison is to a fixed historical baseline, not a
snapshot of the current state.  `bin/region_pixel_bench.exe` is the separate,
unchanged Pixel Kernel no-regression benchmark.

## Validation commands

Reproduce with:

```sh
NO_COLOR=1 opam exec -- dune runtest
make jsoo.runtest
make jsoo.inline-runtest
opam exec -- dune exec bin/region_compute_bench.exe
opam exec -- dune exec bin/region_pixel_bench.exe
make pt2.json-model-support
```

The full test run includes Native, Native4D, Model Explorer, transformation,
and cross-dialect verification.  The benchmark's Native4D Direct counters are
part of its runtime output, so the same reproducible command checks form-policy
parity without presenting Native as the numerical oracle.

## Naming decision

`Transform.Region` remains the established name for a claimed **graph-node
set** with derived inputs, outputs, interior, and convexity.  Region-computation
code uses only qualified nouns — `Region_program`, `Region_partition`,
`Region_execution`, and `Region_trace` — for an **output-coordinate partition**.
No unqualified `Region` module will be introduced by future computation work.
When deferred whole-graph fusion gains a dedicated public module, it must be
called `Node_region` (or a more specific `*_node_region`), and must reuse the
existing `Transform.Region` boundary derivation rather than reimplement it.
This keeps current public APIs stable while removing ambiguity from the point
where both concepts meet.
