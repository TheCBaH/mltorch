# Region computation — landed design

## Status and scope

This record describes the landed Region-computation implementation.  It is the
authoritative tracked companion to `native_compute_design.md` and
`native4d_design.md`; planning notes under `_ai_/` are execution scaffolding,
not required reading for a fresh checkout.

The implementation gives RMSNorm, LayerNorm, and plain Softmax an
operation-authored `Region_program.t`.  Other operations remain Pixel-authored.
Every logical Kernel value has exactly one `Region_program.t`: a Pixel body is
embedded mechanically with `Region_program.pixel`, preserving the original
expression object.  There is no Kernel sum type and no optional Region
admission/fallback path.

Out of scope: loop IR, tiling, shaped locals, multi-output programs, automatic
Pixel-to-Region discovery, graph fusion, GroupNorm and batch normalization
Region forms, safe/log softmax, SDPA, relaxed rounding, tree/parallel
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

## Numerical policy

Programs retain the existing operation order and `f32` materialization policy.
`Identical` means the original expression and reconstructed/specialized Region
expression are structurally the same under the stated limits.  `Equivalent`
is reserved for transformations whose declared arithmetic or materialization
boundaries change.  `Region_program.reconstructs` remains a general
transformation tool and the tests compare authored RMSNorm, LayerNorm, and
Softmax programs against their test-only legacy scalar oracles.  A malformed
authoritative program is a typed construction/limit error, never a request to
run a second handwritten Pixel algorithm.

## Ownership and routing

`Region_computation` is the operation-facing construction boundary.  It takes
the output ordinal and shape plus role-resolved operands, applies actual
`Kernel.Limits`, and returns a program or a typed error.  It contains the
generic construction helpers and the normalization divisor's bounded shared
calculation; dispatchers contain no operation arithmetic.

Native `Eval_direct` and Native4D `Eval_direct4` route RMSNorm, LayerNorm, and
Softmax through that program and materialize it once per key.  Their Symbolic
counterparts put the exact program in `Stage_program.Stage.computation`.
`Stage_program`, grounding, transform verification, Kernel adaptation, and the
Model Explorer consume this structural program.  Pixel-authored operations
remain on their existing specialized Pixel schedule.

Native4D owns Axis4 legality and parameter adaptation, but delegates numeric
Region construction to the shared Native operation definition.  Its Direct and
Symbolic routes therefore have the same operation form and Region sharing as
Native, without a second numeric kernel.

`Regionizer`, candidate maps, optional `regionized` results, candidate
reconstruction admission, synthetic-value lookup, and production scalar
materialization are removed.  `Legacy_pixel` exists only as the test oracle for
the three migrated operations.

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

For reproducibility rather than as a portability claim, one 20-sample local
run on 2026-09-02 at `(R, K) = (64, 32)` recorded these medians:

| Operation | Kernel ms / words | Native Direct ms / words | Native4D Direct ms / words |
| --- | --- | --- | --- |
| RMSNorm | 1.329 / 1214163 | 1.295 / 1200155 | 1.334 / 1200135 |
| LayerNorm | 2.004 / 1791902 | 1.945 / 1771459 | 1.935 / 1771446 |
| Softmax | 1.126 / 1064345 | 1.100 / 1055974 | 1.337 / 1056010 |

GC words and elapsed time are runtime-sensitive; the benchmark command is the
source of a fresh measurement.  The counter caveat above applies beside every
row of this table.

The separate Pixel no-regression benchmark covers the unchanged Pixel Kernel
loop; its 2048-cell run recorded identity `0.401 ms / 202.271 words` per output
and add `0.730 ms / 312.319 words` per output.

## Validation commands

The closeout was validated with:

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
