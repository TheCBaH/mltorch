# Strict vectorization over the Loop IR

Status: Part II of the Wasm/SIMD work. The scalar Wasm backend (see the Wasm
backend design record) is the base; nothing here may change what it computes.
This record fixes the contracts; sections marked *measured* quote the evidence
that decided them.

## Numerical contract (strict mode)

Vectorization changes how many cells are computed at once, never what one cell
computes.

- Working arithmetic is binary64. F32 storage widens on load and narrows on
  store; every `Round_f32` stays an explicit narrow-then-widen. No arithmetic in
  `f32`, no fused multiply-add, no reassociation, no approximate or relaxed
  instruction.
- Lanes are independent output cells. A cell's expression, rounding boundaries
  and reduction order are exactly its scalar program's; a reduction is never
  split across lanes or reordered. Vectorizing the reduction axis is out of
  scope.
- First failure, effects and execution marks keep their scalar order. Anything
  with a failure site, a branch the proofs do not cover, a mark, a local array
  or a scan stays scalar around the vector loop.
- `i64` stays modular and `i32` indices stay in the checked domain; a vector
  loop never computes an index outside the range its scalar loop did.

## Representation (`Loop_vector`)

A *vector program* is a scalar `Loop_program.t` in which some `For` loops are
replaced by vector loops. The scalar program is kept beside it: it is the
fallback for every backend and the reference every check compares with.

A vector loop strip-mines `[lo, hi)` into full vectors of `lanes` consecutive
iterations followed by a scalar remainder:

- `lanes` is the *logical* width: how many cells one vector iteration covers.
  It says nothing about registers. A target maps it to however many physical
  registers it takes (four `f64` lanes are two `f64x2` registers on Wasm).
- The body is straight-line vector statements over typed vector temporaries:
  broadcast of a loop-invariant scalar, contiguous or strided element loads
  (`base + k * stride` for lane `k`, stride 0 is a broadcast), lane-wise
  arithmetic, comparisons into masks, selects, explicit `Round_f32`, and stores.
  There are no physical registers, intrinsics or byte widths in it.
- The remainder is the original scalar loop restricted to the iterations left
  over, so a backend that declines vectors (or a length under one vector) runs
  exactly the scalar program.
- Each vector statement names the scalar statement it came from, so a refusal,
  a failure or a mark count can be traced to its source.

## Legality (`Loop_vector_check`, shared)

A loop is vectorized only when the analysis proves, from the program and not
from emitted text:

1. the body is pure and straight-line: float/int64 assignments and stores of
   pure expressions, no `Fail_if`, `Mark`, `Alloc`, array access, scan
   statement, nested loop or untaken-branch load;
2. every access index is affine in the loop variable (`Loop_linear`) with a
   constant stride, the same cells the scalar iterations touch, so no lane
   reads or writes a cell its scalar iteration would not (memory safety is the
   scalar loop's);
3. no loop-carried dependence: a cell stored in one iteration is not read in
   another (same buffer, different index), and no temporary assigned in the
   body is read before it is assigned in the next iteration;
4. loads and stores of distinct buffers do not overlap. Distinct buffer ids are
   disjoint within one invocation by construction (every operand and the output
   are simultaneously live, so the allocator places them apart); the bundle path
   re-checks the placed byte ranges, and a tensor id is never taken as proof on
   its own. A buffer both read and written in the loop must use the identical
   index.

A refusal is a value with a structured reason, tallied by reason; scalar
retention around a loop is not an unsupported-graph fallback.

## Oracle (`Loop_vector_interp`)

Independent of every backend: a vector loop is expanded into scalar statements
lane by lane (lane `k` of iteration `j` is the scalar body with the loop
variable replaced by `lo + j * lanes + k`) and run by `Loop_interp`. Values,
errors and mark multiplicity must equal the original scalar program's, for
extents 0, 1, `lanes - 1`, `lanes`, `lanes + 1` and non-multiples, offset views
and broadcasts. Malformed vector programs, lane-shape mutations and a dropped
`Round_f32` are rejected or caught by the checks.

## Targets (`Loop_target`)

A target description separates *legality* from *cost*. Legality: which
operations, element types and shapes the target can express (with expansion
into scalar lanes when it cannot). Cost: relative per-operation and per-memory-
access costs, so a profitable plan is chosen without making an unprofitable
legal one illegal. Portable `wasm128` is described first; an installed native
target (the host's 128-bit AArch64 NEON) second.

## Measured baselines (S1)

Linux AArch64 (8 cores, NEON/`asimd`, no SVE), Node 20.19.2, gcc 14.2.0, Clang
19.1.7, one sample per model, warm = median of ten repeats on one instance with
a dirty workspace. Every Wasm run below was bitwise equal to the per-node
reference (`--shadow`) where `--shadow` is noted, and the compile was
`-ffp-contract=off -fno-strict-aliasing`.

| build | mobilenetv2_050 warm ms | fastvit_sa12 warm ms | vector instructions (mobilenetv2_050) | module bytes (mobilenetv2_050) |
|---|---:|---:|---:|---:|
| Clang -O2 scalar Wasm (`-fno-vectorize -fno-slp-vectorize -mno-simd128`) | 81.4 | 1089.5 | 0 | 95,476 |
| Clang -O2 `-msimd128`, vectorizers on (shadow-checked) | 72.0 | 1083.8 | 7,074 | 159,828 |
| Clang -O3 `-msimd128` | 81.0 | 1081.5 | 11,534 | 204,301 |
| gcc -O2 native, vectorizers off | 62.7 | 937.9 | | |
| gcc -O2 native (its default) | 62.4 | 937.1 | | |
| gcc -O3 native (NEON) | 55.0 | 904.3 | | |
| gcc -O3 -march=native | 55.1 | 907.8 | | |

Reading it:

- Compiler auto-vectorization is worth 12% (mobilenetv2_050) and under 1%
  (fastvit_sa12) in Wasm, and 12% / 3.6% natively. Clang's own remarks on the
  mobilenetv2_050 unit: 305 loops vectorized (almost all at width 4), 240 not.
  Of the refusals with a stated reason, 76 are unsafe dependent memory
  operations (it cannot prove two buffers distinct, which this repository can:
  distinct buffers of one invocation are disjoint by construction) and 42 are
  floating-point reductions it will not reorder (which strict mode never
  reorders either, so those stay scalar by design).
- `-O3` bought nothing over `-O2` in Wasm and doubled the module size.
- gcc's default `-O2` already equals its explicitly scalar build: the native
  C baseline in the earlier table is effectively scalar.
- Whatever strict vectorization can add on top of this must come from the
  dense kernels (convolution and matrix multiply), which the compiler's
  loop vectorizer leaves alone, not from the pointwise loops it already
  handles. That is S6's question; S3-S4 build the shared machinery on the
  simple loops first.

## Implemented decisions (S2-S6 first pass)

- **One vector layer, three consumers.** `Loop_vector` (types), `Loop_vector_check`
  (verifier), `Loop_vector_expand` (oracle), `Loop_vectorize` (analysis),
  `Loop_target` (legality and cost) are backend-free and js_of_ocaml-reachable;
  `Loop_wasm_vector` (`f64x2`, two registers per four logical lanes) and the
  vector section of `Loop_c` (GCC/Clang generic vectors) are the only consumers
  that know a register.
- **Nested vector loops.** The unit that vectorizes is a loop whose iterations
  are independent, *including* the loops inside it: a reduction over an inner
  loop runs once, for all lanes in lockstep, each lane's own accumulator chain
  in the scalar order. A vector body is therefore a tree: assignments, stores,
  marks, uniform index temporaries, and `Inner` loops whose bounds do not depend
  on the lane variable. Vector temporaries are mutable (an accumulator is
  assigned again by an inner loop); the verifier checks definite assignment, and
  an inner loop's first assignments do not count after it (it may run zero
  times). A mark inside is bumped once per lane. Loops are tried innermost first;
  a loop that holds a vector loop stays scalar around it.
- **Splats.** A scalar expression independent of the lane variable is broadcast;
  one that also does not reach an inner loop's variable is computed once, before
  the loop, and the others where they are used. A splat may not load from a
  buffer the loop stores.
- **Aliasing.** Distinct buffers of one invocation are disjoint by construction
  (every operand and the output are live together, so the allocator places them
  apart); two accesses of one buffer must be identical.
- **Strided and gathered accesses are legal and costed.** A gather (a weight
  matrix read down a column across lanes) is scalar loads assembled into a
  vector; it is cheaper than it looks because the arithmetic after it is halved.
  The first cost model priced the assembly at two scalar operations per lane and
  rejected the loops that win the most; a measurement with every cost zero
  (`Loop_target.forced`, `--simd-forced`) showed it, and the overhead is now one.
- **The oracle is independent.** It is built from the vector body alone and is
  run against the original scalar program; a dropped inner rounding, a wrong
  stride, and every verifier refusal are tested to be caught.
- **The test corpus is shared.** `test/loop_ir/loop_vector_programs.ml` is run
  through the oracle, SIMD Wasm and vectorized C, bitwise, with and without the
  cost model; every `Loop_check` fixture and the op sweep also run through both
  backends.

## Measured results (strict vectorization)

Same host and method as the S1 table; every row bitwise equal to the per-node
reference (`--shadow`), warm = median of ten repeats, ms.

| route | mobilenetv2_050 | fastvit_sa12 |
|---|---:|---:|
| direct Wasm, scalar | 88.2 | 1126 |
| direct Wasm, leaf loops only | 80.0 | 1104 |
| direct Wasm, nested vector loops, cost model | 52.0 | 670.6 |
| direct Wasm, every legal loop (`--simd-forced`) | 50.5 | not run |
| Clang C->Wasm, scalar | 77-81 | 1090-1119 |
| Clang C->Wasm, auto-SIMD `-O2` | 72.0 | 1083.8 |
| native C gcc -O2 scalar | 62.4 | 937 |
| native C gcc -O3 auto-vectorized | 55.0 | 904 |
| native C, strict vector loops (leaf only, `-O3`) | 49.6 | not run |
| native C, nested vector loops, `-O3` | 132.5 (rejected) | not run |

Nested loops pay on Wasm (the arithmetic of a reduction halves and V8 handles the
gather) and lose 2.7x on native C with GCC generic vectors, so `neon128` declines
them (`Loop_target.inner_loops`). The strict nested Wasm route is 41% faster than
the compiler-vectorized C compiled to Wasm and 1.6x faster than the native
scalar C on mobilenetv2_050; first run (cold, includes V8 tier-up) falls from
235 ms to 118-150 ms.

## Deployment and ISA matrix

- Verified: Wasm SIMD (`f64x2`, standard simd128 only) on Node 20.19.2 and
  headless Chromium; the in-process host picks scalar or SIMD by validating
  `Wasm_features.probe`, never from a version number. All seven CI models are
  bitwise equal to the per-node reference (`make wasm.simd.pt2.runtest`).
- Native C vectors use GCC generic vectors, verified on AArch64 NEON only
  (-O0, -O3 and ASan+UBSan via `make c.runtest.all`). There is no runtime CPU
  dispatch: the vector code is compiled for the build host, so a scalar-only
  CPU needs a scalar build.
- Packing of weights (blocking) is not implemented: the measured nested gather
  already wins (52.0 vs 88.2 ms), so the scratch, prep and cache-identity costs
  are not paid. Revisit only if a model shows gather-bound loops.
- Relaxed SIMD, FMA and F32 arithmetic stay out of scope (strict only).

## Binary32 kernels (pointer)

Everything above is the strict, binary64 contract and it is unchanged: it is the
default and the reference. A kernel the planner vectorizes may instead run in
binary32 under an explicit numerical policy, with sixteen logical lanes, scheduled
sums and optional fused multiply-adds; that has its own contract and oracle, in
the fp32 design record in this directory. The strict shadow commands stay bitwise
against the binary64 reference.
