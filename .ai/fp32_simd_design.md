# Binary32 kernels for SIMD inference; binary64 for general compute

Status: implemented on native C and direct Wasm, promoted to the default of the
generated C/Wasm inference executables. The binary64 reference remains the
default everywhere else and an explicit option (`--reference`) for C and Wasm.

## Contract

Working precision is chosen by role.

- **General compute stays binary64**: the OCaml evaluators, the Loop interpreter,
  generated JavaScript, and every C/Wasm kernel the planner does not vectorize.
  This is the strict contract of the older C, Wasm and vector records,
  unchanged, and it stays under their bitwise shadows.
- **A kernel the planner vectorizes in binary32 computes entirely in binary32**,
  scalar tails and non-vector statements included. Matching the reference's bits
  or its reduction order inside such a kernel is not a requirement; matching its
  *own* oracle bit for bit is, and matching the reference within a frozen
  tolerance.

Precision and schedule are separate axes. A kernel is *ordered* (each output's
operation sequence unchanged) or *relaxed* (reordered sums, fused multiply-add).
Memory safety, exact integer and index arithmetic, shapes, ownership, declared
storage and cast behavior and control dependencies are unaffected: an LSTM
recurrence is not a sum and is never reordered.

## Policy and plan

`Loop_numerics` names three presets and records the permissions each grants.

| Preset | Vectorized kernels | Other kernels | Sums | Contraction |
|---|---|---|---|---|
| `Reference_f64` | f64 | f64 | source order | no |
| `Simd_fp32_ordered` | f32 | f64 | each output's source order | no |
| `Simd_fp32_relaxed` | f32 | f64 | scheduled along their own axis | where the target has it |

Only generated C and Wasm accept a `Simd_fp32_*` policy (`Loop_numerics.check`
is a typed refusal elsewhere). `Loop_plan.resolve` makes the per-kernel decision:
a kernel is binary32 iff the f32 target vectorizes at least one loop (or
schedules one sum) in it and `Loop_numerics.admit` admits it; otherwise it is
planned and emitted exactly as under `Reference_f64`. The policy and the
per-precision kernel and invocation counts (with refusals by reason) are part of
the artifact: `Loop_bundle_c` and `Loop_bundle_wasm` record them, a host prints
them, and the Wasm manifest names the policy and the working precisions.

Admission: every float read of a payload with a decode (`f32`, `f16`, `bf16`,
`bool`, and `f64`, `i32`, `i64` rounded once), an int64 or an index converted to a
float once. Nothing in the CI models is refused.

## The binary32 oracle

`Loop_interp.run ~precision:F32` evaluates every float operation in binary64 and
rounds once to binary32 (`+ - * /` and `sqrt` of binary32 operands are then
correctly rounded: binary64 has more than `2p+2` bits). Constants are pre-rounded;
conversions from int64 go through `Loop_numerics.round32_of_i64` (a sticky bit
keeps a single rounding); transcendentals are the binary64 function on the
widened argument rounded once; `Erf` is `erf32`, the polynomial with a rounding
per operation. A fused multiply-add is `Loop_numerics.fma32` (exact product,
TwoSum, round to odd), checked bit for bit against the host's `fmaf` over a
large generated sample that includes constructed double-rounding cases and shown
able to fail (the `round32 (Float.fma ..)` shortcut differs on the same sample).

For a vector kernel the oracle is the plan's vector program expanded lane by lane
and run by the interpreter (`Loop_plan.oracle`): independent of both backends.
`Loop_interp.run ~fused:false` is the unfused reading a relaxed madd also admits.

## Vector layer in binary32

`Loop_target.f32` plans sixteen logical lanes (four 128-bit registers); C uses
GCC/Clang `v16sf`, Wasm four `f32x4` registers. Sixteen was measured: eight lanes
leave the latency of each cell's own sequential sum exposed (native: 1.5x over
scalar binary64 on a dense matvec against 2.9x at sixteen), thirty-two gives
nothing more (mobilenetv2_050, native C warm: 8 lanes 42 ms, 16 lanes 28 ms,
32 lanes 30 ms; Wasm: 8 and 16 equal, 32 slower). Nested vector loops (an inner
reduction per lane) are allowed at binary32 on NEON and not at binary64, where
they lose.

Small helpers are forced inline (`always_inline`): without it the compiler left
the fused-multiply-add helper as a call passing 64-byte vectors through memory
across 49 sites, which cost 40% on a real model.

## Sums

`Loop_stmt.Reduce_sum` keeps a float sum as one operation (seed, bounds,
accumulator, body, term); `Loop_sum.expand` is exactly the loop the lowering has
always produced, and every consumer reads the expansion. The optimizer runs on
expanded programs; the planner then *recovers* sums from the optimized shape
(`Loop_sum.recover`: a loop opening with a `Reduction` mark and ending with
`acc <- acc + e`, seeded just before it), because guards and index collapsing
must run first for bodies to be admissible. Recovery is checked for every plan of
every suite: the structured lowering expands to the plain one, and a recovered
program expands back to its source exactly.

Under `Simd_fp32_relaxed`, a recovered sum no enclosing loop's lanes took is
scheduled along its own axis (`Loop_vector.Reduction`) when its body is empty, its
bounds constant, its term has a contiguous load and it holds at least four
vectors: `parts` accumulator vectors, rounds of `parts` vectors, leftover
vectors, a fixed adjacent-pair tree over accumulators then lanes, a sequential
tail, and the seed (the definition is in the type's documentation, and
`Loop_vector_expand` spells it out as scalar statements). Verification: bitwise
against the oracle on specials and on moderate values, exact on small integers
against the plain sequential sum (so no term is lost or repeated whatever the
tree), and mutations (a dropped tail, a duplicated leftover vector, a different
lane tree, a dropped Wasm tail) each turn a suite red.

## Contraction

`Loop_expr.Fma` is a guaranteed fused operation; `Loop_contract` rewrites
`a + x * y` on the vector program after sums are scheduled, only where the target
has it (`Loop_target.fma`: native NEON yes, standard Wasm SIMD no). C emits
straight-line `fmaf` per lane, which the compiler turns into `fmla` with the
vector in registers (a rolled loop or a union of NEON quarters round-tripped
through the stack and measured slower). Fusing a scheduled sum's accumulate is a
separate permission (`fuse_reductions`) that measured slower and is off.

Relaxed SIMD (`f32x4.relaxed_madd`, feature `relaxed-simd`, off by default in
node 20) is a separate target, `Loop_target.wasm128_relaxed`, planned only when
`Wasm_node.supports Relaxed_simd` validates the real probe module (a host that
cannot plans standard SIMD). Its results are the fused or the unfused answer, so
its check is cell-wise membership in the two oracles, not equality.

## Measured results

Native C (aarch64 NEON, gcc 14.2 `-O2 -ffp-contract=off`, one thread), warm
latency, binary64 vector baseline against the default policy, every output within
tolerance of the binary64 reference and the release ranking intact:

| model | f64 ms | fp32 ms | speedup | binary32 kernels |
|---|---:|---:|---:|---:|
| test_convnext2 | 24.3 | 12.4 | 2.0x | 56/69 |
| mobilenetv3_small_050 | 13.9 | 7.4 | 1.9x | 108/147 |
| mobilenetv2_050 | 58.8 | 28.2 | 2.1x | 85/123 |
| regnetx_002 | 115.0 | 57.9 | 2.0x | 47/77 |
| efficientnet_b0 | 239.1 | 111.2 | 2.1x | 144/204 |
| edgenext_xx_small | 91.2 | 50.5 | 1.8x | 133/185 |
| csatv2 | 116.0 | 53.6 | 2.2x | 140/206 |
| fastvit_sa12 | 937 | 368 | 2.5x | 97/132 |
| mvitv2_tiny | 2543 | 1053 | 2.4x | 160/299 |

Direct Wasm under node 20 (warm run): mobilenetv2_050 55.5 ms strict f64 SIMD to
38.0 ms; fastvit_sa12 706 ms to 532 ms. Relaxed `madd` matches standard SIMD
there (V8 fuses; no latency change).

Dense microkernels (`make fp32.bench`, native, against scalar binary64): matvec
and 1x1-convolution shapes 2.0-2.9x ordered, the same relaxed; a 4096-term dot
6.3x relaxed (error against binary64 6e-5, the sequential binary32 sum 6.7e-4).
Pointwise loops are a loss for vector code in both precisions on this host.

The same programs on direct Wasm (node 20, aarch64, `make fp32.bench.wasm`, against
scalar binary64): 1x1-convolution, matvec and matmul shapes 7.0-8.4x with
binary32 SIMD against 2.4-3.1x for strict binary64 SIMD; a 4096-term dot 6.4x
relaxed (binary64 SIMD cannot vectorize it, 1.0x); pointwise 4.6x; weights in
`[N, K]` layout 2.7x. Relaxed madd is within 5% of standard SIMD. The whole-model
gains are smaller than these kernels' (1.3-1.45x) because most of a model's time is
not in vectorized dense loops.

Numerical acceptance, frozen from the table above and the sweeps: every graph
output satisfies `|actual - reference| <= 1e-4 + 1e-4 |reference|`, nonfinite
cells match by kind, ranking is the release's; observed worst normalized error
(max abs over the output's scale) is 1.9e-6 and worst absolute error 2.8e-5.
The reference path is the binary64 per-node evaluator.

## Register blocking over rows

`Loop_block` unroll-and-jams a scalar loop with constant bounds that directly
holds a vector loop whose body carries an accumulator (a dense kernel: rows
around output channels around a sum). `Loop_target.row_block` rows run in one
vector iteration; an inner loop all rows have becomes one loop whose body holds
each row's statements in turn, so the rows' multiply-add chains overlap and a
load they share is issued once. Rows left over run unblocked. Each output cell
keeps its own operation order, so the pass is legal under every policy and the
blocked kernel is bitwise the unblocked one (tested: C and Wasm equal the plan's
oracle, and the blocked oracle equals the unblocked plan's).

Independence of the rows is proved, not assumed: the blocked vector loop goes
through `Loop_vector_check`, whose store rules now accept two accesses to one
buffer when they have the same stride and offsets differing by a constant `d`
with `d` not a multiple of the stride within the iteration count (their cells
never meet; an offset mentioning an inner loop's variable is not proved). Rows
whose stores overlap fail it and the loop stays as it was (a test turns red when
that proof is weakened).

Measured, aarch64 NEON, default relaxed policy: a 196x96x96 pointwise convolution
2.9 -> 0.96 ms, a 16-row matmul 3.1x, a convolution in `[N,K]` weight layout
1.3-1.4x; whole models (two rows, native C) 28.2 -> 25.4 ms (mobilenetv2_050),
7.3 -> 6.8 (mobilenetv3_small_050), 55 -> 48 (regnetx_002); two and four rows are
equal on models. `neon128` blocks two rows. `wasm128` plans none: dense kernels
gain about 20% under V8 (`make fp32.bench.wasm`, `--row-block=N` on the model
runners) but whole models did not move, so the default stays off.

Two measured limits. First, GCC's SLP vectorizer, given sixteen scalar `fmaf`
per `v16sf`, stops vectorizing once several accumulators are live (four rows of a
96x96 shape fell to scalar code, 3x slower); a NEON `vfmaq_f32` per quarter
through a union fixed a micro kernel but made whole models 1.4x slower, so
`vs_fma` is unchanged. Second, the headroom this leaves is in the C vector type
itself: a hand-written matmul on native `float32x4_t` quarters runs 17 GMAC/s
unblocked and about 50 blocked, against 5 and 8 for generic 64-byte vectors, so
representing the sixteen lanes as four native vectors in the emitted C is the
next lever (not done: it touches every vector helper and operator).

## Decisions taken on measurement

- Sixteen lanes; unfused sum accumulates; contraction on.
- **No weight packing.** A convolution's own weight layout (`[N, K]`, contiguous
  along the sum) makes an output-axis vector gather; under the ordered policy that
  costs a 1x1-convolution shape 1.32x against 2.11x for the layout the vector
  likes, but under the relaxed policy the planner already schedules the sum along
  K and the gap is 6% (1.98x against 2.10x). Not worth setup time and storage.
- Partial scalar accumulators were not built: they would change an unvectorized
  kernel's arithmetic, which the policy keeps binary64.
- Constant folding runs before precision is resolved and evaluates in binary64;
  the folded constant rounds once at emission.
- Local arrays of a binary32 C kernel are floats, two to a scratch double.

## Commands

| Command | What |
|---|---|
| `make runtest` | the unit suites: oracle, C corpus, op sweep, structured sums |
| `make wasm.runtest` | the Wasm op table (173 ops against JS references, plus relaxed SIMD), the Wasm binary32 corpus under node |
| `make jsoo.inline-runtest` | the same expect suites under node (the numerics, oracle and sums are js_of_ocaml-reachable) |
| `make fp32.bench`, `make fp32.bench.wasm` | binary64 against binary32 dense kernels on native C and on direct Wasm under node, raw samples as JSON (`_build/fp32-bench-*.jsonl`, kept as CI artifacts), results verified at their own precision (the relaxed madd as the fused or unfused answer), `--selftest` proves the check can fail; the exit status never depends on timing |
| `make c.pt2.perf`, `make wasm.pt2.perf` | the default policy on every CI model within the frozen tolerance, coverage printed |
| `make c.pt2.runtest`, `make wasm.pt2.runtest` | the strict bitwise gates, on `--reference` |

`loop_c_pt2` and `loop_wasm_pt2` default to the performance path; `--reference`
selects the binary64 scalar reference, `--numerics=NAME` any policy,
`--shadow-numeric` the tolerance check, `--row-block=N` the rows per blocked iteration, and `--shadow` (bitwise) is refused for a
binary32 policy.

## Limits

GPU execution, threading, dynamic shapes and general scan parallelization are
out of scope. The browser host keeps its own defaults until its checks are run
with the new policy. Wasm gains are smaller than native (V8 does not fuse across
the engine's tiers the way the C compiler does). Models with many small or
layout-bound loops (mvitv2_tiny: 160 of 299 kernels vectorize) gain less.
