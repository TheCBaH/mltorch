# Coral cycle estimate and emulator model

Predicts what a graph costs on a Coral core in the core's own clocks (mcycle
around each kernel call, the unit the route measures), from the computation,
without running it; and what running it on an emulator would cost in host
time. The point: a large graph's cycles are predicted from small measured
kernels, and whether to run it on RTL is a computed answer.

## The estimate

- **Granularity.** One invocation is one planned SSA program, exactly the one
  the route lowers (`Mir_model.prepare ~on_program`). A bundle is the sum of
  its invocations, each its own image and run, so a graph's prediction is the
  sum of its parts' — composability is a property checked, not assumed.
- **Census** (`Coral_estimate.Coral_census`). Every operation of a body is
  multiplied by its loops' trip counts, evaluated from the program's own index
  constants and induction values; a loop or `if` that depends on data is
  counted once (half per branch) and reported. Classes are what costs
  differently: scalar and vector binary32 operations, divides, each math
  function separately (a vector exp is 580 cycles a register group, a vector
  sqrt 11), loads and stores by memory region and stride, scratch cells, loop
  trips, int and index work. Vector classes count 128-bit register groups.
- **Placement is the route's, not assumed.** The DTCM is filled first-fit in
  slot order after the runtime's own data and the rest goes to RAM
  (`Coral_layout`), so the same program costs differently with a big operand.
  The route reports each image's placement (`Rv32_mir_coral.route
  ~on_image`: Machine IR region → memory role) and `on_program` gives each SSA
  buffer's Machine IR region; the census prices every load, store and kernel
  operand by where it actually is. Measured: a vector add costs 3,127 cycles
  in DTCM and 5,671 in testbench RAM; ignoring it put the f32 contractions
  whose operands overflow the 32 KiB DTCM 30% low.
- **Hand kernels** are priced by their own features (`Coral_model`): the
  matrix engine's contraction by 16 x 16 tiles (C's region), A rows times k
  (A's) and B columns times k (B's), split by format and by plain or packed
  function (the route's rule, `Zvt_packed_contraction.fast`, supplied by the
  caller so the estimate does not depend on the kernels); the fused attention
  by query blocks, key blocks, padded score cells and those times d, by
  operand region.
- **Rates** are cycles per unit of feature, fitted by non-negative least
  squares on relative error (`Coral_fit`, Lawson-Hanson), so a 2,000-cycle
  kernel weighs as much as a 4 M one. The memory classes rise together in
  every case, so the fit alone cannot tell them apart; each is pulled toward
  the cost the waveforms show (a prior row of weight two: loads 4 cycles from
  the TCM and 7 from the testbench RAM, stores 3 and 8, vector loads 4 and 7,
  vector stores 4 and 8, read with `coralview commits --profile`).
- **Vector operations overlap external vector accesses.** The vector unit
  runs an operation while earlier loads and stores are in flight, so in one
  innermost loop body the shorter of the operations' cycles and the
  RAM/SRAM accesses' cycles is partly hidden by the longer. The census keeps
  each such body's counts (`Coral_census.t.bodies`, trips included, one entry
  per distinct body); `Coral_model.features ~rates` adds `overlap`, minus
  `sum_bodies min(compute, memory)` priced at those rates, and its rate is
  the hidden fraction (0.72). Because the feature is made from rates the fit is
  a fixed point (fit without it, make it from those rates, refit). Only
  accesses outside the DTCM count: `coralview cost` on `div` against `add` of
  the same kernel shows the divider's extra cycles taken out of the external
  loads' stall by about three fifths and out of the DTCM loads' by about an
  eighth, and a flat per-operation rate had left the vector divide 13% under
  in DTCM and 13% over from RAM. The scalar divide shows no overlap (11.9
  cycles an element at both sizes). The in-order scalar core is not modelled
  this way, and only innermost bodies are paired, so work split across
  loops of one nest is not.
- **The fused attention stages its rows.** Q, K and V are copied into the
  DTCM before they are packed; from RAM that is strided vector loads of about
  140 cycles each (`vlse32.v` in the profile), so the attention's features
  include the staged Q and K, V elements by region.

## Data and validation

`test/coral_estimate_rtl` (gated, `make coral.estimate.calibrate`) runs the
cases of `test/coral_estimate_cases` on the VME core's RTL simulator and
records each invocation's cycles, instructions retired and the buffers placed
outside the DTCM; its golden is the dataset. The cases run as parallel forked
workers under the CPU pool (below): about 120 runs in under 7 minutes.
`test/coral_estimate` (hermetic) re-plans every case, counts it with the
recorded placement, fits on the fit set and reports each case's prediction.
Cycle counts are reproducible: 28 of 28 cases gave identical cycles run
sequentially and seven in parallel; wall times differed by 5–20%.

First results on the VME core (fit set: ~110 kernels; check set held out):

| | error |
|---|---|
| math, elementwise, reductions, transposes | mostly within 1–4% |
| f32 matrix-engine contractions (incl. held-out 32x96x48) | within 0.3% |
| bf16 paired contractions | within 9% (21% one RAM case) |
| fused attention (incl. held-out 41x61) | within 10% |
| decomposed attention layer: 14 invocations summed | +1.4% |
| fused attention layers (norms, RoPE, flags) | −6.3% to +15.4% (Pre LN + RoPE the outlier) |
| fastvit's layer shape (49 x 512, 16 heads; measured by the manual target) | +0.5% |

The rates the estimate uses are committed (`Coral_rates.vme`) and the suite
fails when a fresh fit of the data no longer gives them. fastvit's shape was
first predicted 64% low: the route takes the packed bfloat16 contraction only
when its panels also fit the five eighths of the DTCM all functions' panels
share (20 KiB), so fastvit's 512 x 512 weights run on the plain function,
which the fit had no case of; with that rule and four plain-function cases
the prediction is within 0.2%. One of those cases (49 x 512, k 64) shares
fastvit's rows and columns, so this is an extrapolation in k, not in shape.

Validation strengthened 2026-11-08, after the transformer kernel changes:

- The rates/fit comparison is an inline unit assertion, not an expect snapshot
  whose promoted output can accept a mismatch. All committed rates were
  synchronized with the current fit; `class.iteration` had remained 0.948 when
  its fit was zero. This is consistency repair, not improved model accuracy.
- The per-case diagnostic reports the committed rates, including their use to
  calculate the overlap feature, rather than an uncommitted fresh fit.
- `test/coral_estimate/validation_test.ml` asserts per-case error limits on
  eight measured held-out cases (2--30%, stated individually). Zeroing the
  vector-exp cost is caught by the held-out gate. These limits describe coarse
  cycle estimation; they do not validate ranking close optimization variants.
- Every calibration/check case must have a positive measurement for each
  planned invocation or an explicitly inventoried RTL refusal. Three softmax
  cases exceed the VME core's 8 KiB ITCM (11,352 bytes required). They are not
  silently missing samples or accuracy successes.
- The layer/RMS norm cases are outside the accuracy gate: ordered reduction
  lane extracts remain expensive SSA features although RV32 recognizes their
  chain as `vfredosum`; committed predictions err by +590.6%/+507.2%.
  The decomposed layer also stays outside it: 1,024 data-dependent decisions
  are counted heuristically. Its diagnostic error is -22.2%.

The table above records the original investigation, not current accuracy.
Current committed-rate held-out errors include -9.6% for SDPA, +25.2% for the
pre-LayerNorm/RoPE layer, and +0.6% for the separately measured fastvit shape.
No new RTL measurements or rate tuning to the held-out set were used for this
validation repair. Next, freeze predictions for matched configuration kernel
pairs before running RTL; use their discrepancies to identify missing cost
information rather than immediately refitting rates.

Paired validation added 2026-11-08 (`test/coral_estimate_validation/probe.ml`,
manual, outside `runtest`). It prepares every image and writes every prediction
before any simulator run. Each case records the actual placement's census and
features, rates, configuration/artifact identity, synthetic input bits, and the
entry-call measurement; each output is checked against the binary32 oracle.
Existing evidence directories/files are never overwritten. Both pairs use
`vme-core-mini-axi`, single registers, and ordered f32: the configuration and
register route the existing rates were calibrated for. No refit is performed.

The first eight runs (two shapes, unroll 1/2 and Reference/Fast sequences) were
all bitwise. Predictions preceded the measurements:

| pair | predicted speedup | measured speedup | measured cycles |
|---|---:|---:|---:|
| add 2x160, unroll 1 -> 2 | 0.847x | 1.046x | 909 -> 869 |
| add 1x67, unroll 1 -> 2 | 0.879x | 1.026x | 311 -> 303 |
| GELU/mul 2x160, Reference -> Fast | 1.000x | 2.485x | 54,156 -> 21,796 |
| GELU/mul 1x67, Reference -> Fast | 1.000x | 2.376x | 12,612 -> 5,308 |

For add 2x160 the two censuses differ by ten extra `int_op` and ten fewer
iterations; all vector compute/memory counts are identical. The fitted
iteration cost is zero and int_op is 14.456, so the prediction rises by exactly
144.56 cycles. This does not account for later pointer simplification or the
interleave's overlap benefit. Reference/Fast have byte-identical feature files:
the changed math sequence is a choice made after the counted SSA. Absolute
errors for Fast are +134.8% (2x160) and +146.0% (1x67). These pairs are evidence
of model gaps, not new accuracy successes or training samples.

Reproduce with `CORALNPU_DATA=<archive> opam exec -- dune exec
test/coral_estimate_validation/probe.exe -- --output <new-directory>
--revision <source-label> --run`. Omit `--run` to freeze predictions without
simulation. Research artifacts are under
`ai/coral_estimator_validation/pairs_2026-11-08_vme/`.
The eight measurements reproduced exactly after improving artifact
serialization (`pairs_2026-11-08_vme_serialized/`); the replay tests the recorder,
not independent model accuracy.

bfloat16 storage pairs (`--experiment bf16`, 2026-11-08, after the vector
bfloat16 store): bf16 inputs with binary32 results, then the same with
bfloat16 results, residual add and RMS norm at 2x160 and 1x67, all bitwise,
predictions frozen first:

| case | predicted | measured | error |
|---|---:|---:|---:|
| add 2x160, bf16 in | 821 | 1,451 | -43.4% |
| add 2x160, bf16 in and out | 1,385 | 3,698 | -62.5% |
| RMS 2x160, bf16 in | 7,738 | 4,751 | +62.9% |
| RMS 2x160, bf16 in and out | 8,302 | 6,998 | +18.6% |
| add 1x67, bf16 in | 214 | 439 | -51.1% |
| add 1x67, bf16 in and out | 352 | 1,041 | -66.2% |
| RMS 1x67, bf16 in | 1,706 | 1,230 | +38.7% |
| RMS 1x67, bf16 in and out | 1,844 | 1,832 | +0.6% |

The bfloat16 result adds one invocation (the narrowing `To_copy`), and it
measures the same for both operations: 2,247 cycles at 2x160 and 602 at 1x67
(7.0 and 9.0 an element at LMUL 1), against a predicted 564 and 138. The
census prices a vector access by the binary32 value's register groups and
ignores the encode and the decode: a bfloat16 store (int32 rounding, the
canonical-NaN compare and merge through v0, two narrowings joined by a slide,
and the `vsetivli` changes around each e16 form) counts as one binary32
store, and a bfloat16 load (an e16 load, `vzext`, `vsll`, with their
configuration changes) as one binary32 load, which is most of the bf16-input
add's -43%. RMS norm's overprediction is the ordered-reduction gap above (the
extract chain priced, one `vfredosum` emitted); its +0.6% at 1x67 is two
errors cancelling, not accuracy. The repair this asks for is counting, not
rates: the census should count an access's encode and decode as the vector
operations and configuration changes they lower to, then be validated on
held-out shapes (1x33, GELU) rather than these pairs. Evidence:
`ai/coral_estimator_validation/bf16_2026-11-08_vme/`.

Pipeline audit: the observed SSA is after `Ssa_plan` vectorization, ordered-sum
scheduling, row collapse, unroll/interleave, and prepacking. There is no separate
Coral-only loop vectorizer after that observation. Substantial transformations
do remain after it:

- `Mir_lower.program`: optional generic dot fusion and Reference/Fast math
  expansion. The public `on_program` callback runs after compilation but its
  payload is the SSA saved before this lowering, not a later representation.
- `Rv32_select.program`: generic MIR vector splitting, address hoisting/carried
  pointers (`Mir_offsets`), index narrowing, CSE, and RV32 recognition of a full
  ordered extract/add chain as one `Vfredosum`. These also serve the RV32/qemu
  route; they are target-specific, not Coral-only.
- `Rv32_mir_route.publish`: selected-instruction sink scheduling, allocation,
  spill/copy/frame realization. Emission also inserts vector configurations and
  expands forms and far branches.
- `Rv32_mir_coral.image`: Coral-specific vector/matrix-engine helper selection,
  packed/plain and staged-B choices using the shared DTCM budget, optional DMA
  transfer plans, and subsystem-only ITCM overflow outlining/overlays. The
  standalone VME experiments above do not enable DMA or subsystem overlays.

An SSA estimator remains viable, but a census alone cannot price all of these
choices. The existing contraction/attention descriptor models already cover
some late helper realization. Prefer a shared read-only realization summary
from the compiler/route for remaining choices over duplicating backend
recognition logic in the estimator. Moving the census to generic MIR alone
would still miss RV32 recognition, allocation, and Coral helper selection.

Proposed next boundary: finalize all cost-relevant SSA transformations before
estimation. Keep generic `Contract`/`Attend` semantics but make their selected
execution plan explicit at that stage: vector/matrix engine, kernel variant,
tiles, packing, scratch/staging and transfer decisions. An explicit ordered
vector-reduction operation and recorded math sequence/grouping would eliminate
the concrete hidden choices found in the paired experiments. Reuse compiler
selection rather than duplicate it inside the estimator. Numerical policies
still constrain which choices are legal; a cheaper implementation cannot
silently change the contraction or reduction contract.

Backend bookkeeping is deterministic and must be priced systematically:
allocation's spill/reload traffic, save/restore, offset materialization,
alignment, stack placement/footprint, vector configuration and final form
expansion. Frame layout does not justify unexplained error. Use a read-only
backend cost summary for effects not yet derivable from final SSA. If physical
resource checks force another high-level implementation, refuse or replan and
invalidate the old estimate; do not silently replace the selected plan. These
SSA/API changes are proposed work, not part of the initial validation commits.

## Emulator model

Per simulator binary: `wall = start + cycles / rate`, and the CPUs one run
keeps busy — the Verilator build's `--threads` from the manifest
(`Coral_profile.threads`: 4 for the subsystems, 1 for the standalone cores
and MPACT). Measured on the VME core RTL: start about 1 s, 15,100 cycles/s
alone and 14,300 with seven or eight runs in parallel, individual runs within
about 35% of the line. The manifests give no core frequency (the standalone
cores' clock is a simulation time base; the subsystems record a 30 or 50 MHz
build setting, not a measured one), so a slowdown is stated against a
frequency the caller assumes: 15,100 cycles/s is about 2,000 times slower than
30 MHz.

**The CPU pool** (`Coral_cpus`): every simulator run holds as many slots of a
host-wide pool (lock files under `/tmp/mltorch-coral-cpus-<uid>`, one per CPU,
`MLTORCH_CORAL_CPUS` to override) as its threads, for as long as it runs, so
parallel test runners and builds cannot oversubscribe the host: on eight CPUs,
two subsystem simulators or eight standalone cores at once. A slot is a POSIX
record lock, released when its process dies.

## Waveforms

`coralview` (devcontainer pin a746f9ae) reads the VME core's RTL traces:
`coralview plan --platform vme-core-mini-axi --model commits --model stalls
--model memory --model tiles --catalog <release>/catalogs/
vme_core_mini_axi_sim.catalog.fst` selects 118 signals, and the calibration
suite traces one case with `CORAL_ESTIMATE_ONLY=<case>
CORAL_ESTIMATE_FLAGS="--trace_file=<dir>/{case}.fst --trace_select=<plan>/select"`
(88 KB for a 7.5 k-cycle run, 3.4 MB for a fused layer). `commits
--format=jsonl` gives every retired instruction's word and the cycles charged
to it; summed between the runtime's two `mcycle` reads around the kernel call
they are exactly the measured kernel cycles (3,127; 5,671; 436,619 checked),
so a case's cycles split by instruction class on the hardware itself.
Measured per instruction: vle32.v 3.3 cycles from the TCM and 6.3 from the
testbench RAM, vse32.v 2.3 and 6.3, vector arithmetic 1.8, vfdiv 59, fdiv 17,
fsqrt 18, scalar stores to RAM 8.8, the attention's strided staging loads from
RAM 95. The installed `rivet` must stay pinned to `vendored/rivet`
(`make rivet.install` after installing coralview, whose opam file re-pins it).

## Open

- Overlap on the subsystem cores and for other operations: only the vector divide and exp from RAM calibrate it so far.
- Subsystem calibration (its own rates, DMA and overlay costs, its emulator
  speed with 4 threads).
- Per-class comparison: the census's predicted cycles per class against the
  trace's per-instruction-class cycles for each case (the Pre-LN + RoPE layer,
  15% over, first), then priors or features from the measured costs.
- Whole-model prediction through the route without simulation (`prepare`
  builds the images, so placements are real) and its emulator hours.

**First SSA planning increment implemented (2026-11-08).** Contraction
engines are now selected in `Ssa_contract_plan`, after SSA prepacking and before
MIR lowering, via a custom route's final planning callback. The estimator's
`Contract_call` census retains the recorded engine; its cost model prices that
choice and only uses `Core` to choose for legacy unfinalized SSA. Hermetic
calibration/check planning uses the same final SSA pass with VME capabilities.
No estimator rates or calibration/held-out datasets were changed for this
increment.

The selected engine is part of canonical SSA text and the checked compilation
summary. RV32 helper construction honors it, refusing an unavailable matrix
realization, and Coral validates it against the image's capabilities. Generic
SSA/MIR and qemu tests cover the plan; focused VME matrix/vector and MPACT
vector runs are bitwise. Four mutations were seen red: SSA reselection,
estimator reselection, backend loss of the selected engine, and omission of
the SSA/summary agreement check.

This is engine selection, not complete kernel realization planning. Packed
versus plain, staging and shared DTCM budgets are still late choices. Nor does
this claim to fix the paired experiments' Fast math and unroll accuracy gaps.
Those remain recorded validation failures for subsequent increments.

Engine-planning performance check: the retained legacy route (SSA finalizer
set to identity) and the recorded-choice route were compared on all five
contraction fixtures with both vector and matrix implementations. All twenty
RTL runs were bitwise; each of the ten pairs had byte-identical ELFs and
identical entry-call cycles. This is a hard inline-unit invariant, not a
promotable cycle snapshot. The old VME fixture golden was stale (for example,
linear 9x32 k64 recorded 13,694/4,895 while both paths now measure
13,643/4,844). Five rows in that golden were refreshed after the counterfactual
comparison. This is baseline maintenance, not a performance gain from moving
the engine choice. Full repository and JavaScript inline checks also pass.
