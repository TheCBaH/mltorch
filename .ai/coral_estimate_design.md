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

`coralview` (devcontainer pin 7db47b80) reads the VME core's RTL traces:
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

- Subsystem calibration (its own rates, DMA and overlay costs, its emulator
  speed with 4 threads).
- Per-class comparison: the census's predicted cycles per class against the
  trace's per-instruction-class cycles for each case (the Pre-LN + RoPE layer,
  15% over, first), then priors or features from the measured costs.
- Whole-model prediction through the route without simulation (`prepare`
  builds the images, so placements are real) and its emulator hours.
