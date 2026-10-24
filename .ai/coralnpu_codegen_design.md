# Coral NPU code generation — design record

How the RV32 Machine IR route selects Coral-specific code, and under which
numerical contract. Measurements of the simulators themselves (peak bursts,
DMA, subsystem) are in `coralnpu_simulators_design.md`; this record holds the
compiler's rules. Related: `machine_ir_design.md`, `fp32_simd_design.md`,
`riscv32_process_design.md`.

## Numerical policy: `Fp32_contracted`

The Coral cores have no D extension, so binary64 work is soft-float, and the
fast GEMM forms (`vfmacc`, `.vf` MACs, Zvt tiles) fuse multiply and add. The
policy for them is `Fp32_contracted` (`Ssa_numerics` and `Loop_numerics`,
kept in parity by the bridge suite):

- every kernel binary32, planned exactly as `Fp32_ordered` (same loops, same
  vectorization, same blocking);
- an ordered sum whose term is one product accumulates it with a fused
  multiply-add, in source order from the seed:
  `acc <- fmaf(x, y, acc)`, k ascending, each step rounded once;
- nothing else is fused (no pointwise `a*b + c`), and no sum is reordered.

Contraction scope is a closed variant, `Contraction.t = Dots | Everywhere |
Forbidden`, separate from reassociation (`reassociation_permitted`, only
`Simd_fp32_relaxed`). `contraction_permitted` means `Everywhere` (the
pointwise passes `Ssa_opt_contract`/`Loop_contract`); `Dots` is consumed only
by the Machine IR lowering. The identity string spells it `fma=dots`;
existing policies' identities are unchanged.

Only the Machine IR route runs the policy: `check` refuses it for C, Wasm,
the interpreter and JavaScript, which would compute the ordered sums. The
Loop planner plans it as `Fp32_ordered`.

### Where the fusion happens

`Mir_lower.summary` gives the policy `Mir_planning.Fma.Dots` on any target
(the lowering fuses, so the target's own `fma` flag is irrelevant), with the
`Fused_multiply_add` capability. Under `Dots`, `Mir_lower.program`:

1. runs `Ssa_fuse_dots` over the structured program: a reduction loop (its
   body marks a reduction) carrying an accumulator used once, in
   `acc + x*y` with the product made in the same body, yields
   `fma(x, y, acc)` instead. Blocking and vectorization rewrite an ordered sum
   into such a `for` carrying several accumulators, so fusing only
   `Ordered_sum` misses most of them;
2. lowers with `Ssa_cfg_lower.Accumulate.Fuse_products`: an `Ordered_sum`
   whose term the body defines as one product gets `fma(x, y, acc)` on its
   back edge.

The product instruction stays (dead unless used elsewhere). A target without
an exact fused operation in its selected features refuses (x86-64 at its
default SSE2 feature set: "needs fma").

### The oracle

`Loop_bundle_oracle.run ~accumulate:Loop_sum.Accumulate.Fuse_products`: the
Loop interpreter at binary32 with each single-product sum fused. A bundle's
programs hold expanded sums, so the oracle first recovers them
(`Loop_sum.recover`); without that step nothing fuses. It is cross-checked on
the matrix-product fixtures against `Direct` under `Binary32_sequential`, an
independent evaluator, bitwise, with and without bias.

### Exact binary32 fma

`F32_math.fma32` is the one implementation (Loop, SSA and `Direct` share it):
the binary32 product is exact in binary64, TwoSum splits `p + c` exactly, and
the binary64 sum is rounded to odd before the final rounding to binary32
(53 >= 2 x 24 + 2 bits). Only `Int64` bit operations, so js_of_ocaml gives the
same bits. `Direct.Binary32_sequential` used to round twice (binary64 `fma`,
then binary32), wrong where the binary64 result lands on a binary32 midpoint;
`test/f32_math/fma32_test.ml` holds such triples and an independent
midpoint-decision oracle.

## Calibration

`Rv32_mir_census` counts what a published RV32 artifact emits, statically:
vector instructions by mnemonic, vector configurations, spills, reloads,
rematerializations, frame bytes, helper calls. Its goldens
(`test/machine_rivet_riscv32_mir/census_test.ml`) show a selection change
without a simulator.

`test/machine_rivet_riscv32_coral/rtl_test.ml` measures the matrix-product
fixtures (`Model_test.contraction_fixtures`: Linear and Bmm at the peak RVV
FP32 kernel's C[9,32], K=64, and with M/N/K tails 7/29/63) on the RTL vector
core. Under `Fp32_ordered` the generated Zve32f GEMMs reach 0.3-1.3
operations a cycle (two per multiply-add), against the peak suite's 9.9 burst
at the same shape: the gap register groups and generated microkernels close.
