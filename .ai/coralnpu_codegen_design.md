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

## Register groups

A `Mir_target.View` has a `span` (`Span.t = One | Two | Four | Eight`, RVV's
integer LMUL in `vtype.vlmul` encoding order): a view whose span is not `One`
is a register group, `unit` its lowest member, covering `Span.count`
consecutive units whole (`lo = 0`, `bits` the members' total). A
sub-register view (AArch64 `s`/`d`, x86-64 `xmm` of `ymm`, RVV fractional
LMUL) stays `One` with fewer `bits`. Base alignment is the target's rule on
its register number, not on unit ids (v4 is unit 69).

`View.units` lists the members; `View.overlap` is a shared member unit, and,
between two single-unit views, intersecting bit ranges; `View.same_register`
(base, span and bit range alike) is what a tied operand must be, since a
shared unit alone would make v4 the v4m4 group. The physical verifier's tied
check and the frame's callee-saved write detection use these
(`test/machine_ir/mir_view_test.ml` runs its checks against the base-unit
overlap and the unit-only identity, which fail them). The AArch64 and x86-64
emitters refuse a group.

The physical interpreter keeps state per unit and reads or writes a group
member by member: lane `i` of an `n`-lane group lives in member
`i / (n / count)` at lane `i mod (n / count)`, the lane map `Mir_vsplit`
slices by; a group's frame memory is moved a lane at a time. The checker
keeps a group's holder on its base unit with the number of units it covers,
and any write to a member forgets it.

### Allocating groups (RV32, LMUL 4)

`Rv32.shape` gives sixteen binary32/int32 lanes, or a sixteen-lane mask, the
vector bank at 512 bits; `Rv32_reg.group` builds the view, refusing an
unaligned base. `Mir_scan` takes, for a value wider than one unit, only bases
that are a multiple of its unit count and whose members are all allocatable,
and decides interference by view overlap rather than register-number
equality, so a group and its members exclude each other. Selection makes
such values only when asked (`Rv32_select.program ~register_bytes:64L`); the
portable route keeps sixteen bytes, and its code is unchanged.

The grouped choice has its own pool, `Rv32_regs.Grouped`: a group's copy and
cycle scratch must be groups (v24m4, v28m4), so v24-v31 are not allocated, and
the single-register scratch stays v5 and v6. That leaves five groups (v0, v8,
v12, v16, v20). The reference allocator refuses grouped programs (D12); the
interpreted route is `Mir_model_route.Route.Riscv32_grouped`.

### v0

RVV's mask operand is always v0, so RV32 used to reserve it bank-wide, which
forfeits an m8 group and the peak kernels' B buffer. Now v0 is an ordinary
allocatable register (first in the vector preference order), and the forms
whose sequence writes it in passing (`Rv32_op.vector_scratch`: a compare,
`Vmerge`, `Vinsert`, `Vext` and `Vfmv_f_s` off lane 0) declare it twice: as a
clobber (no value survives them there) and as a
`Mir_target.Constraint.Scratch` (no use or result of the form may be there,
since the sequence overwrites v0 before reading its operands). A clobber alone
is not enough: its fixed interval opens after the uses, so the allocator would
place an operand in v0, which the checker and interpreter accept (they model
the form, not its sequence) and the hardware gets wrong; under qemu that
mutation crashes most Zve32f bundles. Linear scan blocks a scratch view over
the use and result positions; the physical verifier rejects an operand in it.
Masks stay four-lane all-ones/all-zeros vectors, so no form needs a value
fixed in v0.

### Emitting groups (RV32)

Each vector item states the configuration it runs under
(`Rv32_mir_form.requirement`): SEW 32, tail and mask agnostic, `M1` (LMUL 1,
vl 4) for four lanes, `M4` (LMUL 4, vl 16) for sixteen. The module writer
(`Rv32_mir_module.configurations`, a pure function of a block's requirements
and calls) emits `vsetivli` before an item whose requirement is not known to
hold; the state is unknown at a block's start and after a call, and a form
that changes it on its way restores LMUL 1, its own. Portable code needs only
`M1`, so its output is the old one byte for byte. A group's register copy is
`vmv4r.v` (whole registers, no configuration); its spill and reload are
`vse32.v`/`vle32.v` of 64 bytes under `M4`, so they do not depend on VLEN.
The grouped route (`Rv32_mir_route.Registers.Grouped`, also on the Coral
route) is bitwise under qemu at VLEN 128 and 1024 and on MPACT and the RTL
core; a group copy of its first register alone and a group spilled into half
its bytes are caught. No form mixes the two widths in one block today, so the
stale-configuration mutation is checked on `configurations` itself.

Measured on the RTL vector core (`rtl_test.ml`, v0.0.4), LMUL 4 against
LMUL 1, the same plan: relu-then-add 1915 -> 1152 cycles (1.66x), the conv
chain 23903 -> 20966 (1.14x), the matrix products 1.07-1.66x under
`Fp32_ordered`, and 0.92-1.32x under `Fp32_contracted`. The stop rule (under
10% on the pointwise and conv chains: stop at m4) does not trigger. The one
slowdown, linear 9x32 K=64 contracted (34910 -> 38079), is register pressure:
with v24-v31 held as group scratch only five groups are allocatable, and the
kernel spills two groups and copies five inside its loop, where LMUL 1 spills
nothing. Freeing that scratch (a frame slot for the cycle scratch, say) is the
next lever.
