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

## Contractions

`lib/contraction` describes a matrix product as data: batch, M, N, K
(`int64`), each operand's element strides from its first cell (A: batch,
row, k; B: batch, column, k; C: batch, row, column), the element format
(binary32 only so far), the seed (`Zero` or the output's own value) and the
numerical requirement: `Ordered` (product and sum each rounded, k ascending)
or `Fused_sequential` (`fmaf` per step, k ascending). `Contraction.eval` is the
reference for both; it has no SSA, Rivet or tensor dependency.

`Ssa_contract` recognizes products in the scalar binary32 form a plan starts
from (`Ssa_plan.binary32_scalar`: the common passes, guards proved away, then
the exact narrowing), before vectorization and blocking reshape the loops: an
ordered sum from 0 to a constant, seeded with +0, whose body multiplies one
in-bounds binary32 load of A by one of B, each indexed by zeros, the sum's
variable and the variables of the enclosing unit-step loops. A variable on A
only is the row, on B only the column, on both the batch (at most one each);
strides come from the buffers' row-major extents. Every other ordered sum is
refused with a reason (`not_product` for softmax and norms, `bounds` for a
convolution's window), and `--contractions` in the RV32 census binary lists
both per model.

The rewrite (`Ssa_contract.program`) puts each product into a fresh binary32
scratch buffer `[1, 1, 1, batch, m, n]` through one `Ssa_op.Contract`, placed
just before the outermost loop its variables belong to and threaded on that
loop's effect, and replaces the sum by an in-bounds load of the scratch;
whatever the loops did with the sum (a bias, the store) stays. The SSA
verifier checks the operands' formats and that every element a descriptor
names lies inside its buffer; the SSA interpreters run it through
`Contraction.eval`. On the matrix-product fixtures the rewritten programs are
bitwise the originals under `Ordered` on both interpreters, and under
`Fused_sequential` equal the contracted oracle; a transposed A, a K one short
and the wrong requirement are caught. The C, Wasm and JavaScript emitters
refuse a `Contract`, and so does the Machine IR lowering until the helper
call exists.

## Contractions as helper calls

With `Mir_model.prepare ~contractions:true` (`Ssa_plan.resolve
~contractions`), a binary32 plan rewrites each recognized product into a
`Contract` on the narrowed scalar form, before vectorization: ordered under
`Fp32_ordered`, `Fused_sequential` under `Fp32_contracted`. Everything else
in the kernel (a bias, the store) is planned as before, vectorized where it
was.

`Mir_lower` lowers a `Contract` to a call of a helper `Mir_contract`
declares: its name is the descriptor (`Contraction.name`, from which every
consumer rebuilds it), its arguments the addresses of A, B and C, no result,
`Reads_writes`, version 1, ids from 16. The interpreters' model
(`Mir_contract_model`) runs `Contraction.eval` on the cells the pointers
address, so the generic and physical routes need nothing else. The artifact
binds such a helper to a `Generated_function` symbol
(`Contraction.symbol`, a digest of the name); the RV32 qemu and Coral routes
generate it (`lib/coralnpu_kernels`, `Rvv_contraction`) and link it beside
the program's module, and an artifact that calls one takes the vector
profile. AArch64 and x86-64 have no generator.

The generated function (ILP32F; `a0`-`a2` the operand addresses, `0` in
`a0`) is specialized to its descriptor: batch, then 32-column tiles under
e32 m8 (the last one narrower), then row groups, each output's sum k
ascending from +0. Fused: three rows in v8, v16, v24 with `vfmacc.vf`;
ordered: two rows with `vfmul.vf` into v24 then `vfadd.vv`. Per k step the
rows' A values load by `flw` before B's row (`vle32.v` contiguous, else
`vlse32.v`); C stores by `vse32.v`/`vsse32.v`. It saves `s0`-`s3`, uses only
caller-saved registers otherwise, and leaves the vector configuration
undefined, which the module writer already assumes after a call. A seed from
the output is refused (the rewrite never makes one). Under qemu at VLEN 128
and 1024 every matrix-product fixture is bitwise against its policy's
oracle; A's row stride off by one element and a last tile one lane short are
caught.

On the RTL vector core (`rtl_test.ml`, v0.0.4), under `Fp32_contracted`,
the generated functions against the planned Zve32f loops: linear 9x32 K=64
34910 -> 9281 cycles (3.76x, 3.97 operations a cycle), with bias 3.66x,
linear 7x29 K=63 5.62x (2.55), bmm 9x32 K=64 4.97x (7.56), bmm 3x7x29 K=63
10.22x (4.90). The peak suite's FP32 burst at C[9,32], K=64 is 9.9; Linear's
B is the weight [N, K], read by `vlse32.v` at a stride of K, which is what
holds it at four. Packing B (stage 5's prepacked weights) is the lever.

### The matrix engine's binary32 rounding

`make coralnpu.zvt.numerics.runtest` (`test/coralnpu_zvt_numerics`, the VME
core, v0.0.4) runs two `vtfmm.tvv` steps of tk = 1 into one 16x16 binary32
tile from zero and compares every element with a fused step
(`fmaf (a2, b2, c)`) and a separately rounded one (`c + round (a2 b2)`), with
elements built to tell them apart. No element matches the fused step alone;
73 match only the separate one and the rest both, with the tile's rows
indexed by A's elements. A subnormal product survives (no flush to zero).
So a k-ordered chain of `vtfmm` steps is the `Ordered` requirement, bitwise,
and never `Fused_sequential`: Zvt can realize an `Fp32_ordered` contraction,
not an `Fp32_contracted` one. NaN propagation was not probed.

### On the matrix engine

On a Coral configuration whose extensions list `zvt` (the VME core), the
Coral route realizes an `Ordered` contraction with
`lib/coralnpu_kernels_zvt` (`Zvt_contraction`, through coral-isa's Rivet
extension; the qemu route never depends on it) and builds the image with the
matrix unit on; `Fused_sequential` ones stay on the vector unit, since the
engine rounds each product. The function computes each 16x16 block of C in
tile `mt0` from zero: per k step `msettn tm` and A's column slice
(`vle32.v`, or `vlse32.v` at A's row stride; the kernel DSL gained `Vlse`),
`msettn tn` and B's row slice, one `vtfmm.tvv`; then the block's `tm` rows go
to C by `vtse32`. A partial block narrows `tm` (in `mtype`) and `tn`; no
operand is packed or padded, since each slice loads only its own elements.
It refuses non-contiguous C columns and more than 256 output tiles.

On the VME core (`vme_test.ml`, v0.0.4), `Fp32_ordered`, against the vector
unit's function on the same core, bitwise both: linear 9x32 K=64 13694 ->
4870 cycles (2.81x, 7.57 operations a cycle), with bias 2.70x, linear tails
1.98x, bmm 9x32 K=64 2.36x (10.54), bmm tails 2.21x. A partial last tile
written 16 columns wide and A and B exchanged in the tile multiply are
caught. One accumulator tile and unpacked strided slices leave the engine
far below its 86.8 operations a cycle at the engine shape: four tiles and
packed panels are the next levers.

### Which unit

`Contraction.Select.choose` decides from the core's capabilities (matrix
engine, vector unit) and the descriptor, with the reason as data: the matrix
engine for an `Ordered` product on a core that has it, else the vector unit,
else the planned loops. The Coral route asks it per generated function. The
loops choice is not yet acted on: contractions are rewritten at planning,
which does not know the core, so on a core without Zve32f the vector
function is refused at admission rather than replaced.

## Operands in SRAM: staging by DMA

On the subsystem the tensors of a large model live in SRAM. Read in place,
the generated functions slow down 2.7-3.9x against DTCM (linear 9x32 K=64
9281 -> 36505 cycles, bmm 9x32 K=64 4872 -> 13376). The first transfer
schedule is the simplest: resident operands. With `Rvv_contraction ~stage`
the function owns DTCM panels (a `.bss` of each staged operand's span, which
the Coral image places in DTCM), writes one 16-byte-beat descriptor per
staged operand from its own arguments, fences, starts the chain, polls
`STATUS` until done or error and then computes from the panels; an operand
not 16-byte aligned or a chain ending in error leaves the operands in place.
The DMA registers are restated in the generator, which stays clear of the
simulator libraries.

What to stage was measured (`subsystem-vector-normal`, v0.0.4): staging A
saves little (it is read by scalar loads either way); staging B halves a
Linear (its weight's columns are strided, each `vlse32.v` element its own
SRAM access) and costs 8% on a Bmm, whose contiguous rows stream from SRAM
faster than the DMA's 1.14 bytes a cycle copies them. `in_sram` encodes that:
A always, B when strided. With it (`soc_test.ml`): linear 36505 -> 18912
(1.93x), linear tails 34417 -> 18785 (1.83x), bmm 13376 -> 13153 (1.02x),
bitwise; a chain started and not waited for is caught. The Coral route takes
the staging per descriptor (`~stage`); nothing chooses it from the slots'
homes yet. Overlapping transfer with compute (resident A with streamed B
panels, split K) is the next schedule; the copy now runs alone.

## Model census (stage 7)

`machine_rivet_rv32_census <model.pt2> --contractions` over the downloaded
models (`data/pt2-functional`): every CNN yields one product, its classifier
(mobilenetv2_050 m=1 n=1000 k=1280; efficientnet_b0, mobilenetv3_small_050,
regnetx_002, test_convnext2 likewise), and refuses its convolutions
(`bounds`: 106 in mobilenetv2_050). fastvit_sa12 adds its attention-free
MLP Linears (m=49, n=512/1536, k=512); csatv2 yields 87, among them
k=1 outer products and m=1/n=1 vector products. A convolution sums Cin
outermost, then kh and kw over clamped window bounds: not one chain unless
the window is 1x1, where each inner sum holds one product (exactly its
rounded value) and the outer one adds them in Cin order. A 1x1 convolution
is therefore an `Ordered` product even under `Fp32_contracted`; its rows
are two loops (h, w), which flatten when they cover their extents.

The recognizer now takes that form. Bounds and indices are known through
`Ssa_range` as well as constants; a term that is itself a sum over [0, 1)
from +0 is walked into (index additions of a proven zero stand for their
other operand), and its presence makes the product `Ordered` whatever the
policy: the inner sum rounds its one product, and +0 plus a product is the
product (a -0 product becomes +0 in both forms, and an accumulator from +0
never becomes -0). Two row variables flatten when the outer one's A stride
is the inner one's extent times its own; the scratch is then
[1, 1, batch, h, w, n], the same bytes. A sum's own index or a loop's is
kept as an axis even when its range is one value, else a k=1 product loses
its k axis (`conv1x1 3x5 c1 to 4` is the regression). A padded or strided
1x1 window is refused (`bounds`, `non_affine`). Found products: csatv2 118,
efficientnet_b0 65, mobilenetv3_small_050 42, mobilenetv2_050 35,
fastvit_sa12 35, regnetx_002 27, test_convnext2 9. On the VME core
(`vme_test.ml`, contracted policy, so ordered products on the engine): a
1x1 convolution 7x7 C 32 -> 48 runs 426616 cycles as loops, 67686 as the
RVV function, 26195 on the matrix engine (5.75 ops/cycle; 16.3x the loops,
2.58x the RVV function), bitwise all three.

## bfloat16 operands (stage 6.1-6.2)

`Contraction.Element` gains `Bf16`: A and B are bfloat16, widened exactly
to binary32; C stays binary32 and every sum is accumulated there, under
either numerical requirement unchanged (a product of two bfloat16 values is
exact in binary32 except where it overflows or leaves the normal range, so
`Ordered` and `Fused_sequential` still differ only in those cases and both
stay specified). The recognizer accepts two `Bf16_to_f32` loads as it
accepts two `F32_exact` ones and refuses a mix (`format`); the SSA
interpreter, the verifiers (`Ssa_op.contract_operands`) and the helper
model read the operands at their format.

The RVV function widens without Zvfbfwma, so it needs no capability the
manifest does not state (Q4 stays open, and `vfwmaccbf16` would fuse a
product the ordered requirement rounds): A's scalar is `lhu`, `slli 16`,
`fmv.w.x`; B's row is `vle16.v`/`vlse16.v` into v4 (EMUL 4 under the
tile's e32 m8, the high half of v0's group, where a widening may overlap its
source), then `vzext.vf2 v0, v4` and `vsll.vi v0, v0, 16`. The rest of the
function is the binary32 one; `s4` is saved only for bfloat16, so the
binary32 kernels are unchanged byte for byte. The selection sends a
bfloat16 product to the vector unit even on a core with the matrix engine
(`format_off_matrix`, and the Zvt generator refuses one). The reason is
numerical, measured on the VME core with the v0.0.5 release
(`zvt_numerics_test.ml`, `vtfmm.alt bfloat16 accumulation`): at tk = 2 the
engine adds both k steps' products to the accumulator with one rounding.
From an accumulator of 1, two products of 2^-24 reach 1 + 2^-23 (each
requirement ties twice back to 1), and 2^-24 with 2^-48 passes the midpoint
(each requirement, and summing the pair first, stays at 1); every random
element agrees with all four candidates. Neither `Ordered` nor
`Fused_sequential` rounds two steps at once, so a bfloat16 product on the
engine would need a third requirement of its own. The tile's slices are
2 x LMUL groups apart (four registers at e16 m2), as the peak kernels place
them; a probe with them two apart reads the wrong pairs and matched the
sequential candidates on the uniform elements only, so a probe needs
non-uniform elements to show its layout is right. Evidence: qemu at VLEN 128 over bmm and linear
fixtures (contiguous and strided B), both policies, bitwise, with A's row
stride mutation caught; on the VME core RTL against the loops, bmm 3x7x29
K=63 12.1x (ordered) and 13.0x (contracted), linear 9x32 K=64 12.2x and
14.5x, bitwise. No model in the census has bfloat16 tensors, and none is
quantized, so INT8 (6.4) has no consumer yet and waits for one.
