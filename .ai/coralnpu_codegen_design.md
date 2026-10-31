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

## A whole model on the RTL (stage 7)

`bin/machine_rivet_coral_model` (manual, built where the RV32 encoder is)
runs every invocation of a `.pt2` as one image on a Coral configuration,
tensors in the configuration's RAM, outputs bitwise against the binary32
oracle, cycles per invocation. test_convnext2 is the downloaded model whose
tensors fit; on `rvv-core-mini-highmem-axi` (Fp32_ordered, Zve32f,
`--contractions`) its 106 invocations take 285.8 M cycles, bitwise, in 3.7
hours of simulation; the loops-only run did not finish in four hours. The 9
products (37.0 M multiply-adds) take 85.1 M cycles, 30%: 0.87 ops/cycle,
against 4-7.5 for the same functions over DTCM, so operand placement, not
the kernel, bounds them there (the DMA staging of stage 5 is the subsystem's
answer; this core has no DMA). The other 70% is invocations with no
product: the depthwise 7x7 convolutions, the norms and GELU. The next gain
is there, not in more matrix-engine work; tile residency across operations
and epilogue fusion wait until a product dominates a profile.

## Prepacked weights (stage 5.3)

A Linear's (or a 1x1 convolution's) weight is B with strided columns: each
`vlse32.v`/`vlse16.v` element is its own access, which is why a linear runs
at half a bmm's rate. `Ssa_prepack` rewrites, after planning, every
`Contract` whose B is a bundle constant with strided columns to read a new
input buffer, B packed as [batches, k, n] with contiguous columns (one
batch when B's batch stride is 0), and drops the source buffer when nothing
else reads it, so the weight is not bound twice. `Mir_model.prepare
~prepack` (default off) names each packed copy an edge past every edge of
the bundle, one per source and packing whatever buffer ids the kernels
used, and `Context.create` builds it once from the source constant's bytes,
before any invocation; the planning summary is taken over the packed
program, whose digest it binds. The packing is keyed by source edge,
format, batches, k, n and the source strides; the target (Zve32f) and
numerics are the model's, so they need no key of their own. A wrong
packing order goes red on the interpreters (`prepacked weights`). Evidence,
all bitwise: qemu over binary32, bfloat16 and 1x1-convolution weights; the
VME core, contracted (RVV function), linear 9x32 K=64 9281 -> 4961 cycles
(1.87x), tails 1.58x, bfloat16 1.73x; the subsystem with tensors in SRAM
under the staging rule (which, B now contiguous, stages only A) 18912 ->
13242 (1.43x) and 1.29x, so 36505 -> 13242 (2.76x) against reading the
strided weight in place. Activations are not packed: an input B (a bmm's)
stays as it is.

Staging is bounded by DTCM. A staged panel is `.bss` of the function, which
the image places in DTCM, so an operand larger than DTCM would make the
image fail. The Coral image therefore gives all of its generated functions
five eighths of the DTCM for panels (the stack, the staging area and the
program's data keep the rest), shared in the order the image builds them,
and `Rvv_contraction.within` cuts each function's staging to what fits, B
before A; with the budget lifted, a Linear of 200 rows at K=64 makes the
image refuse (55360 bytes of DTCM needed, 28608 available).

A is then streamed rather than copied whenever its rows are contiguous
(`Stage.Mode.Streamed`): blocks of rows through two DTCM halves, the chain
of block j + 1 started once block j has arrived and before block j's
product, into the half block j - 1 used, so the copy runs beside the
product. The loop order becomes blocks, column tiles, row groups; B, copied
once, is read from DTCM by every block. A block is the fewest rows, a
multiple of a k step's rows, whose bytes are whole beats, times as many as
the halves hold. The ring has two obligations, each a mutation the
subsystem catches: wait for a block before reading it (`No_ring_wait`) and
fetch into the other half (`Aliased_ring`). The ring's source is a buffer
base, which every Coral layout aligns; a misaligned one or a chain error
executes `unimp`. `in_sram` now also copies a contiguous B that more than 16
rows reuse. On `subsystem-vector-normal`, operands in SRAM, the weight
prepacked: linear 192x32 K=128 554842 -> 226815 cycles (2.45x), 2.83 -> 6.93
ops/cycle, against the DMA suite's hand-built resident-B kernel at 9.3
(which packs A for its block and copies C out by the DMA; the generated
function stores C from the core); linear 200x16 K=64 457172 -> 148565
(3.08x). With A streamed the staged copy is no longer the only chain, so
the copy-not-waited check runs over a staging of B alone.

The RVV function reads a row group's A values through one pointer and
immediate offsets when the last row's offset fits a load's 12 bits (rows of
at most 1023 bytes for three rows), saving two `addi` a k step. The figures
above predate it; after it, on the VME core: bmm 1x9x32 K=64 4874 -> 4678
cycles (7.88 ops/cycle), linear 9x32 K=64 prepacked 4961 -> 4765, bfloat16
linear contracted 12738 -> 11226 (16.4x the loops); on the subsystem the
streamed 192x32 K=128 226815 -> 218311 (7.20 ops/cycle). Products bound by
SRAM move by a cycle. What is left against the suite's 9.3 is mostly the
vector unit's own rate at m8 (three `vfmacc.vf` a k step) and C's stores to
SRAM, which the suite's kernel hands to the DMA.

## Transfer plans and their model (stage 5.1, 5.4)

Staging is a plan, not a rule inside the generator. `lib/coral_transfer`
(pure: no Rivet, no simulator) holds `Transfer_plan.t`: A in place, copied
whole into a DTCM panel, or streamed by blocks of rows through the ring; B
in place or copied. `check ~budget d` is the only way to a `Witness.t`: it
refuses a copy past one DMA transfer, a streamed A whose rows are not
contiguous, a block that is not a positive multiple of the fewest rows whose
bytes are whole beats (and of a k step's rows), and panels past the DTCM
given (`Refusal`, typed). `Rvv_contraction.module_ ?plan` takes only a
witness, and refuses one checked for another descriptor. The Coral image asks
its `transfer` chooser for each generated function's plan within what the
functions before it left of the five-eighths budget, checks it, and makes a
refusal an error rather than a silent fallback. `candidates` offers every
plan worth ranking (B in place or copied; A in place, copied, or streamed by
the largest block the rest holds), the in-place one always among them.

`Transfer_model.predict` times a plan on the subsystem and `choose` takes the
cheapest candidate, replacing the old staging rule. Each k step of a row
group costs the longer of its vector work (per row, a fixed part and a part
per four lanes) and its loads (A's values, B's row by beats when contiguous
or by elements when gathered, each rate by where the operand is), since the
two overlap, plus the step's scalar instructions; each group stores C by
beats; copies run by the DMA alone first; a streamed A's later blocks move
beside the block before, costing what it does not hide and, when the product
reads B from SRAM, a contention per byte (the SRAM term stage 5.1 asked for).
The structure came from the measurements, each fix forced by a residual: a
strided B copied to DTCM still gathers there (`b_element_dtcm`); A's SRAM
latency hides behind a longer ordered step and does not behind a short tail
tile, which an additive model cannot say and a `max` can; B and C cost by
beats, so a 8-lane tail tile is not priced as a full one. The rates are
fitted by Nelder-Mead on relative error to 85 plans of 16 shapes
(`test/machine_rivet_riscv32_coral_transfer`, five hours, `make
coralnpu.transfer.runtest`, manual), the DMA's own rate and start held at the
DMA suite's.

The model's own error is not what stage 5.4 bounds (see below), but for the
record: the worst plan is off by 8.8% in the fit, 9.5% against the function
after its k loop was pipelined (the rates were fitted before; a refit does no
better); left out a shape at a time it is 6.1% at the median shape
and 19.1% at the worst, an ordered shape (five of the sixteen are ordered,
too few to pin their rates down). Two earlier held-out sets were spent on
finding the structure, which is why the final figure is a cross-validation
rather than a fresh set. As a chooser it does what it is for: under the
fitted rates it picks the measured best plan on all sixteen shapes (in the
fit; on the two held-out sets, before they joined it, the best or within
5.7% of it).

The evidence stage 5.4 asks for is the DMA suite's committed rates
reproduced through the generated path, and it is a kernel-to-kernel
comparison: `kernel_test.ml` (beside the subsystem suite) calls the
generated function alone, A, B and C in SRAM, from a caller handed the slot
table, at the suite's resident-B shape (M=192, W=32, K=128, binary32,
fused). Under the plan the model chooses (B copied by the DMA, A streamed by
three rows, C stored by the core) it takes 186304 cycles, 8.44 ops/cycle,
bitwise; the suite's kernel at the same geometry (`gemm_rvv_float.t`, B by
the DMA alone, C by the core) takes 186003, 8.5: within 0.2%. Both in place
it takes 526546 cycles, 2.99, against the suite's direct 527868, 3.0. The
suite's faster variants (B shared with the core's copy, C out by the DMA:
9.0 and 9.3) are geometries the generator does not produce. The earlier
figure of 7.2 ops/cycle was a whole program's, which also copies C out of
the contraction's scratch buffer. Reaching it took the k loop pipelined as
the peak kernels' is: k steps in pairs, the next step's A values loading
into a second register set (fa3-fa5) while the current step's multiply-adds
run, A's pointers advanced once a pair, the last one or two steps peeled so
that no load reads past A's last step.

## Integer contractions (stage 6.4)

`Contraction.Element.I8 { a; b }` names 8-bit operands, each signed or not;
C is a 32-bit signed integer and the sum exact, so the numerical requirement
does not apply and any order computes it. `make` refuses a k at which the
largest products' sum could pass 2^31 - 1 (`Overflow`: 131071 signed by
signed is the last k accepted). The selection sends an integer product to the
matrix engine when the core has one (`integer_on_matrix`): unlike binary32
and bfloat16, nothing in the engine's rounding stands in the way.

The RVV function loads A's value with `lb`/`lbu` into an integer register
(t1, t2, s4, free when the group's rows share t0, which an integer A with rows
under 1 KiB always does), B's row with `vle8.v`/`vlse8.v` into v6, the top
quarter of v0's group, widens it with `vsext.vf4`/`vzext.vf4` and adds with
`vmacc.vx` into 32-bit accumulators: no INT16 intermediate, so no bound is
needed beyond `make`'s. The Zvt function multiplies four k steps a tile
multiply at e8 m1, their slices two registers apart, zero past k, with
`vtmms` for a signed A and `vtmmu` for an unsigned one, `altfmt` for a signed
B, and stores the 32-bit accumulators under the binary32 configuration.

No recognized loop nest is an integer product, and no downloaded model is
quantized, so the functions are reached directly: a caller module hands them
A, B and C from the slot table (`int8_test.ml` beside the qemu and the Coral
suites). Under qemu all 32 cases are exact (four sign pairings, B contiguous
and strided, two shapes with tails, VLEN 128 and 1024) and A's row stride
mutation is caught; swapping sign and zero extension turns all 32 red. On
the RTL both functions are exact for every pairing (`vtmmu` with `altfmt`
included): at 32x32 K=64 the vector unit takes 12855 cycles (10.2 ops/cycle)
and the matrix engine 5907 (22.2), at 19x29 K=61 8.8 against 16.2.

## The DMA suite's kernels in the libraries (stages 4.1, 5.1)

The peak suite's GEMM and tile kernels and the DMA suite's RVV block emitters
moved into `lib/coralnpu_kernels` (`Peak_kernels`, `Dma_block`; no coral-isa:
the harness's `build_with ~extensions` takes the Zvt encoder from its caller)
and `lib/coralnpu_kernels_zvt` (`Peak_zvt`, with the encoder and `build`); the
DMA-fed multiplies and their model followed (`Dma_gemm`, `Dma_model` in the
Zvt library, since they emit tile kernels), their geometry types as pure data
in `lib/coral_transfer` (`Dma_geometry`). The suites re-export every module
under its old name, so their byte baselines are unchanged. `Dma_model` gained
what stage 5.1 asked of it: `variants` returns a typed refusal (`Rows`,
`Nothing_fits` with the least DTCM any variant needs) where it returned `[]`;
`~a_row_major` ranks the row-major geometry where resident B's applies, as an
activation arrives; and the core's stores of C contend with a chain carrying
C's other rows (`store_contention`, above). It is within 0-5% of every
measured run but split K's small RVV steps (+6%), as before. The modules keep
`int` extents and no interface: every caller is a suite whose byte baseline
pins their code.

## The suite's resident-B geometries, generated (stages 5.2, 5.4)

Stage 5.4 asks the generated path to reproduce the suite's committed rates
within `Dma_model`'s recorded error (0-5%) for every geometry the model
picks. For the binary32 resident-B shape on the 32 KiB SoC the model picks B
shared and C out by the DMA, so the plan gained both. `B.Shared`: the chain
copies B's head while the core copies the rest, three quarters in 256-byte
trips of two e8 m8 groups, as the suite's `b-shared` prologue. `Out.Dma`
(with A streamed, C's rows contiguous and all of C whole beats, else a typed
refusal): a ring half holds a block's A and then its C, the core stores the
block's C into its half, and the chain started before block j's product
also carries block j - 1's C out of the other half (a second descriptor,
from the second block on); a last chain carries the last two blocks' C out.
The model prices the shared copy and C's DTCM stores from the suite's rates
(the core's copy 2.42 B/cycle, the DMA 1.03 beside it; a DTCM store a beat a
cycle, unmeasured) and picks, at 192x32 K=128, A streamed by three rows, B
shared, C by the DMA, the suite's pick.

`kernel_test.ml` runs every candidate there, bitwise, and catches two new
defects (the core's share of B not copied; C carried out of the half being
computed). Against the suite at the same geometry: B shared, C by the DMA
170634 cycles against 168368 (1.3%); B shared, C by the core 177041 against
175628 (0.8%); B by the DMA, C by the DMA 179896 against 178743 (0.6%); B by
the DMA, C by the core 186304 against 186003 (0.2%). Both in place, 2.99
ops/cycle against the direct control's 3.0.

## The matrix engine's bfloat16 pair (numerics `Paired`)

The earlier probe showed `vtfmm.alt` at tk = 2 agrees with neither ordered nor
fused rounding, and four hand-picked candidates could not tell what it does
instead. Random bfloat16 data over wide exponents on the VME RTL (v0.0.5) can:
a tile multiply from a zero accumulator rounds the two products' sum **to
odd** in binary32 (truncated, last bit set when anything was cut off), and
with a nonzero accumulator that pair sum is then added to it **to nearest**:
`acc <- RNE (acc + RTO (p0 + p1))`, each product exact. At exponent spread 12 a single
rounding of the three terms to nearest (what the probe's cases allowed) is
wrong on 213 of 1024 cells at k = 4, and rounding the pair to nearest on 307
at k = 2; the rule above is bitwise on all 1024 at both, and on 32x32 k64
and 19x29 k61 with partial tiles. A result does not depend on its neighbours
(a cell run alone gives the same bits). Found by sweeping `1 + small` with the
small product's depth and bits known: the added bit below the result's last
set it however far down, and the bits above were truncated.

`Contraction.Numerics.Paired` names it (bfloat16 operands only; a last odd
step adds one product; a zero sum is +0, the engine's adder never gives -0,
measured with seeds and products of both signs), `F32_math.add_odd32` computes the pair sum (checked
against exact integer arithmetic over 400,000 pairs, 143,346 of which
rounding to nearest gets wrong), and `Select` sends it to the matrix engine,
or to the planned loops on a core without one (the vector unit has no way to
round to odd). `Zvt_contraction` emits it: e16 m2 with `altfmt`, two k steps a
multiply, slices four registers apart, a last odd step's second slice zeroed.
`bf16_pair_test.ml` runs it on the RTL against the reference, with the
partial-tile and swapped-operand defects caught. No policy produces `Paired`
from a model yet: ordered and contracted are the only source policies, so a
recognized bfloat16 product stays on the vector unit until a policy grants it.

## The matrix engine's bfloat16 function at block rate (`Zvt_packed_contraction`)

`Zvt_contraction` loads two slices of A and two of B for every tile multiply,
one tile at a time: 16.6 ops/cycle at 64x64 K=128 with A row-major, 30 with A
transposed, against a 150 peak, issue- and load-bound. `Zvt_packed_contraction`
instead reuses the suite's `Tile_gemm.block_items`: two tile rows by two tile
columns, the next step's slices loading while the multiplies read the current
ones, so a multiply costs about one load. Those blocks read A and B packed per
tile, k-major, the two tiles of a row or column side by side; the function
packs them itself, B once a batch and A a block of 32 rows at a time (a tail
in blocks of at most 16, k taken to a multiple of four with zero steps, which
add nothing to a pair), into `.bss` panels of its own. The ABI is the other
functions': three addresses, one symbol.

Where the time goes is measured by skipping one phase at a time (32x32 K=64, 4770 cycles: packing A 2620, packing B 830, the blocks
1200 against 873 at peak). Packing A from a row-major matrix is a strided
load, and none of Zvt's 22 instructions transposes. Read with coralview on the
VME RTL (`commits`, `memory` over a `--trace_file` run), a strided
load is one 4-byte TCM read a cycle, a request per element, and the next k
step re-reads the same words: 41 cycles for 32 elements of `vlse16`, 64 times
for 2616 of 4770 cycles. Where two k steps share a word (rows a whole number
of words apart, k adjacent) one `vlse32` brings both, and two `vnsrl.wi` (0
and 16) split them into the two slices: 45 cycles for two k steps, packing
A in 1680 cycles, 32x32 K=64 from 3933 to 3003 cycles (43.6 ops/cycle). That
is the LSU's one request a cycle; going lower means overlapping the packing
with the multiplies. Rows an odd number of elements long are not word
aligned and keep the slow path (14-18 ops/cycle at K=61 and 33). An operand whose k rows are contiguous runs of the
tile's indices is not packed at all: the blocks' loads advance by the row
pitch instead of 64 bytes (`block_items ~a_pitch ~b_pitch ~b_pass`), so B as it
is, and A when it is kept transposed, are read in place. That needs k a
multiple of four (a zero step is not in the operand) and leaves the blocks
reading a few k rows past its end. With both in place and the operands in
DTCM the function runs at 99.8 ops/cycle (32x32 K=64) and 104.5 (32x64), the
suite's best DMA-fed BF16 rates (91-113), bitwise against the paired
reference; with A row-major and packed, 33 at 32x32 K=64. Operands in
SRAM cost the core a read each (coralview `axi`: the load unit issues one
16-byte read at a time, one every 4 cycles, 4 bytes a cycle, with no
overlap with the multiplies), so 32x32 K=64 with A in SRAM runs at 62
ops/cycle, A and B 45, with C stored to SRAM too 33.5; 192x32 and 192x64 at
K=128 with everything in SRAM 42.8. Staging them is bounded by the DMA's
1.14 bytes a cycle (the core's copy 2.42): an A block streamed by the DMA
is slower than the core reading it in place unless N is above about 90, and
C stored by the core costs 4 bytes a cycle, a fifth of the block at K=64.
The bound a staging could reach is B resident in DTCM with A and C in SRAM:
57.5 ops/cycle at 192x64 K=128 before the copy of B (about 51 with it), 1.2x
the in-place rate, so a ring composition for this function is not built. What is built
is the copy: `module_ ~stage_b:true` copies a B read in place, rows and pitch
as they are, into a panel of the function's own by the core (6.3 bytes a
cycle, 649 cycles for 4 KB), for a B in SRAM reused by more than one block of
A. Bitwise on the RTL with everything in SRAM: 192x32 and 192x64 at K=128
42.8 -> 52.0 and 42.9 -> 52.1 ops/cycle (1.2x, as bounded); one block (32x32
K=64) loses, 33.5 -> 28.7, so the caller grants it only past one block.
A kept transposed through a model removes the packing: the layer computes the
product with its operands swapped, C^T = B^T A^T, so its output is the next
layer's k-major A. `Contraction.transpose` is that swap (the operands, M and N,
and C's strides exchanged; every numerics is symmetric in a product's
operands, so the bits written are the same), and the function accepts a C
stored transposed by running the swapped descriptor: bitwise on the RTL at
the same 99.8 and 104.5 ops/cycle as the plain layout, and 43.6 with the
row-major operand packed word-wise on either side. Which layers store their
output transposed is a layout decision for the graph, not for this function.

The typed kernel DSL gained the data this needed: `Item.bss`, `Item.zero`
(refused when negative) and `Insn.La` of a label, so a function declares its
own panels and takes their addresses (a workspace pointer argument was tried
first and dropped: it gave the function an ABI the RVV helper does not have).
`block_items` takes a label prefix so a function may hold several, and
defaults reproduce the suite's bytes.

### Granting `Paired` from a model

`Paired` is a relaxation (the result differs from the binary32 oracle), so a
policy grants it, not the recognizer: `Ssa_plan.resolve ~bf16_pairs:true`
(with `~contractions`, through `Ssa_backends.plan` and `Mir_model.prepare`)
makes each bfloat16 product the recognizer finds `Paired`; every other
product keeps the requirement of its policy. It is a flag beside
`contractions`, not a new `Ssa_numerics` preset: the presets are shared with
the Loop planner, whose interpreter has no pair accumulation, so the
comparison oracle of a granted run is the descriptor's `Contraction.eval`
(`Model_test.Custom`). Without a matrix engine the product falls back to the
loops, ordered, which is not the pair rounding: the grant means the matrix
engine's rounding where the engine runs the product.

The Coral image prefers `Zvt_packed_contraction` to the per-tile function
when every operand it packs goes by words (`fast`) and its `.bss` panels
fit what the earlier functions left of the DTCM's five eighths; an operand
read in place is read past its end, so every slot gets `overread` bytes more
room. On the VME RTL, bitwise against the reference over a model's own
operands: linear bf16 9x32 K=64 in 4236 cycles against 11130 for the same
product rounded step by step (granting nothing differs from the pair
reference, the control), 64x64 K=64 37784 against 58773 for the per-tile
function; bmm 3x7x29 K=63, with odd rows, stays on the per-tile function.

Packing A cannot be overlapped with the tile multiplies in this block loop.
The loop's double-buffered slices take all 32 vector registers (A at v8/v12
and v0/v4, B at v16/v20 and v24/v28, each a group of four), so the packing's
vectors have nowhere to live; doing it on the scalar core instead costs two
instructions an element, 4100 cycles a block against the vector path's 1680.
The vector path is itself near the load unit's bound (32 requests and the
stores of an iteration in 45 cycles; software pipelining could take 8-10%
off). Nor does re-blocking help: every element of A and B is packed once
whichever operand plays which role, so the cost per multiply-add is fixed by
N, and packing (0.8 cycle an element) bounds a product with N under about 50
no matter how the work is ordered. What removes it is a k-major A, which is
the graph's layout decision.

The image does grant the copy. It cannot tell which slot a call's B is (the
call's arguments are addresses computed in the program), so the function
checks at run time: `Stage.Outside {base; bytes}` copies a B read in place into
the panel unless its address lies in the DTCM (a subtract and a shift, four
cycles when it does). The image passes the profile's DTCM region for a product
that is `reuses_b` (B read in place, more than one 32-row block of A) and
whose panel fits what the DTCM budget has left; otherwise B is read where it
is. On the RTL, 96x32 K=64 with A and B in SRAM: 11507 -> 10588 cycles; with
both in the DTCM 6776 -> 6780, so the check costs nothing it would not save;
a bmm of 64x32 K=64 from a model is bitwise. Only a bmm's B is read in place
(a Linear's weight is packed), so the grant is for attention-shaped products.

A model-level profile (coralview `profile` over a `--trace_file` run of the
64x64 K=64 product) shows the work around the function: the runtime zeroing
the image's `.bss` (the panels included) a word at a time, B packed from SRAM
at 4.4 cycles an element (a strided word load is 140 cycles for 32 rows
there, 45 in DTCM), and the epilogue the contraction rewrite leaves, `out =
scratch + 0.0` over the whole result. The epilogue looks removable (a sum
seeded with positive zero is never negative zero, so the add changes no
value, and the contraction could write the output itself), and it is not: the
output slot is in SRAM and the matrix engine's strided tile stores there cost
more than the streaming copy from a DTCM scratch (9x32 K=64 4236 -> 5561
cycles, 64x64 37784 -> 51198, the ordered contraction 9280 -> 10605), so the
scratch stays.

## Fused streaming attention (`Zvt_sdpa`)

The generated `Sdpa` is refused on the VME core (its one kernel exceeds the 8 KiB
instruction memory by a few hundred bytes), and its per-op composition (a paired
scores product, scale, softmax, a binary32 context product) takes 4.2 M cycles at
64x64 d64, 97% of it the generated softmax at 1,023 cycles an element. `Zvt_sdpa`
is one function per `Attention_block` descriptor (contract: the attention design
record, section on the bfloat16 streaming grant) that takes 386 k cycles there.

Layout. Q is taken 16 queries at a time (`Bq` is fixed at 16: a 16-lane row is
one LMUL 4 vector, and `Bq` changes no bit). Q, K and V rows are always staged
dense into one buffer in the DTCM before they are packed or read in place: the
staging makes the pack independent of the operand strides, gives V's block its
zero rows past the valid keys and readable bytes past its end (a V in the DTCM
needs both as much as one in SRAM), and is 4% of the run. A ragged block
pre-zeroes its panel and packs with a run-time `vl`. The scores block writes
`z` rows of 16 f32 into the tile; pass 2 overwrites them in ascending row order
with the bfloat16 `P^T` rows (row r of P lies in z rows at most r/2); the
context block reads that tile and the staged V in place and accumulates into
`O_acc` from its loaded tiles. `m`, `m'`, `l`, `a`, `s` and the masked `m'` are
one 16-lane array each in `.bss`, so the `exp` code exists once per use.
Budget at d 64, `block_keys` 32: tile 2,048, `O_acc` 4,096, Q panel 2,176, K
panel 4,224, staging 4,736, arrays 384 = 17,664 of the 20,480-byte allowance;
`.text` 7,484 of 8,192 bytes with the runtime. `.bss` is explicit zero bytes in
the ELF and costs no run time; the runtime zeroes only output and scratch slots
(six instructions a word).

Cost at 64x64 d64, by leaving a phase out: exp pass 291 k (75%), scores 32 k,
context 29 k, normalize 18 k, the rest 16 k. `F32_math.exp` emitted over vector
registers in construction order needs 7 register groups (29 in a depth-first
walk), so it runs at LMUL 4 at 65.8 cycles an element; at LMUL 1 it is 189, slower
than the generated scalar exp (148). The softmax is exp-bound; the levers not
taken are `.vf` operand forms (the constants are rebuilt with `li` + `vmv.v.x` at
each use) and one shared copy of the code.

Two softmax identities keep a register group free: `p`'s select on `z = -inf` is
`exp (z - ms)` with `ms = m'` and 0 where `m' = -inf`; `a`'s select on
`m' = -inf` is `exp (m_eff - mn_eff)` with both 0 there. Each equals the
oracle's select bit for bit.

Route. `Ssa_op.Attend` (Q, K, V, optional mask, O) is interpreted by
`Attention_block.eval`, refused by the C, Wasm, JavaScript and Loop backends,
lowered to the helper call `attend.<descriptor>` (`Mir_attention`, ids from
1024), and bound in the artifact as `Generated_attention`. The Coral image gives
the function what the contractions before it left of the DTCM allowance;
`Mir_model.prepare ?attention` substitutes the SSA program of a selected node.
The function's argument order is Q, K, V, O, then the mask.

## What is left open

The suite's geometries beyond resident B (B by column panels with A
resident, split K) are not generated: a contraction whose B does not fit
DTCM is read in place. The matrix engine's bfloat16 multiply is generated for
operands already in memory but not yet with the DMA schedules, which are the
vector unit's, and no policy yet grants `Paired`. Integer operators wait for a
quantized model to recognize.

## A granted attention layer

A granted bfloat16 `Attention_layer` is one invocation whose body holds paired
products, one `attend.<descriptor>` call and generated loops (norm, bias,
rotation, bfloat16 copies, epilogue). Its image is past the 8 KiB ITCM even
with the kernel's regions outlined on the subsystem (8,592 to 9,648 bytes at
L 16, E 64, two heads of 32), because the attention helper stays resident;
the route refuses it with the `itcm` overrun. The open step is loading helper
modules, not only kernel regions, as overlays.
