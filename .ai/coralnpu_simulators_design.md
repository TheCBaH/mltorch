# Coral NPU simulators — design record

mltorch runs RV32 guest images on the Coral NPU simulators published by
`devcontainer.coralnpu`: MPACT (functional, instruction counts) and the
Verilator RTL CoreMini testbenches (cycles). The guest ISA is RV32IMF with no
D extension, the default RV32 profile of `riscv32_process_design.md`. This
record grows with the route; today it covers the release pin, the manifest
decoder, the simulator profiles, the memory layout, the wire protocol, the
ELF, the guest runtime, the runner, the CompCert loader, the RV32 Machine IR
route, the peak-throughput suite and the full subsystem.

## The release pin

`coralnpu.lock` (repo root, same shape as rivet-compcert's `compcert.lock`)
names the release tag, the producer repository and the SHA-256 of each
host architecture's emulators archive. `make coralnpu.download` fetches this
host's archive into `data/coralnpu/<tag>/` (gitignored), checks the pin with
`sha256sum -c`, and extracts beside the old tree before swapping, so an
interrupted run leaves no partial archive under the name tests read. Other
archives of the release (simulator libraries, subsystem) are not pinned yet.

Tests that need the release read `CORALNPU_DATA`, the extracted archive
root. `make coralnpu.runtest` sets it and fails when the download is absent;
the default `make runtest` never needs it.

## Manifest and configuration IDs (`lib/coralnpu`)

`Coral_manifest` decodes the subset of `hardware-configurations.json` a run
needs (backend, ISA extensions, memory regions with their access flags,
artifacts with their launch command and SHA-256). Every configuration's ID is
recomputed and compared with the declared one, so a decoded configuration is
the one its ID names.

The producer's rule: `coralnpu:<name>:` plus the first 20 hex digits of the
SHA-256 of the record without `id`, as Python's `json.dumps(sort_keys=True,
separators=(",", ":"), ensure_ascii=False)` prints it. `Coral_json` is that
printer in OCaml: members sorted by code point (byte order on UTF-8), raw
UTF-8, Python's escapes (`\"`, `\\`, `\b`, `\f`, `\n`, `\r`, `\t`, other
controls as lower-case `\u00xx`). A decoded number is a float with its
spelling lost, so only an integral value below 2^53 is printed (as an
integer); any other number, and a duplicate member, is refused rather than
guessed. A source `1.0` would print as `1`; the recomputed ID then differs
and the configuration is refused, so the check fails closed. The hermetic test
carries a fragment whose IDs were computed by the producer's rule over
non-ASCII names, escapes and mixed-case keys; dropping the member sort or the
`\t` escape turns it red.

## Profiles

`Coral_profile.t` is one configuration plus the artifact that simulates it:
the simulator (`Mpact` | `Rtl`), the extension list, the regions, and the
launch command with its `<elf>` hole. `load` verifies the artifact's SHA-256
in pure OCaml. The baseline profiles are `core-mini-axi`,
`core-mini-highmem-axi` and `mpact-v2`; any configuration with a known
backend and known region types decodes.

Two facts per region decide where an image may put things:

- `loadable` (an ELF PT_LOAD may lie there) is the manifest's `host_load` of a
  memory region (ITCM, DTCM, RAM). Unknown (`null`) counts as no. A CSR or
  MMIO block is never loadable, even with `host_load` true: v0.0.2 marked the
  RTL core CSRs so, for the host's run control.
- `htif_reachable` (an HTIF buffer may lie there) is mltorch's, because the
  manifest does not record it. Guest runs settled it: DTCM and the external
  RAM, on both simulators.

Since release v0.0.3 both hold for the RTL testbench RAM. In v0.0.2 the
testbench loaded ELF segments and served HTIF through the core's AXI slave,
which reaches only the TCMs: a segment in its RAM aborted the loader, an HTIF
read into that RAM failed, and a write from it aborted the simulator. Any
region not shown to work is treated as unreachable, never as allowed.

## Layout (`Coral_layout`)

Pure: segments (size, alignment, executable) and slots in, addresses out. A
slot is `In` (read from the input frame), `In_out` (read, then written to the
result), `Out` (zeroed, then written) or `Scratch` (zeroed).

- Executable segments go to ITCM. There is no spill to external memory: a
  program larger than ITCM is `Over_capacity`.
- DTCM from its base: data segments, the HTIF block (`tohost`, `fromhost`,
  `tohost_ready`, `fromhost_ready`, 64 bytes apart), the HTIF packet, the
  staging buffer (4 KiB by default), the trap scratch, then slots. The stack
  (4 KiB by default) takes the top.
- A slot that does not fit in DTCM goes to the profile's writable RAM region,
  and is staged there when that region is not `htif_reachable`. No baseline
  profile stages since v0.0.3; the kernel tests keep the path covered by
  marking the RAM unreachable. A slot is never split.

The digest is 32 bits of SHA-256 over the configuration ID and every address
and size the layout assigns. It travels in both directions, so a guest built
for one layout cannot answer a host expecting another. A size change that
moves no address leaves the digest unchanged.

## Wire protocol (`Coral_wire`)

The guest reads one stream on stdin and writes one frame on stdout, both
through HTIF. Fields are little-endian; `u32` is `int32`, `u64` is `int64`.

```text
input:  "CNPUIN01" version digest length payload "CNPUEND1"
result: "CNPUOUT1" version digest status mcycle minstret length payload "CNPUEND1"
trap:   "CNPUTRP1" version mcause mepc mtval "CNPUEND1"
```

`mcycle`/`minstret` are `u64` deltas across the entry call. The other fields
are `u32`. Payloads are the read slots' bytes (`In`, `In_out`) and the
written slots' (`In_out`, `Out`), in slot order. A
trap frame carries no digest, because a trap may come before the input is
read.

Simulators print their own text around the guest's, and MPACT has printed
its banner after guest bytes. The host takes the first frame magic and keeps
the text around it. Every way the frame can break is its own `Malformed`
row, never a wrong answer: no frame, truncated, bad trailer, version,
digest, payload size.

## ELF (`Coral_elf`)

The ELF has one PT_LOAD per memory region, holding that region's sections in
address order. The layout packs DTCM, so data and the HTIF block share a
page, which separate PT_LOADs could not. Every byte is explicit, zero-filled
tails included, because the RTL testbench loads by copying file bytes. Zero
sections (`.fill_head.<region>`, `.fill.<region>`) stretch each PT_LOAD to
128-byte bounds with at least 128 bytes of zeros past its end, within its
region, so the core never fetches TCM the run left random (see the peak
suite's notes on run-to-run variation). A segment outside a `loadable` region
is refused before anything is built.

`.htif` is 256 zero bytes at the layout's address, under the four HTIF
symbols; `_start` marks the entry. The ABI is ILP32F with arch
`rv32i2p1_m2p0_f2p2_zicsr2p0_zifencei2p0`, without RVC. Both simulators load
such an image; binutils' `readelf` reads its flags, symbols and attributes
as written.

## Guest runtime (`Coral_runtime`)

The runtime is a typed Rivet module generated per layout. `_start` does the
following, in order:

1. Sets `sp`, `mtvec`, `mstatus.FS` = Initial and `fcsr` = 0.
2. Zeroes `Out` and `Scratch` slots, as the QEMU harness's files and bss are.
3. Reads the input header, refusing a wrong magic, version, digest or length
   before any slot is touched.
4. Reads each `In` and `In_out` slot, directly or in staging-sized chunks
   copied by word.
5. Calls the entry (slot table or first slot, as `Rv32_harness.Abi`) between
   two reads of `mcycle` and `minstret`. Each counter is read high/low/high
   and retried on a carry.
6. Writes the result frame, with each `In_out` and `Out` slot.
7. Exits through HTIF with 0, then `mpause` forever.

HTIF calls follow the simulators' proxy-kernel packet (eight `u64`, address
in `tohost`, `tohost_ready` = 1, poll `fromhost`/`fromhost_ready`).

A transfer loops on short counts. A count of zero or less, or one above what
was asked, ends it: an input stream that ends early is a refusal. A failed
write exits 2, since there is nothing left to report on.

The trap handler uses no stack. It writes a trap frame from the trap scratch
and exits 1. An input refusal is the same frame with `mcause` `0xFFFF_FFFF`
and the reason in `mepc`. No `ecall` is ever emitted.

Every address and constant is loaded by a `lui`/`addi` pair, so the module's
size does not depend on the layout. `Coral_image` iterates
runtime → plan → layout, starting from "nothing staged", until the layout
repeats. DTCM placement does not depend on code size, so this settles on the
second round.

Labels are generated fresh per use. A builder that reuses one item list
twice gives two loops one label. Rivet accepts that today and resolves a
reference to one of them: the runtime's second write loop once ran the
first's code and flooded stdout.

## Runner (`Coral_run`)

The runner writes the ELF to a fresh directory and spawns the simulator as a
session leader. One `select` loop feeds stdin while draining stdout and
stderr. When the deadline passes, or stdout exceeds the frame plus 4 MiB, it
kills the whole process group.

The frame decides the outcome, never the exit status. Since v0.0.3 the status
carries the low eight bits of the guest's exit code, but the simulators' own
statuses (124 at the RTL cycle limit and at MPACT's instruction limit, 125 on
a fault halt, 126 on a loader failure) can equal a guest code. The RTL also
still reports 0 when `mpause` follows the HTIF exit at once, as this runtime's
does. Rows, closed:

- `Cycle_limit`: the testbench printed "loop limit" (and, since v0.0.3, exits
  124 there; v0.0.2 hung until the deadline).
- `Guest_trap`: `mcause`, `mepc`, `mtval`.
- `Input_refused`.
- `Instruction_limit`: MPACT printed "Stop reason: instruction_limit" (and
  exits 124). Since v0.0.4 the manifest's MPACT launch command carries
  `--max_instructions=100000000`, so a guest that never exits stops there, not
  at the deadline.
- `Malformed_output`.
- `Output_limit`.
- `Simulator_crashed`: no frame and a failed process.
- `Simulator_missing`.
- `Timeout`.

`make coralnpu.runtest` runs typed binary32 kernels (negate through a
scratch slot) on `core-mini-axi`, `core-mini-highmem-axi` and `mpact-v2`.

- **Sizes:** eight floats, with every slot in DTCM; and 64 KiB slots, which
  sit in testbench RAM on `core-mini-axi` and in `extmem` on MPACT, read and
  written there by HTIF, and in DTCM on highmem. The 64 KiB runs repeat on
  `core-mini-axi` and MPACT with the RAM marked unreachable, so the slots are
  staged. Results are bitwise.
- **Counters:** `minstret` is equal on all three, 133 and 213021 for the two
  sizes. `mcycle` counts on both simulators.
- **Purpose-built guests**, one row each: illegal instruction (`mcause` 2),
  store to ITCM (`mcause` 7), an RTL spin (`Timeout`), an MPACT spin
  (`Instruction_limit`), RTL `--cycles` (`Cycle_limit`), wrong digest
  (`Input_refused`), missing simulator.
- **Store-fault `mtval`:** for the ITCM store at `0x100`, both report
  `0x100` (v0.0.2's RTL reported `0x10`).
- **In place:** the same negation in one `In_out` slot, at both sizes, and
  staged on `core-mini-axi` and MPACT; bitwise, `minstret` 72 and 98328.
- **Mutation:** shortening a staged chunk's copy by one word makes the
  staged profiles fail at element 1023, the last word of the first chunk.

## CompCert loader (`lib/loop_c_embed_compcert_coral`)

`Compcert_coral_loader.Make` is a `Loop_c_embed` LOADER for one archive
directory and configuration. The unit is compiled and lowered once by the
F-only RV32 loader (`Compcert_rv32_loader.Ilp32f`: the separate
`Compcert_embed_riscv32_fp32` extraction, its runtime helpers and the binary32
`Rv32_libc`, all lowered with
Rivet's `d` component disabled). A call builds the guest image around the
caller's region as one `In_out` slot, whose address the entry gets
(`Abi.Slot0`), and runs it; the image is kept for the next call of the same
region size. `Coral_image.build ~lowered` takes the already-lowered modules
beside the typed runtime.

`make coralnpu.compcert.runtest` runs the binary32 half of `test/loop_c_embed`
(`embed_fp32_test.ml`: the chain graph, a conv/batch-norm/relu bundle, against
the binary32 oracle at three salts; the failure record; softmax and
layer_norm) on `mpact-v2` and `core-mini-highmem-axi`, with the expect output
unchanged. One call of the chain is 25438 instructions on both, 22941 RTL
cycles. Dropping the region's copy-back, or making `memcpy` a no-op, fails
the suite on the simulator.

## The RV32 Machine IR route (`lib/machine_rivet_riscv32_coral`)

An invocation published on the RV32 qemu route (selected onto ILP32, linear
scan, frames, its module from the same writer) is built into a guest image
whose runtime calls `mir_entry` with the table of region addresses — the
`Table` ABI — so one artifact runs under qemu and on a Coral core unchanged.
A bound region is an `In_out` slot, the failure record `Out`, every other
region `Scratch`. An artifact with Zve32f forms is lowered under the vector
profile, its runtime also sets `mstatus.VS` to Initial (the privileged spec
requires it before any vector instruction; neither MPACT nor the RTL core
traps when it is Off, so no run shows it), and a configuration whose
extensions lack `zve32f` (`mpact-v2`) refuses it before running.
`make coralnpu.rv32.runtest`: the binary32 bundles are bitwise equal to the
binary32 oracle on `mpact-v2` (scalar), `mpact-m3` (scalar and Zve32f) and the
RTL vector core `rvv-core-mini-highmem-axi`, where the cycles were measured
(v0.0.4): Zve32f takes relu-then-add from 7572 to 1909 cycles (3.97x) and
conv-batch-norm-relu from 192316 to 23807 (8.08x); softmax and layer_norm hold
local arrays, have no vector loops, and cost the same.

## Peak throughput (`test/coralnpu_peak`)

`make coralnpu.peak.runtest` measures the RTL vector core (`rvv-core-mini-axi`)
and the VME core's matrix engine (`vme-core-mini-axi`) at their arithmetic
ceilings. Every kernel is built from typed instructions (the
typed RV32 kernel AST: registers, immediates and instructions as data, validated
once, lowered to a Rivet module; never assembly text) under the BF16 vector
profile, built by `Coral_image` like any other guest. An expect suite pins each
kernel's encoded bytes, entry and exports with Rivet alone, so CI, which lacks
these emulators, still catches a change; this suite runs the kernels. Its loop sits between the exported labels `peak_body` and
`peak_body_end`. `Coral_image.t` carries the bound image, so the cram prints the
loop as coralview's own decoder (`coral-isa`'s, which knows Zvt) reads those
bytes. The run passes
`--trace_file` (v0.0.4), and `coralview ops` reads the waveform back: ops per
cycle over the trace, in bursts, and the best 256-cycle window. The host checks
the kernel's `Out` slot against bytes it computed, so a wrong kernel fails before
any rate is read (a swapped accumulator in INT16 and a short FP32
multiply-add loop each turn the suite red).

The ceiling is 2 x 128 / SEW operations a cycle per kind of unit, twice that
for multiply-adds, and a widening instruction is limited by its destination's
SEW. One instruction repeated at LMUL=4 (two units, four uops) reaches it for
every type: INT8/16/32 add and multiply, their multiply-adds and the widening
ones, FP32 add, multiply and FMA, BF16 into FP32. At LMUL=1 an FP32 FMA reaches
8.6 of 16, because the scalar core passes about one vector instruction a cycle.

### Zvt tile multiplies

coralview's `coral-isa` package is Coral's Zvt as a Rivet vendor extension: the
rows of MPACT's decoder table, validated by Rivet under the feature
`xcoralzvt`. The kernel AST has typed forms for the part the suite uses
(`msetmtype`, `msettn`, `vtzero`, `vtse<N>`, and the tile multiplies `vtfmm`,
`vtfmm.alt`, `vtmmu`, `vtmms` over a tile-register class `mt0`-`mt15`), and the
lowering takes extensions as values (`Rv32_kernel_lower.Extension`), so
`machine_rivet_riscv32` needs Rivet only. An extension instruction cannot go
through the standard encoder, whose types are another functor instance's, so it
is encoded alone by an encoder over the standard rows and the extension's, then
placed as a word; Zvt has no fixups. Its configuration is the profile's
features plus the extension's own, and the lowering refuses one that would
bring anything else: Rivet enables a feature's requirements, so `xcoralzvt`
would otherwise switch `zve32x` on under a scalar profile. The expect suite
checks each form against Coral's words (`matrix.elf` and the VME tests). The
runtime sets `mstatus.MS` to Initial for such a kernel (`Coral_image.build
~matrix`), as it does `VS` for vector code.

Each Zvt kernel repeats one tile multiply on the four independent 32-bit
accumulator tiles `mt0`, `mt4`, `mt8`, `mt12` (16 x 16 elements, four tile
registers each), with A and B in vector registers at the configuration the
RTL needs: INT8 e8 m1, `tk` = 4; BF16 e16 m2 with `altfmt` (bit 8 of `vtype`,
which only `msetmtype` can set), `tk` = 2; FP32 e32 m4, `tk` = 1. A's `tk`
slices are a slice every two register groups from v8, B's from v16.

The PE array at VLEN 128 has 64 PEs (4 blocks of 2 x 8), each taking four
INT8, two BF16 or one FP32 product a cycle, and serves a 16 x 16 tile in four
steps; so a tile multiply issues every 4 cycles at best whatever the type, and
the ceiling is 2 x 16 x 16 x `tk` / 4 = 128 x `tk` operations a cycle. Each
kernel reaches it over its best 256-cycle window (burst rate in brackets): FP32
128 (127.1), BF16 256 (254.3), INT8 512 (508.0) signed or unsigned, eight times
the vector unit's INT8 multiply-add. In the trace, the 64 results a cycle are the
tile register file's 16 write ports of 16 bytes (4 blocks of 4 ports), all of
them busy in 1022 of the burst's 1026 write cycles. Independent tiles are not
needed for it: one tile updated back to back also reached 512 for INT8, so the
array forwards its own accumulation without a bubble. The four tiles keep the
kernel the shape of the elementwise ones. Nothing streams from memory here;
these are the engine's ceilings, not a GEMM's.

### Zvt matrix multiplies

`tgemm_<type>` computes C = A x B for INT8, BF16 and FP32 in 16 x 16 tiles: M
up to 16 is one row of tiles `tm` = M deep, a multiple of 32 blocks of two full
rows; W runs in passes of two tile columns, the last `tn` = what is left of W
wide. A pass keeps its tiles in the four accumulators, each loaded slice of A
or B serving every tile of its row or column; with four 32-bit tiles, two by
two is the arrangement that loads least per tile multiply. The tile multiplies
get the real `tm` and `tn`, so `ops` counts what a partial tile computes. A and
B are packed per tile, k-major, zero beyond M and W. A k step (the `tk` values a
tile multiply reduces) of two tiles is 128 contiguous bytes, taken by two
`vle32.v` of 64: at the tile configuration's SEW a 32-bit element spans four
registers (EMUL 4), so one load fills both tiles' slices, which interleave (the
second tile's start one register group past the first's), and `vtype` never
changes, which only `msetmtype` could do without losing `mtype`. One tile's
slices load one at a time at their own width. The next step loads into a
second buffer while the tile multiplies read the first. INT8 is signed on both
sides, `vtmms` with `altfmt`, which makes B signed too; without it B is
unsigned. It accumulates into INT32 where the RVV kernel uses INT16; these
inputs give the same values in either.

The loads set the pace. At the engine's own shape (C 64 x 32, two full tile
rows by two full columns) a k step is four 64-byte loads, about 6 cycles each,
against 16 cycles of tile multiplies: about 24 cycles a step whatever the type,
2/3 of the engine's ceiling. Over the bursts: INT8 347 (best window 352), BF16
174 (176), FP32 87 (88). Each has 32 k steps a block and keeps A, B and C in
DTCM: with B in testbench RAM (BF16 at K = 128, FP32 at K = 64), each of its
loads took about 18 cycles instead of 6 and halved the rate.

At the RVV kernels' shapes (`tgemm_<type>_rvv[_w<N>]`, the same A, B and C as
`gemm_<type>[_w<N>]`, from the same table) M is 8 or 9: one row of tiles, a
step two tile multiplies half used or less, for one tile of A and two of B.
INT8 takes 4 x 3 + 2 x 6 = 24 cycles a step for 2 x 8 x 16 x 4 multiply-adds,
85 operations a cycle at best (best window 88, burst 79.6 with the second
pass's start); BF16 2 x 4 + 12 = 20 cycles, 58 (58.4); FP32 6 + 12 = 18, 32
(32.5). That is 2.5, 4.6 and 3.3 times the RVV kernels on the same problem,
against 11, 14 and 9 at the engine's shape: a small M costs the engine most of
its advantage, since it keeps the loads and leaves the tiles' rows empty.
W = 24 leaves the second tile column half used at the same cycles (BF16 43.8,
FP32 24.3, against RVV's 9.5 and 7.4); W = 48 gives INT8 a pass of one column,
whose steps load as much for half the tiles (59.7 over the burst); a wider W
adds passes, each with its own start (INT8 W = 128 76.1, BF16 and FP32 W = 64
55.0 and 30.4). A single row of tiles would load less per tile multiply four
columns a pass, one tile of A for four of B, but four columns of B do not leave
registers for a second buffer.

The RTL simulators fill TCM the ELF does not load with different bytes on
every run, and the core fetches up to 76 bytes past the last instruction. A
stray word there that predecodes as a jump (a `jal` after the final `ret`)
sends a fetch to a garbage address, which fails with DECERR and costs 3 cycles:
the same image took 84070 or 84073 cycles on the RVV core and 1473 or 1476 on
the VME. `Coral_elf` therefore zero-fills each PT_LOAD to 128-byte bounds, at
least 128 bytes past its end, and every core's `mcycle` repeats exactly. The
traces still differ from run to run: a load returns its whole 16-byte DTCM row,
and the row's unwritten bytes (stack, past an output) are random, with no
effect on timing.

### RVV matrix multiplies

Matrix multiplies, RVV only (no Zvt), keep two or three rows of C in LMUL=8
accumulators, stream B a row at a time and take A's values as scalars
(`.vx`, `.vf`). Over the multiply's burst (best window in brackets):

| C = A x B | ops/cycle | ceiling | bound by |
|---|---|---|---|
| INT16 = INT8 x INT8 | 31.3 (32.5) | 32 | the multiply-add units |
| INT32 = INT16 x INT16 | 15.7 (16.25) | 16 | the multiply-add units |
| FP32 = BF16 x BF16 | 12.6 (13.0) | 16 | loads: about 15 cycles a k step against 12 |
| INT32 = INT32 x INT32 | 9.9 (10.25) | 16 | loads: about 19 cycles a k step against 12 |
| FP32 = FP32 x FP32 | 9.9 (10.25) | 16 | as INT32 |

The load unit takes about 2 + n cycles for a vector of n 16-byte beats and 3
for any scalar load. Integer A values pack two or four to a word, unpacked by
`srli`, so the widening kernels stay ahead of their multiply-adds. A 32-bit or
BF16 value takes a scalar load of its own. A scalar load issued after a vector
load waits for it, so each k step's scalar loads come first. Taking A through
`vfmv.f.s` or `fmv.h.x` instead is slower: those issue from scalar lane 0 with
the `.vf` forms and saturate it.

The default width of C is the lanes of one vector (64 for INT8, 32 for the rest).
`gemm_<type>_w<N>` sets another. A narrower W runs one tile with `vl` = W: the
instructions and cycles do not change, so the rate falls in proportion (INT8 at
W = 48: 23.5, three quarters of 31.3). A wider W must be a multiple of the lanes
and runs one pass over all blocks of rows per column tile (tiles outermost, so
moving between blocks costs what it does with one tile), rewinding A between
passes. The instructions scale with W and the rate stays within 0.5% of the
default's (INT8 at W = 128: 31.2). That is a counting effect, not a tile cost: a
burst leaves out its first block's ramp, so a burst of n blocks takes about
1960 n - 42 cycles in BF16, and a W = 64 run (3 blocks x 2 tiles) is within 18
cycles of the same kernel at W = 32 with 6 blocks. Strides come from W.

A block's end overlaps the next block's start: B's first row is loaded before
the stores, since the accumulators it does not touch and v0 is free once the
loop exits (A's first values are already loaded by the loop's last step). That
shortens every block transition, default kernels included (INT8 31.2 -> 31.3,
BF16 12.57 -> 12.63). The INT32 and FP32 kernels reuse v0 for every row and keep
the plain order.

Tried and dropped, INT8: rotating the accumulators through v8, v16 and v24 (a
period-three super-block with a copy of the loop per phase and a peeled first
trip) so one row's store overlaps the next block's multiply-adds. The output
was right at every width, but six blocks took 6303-6323 cycles against 6289 for
the plain kernel. A block transition costs about 24 cycles of idle multiply-add
unit either way: removing the stores altogether saves only 14 of the plain
kernel's 30 overhead cycles, and the rotation splits that gap in two. A C-row
store keeps the load/store unit for about 10 cycles, and with only two B
buffers (v0, v4) the next B row, which the multiply-adds wait for, queues behind
it wherever it sits, so each store costs about 15 idle cycles. Fewer idle cycles
need a third B buffer, which no register group is left for. Two smaller ideas
did not help either: zeroing each accumulator right after its store, and
dropping the `vsetvli`s around a block (the kernel's own `vse16.v` already
stores 64 halfwords at e8/m4, but zeroing the whole accumulator needs the wide
type once `vl` is below the lanes).

## The subsystem (`Coral_soc_run`)

The release's subsystem archive, pinned in the same lock file (`make
coralnpu.subsystem.download`, `CORALNPU_SUBSYSTEM_DATA`), packages
`Vchip_{vector,matrix}_{normal,highmem}`: the reference SoC around the same
cores, with 4 MiB of RTL SRAM at `0x2000_0000`, a ROM, peripherals and the DMA
engine. Its manifest decodes like the emulators': interface `subsystem-cli`
makes the `Subsystem` simulator, ROM and DDR are regions that are never
loadable (the DDR aperture has nothing behind it), and each region's
`dma_reachable` is the manifest's `access.dma`. A slot's `Home` (`Any`, `Dtcm`,
`Ram`) overrides first fit, so a kernel can keep an operand in SRAM while DTCM
has room, or be refused rather than moved out of DTCM.

There is no HTIF. The simulator backdoor-loads the ELF (`--load_elf`), then
waits on a TCP port (`SPI_DPI_PORT`) for a host to start the core through the
SoC's SPI-to-TileLink bridge: the PC to the core CSR base + 4, then 1 and 0 to
the base (clock gate, reset). The release does that with a Python loader;
`Coral_spi` is the same protocol in OCaml. A read of several beats puts a
framing byte between them, which the release's loader never meets because it
reads one beat at a time, so `Coral_spi` does too. The simulator never exits;
the host kills its process group.

The bridge moves about a kilobyte a second, so no slot bytes cross it. Every
slot is a section of the ELF (read slots with the caller's bytes, the others
zero), and the guest's runtime (`Coral_runtime.module_ ~mailbox:true`) neither
reads nor zeroes them. After the timed call it writes a 256-byte record
(`Coral_mailbox`) where the HTIF block would be: magic, layout digest, status,
`mcycle` and `minstret`, a hash of each written slot, and last a done word,
then sends one byte on UART1 (its baud set from the clock table). The host
waits for that byte in the simulator's UART log, then reads the record. A trap
leaves a trap record the same way. Polling the done word over the bridge
instead made DMA-heavy runs vary by a beat (14 cycles) from run to run: each
bridge read takes the DTCM device port the DMA writes through, at a wall-clock
moment the run cannot repeat. Watching the log reads nothing over the bus, and
the counts repeat, even with several runs at once. The hash is FNV-1a over 32-bit words with a fold, `h xor
(h lsr 15)`, after each multiply: without it a word's top bit never reaches the
others, and the suite's own check (an output hashed against the kernel's input,
eight floats negated) passed, because the sign flips cancel in pairs.

`make coralnpu.soc.runtest` negates binary32 floats on all four variants, with
the slots in DTCM and in SRAM, then fails one way each: illegal instruction
(`mcause` 2), a store to ITCM (7, `mtval` 0x100), a spin (`Timeout`), a wrong
digest, a missing simulator, and an output hashed against the wrong bytes.
Cycles repeat exactly from run to run: eight floats take 202 cycles in DTCM and
322 in SRAM, 4096 in SRAM 139314. The simulator runs about 20 000 cycles a
second on the aarch64 devcontainer, which bounds every workload it measures.

The peak suite's kernels, unchanged and with everything in DTCM, take the
standalone cores' cycles on the subsystem's normal-TCM cores, within four
cycles of code placement (`test/coralnpu_soc/peak.t`): RVV GEMMs for INT8,
FP32 and BF16 on both the vector and the matrix SoC, Zvt tile multiplies and
GEMMs for INT8 and FP32. So the peak rates are the subsystem's compute
ceilings. Zvt BF16 (`vtfmm.alt`) is wrong there, although the manifest lists
BF16 among the matrix formats: the engine's BF16 tile multiplies need the
build define `-DZVTBF16FMM_ON`, which the producer adds in a wrapper patch
later than v0.0.4 (whose provenance ends at patch 0024). A define is not a
generator option, so the manifest does not show it. Consistently,
`tile_bf16` takes 639 cycles against the standalone 1470, under the 1024 its
256 tile multiplies need at one per 4 cycles: they never occupy the PE array.

### Raw transfer speed and the DMA (`test/coralnpu_dma`)

`Coral_dma` is the engine as data: its registers, and descriptor chains the
host builds for the layout and loads with the image (an `In` slot in SRAM), so
the guest's whole driver is two stores and a poll of `STATUS.done`, which a
start clears. Descriptors are checked before loading: a length of one to 16 MiB
- 1 bytes in whole beats, both ends beat-aligned and each within one region the
manifest marks `dma`. The kernel DSL gained `andi`, `beqz`, `csrr mcycle`,
`fence` and `sub` for it.

`make coralnpu.dma.runtest` times each transfer inside its kernel (mcycle
around it; the DMA from the descriptor address written to `done` seen), the
timings read back over the bridge (`Coral_soc_run.run ~read_back`) and the
destination checked by hash. Shortening a chain's last descriptor by a beat
turns it red. All four subsystem variants give the same cycles. In bytes a
cycle, at sizes where the start no longer counts:

| Path | DMA, 16-byte beats | RVV copy, e8 m8 | scalar copy |
|---|---|---|---|
| SRAM -> DTCM | 1.14 | 2.42 | 0.44 |
| DTCM -> SRAM | 1.14 | 2.12 | 0.40 |
| SRAM -> SRAM | 1.77 | 1.39 | 0.29 |
| DTCM -> DTCM | 0.84 | 6.15 | 0.80 |

The DMA starts in about 57 cycles and then moves one beat at a time, read then
written: about 14 cycles a beat between SRAM and DTCM, 9 SRAM to SRAM, 19 DTCM
to DTCM. A narrower beat takes as long, so only 16-byte beats are worth using.
Each further descriptor of a chain costs about 8 cycles, so a chain of
row-sized pieces keeps most of the rate (64-byte pieces: 0.99 B/cycle). The
core's RVV loads read SRAM at 3.0 B/cycle and DTCM at 12.6; its copy needs
LMUL 8 to reach 2.4 (1.45 at m1). So the DMA is the slower mover: its use is
that it runs beside the core.

Beside the core, an 8 KiB SRAM-to-DTCM chain against as many cycles of work:
ALU work, or the core reading SRAM, leaves both at full speed. While the core
streams DTCM at its full rate (RVV loads or stores), the core is not slowed,
but the DMA's writes into DTCM lose arbitration and it advances at about 70% of
its rate until the core stops. A kernel paced by its DTCM loads therefore gets
about 0.8 B/cycle of DMA in the background, not 1.14.

Descriptor polling (`Coral_dma.Poll`, `poll_en`): the engine reads the polled
word before every beat and waits until `word land mask = value`, as upstream's
DMA document says. A one-beat descriptor on a flag is a gate: the chain stays
on it (`CUR_DESC`) until the core writes the flag, then runs on about as fast
as a fresh start. Polling every beat costs about 4 cycles a beat on an SRAM
word and 9 on a DTCM one, so it is for pacing peripherals, not memory copies.
A DMA spinning on a flag, in SRAM or DTCM, costs the core's DTCM stream 1%.
A flag-and-token handshake (51 cycles to the token seen) is slower than
starting a one-beat chain and polling `STATUS` (35), so the matrix multiplies
keep a chain a step started by the core: gating would save nothing.

### Matrix multiplies fed by the DMA (`test/coralnpu_dma/gemm_*.t`)

`Dma_gemm` computes C[M x W] = A[M x K] B[K x W] with A, B and C in SRAM. B is
resident in DTCM, copied in a prologue. A streams a block of rows at a time
into a two-block ring in DTCM. Step i computes block i while one chain fetches
block i + 1 (and, with `c-by-dma`, writes out block i - 1's rows from a second
ring); the chain starts before the block (after a `fence`, since it may copy
rows just stored) and is waited for after it. A start while the engine is busy
is ignored, so the wait is what keeps a chain from being dropped: removing it
fails the DMA-bound cases (the compute-bound RVV one hides it), and swapping
the ring halves fails every case. Each matrix crosses the bus once, the least
any order moves. The block is the peak kernels' own
(`Peak_zvt.Tile_gemm.block_items` was split out of its kernel, the peak bytes
unchanged), packed as they pack it, and C is checked against the host's
product.

Two choices of mover, since the core copies faster than the DMA and is idle
while it waits: B's prologue by the DMA alone (`b-by-dma`), or a quarter by the
DMA and the rest by the core's RVV copy beside it (`b-shared`, 2.4 and 0.8
B/cycle); C's rows by the DMA (`c-by-dma`) or by the block's own stores
straight to SRAM (`c-by-core`, no C ring). All cycles are the whole call's,
prologue included:

| Engine, SoC | M x W x K | b-by-dma c-by-dma | b-shared, best C | ops/cycle |
|---|---|---|---|---|
| RVV INT8 -> INT16, 32 KiB | 256 x 64 x 256 | 29.0 | c-by-dma | 30.0 |
| Zvt INT8 -> INT32, 32 KiB | 128 x 32 x 128 | 31.1 | c-by-core | 56.1 |
| Zvt INT8 -> INT32, 1 MiB | 256 x 128 x 512 | 107.5 | c-by-core | 182.7 |
| Zvt INT8 -> INT32, 1 MiB | 256 x 128 x 1024 | | c-by-core | 202.5 |
| Zvt FP32, 1 MiB | 256 x 128 x 256 | 32.8 | c-by-core | 50.6 |

RVV hides the DMA entirely and runs at 96% of the peak suite's DTCM-resident
31.3; its stores of C to SRAM cost more than the DMA's copy, so C stays with
the DMA. Zvt consumes operands four to eight times faster, so the movers set
its pace. With the DMA alone a kernel takes at least the bytes moved over
about 1.1 B/cycle, A (M K s), B (K W s) and C (M W c) for s-byte inputs and
c-byte results, so its rate is at most 2 / (s/W + s/M + c/K) times that: Zvt
INT8 no better than RVV on 32 KiB, 107.5 on 1 MiB (the 320 KiB moved at 1.03
B/cycle). With the core writing C and sharing B, a block costs about what the
DMA takes for its A and the core for its tile multiplies and C's stores, and
INT8 reaches 202.5 with K = 1024: 58% of the engine's DTCM-resident burst with
every operand in SRAM, 6.7 times RVV. Widening W to 256 at K = 512 makes the
core's stores of C the bound instead (190.4; with C by the DMA, 154.9). FP32
moves four bytes an input element and is held near 50 whatever moves them.
INT8 at K = 1024 and FP32 at K = 256 take the same cycles: the same bytes, the
same tile multiplies (an INT8 one reduces four k values, an FP32 one one).

RVV BF16 -> FP32 and FP32 (three rows of 32 lanes a block, the peak kernels'
blocks) run at 12.2 and 9.3 on the 32 KiB vector SoC, 97% and 94% of their
DTCM-resident 12.6 and 9.9. The control is the peak kernel itself at the same
shape with A, B and C left in SRAM for the core's own loads and stores: RVV
INT8 9.9, BF16 4.4, FP32 3.0, so the DMA-fed kernels are about three times
faster; Zvt INT8 80.5 and FP32 21.6 on 1 MiB, 2.3 times slower than fed. On
the 32 KiB SoC, Zvt INT8 at 128 x 32 x 128 reads SRAM in place as fast as the
DMA feeds it (57.4 against 56.1): there the DMA gains nothing.

`Dma_model` predicts a variant's cycles from the measured rates alone: the DMA
at 16/14 B/cycle, 0.8 beside the core's DTCM streaming and 1.03 beside its
copy, 57 cycles a start; the core's copy at 2.42 and its stores of C to SRAM at
2.34; each engine's DTCM-resident burst; 50 cycles a step. A step overlaps the
block's multiplies (contended) and its stores of C (not) with the chain, the
chain's start included; the rest of the chain moves at the full rate. It is
within 0 to +4% of every run in the crams and of five runs it was not fitted
to, and `Dma_model.variants`
ranks the variants that fit a DTCM budget: on 32 KiB, RVV with the shared
prologue and C by the DMA, as measured; on 1 MiB, Zvt with the shared
prologue and C by the core up to W = 256. Its pick for 512 x 256 x 1024 INT8,
measured afterwards, took 1097986 cycles against 1092949 predicted: 244.5
operations a cycle, 70% of the engine's DTCM-resident burst with A, B and C in
SRAM. At W = 512 it predicts the core's stores of C would bound the block
and picks C by the DMA instead: measured, 1961336 cycles (273.7 operations a
cycle, 79% of the burst) against 2173969 with C by the core, the model 3%
and 1% off. A C row of 2048 bytes or more needs its tile stores' stride in a
register, and a block that wide is more than a branch reaches, so the step
loop closes with a `j`.

### Geometries (`Dma_gemm.Geometry`, `Dma_gemm.Out.Split`)

Resident B needs B in DTCM, and A packed. Four further shapes lift those
limits; each is the same step pipeline with another piece streamed:

- **C split by rows** (`c-split-N`, Zvt): a block's first N of 32 rows go
  straight to SRAM by the core's tile stores, the rest to the C ring for the
  next chain (`Peak_zvt.Tile_gemm.block_items ~split`). At 512 x 256 x 1024
  INT8, N = 20 takes 1047424 cycles (256.3 ops/cycle) against 1097986 with
  the core storing all of C. At W = 512 the model's pick, N = 8, gained
  nothing (1971328 against 1961336 with C by the DMA): there the core's stores
  and the DMA contend for SRAM, which the model does not price (it is 5% low).
- **Resident A** (`resident-a/P`): A stays in DTCM; B streams by panels of P
  columns, packed so a panel is one descriptor, and a step runs every block of
  A against its panel. C's rows, a panel wide, go out by the core's stores at
  the full row stride, or from the C ring a descriptor a row (8 cycles each,
  now in the model). It is how RVV, whose block is one vector wide, reaches W
  past 64: RVV INT8 at 32 x 512 x 128 on 32 KiB runs at 28.3 (B is 64 KiB),
  the direct control 9.7; RVV BF16 at 24 x 1024 x 128, 12.1. Zvt INT8 at 128 x
  1024 x 1024 on 1 MiB, with a 1 MiB B, runs at 233.0.
- **Split K** (`split-k/C`): B in chunks of C rows, two in DTCM; the next
  chunk arrives a share a step beside the current chunk's steps, so no step
  waits for a whole chunk. A streams a block of a chunk's columns a step, and
  each block's partial sums come back from SRAM through the C ring before its
  step (from a zero slot for the first chunk) and go out after it: the
  ring's half is refilled after it is emptied, in the same chain, which needs
  two blocks or more. The RVV block starts its accumulators by vector loads,
  the Zvt one by tile loads (`vtle32`, `Peak_zvt.Tile.load`, `block_items
  ~accumulate`). RVV INT8 at 128 x 64 x 1024 on 32 KiB runs at 28.7, 92% of the
  DTCM-resident rate, with B twice the DTCM; Zvt INT8 at 256 x 32 x 1024 on 32
  KiB, 22.6, bound by the DMA carrying four-byte partial sums both ways; Zvt
  INT8 at 256 x 256 x 4096 on 1 MiB, where A and B are 1 MiB each, 137.2 with
  chunks of 1024.
- **Row-major A** (`row-major-a`): A as an activation arrives. A block's rows
  are contiguous, so one descriptor still brings them; the core packs them in
  DTCM (a vector load and a strided `vsse` a row, VLMAX elements a trip)
  before the block. A strided store writes an element a beat, several when
  they share one: RVV INT8's 2-byte stride packs 4.4 elements a cycle (5% on
  the step, 294312 cycles against 279456), Zvt's 32-byte one 0.9 (328054
  against 183697). For Zvt, A is worth packing where it is produced.

`vtle32` is `vtse32` with the load opcode, from MPACT's decoder row in
coral-isa; `vsse` goes through the standard encoder. Both are checked on the
SoC by the results. Mutating each mechanism turns its case red: the C split's
descriptor covering the core's rows, resident A's panel halves collapsed,
split K without its partial sums or writing the next chunk into the current
one, the packing's stride. The model now overlaps a chain's start with the
step's work, which brought the RVV runs from +3-4% to 0%; it is within 5% of
every new run, cram or not, but split K's small RVV steps (+6%).
`Dma_model.variants` adds the geometries. It picks resident A where B does
not fit, and (unmeasured) at 512 x 64 x 1024 where it does, since the larger
operand then arrives at the core's copy rate in the prologue; split K where
neither operand fits.
