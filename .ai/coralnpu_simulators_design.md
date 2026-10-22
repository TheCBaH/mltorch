# Coral NPU simulators — design record

mltorch runs RV32 guest images on the Coral NPU simulators published by
`devcontainer.coralnpu`: MPACT (functional, instruction counts) and the
Verilator RTL CoreMini testbenches (cycles). The guest ISA is RV32IMF with no
D extension, the default RV32 profile of `riscv32_process_design.md`. This
record grows with the route; today it covers the release pin, the manifest
decoder, the simulator profiles, the memory layout, the wire protocol, the
ELF, the guest runtime, the runner, the CompCert loader, the RV32 Machine IR
route and the peak-throughput suite.

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
