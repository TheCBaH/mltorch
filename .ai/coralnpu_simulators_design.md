# Coral NPU simulators — design record

mltorch runs RV32 guest images on the Coral NPU simulators published by
`devcontainer.coralnpu`: MPACT (functional, instruction counts) and the
Verilator RTL CoreMini testbenches (cycles). The guest ISA is RV32IMF with no
D extension, the default RV32 profile of `riscv32_process_design.md`. This
record grows with the route; today it covers the release pin, the manifest
decoder, the simulator profiles, the memory layout, the wire protocol, the
ELF, the guest runtime, the runner and the CompCert loader.

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
tails included, because the RTL testbench loads by copying file bytes. A
segment outside a `loadable` region is refused before anything is built.

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
statuses (124 at the RTL cycle limit, 125 on a fault halt, 126 on a loader
failure) can equal a guest code. The RTL also still reports 0 when `mpause`
follows the HTIF exit at once, as this runtime's does. Rows, closed:

- `Cycle_limit`: the testbench printed "loop limit" (and, since v0.0.3, exits
  124 there; v0.0.2 hung until the deadline).
- `Guest_trap`: `mcause`, `mepc`, `mtval`.
- `Input_refused`.
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
  store to ITCM (`mcause` 7), spin (`Timeout`), RTL `--cycles`
  (`Cycle_limit`), wrong digest (`Input_refused`), missing simulator.
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
RTL vector core `rvv-core-mini-highmem-axi`, where the cycles were measured:
Zve32f takes relu-then-add from 7572 to 1901 cycles (3.98x) and
conv-batch-norm-relu from 192313 to 23786 (8.09x); softmax and layer_norm hold
local arrays, have no vector loops, and cost the same.
