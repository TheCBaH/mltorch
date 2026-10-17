# Coral NPU simulators — design record

mltorch runs RV32 guest images on the Coral NPU simulators published by
`devcontainer.coralnpu`: MPACT (functional, instruction counts) and the
Verilator RTL CoreMini testbenches (cycles). The guest ISA is RV32IMF with no
D extension, the default RV32 profile of `riscv32_process_design.md`. This
record grows with the route; today it covers the release pin, the manifest
decoder, the simulator profiles, the memory layout, the wire protocol and the
ELF.

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

Two facts per region are mltorch's, because the manifest leaves them unknown
(`null`) and guest runs settled them:

| Fact | MPACT | RTL |
|---|---|---|
| `loadable` (an ELF PT_LOAD may lie there) | ITCM, DTCM, RAM | ITCM, DTCM only: the testbench loads through the core's AXI slave and aborts on a segment in its RAM |
| `htif_reachable` (an HTIF buffer may lie there) | DTCM | DTCM: the HTIF initiator reaches only the core's AXI slave; a read into testbench RAM fails, a write from it aborts the simulator |

Any region not shown to work is treated as unreachable, never as allowed.

## Layout (`Coral_layout`)

Pure: segments (size, alignment, executable) and slots (`In`/`Out`/`Scratch`)
in, addresses out.

- Executable segments go to ITCM. There is no spill to external memory: a
  program larger than ITCM is `Over_capacity`.
- DTCM from its base: data segments, the HTIF block (`tohost`, `fromhost`,
  `tohost_ready`, `fromhost_ready`, 64 bytes apart), the HTIF packet, the
  staging buffer (4 KiB by default), the trap scratch, then slots. The stack
  (4 KiB by default) takes the top.
- A slot that does not fit in DTCM goes to the profile's writable RAM region,
  and is staged there when that region is not `htif_reachable`. A slot is
  never split.

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
are `u32`. Payloads are the `In` and `Out` slots' bytes, in slot order. A
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
