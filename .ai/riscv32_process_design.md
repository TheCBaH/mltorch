# RISC-V 32 processes under qemu-riscv32

Status: **the embedded-CompCert route runs**. The Rivet Machine IR route does not
exist yet (see "Not done"). RV32 code cannot run on the host CPU, so every run is a
static ELF executed by `qemu-riscv32`; it checks instruction bytes and control
flow, not an RV32 core's timing or memory ordering.

## The process

`lib/machine_rivet_riscv32` builds the process from typed Rivet modules, with no
assembly text on the way, and writes it with `ocaml-elf` (`Elf.Exec`, the
`rv32imafd`/`ilp32d` profile):

```text
typed modules (kernel, libc subset, harness)  -> Rv32_image   Pipeline_direct plan, bind at 0x10000
                                              -> Rv32_elf     one PT_LOAD per segment
                                              -> Rv32_process qemu-riscv32, exit code and stdout
```

`Rv32_harness` generates `_start`. It takes two file names from `argv`: the input
file is mapped read-write and private, the output file shared and sized by the
host beforehand. Regions are placed in the input file, the output file or the
ELF's zero-filled data (`Placement`); a table of their addresses is filled and the
entry called, either with the table or with the first region's address alone
(`Abi`). The entry's return value is stored at the front of the output mapping and
the host reads that file back; standard output is not used. A failing step exits
with its own code (100 arguments, 101/102 opening or mapping the input, 103/104
the output). The system calls are raw RV32 Linux (openat 56, mmap2 222, exit_group
94), so the process links no libc; `Rv32_libc` supplies the few functions the
generated C declares (`memcpy`, `memset`, `fabs`, `fma`, `sqrt` and their
binary32 forms). A unit that references another function is refused when the
image is planned, never silently stubbed.

Rivet's `li` covers only a signed 12-bit constant or one with the low 12 bits
zero, so `Rv32_asm.load_imm` expands any other constant to `lui` + `addi`.

## CompCert on RV32

`Loop_c_embed.Make` takes a LOADER, now with a `dialect`. `Compcert_ilp32` is the
`Compcert_scalar` C for a target whose `long` is 32 bits: `int64_t` is `long long`
and constants end in `LL`. `lib/loop_c_embed_compcert_rv32` is the loader: the
unit is compiled by the embedded riscv32 CompCert, the runtime helpers CompCert
needs (`__i64_*`) are added, the assembly is lowered by Rivet's text pipeline and
planned together with the typed harness and libc, and a call writes the caller's
whole region as the shared output file, runs the ELF and reads the file back. The
model reads its weights and inputs from, and writes its outputs into, that mapping.

The riscv32 and host extractions are separate libraries with module names that do
not collide, so they link into one program (`test/compcert_rv32`). The riscv32
variant is built beside the host one (`make compcert.rv32.install`). Building
the extraction preprocesses CompCert's own runtime helper sources
(`runtime/c/i64_*.c`, which use `#include` and macros; the generated model C
never needs a preprocessor) with the configuration's `riscv32-linux-gnu-gcc`.
`scripts/rv32-prepro` stands in for that name with the host `gcc -E` and refuses
anything but preprocessing, so no RV32 cross toolchain is installed. The helpers'
text does not depend on the target's headers: the generated runtime data is
identical to the cross compiler's apart from the path in one line marker. A
producer that shipped the helpers already preprocessed would remove the step.

## Evidence

`test/loop_c_embed_compcert_rv32` runs the tests of `test/loop_c_embed` unchanged:
the chain graph through CompCert, Rivet and qemu is bitwise equal to the
reference evaluator, the unit is reused across contexts, and a failing
invocation is decoded from the failure record. The check was seen to fail with
`memcpy` made a no-op.

## Data model

`Mir_layout.Data_model` has `Ilp32_le` beside `Lp64_le`: 4-byte pointers, a
`ptr.add` offset as wide as a pointer (checked by the generic verifier), objects
below 2 GiB. Nothing lowers to it yet; Machine IR is still produced in LP64.

## Not done

- A Machine IR RV32 target and the Rivet route (`rivet-rv32`). It needs a pass
  that splits i64 values into i32 pairs (with ops to join and split an f64, since
  every float access is an integer load plus a bitcast), failure payloads in two
  words, the RV32 selector, and the interpreter's pointer width.
- The comparison-tool variant word for the CompCert RV32 engine, and its
  coverage-manifest row.
- `exp`, `log`, `sin`, `cos`, `ldexp`, `trunc`: not in `Rv32_libc`.
