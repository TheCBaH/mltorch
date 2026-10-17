# RISC-V 32 processes under qemu-riscv32

Status: **the embedded-CompCert route runs**, D-free for binary32 programs. The Rivet Machine IR route does not
exist yet (see "Not done"). RV32 code cannot run on the host CPU, so every run is a
static ELF executed by `qemu-riscv32`; it checks instruction bytes and control
flow, not an RV32 core's timing or memory ordering.

## The process

`lib/machine_rivet_riscv32` builds the process from typed Rivet modules, with no
assembly text on the way, and writes it with `ocaml-elf` (`Elf.Exec`), for one
`Rv32_profile`:

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

## Profiles: binary32 by default, no D instruction

`Rv32_profile.t` is the one choice the assembler's feature set, the ELF flags,
`Rv32_libc` and qemu's CPU are read from, so they cannot disagree:

| Profile | Rivet features | ELF ABI | `Rv32_libc` | qemu |
|---|---|---|---|---|
| `Rv32imf_ilp32f` (default) | `-d` | ILP32F, `rv32imf` | `memcpy`, `memset`, `fabsf`, `fmaf`, `sqrtf`, `truncf` | `-cpu rv32,d=false` |
| `Rv32imafd_ilp32d` | all | ILP32D, `rv32imafd` | the above plus `fabs`, `fma`, `sqrt` | default |

The default targets an RV32 core without the D extension (Coral NPU's are
RV32IMF). Rivet's RISC-V `d` component owns every form that reads or writes a
binary64 value, so under `-d` a D instruction in a typed module is refused when
the module is lowered, naming each form; qemu with D disabled is the second,
independent check, turning one that reached execution any other way into
SIGILL. Each was shown to fail alone: without `-d` a binary64 kernel dies in
qemu; with neither, it runs and the test's expectation breaks.

`truncf` converts toward zero only below 2^23 in magnitude (where a binary32
can have a fraction) and copies the input's sign onto the result, so a zero
result keeps it; larger values, infinities and NaNs come back unchanged. Its
test compares it bitwise with the host's `Float.trunc` at the binary32
boundaries and over a seeded sweep, under `d=false`.

The embedded-CompCert loader has one instance per CompCert profile, each paired
with the `Rv32_profile` its code needs. `Ilp32d` is CompCert's riscv32 target,
which is ilp32d and moves binary32 values with `fmv.d`, spills FP callee-save
registers with `fld`/`fsd` and expands aligned copies through them, so its
binary32 programs still contain D instructions: it keeps `Rv32imafd_ilp32d`.
`Ilp32f` is CompCert's F-only profile (`riscv32_fp32`, `rv32imf`/ILP32F), on
`Rv32imf_ilp32f`. Both lower CompCert's assembly through Rivet's text pipeline
from the profile's state, so under `Ilp32f` a D instruction is refused at
lowering (`fmv.d requires feature d`) before qemu would trap on it.

Rivet's `li` covers only a signed 12-bit constant or one with the low 12 bits
zero, so `Rv32_asm.load_imm` expands any other constant to `lui` + `addi`.

## CompCert on RV32

`Loop_c_embed.Make` takes a LOADER, with a `dialect`. `Compcert_ilp32` is the
`Compcert_scalar` C for a target whose `long` is 32 bits: `int64_t` is `long long`
and constants end in `LL`. `lib/loop_c_embed_compcert_rv32` is the loader: the
unit is compiled by the embedded riscv32 CompCert, the runtime helpers CompCert
needs (`__i64_*`) are added, the assembly is lowered by Rivet's text pipeline and
planned together with the typed harness and libc, and a call writes the caller's
whole region as the shared output file, runs the ELF and reads the file back. The
model reads its weights and inputs from, and writes its outputs into, that mapping.

The riscv32 and host extractions are separate libraries with module names that do
not collide, so they link into one program (`test/compcert_rv32`). Both riscv32
variants are built beside the host one (`make compcert.rv32.install`). Building
an extraction preprocesses CompCert's own runtime helper sources
(`runtime/c/i64_*.c`, which use `#include` and macros, and the F-only profile's
`riscv-fp32.c` conversions; the generated model C never needs a preprocessor)
with the configuration's `riscv32-linux-gnu-gcc`. `scripts/rv32-prepro` stands
in for that name with the host `gcc -E`, dropping the RISC-V `-march`/`-mabi`
flags the host compiler rejects, and refuses anything but preprocessing, so no
RV32 cross toolchain is installed. The helpers'
text does not depend on the target's headers: the generated runtime data is
identical to the cross compiler's apart from the path in one line marker. A
producer that shipped the helpers already preprocessed would remove the step.

## Evidence

`test/loop_c_embed_compcert_rv32` runs the tests of `test/loop_c_embed` unchanged
with `Ilp32d`: the chain graph through CompCert, Rivet and qemu is bitwise equal
to the reference evaluator, the unit is reused across contexts, and a failing
invocation is decoded from the failure record. The check was seen to fail with
`memcpy` made a no-op. `test/loop_c_embed_compcert_rv32_fp32` runs their binary32
half (`embed_fp32_test.ml`: the binary32 oracle, Region nodes, the failure
record under `Fp32_ordered`) with `Ilp32f` under `-cpu rv32,d=false`; pointed at
the ilp32d embedding instead, it fails at lowering on CompCert's `fmv.d`.

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
- Binary32 Loop C still uses binary64 helpers for the f16 and bf16 decoders and
  int64-to-float conversion, which the F-only CompCert refuses (binary64
  arithmetic) or cannot link (its runtime has no `__compcert_i64_stof`); the
  graphs the F-only suite runs need neither.
