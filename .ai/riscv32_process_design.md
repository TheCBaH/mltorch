# RISC-V 32 processes under qemu-riscv32

Status: **the embedded-CompCert route and the Rivet Machine IR route run**, both
D-free for binary32 programs. RV32 code cannot run on the host CPU, so every run is a
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
| `Rv32imf_ilp32f` (default) | `none,+m,+f` | ILP32F, `rv32imf` | `memcpy`, `memset`, `fabsf`, `fmaf`, `sqrtf`, `truncf` | `-cpu rv32,d=false` |
| `Rv32imf_zve32f_ilp32f` | `none,+m,+f,+zve32f` | ILP32F, `rv32imf_zve32f` | as the default | `-cpu rv32,d=false,zve32f=true,vlen=128,elen=32` |
| `Rv32imf_zve32f_zvfbfwma_ilp32f` | `none,+m,+f,+zve32f,+zfbfmin,+zvfbfwma` | ILP32F, `rv32imf_zfbfmin_zve32f_zvfbfwma` | as the default | the vector one plus `zfbfmin`, `zvfbfmin`, `zvfbfwma` |
| `Rv32imafd_ilp32d` | `none,+m,+a,+f,+d` | ILP32D, `rv32imafd` | the above plus `fabs`, `fma`, `sqrt` | default |

Every feature set starts from `none`: Rivet's default configuration enables
every component it has (C, B, the vector profiles and more), which no profile
claims, and its requirements close each set (F brings Zicsr, Zve32f Zve32x,
Zvfbfwma Zvfbfmin). The ELF ISA string is Rivet's `Riscv_isa.to_string` of that
set under the 20191213 unprivileged specification, the edition GNU as and LLVM
default to (`rv32i2p1_m2p0_f2p2_zicsr2p0_zmmul1p0` for the default), so it
names exactly what the assembler admits; Rivet pins its output against LLVM 19.

The default targets an RV32 core without the D extension (Coral NPU's are
RV32IMF). Rivet's RISC-V `d` component owns every form that reads or writes a
binary64 value, so without `d` a D instruction in a typed module is refused when
the module is lowered, naming each form; qemu with D disabled is the second,
independent check, turning one that reached execution any other way into
SIGILL. Each was shown to fail alone: without `-d` a binary64 kernel dies in
qemu; with neither, it runs and the test's expectation breaks.

The vector profile is Coral NPU's vector core: Zve32f (binary32 elements, no
binary64 vector forms) at VLEN 128. Rivet's `zve32x`, `zve32f` and `zve64x`
components own the vector forms, so the two scalar profiles refuse a vector
instruction when it is lowered, and qemu without Zve32f turns one into SIGILL
(`test/machine_rivet_riscv32/vector_test.ml`; enabling `zve32f` in the default
profile turns that test red through qemu). The same test runs one
`vsetvli`-strip-mined image at VLEN 128, 256 and 1024, LMUL 1 and 8, with
qemu's `rvv_ta_all_1s`, which fills tail elements with ones.

The BF16 vector profile adds what the same core runs for BF16: Zvfbfwma's
widening multiply-add (`vfwmaccbf16`) and the Zfbfmin and Zvfbfmin it requires,
the BF16 part of upstream Coral NPU's `-march`
(`rv32imf_zve32f_zicsr_zifencei_zbb_zfbfmin_zvfbfmin_zvfbfwma`). Zfbfmin brings
`flh`, `fsh` and the half-word moves, so the core's missing Zfhmin stays off and
its conversions (`fcvt.s.h`, `fcvt.h.s`) are refused. The emulator release's
manifest lists no Zvfbfwma for the RVV cores, though their RTL runs
`vfwmaccbf16`.

The text pipeline (`Riscv32`) and the encoder (`Riscv32_encode`) each have a
state; `Rv32_profile.config` is the one feature set both are made from.

The vector code assumes VLEN 128 at least (vl 4 at SEW 32), so under the
vector profile `_start` reads `vlenb` (CSR 0xc22) and exits 105 below 16
bytes: a qemu run at `vlen=64` does, and the same image is exact at 128, 256,
512 and 1024. qemu starts a process with `vill` set: a vector form before any
`vsetivli` is SIGILL (the `Unset_vtype` mutation shows it), so a missing
configuration fails loudly rather than computing at a stale one.

`truncf` converts toward zero only below 2^23 in magnitude (where a binary32
can have a fraction) and copies the input's sign onto the result, so a zero
result keeps it; larger values, infinities and NaNs come back unchanged. Its
test compares it bitwise with the host's `Float.trunc` at the binary32
boundaries and over a seeded sweep, under `d=false`.

The embedded-CompCert loader uses only `Compcert_embed_riscv32_fp32`, the
separate extraction for the F32 ABI, through `Compcert_rv32_loader.Ilp32f` on
`Rv32imf_ilp32f`. Neither its build dependencies nor `make compcert.rv32.install`
select the ordinary `Compcert_embed_riscv32` extraction. That ordinary D-ABI
backend can emit `fmv.d` and `fld`/`fsd` even for binary32 source; those
observations do not describe the F32 ABI extraction. Assembly from the F32
extraction and its runtime is lowered through Rivet's text pipeline with D
disabled, then QEMU executes with `d=false`.

RISC-V comparison execution requires `Fp32_ordered` (`compcert-rv32,fp32` or
`compcert-rv32-fp32`); the binary64 `compcert-rv32` word refuses before compiling.
The optional D profile of the typed assembly harness remains useful for
negative ISA tests; it does not select a CompCert extraction.

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

Binary32 Loop C in the `Compcert_ilp32` dialect takes no binary64 step
(`Loop_c_dialect.binary32_only`): the F-only CompCert refuses binary64
arithmetic, compiles a 64-bit integer to `float` as a call to
`__compcert_i64_stof` that its runtime lacks, and refuses `float` to a 64-bit
integer. So under `Fp32_ordered` that dialect decodes f16 and bf16 with
`f16_to_f32`/`bf16_to_f32`, converts an `int64_t` to `float` with
`f32_from_i64` (round to nearest even in integer arithmetic; within 32 bits the
target's own conversion) and back with `i64_from_f32` (and
`i64_from_f32_failure`, whose record holds the value widened to binary64, as
before). Every dialect computes the same bits; only this one's text changes.
`test/loop_c/binary32_only_test.ml` checks the helpers against the host's
conversions and the binary64 helpers: every f16 and bf16 pattern, and the
int64 and float boundaries plus a seeded sweep (about 400k values each);
rounding ties up, or shifting by one bit, fails it. Through it
`test_convnext2`'s 49 units equal `loop-c-fp32` bitwise under qemu with D
disabled and on MPACT.

The riscv32 and host extractions are separate libraries with module names that do
not collide, so they link into one program (`test/compcert_rv32`). The F32 ABI riscv32
variant is built beside the host one (`make compcert.rv32.install`). Building
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

`test/loop_c_embed_compcert_rv32_fp32` runs the binary32 embedding tests
(`embed_fp32_test.ml`: the binary32 oracle, Region nodes and the failure record)
with `Ilp32f` under `-cpu rv32,d=false`. Pointing that loader at the ordinary
D-ABI extraction instead was seen to fail at lowering on `fmv.d`. The legacy
D-ABI embedding suite has been removed; RISC-V CompCert execution uses the F32
ABI exclusively. `test/compcert_rv32` links that extraction with the host's.

## The Machine IR route

`lib/machine_rivet_riscv32_mir` makes each invocation of a bundle a process:
the generic program is selected onto ILP32 (`Rv32_select`), allocated (sink
scheduling and linear scan by default), framed and published, then
`Rv32_mir_module` emits it as one typed Rivet module and `Rv32_harness` plans
it into a static ELF, built once per invocation. The image is table-bound:
s11 holds the table of region addresses, which the module's `mir_entry` takes
from a0 (saving s11 and ra around the kernel), so a view's address is a load
from its region's slot plus its offset, and only constants are in the image.
A bound region and the failure record are slots of the shared output file
(written before the run, read back after); scratch regions are the ELF's
zero-filled data. Each run is one qemu process with D disabled.

RISC-V has no short and long branch and Rivet no RV32 relaxation, so the
module writer lays each function out itself: every item's size is known (all
instructions are four bytes, `la` and `call` eight), a conditional branch
beyond 4000 bytes becomes the inverted branch over a `jal`, and the layout
repeats until no branch changes. `test/machine_rivet_riscv32_mir`: source
kernels (failures included) and i64 programs agree with the generic route and
the SSA oracle; the binary32 bundles and transcendentals are bitwise equal to
the binary32 oracle; a branch over 1100 and 3000 instructions is promoted and
taken both ways; a branch's sense, a select's arms, a binary32 subtraction's
operands and a region's table slot, each mutated, are caught by the
processes' outputs.

Per-form conformance (`conformance_test.ml`, the counterpart of the AArch64
native harness, with no cross compiler): every admitted form instance —
immediates, select aliasings, access sizes and offsets, branch conditions —
is emitted by the route's own mapping and run in one process over boundary
operands (pairs of 20 integers, 24 binary32 patterns with signed zeros,
subnormals, infinities and quiet and signalling NaNs, triples for `fmadd.s`),
and its results and the memory a store wrote equal `Rv32_sem`'s bit for bit:
136 instances, 17619 vectors. maximum for `fmax.s`, 0 for `fcvt.w.s` of a
NaN, 0 for a division by zero, a zero-extended `sltiu` immediate and a
non-canonical NaN result, each put in the semantics, and the mapping's commuted
subtraction, swapped select arms and branch sense, are each caught.


Vectors: an artifact whose features include Zve32f is built and run under the
vector profile. Under qemu, the Zve32f kernels (pointwise, masks, truncation,
matmul) agree with the generic route and the oracle at VLEN 128, 256, 512 and
1024 with `rvv_ta_all_1s` and `rvv_ma_all_1s`, and the binary32 bundles and
transcendentals planned for `zve32f128` (forced, for the transcendentals) are
bitwise equal to the binary32 oracle at 128 and 1024. The integer-lane decodes (i8 and i16
dequantization, every binary16 and bfloat16 encoding) agree at 128 and 1024. Vector per-form
conformance (`vconformance_test.ml`) runs 77 instances, 3381 vectors of lanes
from the scalar boundary values (select and insert destinations aliasing every
operand, strides 0 to 12), exact at VLEN 128, 512 and 1024; maximum for
`vfmax.vv`, compare lanes of 1, a zero extension for `vsext`, a strided load
read contiguously, and the
mapping's commuted subtraction and swapped merge arms are each caught.

In the comparison tool the route is the variant word `rivet-rv32-fp32`
(`Inference_compare_rv32.Mir_rv32_engine`: the scalar `Fp32_ordered` plan,
linear scan, one qemu process per invocation); `rivet-rv32` names no D
extension and refuses. On the chain fixture it is bitwise equal to
`loop-c-fp32` and to `compcert-rv32-fp32`; the coverage manifest pairs it
with the CompCert engine. `bin/machine_rivet_rv32_census` (manual, needs the
model data) compiles a model's every invocation, tallying refusals, and runs
the schedule once against the binary32 oracle. `make rv32.pt2.runtest` runs it on
`RV32_PT2_MODELS` (the fixture models by default) scalar and Zve32f at VLEN
128 and 1024, failing on any difference. Measured 2026-10-18, every CI model
(test_convnext2, mobilenetv2_050, regnetx_002, efficientnet_b0,
mobilenetv3_small_050, fastvit_sa12) compiles every invocation and is bitwise
equal to the binary32 oracle on all three; under Zve32f between 20% and 51% of
the selected forms are vector forms (efficientnet_b0 46385 of 91701).

## Data model

`Mir_layout.Data_model` has `Ilp32_le` beside `Lp64_le`: 4-byte pointers, a
`ptr.add` offset as wide as a pointer (checked by the generic verifier), objects
below 2 GiB. Generic Machine IR is still produced in LP64; the RV32 selected
program is `Ilp32_le` (see the Machine IR design record).

## Not done

- `exp`, `log`, `sin`, `cos`, `ldexp`, `trunc`: not in `Rv32_libc`.
