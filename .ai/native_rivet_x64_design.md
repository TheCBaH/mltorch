# Native x86-64 through typed Rivet modules — implemented design

Status: **runs on an x86-64 CPU**. A published x86-64 `Mir_artifact` becomes a
typed `Normalized_ast.module_` of Rivet `X86_64.Instruction.t` values, built
through the encoder's surface constructor (a mnemonic and typed operands, never
assembly text). Rivet lowers and lays it out, and it runs two ways. On an
x86-64 host `Rivet_x64_image.load` maps it into this process (`Native_exec`) and
the table-bound `Rivet_x64_host` calls it with the context's tensors passed by
address, as the AArch64 host does. Anywhere, `Rivet_x64_process` writes the
bound image as a static ELF and runs it as a process: directly on an x86-64
host, under `qemu-x86_64` (binary translation of the same bytes, not a CPU)
elsewhere. `Rivet_x64_process.runner` says which, and every report prints it:
only a `Cpu` run is evidence about flags, MXCSR, timing or memory ordering. The
shared parts (the table of region addresses, the runtime mode, CFI) are
`lib/machine_rivet_common`; the AArch64 design is in the sibling record.

```text
Mir_artifact (checked, realized physical program)
  -> Rivet_x64_form    selected/late/move/save/sp forms -> X86_64.Instruction.t
  -> Rivet_x64_module  functions, blocks, data, entry wrapper -> Normalized_ast.module_
  -> Rivet_x64_image   Pipeline_direct plan; load into this process, or bind at fixed addresses
  -> Rivet_x64_host    table-bound model host: loaded kernels, contexts, tensors by address
  -> Rivet_x64_elf     a bound image -> a static ELF executable
  -> Rivet_x64_process harness module, prepare once, launch per run (CPU, else qemu-user)
  -> Rivet_x64_route   one prepared process per bundle invocation, as a Mir_model route
```

`Rivet_x64_cpu` checks the features an artifact was selected for against the
flags in `/proc/cpuinfo` (`sse2`, `sse4_1`, `fma`, `avx`, `avx2`) before the
host loads anything; a CPU whose flags cannot be read is refused.
`Rivet_x64_manifest` is the AArch64 manifest for this target: features,
planning, runtime mode, region binding, declared and carried helpers, a digest
of the code.

`lib/machine_rivet_x86_64_gnu` prints the same module through Rivet
(`Gnu_module` with `Instruction.pp_gnu`), has the cross binutils assemble and link
it at Rivet's addresses, and compares loadable bytes and global symbol addresses;
Rivet's own text route reads the print back and must agree too.

## Form mapping

Operands are AT&T-ordered (source first). Tied forms have the destination in the
location of the tied use; Cmov and Fmadd tie to their last use. 8- and 16-bit
register names exist only inside `Ext`, `Trunc_zx`, `Setcc_zx` and byte/half
stores; those need an empty REX for `%spl..%dil`, which Rivet now emits. A 64-bit
immediate is `movq $imm, reg` (Rivet accepts `movabsq` only outside the signed
32-bit range). The compare predicates are `cmpsd $0|3`, not the pseudo-mnemonics.
`Lea_view` is `leaq sym(%rip)` for resident data and a load from the table (`rbp`
slot) for a table-bound region. Branches fall through when the target follows,
inverting the condition.

## Entry wrapper and data

Regions that a caller owns are reached through a table of base addresses held in
`rbp`, which the target reserves and the System V ABI makes callee-saved. The
`mir_entry` wrapper pushes `rbp`, reserves 16 bytes (the kernel is then entered
with the stack at 8 mod 16, as a callee expects), saves MXCSR, sets it to round
to nearest with every exception masked, calls the kernel, and puts everything
back. The kernel itself never touches MXCSR (`X64_frame.fp_control = None`).

The process is the artifact's module plus a harness: one `.data` block holding a
status word, a 128-byte ABI-probe record and every mutable region, the table of
their addresses, and a `_start` that calls `mir_entry`, stores the status and
writes the block to standard output. The code does not change between runs, so an
image is planned and bound once (`prepare`) and each run (`launch`) patches the
block's bytes in the bound segment, writes the ELF and reads the block back.

## The ABI probe

With `~probe`, `_start` first sets `rbx rbp r12-r15` to sentinels and MXCSR to
0xFFC0 (round toward zero, flush to zero, denormals are zero, all masked), then
calls the entry and records the registers, `rsp` and MXCSR. The libm helpers the
artifact names are replaced by stubs that record `(rsp+8) land 15` and clobber
every caller-saved register and `xmm0-15` before returning zero, so a kernel
that relies on a caller-saved value, or is entered misaligned, is seen. Only the
probe record is meaningful for such a run.

## Runtime mode

As on AArch64: the default refuses a kernel that calls a C-library helper the
project does not own. `exp` is owned (see the owned `exp` design below), so a
default image carries it and is not refused. A loaded image binds the other
helpers through Rivet's typed trampolines to this process's libm
(`system_libm`); the static process has no libm, so there `system_libm` is only
meaningful with the probe's stubs.

## Vectors

Packed forms are SSE2 on XMM registers (FMA and SSE4.1 only under their
features). The selector takes, besides the arithmetic: packed compares into masks
(`CMPPS/CMPPD`, predicates equal, less, less-or-equal, unordered; a mask is a
register with every bit of a lane set or clear, so a NaN is false for every
ordered compare); mask logic and not (`pcmpeqd` makes the ones); a mask select as
`b ^ ((a ^ b) & m)`; a predicate splatted to a mask; IEEE maximum from `MAXPS`
repaired as the scalar `fmax` is, with masks; a strided load or store as one
scalar access per lane in lane order; and a lane insert from `SHUFPS/SHUFPD`
alone (the scalar is splatted, then lanes are picked from it and the vector;
`INSERTPS` would need SSE4.1). A tie needs equal types, so the destructive
compare takes a mask-typed copy of its first operand (`Movap_to_mask`). Still
refused: half-width compares, and any vector the split leaves in a shape that is
not four binary32 or two binary64 lanes.

## Host

`Rivet_x64_host` is the AArch64 host over this target: each invocation is
published and loaded once, tensors are the context's own storage passed through
the table in `rdi` (the entry moves it to `rbp`), scratch and the failure record
belong to the context, an image is immutable and shared, a first failure
names its record and the context goes on. `?features` and `?cpuinfo` choose what
the artifact may use and what the CPU is taken to report.

## Owned `exp`

The project's own binary64 `exp` is the algorithm and the constants of glibc
2.41's table-driven `exp` (N = 128, degree-5 polynomial, round-to-nearest on
the reduction, fused multiply-adds where the AArch64 library fuses). The
specification is `Machine_ir.Mir_exp.model`, checked equal to the host libm on
1.6 million inputs (random, bit patterns, the two thresholds, every
half-integer reduction). Each realization is a leaf of typed Rivet
instructions that follows the model operation for operation: `Rivet_x64_exp`
here (it needs FMA3, and rounds half away from zero by truncating and
correcting, since SSE4.1 rounding has no such mode) and `Rivet_a64_exp` on
AArch64. Each is checked bit for bit against the model through a driver that
calls it over a batch; each check was seen to fail under a changed constant,
a changed rounding and an unfused step. The image carries the code in its own
`.rodata`; the AArch64 manifest records it by digest. Another libm (a
different rounding of the same algorithm, or a different algorithm) is a
documented disagreement with the reference, never a tolerance. `cos`, `log`
and `sin` remain refused by the default mode.

## Conformance

`make machine.rivet.x64.conformance` runs the form instances over boundary and
random vectors, on the CPU of an x86-64 host. Each form is a batch process: a loop over records
loading operands into the registers the form's locations name, seeding RFLAGS and
the destination, running the form's Rivet instructions, and storing results,
flags and the buffer. The model is `X64_sem.exec` on the same operands under the
form's defined bits, `result_write` mask and flag mask. A NaN is one class
(payload and sign are not modelled and the CPU's default NaN differs from the
interpreter host's); a pointer result is compared as its offset from the base.
`--gnu` checks each form's module against the cross binutils. Five model
mutations and two mapping mutations must be seen; the immediates around the byte
rung (127, 128, 255, 256, -128, -129) are enumerated because the form that
motivated them was wrong only there.

## Rivet additions

All local on branch `native-asm`, with tests: the empty REX for a byte-register
source of `movzx/movsx`; `imul` three-operand immediates reduced to the operand
width and required to read back through the sign-extended byte rung; `pp_gnu`
(size suffixes the text parser needs, `movq` for a 64-bit `movd`, SSE mnemonics
whole); `.2byte/.4byte/.8byte` in the x86 and AArch64 directive tables.

## Not done

A derived object file kept as an artifact; branch-range negatives; the other libm
helpers (`cos`, `log`, `sin`); AVX and AVX2; half-width compares; frequency-pinned
timing.
