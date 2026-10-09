# Native AArch64 through typed Rivet modules — implemented design

Status: **in progress**. A published AArch64 `Mir_artifact` becomes a typed
`Normalized_ast.module_` of Rivet `Aarch64.Instruction.t` values, which Rivet
lowers, lays out and (natively) loads. No assembly text is produced on this
route, and CompCert is not involved. The scalar slice of the machine stages
runs on the CPU and agrees with the physical interpreter; vectors, x86-64,
whole models and GNU export are later work.

```text
Mir_artifact (checked, realized physical program)
  -> Rivet_a64_form    selected/late/move/save forms -> Aarch64.Instruction.t
  -> Rivet_a64_module  functions, blocks, data symbols -> Normalized_ast.module_
  -> Rivet_a64_image   Driver.Pipeline.lower/plan -> Image.laid_out -> Native_exec
  -> Rivet_a64_route   one image per bundle invocation, as a Mir_model route
```

## Library and build gating

`lib/machine_rivet_aarch64` depends on the installed `rivet.*` packages plus the
machine libraries; it is built only where `rivet.aarch64` is installed
(`%{lib-available:...}`), so a checkout without Rivet builds as before. `make
rivet.install` installs Rivet alone from `vendored/rivet`; the CompCert combined
install is not a prerequisite. `test/machine_rivet_aarch64` additionally needs an
AArch64 host (`%{architecture}`), because it executes the generated code.

## Form mapping

The selected form says the operation and its immediates; the allocator's
locations say the registers, by operand position (the same contract the physical
interpreter reads). A form Rivet cannot encode is a typed `Rivet_a64_refusal`
naming it, thrown through `Err.Escape` before anything is published. Nothing is
approximated or hidden in raw instruction words.

- Immediate shifts are the bitfield aliases (`ubfiz`, `ubfx`, `sbfx`), `Ext` and
  `Trunc` are `ubfx`/`sbfx` at lsb 0, `Uxtw` and `Wtrunc` are a W `mov`.
- A symbol operand is an `Expr` (`Symbol`, plus an addend, under the `lo12`
  `Modifier` for the low twelve bits) taken from the artifact's relocation table by
  (function, block, instruction) and reference form.
- Frame traffic (`Move`, `Save`) is `ldr`/`str` of X, W, D or S against the stack
  pointer; `Sp` is `add`/`sub sp`. Quad-width transfers and NEON forms refuse.
- Late forms carry the FPCR save and restore (`mrs`/`msr fpcr`) and a large
  offset's address.

Blocks are labels in function order; a branch whose target follows falls through
(with the condition inverted when the taken side follows). Conditional branches
reach 1 MiB and `b` 128 MiB: Rivet declares no relaxation ladder for AArch64, so
a function larger than that is a planning error, not a silent truncation.

## Data and entry

Each data symbol is a global object: `Bss` and `Bound` regions in `.bss`,
constants in `.rodata`. A bound region is therefore *image-resident*: the host
writes its bytes with `Native_exec.write_global` before a call and reads results
with `read_global`; the entry takes no arguments (`Native_exec` passes an
unused pointer) and returns the status word in the integer result register. This
copies tensors in and out per call. The planned replacement binds caller tensor
regions through the entry's pointer table and moves invocation-private storage
into per-context state, so shared code owns no shared mutable scratch.

A libm helper is a host-symbol stub in a second module, because its address
belongs to the process: `movz`/`movk x16` then `br x16`, as typed instructions.
It must not be cached across processes.

## The model route

`Mir_model_route.Route.Custom` is the extension point for a back end the machine
libraries do not link. `Rivet_a64_route.route ()` supplies one: each invocation's
generic program is selected, allocated (the reference allocator, or sink
scheduling then split linear scan), framed, published and loaded as an image of
its own at `prepare` time, so a refusal is a route refusal before any call. A run
copies the invocation's bound regions (tensors, in the context's own
`Mir_memory`) into the image, calls it, copies the bound regions and the failure
record back, and decodes the status through the same `Mir_record` the
interpreters use. The CPU cannot tell which bytes a kernel wrote, so after a run
every byte of a bound region is defined: the interpreters' "an output has a byte
no invocation wrote" check is not made natively. A loaded image is closed by a
finaliser when its kernel is unreachable. The copy per call is the interim
binding described above and the dominant cost.

## Runtime mode

`Rivet_a64_runtime` declares what an image may depend on. `Dependency_free`, the
default, refuses (a typed `Helper` refusal at `prepare`) any artifact that calls a
helper bound to a C library symbol; `System_libm` admits it and the image resolves
the symbol with `dlsym` in the host process. Of the seven CI models, three
(`mobilenetv3_small_050`, `regnetx_002`, `mobilenetv2_050`) compile fully
dependency-free; the others refuse their `exp` invocations. Owning the math
helpers is what retires the system mode for them.

## GNU as an independent encoder

`lib/machine_rivet_aarch64_gnu` prints the same `Normalized_ast` module with
Rivet's `Gnu_module` (and the AArch64 `Instruction.pp_gnu`, which spells a
relocation modifier the GNU way), assembles it with GNU as, links it with GNU ld
with every section at the address Rivet bound it, and compares the loadable bytes
and the global symbol addresses. It runs processes, so it is apart from the pure
mapping library. `Gnu_module` carries no CFI because `Directive.t` has none; an
unwind description is not claimed. Agreement says two encoders read the module
alike, not what the code does.

## Rivet additions

The vendored Rivet gained what the machine stages emit and its corpus did not:
`mrs`/`msr` of FPCR, `fsqrt`, `frintz`, `fmadd`, `fmax` and `fmov` of FP bits to
a general register, each checked byte for byte against GNU `as`, plus
`Native_exec.write_global`. NEON, which the vector slice needs, is not yet there.

## Evidence

`test/machine_rivet_aarch64` runs lowered source cases (pointwise with signed
zero, NaN and binary32 boundaries, coordinate failures, matmul at odd shapes, an
`exp` kernel through libm) on the physical interpreter and on the CPU over the
same bound bytes, and compares the observations: status, the decoded failure
record and the output tensors. The native run is forked, so a faulting kernel is
a reported verdict, not the end of the test process. Four mapping defects
(`Rivet_a64_form.Mutation`: a commuted subtraction, a wrong branch sense, a
dropped low-12 address, a narrowed spill) each yield a disagreement or a
signal, which is the evidence the comparison can fail.

Whole bundles (`model_test.ml`) run the model tests' graphs through the route
against the reference evaluator, bitwise: convolution with batch norm and relu
(both pipelines, three calls with different inputs), softmax, layer norm, RMS
norm, SDPA with masked rows, `bmm`, a padded strided convolution, and a gather
whose runtime index fails in the second invocation with the same record the
interpreters store; the chain and several kernels also under the scanned
allocation. Two mapping mutations change a bundle's answer.

The census (`bin/machine_rivet_a64_census`, manual, needs the model data) runs a
model's invocations through the route against the per-node reference and, with
`--gnu`, refuses any invocation whose Rivet and GNU images differ. All seven CI
models run bitwise under both allocators with `system_libm`, and their images
agree with GNU's.

Logical work counters (`Event`) are not executed natively and are not compared.
x86-64 and are not covered here.
