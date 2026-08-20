# Building the ATen "core" C++ subset

Goal: compile a minimal slice of PyTorch's C++ (`modules/pytorch`) into a static
archive `libaten_core.a`. The C++ sources and build scripts live alongside the
OCaml bindings in [`lib/aten/`](../lib/aten/) (so the library picks up the
artifacts directly — no cross-directory copies).

## What "core" means (and what it is NOT)

The actual operator *kernels* live in `aten/src/ATen/native/` (463 `.cpp`) plus the
codegen-generated `RegisterCPU_*.cpp` / `Operators_*.cpp` glue. Building those drags
in the whole dependency forest (MKL, sleef, fbgemm, XNNPACK, …).

The **core** subset is the layer every kernel registers *into* — the dispatcher,
IValue, type system, and Tensor metadata. It is self-contained and small:

| Layer | Files | What |
|---|---|---|
| `c10/core` (incl. `core/impl`) + `c10/util` + `c10/mobile` | 89 `.cpp` | foundation: Device, ScalarType, Storage, Allocator, TensorImpl, PyObjectSlot/COW/SizesAndStrides, caching allocators |
| `aten/src/ATen/core` | 45 `.cpp` | dispatcher, `ivalue`, `function_schema`, `Tensor`, boxing |
| generated core sources | 2 `.cpp` | `core/ATenOpList.cpp`, `core/TensorMethods.cpp` |
| shim | 1 `.cpp` | the extern "C" binding surface (see below) |

137 translation units total (the c10 globs are recursive — an earlier version
used `-maxdepth 1` and silently dropped `c10/core/impl` + `c10/mobile`, which
broke real tensor construction). Verified: all compile to `.o`, zero failures
(g++ 14.2, `-std=c++17`, CPU/static). `-Os` → ~8 MB archive.

This subset is the smallest that can **construct CPU tensors and run trivial
ops** with no dispatcher, no native kernels, no `cpuinfo`, no `at::empty`:
build a tensor straight from c10 (`StorageImpl` + `TensorImpl` +
`set_sizes_contiguous`) and access its buffer via `storage().mutable_data()`.
Note: using `data_ptr<T>()` instead would pull `TensorMethods.cpp`'s
`Tensor::item()` → `at::_ops::item::call` → the whole `Operators_*.cpp` +
`record_function` layer, so the shim deliberately uses raw storage access.

## The dependency closure (empirically determined)

Found by bulk-compiling and reading the missing-header on each failure, not by
guessing. The complete set of *direct* dependencies of core is surprisingly thin:

1. **Generated config header** `c10/macros/cmake_macros.h` — CMake configures this
   from `cmake_macros.h.in`. For a default CPU/static build every `#cmakedefine`
   becomes `/* #undef */`. We reproduce it with one `sed`.
2. **torchgen-generated headers** — `ATen/core/*.cpp` won't even *parse* without
   `ATen/core/TensorBody.h` and `aten_interned_strings.h`. These come from
   `python3 -m torchgen.gen`.
3. **fmt** (header-only, `-DFMT_HEADER_ONLY=1`) — 4 c10 files; **cpuinfo** (headers)
   — `c10/core/thread_pool.cpp`. Both are PyTorch `third_party` submodules, empty by
   default: `git submodule update --init third_party/fmt third_party/cpuinfo`.

ATen/core pulls c10 only from `core` / `util` / `macros` — nothing else in the tree.

Include roots:
`-I<pt-root> -Iaten/src -Igen -Iinc -Ithird_party/fmt/include -Ithird_party/cpuinfo/include`
(`gen` holds the codegen output as `gen/ATen/...`; `inc` holds the macro header as
`inc/c10/macros/...`).

### Adding an op can mean adding C++, not just a selection line

`bin/aten_ops_gen.ml`'s `selection` decides what gets a *binding*.
`lib/aten/build_archive.sh` decides what gets *linked*, and the two are separate
lists. A one-line selection change that names an op whose kernel is not in the
archive produces a **link error**, not a build success with a missing arm — the
opposite failure mode from `Interp_decode`'s silent skip (see
`.ai/interpreter_design.md` §3a), and much easier to notice.

`aten.pad.default` is the worked example, and it shows all three moves:

- **add the sources.** `native/PadNd.cpp` (`pad_symint`, `_pad_enum_symint`,
  `constant_pad_nd`), plus `ReflectionPad.cpp` and `ReplicationPadding.cpp`,
  plus `cpu/PaddingKernel.cpp` in the CAP list. Replicate and circular are
  refused by the native engine and still have to link: `_pad_enum_symint`'s mode
  switch is straight-line, so every arm is *referenced*.
- **remove a stub that became a duplicate.** `constant_pad_nd` was in
  `atg_stubs.cpp` only because `Convolution.cpp` references it for `same`
  padding; with `PadNd.cpp` real, keeping the stub is a duplicate-symbol error.
- **add a stub for what the new sources drag in.** `ReflectionPad.cpp` has
  quantized variants calling `at::_empty_affine_quantized` straight-line, which
  static dispatch resolves to `empty_affine_quantized_other_backends_stub` —
  a cold path the dense-CPU float engine never takes, and one whose real
  definition is itself only a better error message.

The general shape: link, read the undefined symbol, decide *source or stub* by
asking whether the dense-CPU float path can reach it, and record the answer in a
comment beside the line. Guessing from the op's name does not work — the closure
is empirical, exactly as the section above says.

## torchgen invocation quirks

`python3 -m torchgen.gen --source-path <pt>/aten/src/ATen --install_dir <abs> --generate <what>`

- **`--install_dir` must be ABSOLUTE.** With a relative path, header generation is
  silently skipped (sources still appear) — a long debugging trap.
- **Run `headers` and `sources` as two separate invocations.** Passing
  `--generate headers --generate sources` in one call drops the headers.
- **Omit `--per-operator-headers`** — keeps the 25 monolithic headers and avoids the
  `ops/` per-operator explosion. Simpler and sufficient for core.
- Needs python `pyyaml` + `typing_extensions` (`apt: python3-yaml
  python3-typing-extensions`).
- `torchgen` is a package inside the submodule, so set `PYTHONPATH=<pt-root>`.

## Dune wiring

Three rules in [`lib/aten/dune`](../lib/aten/dune), each backed by a
small shell script so the bash stays out of the sexp and is independently runnable:

| Rule | Output | Script |
|---|---|---|
| 1 | `inc/` (dir target) | `gen_macros.sh` — sed the `.in` |
| 2 | `gen/` (dir target) | `run_codegen.sh` — torchgen |
| 3 | `libaten_core.a` + `dllaten_core.so` | `build_archive.sh` — `clang++ -O3` (+ `-mavx2 -mfma` if the build machine supports it, see [pt2_inference_perf.md](pt2_inference_perf.md)) + `ar` |

These rules emit the archive into `lib/aten/`'s build dir, exactly where the
`(library (name aten) ...)` stanza's `(foreign_archives aten_core)` looks for it.

See [dune_cram_patterns.md](dune_cram_patterns.md) for the directory-target and
`%{project_root}` mechanics that make this work.

## OCaml bindings (ctypes, static stubs)

Bindings are added incrementally, dependencies first, working toward
`aten::add.Tensor` (per-element add: `add.Tensor(Tensor, Tensor, *, Scalar alpha=1)`
-> `at::add`). The C++ surface is C++ (mangled, non-POD by value), so every binding
goes through a hand-written `extern "C"` shim ([shim.h](../lib/aten/shim.h) /
shim.cpp = Tensor API; [ops.h](../lib/aten/ops.h) / ops.cpp = operations)
trading only C scalars and opaque pointers. Pattern mirrors
[TheCBaH/ocaml-ggml](https://github.com/TheCBaH/ocaml-ggml/tree/main/lib/ggml).

Mechanics (no runtime `.so` loading — static linking + generated stubs):
- `shim.cpp` is compiled into `libaten_core.a`; `build_archive.sh` also emits
  `dllaten_core.so` (`--whole-archive`) for ctypes' bytecode path.
- The `(library (name aten) ...)` uses dune's `(ctypes ...)` stanza:
  `type_description.ml` (`Types` functor) + `function_description.ml` (`Functions`
  functor, one `foreign "atc_..."` per shim fn), `(foreign_archives aten_core)`,
  vendored resolver with `c_library_flags ... -lstdc++ -lpthread`, preamble
  `#include "shim.h"`, `(generated_entry_point C)` -> callable as `Aten.C.Functions.*`.
- Needs `(using ctypes 0.3)` in dune-project, opam `ctypes` + `ctypes-foreign`, and
  apt `libffi-dev` + `pkg-config` (wired into devcontainer.json + Dockerfile).

The library, its C++ sources, and the demo all live under [lib/aten/](../lib/aten/)
(binding) and [lib/aten/demo/](../lib/aten/demo/). Because the archive rules and
the library share one directory, `libaten_core.a` / `dllaten_core.so` are produced
right where `(foreign_archives aten_core)` and the preamble `#include "shim.h"`
need them — no copy rules. (`modules/` is reserved for git submodules.)

## Result ownership across the C ABI

Every handle crossing into OCaml is owned by OCaml and freed exactly once, via
`Aten_tensor.manage`'s finaliser. There are three result shapes, and they differ
in more than arity:

| Schema return | C ABI | OCaml |
|---|---|---|
| `Tensor` | returns `atc_tensor` | `check \|> manage` |
| `(Tensor, Tensor[, Tensor])` | `int` status + `struct atc_tensorsN*` out-param | `bind_many`, one `manage` per field |
| `Tensor[]` | returns `atc_tensor_list` | `Aten_tensor_list.to_list` |

The `Tensor[]` case is the only one whose length is not known statically, so it
needs a container rather than an out-param struct. `atc_tensor_list` is an opaque
owned `std::vector<at::Tensor>` with a deliberately small lifecycle API
(`_len` / `_get` / `_free` / `_live_count`).

**Container ownership and storage aliasing are separate.** `atc_tensor_list_get`
returns an independently freeable `atc_tensor` that still shares ATen storage and
view metadata with the op's input — `at::unbind` returns views, and this ABI
preserves them. `at::Tensor` is intrusive-reference-counted, so destroying the
container drops only its own handles; tensors already extracted stay valid. That
is why `Aten_tensor_list.to_list` can free the container before returning.

Do **not** reuse `atc_to_tensor_vector` for this: it is input-only and borrowing
(the vector is a temporary in the wrapper's `at::<op>(...)` full-expression, and
the `ArrayRef` borrows it), so it cannot express ownership of a returned vector.

Two counters, because a container is not a tensor handle: `atc_live_count` covers
`atc_wrap` allocations, `atc_tensor_list_live_count` covers containers. The
leak tests assert both — the container synchronously (`Fun.protect` frees it), the
tensors only after the GC has run their finalisers.

`atc_tensor_list_len` / `_get` check for NULL **before** dereferencing rather
than relying on `ATC_TRY`: that macro catches `std::exception`, and a null
dereference is undefined behaviour, not a throw. `ATC_TRY` still guards the
`at::Tensor` copy inside `_get`, which genuinely can throw. The sentinels are
`-1` and `NULL` with `atc_last_error` set, which `Aten_tensor_list` turns into
`Aten_tensor.Error`.

## The flat-`Bigarray` boundary

> Before exposing an ATen tensor through the flat `Aten_tensor.data` Bigarray
> boundary, materialize it when its layout or storage offset makes that flat
> representation unsafe.

`atc_data_ptr` returns `storage().mutable_data()` — the storage **base**. It
applies neither strides nor the storage offset, and `Aten_tensor.data` builds a
`numel`-long Bigarray straight off it. So the flat view is faithful only for a
tensor that is contiguous **and** at offset zero, and a reader must check both:

- a **strided** view (`permute`, `expand`) yields the right elements in the wrong
  order;
- an **offset** view (`select`, `slice`, `unbind`) yields another slice's
  elements entirely — and `atc_is_contiguous` reports it **contiguous**, because
  `is_contiguous` does not consider the storage offset. An `is_contiguous`-only
  guard therefore passes exactly the dangerous case straight through. That was a
  live defect: `Tensor_bridge.of_aten` and `Verify.logical_tensor` both had it,
  and `select.int` is in the curated selection.

`atc_storage_offset` exposes the missing half, and `Aten_tensor.materialize_for_raw_read`
is the single guard: it returns `t` unchanged when a flat read is faithful, and
otherwise a **managed** dense copy at offset zero. It uses `clone`, not
`contiguous` — `at::contiguous` no-ops on anything already contiguous and hands
back the same offset view, so only a copy relocates the data.

`Aten_tensor.data` stays unguarded on purpose: it means "expose this tensor's
underlying flat storage", write-through included, and cloning inside it would
silently break the callers that want the view. Materializing is the reader's
explicit decision — see `native_aten_bridge_design.md` and
`aten_native_verify_design.md` for the reader-side rule.

It is **not** required that every `Aten_tensor.t` own dense storage. Views are
valid inputs to other ATen ops, and to `shape` / `strides` / `equal` / `allclose` /
`to_string`, all of which understand layout natively. Only the flat boundary
needs the copy.

Note the resulting coupling: `Aten_tensor` now depends on the *generated* ops for
`clone`. Dropping `op "clone"` from the curated selection breaks compilation of
`aten_tensor.ml` — a loud failure, which is why it is acceptable.

Done — **Step 0 (plumbing)** + **Step 2 (minimal tensor runtime)**: the shim now
exposes `atc_new_float` / `atc_free` / `atc_numel` / `atc_data_float` /
`atc_fill_float` (scalar) / `atc_add_float` (tensor). [lib/aten/demo/main.ml](../lib/aten/demo/main.ml)
creates two `2x3` CPU tensors, fills one with a scalar, adds them, and prints
`a+b = [10; 11; 12; 13; 14; 15]` — verified by [demo.t](../lib/aten/demo/demo.t).
The op is a direct buffer loop (no dispatcher) — intentionally, to stay on the
minimal "core" subset.

Decided plan (user): keep the hand-written shim; then build a **parser** (extend
[lib/aten_schema](../lib/aten_schema/)) and an incremental **generator** gated by a
controlled allow-list of funcs/types (rest handled manually) that emits extern "C"
shims + ctypes from the schema — ocaml-torch's hand-core + generated-ops split.
Dispatched `at::add` (which needs growing the build to CPU `aten_cpu`: native
kernels + `RegisterCPU` codegen + `cpuinfo`) is deferred until the generator path
needs it. For calling *real* ATen ops dispatch-free and SIMD-free when that time
comes, see the proven recipe in [aten_static_dispatch.md](aten_static_dispatch.md).
