# Building the ATen "core" C++ subset

Goal: compile a minimal slice of PyTorch's C++ (`modules/pytorch`) into a static
archive `libaten_core.a`, with **no OCaml bindings yet** — just prove the closure
builds under dune. Lives in [`modules/aten_core/`](../modules/aten_core/).

## What "core" means (and what it is NOT)

The actual operator *kernels* live in `aten/src/ATen/native/` (463 `.cpp`) plus the
codegen-generated `RegisterCPU_*.cpp` / `Operators_*.cpp` glue. Building those drags
in the whole dependency forest (MKL, sleef, fbgemm, XNNPACK, …).

The **core** subset is the layer every kernel registers *into* — the dispatcher,
IValue, type system, and Tensor metadata. It is self-contained and small:

| Layer | Files | What |
|---|---|---|
| `c10/core` + `c10/util` | 75 `.cpp` | foundation: Device, ScalarType, Storage, Allocator |
| `aten/src/ATen/core` | 45 `.cpp` | dispatcher, `ivalue`, `function_schema`, `Tensor`, boxing |
| generated core sources | 2 `.cpp` | `core/ATenOpList.cpp`, `core/TensorMethods.cpp` |

122 translation units total. Verified: all compile to `.o`, zero failures
(g++ 14.2, `-std=c++17`, CPU/static). `-Os` → ~7.9 MB archive.

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

Three rules in [`modules/aten_core/dune`](../modules/aten_core/dune), each backed by a
small shell script so the bash stays out of the sexp and is independently runnable:

| Rule | Output | Script |
|---|---|---|
| 1 | `inc/` (dir target) | `gen_macros.sh` — sed the `.in` |
| 2 | `gen/` (dir target) | `run_codegen.sh` — torchgen |
| 3 | `libaten_core.a` | `build_archive.sh` — `g++ -Os` + `ar` |

See [dune_cram_patterns.md](dune_cram_patterns.md) for the directory-target and
`%{project_root}` mechanics that make this work.

## Status / next step

Done: the closure builds end-to-end via `dune build modules/aten_core/libaten_core.a`.
Not yet: wrap the `.a` as a dune `(foreign_archive)` so an OCaml library/executable can
link it (that is the point at which bindings would start).
