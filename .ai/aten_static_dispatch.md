# Dispatch-free, SIMD-free ATen: the static-dispatch recipe

How to call *real* ATen CPU ops from the minimal build **without** the operator
`Dispatcher` and **without** vectorized SIMD kernels. Empirically validated
2026-06-13 (scratch, g++ 14.2, aarch64). Companion to
[aten_core_build.md](aten_core_build.md); the goal is to eventually let the
binding generator emit real ops on top of the minimal `libaten_core.a`.

## TL;DR of the findings

| Concern | Result |
|---|---|
| Remove the `Dispatcher` | `torchgen --static-dispatch-backend CPU` makes each `at::_ops::X::call` a **direct** `at::cpu::X(...)` call. Verified: 0 dispatcher symbols left. |
| Avoid force-linking all kernels | `--skip-dispatcher-op-registration` emits **empty** `TORCH_LIBRARY_IMPL` (0 `m.impl`) — no static-init pulls kernels. |
| Bound the closure | `-ffunction-sections -fdata-sections` + `-Wl,--gc-sections` drops every unreached `at::cpu::`/kernel. Closure collapsed **2565 → ~24 undefined**. |
| `cpuinfo` (needed by `parallel_for`) | builds standalone in one step (below). Not a blocker. |
| SIMD | avoid `add_stub` by driving `TensorIterator` with a scalar `iter.for_each(loop)` (no `Loops.h`). |
| Residual | `TensorIterator` unconditionally references cold-path ops → stub them. |

## What does NOT work (tried, ruled out)

- **`--op-registration-whitelist`** — only filters dispatcher *registration*
  (which static dispatch doesn't use). It does **not** trim the generated
  `at::cpu::` wrappers: RegisterCPU still defines all 2669 of them → 648 native
  kernels.
- **`--op-selection-yaml-path`** (selective/mobile build) — trims `m.impl`
  registrations to the selected ops, but **also keeps all 2669 `at::cpu::`
  defs**. Doesn't bound the static-dispatch closure either.
- Relying on archive on-demand linking alone — the generated `Operators_*.cpp`
  / `RegisterCPU_*.cpp` bundle hundreds of ops per TU, so pulling one pulls the
  whole object. **`--gc-sections` (function-level)** is what actually bounds it.

## The recipe

### 1. Codegen (static dispatch, no registration)
```sh
python3 -m torchgen.gen --source-path aten/src/ATen --install_dir GEN \
  --generate headers --static-dispatch-backend CPU --skip-dispatcher-op-registration
python3 -m torchgen.gen --source-path aten/src/ATen --install_dir GEN \
  --generate sources --static-dispatch-backend CPU --skip-dispatcher-op-registration
```
Yields `Operators_*.cpp` whose `::call` directly invoke `at::cpu::*`, and
`RegisterCPU_*.cpp` with the `at::cpu::*` definitions and empty registration
blocks.

### 2. Compile flags
```
-std=c++17 -O1 -ffunction-sections -fdata-sections
-DFMT_HEADER_ONLY=1 -DCPU_CAPABILITY=DEFAULT -DCPU_CAPABILITY_DEFAULT
```
`CPU_CAPABILITY=DEFAULT` = the scalar (non-AVX) baseline for any kernel TU.
Also needs the generated `ATen/Config.h` (all features off, `AT_PARALLEL_NATIVE=1`)
alongside `c10/macros/cmake_macros.h`.

### 3. Link
```
g++ -Wl,--gc-sections  test.o  libgen.a  libaten_core.a  libcpuinfo.a  -lstdc++ -lpthread
```

### 4. cpuinfo
```sh
cmake -S third_party/cpuinfo -B build_cpuinfo \
  -DCPUINFO_BUILD_UNIT_TESTS=OFF -DCPUINFO_BUILD_MOCK_TESTS=OFF \
  -DCPUINFO_BUILD_BENCHMARKS=OFF -DCPUINFO_LIBRARY_TYPE=static -DCMAKE_BUILD_TYPE=Release
cmake --build build_cpuinfo -j   # -> libcpuinfo.a
```

## The residual: stubbing TensorIterator's cold paths

`TensorIterator.cpp` references ~33 ops in its error/type-check helpers, many
never executed for a dense CPU float op: `is_nonzero`, `item`, `to_dense`,
`dequantize`, `coalesce`, `values`/`indices`/`*_indices`, `copy_`, `sum`, and
(directly) `eq`/`lt`/`ne`. Pulling them drags in **sparse**, **quantized**
(`QTensorImpl`, quantizers), **reduction** (`sum`), **comparison**, and a few
`DispatchStub` SIMD kernels (`fill_stub`, `eq/lt/sum`).

Strategy (chosen): provide throwing stubs so `--gc-sections` removes the
machinery behind them. Two subtleties learned the hard way:
- Stub at the right **level**. Some of these are reached via `TensorIterator`'s
  *direct* `eq`/`lt` references → `at::cpu::eq` → `structured_eq_*::impl` /
  `at::meta::structured_eq::meta`. Stubbing the high-level leaf isn't enough;
  you must also stub the `structured_*::impl`/`meta` (methods of generated
  classes) or avoid generating those `::call`.
- **Monolithic native TUs bundle hot + cold** symbols (e.g. `TensorShape.cpp`
  has both `select_symint` *and* `col_indices_default`). Compiling it for the
  hot symbol keeps the cold one reachable, which then **collides** with a stub
  of the same name. So either stub at the `::call`/`at::cpu::` layer or don't
  compile the bundling TU.

Status: 648 → ~24 residual; reaching 0 is a finite but fiddly stubbing pass.

## Build closure (so far) beyond the 137-file core

generated: `Operators_*`, `RegisterCPU_*`, `RegisterComposite{ExplicitAutograd,
ExplicitAutogradNonFunctional,ImplicitAutograd}_0`; hand-written native:
`TensorIterator`, `EmptyTensor`, `Fill`, `Resize`, `TensorFactories`,
`TensorProperties`, `Scalar`, `TypeProperties`, `MemoryOverlap`, `ExpandUtils`,
`NamedTensorUtils`, `TensorUtils`, `Context`, `Parallel*`, `record_function`,
`SequenceNumber`, `DeviceAccelerator`, `detail/*HooksInterface`,
`cpu/FlushDenormal`; plus `libcpuinfo.a` and a `stubs.cpp` for the cold paths.

## Net

Real `TensorIterator`-based CPU ops are reachable **dispatch-free and
SIMD-free**, at the cost of `TensorIterator`'s closure (~30 native files +
cpuinfo) and a cold-path stub TU. The cost driver is `TensorIterator`, never
dispatch or the add kernel. When the generator needs a real op, this is the
recipe to grow `lib/aten` (the C++ shim/build) by.
