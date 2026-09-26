# Project overview

## Goal

Read PyTorch `.pt2` export archives and execute supported graphs through either
C++ ATen bindings or pure OCaml Native evaluators. Archive reading itself does
not require libtorch. A typed OCaml decoder for `model.json` (the serialized
`ExportedProgram` graph) is generated
automatically from the checked-out PyTorch `schema.yaml`, keeping the decoder
aligned with that schema version.

The parsed graph feeds two independent consumers, described below: a direct
ATen-op interpreter, and a graph-IR pipeline with its own transform and
verification layer. The pure OCaml graph pipeline also compiles to JavaScript;
the C++ ATen interpreter does not. Two JS entry points exist and reach
different depths: the browser Model Explorer bundle (`js/webapp`) decodes,
lowers and verifies but runs no inference, while the node runner (`js/run`,
built as `(modes js)` in `js/jsoo`) executes a real model through
`Native_interp` — that is the only compiled-to-JavaScript execution path.

## Pipeline

```
modules/pytorch/torch/_export/serde/schema.yaml   ← upstream schema (git submodule)
                │
                ▼
         bin/schema_gen.ml                         ← reads schema.yaml, emits OCaml
                │
                ▼
         lib/generated/pytorch_types.ml            ← generated decoder library
                │
                ▼
             lib/pt2                               ← archive reader
                │
     ┌──────────┴────────────────────┐
     ▼                               ▼
 lib/pt2_aten                  lib/native_interp    ← PT2-to-Native importer
     │                               │
     ▼                               ▼
 lib/interp                    lib/native          ← six-axis graph IR
 (C++ ATen execution)                 │
                                     ▼
                               lib/native/transform
                               (passes + symbolic verification)
                                     │
                                     ▼
                               lib/native4d        ← checked four-axis lowering
```

Native graphs also support concrete and symbolic evaluation. The
`lib/native_aten_bridge` library converts tensors and dispatches Native
implementations for comparison with ATen. `lib/model_explorer_export` exposes
pipeline snapshots and computation details to `js/webapp`; visualization does
not require running inference first.

### Libraries

| Path | Role |
|---|---|
| `lib/pytorch_schema/` | Schema meta-parser + code generator |
| `lib/schema_runtime/` | Runtime support for generated decoders (`String_map`) |
| `lib/generated/` | Generated decoder library (build-tree artifact) |
| `lib/pt2/` | Libtorch-free `.pt2` reader: ZIP, pickle, model.json/weights-config |
| `lib/pt2_aten/` | Bridges `pt2`'s raw strided tensors to runnable `Aten_tensor.t` |
| `lib/aten/` | Generated ATen op bindings + Tensor API, over a locally built C++ subset including selected kernels |
| `lib/interp/` | Direct interpreter: walks the raw `ExportedProgram` graph, dispatches to ATen |
| `lib/expr/` | Symbolic expression language: typed indices, values, reductions, intrinsics (public façade over `lib/expr_internal/`) |
| `lib/native/` | The Native graph IR (six-axis, channels-last): representation, ops, transform passes, symbolic verification |
| `lib/native4d/` | Native4D: a checked four-axis dialect that Native legalizes into |
| `lib/native_interp/` | Pure OCaml PT2-to-Native graph importer; captured payloads are resolved separately |
| `lib/native_graph/` | PT2 provenance sidecar (names, paths, metadata, payload targets) kept out of `lib/native` so graph execution stays PT2-free |
| `lib/native_aten_bridge/` | Converts ATen/Native tensors and dispatches Native operations for parity checks |
| `lib/model_explorer_export/` | Exports graph snapshots and computation details for the browser UI |
| `lib/walk_core/` | Backend-neutral verification foundation: PCG32, bit-exact Float32, the random-walk run loop |
| `lib/core/` | Shared printer glue (`Core.Pretty`) |
| `vendored/err_trace` | `Err`, the repo-wide result/error framework; `lib/err_host` reads its configuration at an executable's entry point |
| `js/webapp/`, `js/jsoo/`, `js/melange/` | The browser-compiled frontend and its two JS backends |

See `.ai`'s other docs for each area in depth — `native_graph_design.md`,
`native_transform_design.md`, `native4d_design.md`, `native_compute_design.md`,
`interpreter_design.md`, `model_explorer_design.md`, `js_backends_design.md`,
`domain_int_design.md` (one `int`, one meaning) —
and grep `.ai/` by topic for anything not listed here.

## ATen C++ build

[`lib/aten/`](../lib/aten/) builds a subset of PyTorch's C++ (the dispatcher,
IValue, type system, and required CPU kernels) into
`libaten_core.a` via dune and binds it with ctypes (the C++ sources sit next to
the OCaml bindings). This supports the direct interpreter and ATen parity
tests. See [the ATen build design](aten_core_build.md).
