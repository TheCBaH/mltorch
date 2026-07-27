# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Working Principles

1. **Think before coding** — state assumptions explicitly, surface tradeoffs, halt if something is unclear.
2. **Simplicity first** — write minimal code solving only the stated problem; no speculative abstractions.
3. **Surgical changes** — modify only essential code; match existing style.
4. **Goal-driven** — turn vague requests into verifiable criteria before starting.

## Conventions

- **Record types get their own module, named `t`** (e.g. `Node.t`, `Graph.t`),
  matching the existing `Module.t` style (`Tensor.t`, `Vec6.t`). This guarantees
  field-label uniqueness by construction (`Node.outputs` vs `Graph.outputs`).
  **Do not silence warning 30** (duplicate labels) — give the types distinct
  namespaces instead. For mutually recursive records, use `module rec`; if a
  variant only references the others, parametrise it so it can stay outside the
  recursive group (see `lib/native/graph_ir.ml`: `'g gop`).
- **Result/Option handling**: don't hand-roll `match ... Ok/Error`/
  `Some/None` purely to print — compose through `Fmt.result`/`Fmt.option`/
  `Core.Pretty` instead. Don't cross a `Core.result` into an exception
  boundary with an open-coded `match ... | Error e -> failwith (...)` —
  use `Core.or_raise`. Never rebuild an `Error` by hand-unwrapping
  `e.Core.Error.kind` when `Core.map_error` applies — it preserves the
  original detection backtrace; the hand-rolled form silently doesn't. See
  `.ai/fmt_migration_plan.md`.

## Exploration & Planning — start in `.ai/`

The `.ai/` directory is the canonical design record for this repo. **Before exploring
unfamiliar code or planning any non-trivial change, list `.ai/` and read the docs
relevant to the area you're touching** — they capture the intended design, rationale,
and tradeoffs that the source alone doesn't show, and the place to look for an existing
plan before writing a new one.

The set of docs grows and changes over time, so discover them at the start of a task
rather than relying on a fixed list. Filenames are descriptive (e.g. `*_design.md` for
designs, `*_plan.md` for plans, area prefixes like `aten_*`, `native_*`, `schema_*`);
grep `.ai/` by topic when you're unsure which file applies.

When a design or plan changes materially, update the corresponding `.ai/*.md` doc — or
add a new one — in the same change so this record stays the source of truth.

## Commands

```sh
make build      # build
make runtest    # run all tests
make format     # format — mandatory before every commit

# Single cram test or promote (no make target)
dune runtest test/model_cram.t
dune promote test/model_cram.t
```

**Run `make format` before every commit.** Formatting is enforced; unformatted diffs are noise.

### Working in a git worktree

If you work in a `.claude/worktrees/<name>` checkout, **read `.ai/worktree_setup.md`
first** — `make` from the repo root won't see worktree edits, dune needs `--root .`
there, and submodules/base-ref have gotchas. Quick form:

```sh
git submodule update --init --recursive vendored/opickle   # once per worktree
opam exec -- dune build   --root . lib/native
opam exec -- dune runtest --root . test/native
```

### .pt2 / interpreter (gated on real model data)

```sh
make pt2.download-cram   # fetch the 4 models the cram tests need (~150 MB)
make pt2.runtest         # run pt2_load_cram.t + interp_*_cram.t against them
make inference           # timed smoke run over every model in PT2_MODELS_ALL
```

These need real release weights (`data/pt2/`, gitignored) and are not part of
`make runtest`; CI runs them as separate steps after the main build.

## Architecture

```
modules/pytorch/torch/_export/serde/schema.yaml   ← git submodule (excluded from dune scan)
        │  (shell copy rule in data/dune)
        ▼
bin/schema_gen.ml                                  ← reads YAML, emits OCaml source
        ▼
lib/generated/pytorch_types.ml                     ← build-tree artifact (compiled library)
test/schema_pytorch.ml                             ← build-tree artifact (used via #use in cram)
        ▼
test/*_cram.t                                      ← cram tests decode real model.json files
```

### Libraries

| Path | Dune name | Role |
|---|---|---|
| `lib/pytorch_schema/` | `pytorch_schema` | Schema meta-parser + code generator |
| `lib/schema_runtime/` | `schema_runtime` | Runtime: `String_map = Map.Make(String)` |
| `lib/generated/` | `pytorch_types` | Generated decoder library |
| `bin/schema_gen.ml` | `schema_gen` (exe) | CLI: reads YAML, calls generator |
| `lib/pt2/` | `pt2` | Libtorch-free `.pt2` reader: ZIP (via `zipc`), pickle (via vendored `opickle`), model.json/weights-config decoding |
| `lib/pt2_aten/` | `pt2_aten` | Bridges `pt2`'s raw strided tensors to runnable `Aten_tensor.t` (via `of_storage`), kept separate so `pt2`'s own tests need no C++ build |
| `lib/interp/` | `interp` | Walks an `ExportedProgram` graph and dispatches each node to the bound ATen ops |

Key modules in `pytorch_schema`: `Pytorch_schema` (YAML→type map), `Schema_codegen` (type map→OCaml source), `Type_expr`/`Type_expr_lexer`/`Type_expr_parser` (type-string parser), `Scc` (Tarjan SCC for recursive type detection).

`lib/interp`'s per-node dispatch (`interp_dispatch.ml`) is generated by `bin/aten_ops_gen.ml` from `lib/aten_gen/aten_decode_gen.ml`, the dual of the ATen binding generator: it reads the same `native_functions.yaml` entries that produce a binding and emits the matching decode-and-call arm, so dispatch can't drift from the bindings it calls. Same dep-tracking caveat as `lib/aten/dune` — rebuild manually when the yaml changes.

### Tests

- **Inline expect tests** (`test/scc_test.ml`, `codegen_test.ml`, `type_expr_test.ml`) — `ppx_expect`; promote with `dune promote`.
- **Cram tests** — isolated decode scenarios (`parse_cram.t`, `default_cram.t`), full model parsing (`model_cram.t`, `models_cram.t`), weights config (`weights_config_cram.t`).
- **pt2/interp tests** — `pt2_test.ml` is hermetic (in-memory zip + hand-encoded pickle); `pt2_load_cram.t` and `test/interp_*_cram.t` are gated on `PT2_DATA` (set by `make pt2.runtest`) and run against real downloaded model weights.

Cram tests run `ocaml schema_runtime.cma script.ml 2>/dev/null`. Both `schema_runtime.cma` and `.schema_runtime.objs/byte/schema_runtime.cmi` must be copied into the cram sandbox via dune rules. Source-tree files (e.g., `model_test_utils.ml`) go in `(deps ...)` directly — no copy rule needed.

## Key Constraints

- `modules/pytorch/` and `modules/pytorch.models.pt2/` are git submodules excluded from dune's scan. Reach them via shell rules using `%{project_root}/../../`. No automatic dep-tracking — `touch data/dune` to force a rebuild when `schema.yaml` changes.
- `yamlt` is vendored under `vendored/ocaml-yamlt/` (not on opam); `opickle` (pickle decoding) is vendored under `vendored/opickle/` the same way.
- `data/pt2/<model>/` (downloaded release weights/images/results) is gitignored and not part of the default build — fetch via `make pt2.download-cram` or `make pt2.download-all` before running the gated tests above.

## Git Rules
- Use conventional commit messages.
- NEVER include "Claude-Session" links, URLs, metadata in any commit.

