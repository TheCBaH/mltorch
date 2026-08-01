# Testing Strategy

## Three Test Layers

### 1. Inline expect tests (`test/scc_test.ml`, `test/codegen_test.ml`, `test/type_expr_test.ml`)

Unit tests using `ppx_expect` / `[%expect ...]`. Run with `dune test` or
`dune runtest test/`. These test:
- `Scc.Make(String).run` on hand-crafted graphs (topological order, self-loops, multi-SCCs)
- `Schema_codegen.generate` on small hand-crafted `Type_def.t String_map.t` inputs
- `Type_expr_parse.of_string` round-trips

Snapshots are auto-promoted by `dune promote`.

### 2. Cram tests — incremental parser tests (`test/parse_cram.t`, `test/default_cram.t`, `test/jsont_explore_cram.t`)

Test specific decode scenarios in isolation before testing on full models. Each test:
- Writes a small `.ml` script via heredoc
- Runs it with `ocaml schema_runtime.cma script.ml 2>/dev/null`
- Compares stdout against recorded expected output

The `parse_cram.t` tests cover one type per test, in dependency order:
1. `SchemaVersion` — plain struct, two ints
2. `TensorArgument` — struct, single string
3. `ArgumentKind` — enum decoded from integer
4. `SymIntArgument` — union, single-key dispatch
5. `RangeConstraint` — optional fields (absent key → None)
6. `Node` — optional field with explicit JSON null → None

`jsont_explore_cram.t` documents the correct `Jsont.option` + `Option.join` pattern
and regression-tests it against four input cases (absent, present, null, both).

`default_cram.t` regression-tests the `Mem_default` fix: decodes a `Graph` JSON with
all optional-defaulted fields present and verifies `nodes=0`.

### 3. Cram tests — full model parsing (`test/model_cram.t`, `test/models_cram.t`)

End-to-end tests that decode real exported model files.

`model_cram.t` — resnet18 (`model.json`):
- Prints `schema=8.14 nodes=69` followed by the op-type histogram (sorted by count)
- Expected output is committed (auto-promoted from `dune promote`)

`models_cram.t` — all 17 models in a single OCaml run:
- Each model prints one line: `name: schema=8.14 nodes=N` + indented histogram
- Expected output is committed

Both tests share helper code from `test/model_test_utils.ml` via `#use`.

### 4. Per-node op-spec fixtures (`test/pt2_node_spec_cram.t`)

Committed JSON fixtures under `test/data/<model>/`, one per graph node, derived from a
real `.pt2` model by `bin/pt2_spec_gen` (real op + hyperparameters, tensor contents
synthesized from a distribution fit to that tensor's own real data when it's a
parameter/buffer — see [[pt2_node_spec_design]]). The cram test evaluates every fixture
through the ATen kernel only (`aten_spec_verify --eval`) and promotes the pretty-printed
call + output shape + status as the golden reference — unlike `model_cram.t`/
`pt2_load_cram.t`, this needs no `PT2_DATA` download to *run* (only to *generate* new
fixtures), just the ATen C++ build.

### PT2_DATA-gated crams: `(universe)` is required, not optional

`pt2_load_cram.t`, `interp_*_cram.t`, `native_graph_cram.t`, `native_transform*_cram.t`
and `native4d_to4d_cram.t` all read files under `$PT2_DATA` (set by `make pt2.runtest`)
from inside their `$`-line shell commands, not from a dune-visible path — the `(cram
(deps ...))` stanza only lists the exe being tested. Dune's incremental build treats a
rule as up to date whenever its *declared* deps are unchanged, so without an explicit
`(universe)` dep it cannot tell that `$PT2_DATA` content changed (or appeared) between
runs: once one of these rules is executed while a model file is absent/incomplete, dune
memoizes the failure and keeps replaying it — even across `--force` and
`DUNE_CACHE=disabled` — until the failure is masked by declaring `(universe)`, or the
build tree is wiped with `dune clean`. Combined with `make pt2.runtest`'s
`--auto-promote`, a replayed stale failure gets written straight into the checked-in
`.t` golden, destroying the real expected output. Every PT2_DATA-gated `(cram ...)`
stanza in `test/dune` therefore lists `(deps (universe) %{exe:...})`, forcing a genuine
re-check of `$PT2_DATA` on every run. `pt2.runtest` deliberately does *not* depend on
`pt2.download-cram` — that target's `unzip -o` always re-extracts even when the zip is
already present, which would churn every file's mtime (and briefly remove it) on every
`pt2.runtest` invocation. Instead `pt2.runtest` checks each `PT2_MODELS_CRAM` model's
`.pt2` file exists up front and fails fast with a `make pt2.download-cram` hint if not,
so `--auto-promote` can never run against missing data in the first place.

## Shared Test Utilities: `model_test_utils.ml`

`test/model_test_utils.ml` provides:

```ocaml
val node_type_counts : Graph_Type.t -> (string * int) list
(* returns (op_target, count) pairs sorted by count descending *)

val pp_model : Format.formatter -> ExportedProgram.t -> unit
(* prints: schema=M.N nodes=K\n  op1: count\n  op2: count\n ... *)
```

The file lives in the source tree and is listed in cram `(deps ...)`. There is no
copy rule for it — dune handles source-tree files directly.

## Fixture failures abort; they do not become absence

A fixture that will not build is a broken test, not a test result. Unwrap it
with `Core.or_raise`, which raises `Failure` carrying the error payload:

```ocaml
let build name m =
  Graph_builder.build ~name ~outputs:(fun o -> [ o ]) m
  |> Core.or_raise (fun ppf e ->
         Fmt.pf ppf "fixture %s: %a" name Graph_builder.pp_error e)
```

Do **not** turn a fixture failure into `None` or a default — the test then
reports a wrong answer instead of a broken setup. Where a helper genuinely must
return an `option` (because every caller pattern-matches it), say in a comment
why the diagnostic is discarded and what still detects the failure;
`test/native/permute_passes_test.ml` and
`lib/native/transform/passes/fold_const.ml` are the models.

Per-directory helpers are deliberately **not** consolidated: they cross
different boundaries with different error types, and one shared "build or die"
would have to be untyped or a functor. Only byte-identical copies were merged
(`test/native4d/{lower,verify}_test.ml` now use `Fixtures.build`).

`Core.or_raise` is scoped to `Core.result`. A plain `(_, string) result` from
Zipc/Jsont has no `Core.Error.t` to unwrap, so those keep an explicit
`match … | Error e -> failwith e`.

## When to Add a New Cram Test

- After fixing a code generator bug: add a cram test that exercises the exact JSON
  pattern that was failing (see `jsont_explore_cram.t`, `default_cram.t`).
- After adding a new schema type support: add a case to `parse_cram.t`.
- After adding a new model: add a rule in `test/dune` and list it in `models_cram.t`.

## Workflow

Run Dune commands serially for a given checkout/build root. Do not keep a
`dune build`, `dune runtest`, `dune promote`, `dune fmt`, or Makefile target that
wraps Dune running while starting another one; they share `_build` state and
promotion output.

```sh
# Run all tests
opam exec -- dune runtest

# Run a specific cram test (shows diff if output changed)
opam exec -- dune runtest test/model_cram.t

# Accept new expected output
opam exec -- dune promote test/model_cram.t

# Verify promotion worked
opam exec -- dune runtest test/model_cram.t
```

## What Each Model Shows (as of schema 8.14)

From `models_cram.t` expected output:

| Model | Nodes | Dominant ops |
|---|---|---|
| resnet18 | 69 | conv2d:20, batch_norm:20, relu_:17, add_:8 |
| efficientnet_b0 | 240 | conv2d:81, silu_:49, batch_norm:49, adaptive_avg_pool2d:17 |
| efficientnet_b1/b2 | 342 | conv2d:115, silu_:69, batch_norm:69 |
| efficientnet_b3 | 387 | conv2d:130, silu_:78, batch_norm:78 |
| efficientnet_b4 | 477 | conv2d:160, silu_:96, batch_norm:96 |
| efficientnet_b5 | 579 | conv2d:194, silu_:116, batch_norm:116 |

ResNet uses `relu_` (in-place ReLU) and `add_` (in-place residual add).
EfficientNet uses `silu_` (Swish) and `sigmoid+mul` for SE blocks.
