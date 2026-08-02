# Testing Strategy

## Test Layers

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

### 5. Native-vs-JavaScript differential (`js/probe`, `make js.runtest`)

The only layer with **no committed golden at all**. `js/probe` is one source set built
natively and again for js_of_ocaml (and, for its pure half, for Melange); the harness
runs both and diffs. The native binary *is* the reference, so the two sides cannot
drift apart the way a promoted golden and its producer can — and a stale expectation
cannot be promoted over a real regression.

That property is also its constraint: a check here must produce the *same* text on
every backend, so it cannot print anything backend-specific. `probe_core` prints
`backtrace-path-valid=true` rather than a slot count, because native captures frames
and both JS runtimes take `Core.Error.pp_backtrace`'s "unavailable" branch — a count
would differ by design and the diff could never pass.

**A differential harness is blind to any fault that reproduces on both sides.** Both
executables run the same source, so a broken encoder, a failed evaluation or a
Direct-vs-Symbolic mismatch prints *identical* text either side and diffs clean. The
diff only ever answers "do the backends agree", never "is the answer right". Every
correctness verdict in `js/probe` therefore has to reach the **exit status**: fixtures
abort through `Core.or_raise`/`failwith` per the section below, boolean verdicts are
asserted rather than printed, and `Walk_core.Walk.run` returns whether every step
verified so the entry point can fail on it. Adding a check that only prints is adding
nothing.

Gated on node, outside `make runtest`, run as two parallel CI jobs. What it covers and
what it has already caught is in [[js_backends_design]].

### 6. Expect tests under js_of_ocaml (`make jsoo.inline-runtest`)

Layer 5's blind spot is precise — it cannot see a fault both backends share — and this
is what covers it. `(inline_tests (modes best js))` on `test/native`, `test/native4d`
and `pt2_test` runs the *existing* expect blocks under node against the *same committed
goldens* the native run uses. So unlike the probe, it does answer "is the answer right",
and it costs no new test code.

The two layers are complementary, not redundant, and the split is worth keeping straight:

| | layer 5 (probe) | layer 6 (expect under node) |
|---|---|---|
| reference | the native binary, same source | a committed golden |
| answers | "do the backends agree" | "is the answer right" |
| blind to | anything reproducing on both | anything not covered by an expect block |
| backend-specific text | forbidden | forbidden *for the same reason* |

**One `[%expect]` cannot hold two backends' answers.** That constraint is layer 5's rule
arriving in a new place, and it bites wherever a value is legitimately backend-dependent:
`Printexc` backtraces (js_of_ocaml captures none) and anything int-width- or
float-formatting-dependent. The fix is the same shape every time — assert against an
*oracle* rather than a spelled-out answer, and print only whether they agree.
`backtrace_path_valid` in `test/native/core_test.ml` and the
`Int32.unsigned_to_int` comparison in `test/pt2_test.ml` are the worked examples.

**Two dune traps**, both of which fail silently rather than loudly:

- **Never put `enabled_if` on the `inline_tests` stanza to gate the js mode.** It gates
  the whole stanza, not one mode, so on a machine without node it disables `best` too and
  the native suites quietly become empty — a green run that tested nothing.
- **The root `dune` must set `(js_of_ocaml (runtest_alias runtest-js))`.** Without it the
  js runs attach to `runtest`, and every `make runtest` starts linking js_of_ocaml.

When adding a `(modes best js)` stanza, verify the native side still runs — break one
golden on purpose and confirm `dune runtest` catches it. An empty stanza exits 0.

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
