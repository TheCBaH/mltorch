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
and both JS runtimes end up on `Err.Stack.pp`'s "stack unavailable" branch — a count
would differ by design and the diff could never pass.

**A differential harness is blind to any fault that reproduces on both sides.** Both
executables run the same source, so a broken encoder, a failed evaluation or a
Direct-vs-Symbolic mismatch prints *identical* text either side and diffs clean. The
diff only ever answers "do the backends agree", never "is the answer right". Every
correctness verdict in `js/probe` therefore has to reach the **exit status**: fixtures
abort through `Err.or_raise`/`failwith` per the section below, boolean verdicts are
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
with `Err.or_raise`, which raises `Err.Exn.E` carrying the whole error — payload,
detection origin and semantic trace — and registers a `Printexc` printer, so a runner
that only calls `Printexc.to_string` still reports all of it:

```ocaml
let build name m =
  Graph_builder.build ~name ~outputs:(fun o -> [ o ]) m
  |> Err.or_raise ~pp_error:(fun ppf e ->
         Fmt.pf ppf "fixture %s: %a" name Graph_builder.pp_error e)
```

A test that catches this must match `Err.Exn.E`, not `Failure`. Assert its *shape* and
that the payload survives — never the rendered text, which embeds a backtrace and
shifts on every build (`test/pretty_test.ml` is the model).

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

`Err.or_raise` is scoped to `Err.t`. A plain `(_, string) result` from
Zipc/Jsont has no `Err.Error.t` to unwrap, so those keep an explicit
`match … | Error e -> failwith e`.

## A test that changes the trace policy must restore it

`Err.Config` is process-wide and expect tests share a process, so a test that sets a
policy and leaves it set silently changes every later test in the same executable.
Wrap it in `Fun.protect`; `test/native/core_test.ml`'s `with_config` is the pattern,
and the test immediately after it asserts that the restore actually happened. The same
rule applies to `Err.Monitor.install`, whose handle must be removed in the same way.

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

### Promoting from an interactive shell: always `NO_COLOR=1`

`cmdliner` and `jsont` both style error text with raw ANSI escapes
(`\x1b[1m...\x1b[0m`) whenever they detect a color-capable terminal (`$TERM`
set to anything but `dumb`, `$NO_COLOR` unset) -- see `Cmdliner_base.styler'`
and `Jsont_base.Fmt.styler'`, independently-vendored copies of the same
heuristic. `dune promote`/`--auto-promote` writes whatever the test printed
verbatim into the golden. Run it from an interactive terminal and the escape
bytes go straight into the committed `.t`/`.ml` file -- invisible in any
viewer or in dune's own diff rendering (both interpret or strip them), but
real bytes that only mismatch on a run where `$TERM` looks different, e.g.
headless CI. Two goldens broke exactly this way (`me_diagnostic_test.ml`,
`me_wire_limits_test.ml`): promoted from a colored shell, silently correct
against every subsequent local run (same `$TERM`, same color decision), wrong
against CI's uncolored one. `make runtest`, `make pt2.runtest` and `make
jsoo.inline-runtest` set `NO_COLOR=1` for this reason; do the same for any ad
hoc `dune promote`. The JS target needs it as much as promotion does: it is a
second execution of those same committed byte-for-byte goldens, under Node's
headless environment.

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

**None of the five downloadable models serializes the FUNCTIONAL spelling of
any Group 5 activation (op5.md).** `silu_.default` is real coverage (the table
above), but `silu.default` and `hardsigmoid.default` appear nowhere, and
`hardswish.default` appears nowhere either — mobilenet_v3_small exports it
pre-decomposed as `mul(x, div_scalar(clamp(add_scalar(x,3),0,6), 6))`
(`test/native_graph_cram.t:1003`). So `test/native/compute_test.ml`'s
deterministic fixture, `test/native_bridge_test.ml`'s dispatch/verify tests,
`test/native_interp/activation_test.ml` and `test/me_group5_cram.t` are the
coverage for the three functional targets, not a supplement to a zoo model —
the same shape as Group 2 (`op2.md`) and Group 3 (`op3.md`) before them. The
in-place spellings (`silu_.default` etc.) get the same bridge/importer arm as
their functional twin and are additionally checked against the real
efficientnet counts above via `make pt2.runtest`.

**Group 6 (`op6.md`) is worse: neither target occurs in ANY graph this checkout
can reach.** `pad.default` and `slice.Tensor` appear in none of the 23 core-ATen
graphs under `modules/devcontainer.pytorch-image-models/models/*/models/model.json` and in none
of the five downloadable release models. The 30 targets that do occur run
`add.Tensor … where.self`; the nearest structural ops are `select.int` (220
nodes across four ViT models), `expand.default`, `squeeze.dims` and
`unsqueeze.default`. There is no `slice`, no `pad`, and no `narrow`.

So for Group 6 the hand-built fixtures are not the coverage *for now* — they are
the coverage, full stop, and **no cram will regress if either op breaks**.
`make pt2.runtest`, `test/pt2_node_bridge_cram.t` and every `interp_*_cram.t`
reach neither target. What stands in for real-model evidence:

| layer | evidence |
|---|---|
| numeric, independent oracle | `test/native_bridge_test.ml` — 14 pad and 20 slice configurations against real ATen |
| numeric, fuzzed | the `Recipe_pad` / `Recipe_slice` ATen walks (`test/native_walk_test.ml`) |
| staging | `Pad_nwalk` / `Slice_nwalk` (Direct vs Symbolic — **not** an oracle: both sides run the same `Compute` functor) |
| serialized lowering | `test/native_interp/{pad,slice}_test.ml` |
| dialect | `test/native4d/{compute,lower,verify,mutation}_test.ml` |
| session | `test/me_group6_cram.t` |

The configuration counts `op6.md` quotes (`constant` 74 / `reflect` 4 in
`resnetblur18`; `dim ∈ {1,2,3}`) come from a **clone of
`TheCBaH/devcontainer.pytorch-image-models`** — 100 functional-ATen graphs that
are not a submodule here and that no `make` target fetches. Treat them as
provisional; re-derive them if that corpus is ever added.

**Group 7 (`op7.md`) splits, and the two halves must not be conflated in a
coverage report.** The group has two targets and they sit at opposite ends of
this section's spectrum:

| target | occurrences | what stands in |
|---|---|---|
| `layer_norm.default` | **zero**, in all 23 core-ATen graphs and all five downloadable models | hand-built fixtures, exactly like Group 6 |
| `native_layer_norm.default` | **148** — `vit_b_16` 25, `vit_b_32` 25, `vit_l_16` 49, `vit_l_32` 49 | the same fixtures, *plus* real weights via `vit_b_32` |

Every corpus node is one configuration, checked argument by argument:
`normalized_shape` of arity 1 (`[768]` / `[1024]`), `eps` spelled `1e-06`, both
affine operands present, three outputs of which the trailing two are dead in
every occurrence, and **no `cudnn_enable` argument survives export** — it exists
only on the functional overload's schema. `metadata.from_node` records
`aten.layer_norm.default` as the pre-decomposition source, so the functional
target exists only *upstream* of export: `ExportedProgram.module()` lowers it
before serialization.

`op7.md`'s own "measured starting scope" (`eps ∈ {1e-5, 1e-6}`,
`cudnn_enable = false`) comes from the same out-of-tree
`TheCBaH/devcontainer.pytorch-image-models` clone as Group 6's counts. Same
caveat: provisional, re-derive if that corpus is ever added. Both epsilon values
are walked regardless.

**And even the covered half moves no gate yet.** Importing
`native_layer_norm.default` takes `vit_b_32` from 559 to 584 of 839 nodes, but
the first `expand.default` is node 3 and the first `native_layer_norm.default`
is node 7 — so `native_graph print` still stops in the same place and `make
pt2.runtest` produces no diff. Fourteen targets remain. The claim to make is
"removes one of fifteen blockers", never "vit_b_32 imports".

What stands in for real-model evidence, for both targets:

| layer | evidence |
|---|---|
| numeric, independent oracle | `test/native_bridge_test.ml` — 11 `layer_norm` and 4 `native_layer_norm` configurations against real ATen, including the corpus-shaped `[1, 50, 768]` |
| numeric, fuzzed | the `Recipe_norm` ATen walks for both targets (`test/native_walk_test.ml`), 20 and 26 steps, all four affine states reached |
| numeric, hand-computed | `test/native/compute_test.ml` — the formula itself, which no Direct-vs-Symbolic or Native-vs-Native4D comparison can check |
| staging | `Layer_norm_nwalk` (Direct vs Symbolic — **not** an oracle: both sides run the same `Compute` functor) |
| serialized lowering | `test/native_interp/layer_norm_test.ml`, both targets |
| dialect | `test/native4d/{compute,lower,verify,mutation,domain}_test.ml` |
| session | `test/me_group7_cram.t` |

**Group 8 (`op8.md`) is a starker version of Group 7's split: the target
occurs in NO reachable graph, and its NAME's presence is what makes that a
measured fact rather than an absence.**

| target | occurrences | what stands in |
|---|---|---|
| `scaled_dot_product_attention.default` | **zero** targets, in all 23 core-ATen graphs and all 13 downloaded archives — but the STRING appears 600-1200 times inside the four `vit_*` graphs' `metadata.from_node` provenance | hand-built fixtures only |

`from_node` records what a node decomposed FROM, so those 600-1200
occurrences are proof, not a near-miss: in `vit_b_32`, 300 of 839 nodes (36%
of the graph) carry `from_node = aten.scaled_dot_product_attention.default`,
and their actual targets are twelve primitives —
`view`/`expand`/`permute`/`mul.Scalar`/`bmm`/`logical_not`/`_softmax`/
`eq.Scalar`/`any.dim`/`full_like`/`where.self`/`clone`.default — none of
which this row touches. `op8.md:17-18`'s premise ("SDPA is the common final
interpreter blocker for the two transformer families") is false in this
checkout: the blockers are the decomposition, a separately-ranked plan.

Unlike Group 7, there is no covered half to report separately — every
fixture below, at every layer, is hand-built:

| layer | evidence |
|---|---|
| numeric, independent oracle | `test/native_bridge_test.ml` (the ATen-live-tensor importer) and `test/native_interp/sdpa_test.ml` (the serialized-metadata importer) |
| numeric, real ATen kernel | five fixtures under `test/data/sdpa/`, run through `aten_spec_verify --eval` (`test/sdpa_spec_cram.t`) — every flash-admissible mask/scale form, since no model serializes one |
| numeric, fuzzed against real ATen | `Recipe_sdpa` (`test/native_walk_test.ml`) — the recipe's types make an off-oracle (non-flash) configuration unrepresentable, checked independently with a one-time scratch probe over the recipe's full axis product (op8-impl.md commit 4) |
| numeric, hand-computed | `test/native/compute_test.ml` — the formula itself, PLUS an 11-mutation battery each observed changing a result then reverted, since neither Direct-vs-Symbolic nor Native-vs-Native4D can see a wrong formula |
| staging | `Sdpa_nwalk` (Direct vs Symbolic — **not** an oracle, same caveat as `Layer_norm_nwalk` above) |
| serialized lowering | `test/native_interp/sdpa_test.ml` |
| dialect | `test/native4d/domain_test.ml` — unconditional rejection, no legalization at any batch extent (unlike Bmm's `batch = 1` case) |
| session | `test/me_group8_cram.t` |

**This row delivers no model-coverage movement whatsoever**, not even Group
7's "removes one of fifteen blockers" — `make pt2.runtest`, `make inference`
and `make jsoo.pt2.runtest` show no diff, because no downloaded model
contains the target. The four `vit_*` models stay blocked on the twelve
primitives above, none of which is in scope here. Do not report this row's
completion as model progress; report it as what it is — a fully specified,
independently verified operation with zero corpus reach today.
