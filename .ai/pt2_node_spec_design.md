# Per-node op-test JSON fixtures from a real `.pt2` model — Design

## Purpose

[[aten_spec_design]] gives a JSON format for one ATen operation invocation, hand-written
per test case. This document covers the piece that *derives* such specs from a real
exported model instead: walk every node in a `.pt2` model's graph, keep its real op
target and hyperparameters, and emit one `Aten_spec.Op_spec.t` JSON file per node —
tensor *contents* are always synthesized (never the trained weight/buffer values
themselves), but the distribution used to synthesize them is fit to that tensor's own
real data whenever real data exists. The resulting fixtures need no downloaded model
weights to *evaluate* (only to *generate*), and are re-run through the real ATen kernel
by a cram test whose promoted output is the golden reference — there is no PyTorch
ground truth involved anywhere in this pipeline.

## Why per-node, not whole-graph

Each JSON fixture tests exactly one op invocation, standalone — not by replaying the
full network. This means a tensor argument that is a graph-level parameter/buffer (a
real weight) can be sampled from that weight's own statistics, but a tensor argument
that is an *intermediate activation* (another node's output in the real graph) has no
recorded real data at all, since producing it would require running the whole model up
to that point. Those fall back to a fixed `Normal(0,1)`.

## Distribution: data-driven, not a fixed guess

For every tensor argument, the generator (`lib/pt2_spec_gen/pt2_spec_gen.ml`) first asks
whether the argument's SSA name is a graph-level `Parameter`/`Buffer` input (via
`GraphSignature.input_specs` — the same SSA-name → weight-config-name map
`Interp.run` builds to seed its own environment, `lib/interp/interp.ml:50-62`,
mirrored locally so this generator stays pure/no-ATen).

- **Real parameter/buffer**: load the actual tensor (`Pt2_archive.load_weight`) and
  compute its sample statistics — min, max, mean, variance — over every element. Then:
  - if the real data is **always positive** (e.g. batch_norm's `running_var`, which must
    stay positive since batch_norm computes `1/sqrt(running_var+eps)`): `Uniform{low=min; high=max}`,
    reproducing that guarantee exactly instead of guessing a safe constant;
  - otherwise: `Normal{mean; variance}` fit to the real values (a `1e-6` variance floor
    guards a degenerate all-equal tensor, e.g. a zero-initialized bias).
- **Everything else** (true intermediate activations, and the graph's own user-input
  placeholder): `Normal{mean=0; variance=1}` — a fixed, conservative fallback, since no
  real data exists to fit.

Only tensor *contents* are ever synthesized this way. Every non-tensor argument
(stride/padding/dilation/groups/dims/eps/alpha/momentum/...) is copied **verbatim**
from the real node, so the op is exercised with real-world hyperparameters — only its
"weights" and "activations" are resampled.

A worked example (resnet18's first conv + its batch_norm; `test/data/resnet18/000_*.json`,
`001_*.json`): `conv1.weight`'s stats (`mean≈2.9e-5, variance≈0.017`) come straight from
the checkpoint; the same conv's `input` (the graph's own user-input image) has no
checkpoint data, so it gets `Normal(0,1)`; the fc layer's real weight
(`test/data/resnet18/068_permute_default.json`, since PyTorch's export decomposes a
`Linear` into `permute(weight) ; addmm(bias, input, permuted_weight)`) is stats-fit on the
`permute` node, while `addmm`'s own `mat2` argument — the permute's *output*, not the raw
parameter — correctly falls back to the activation default. This is expected, not a gap:
it is the direct consequence of testing each node standalone.

## Argument decoding

Every param an op takes comes from `Aten_op_config.find target` (the same generated,
runtime schema the ATen bindings and dispatch use — see [[aten_op_config_design]]), in
declared order. `ScalarType?`/`Layout?`/`MemoryFormat?`/`Device?` params are dropped
entirely, exactly like the generated per-op specs (`Aten_op_config.param_type` excludes
them from `Arg_value`). A genuinely-absent `_opt` argument is **also dropped** rather
than encoded as an explicit JSON `null`: the generated per-op codecs decode `_opt` fields
with `Jsont.Object.opt_mem`, whose semantics are "absent key → `None`" — a present key
with a `null` value is *not* equivalent, since `opt_mem` still hands it to the value
codec (e.g. `Tensor_spec.jsont`), which then fails to decode a tensor from `null`. So
`{"bias": null}` is wrong; simply omitting `"bias"` is what round-trips correctly.

## Errors are threaded, not exceptions

`Pt2_dtype`/`Pt2_tensor`/`Pt2_archive` report failures as `Err.t` (see
`lib/core/core.mli`). `lib/pt2_spec_gen/pt2_spec_gen.ml` follows the same convention
throughout — its own `error` row extends theirs, and every fallible function threads
`Err.t` via `Err.Syntax`'s `let*`, rather than converting to exceptions internally.
Only `write_dir`'s outermost per-node loop pattern-matches `Ok`/`Error` directly, and
deliberately does *not* thread a single result through the whole node list: unlike
`Err.List.map`/`fold_left` (which short-circuit on the first `Error`), generation must
keep going past one bad node, writing every node that succeeds and warning (to stderr)
about the ones that don't — coverage gaps stay visible instead of aborting the whole run
or getting silently dropped.

## File layout

One file per node, written to `test/data/<model>/`, committed to git (fixtures are
derived once from a real `.pt2`, not regenerated on every test run):

```
test/data/resnet18/000_convolution_default.json
test/data/resnet18/001__native_batch_norm_legit_no_training_default.json
...
```

`NNN_<target-with-dots-as-underscores>.json`, `NNN` the node's index in `graph.nodes`
(`torch.ops.aten.` stripped). A node whose target isn't in `Aten_op_config` (not yet
bound), or whose only failure mode fires (symbolic shape, non-contiguous weight, ...),
is skipped with a one-line stderr warning rather than aborting the run — every other
node still gets its file.

## How to (re)generate a model's fixtures

```sh
opam exec -- dune exec bin/pt2_spec_gen.exe -- <path/to/model.pt2> test/data/<model>
```

Pure OCaml — no ATen/libtorch build needed, only the `.pt2` file (already present under
`data/pt2-functional/<model>/` once downloaded via `make pt2.download-cram`/`make pt2.download-all`).

## Evaluating the fixtures: `--eval`

`bin/aten_spec_verify.exe --eval <file>` (a third mode alongside the existing default
verify / `--print`, in `lib/aten_spec_run/aten_spec_run.ml`'s `eval_report`) runs the spec
through the **ATen path only** — no native-engine comparison at all, unlike `run`/
`eval_print` — and prints the op's pretty-printed call, each output's shape, and a status
line:

```
[node] torch.ops.aten.convolution.default(input=f32[1,3,224,224], weight=f32[64,3,7,7], bias=none, stride=[2,2], padding=[3,3], dilation=[1,1], transposed=false, output_padding=[0,0], groups=1)
  -> out0: [1,64,112,112]
  status: ok
```

`test/pt2_node_spec_cram.t` runs this over every file in `test/data/resnet18/` (via
`for f in data/resnet18/*.json; do ../bin/aten_spec_verify.exe --eval "$f"; done`) and
commits the output via `dune promote`. That committed output **is** the reference — there
is no PyTorch ground truth anywhere in this pipeline, so a failure means the op's real
behavior (or the ATen build) changed since the fixtures were generated, not that the
fixtures are wrong. Unlike `pt2_load_cram.t`/`interp_*_cram.t`, this cram test is **not**
gated on `PT2_DATA`: the fixtures are committed JSON, so only the ATen C++ build is
needed to run it (same category as `aten_spec_verify_cram`).

## Files

| Path | Role |
|------|------|
| `lib/pt2_spec_gen/pt2_spec_gen.ml` | the generator: node → `Op_spec.t`, data-driven distributions, `write_dir` |
| `bin/pt2_spec_gen.ml` | CLI (`<model.pt2> <out_dir>`), no ATen/C++ build needed |
| `lib/aten_spec_run/aten_spec_run.ml` | `eval_report` (ATen-only pretty-print + shape + status) |
| `bin/aten_spec_verify.ml` | `--eval` flag wiring |
| `test/data/<model>/*.json` | committed per-node fixtures |
| `test/pt2_node_spec_cram.t` | the evaluation cram test (promoted golden output) |

## Known limitations / future extension

- **i64 tensor inputs** (e.g. embedding indices) aren't handled yet: `Aten_spec_run`'s
  `fill_i64` only supports `Values`/`Sequence` payloads, not `Random`. A future model with
  such ops (e.g. a transformer's embedding lookup) needs those tensor args synthesized as
  `Sequence`/`Values` instead of `Random`, in `tensor_spec_of`.
- **Numeric safety for activation defaults**: `Normal(0,1)` is a reasonable default for
  resnet18's op set, but a future op family sensitive to input range (`log`, `div`,
  `rsqrt`, softmax overflow, ...) may need a per-op override on the activation-fallback
  side specifically (the weight side already adapts automatically via real stats).
- Only resnet18 has been generated so far. Repeat "How to (re)generate" for
  efficientnet/mobilenet/vit once new op families are covered.
