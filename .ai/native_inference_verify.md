# Whole-graph native inference verification

`bin/native_graph` remains a pure-OCaml executable. The separate
`bin/aten_graph_ref` executable calls the existing ATen `Interp.run` evaluator
and serializes its final tensor with `Graph_json.encode_tensor`.

`native_graph eval --expect FILE` decodes that tensor and compares the single
native output at an absolute tolerance of `1e-4`. Only the reference producer
links ATen/libtorch.

Every compatible comparison prints `count`, `max_abs`, signed `bias`,
`mean_abs`, `rmse`, signed-error `stddev`, relative L2 error, and cosine
similarity/distance. Passing remains defined by `max_abs <= atol`; the other
values explain the magnitude and direction of a failure without changing that
criterion. Metrics requiring a non-zero norm print `n/a` for zero-norm tensors.

`make native-infer-verify` expands `native-infer-verify.<model>` for the
explicit `PT2_NATIVE_VERIFY_MODELS` list. Each target creates a temporary
reference from one sample image before invoking the pure native executable.
Add models only after their importer coverage and native evaluation time are
suitable for CI.

## Choosing the CI model list

Both `native-infer-verify` and `native-transform-verify` expand the same list,
so an entry costs one native inference plus the four evaluations the two
transform pipelines make. The list is therefore chosen from measurement, not
from node counts.

### Agreement

All four models agree with `Interp.run` on `images/000000000149.pt`, at an
absolute tolerance of `1e-4`:

| model | max_abs | relative_l2 | cosine_similarity |
|---|---|---|---|
| `resnet18` | 9.06e-06 | 1.04e-06 | 1 |
| `mobilenet_v2` | 8.58e-06 | 9.23e-07 | 1 |
| `mobilenet_v3_small` | 1.91e-05 | 1.93e-06 | 1 |
| `mobilenet_v3_large` | 7.93e-06 | 1.70e-06 | 1 |

The MobileNets sit in the same residual band as ResNet-18, i.e. ordinary f32
reassociation, not a systematic difference from the new ops.

### What the static work predicts

Summed over graph convolution nodes as output elements × Kh × Kw ×
(Cin/groups), and over every node output's declared elements:

| model | conv nodes | convolution MACs | vs resnet18 | all node-output pixels |
|---|---:|---:|---:|---:|
| `resnet18` | 20 | 1,813,561,344 | 1.00x | 8,943,592 |
| `mobilenet_v2` | 52 | 299,494,272 | 0.165x | 20,963,240 |
| `mobilenet_v3_small` | 52 | 54,896,576 | 0.030x | 9,263,288 |
| `mobilenet_v3_large` | 62 | 214,080,960 | 0.118x | 22,231,736 |

MACs alone do not predict wall time here: the engine pays per-output scheduling,
pointwise, layout and grouped/depthwise overhead that MAC counts miss, and
MobileNet has 2-3x resnet18's total node-output volume despite far less
convolution arithmetic. So the numbers below are measured, not derived.

### Measured cost

Measured with `/usr/bin/time` on one machine, same sample image, after warming
the build and the downloaded data. **Every model was run three times and is
reported as the median**; run-to-run spread was under 3% throughout. The engine
is single-threaded, so elapsed and CPU time track each other closely — both are
recorded rather than one being asserted from the other.

Median seconds of CPU (user+sys), and of elapsed time, per target and combined:

| model | infer CPU | transform CPU | **CPU total** | infer elapsed | transform elapsed | elapsed total | vs resnet18 | peak RSS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `resnet18` | 86.2 | 337.1 | **423.3** | 86.2 | 337.1 | 423.3 | 1.00x | 241 MB |
| `mobilenet_v2` | 49.9 | 158.2 | **208.0** | 49.0 | 158.2 | 207.2 | 0.49x | 221 MB |
| `mobilenet_v3_small` | 10.0 | 30.9 | **40.9** | 9.4 | 30.9 | 40.3 | **0.10x** | 99 MB |
| `mobilenet_v3_large` | 34.4 | 110.9 | **145.3** | 33.5 | 110.9 | 144.4 | 0.34x | 207 MB |

`mobilenet_v3_small` is cheapest on both metrics and on peak RSS, so the ranking
is unambiguous and does not depend on which is used.

`transform-verify` costs about 3.3x `infer-verify` for every model, which is
what its four evaluations across two pipelines predict.

The ordering follows the MAC estimate rather than node or output-pixel counts:
`mobilenet_v3_small` has 2.6x ResNet-18's node count and about the same
output-pixel volume, yet costs a tenth as much. Convolution dominates by enough
that the per-output overhead the estimate omits does not reorder anything —
though it does compress the ratios (v3_small is 0.10x here against 0.03x of the
MACs).

### The decision

`PT2_NATIVE_VERIFY_MODELS := mobilenet_v3_small`, replacing `resnet18`.

`mobilenet_v3_small` is the cheapest of the four by a wide margin — a tenth of
`resnet18` — and it is already the downloaded cram representative. It also
reaches importer arms `resnet18` cannot: `clamp`, `clone`, `mul`, `div` and the
compile-time scalars the exporter writes into Tensor slots.

The list stays at one model. Whole-model runtime verification is the most
expensive thing in the gated suite, and a second entry buys less than its cost:
the ops are already verified individually, against real ATen, by cheaper means.

`resnet18` is the only model here with `max_pool2d_with_indices`, and so the
only whole-graph exercise of a multi-output op and of the `Discard` sink. That
coverage does not disappear with it — it moves to where the rest of the per-op
coverage already lives:

| what | where |
|---|---|
| pixel semantics, both outputs | `test/native/compute_test.ml` |
| graph build + `Discard` of the dead output | `test/native/graph_test.ml` |
| Symbolic path agrees with Direct | `test/native/graph_symbolic_test.ml` |
| dispatch from a real ATen node | `test/native_bridge_test.ml` |
| ATen-vs-native over a random config space | the generated `Max_pool2d_with_indices_walk` |
| whole-graph import, structurally | `test/native_graph_cram.t` (still ResNet-18) |

What is genuinely lost is only whole-model *numeric* agreement for ResNet-18,
which is why it stays one command away: `make native-infer-verify.resnet18`.
Run it, and the two MobileNets below, when the op set or the evaluator changes.

`mobilenet_v2` and `mobilenet_v3_large` likewise stay manual
(`make native-infer-verify.mobilenet_v2`, and likewise for v3_large). v2 is the
only model with `hardtanh`, so it is the one to run when the pointwise ops
change; its *structural* import and transform are covered in the gated crams,
which is why it is in `PT2_MODELS_CRAM`.

For profiling a long native run, `native_graph eval --verbose` prints each
completed node's canonical native-IR rendering (operation parameters, operands,
outputs, and PT2 provenance) plus CPU compute time to stderr. Evaluation hooks
have paired `on_start`/`on_end` callbacks, with the start callback's
caller-owned state passed unchanged to the matching end callback; timing is
therefore a CLI concern rather than evaluator API data.
