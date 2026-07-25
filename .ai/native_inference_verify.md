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
The initial CI list is `resnet18`; add models only after their importer coverage
and native evaluation time are suitable for CI.

For profiling a long native run, `native_graph eval --verbose` prints each
completed node's canonical native-IR rendering (operation parameters, operands,
outputs, and PT2 provenance) plus CPU compute time to stderr. Evaluation hooks
have paired `on_start`/`on_end` callbacks, with the start callback's
caller-owned state passed unchanged to the matching end callback; timing is
therefore a CLI concern rather than evaluator API data.
