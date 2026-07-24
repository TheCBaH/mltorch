# Multi-output ops, the `Discard` sink, and dead-output conversion

## Why

Some ATen core ops return a *tuple* of tensors, not one. resnet18 has two:

- `_native_batch_norm_legit_no_training` → `(output, save_mean, save_invstd)`
- `max_pool2d_with_indices` → `(values, indices)`

In an inference (forward-only) graph the non-primary results are **dead** — no
downstream node consumes them. But the goal of the pt2→native conversion is to
**preserve the semantics of the ATen core op**, so the conversion represents the
node's full arity rather than silently pruning; a later optimization pass is
responsible for removing what is unused. This doc records the two mechanisms
that make that possible.

## 1. Multi-output evaluation

`Graph_ir.Node.t` has always carried `outputs : Tensor_id.t list` (a list, "to
admit a future multi-output op"). This is now honoured end to end:

- `Graph_shape.output_shape : op -> … -> Vec6.shape list` already returned one
  shape per output — unchanged.
- `Eval_direct` and `Eval_symbolic` no longer hard-fail on a non-singleton
  output list. Each iterates `Node.outputs` (zipped with the shape list in
  Direct), producing one materialised tensor / one stage per output edge. A
  single-output op runs the loop exactly once; a zero-output op (see `Discard`)
  runs it zero times. `Eval_direct` reports a `` `Output_arity_mismatch `` if
  the shape-list and id-list lengths disagree (a builder bug).
- `Eval_op.pixel` takes an `~output` index (0-based, in `Node.outputs` order)
  so an op can build a *different* pixel per output. Single-output ops ignore
  it; `Max_pool2d_with_indices` selects `value_pixel` for out0 and `index_pixel`
  for out1. `Eval_direct`/`Eval_symbolic` pass `~output:i` as they iterate.

## 2. The `Discard` sink

`Discard of { x : tensor_ref }` is a node that **consumes one edge and produces
no output** (`Node.outputs = []`). It marks an edge as intentionally unused so a
future pruning pass can find and remove the sub-DAG feeding only sinks.

Like `Subgraph`, `Discard` is not a registry op — it is handled inline wherever
`op_registry` is folded (`operands` / `map_operands` / `pp` / JSON in
`graph_ir.ml`), returns `[]` from `Graph_shape.output_shape`, and raises from
`Eval_op.pixel` (it is never evaluated as a pixel; the zero-length output loop
skips it). Build one with `Graph_builder.discard : tensor_ref -> unit t`.

Direct eval still materialises the discarded producer's edge (it is a normal
node output), so a discarded value remains inspectable in the returned env — the
sink only records deadness, it does not suppress computation.

## 3. Dead-output conversion rule (bridge)

The exported graph names dead tuple elements `<op>_unused_<n>` and records their
metadata. There are two shapes of deadness, handled differently by the bridge:

- **Size-0 outputs → dropped.** `_native_batch_norm_legit_no_training`'s
  `save_mean`/`save_invstd` are recorded with `sizes: [0]` (a CPU-export
  artifact; these outputs have no device-independent forward semantics — CUDA
  emits `[C]` running-stats instead). The native engine forbids size-0 tensors
  by construction (`Dim.extent` ≥ 1), so the bridge simply **does not emit**
  these outputs. This is faithful: the graph itself declares them empty.
- **Real-shaped dead outputs → materialised + `Discard`.**
  `max_pool2d_with_indices`'s `indices` output is a full-size `[N,C,H,W]` tensor
  that is representable. The bridge materialises it (a genuine argmax-index
  computation) and routes it into a `Discard` node, so the op keeps its full
  two-output arity while the edge is explicitly marked dead.

  The argmax needs an index carried into the value domain, which the abstract
  `Semantics` deliberately separates from indices (`load` was the only bridge).
  So one minimal primitive was added — `value_of_index : delta index -> t`
  (Direct: `float_of_int`; Symbolic: an `Expr.Value_of_index` node). The index
  pixel takes, over each window, the minimum flat index `ih*in_W + iw` among
  positions whose value equals the max (non-max positions excluded with `+inf`;
  min via `-max_reduce(-·)`, since there is no `min_reduce`).

## 4. ATen-vs-native verification of dropped outputs

`Verify.verify_node` (`lib/aten_native_verify`) compares the outputs the bridge
*exposes* (`graph.Graph.outputs`) against the **leading** tensor outputs of the
ATen node — a native graph may legitimately expose FEWER outputs than the op
has, because a dead output is dropped (size-0) or routed to a `Discard` sink.
Only exposing *more* outputs than the op has is an error. So
`max_pool2d_with_indices` verifies its pooled values against ATen's out0, and
its argmax indices — dropped to `Discard`, and int64 where the native engine is
F32 — are simply not compared. The bridge-coverage walk shows `matched` (not an
`output count` mismatch) for it.

See `native_aten_bridge_layout.md` for the per-op relayout tables and
`native_add_op.md` for the full site list when adding an op.
