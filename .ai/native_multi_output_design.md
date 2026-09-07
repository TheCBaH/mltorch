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

## 1a. Fixed arity vs. arity from the input signature

`Max_pool2d_with_indices` has **two** outputs because the op says so. `Unbind` has
as many outputs as the selected axis has coordinates — its arity is a function of
its *operand signature*, and no other op's is. Three consequences, all of which
the fixed-tuple ops never had to face:

- **The builder cannot take a count.** `Graph_builder.opN` allocates one edge per
  shape returned by `Graph_shape`, with no expected arity to check against. That
  is deliberate: because the count is *derived*, a serialized graph's list of SSA
  output names can be **checked** against it rather than zipped with it, which is
  what turns a malformed export into a diagnostic instead of silent truncation.
  `op1`'s loud singleton check stays for the ops whose arity really is fixed.
- **`Eval_op.pixel`'s `~output` is data, not a selector.** The two max-pool
  outputs run *different* algorithms, so its arm branches on the ordinal. Every
  unbind output runs the same algorithm and differs only in the coordinate it
  reads, so the ordinal goes straight in as a value.
- **The count is untrusted, so it is bounded before it is used.** A declared size
  in a serialized graph decides how many edges shape inference is asked to
  produce, so it is an aggregate over model data. The ceiling is
  `Kernel.Limits.Hard.outputs` (4096) and the rule is **exclusive**, matching
  `Kernel.Limits.create`'s own `v >= hard` test — 4095 is the largest accepted
  count. It is enforced inside `Split.Unbind.output_shapes`, *before* the shape
  list is built: a check in the builder would already be too late, since the
  builder asks `Graph_shape` for the list first and the allocation it means to
  prevent has happened by the time it can measure one. The row is
  `Shape_error.Output_count`, whose `observed` field distinguishes `Exact` (shape
  inference knows the extent) from `At_least` (a bounded list traversal stops at
  the limit and never learns the real length).

Separately, `Tensor_id.check_room` guards the **id space** as both `opN`s advance
their counters. That is an invariant, not a reachable ceiling — with per-node
count bounded at 4095, exhausting the id space needs ~2^31 live edges and memory
goes first — so it raises rather than returning a row. It is written
`count > max_int - next`, never `next + count > max_int`: js_of_ocaml's `int` is
32-bit and a wrapped sum passes the naive test.

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

  Pool values and indices are now separate compact symbolic `Max_pool` nodes,
  both retaining the same input signature and window geometry. The index node
  directly evaluates the clipped window and chooses the smallest flattened
  `ih*in_W + iw` among tied maxima. Unlike max pooling, `avg_pool2d` remains a
  single-output operation: ATen provides no average-pool index tensor.

- **Real-shaped dead outputs with nothing to compute → dropped, and liveness
  refused.** `native_layer_norm`'s `mean`/`rstd` are the third shape and were
  not covered by either rule above. They are *not* size-0 — `vit_b_32` records
  them as real `[1, 50, 1]` f32 tensors, so the batch-norm rationale ("the graph
  itself declares them empty") does not transfer. Nor are they materialised into
  a `Discard`: the native `Layer_norm` op computes the mean and the reciprocal
  standard deviation as *intermediates* of one fused reduction and has no output
  for them, and adding two so a `Discard` could eat them would decompose an op
  the dialect deliberately keeps fused.

  So the bridge exposes one output, which the leading-outputs rule (§4) already
  permits, and the importer keeps only the head in
  `materialized_output_names`. What is new is that the dropped names then have
  to be *proven dead* rather than assumed so: `Native_interp` collects every SSA
  name any node input or graph output reads, and refuses a graph that reads
  either statistic with `` `Live_layer_norm_stats ``. All 148 corpus occurrences
  are dead, which is why the check has only a hand-built fixture — and why that
  fixture was observed failing (without the check the failure is
  `` `Undefined_ssa `` at the *consumer*, a diagnostic naming a node that did
  nothing wrong) before the check was trusted.

  The set is collected lazily and once per graph, not per node: scanning the
  remaining nodes from inside the arm is quadratic in the node count on exactly
  the four ViT models this row exists for.

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

**That leniency is for FIXED tuples only.** A dynamic `Tensor[]` return has no
dead-output story: its arity is the *input's*, not the op's, so there is nothing
a bridge could legitimately decline to expose, and under the leading-outputs rule
a bridge returning only the first slice reports `matched`. Measured, not
supposed: dropping the last output from the `unbind.int` arm printed
`status: matched` until the rule was tightened.

So `verify_node` requires **exact** cardinality when the node's outputs are a
single `Argument.Tensors`, and keeps leading comparison otherwise. The
discriminator is the node's **output structure**, read directly — not its target
string, which would be a second list to keep in sync with the bindings.

See `native_aten_bridge_layout.md` for the per-op relayout tables and
`native_add_op.md` for the full site list when adding an op.

## 5. Shared multi-output computation (project step 19)

§2's own claim — "Direct eval still materialises the discarded producer's
edge... the sink only records deadness, it does not suppress computation" —
described the ORIGINAL LSTM landing, where `output`/`h_n`/`c_n` each rebuilt
an independent copy of the whole layer/direction recurrence. That is no
longer the implementation, though it remains true today's four callers
(Direct, Native4D Direct, `Stage_program.ground`, `Kernel_eval.run`/
`run_plan`) always request all three ordinals regardless of `Discard` — so
the OBSERVABLE per-caller behavior described in §2 is unchanged; what
changed is how much work computing "all three" actually costs.

`Region_group.t` is a validated bundle of shared Region locals plus several
ordered emitters, each reading those locals through a checked canonical-to-
physical axis mapping (`lib/native/region_group.mli`). LSTM's shared
layer/direction scans are built once (`Lstm.Lstm.Computation.group`); the
single-output `Computation.program` compatibility entry point is now a thin
projection of it. `Region_group.Ref.t = Grouped of Region_group.t * int |
Solo of Region_program.t` is the storage type both `Stage_program.Stage.t`
and `Kernel.Value.t` carry instead of a bare `Region_program.t`.

All four tensor-execution paths evaluate the shared locals exactly once per
canonical batch key and read every requested emitter off that one
evaluation (`Region_execution.lower_group`/`materialize_group`):

- Native Direct / Native4D Direct: `Eval_direct`/`Eval_direct4` detect a
  multi-output region-authored node (`List.length outs > 1`) and route it
  through one `Region_computation(4).group` + `materialize_group` call
  instead of one `program` call per output.
- `Stage_program.ground`: both symbolic builders (`Eval_symbolic`,
  `Eval_symbolic4`) build one group per multi-output node and construct
  sibling stages as `Grouped` references into the SAME physical instance;
  `ground` partitions its stage list into maximal contiguous runs sharing one
  physical group (`Region_group.runs`) and materializes each run once.
- Kernel (`Kernel_eval.run`/`run_plan`): `execute`'s eager materialize loop
  partitions `Kernel.Value.t` the same way and shares the evaluation across a
  run's stored members. `Kernel_elab`'s fusion admission rejects any
  `Grouped` value unconditionally (its `pixel_expression` is always `None`,
  tripping the `Regional_computation` check before `has_scan` is ever
  consulted) — a `Grouped` value can never be virtualized on either side of a
  `Kernel.Use.t`, which is what makes sharing at this layer safe without any
  fusion-side change. `Kernel_eval.value_at`'s on-demand scalar path keeps
  using the per-emitter projection (`Region_group.Ref.project`) — an
  explicit, disclosed scalar-path exception, not a gap.

Measured effect (project step 19's own exit criterion): `lstm_scale_test.ml`
went from `keys=3 scans=6 scan_updates=18432` (each of the three outputs
independently costed) to `keys=1 scans=2 scan_updates=6144` for both the
tiny and fast-family fixtures, matching the shared multi-output
implementation plan's own predicted formula (`shared scans = B*L*R`,
`shared scan_updates = B*L*R*T*2*K`) exactly; `stage_group_scale_test.ml`/
`kernel_group_scale_test.ml` show the identical reduction for the Stage and
Kernel paths.

Two disclosed, deliberate simplifications remain, both sound (never admit an
actually-oversized program) but not tight:

- `Region_execution.lower_group`'s preflight projects and preflights each
  emitter independently, so it counts the shared locals' cost once PER
  EMITTER rather than once per group — an overcount, not an undercount.
  `Kernel.create`'s `max_scan_updates_total` aggregation inherits the same
  overcount (it preflights each `Grouped` value via the same projection).
  The plan's own §4.2 formula (count locals once, sum only the emitter
  deltas) is not implemented; nothing currently depends on the tighter bound.
- A group's canonical key is always `W` for LSTM specifically (the only
  Region-authored op with a `group` arm today); `Region_group` itself is
  general over any set of declared canonical axes, but no second op has
  exercised that generality yet.

Status: landed and verified across all four execution paths, project step
19. The working plan this section summarizes lives in the project's
execution ledger, outside this tracked design record.
