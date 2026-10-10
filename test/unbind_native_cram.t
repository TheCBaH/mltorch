Native lowering of a serialized `torch.ops.aten.unbind.int` node.

UNGATED and hand-built, for the same reason as me_visualize_unsupported_cram: what
this has to exercise is a node shape no released model in `data/pt2/` contains in a
form small enough to read, and — for the malformed variants — one no exporter emits
at all.

`unbind.int` returns `Tensor[]`, which the PyTorch serializer represents as ONE node
output of kind `as_tensors` holding every result name in order. That is a different
shape from a fixed tuple, whose elements are separate entries in `Node.outputs`, so
`Native_interp.output_names` has to flatten it rather than reject it — which is what
it used to do, uniformly, for every fixture here.

  $ ./cram_probe.exe fixture unbind

Each fixture now reaches its own outcome. The four well-formed ones lower, including
the rank-five ViT case — Native has no four-axis restriction, so `dim=0` on a rank-five
tensor is an ordinary `T`-axis unbind here; it is the Native4D conversion that has to
refuse it, and this file is not where that shows.

`count-mismatch` is the one shape only this op can have: the node's single `as_tensors`
output declares three names while `x`'s extent at `dim=0` is two. Both numbers are model
data, so it is a malformed graph rather than a defect, and it is checked before the
builder allocates anything.

`over-limit` is refused by Model Explorer's OWN per-node ceiling
(`max_outputs_metadata_per_node` = 1024), not by the lowering bound, which sits higher
at 4096. That is the honest result for this driver and worth recording: the projection
is bounded before the lowerer gets the chance. The lowering bound protects every caller
that has no Model Explorer limits — the `.pt2` interpreter and `native_graph`'s
print/eval/transform/to4d — and is driven directly in
`test/native_interp/output_limit_test.ml`.

  $ for m in dim-absent dim-pos dim-neg vit-rank5 count-mismatch over-limit graph-list; do
  >   printf '%-16s ' "$m"
  >   if ../bin/native_graph.exe visualize --model u-$m.json >/dev/null 2>err.txt; then
  >     echo lowered
  >   else
  >     head -1 err.txt
  >   fi
  > done
  dim-absent       lowered
  dim-pos          lowered
  dim-neg          lowered
  vit-rank5        lowered
  count-mismatch   native_graph: malformed PT2 graph: torch.ops.aten.unbind.int declares 3 outputs but produces 2
  over-limit       native_graph: session outputsMetadataPerNode = 4096 is over the ceiling
  graph-list       lowered

The whole projection, over the two fixtures that differ only in what the four-axis
dialect can represent. Nothing in Model Explorer needed a special case for a
multi-output node: the source projection reads `Argument.Tensors` through
`Me_source.tensor_names`, and every later stage works from the registries and the
generic output lists.

`u-dim-neg` unbinds C on a rank-three input, so every slice stays four-axis and the
graph converts the whole way down. `u-vit-rank5` is the motivating ViT node: Native
represents it happily — `dim=0` on a rank-five tensor is an ordinary `T`-axis
unbind — and only the Native4D row goes unavailable.

  $ caps() {
  >   ../bin/native_graph.exe visualize --model "$1" --output "$2" 2>/dev/null
  >   ./cram_probe.exe brief "$2"
  > }
  $ caps u-dim-neg.json neg.json
  stage:source             available graph
  stage:initial_native     available graph
  stage:native4d           available graph
  $ caps u-vit-rank5.json vit.json
  stage:source             available graph
  stage:initial_native     available graph
  stage:native4d           unavailable outside_dialect_domain
    diagnostic: outside_dialect_domain | node n0: axis T is outside the N/H/W/C dialect

The source node carries ONE `Tensor[]` output, and it renders as its ordered SSA
elements rather than as a single edge or as none. Filtering for `Argument.Tensor`
alone would print an empty list here — a node with no outputs at all, which reads
as a graph that simply does not use its result.

  $ ./cram_probe.exe unbind-source neg.json
  label: torch.ops.aten.unbind.int
    slot 0 ssa ['u0']
    slot 1 ssa ['u1']
    slot 2 ssa ['u2']
    slot 3 ssa ['u3']

One Native node with every output edge, and the ids stay distinct — a projection
that collapsed a multi-output node's slots would show fewer, or repeat one.

  $ ./cram_probe.exe unbind-native neg.json
  label: Unbind slots: ['0', '1', '2', '3'] unique: True
