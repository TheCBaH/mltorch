A model this repository cannot lower is a SUCCESSFUL session, not a usage error.

UNGATED, and hand-built rather than downloaded: what it has to exercise is a
graph the lowerer rejects, and no released model is one. The exported program
decoded, which is the whole of what stage:source claims, so the session carries
a source view and a capability vector saying what is missing and why. The
browser shell cannot surface a usage error, and the same code path serves both.

  $ ./cram_probe.exe fixture unsupported

  $ ../bin/native_graph.exe visualize --model operator.json --output session.json

The COMPLETE vector, not the failing key: a test checking only initial_native
would let every downstream row drift, and the whole point of propagating a
prerequisite is what happens to the rows the failure did not touch.

  $ ./cram_probe.exe caps session.json
  stage:source                 available graph
  stage:initial_native         unavailable unsupported_operator
  stage:canonical              unavailable prerequisite_unavailable
  stage:native4d               unavailable prerequisite_unavailable
  stage:stage_program          unavailable prerequisite_unavailable
  stage:kernel                 unavailable prerequisite_unavailable
  stage:fusion                 unavailable prerequisite_unavailable
  feature:flow                 unavailable prerequisite_unavailable
  feature:verification         unavailable prerequisite_unavailable
  feature:pass_audits          unavailable prerequisite_unavailable
  feature:fold                 unavailable prerequisite_unavailable
  feature:expression_detail    available present
  feature:generated_js         unavailable prerequisite_unavailable
  feature:loop_ir              unavailable not_implemented
  feature:codegen              unavailable not_implemented

initial_native carries the reason lowering actually gave, not
prerequisite_unavailable -- "its prerequisite is unavailable" would be circular
for the row that IS the lowering, and the classified reason is the actionable
one. The free text lands in a diagnostic, which is the bounded type that crosses
every boundary of this design.

  $ ./cram_probe.exe diagnostics session.json
  unsupported_operator | unsupported PT2 operator: torch.ops.aten.bogus_operator.default | pt2/root | truncated False

One graph, one view, and NO flow: with no Native state the spine would hold
s/pt2/000 and no transitions, and a one-node flow graph asserts a navigability
that does not exist.

  $ ./cram_probe.exe unsupported-summary session.json
  graphs ['pt2/root']
  views ['v/source'] default v/source
  comparisons [] flow None
  opTargets 1

The other recoverable row. An input spec the lowerer does not handle is
Unsupported_input, and it reaches the same shape through the same classifier --
which is what makes the two rows one code path rather than two.

  $ ../bin/native_graph.exe visualize --model input.json --output input-session.json
  $ ./cram_probe.exe unsupported-input input-session.json
  unavailable unsupported_input
  unsupported_input | unsupported PT2 input: not a tensor

A defect is NOT downgraded to a capability. Reporting an internal invariant
failure as "this model is outside what we support" tells the user to change
their model to work around our bug, so those rows still exit -- here a graph
whose output names nothing, which the lowerer rejects as malformed.

  $ ./cram_probe.exe fixture malformed
  $ ../bin/native_graph.exe visualize --model malformed.json
  native_graph: malformed PT2 graph: SSA tensor "y" is not defined
  [123]

Every OTHER row of [Native_interp.malformed], each with its own witness.

The rows were one `Malformed_graph of string` carrying a ksprintf sentence from
32 sites, so no test could distinguish them and none tried. They are now thirteen
typed cases, and a case no input reaches is indistinguishable from one that
cannot be constructed -- so each is driven here, from the smallest mutation of a
valid program that produces it.

The two most recent are the ones a `Tensor[]` return forced. `zero-dim` is its
own `dim_fault` rather than a negative one: the engine has no empty tensor at
all, and unlike a negative size this does arrive from real models, since ATen's
unbind of a zero-length dim returns an empty list. `output-arity` is the shape
only a list-returning node can have -- one `as_tensors` output whose name count
is model data, checked against a count derived from the operand's extent.

  $ ./cram_probe.exe fixture malformed-rows

  $ for m in missing-arg wrong-kind node-output graph-output no-metadata \
  >          negative zero-dim symbolic rank-seven axis alpha memory-format arity \
  >          conv-rank config-pos config-neg output-arity; do
  >   ../bin/native_graph.exe visualize --model m-$m.json 2>&1 >/dev/null | head -1
  > done
  native_graph: malformed PT2 graph: torch.ops.aten.relu.default: missing argument "self"
  native_graph: malformed PT2 graph: torch.ops.aten.relu.default.self is not a tensor
  native_graph: malformed PT2 graph: torch.ops.aten.relu.default has a non-tensor output
  native_graph: malformed PT2 graph: non-tensor graph output
  native_graph: malformed PT2 graph: no tensor metadata for "x"
  native_graph: malformed PT2 graph: x has negative dimension -1
  native_graph: malformed PT2 graph: x has a zero-length dimension
  native_graph: malformed PT2 graph: x has a symbolic dimension
  native_graph: malformed PT2 graph: x has rank greater than six
  native_graph: malformed PT2 graph: invalid dimension 9 for rank 2
  native_graph: malformed PT2 graph: torch.ops.aten.add.Tensor: alpha=2 is not supported (only 1)
  native_graph: malformed PT2 graph: torch.ops.aten.clone.default: memory_format=channels_last is not supported
  native_graph: malformed PT2 graph: stride must have one or two values, got 3
  native_graph: malformed PT2 graph: w is rank 2, expected 4
  native_graph: malformed PT2 graph: torch.ops.aten.convolution.default: groups must be positive, got 0
  native_graph: malformed PT2 graph: torch.ops.aten.convolution.default: padding must not be negative, got -1
  native_graph: malformed PT2 graph: torch.ops.aten.unbind.int declares 3 outputs but produces 2

[`Output_not_evaluated] has no witness here and is marked as such rather than
left to look covered. It fires when [Eval_direct] returns an environment missing
a graph output -- a graph declaring an output no node produces and that is
neither an input nor a constant.

No model can reach it. [Native_interp.lower]'s body resolves every output
through [env_find], which returns ids [add_env] bound from node outputs, so a
lowered graph's outputs are produced by construction.

It is kept anyway, and that is a different judgement from the one that deleted
[`Dangling_edge] in the same migration. [`Dangling_edge] had no construction
site at all -- nothing in the tree could raise it, and its condition was already
caught by [`Unknown_node]. This one IS raised, at both run sites, and NOTHING
checks it earlier: [Graph_builder.build] does not verify that outputs are
produced, and [Map_verify] runs only under verification. What stands between a
pass that drops a producer and a silent wrong answer is this row.
