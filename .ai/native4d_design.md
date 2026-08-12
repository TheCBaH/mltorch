# Native4D dialect — goal, feasibility review, and high-level plan

## Status

Design and implementation plan.

This document describes a second native graph dialect whose tensors occupy only
the `N`, `H`, `W`, and `C` axes of the existing six-axis tensor frame. It reviews
the plausibility of sharing the current transformation, verification, and
compute infrastructure, defines the boundary between the dialects, and scopes
the work required to implement it.

Related design records:

- `.ai/native_tensor_design.md`
- `.ai/native_compute_design.md`
- `.ai/native_graph_design.md`
- `.ai/native_transform_design.md`
- `.ai/native_transform_verify.md`
- `.ai/native_transform_versioning.md`
- `.ai/native_multi_output_design.md`

## 1. Goal

The native framework is intended to hold several representations of ML
computations and to provide a grounded way to:

1. transform within one representation;
2. convert between representations;
3. retain an explicit correspondence and provenance map for every change; and
4. validate the claimed relation between the source and destination graphs.

Native4D is the first additional dialect. It is a deliberately reduced
representation:

- every tensor uses the existing dense `Tensor.packed` storage;
- tensor shapes use the existing `Vec6` physical frame, but `T = 1` and `D = 1`;
- only the axes `N`, `H`, `W`, and `C` carry non-unit extents;
- reduction is represented only in keep-dimensions form;
- convolution is split into `Conv2D`, `DepthwiseConv2D`, and
  `TransposedConv2D`;
- there is no general grouped-convolution operation;
- `Linear`/fully connected operations are legalized to `Conv2D`;
- the supported single-batch form of `Bmm` is legalized to `Conv2D`;
- operations outside the reduced dialect are either decomposed into legal
  Native4D operations or cause conversion to fail with a typed diagnostic.

Native4D is not an alternative PT2 importer. The staging is always:

```text
PT2
  |
  v
Native (existing six-axis dialect)
  |
  | existing canonicalization, folding, relayout simplification, and DCE
  v
canonical Native
  |
  | partial, checked dialect conversion
  v
Native4D
  |
  | Native4D-specific transformations
  v
optimized Native4D
```

The conversion is intentionally partial. A Native graph that does not satisfy
the shape precondition or contains an operation without a sound legalization is
not a failed implementation of Native4D; it is outside the dialect's domain.

## 2. Non-goals

- Direct PT2-to-Native4D conversion.
- Converting every possible Native graph.
- Representing arbitrary rank by silently flattening or changing layouts.
- Adding general grouped convolution merely to make the conversion total.
- Treating `T` or `D` as usable axes because their current extent happens to be
  one.
- Reimplementing numeric kernels already expressed by the Native
  `Compute (S)` functors.
- Claiming bit identity for algebraic decompositions that change f32
  materialization or reduction order.
- Making Native and Native4D share the same closed `op` variant.

## 3. Feasibility conclusion

The approach is plausible and fits the current architecture, with two important
constraints.

First, Native4D must be a real dialect with its own closed operation type and a
checked conversion boundary. Making it a flag on `Graph_ir.op` would allow mixed
or invalid graphs and would weaken exhaustiveness at every operation dispatch.

Second, sharing should happen at narrow interfaces. Numeric computation is
already factored at the right boundary: an operation defines a scalar algorithm
over `Semantics.SEMANTICS`, and Direct and Symbolic instantiate it. Native4D
operators can adapt their parameters to the existing Native operator modules and
call the same `Compute (S).pixel`. Transformation structure and verification can
also be shared, but the current implementation is concretely coupled to
`Graph_ir.op`, `Graph_shape`, and `Eval_symbolic`; extracting those dependencies
behind a dialect interface is substantial work.

The recommended implementation therefore proceeds in two steps:

1. implement a working Native-to-Native4D converter and dialect-specific
   evaluator using the smallest explicit shared interfaces;
2. extract the reusable transformation framework from the concrete needs shown
   by the second dialect, rather than first parameterizing the entire current
   framework speculatively.

## 4. The four-axis invariant

### 4.1 Physical representation

Native4D reuses the current six-axis storage order and tensor payload:

```text
N T D H W C
```

with the invariant:

```text
extent(T) = 1
extent(D) = 1
```

No data copy is required merely to cross the dialect boundary. A tensor accepted
by Native4D has the same shape, flat order, format, quantization metadata, and
coordinates it had in Native.

This shape preservation is important for verification: a correspondence member
in Native and its Native4D counterpart can continue to be compared at the same
`Vec6.coord`. The conversion does not need a new layout- or index-correspondence
language.

### 4.2 Enforcing rather than documenting the invariant

Reusing `Vec6.shape` directly in every public Native4D constructor would make an
invalid Native4D graph constructible. Native4D should instead expose an abstract
validated wrapper:

```ocaml
module Shape4 : sig
  type t

  val of_vec6 : Vec6.shape -> (t, error) Err.t
  val to_vec6 : t -> Vec6.shape
  val make :
    n:int -> h:int -> w:int -> c:int -> (t, Dim.error) Err.t
end
```

Runtime evaluation may unwrap `Shape4.t` to `Vec6.shape` before calling shared
compute. The tensor payload remains `Tensor.packed`; only the graph-facing shape
type enforces the dialect contract.

The graph validator must check:

- every live graph input, constant, node output, and graph output has `T=D=1`;
- every tensor-map key agrees with its recorded ID, as in the current
  `Graph_view`;
- every operation's inferred outputs satisfy the four-axis invariant;
- operations that carry axes name only `N/H/W/C`;
- a four-axis permutation does not use `T` or `D` as semantic axes;
- all operands resolve and the graph remains topological.

Dead-code elimination should run first. An unreachable tensor should not prevent
conversion, and dead real-shaped outputs such as max-pool indices should be
removed before the Native4D domain check.

#### Which tensors the invariant covers

"Every live graph input, constant, node output, and graph output" above is not
precise enough to implement, and the two natural readings are both wrong.
"Every tensor" rejects a graph over a captured weight nobody reads. "Every
tensor reachable from the outputs" accepts an unused non-four-axis graph
*input* — and dead-code elimination will not remove one: `Rewrite.apply` keeps
an unused user input deliberately, "because the graph's signature is externally
meaningful". The lowerer would then have no sound move, since copying the input
yields a graph the dialect rejects and omitting it silently narrows the
interface.

The rule is therefore:

> every effective `Input.Input`, read or not, **plus** every tensor reachable
> from `Graph.outputs`.

Constants are `Graph.inputs` entries too, discriminated only by `input_kinds`,
so the rule is stated in terms of the effective kind rather than of membership.
It cuts along the same seam `Rewrite.apply` already uses:

| tensor | policy | why |
|---|---|---|
| unused user input | **reject** | `Graph.inputs` is the graph's interface; a converter that quietly narrows an interface is worse than one that refuses |
| unread constant | **omit** | model-bound state, not interface — "a constant nobody reads is gone" |

Constant omission is the *lowerer's* job, not dead-code elimination's:
constant-input pruning happens inside `Rewrite.apply`'s graph rebuild, which
only runs when some recipe applies, so an isolated unread constant triggers
nothing and survives the whole pipeline. The lowerer omits it from the
destination graph, records a deletion cluster, and filters it out of the
returned constants.

This is why the domain check consumes a validated `Graph_view.t` rather than a
`graph`: given a graph it would have to validate it and so report
`Graph_view.error` as well, and every caller already holds a view.

### 4.3 Unit axes are not semantic axes

`T=D=1` alone is insufficient. For example, reducing `D` is numerically an
identity today, but retaining such an operation would make Native4D depend on a
non-Native4D axis. The converter should normalize harmless uses away where the
equivalence is obvious and otherwise reject them.

## 5. Dialect architecture

### 5.1 Common graph structure

The structural graph concepts are not dialect-specific:

- tensor, node, and group IDs;
- input kinds;
- tensor signatures and payload formats;
- SSA node outputs;
- the ordered group tree;
- graph inputs and outputs.

They should live in a common parameterized graph representation:

```ocaml
module Node : sig
  type 'op t = {
    id : Node_id.t;
    op : 'op;
    outputs : Tensor_id.t list;
  }
end

module Graph : sig
  type ('op, 'shape) t = {
    nodes : 'op Node.t list;
    root : Group.t;
    tensors : 'shape Tensor_sig_common.t Tensor_id.Map.t;
    inputs : Tensor_id.t list;
    input_kinds : Input.kind Tensor_id.Map.t;
    outputs : Tensor_id.t list;
  }
end
```

The exact type parameters may differ, but the representation must preserve the
repository rule that record types live in their own module as `t`.

The existing Native API can remain a type alias or thin specialization, avoiding
a broad rename from `Graph_ir` in callers.

### 5.2 Dialect interface

The reusable structural algorithms need a small, explicit dialect interface:

```ocaml
module type DIALECT = sig
  type op
  type shape
  type shape_error

  val operands : op -> Tensor_id.t list
  val map_operands : (Tensor_id.t -> Tensor_id.t) -> op -> op

  val output_shapes :
    op ->
    sig_of:(Tensor_id.t -> tensor_sig) ->
    (shape list, shape_error) Err.t

  val output_transfer :
    op -> output:int -> Output_transfer.t

  val eval_symbolic : graph -> Stage_program.t
  val pp_op : Tensor_id.t Fmt.t -> op Fmt.t
end
```

JSON support and direct evaluation may be separate interfaces if a generic
algorithm does not need them. Keeping the signature capability-based prevents a
printer or relation algebra from depending on evaluation.

### 5.3 What should and should not be functorized

Useful functors:

- `Graph_view.Make (Dialect)` — validation and indexing;
- `Transform.Make (Dialect)` — same-dialect recipe application;
- `Pattern.Make (Dialect)` and `Region.Make (Dialect)` — structural matching;
- `Output_transfer.Make (Dialect)` — claim closure;
- `Map_verify.Make_pair (Source) (Destination)` — cross-dialect symbolic
  validation.

Dialect-specific code:

- the closed operation variant;
- operation constructors and builder conveniences;
- shape dispatch;
- value dispatch;
- constructor-specific patterns and passes;
- Native-to-Native4D legalization rules.

The numeric operation modules should not acquire another dialect functor. They
are already functorized over `SEMANTICS`; Native4D wrappers should translate
configuration and delegate.

### 5.4 Same-dialect rewrite versus cross-dialect lowering

The current `Rewrite` recipes insert the same `Graph_ir.op` type they remove.
Native-to-Native4D conversion changes the operation type and therefore should not
be disguised as an ordinary Native rewrite pass.

Use two concepts:

```ocaml
Transform.Make (D)
(* D graph -> D graph, recipes and fixpoints *)

Native_to_4d.lower
(* Native graph -> Native4D graph * cross-dialect Graph_map *)
```

The lowerer constructs the destination graph in topological order, records value
and node clusters as it legalizes each source node, and records provenance for
created edges. Native4D transformations can then run through
`Transform.Make (Native4d_dialect)`.

## 6. Compute reuse

Native4D evaluation should reuse the existing scalar algorithms without wrapping
or copying tensors per operation.

### 6.1 Convolution

- `Conv2D` constructs `Conv.Conv2d.params` with `groups=1`.
- `DepthwiseConv2D` constructs `Conv.Conv2d.params` with
  `groups=in_channels`.
- `TransposedConv2D` constructs `Conv.Convolution.params` with
  `transposed=true`.

The Native4D weight layout should match the current Native layout wherever
possible:

```text
forward:    [N=out_channels, H=kernel_h, W=kernel_w, C=in_per_group]
transposed: [N=in_channels,  H=kernel_h, W=kernel_w, C=out_per_group]
```

Choosing a different public weight layout would require a conversion or
permutation before every shared compute call. If another backend eventually
requires a different layout, that change belongs in its lowering, not in the
semantic Native4D representation.

### 6.2 Other operations

Pointwise, pooling, reshape, permute, and keep-dimensions reduction can likewise
delegate to their existing Native compute modules after converting `Shape4.t`
to `Vec6.shape`.

This makes the Direct/Symbolic equivalence of the existing operations reusable
and lets the cross-dialect verifier compare expressions produced from the same
semantic primitives.

## 7. Native operation legalization

This section covers every operation currently in `Graph_ir.op`.

### 7.1 Direct or trivial legalization

| Native operation | Native4D result | Expected claim |
|---|---|---|
| `Add`, `Div`, `Mul`, `Sub` | Four-axis broadcast pointwise operation | `Identical` |
| `Add_scalar`, `Div_scalar` | Scalar form, or binary op with a captured scalar constant | `Identical` |
| `Clamp`, `Hardtanh`, `Relu`, `Sqrt` | Direct counterpart | `Identical` |
| `Clone` | Remove and tie its output to its input | `Identical` |
| `Max_pool2d`, `Avg_pool2d` | Direct counterpart | `Identical` when shared compute is used |
| `Permute` | `Permute4`, after proving it acts only on the four-axis domain | `Identical` |
| `Reshape` | `Reshape4`, when source and target both satisfy the invariant | `Identical` |
| `Discard` | Removed by DCE | Vacuous deletion |

Average pool should remain an operation. Replacing it with a depthwise
convolution multiplies every tap before accumulation instead of summing and then
dividing, changing f32 rounding.

### 7.2 Convolution family

`Conv2d`, `Conv2d_padding`, and non-transposed `Convolution` normalize as follows:

1. `groups = 1` → `Conv2D`;
2. `groups = in_channels` and weight input channels per group is one →
   `DepthwiseConv2D`;
3. any other grouping → reject.

`Conv2d_padding` first resolves `"same"` or `"valid"` to explicit axis windows.
The generic forward `Convolution` first resolves its kernel, padding, dilation,
and channel conventions to the same explicit form.

For transposed `Convolution`:

1. `groups = 1` → `TransposedConv2D`;
2. grouped or depthwise-transposed forms → reject unless Native4D later adds an
   explicit operation for them.

All three Native4D convolution operations delegate to the existing compute, so
normalization is expected to be `Identical`.

### 7.3 Linear

`Linear` becomes a `Conv2D` with:

```text
kernel      = 1x1
stride      = 1x1
padding     = 0
dilation    = 1x1
in_channels = in_features
groups      = 1
```

The existing Native linear weight is already laid out with output features on
`N` and input features on `C`, exactly the layout of a 1x1 Native convolution.
The spatial singleton loops add no arithmetic, so the input-channel reduction
order remains unchanged. This legalization should be `Identical`.

### 7.4 Batch matrix multiplication

The existing `Bmm` shape is:

```text
input: [H=batch, W=rows, C=contract]
mat2:  [H=batch, W=contract, C=columns]
out:   [H=batch, W=rows, C=columns]
```

It can become a 1x1 convolution only when `batch = 1`:

1. permute `mat2[H=1,W=contract,C=columns]` to the convolution weight layout
   `[N=columns,H=1,W=1,C=contract]`;
2. apply a 1x1 `Conv2D` to `input`.

The permutation is a value reindexing, and the convolution retains the same
contract-axis reduction order. This case is expected to be `Identical`.

For `batch > 1`, `mat2` varies with the output's `H` coordinate while
convolution weights are shared across spatial positions. Ordinary Conv2D cannot
represent that computation. Reject it. Supporting it requires a retained
MatMul/BMM operation or a materially broader dialect.

### 7.5 Mean

`Mean { keepdim=true }` becomes `MeanKeepDims`.

`Mean { keepdim=false }` becomes:

```text
MeanKeepDims dims
  ->
Reshape4 original_native_output_shape
```

The keep-dimensions result and packed result have the same surviving values in
the same row-major order; unit axes do not contribute to the offset. The
reshape is a reindexing/materialization of an already stored value.

The converter must still reject a reduction naming `T` or `D` after
normalization.

### 7.6 Batch normalization

Preferred path:

1. run constant folding so relaid convolution weights are constants;
2. run the existing batch-normalization fold;
3. fold the resulting parameter arithmetic;
4. convert the resulting convolution.

This covers the usual inference graph where batch normalization immediately
follows convolution.

A standalone inference `Batch_norm` with constant scale, bias, running mean, and
running variance can be turned into a depthwise 1x1 convolution by precomputing
one scale and offset per channel. That changes the arithmetic association and
rounding points, so the relation is generally `Equivalent`, not `Identical`.

If the parameters are dynamic and Native4D has no BatchNorm operation, reject.

### 7.7 RMS normalization

RMS normalization can be decomposed into:

```text
squared = Mul x x
mean    = MeanKeepDims dims squared
denom   = Sqrt (Add_scalar eps mean)
norm    = Div x denom
result  = Mul norm weight
```

This decomposition materializes `squared` before the reduction, while the
current fused Native operation computes products inside the reduction. It
therefore introduces a different f32 rounding boundary. Its claim is
`Equivalent`, not `Identical`.

If bit identity is required, retain a Native4D `RmsNorm` operation that delegates
to the existing fused compute.

### 7.8 Max pool with indices

When the indices output is consumed only by `Discard`, DCE removes the sink and
unused edge and the value output becomes ordinary `MaxPool2D`.

If the index output is live, conversion requires a Native4D argmax-pool
operation. The initial dialect should reject this case instead.

## 8. Operations actually required by the reduced dialect

The three convolution forms are not sufficient to represent the current Native
graphs. A practical initial Native4D dialect also needs:

- four-axis pointwise arithmetic with broadcasting;
- scalar constants or scalar pointwise forms;
- `Relu`, `Clamp`/`Hardtanh`, and `Sqrt`;
- `MaxPool2D` and `AvgPool2D`;
- `MeanKeepDims`;
- `Permute4`;
- `Reshape4`;
- graph constants, inputs, and the structural notion of discarded/dead output.

`Permute4` and `Reshape4` are essential legalization operations, not merely
optimizations. The existing PT2-to-Native lowering emits permutations around
layout-sensitive operations, and rank-changing Native reductions and views
produce reshapes. Existing Native passes remove many of them, but boundary
relayouts and real shape changes can remain.

The initial dialect does not need:

- general grouped convolution;
- `Split`, `Slice`, or `Concat`, if unsupported grouped convolution is rejected;
- general BMM/MatMul, if only the single-batch legal form is accepted;
- argmax-pool indices, if live indices are rejected;
- BatchNorm, if conversion requires it to be folded;
- RMSNorm, if `Equivalent` decomposition is acceptable.

If conversion later needs to become more complete, the smallest additions are:

| Missing source case | Smallest honest extension |
|---|---|
| General grouped convolution | Retain grouped convolution, or add channel split plus concat |
| Batched BMM | Retain BMM/MatMul |
| Live max-pool indices | ArgMaxPool/MaxPoolWithIndices |
| Dynamic standalone BatchNorm | BatchNorm or per-channel affine op |
| Bit-identical RMSNorm | Fused RMSNorm |

Expanding constant grouped-convolution weights into a dense zero-filled Conv2D
weight is a possible optimization for captured weights, but it increases storage
and computation and does not handle dynamic weights. It should be an optional
legalization, not the core definition of grouped convolution.

## 9. Mapping and verification

### 9.1 Conversion result

Native-to-Native4D conversion returns:

```ocaml
type result = {
  graph : Native4d_graph.t;
  map : (native_version, native4d_version) Graph_map.t;
  constants : Tensor.packed Tensor_id.Map.t;
}
```

The exact packaging should preserve the existing version-indexed-ID discipline.
Every source and destination ID is lifted through the snapshot that owns it;
conversion must not forge tags or rely on equal raw integers to identify
members.

### 9.2 Value claims

Use the existing relation meanings:

- `Identical` — bit-for-bit equal at every corresponding coordinate;
- `Equivalent` — equal in exact arithmetic, but rounding may differ;
- `Approximate` — a declared lossy representation was crossed;
- `Unverifiable` — structural correspondence without a value guarantee.

Direct wrappers, convolution normalization, clone removal, linear legalization,
and the supported BMM legalization should claim `Identical`.

Batch-normalization folding and decomposed RMS normalization should claim
`Equivalent`.

Created/deleted intermediates form vacuous clusters but still carry provenance.

### 9.3 Cross-dialect symbolic verification

The verifier needs two dialect-specific ways to produce `Stage_program.t`, but
the grounding, normalization, coefficient, probing, and report machinery can be
shared.

Because the converter preserves physical shapes and coordinates:

- the existing cluster shape-equality requirement remains valid;
- both sides can be checked at the same `Vec6.coord`;
- no axis map is needed in a correspondence cluster;
- constants remain bound per graph, as in current transformation verification.

The destination operation's `output_transfer` classification drives claim
closure after an `Equivalent` or weaker conversion. That classifier must be
exhaustive on the Native4D operation variant just as it is on Native.

### 9.4 Whole-graph validation

Use both layers:

1. symbolic per-cluster validation of the conversion map;
2. end-to-end evaluation of Native and Native4D on the same captured constants
   and test input.

The end-to-end comparison is a regression check, not a replacement for the
mapping verifier. It observes one input and only graph outputs; the symbolic
map check validates local obligations for every input within its proof and
budget limits.

## 10. Conversion errors

Conversion should return typed errors with enough context to diagnose the first
unsupported node:

```ocaml
type error =
  [ `Non_four_dimensional_tensor of Tensor_id.t * Vec6.shape
  | `Axis_outside_dialect of Node_id.t * Axis.t
  | `Unsupported_op of Node_id.t * Native.op
  | `Unsupported_grouped_conv of Node_id.t * int
  | `Unsupported_grouped_transposed_conv of Node_id.t * int
  | `Unsupported_bmm_batch of Node_id.t * Dim.extent Dim.t
  | `Live_max_pool_indices of Node_id.t * Tensor_id.t
  | `Dynamic_batch_norm of Node_id.t
  | `Shape of Native4d_shape.error
  | `Build of Native4d_builder.error
  | `Map of Graph_map.error ]
```

Diagnostics should include source node and tensor IDs. PT2 names remain
recoverable through the composed PT2-to-Native and Native-to-Native4D maps rather
than being copied into either graph IR.

### 10.1 The row is built progressively, and `Shape` is not a tag

Two corrections to the sketch above, both learned from writing it.

**The set cannot be declared in one place.** `Shape4`, the builder and the map
arrive in different stages, so the domain-check cases land first and each later
stage unions its own row in as the module that names it appears. Each module owns
its own row and callers widen — the idiom `Graph_shape.widen` already uses — so
`Shape4.of_vec6` returns *its* row, never this aggregate. The reverse would make
the error module depend on `Shape4` and `Shape4` depend on the error module.

**`` `Shape `` must not be a tag here.** The generic `Graph_view.Make` needs a
wrapper for its dialect's abstract shape error, because OCaml can inherit only a
row it knows expands to one; and `Rewrite.error` unions `Graph_map.error` with
`Graph_view.error`, so two same-named tags with different payloads do not
typecheck. Resolved one way, once:

- `Shape4.error` is **inherited directly**, with no tag at all;
- the generic view's is wrapped `` `Graph_shape of D.shape_error ``.

More generally: any new tag in a `transform/` error row has to be checked against
`Rewrite.error`'s full transitive union first. Three separate collisions were
found this way while planning — `` `Shape ``, `` `Graph_shape ``, and
`` `Output_arity `` (which `Graph_view.error` already owns with an incompatible
payload, so the map-level output errors are named `` `Graph_output_arity `` and
`` `Graph_output_mismatch ``).

## 11. High-level implementation plan

### Stage 0 — pin the domain with tests

Before changing graph infrastructure, add table-driven tests for the intended
conversion outcomes:

- reject a graph with non-unit `T`;
- reject a graph with non-unit `D`;
- accept a graph whose every live tensor has `T=D=1`;
- ignore an unreachable invalid tensor after DCE;
- accept `groups=1`;
- accept depthwise grouping;
- reject an intermediate group count;
- accept single-batch BMM;
- reject multi-batch BMM;
- lower `Mean keepdim=false` through `MeanKeepDims` plus reshape;
- drop dead max-pool indices and reject live indices.

These tests define the partial conversion contract independently of functor
design.

### Stage 1 — four-axis core and dialect

1. Add `Shape4` and its validation tests.
2. Define the Native4D closed operation variant and per-operation payload
   modules.
3. Add Native4D operand traversal, shape inference, printer, and JSON codec if
   persistence is required.
4. Add a validating Native4D graph builder.
5. Add Direct and Symbolic evaluation.
6. Implement convolution wrappers by delegating to existing Native compute.
7. Verify Direct versus Symbolic for every Native4D operation.

Acceptance criteria:

- invalid four-axis shapes cannot be built through the public API;
- every operation's inferred output is a `Shape4.t`;
- Direct and grounded Symbolic results agree bitwise on the direct-wrapper
  operations.

### Stage 2 — Native canonicalization prerequisites

1. Add graph-level DCE, including `Discard` sinks and unused multi-output edges.
2. Reuse the existing reshape-to-permute and relayout simplification pipeline.
3. Run constant folding before batch-normalization folding and again afterward.
4. Define one documented canonical pipeline that precedes Native4D conversion.

Acceptance criteria:

- dead max-pool indices do not reach the converter;
- constant convolution-weight permutations are folded;
- eligible convolution/batch-normalization pairs contain no BatchNorm before
  conversion;
- running the canonicalization pipeline twice is stable.

### Stage 3 — partial Native-to-Native4D lowerer

1. Walk the canonical Native graph in topological order.
2. Reuse or recreate graph inputs and constants under `Shape4`.
3. Implement direct operation translations.
4. Implement convolution classification.
5. Implement Linear and single-batch BMM legalization.
6. Implement keep-dimensions Mean legalization.
7. Implement RMSNorm policy: retained fused op or `Equivalent` decomposition.
8. Produce value clusters, node clusters, and provenance as conversion occurs.
9. Return typed errors at the first unsupported node.

Acceptance criteria:

- every successfully produced graph passes Native4D graph validation;
- all source graph outputs have destination correspondents;
- every created/deleted edge is represented in the map;
- conversion is deterministic, including IDs, node order, and diagnostics.

### Stage 4 — shared graph and transformation extraction

1. Extract common graph records and ID modules without changing serialized
   Native behavior.
2. Parameterize `Graph_view`, `Snapshot`, and ID universes by dialect.
3. Parameterize same-dialect `Pattern`, `Region`, `Recipe`, `Rewrite`, and
   `Pass`.
4. Keep existing Native passes as specializations of the generic framework.
5. Add one simple Native4D optimization pass to prove the abstraction is usable.

Acceptance criteria:

- all existing Native transform tests remain unchanged or differ only in module
  qualification;
- version-safety negative compilation tests still fail for the intended reason;
- Native and Native4D can each run a same-dialect rewrite through the shared
  framework.

Do not combine this refactor with an unrelated change to the verifier's proof
semantics. Stabilize the local-verification contract first, then parameterize it.

### Stage 5 — cross-dialect map verification

1. Generalize snapshots and graph maps to distinct source and destination
   dialects.
2. Make claim closure use the destination dialect's output-transfer table.
3. Make `Map_verify` obtain one symbolic program from each dialect.
4. Reuse shared grounding and local-frontier comparison.
5. Add mutation tests for each legalization: revert or deliberately break the
   lowering and demonstrate that verification no longer proves it or finds a
   counterexample where the claim is `Identical`.

Acceptance criteria:

- every `Identical` legalization is proved exhaustively on small fixtures;
- `Equivalent` legalizations are never reported as bit-identical;
- a deliberately wrong convolution, BMM weight permutation, or Mean reshape is
  detected;
- composing PT2-to-Native provenance with Native-to-Native4D provenance resolves
  final tensors back to PT2 origins.

### Stage 6 — representative model gates

Start with graphs already covered by the Native importer:

1. a small synthetic graph for every legalization;
2. ResNet-18 after constant and batch-normalization folding;
3. MobileNet-v2 or MobileNet-v3 to exercise depthwise convolution and scalar
   pointwise operations.

For each model report:

- whether conversion succeeded;
- the first unsupported operation when it did not;
- counts of each Native4D operation;
- conversion-map verification summary;
- Native versus Native4D output distance;
- runtime and memory changes, reported separately from correctness.

Model conversion should remain a gated real-data test, consistent with the
existing PT2 test strategy.

## 12. Scope estimate

The compute adapters themselves are small. The major cost is making graph
transformation and symbolic verification dialect-polymorphic without weakening
the current version-safety and claim-closure invariants.

Approximate implementation size:

| Work | Expected size |
|---|---|
| `Shape4`, Native4D IR, builder, shape/eval dispatch | 1,500–2,500 lines |
| Canonicalization/DCE and Native-to-Native4D lowering | 1,000–2,000 lines |
| Generic graph/transform extraction | 2,000–4,000 changed or new lines |
| Cross-dialect verifier adaptation | 1,000–2,000 changed or new lines |
| Unit, mutation, expect, and model tests | 1,500–3,000 lines |

A conversion/evaluation MVP without fully shared transformations is a medium
change, approximately 2,000–4,000 lines.

The complete design — reusable transformations in both dialects,
cross-dialect maps, and symbolic validation — is a large refactor, approximately
6,000–10,000 changed or new implementation lines plus tests. For one engineer
already familiar with the repository, a rough planning range is six to ten
engineer-weeks:

- one to two weeks for the Native4D core and shared-compute adapters;
- two to three weeks for canonicalization and the partial lowerer;
- two to four weeks for transformation/map/verifier generalization;
- one to two weeks for model gates, diagnostics, and stabilization.

These are planning ranges rather than delivery commitments. The main uncertainty
is not operator compute; it is how much of the current concrete transform API can
be parameterized without forcing churn into every existing pass.

## 13. Risks and mitigations

### Accidental weakening of the four-axis invariant

Mitigation: abstract `Shape4.t`, validate operation axes, and require every
output-shape function to return `Shape4.t`.

### Over-generalizing before a second dialect works

Mitigation: implement a vertical conversion slice first, then extract only the
interfaces it exercises.

### Treating algebraic equivalence as bit identity

Mitigation: document each legalization's rounding boundaries and make the
relation label part of its implementation test.

### A source operation looks decomposable but needs missing data movement

Mitigation: keep `Permute4` and `Reshape4` in the initial dialect. Reject cases
requiring split/concat or batch-varying convolution weights.

### Dead outputs make an otherwise valid graph fail conversion

Mitigation: make DCE a mandatory pre-conversion stage rather than teaching the
lowerer to ignore arbitrary live-looking structure.

### Functor-generated types become difficult to compose

Mitigation: keep IDs and relation algebra in non-generative common modules;
parameterize graph-owning modules only, and package version tags existentially
as the current `Snapshot` design does.

### Existing Native behavior changes during extraction

Mitigation: preserve the existing Native public specialization, JSON goldens,
version-safety compile failures, and full transform suite before adding
Native4D-specific behavior.

## 14. Recommended first milestone

The first milestone should demonstrate the complete idea on a narrow graph, not
implement every operation:

1. `Shape4` and a Native4D graph with `Add`, `Relu`, `Conv2D`,
   `DepthwiseConv2D`, `MeanKeepDims`, `Reshape4`, and `Permute4`;
2. Direct and Symbolic evaluation through existing compute;
3. a Native-to-Native4D converter for those operations plus Linear;
4. an explicit conversion map;
5. symbolic verification of that map;
6. one deliberately incorrect lowering that the verifier detects.

That milestone proves the three uncertain architectural claims:

- a second dialect can share compute without copying tensors;
- one conversion can produce the existing style of correspondence and
  provenance evidence;
- the verifier can compare different closed operation types at the same
  coordinates.

Only after those hold should the rest of the Native operation matrix and the
full same-dialect transformation functors be completed.
