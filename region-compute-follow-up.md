# Operation-owned Pixel and Region computation

## Adopted direction

An operation should expose its computation in the form natural to that
operation. Pixel computation is the right source form for many pointwise,
indexing, pooling, convolution, and ordinary reduction operations. It is not
the right source form for an operation whose algorithm intrinsically computes
and shares values over several outputs.

RMSNorm, LayerNorm, and plain Softmax should therefore own an authoritative
Region computation rather than an authoritative Pixel computation plus an
optional operation-specific regionization. Native4D counterparts should make
the same choice: after their checked four-axis parameter conversion, they reuse
the same Region computation instead of delegating to a Pixel computation.

The core rule is:

```text
operation source form:
  Pixel operation  -> operation-owned Pixel computation
  Region operation -> operation-owned Region computation

common Kernel form:
  Pixel computation  -> Region_program.pixel pixel
  Region computation -> Region program unchanged
```

`Kernel.Value.computation` remains one `Region_program.t`. Pixel remains the
degenerate, singleton/no-local subset of that common representation; there is
no need to add a semantic `Pixel | Region` sum to Kernel merely because
operations may author their computations differently.

## Why Pixel is not the universal operation source form

`Compute(S).pixel` is a good abstraction when one output can be computed
independently and efficiently from its coordinate. For RMSNorm, LayerNorm, and
Softmax, one output is observable independently, but independent Pixel
evaluation is not the natural algorithm:

| Operation | Shared Region work for a slice of extent `K` | Fresh scalar projection for all `K` outputs |
|---|---:|---:|
| RMSNorm | one `K`-term reduction plus `K` emissions | `K` separate `K`-term reductions |
| LayerNorm | two `K`-term reductions plus `K` emissions | `K` separate pairs of `K`-term reductions |
| Softmax | max and denominator passes plus `K` emissions | `K` separate max and denominator pairs |

The current implementation keeps two handwritten descriptions for these
operations. For example, `Norm.RmsNorm.Compute(S).pixel` builds the scalar
formula and `Norm.RmsNorm.Region.try_regionize` separately builds structurally
equivalent Region syntax. The retained `Region_context.pixel` is not used to
extract those locals or the emitter; reconstruction checks the two trees only
after the Region tree has been constructed.

That was a useful migration path, but it should not be the final ownership
model. Maintaining an intentionally inefficient Pixel algorithm solely as a
second semantic source adds drift risk and prevents Direct execution from
receiving Region sharing.

## Scalar projection is required; scalar authorship is not

Every supported Region program must still define the value at an individual
output coordinate. Its strict scalar projection is:

```text
project(program, output):
  key = Region_partition.key_of_output program.partition output_shape output
  locals = evaluate program.locals afresh, in declaration order, at key
  return evaluate program.output once at (locals, output)
```

This projection receives the same essential information as a Pixel
computation: the input environment, output shape, and requested output
coordinate. It computes the owning key and evaluates every local needed by
that output. It need not emit or store the other outputs owned by the key.

Calling `project` independently at every Pixel boundary is correct but can be
extremely inefficient. It deliberately recreates the repeated work that
Region materialization removes. It is appropriate for:

- an API that fundamentally requests one value;
- reference evaluation and differential tests;
- diagnostics and inspection;
- a compatibility adapter for a legacy Pixel-only consumer.

It is not an acceptable production materialization strategy. Production
execution must evaluate locals once per key and emit all outputs owned by that
key. The existing `Region_execution.value_at`/`Region_eval.value_at` path is
the model for scalar projection; `Region_execution.materialize` is the model
for shared execution.

The projection law is part of the Region language contract:

```text
materialize(program)[output] = project(program, output)
```

The equality is bitwise in strict mode, including the final
`Kernel.Result_conversion` boundary. Shaped locals preserve this law: scalar
projection may have to reconstruct an entire local vector or tile for one
output, which is inefficient but semantically valid. A future multi-output
program adds an output/result ordinal to the projection.

This law depends on pure emitters and region-scoped locals. One output may not
observe mutation performed while emitting another output. Ordered reductions
and folds remain semantic; traversal among independent keys and pure emitted
outputs remains lowering and scheduling policy.

## Region-domain coverage and traversal

A Region computation is defined over the whole logical output domain, not only
over a representative key. For output domain `D`, partition `P`, keys
`keys(P, D)`, and owned outputs `outputs(P, D, key)`, validation and tests
must establish:

```text
coverage:
  union { outputs(P, D, key) | key in keys(P, D) } = D

disjointness:
  outputs(P, D, key_a) intersect outputs(P, D, key_b) = empty
  whenever key_a <> key_b

ownership:
  key_of_output(P, D, output) = key
  for every output in outputs(P, D, key)
```

These are representation invariants independent of the chosen physical scan
order. The current canonical `Vec6` N/T/D/H/W/C traversal with C innermost is
the deterministic reference schedule. An optimized schedule may reorder
independent keys or pure emissions, but it must preserve the same coverage,
single ownership, local lifetime, ordered reductions, and result conversion.

Foundation tests over a synthetic whole-C partition are necessary but not
sufficient once operations author Region programs. Add operation-level trace
tests for RMSNorm, LayerNorm, and Softmax, and for their Native4D counterparts.
Use non-trivial shapes and configurations so unit axes do not conceal mistakes:

- several independent region keys;
- every supported Softmax axis;
- single- and multi-axis normalization;
- non-adjacent normalized axes;
- extent-one normalized and non-normalized axes;
- optional affine operands;
- Native4D N/H/W/C axis mappings, with T/D visibly singleton;
- configurations where the same number of outputs is partitioned differently.

A deterministic trace should show the program before execution and every
logical traversal step. For example:

```text
program:
  region [N=singleton T=singleton D=singleton H=singleton W=singleton C=whole]
  let l0 = reduce_sum ...
  let l1 = ...
  emit ...

key 0: (n=0,t=0,d=0,h=0,w=0,c=0)
  local l0: <open expression at this key>
  local l1: <open expression using l0>
  outputs:
    (n=0,t=0,d=0,h=0,w=0,c=0)
    (n=0,t=0,d=0,h=0,w=0,c=1)
    (n=0,t=0,d=0,h=0,w=0,c=2)
  emitter: <open expression using l0/l1 and each output coordinate>

key 1: (n=0,t=0,d=0,h=0,w=1,c=0)
  ...

summary:
  output-domain cells: 6
  keys: 2
  visited outputs: 6
  duplicate outputs: 0
  missing outputs: 0
```

The trace need not expand a reduction once per iteration or substitute locals
into a potentially exponential Pixel tree. It should print:

1. the stable `Region_program.pp` form once;
2. each canonical key;
3. the ordered local declarations evaluated for that key, with stable local
   names and their open expressions;
4. the outputs owned by that key;
5. the emitter expression and, optionally, evaluated local/output values;
6. a coverage/disjointness summary computed independently from the executor's
   stores.

The independent summary matters. Merely checking that materialization produced
a dense tensor cannot detect that one output was stored twice and another was
left at an initialization value. Tests should maintain a visit count for every
coordinate, require exactly one visit, and compare every enumerated output's
`key_of_output` with the current key.

Keep structural and numerical evidence separate. A traversal trace proves
partition shape, local order, and output ownership. Differential tensor tests
prove values. Execution counters prove locals were shared once per key. All
three are required for a non-trivial Region-authored operation.

## Native execution ownership

For a Pixel-authored Native operation, the current paths remain useful:

```text
Compute(Direct).pixel   -> direct Pixel schedule
Compute(Symbolic).pixel -> Expr.Value.t -> Region_program.pixel
```

For a Region-authored Native operation, Direct and Symbolic consume the same
operation-owned program:

```text
Norm.RmsNorm.Computation.program
Norm.LayerNorm.Computation.program
Reduce.Softmax.Computation.program
                 |
                 v
          Region_program.t
            /          \
   Direct materialize   Symbolic/Kernel construction
```

`Direct` should execute the Region traversal, not call scalar projection once
per output. `Symbolic` should carry the Region program directly. If the legacy
`Stage_program.Stage.body : Expr.Value.t` boundary temporarily requires a
Pixel expression, it may receive a derived specialization, but that
specialization is compatibility plumbing and not the operation's source
definition.

The common Region runtime continues to own:

- validation of partitions, local scope, invariance, and limits;
- lowering and one-time Pixel/Region execution classification;
- key traversal and local allocation/evaluation;
- traversal and stores for outputs owned by a key;
- result conversion;
- backend scheduling, tiling, placement, and vectorization.

An operation owns its partition, ordered local declarations, emitter, and any
operation-specific parameter checks. It does not own physical loops or stores.

## Native4D is a peer computation owner

Native4D has its own closed operation type and must remain a real dialect.
`Ops4.Rms_norm`, `Ops4.Layer_norm`, and `Ops4.Softmax4` are therefore not
silently treated as Native graph operations. They expose their computation as
Native4D operations while reusing the common numeric Region definition after a
checked mapping:

```text
Ops4.Rms_norm
  Axis4 dims -> Native axes
  Native4D operands/signatures -> common sources
  -> shared RMSNorm Region computation

Ops4.Layer_norm
  Axis4 dims -> Native axes
  Native4D operands/signatures -> common sources
  -> shared LayerNorm Region computation

Ops4.Softmax4
  Axis4 axis -> Native axis
  Native4D operand/signature -> common source
  -> shared Softmax Region computation
```

This is operation ownership with implementation reuse. Native4D owns the typed
operation, its four-axis domain, parameter mapping, optional-operand wiring, and
rejection behavior. The arithmetic, local order, emitter, and strict numerical
behavior have one shared definition. No second Region algorithm belongs in
`Eval_op4`.

The present `Graph_shape4.rms_params`, `layer_norm_params`, and
`softmax_params` adapters already demonstrate the required checked axis
translation, and `Eval_op4` uses them before delegating to Native
`Compute(S).pixel`. The Region direction changes the delegated computation
form, not the four-axis legality rule:

- only N/H/W/C may be named by Native4D parameters;
- mapped T/D remain unit, non-semantic axes and stay `Singleton` in the
  resulting partition;
- shape and computation must use the same parameter adapter;
- unsupported Native semantics do not become legal merely because the common
  Region representation uses `Vec6`;
- Native4D Direct must share Region locals just as Native Direct does;
- Native4D Symbolic must carry the same Region program rather than recreate a
  Pixel-only stage.

Pixel-authored Native4D operations continue to translate their parameters and
delegate to the corresponding Native Pixel computation. The rule is based on
the natural computation form of the operation, not on the dialect.

## Operation-facing API direction

The exact OCaml packaging can be introduced incrementally, but the semantic
shape should be explicit. For example:

```ocaml
module RmsNorm = struct
  module Computation = struct
    val program :
      limits:Kernel.Limits.t ->
      output:int ->
      output_shape:Vec6.shape ->
      params ->
      x:Tensor_sig.t ->
      weight:Tensor_sig.t ->
      (Region_program.t, error) result
  end
end
```

The Native4D operation exposes a corresponding computation entry which maps
its typed parameters and delegates to that shared program builder. Resolved
optional operands should be passed by role; a shared context should not search
an untyped list of synthetic constants by their floating-point value.

The graph evaluator still needs an exhaustive dispatcher over each dialect's
closed operation type. That dispatcher owns only provenance decoding,
operand/signature resolution, and the call to the operation-owned computation.
It does not own the formulas. The existing generic `Regionizer` therefore
either becomes a computation-form dispatcher or disappears into the normal
graph-to-computation boundary; it is no longer an optional optimization
selector for Region-authored operations.

## Validation, fallback, and oracles

For a Region-authored operation, admission requires:

- valid typed parameters and resolved operands;
- `Region_program.check` under the actual requested Kernel limits;
- region-domain coverage, disjointness, and single ownership;
- a supported lowering or an explicit reference-execution fallback;
- the operation's numerical and result-conversion contract.

It does not require reconstruction against a second handwritten Pixel
definition. `Region_program.reconstructs` remains useful for transformations,
migration checks while the old Pixel implementation still exists, and proofs
that a derived Region candidate preserves a Pixel-authored operation.

A construction or validation failure for an operation whose authoritative
form is Region is an implementation error or a structured unsupported-limit
error, not a silent request to run a different handwritten algorithm. A
backend may use the reference Region interpreter or derived scalar projection
when optimized Region lowering is unavailable. That choice can be slow, but it
preserves one computation definition.

Removing the independent Pixel definition also changes the oracle matrix.
Validation should include:

1. external ATen/reference fixtures for operation semantics;
2. Region reference evaluation versus lowered Region execution;
3. Region materialization versus fresh scalar projection at every output;
4. Native versus Native4D comparison after the checked axis mapping;
5. operation-level traversal traces and independent coverage proofs;
6. adversarial floating-point, optional-operand, degenerate-dimension, and
   multi-key cases;
7. counters proving production Direct, Symbolic/Kernel, Native, and Native4D
   paths share locals rather than invoking scalar projection per output.

The old Pixel implementation remains valuable during migration as a temporary
differential oracle. It should not remain the permanent semantic authority for
an intrinsically Region-authored operation.

## Consequences for the existing Region documents

This decision supersedes statements that make `Compute(S).pixel` the universal
operation semantic source or describe Region construction for RMSNorm,
LayerNorm, and Softmax as an optional Kernel optimization. The main
`region-compute-design.md` and `region-native-implementation-guide.md` now
record the adopted direction; the completed optional-regionizer plan is marked
as historical migration evidence.

The Native4D design documents also need a narrow correction. Their rule against
duplicating Native numeric computation remains valid, but delegation is no
longer universally delegation to `Compute(S).pixel`. A Native4D counterpart
delegates to the Native operation's natural computation form: Pixel or Region.

The completed Foundation remains valid:

- `Region_program.pixel` still embeds Pixel-authored operations;
- every Kernel value still stores one `Region_program.t`;
- scalar projection and the reference interpreter remain valid;
- Pixel lowering still preserves the existing fast path;
- Region lowering still owns common traversal and stores.

The completed scalar-Region implementation and its logs remain useful migration
evidence. Their optional regionizer, reconstruction admission, and unchanged
Direct/Native4D Pixel paths describe the implemented stepping stone, not the
final operation-computation ownership model adopted here.

The remaining implementation gap is specified by
[`region-operation-computation-implementation-plan.md`](region-operation-computation-implementation-plan.md)
and tracked in
[`region-operation-computation-implementation-todo.md`](region-operation-computation-implementation-todo.md).
