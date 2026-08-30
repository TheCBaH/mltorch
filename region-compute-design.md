# Region computation: shared work, normalization, softmax, and attention

## Design goal

Extend the computation language from independent per-output-scalar evaluation
to structured **output-region computation**.  An output region computes shared
locals once, then emits all outputs that consume them.  The near-term target is
`Theta(K)` evaluation of RMSNorm, LayerNorm, and Softmax over a normalized slice
of length `K`, replacing the current `Theta(K^2)` repeated reductions.  The
same representation must later support shaped node-local tiles, efficient
SDPA, and restricted whole-graph lowering to one region node.

The design uses **region** and **tile** for complementary concepts:

- an **output region** is a semantic set of output coordinates that may share
  local values;
- a **schedule tile** is a physical loop/memory block selected by a backend;
- a **reduction domain** is the ordered set of coordinates folded to construct
  a local value;
- a **local** is a scalar, tuple of scalars, or region-local tensor whose
  lifetime is one output-region invocation.

A schedule tile is the positive locality mechanism used for Conv and fused
stencil execution: it bounds an output block, its required input/weight
footprints, its accumulators, and its node-local storage.  An output region
states semantic sharing and ownership; a schedule tile selects a physical
blocking of that computation.  The selected Kernel Region records the tile as
a concrete `Block` partition and explicit locals.

"Sharing region" is slightly more descriptive, but `output region` fits the
existing vocabulary (`output shape`, `output coordinate`) and does not imply a
particular hardware workgroup.  A normalization **slice** or one-axis **fiber**
is then a common special case of an output region, not the general term.

Regions are a structured semantic/kernel construct; a separate schedule lowers
them to loops and selects physical tiles.  Cross-output sharing is not an
opaque `SEMANTICS` callback, and a mutable scan of output pixels is not
observable.  Those forms would mix meaning with schedule and would make fusion,
verification, and alternate backends substantially harder.

Near term, support axis-aligned regions, ordered reductions, several named
scalar locals, and a pure output emitter.  That is enough for RMSNorm,
LayerNorm, and Softmax.  Design locals as shaped values from the start, even if
the first implementation admits only shape `[]`: attention and fused
`conv + add + rmsnorm` both eventually need a local vector or tensor.

Graph fusion is deliberately restricted: admit a candidate only when its
**entire graph can be lowered as one region node**.  There is no initial search
for several interacting fusion clusters and no partially materialized interior
cut.  A strict numerical mode preserves every original node's `round_f32`; an
optional relaxed mode may elide internal f32 materialization rounds and makes
an `Equivalent`, never `Identical`, claim.

"One region node" does not mean one scalar expression: the node may contain a
structured DAG of locals and phases.  It means the admitted graph has one
execution boundary, no graph-visible intermediate tensors, and one coherent
region schedule.  The original graph may remain intact for inspection; the
single node can be a compilation representation rather than a destructive
graph rewrite.

## 1. Why the current model is inefficient

`Compute(S).pixel` is the algorithm for one scalar.  `Schedule.evaluate` owns
the dense six-axis output traversal and materializes each result.  `Direct`
executes reductions immediately; `Symbolic` places reductions in a scalar
expression that is later grounded once per output coordinate.  Neither layer
has a lifetime in which a value can be computed once and reused by neighboring
outputs.

For one independent slice of length `K`:

| Operation | Current direct reduction work | Region computation |
|---|---:|---:|
| RMSNorm | `K` outputs x `K` sum-of-squares terms = `K^2` | one `K`-term reduction + `K` outputs |
| LayerNorm | `K` outputs x two `K`-term passes = `2K^2` | two `K`-term passes + `K` outputs |
| Softmax | `K` outputs x max and denominator passes = `2K^2` | max pass + denominator pass + `K` outputs |

Ignoring small pointwise costs, the new work is respectively about `2K`,
`3K`, and `3K`.  This is an asymptotic improvement, not just fewer closures or
faster indexing.

The symbolic path can be worse than the table suggests.  An OCaml binding of a
`Symbolic.t` construction is not an expression-level `let`.  Reusing it can
place a reduction beneath another reduction, so grounding may reevaluate the
inner reduction for each iteration of the outer one.  A region representation
needs explicit local identity/DAG sharing; host-language sharing is not a
sufficient contract.

A specialized dot can make one contraction much cheaper by avoiding generic
coordinate construction and loads.  It does not stop the same contraction or
normalization statistic from being invoked once per dependent output.  Dot and
regions are complementary: regions remove repeated invocations; contraction
intrinsics optimize the remaining invocation.

## 2. Semantic shape of a region program

A region program should describe four things:

1. how the output tensor is partitioned into regions;
2. the ordered computations performed once for each region;
3. the local values made available to later phases;
4. the pure value emitted at every output coordinate in that region.

An explanatory syntax is:

```text
region axes [C] in output x.shape {
  let sumsq : scalar = reduce_sum c in [0, C) {
    x[outer..., c] * x[outer..., c]
  }
  let inv : scalar = 1 / sqrt(sumsq / C + eps)

  emit c in [0, C) {
    x[outer..., c] * inv * weight[c]
  }
}
```

`axes [C]` means that coordinates differing only in `C` belong to one region;
the other output axes identify the region.  For normalization over several
axes, the region varies over all of those axes and the remaining axes identify
it.  This covers the current Native `dims : Axis.t list` rather than assuming
normalization always has exactly one non-unit dimension.

The initial axis-aligned partition can be represented by one policy per output
axis:

```ocaml
type axis_region =
  | Singleton                 (* one region per coordinate on this axis *)
  | Whole                     (* full axis belongs to one region *)
  | Block of Op_config.Pos.t  (* later; final block may be shorter *)
```

`Singleton`/`Whole` is sufficient for the near-term operation definitions.
`Block` appears in an instantiated Kernel Region after a tiling transformation.
The Conv source program does not hard-code one machine tile size: schedule
selection chooses block extents, inserts the corresponding locals/phases, and
produces the Region program that the selected Kernel records.  Loop lowering
may still sub-block copy or emission loops without changing the declared local
lifetime.

The representation must establish that regions form a disjoint, complete
partition of the output domain.  This makes every output have exactly one
writer and avoids assigning meaning to output traversal order.

## 3. Phases, locals, and reductions

One temporary scalar is too restrictive even for the immediate targets:
LayerNorm needs a mean followed by a variance, and Softmax needs a maximum
followed by a denominator.  The minimum useful model is an ordered sequence of
named scalar bindings followed by an emitter.  A binding may use earlier
bindings.

```text
region axes [C] {
  let m = reduce_max C { x[C] }
  let z = reduce_sum C { exp(x[C] - m) }
  emit C { exp(x[C] - m) / z }
}
```

Reduction iteration order is semantic because floating-point addition is not
associative and max has repository-specific NaN/tie behavior.  Initially,
retain the current lexicographic, left-to-right order.  Parallel or tree
reductions are alternative numerical schedules and require an explicit
equivalence/tolerance policy; they are not automatically `Identical`.

The emitter should be pure: it may read inputs and locals and compute the value
for its coordinate, but one emitted output must not mutate state observed by
the next.  Consequently, the output scan order is a scheduling choice and does
not need to appear in operation semantics.  A backend must nevertheless make
its chosen scan explicit in Loop IR.  Only ordered reductions/folds expose an
order at the semantic level.

Locals should have explicit identity and shape:

```ocaml
type local_shape = Scalar | Tensor of region_relative_shape
type local = { id : Local_id.t; shape : local_shape; body : ... }
```

Phase 1 can validate that every local is `Scalar`.  Keeping shape in the model
avoids a redesign when a region needs cached logits, a vector accumulator, or a
producer tile.  A tuple of scalar state should either be a first-class product
or several locals computed by one multi-result fold; encoding it as a tiny
tensor would obscure register placement and synchronized updates.

Useful reduction forms are:

- built-in ordered `sum` and `max`, preserving current behavior;
- later, a typed multi-accumulator `fold` with explicit initialization,
  update, and finalization;
- rectangular, possibly nested reduction domains rather than only a single
  integer interval.

Arbitrary mutation should not be the source language.  Structured folds make
liveness, bounds, cost, and numerical order inspectable.

## 4. Where this belongs in the architecture

### Region Kernel IR, lowered by a schedule

Region program is the Kernel computation boundary.  Existing
`Compute(S).pixel` definitions enter it through the singleton-region
constructor.  A separate regionization pass converts selected Pixel-form
Kernel programs into the extended form with explicit partition and locals.
Kernel IR plus a schedule then lowers to Loop IR.  The current Kernel design
already says scalar expressions describe meaning while Loop IR owns final
loops, placement, and stores.  Regionization identifies required sharing and
legal local lifetimes; schedule selection may then transform the program into
an instantiated Region variant with concrete block sizes and cache locals.
Loop IR fixes the remaining machine-level loop and placement details.

Conceptually:

```text
Graph operation -> existing Compute.pixel -> Expr body -> Region.pixel
                                                        -> Pixel-form Kernel
                                                        -> regionize selected Kernel
                                                        -> Region Kernel program
                                                           (partition, locals,
                                                            reductions, emitters)
  -> schedule (loop order, physical tiles, placement, vectorization)
  -> Loop IR
  -> interpreter / JS / C
```

Mechanically check the converted Region form against the existing Pixel
program.  Expression comparison substitutes Region locals into the emitter and
compares the result with the source Pixel expression; concrete differential
tests remain an independent oracle.  Operation-specific `Compute.pixel`
definitions remain unchanged and authoritative for operation semantics.

The derived Pixel path evaluates the region containing the requested coordinate
and is intentionally inefficient.  It remains useful as a reference.

### Transitional alternative: `prepare` plus `emit`

For a small Native-only experiment, an operation can expose:

```ocaml
val region_spec : ... -> Region_spec.t
val prepare : region_coord -> locals
val emit : locals -> output_coord -> value
```

and a region-aware direct schedule can invoke `prepare` once.  This can prove
the performance thesis quickly.  It should not become the permanent public
language: heterogeneous `locals` complicate `Eval_op` packaging, the symbolic
path still needs explicit local nodes, and fusion cannot analyze opaque OCaml
closures.

### What not to put in `SEMANTICS`

`SEMANTICS.t` is a scalar value domain.  A scalar primitive for cross-output
sharing cannot define who invokes it, how long its value lives, or which
outputs consume it.  A primitive that also emits outputs would make
`SEMANTICS` own loops and stores, undoing the current separation between
algorithm and schedule.  Keep scalar intrinsics such as a future dot inside
`SEMANTICS`; put cross-output sharing in Kernel/schedule structure.

## 4a. Pixel and Region programs coexist

Pixel is a strict subset of Region.  There is one underlying computation
language and one Kernel computation type.  The existing scalar/index/boolean
expressions, loads, intrinsics, and ordered reductions remain the expression
language used inside Region.  Region extends it with:

- an output partition;
- region keys and coordinates;
- named locals with region lifetime;
- ordered preparation phases;
- one or more output emitters.

A Pixel program is the degenerate Region program whose partition is
`Singleton` on every output axis and which has no region locals.  Its emitter
is the existing per-output expression, including any reductions nested inside
that expression:

```ocaml
type Region_program.t = {
  partition : Region_partition.t;
  locals : Region_local.t list;
  outputs : Region_output.t list;
}

val pixel : output:Expr.Value.t -> Region_program.t
(* partition = singleton on every axis; locals = [] *)
```

This is an explanatory interface; the final representation must also cover
multi-output programs and logical Kernel values.  The invariant is that there
is no semantic sum type `Pixel | Region`.  `Region_program.pixel` is a
compatibility/smart constructor for the restricted form.

Kernel carries the computation it implements by storing its Region program,
including the semantic output partition and locals.  A printer or analysis may
classify a program with an all-singleton partition and no locals as `Pixel`, but
that classification is derived.  The schedule reads the declared partition;
it does not infer sharing by comparing independent Pixel calls.

The graph operation, parameters, operands, output shape, and externally visible
result remain unchanged.  Region replaces the Kernel computation description,
not `Graph_ir` and not shape inference.

A Region program contains a Pixel-shaped emitter parameterized by explicit
locals.  Its semantic relationship to Pixel is:

```text
pixel(out):
  key    = region_of(out)
  locals = prepare(key)
  return emit(locals, out)

evaluate_region(key):
  locals = prepare(key)
  for out in outputs_of_region(key):
    store out = emit(locals, out)
```

`prepare`, its reductions, and `emit` are structured IR, not opaque callbacks.
The two evaluations differ only in local lifetime: reference Pixel
specialization reconstructs the locals for the requested coordinate, while
Region evaluation shares them across all emissions.  This relationship makes
Pixel-subset semantics part of the language definition rather than an
evaluator convention.

One evaluator handles both cases:

```text
for key in program.partition.region_keys:
  locals = evaluate program.locals at key
  for out in program.partition.outputs(key):
    evaluate and store program.outputs at (locals, out)
```

For `Region_program.pixel body`, every key owns exactly one `out`, `locals` is
empty, and this loop is the current dense Pixel schedule.  No runtime feature
test or alternate semantic dispatch is necessary.

### Incremental compatibility with the current implementation

The unified Kernel representation does not require an immediate rewrite of
all `Compute(S).pixel` modules:

1. `Eval_symbolic` continues producing the existing `Expr.Value.t` stage body.
2. The `Stage_program -> Kernel` adapter wraps that body with
   `Region_program.pixel`.
3. The Region interpreter first supports this degenerate form, reproducing the
   current schedule.
4. Converted operations emit a non-degenerate Region program directly.
5. Existing `Eval_direct` and `Eval_symbolic` Pixel paths remain independent
   differential oracles during migration.

Thus Pixel and Region coexist at the source/front-end boundary temporarily,
while Kernel and every downstream schedule see only Region programs.  Existing
Pixel code has a mechanical embedding; no operation becomes unsupported during
migration.

### Pixel performance is a lowering invariant

Pixel is a semantic subset of Region, but it does not execute through the
general Region interpreter's machinery in a production hot loop.  Validate and
classify the program once before execution, then select a specialized lowering:

```ocaml
(* Derived execution form, not a second semantic program type. *)
type Lowered.t =
  | Pixel_loop of Pixel_loop.t
  | Region_loop of Region_loop.t
```

For an all-singleton partition with no region locals, Pixel lowering performs
these erasures:

- fuse the region-key loop with the output-coordinate loop;
- identify the region key directly with the output coordinate;
- remove `region_of`, `outputs_of_region`, preparation, and local-environment
  construction;
- compile/call the emitter directly at that coordinate;
- retain the existing dense six-axis traversal with C innermost;
- pass the coordinate directly to existing loads and index expressions.

The classification branch occurs once per Kernel invocation, never once per
output.  The Pixel loop allocates no region key, bounds object, local map, or
callback closure per output.  `Region_program.pixel` is a private smart
constructor that can carry a validated singleton witness; any transformation
that changes partition or locals invalidates and recomputes that witness.

The semantic IR remains uniform while the execution IR is specialized.  This
is the same separation by which a compiler represents a constant uniformly but
still lowers it without a runtime lookup.

During incremental adoption, preserve the current fast paths explicitly:

- `Eval_direct` continues invoking `Compute(Direct).pixel` and
  `Schedule.evaluate` for existing Pixel-form operations;
- `Eval_symbolic` continues building one expression once and
  `Schedule.ground` evaluates it over the current coordinate loop;
- `Stage_program -> Kernel` wrapping is O(1) per stage and performs no AST copy;
- Region-native execution is selected only for a genuinely non-degenerate
  partition or for an admitted fused single-node graph;
- an unfused graph of Pixel operations keeps its existing evaluation path.

After a selected Kernel is regionized, its production path comes from
ahead-of-loop lowering of the Region emitter, not from repeatedly invoking the
general reference specialization shown above.  The original operation Pixel
program remains the semantic oracle.  A Region-to-Pixel specialization may
reconstruct locals for equivalence checking; the optimized lowering must
inline/scalarize singleton locals or reject that fast classification.

Pixel performance acceptance requires, for representative pointwise,
convolution, and indexing operations:

- unchanged asymptotic work and input-load count;
- zero additional allocation in the per-output hot path;
- no new dynamic dispatch or partition test per output;
- identical output scan and arithmetic order;
- benchmarked runtime/allocation results no worse than the existing evaluator
  beyond a small declared noise threshold.

Keep a benchmark for a trivial pointwise operation as the most sensitive
overhead detector: expensive convolution can hide a region-wrapper cost that
would dominate `Relu`, `Add`, or `Clone`.

Distinguish a Pixel-form program from Pixel specialization of a true Region
program.  The former receives the zero-overhead Pixel loop above.  The latter
computes one output of a multi-output-sharing region and may necessarily repeat
preparation; it exists for reference evaluation, debugging, and equivalence
checking, not as a production schedule.  For RMSNorm this reference
specialization has the same repeated-reduction complexity as the current Pixel
implementation, while normal execution selects the non-degenerate Region
schedule.

Pixel-form programs also remain useful inside a larger Region program.  A
per-coordinate producer can be specialized at every coordinate requested by a
consumer region and inserted as a scalar expression.  This directly handles
pointwise producers and, after footprint analysis, bounded stencil producers.
It does not by itself share a reduction whose result is invariant over several
outputs: RMSNorm and Softmax require regionization that hoists the invariant
subexpression into a Region local.  Consequently:

- a Pixel-form operation is a degenerate executable Region program;
- a Pixel pointwise/stencil operation can be lifted into an admitted
  whole-graph Region node;
- a reduction-coupled Pixel Kernel needs a successful regionization before it
  can supply efficient cross-output sharing;
- an unsupported or unbounded footprint rejects the complete
  single-node lowering without affecting ordinary graph execution.

### Operation semantics remain Pixel; selected Kernels are regionized

Keep every operation-specific `Compute(S).pixel`, shape rule, operand wiring,
and Graph IR form unchanged.  Pixel remains the operation's semantic source and
the universal Direct/Symbolic implementation.  Regionization is a Kernel
optimization:

```text
operation Compute.pixel -> Pixel-form Region Kernel -> ordinary Pixel schedule
                                                   \-> regionize selected Kernel
                                                       -> Region schedule
```

The converter reuses the source Pixel expression rather than restating the
operation formula.  For exact scalar sharing it:

1. selects a legal output partition;
2. identifies source-expression subtrees invariant over that region;
3. hoists those existing subtrees into ordered Region locals;
4. replaces their occurrences in the emitter with local references;
5. proves that substituting the locals reconstructs the source Pixel
   expression and that each local is invariant over its declared region.

RMSNorm, LayerNorm, and Softmax conversion may use operation identity,
parameters, and shapes to select candidate region axes.  The arithmetic still
comes from `Compute.pixel`.  Perform conversion while this provenance is
available, or retain bounded origin metadata through Pixel-form Kernel
construction; do not recover operation identity by brittle pretty-printed AST
matching.

A later generic dependence/loop-invariance analysis can discover the same
regions without operation-specific selection.  It is not required for the
first converters.  An algorithm-changing conversion, such as Welford variance
or online Softmax, is a different `Equivalent` optimization and needs its own
numerical contract; it is not scalar-region hoisting.

Unselected operations remain mechanically wrapped singleton Region programs
and use the specialized Pixel loop.  A failed conversion returns the original
Pixel-form Kernel unchanged.  Region language implementation and operation
selection are therefore separate bounded tasks.

## 5. Phase 1 operation and language scope

Phase 1 covers exactly three existing single-output Native operations:

| Native operation | Region partition | Scalar locals | Region work for normalized extent `K` |
|---|---|---|---:|
| `Rms_norm` | `Whole` on `params.dims`; `Singleton` elsewhere | `sumsq`, `mean_square`, `inv` | `K` reduction terms + `K` emissions |
| `Layer_norm` | `Whole` on `params.dims`; `Singleton` elsewhere | `sum`, `mean`, `variance_sum`, `variance`, `inv` | `2K` reduction terms + `K` emissions |
| `Softmax` | `Whole` on `params.axis`; `Singleton` elsewhere | `max`, `denominator` | `2K` reduction terms + `K` emissions |

Derived arithmetic bindings such as `mean_square`, `variance`, and `inv` may
be folded by lowering, but they are scalar Region values in the source
program.  Phase 1 permits any finite number of scalar locals with dependencies
on earlier locals.

**Scalar-only describes local shape.**  It does not mean that the operation
has a scalar output, that a region contains one output, or that preparation has
one pass.  Phase 1 regions emit complete tensor slices, and their scalar locals
are produced by ordered traversals over tensor reduction domains.  It excludes
only locals shaped as vectors or higher-rank tensors.

The closed Phase 1 language subset is:

- one existing graph operation and one tensor output;
- `Singleton` or `Whole` on each output axis; no `Block` partition;
- statically shaped, possibly multi-axis nested reduction domains;
- ordered `sum` and `max` reductions;
- scalar `let` bindings that may reference earlier scalar bindings;
- the existing scalar/index/boolean primitives and tensor loads;
- one pure emitter evaluated for every output in the region;
- strict numerical mode with current reduction and arithmetic order.

Phase 1 explicitly excludes:

- `Group_norm`, whose channel groups need a `Block`-like partition;
- `Batch_norm_no_stats`, whose useful Region form also has live statistic
  outputs with different shapes;
- running-stat `Batch_norm`, which has no repeated data-derived normalization
  reduction to remove in inference;
- `Mean`, `Amax`, and other tensor-reducing output operations, which already
  perform one reduction per produced output and do not have the same
  cross-output duplication;
- `log_softmax`, SDPA's safe softmax, and the complete SDPA operation;
- shaped locals, cached exponent rows, custom/online folds, reduction
  splitting, parallel/tree reductions, tiling, and graph fusion.

The Phase 1 implementation order is RMSNorm, LayerNorm, then plain Softmax.
Completion requires all three; RMSNorm alone proves only one reduction and one
dependent scalar finalization.

### RMSNorm

Region axes are `params.dims`.  Compute one ordered sum of squares over the
same axes, derive `inv`, and emit every value in the region.  This preserves
the existing formula and reduction order while changing only the lifetime of
the result.

No local tensor is required.  Weight indexing remains the current normalized
coordinate projection.

#### Effect on the existing RMSNorm formulation

The current `Norm.RmsNorm.Compute(S).pixel` computes, for every output
coordinate `out`:

```text
sumsq(out) = sum over params.dims of x[out with normalized coordinates replaced]
                                      ^ 2
inv(out)   = 1 / sqrt(sumsq(out) / normalized_count + eps)
y[out]     = x[out] * inv(out) * weight[normalized coordinates of out]
```

`sumsq(out)` and `inv(out)` depend only on the non-normalized coordinates.
They are identical for every `out` in the same normalized slice, but Pixel
recomputes them independently.

The Region form makes that invariance structural:

```text
region whole=params.dims, singleton=other axes {
  let sumsq = ordered_reduce_sum params.dims {
    let v = load x at (region key + reduction coordinates)
    v * v
  }
  let inv = 1 / sqrt(sumsq / normalized_count + eps)

  emit out over params.dims {
    load x at out * inv * load weight at project(out, params.dims)
  }
}
```

For `dims=[C]`, a region is one complete C fiber at fixed `N,T,D,H,W`.  For
`dims=[W,C]`, it is one complete W-by-C slice at fixed `N,T,D,H`.  The design
therefore preserves the existing multi-axis RMSNorm contract instead of
special-casing one axis.

The following RMSNorm components do not change:

- `params`, JSON, graph operands, and `Graph_ir.Rms_norm`;
- `output_shape`, including the checked normalized-count bound;
- `check_weight` and the normalized-only affine layout;
- importer behavior and absent-weight semantics;
- the scalar formula, epsilon, and weight projection.

The changed components are localized:

- the RMSNorm Region converter selects `params.dims` as `Whole` axes and hoists
  the existing `sumsq`/`inv` expression subtrees into locals;
- Pixel-form Kernel construction retains enough bounded operation
  provenance—or invokes conversion while building the Kernel—to supply the
  parameters and operand/shape correspondence;
- the region schedule enumerates non-normalized region keys before normalized
  output coordinates;
- efficient Kernel execution selects the converted Region instead of invoking
  Pixel for every output;
- `Norm.RmsNorm.Compute.pixel` remains unchanged as fallback and semantic
  oracle.

For normalized size `K` and `R` independent slices, Pixel performs `R*K*K`
square/reduction terms.  Region performs `R*K` reduction terms and `R*K`
emissions.  Output storage remains `R*K`; only redundant computation is
removed.

This conversion is bitwise `Identical` when the Region interpreter:

- nests reduction axes in the same `params.dims` order;
- visits each extent left-to-right as current `S.sum` does;
- evaluates `sumsq / count`, epsilon addition, square root, division, and
  output multiplications in the current order;
- keeps `sumsq` and `inv` in the existing working value domain and rounds only
  at the final RMSNorm tensor store.

Sharing a deterministically recomputed `inv` does not itself change rounding.
Storing `inv` in an f32 local would change the contract and is not allowed in
strict mode.  Parallel/tree reductions and reassociated arithmetic likewise
require the relaxed numerical mode rather than being consequences of Region
execution.

### LayerNorm

Use two phases to preserve the current numerically stable definition:

1. ordered sum and division to obtain `mean`;
2. ordered sum of `(x - mean)^2` and division to obtain biased `variance`;
3. emit normalized and affine-transformed outputs.

Do not rewrite variance as `E[x^2] - E[x]^2` merely to combine reductions; it
changes stability and existing results.  Welford is a valuable later algorithm
variant but is numerically `Equivalent`, not `Identical`.

### Softmax

The region is the full selected axis at fixed coordinates on every other
axis:

1. ordered max reduction;
2. ordered exponential sum using that max;
3. emit the normalized exponential.

Only two shared locals are required:

```text
region whole=params.axis, singleton=other axes {
  let m = ordered_max k in axis { x[k] }
  let z = ordered_sum k in axis { exp(x[k] - m) }

  emit k in axis {
    exp(x[k] - m) / z
  }
}
```

`m` and `z` are scalars for the fixed non-axis coordinates.  Phase 1 does not
cache the `K` exponentials in a local vector: it evaluates `exp(x[k] - m)` once
during the denominator pass and once during emission.  This gives `K` max
terms, `K` denominator terms, and `K` emissions—`Theta(K)` work and scalar
storage.  A shaped exponential/probability cache can reduce transcendental
recomputation later, but it is not required to remove the current
`Theta(K^2)` repeated reductions.

Plain Softmax retains its current all-`-inf` behavior.  SDPA's safe-softmax
zero-row rule remains a separate semantic choice in the attention program.

### Multi-output operations

The emitter should permit several result values/stores because the existing
Kernel already supports multiple outputs and pooling-with-indices establishes
the need.  All stores must use the same partition/ownership proof or name their
own output region.

## 6. SDPA requires more than scalar sharing

For query length `Q`, key length `K`, query/key feature size `E`, and value
feature size `V`, the current per-pixel implementation performs three score
passes for every `(query, value-feature)` output.  Its leading contraction work
is therefore `Theta(Q * V * K * E)`.  With `V = E`, that is
`Theta(Q * K * E^2)`, even though ordinary attention work is
`Theta(Q * K * (E + V))`.

Scalar region locals can compute a row's `m` and `z` once, removing their
repetition across `V`, but they do not solve the numerator: computing each
output feature independently still recomputes every score.  Efficient SDPA
needs one of these region-local strategies:

1. cache a `K`-element score/probability row, then multiply it by all value
   columns;
2. keep a `V`-element output accumulator while scanning keys;
3. use blocked online softmax, with state `(row_max, row_sum, output[V])`, as
   in flash-style attention.

The third avoids materializing the `Q x K` score tensor and is the desired
long-term form.  It proves that shaped locals and multi-result folds are part
of the fundamental design, not optional embellishments.

A first attention region can be one query row, varying over output feature
`V`.  A later physical schedule may combine several query rows and key blocks
to improve Q/K/V locality.  The semantic online fold is over the ordered key
domain; query/key block sizes and placement of the accumulator in registers,
stack memory, or shared memory are schedule decisions.

There is an important numerical boundary.  Online softmax rescaling changes
the grouping and rounding of max, sum, and output accumulation relative to the
current three-pass definition.  Likewise, moving SDPA's split scale outside a
dot changes f32 results.  The IR must retain:

- ordered reduction/fold update rules;
- explicit `round_f32` stage boundaries;
- the existing split scaling on both Q and K operands where that is the op's
  contract;
- an `Identical` versus `Equivalent` claim for algorithm substitutions.

The current SDPA oracle already uses tolerance against ATen flash attention,
but that does not make arbitrary reassociation silently legal.  Add a targeted
numerical verification policy for the online algorithm.

## 7. Relation to dot/contraction intrinsics

The dot-product design remains useful for Bmm and individual attention scores.
Its role in the complete design is a scalar/reduction leaf optimization:

```text
region sharing and streaming   reduce number of reductions/contractions
contraction intrinsic          make each remaining contraction efficient
schedule tiling/vectorization  improve locality and machine utilization
```

For normalization and Softmax, region sharing should come first because it
changes `Theta(K^2)` work to `Theta(K)`.  A self-dot for RMSNorm only makes
each of the `K` repeated reductions faster and leaves `Theta(K^2)` work.

For attention, a plain section dot is not automatically exact because the
current score reduces `(q[e] * sf) * (k[e] * sf)`.  Either:

- recognize this complete scaled contraction in lowering and generate a tight
  loop preserving the per-term operation order; or
- introduce a contraction intrinsic whose element transforms are explicit.

Do not pull `sf * sf` outside the reduction under an `Identical` claim.

Longer term, a general `map_reduce` or contraction node in Kernel IR is more
composable than adding many narrowly named `SEMANTICS` methods.  A specialized
Direct `dot` remains a reasonable short-term benchmark experiment.

## 7a. Attention and the restricted single-node graph domain

Attention is usefully viewed as an already fused computational graph:

```text
Q, K -> scaled QK^T -> add mask -> softmax -> multiply by V -> output
```

The unmaterialized score/probability values, their region-local lifetimes, and
the schedule coordinating their producers and consumers are the same core
problems that occur when representing a multi-node graph as one computation.
Therefore decisions made for attention extend to the deliberately restricted
fusion domain **if** they are expressed using generic region concepts:

- logical values connected as a DAG;
- explicit scalar or shaped locals;
- producer-to-consumer index maps and derived footprints;
- ordered reductions/folds;
- one or more region outputs;
- schedule-selected placement (recompute, register, or bounded node-local
  buffer).

In that model, a single complex operation and an admitted whole graph differ at
the front end, not in the scheduled representation:

```text
complex op definition ----\
                           -> region Kernel DAG -> placement/schedule -> Loop IR
admitted whole graph -----/
```

This design intentionally avoids general graph partitioning.  The complete
candidate either lowers to one region node or is not fused.  Internal fan-out
is an ordinary logical-value DAG; only the node's declared inputs and outputs
cross its boundary.  There is no choice of where to cut or which subset to
fuse.  Attention starts with a known internal DAG and therefore remains a good
expressiveness test for this smaller domain.

Initial single-node admission should require:

- pure, deterministic, acyclic tensor computation;
- all externally visible results owned by the region node (initially one
  result, or several results with a compatible region grid);
- statically known shapes and producer-to-consumer index maps;
- every internal value expressible as a scalar expression, a reduction local,
  a bounded region-local tensor, or deliberate recomputation;
- no required interior global materialization, stateful update, scatter, or
  data-dependent output ownership;
- a statically bounded input footprint and local storage requirement for every
  output region;
- all operations and numerical forms supported by the region/Loop IR.

These rules admit straight-line pointwise graphs, normalization/Softmax tails,
bounded stencil chains, and eventually attention.  A graph with incompatible
global communication patterns simply remains as several ordinary nodes; the
first design does not attempt a partial fusion.

There is a critical rounding distinction:

- locals inside one graph operation have only the conversions stated by that
  operation's semantics;
- an internal edge originating as an original graph node carries the
  producer's `round_f32` result conversion in strict mode even when no tensor
  is stored.

For example, an optimized attention operation need not behave bitwise like a
materialized `matmul -> softmax -> matmul` graph.  Conversely, lowering those
three existing nodes to one region node under an `Identical` claim must
preserve both eliminated stage boundaries as explicit rounds.  Region IR
should carry result conversions on logical values so the scheduler cannot
confuse these cases.

The compilation request selects one of two modes:

```ocaml
type numerical_mode =
  | Preserve_stage_rounding
  | Relax_internal_f32
```

`Preserve_stage_rounding` retains every original logical node's `round_f32`
and can claim `Identical` when loop order and primitives are also preserved.
`Relax_internal_f32` permits an internal graph-edge round to be removed or
deferred.  Input decoding, explicit dtype conversions, quantization
boundaries, and final output conversion are never removed.  The resulting
single node carries an `Equivalent` claim and must be checked under an explicit
absolute/relative tolerance and NaN/infinity policy.

The first relaxed subset should exclude a removed round that can reach an
index, shape, gather/scatter address, or other data-dependent control decision.
Such a change is not well described as ordinary floating-point error.  It can
be widened later with operation-specific evidence.  Floating comparisons,
max/min, and selects also deserve targeted boundary tests even when allowed,
because a one-bit input difference may select a different arm.

Use attention as a late expressiveness and scheduling test; its specialized
algorithm does not define the first API.  Normalization and Softmax establish
the reusable scalar-region core with much less machinery.  Reduction-coupled
graph fusion and attention then add shaped local state on the same foundation.

## 8. Operation tiling and fusion

Source-level sharing regions and physical schedule tiles solve different
problems and compose through Region-to-Region transformation.  Required sharing
comes from the operation; chosen locality blocks are reified in the selected
Kernel Region before Loop lowering.

### Tiling one operation

A convolution's output region may be a single point semantically; the schedule
can compute a block of output spatial/channel coordinates to reuse input and
weight data.  Conversely, a normalization has a full normalized slice as its
semantic sharing region; the schedule may block the outer region grid, but it
cannot independently split the reduction axis without introducing partial
states and a combine/finalize phase.

Thus a schedule needs to distinguish:

- partitioning independent output regions;
- blocking emission within a region;
- splitting a reduction into partial folds;
- placement and lifetime of locals across those loops.

### Conv tiling is a Region-to-Region transformation

Conv2D begins as a Pixel-form Region program.  Its emitter contains the current
ordered nested contraction for one output coordinate.  A tiling pass selects an
output block and rewrites that source program into a non-degenerate Region
program:

```text
Conv Pixel Region
  partition: Singleton on H, W, Cout
  emitter: ordered contraction for one output

        -- tile_conv(Bh, Bw, Bco, cache choices) -->

Tiled Conv Region
  partition: Block Bh on H, Block Bw on W, Block Bco on Cout
  locals: input footprint tile, optional weight tile, output accumulators
  phases: populate locals, ordered contraction, bias/finalize, emit block
```

Both programs use the same load/index/arithmetic/reduction language.  The
second program makes reuse and lifetime explicit; it is not a new Conv graph
operation.  The tiling transformation carries an `Identical` or `Equivalent`
claim and is checked against the source Pixel-form program.  The selected
Kernel stores the transformed Region program, so the implemented computation
and its local reuse are inspectable rather than implicit backend behavior.

For an output-height block `[oh0, oh1)`, the required input-height coordinates
are the union of the current Conv windows:

```text
ih = oh * stride_h - pad_before_h + kh * dilation_h
```

for every `oh` in the block and every valid `kh`.  Ignoring clipping, the
bounding extent is:

```text
(Bh - 1) * stride_h + (Kh - 1) * dilation_h + 1
```

and likewise for width.  Bounds are still clipped according to
`Window_axis`; strict lowering must not replace clipping with extra padded-zero
multiply terms because that can change NaN behavior and reduction grouping.
The input-channel footprint is the complete channel range of the groups touched
by the output-channel block.  The weight footprint covers that output-channel
block, the same input channels per group, and the complete kernel window.

The principal reuse then becomes explicit:

- neighboring H/W outputs reuse overlapping values in the input tile;
- different output channels in the same group reuse the input tile;
- all spatial outputs in the block reuse the selected weight tile;
- an output-accumulator tile survives for the complete contraction.

The local definitions carry element format as well as shape.  In strict mode,
an input or weight cache stores the decoded working-domain value without an
additional f32 round, and accumulators use the current working domain.  A
cache represented as f32 when the repeated load previously produced a wider
working value is a numerical change.  Final Conv output retains the existing
store round.

### Ordered tiled Conv computation

Local reuse does not authorize reduction reassociation.  The current Conv2D
Pixel expression nests reductions in this order:

```text
sum local_input_channel {
  sum kernel_h {
    sum kernel_w {
      input * weight
    }
  }
}
```

and adds bias after the contraction.  A strict tiled Region preserves that
nesting independently for every output.  It may interleave work for several
outputs, but the sequence and grouping of terms observed by each accumulator
remain unchanged.  Conceptually:

```text
region output_tile {
  let input_tile  = cache required_input_footprint
  let weight_tile = cache required_weight_footprint   // optional

  let contracted[out in output_tile] =
    ordered_sum local_ic {
      ordered_sum kh(valid for out) {
        ordered_sum kw(valid for out) {
          input_tile[input_index(out, local_ic, kh, kw)]
          * weight_tile[weight_index(out.C, local_ic, kh, kw)]
        }
      }
    }

  emit out in output_tile {
    contracted[out] + bias[out.C]
  }
}
```

`cache` is a pure shaped local definition and `ordered_sum` is a structured
fold; neither is arbitrary mutable code.  Lowering may realize `contracted` as
an array of accumulators and interchange an output-tile loop with reduction
loops only when it proves that each output observes the required nested update
order.  Flattening the three reductions, using partial/tree sums, changing the
accumulator precision, or introducing fused multiply-add belongs to relaxed
mode.

Cache-population order is normally not numerically observable because it only
copies/decodes pure input values.  Phase order is observable: caches must be
defined before contraction, the complete contraction before bias/finalization,
and emission after its dependencies.  Region therefore specifies the ordered
computation and lifetimes; Loop IR chooses concrete loop syntax,
vectorization, parallel output tiles, and physical placement.

Tile selection is constrained by checked local-resource estimates:

```text
input_tile_bytes + optional_weight_tile_bytes + accumulator_bytes
  <= target_local_capacity
```

The estimator also accounts for group boundaries, tail blocks, clipped
windows, and aggregate-size overflow.  If no profitable tile is admitted, the
Kernel keeps the Pixel-form Region program and its specialized Pixel loop, so
adding Conv tiling cannot degrade the established untiled path.

### Expressing a whole graph as one region node

For pointwise chains, producer values can remain scalar/virtual.  For stencils,
a consumer tile requests a producer tile plus a halo.  For reduction-coupled
consumers, the producer footprint may cover a whole reduction region.  The
entire admitted graph is nevertheless one compilation unit and produces no
interior global tensor.

`conv + add + rmsnorm` is the revealing case.  To avoid a full intermediate
tensor while also avoiding recomputation, a schedule for one spatial output
region must:

1. compute the needed `conv + add` channel values;
2. retain that channel vector in a region-local buffer while accumulating
   sum-of-squares;
3. normalize the retained values and store the final output.

With only a scalar local it must either recompute convolution during emission
or fail this single-node admission.  A local tensor therefore matters for the
stated fusion goal even before SDPA.  Profitability chooses between local
buffering and recomputation based on footprint and local capacity.  If neither
is acceptable, the whole-graph fusion attempt is rejected rather than split
into smaller fused clusters or given an interior graph-visible tensor.

The existing Kernel logical-value DAG and explicit `round_f32` boundaries are
good foundations.  Extend placement with a bounded region-local buffer.  The
value remains a logical Kernel value; the schedule decides whether its physical
realization is recomputed, a scalar/register vector, or a node-local tile.  A
graph-visible global buffer is outside this restricted fusion form.

Single-node admission needs dependency/footprint analysis.  Given a requested
final output region, recursively derive the coordinates required from every
internal producer.  Classify each step as pointwise, bounded halo,
reduction-coupled, or data-dependent.  Reject the complete lowering when any
internal footprint is not representable or exceeds resource limits.  Do not
require operators to duplicate this footprint as an unchecked annotation;
derive it from load indices where possible.

### Stencil-to-stencil acceptance cases

Convolution/depthwise-convolution and convolution/max-pool pairs are good
single-node tiling cases.  Unlike `conv + add`, they require a consumer tile to
request a bounded **region** of producer outputs, usually with a spatial halo.
Unlike attention, all footprints are static affine windows, so they test the
tiling machinery without shaped online state or data-dependent addresses.

| Whole graph | What it tests | Relative difficulty |
|---|---|---|
| `depthwise 3x3 -> pointwise 1x1 conv` | A local depthwise tile consumed by a channel contraction; the second op adds no spatial halo but needs all intermediate channels | Best first convolution pair |
| `conv2d -> depthwise conv2d` | Producer spatial halo, channel-local consumption, padding/stride composition, and avoidance of expensive producer recomputation | Best first true stencil-to-stencil case |
| `conv2d -> maxpool2d` | Producer halo feeding a nonlinear ordered max reduction, including max/NaN/tie semantics | Good after value-only max-pool regions work |
| `maxpool2d -> conv2d` | A cached pool-output tile, overlapping source windows, then spatial/channel contraction | Harder composed-footprint case |
| `depthwise conv2d -> general conv2d` | The consumer normally needs every intermediate channel and, unless pointwise, a spatial halo | Larger local tile; later case |
| `general conv2d -> general conv2d` | Full spatial and channel coupling | Useful final stress case, not an initial target |

For `conv2d -> depthwise conv2d`, a final output tile determines an
intermediate spatial tile expanded by the depthwise kernel's effective halo.
The depthwise consumer does not generally mix intermediate channels, so a
channel-tiled final output requests only the corresponding producer channels
(with the depth-multiplier mapping where supported).  The first convolution
computes that expanded tile once into node-local storage; the depthwise
consumer then reuses overlapping values.  Start with multiplier one, stride
one, dilation one, and simple padding before exercising the complete window
arithmetic.

The reverse `depthwise -> conv2d` order is also valid, but a general second
convolution mixes channels and therefore usually requires every intermediate
depthwise channel.  The important easy special case is
`depthwise 3x3 -> pointwise 1x1`: it is a common separable-convolution shape,
needs no additional spatial halo at the second operation, and exercises local
tile production followed by a channel reduction.

For `conv2d -> maxpool2d`, each pool output requests a fixed window of rounded
convolution outputs.  Adjacent windows may overlap, making local caching
profitable.  In strict mode, apply the convolution node's `round_f32` **before**
the max comparison.  Removing that round can change which element wins, not
only the returned value, so this edge should initially be excluded from
`Relax_internal_f32`.  Start with value-only max-pool.  Adding pool indices
also has to preserve first-index tie behavior and the index coordinate system.

For `maxpool2d -> conv2d`, the final convolution tile first determines a tile
of pool outputs; each of those expands to a source-input pooling window.  This
tests recursive footprint composition and whether overlapping pooling windows
should be cached or recomputed.  It is a good later test because padding,
stride, and two nested window maps make the required source footprint less
obvious, even though it remains static and bounded.

These examples should remain one-node/all-or-nothing admissions.  They may use
node-local tiles, but they do not introduce a graph-visible intermediate.  If
the expanded producer tile exceeds the local-resource limit and recomputation
is not profitable, leave the original two-node graph unchanged.

## 9. Native and Native4D

The region construct should be dimension-parametric at the Kernel/Loop level.
Native contributes a six-axis domain; Native4D contributes its four-axis
domain or maps its axes into the common representation.  Both should lower to
the same local/reduction/emission concepts and schedule machinery.

This does not make an unsupported Native operation legal in Native4D.  An
operation whose semantic region or reduction names an axis absent from the
four-axis dialect still needs legalization or a typed rejection.  In
particular, efficient SDPA still has the existing Native4D domain problem; a
region API does not invent the missing batch/attention semantics.

Avoid copying a second handwritten region algorithm into `Eval_op4`.  As with
current Native4D delegation to Native `Compute(S)`, reuse a common region
definition when the axis mapping is sound.  Longer term, both dialects should
produce common Kernel IR before schedule selection.

## 10. Validation obligations

A region program is accepted only if the system can establish:

- region coverage and disjointness for every output;
- local definitions are acyclic, scoped, and used only within their region;
- every reduction/fold bound is valid and its iteration order is explicit;
- every input/local load is in bounds over all region and reduction indices;
- each output is stored exactly once;
- local tensor sizes and aggregate work are overflow-checked before
  allocation/lowering;
- a schedule preserves required local lifetimes and reduction order;
- strict mode retains every original graph-stage `round_f32` even when its
  storage is eliminated;
- relaxed mode retains input/output and explicit conversion boundaries,
  records every elided internal round, and satisfies its declared numerical
  comparison policy;
- backend resource limits admit the chosen tile and local placement.

The reference interpreter should execute region programs directly before an
optimizing Loop lowering is trusted.  Differential tests should compare:

1. existing per-pixel Direct;
2. existing per-pixel Symbolic grounding;
3. the region interpreter;
4. naive Loop IR lowering;
5. optimized schedules.

For exact region hoisting of the current normalization/Softmax formulas, these
should be bitwise equal.  Parallel reductions, Welford, contraction
reassociation, and online attention use explicit tolerance/claim tests.

Performance tests must measure operation counts or representative runtime, not
only expression size.  Expression AST size is constant in reduction extent and
therefore does not expose repeated dynamic work.

## 11. Incremental implementation plan

### Phase 0: measure the current defect

- Add counters/benchmarks for input loads, reduction iterations, allocation,
  and elapsed time on RMSNorm, LayerNorm, and Softmax at several slice lengths.
- Measure Direct and symbolic grounding separately; their dynamic work can
  differ even when they share `Compute`.
- Keep the dot benchmark separate so invocation reduction and inner-loop
  optimization are not conflated.

### Foundation task: implement the Region language

The reviewable gate sequence, concrete APIs, file map, tests, limits, and
no-regression checks are specified in
[region-foundation-plan.md](region-foundation-plan.md).

- Add the Region AST, private constructors, scope/type validation, limits, and
  deterministic printer.
- Add axis-aligned `Singleton`/`Whole` partitions, ordered scalar `let`, nested
  `sum`, `max`, and one pure tensor-output emitter.
- Add `Region_program.pixel` and validate that every existing Pixel-form Kernel
  embeds mechanically without AST copying.
- Make validated `Region_program.t` the Kernel computation field and adapt
  existing `Stage_program` values through the Pixel constructor.
- Implement the Region reference interpreter and specialized singleton Pixel
  execution.
- Implement Region-local substitution/specialization and the checks needed to
  prove region invariance and reconstruct a Pixel expression.
- Test synthetic singleton, whole-axis, multi-axis, dependent-local, invalid
  scope, invalid partition, bounds, and limit cases.
- Benchmark trivial Pixel-form kernels to enforce the no-regression invariant.
- Do not modify RMSNorm, LayerNorm, Softmax, or any other operation module in
  this task.

This task is complete when Region is a usable computation language for both
degenerate Pixel programs and synthetic scalar-local programs.  It does not
claim an operation speedup.

### Phase 1: regionize selected scalar-local Kernels

- Phase 1a: convert RMSNorm Kernels by selecting `params.dims` and hoisting the
  existing sum-of-squares/inverse expression; prove one shared reduction plus
  dependent scalar finalization.
- Phase 1b: convert LayerNorm Kernels and prove a later reduction can reference
  an earlier hoisted scalar local (`mean`).
- Phase 1c: convert plain Softmax Kernels and prove mixed max/sum phases, scalar
  reuse in the emitter, arbitrary selected axis, and existing all-`-inf`
  behavior.
- Leave all three `Compute(S).pixel` implementations unchanged as semantic
  sources, fallbacks, and differential oracles.
- Return the original Pixel-form Kernel unchanged whenever selection,
  invariance proof, reconstruction, validation, or limits fail.
- Measure reduction iterations and loads to establish `Theta(K)` Region work
  for all three operations.

Phase 1 is complete only when all three operations pass Direct Region versus
Pixel Direct/Symbolic differential tests over single- and multi-axis
normalization, optional affine operands, several epsilons, every Softmax axis,
and all-`-inf` Softmax input.

### Phase 2: Loop IR integration

- Lower region programs to naive structured Loop IR.
- Add local liveness, loop-invariance, bounds, and cost analysis.
- Make the dense current schedule a valid degenerate policy.
- Route Native and representable Native4D operations through the common path.

### Phase 3: first whole-graph single-node lowering

- Form one region Kernel DAG from an entire small input graph instead of from
  one operation definition.  Reject it as a unit if any node is unsupported.
- Start with a graph such as `pointwise -> RMSNorm` or
  `pointwise -> Softmax`; the producer can be recomputed in the reduction and
  emitter, making the first version legal without a local tensor.
- Support `Preserve_stage_rounding` first and retain the producer's
  `round_f32` as an internal logical conversion.
- Add a whole-graph cost decision between the single-node lowering and leaving
  the original graph unfused.
- Reuse the same region interpreter and Loop lowering as the single-op cases;
  only whole-graph construction should be new.

This is the smallest proof that the region design is not an attention-only or
operator-only facility.  It also exposes graph-boundary rounding and fan-out
before local buffers make debugging more difficult.

### Phase 3a: optional relaxed rounding

- Add `Relax_internal_f32` as a distinct compilation mode, never as an
  implicit optimization in strict mode.
- Initially elide rounds only along floating value paths that cannot affect
  indices, shapes, or data-dependent memory access.
- Always round declared graph outputs and retain explicit dtype/quantization
  conversions.
- Report which logical-edge rounds were removed and validate the complete
  output with declared `atol`, `rtol`, NaN, and infinity rules.
- Classify the result as `Equivalent`; do not infer `Identical` from a passing
  finite test set.

### Phase 4: shaped locals and region-local placement

- Admit bounded local vectors/tensors and multi-result folds.
- Add placement choices: recompute, scalar/register, and bounded node-local
  buffer.
- Exercise `depthwise 3x3 -> pointwise 1x1 conv` as the first local convolution
  tile and `conv2d -> depthwise conv2d` as the first spatial-halo case.
- Add `conv2d -> value-only maxpool2d`, preserving the strict round before max;
  then test `maxpool2d -> conv2d` footprint composition.
- Exercise `conv + add + rmsnorm` without a full intermediate tensor after the
  static stencil cases.
- Add schedule blocking, halo-aware producer footprint derivation, and local
  resource rejection.

At this point an admitted whole graph and a complex single operation use the
same region-local storage model.  `conv + add + rmsnorm` tests a local producer
tile; attention tests a local reduction accumulator or score tile.

### Phase 5: efficient SDPA

- First cache/share row scores or use a vector accumulator to remove the
  `V`-fold score recomputation.
- Then add blocked online-softmax state and query/key block schedules.
- Integrate a contraction intrinsic or pattern-lowered tight dot loop.
- Validate numerical behavior against both the existing scalar definition and
  the selected ATen oracle under an explicit equivalence policy.

### Phase 6: schedule search and advanced reduction splitting

- Add target resource models, vectorization, parallel region-grid execution,
  partial reductions, and combine/finalize phases.
- Keep schedule choice outside graph semantics and make every chosen schedule
  inspectable in Loop IR.

## 12. Adopted design direction

Use **region computation with explicit locals and phases**.  Scalar locals
cover the initial RMSNorm, LayerNorm, and Softmax cases; shaped locals cover
efficient attention, tiled convolution, and reduction-coupled fusion.

The first implementation is deliberately narrow: whole/singleton axis regions,
named scalar locals, ordered sum/max reductions, and pure emission.  It removes
the normalization and Softmax asymptotic defect while preserving the current
arithmetic.  Locals have a shape in the IR, and region semantics remain separate
from schedule tiles.  These choices support later online attention,
convolution tiling, and restricted whole-graph single-node lowering without
another language redesign.
