# Region-native execution and locality implementation guide

## Purpose and status

The Foundation implementation is complete: every logical `Kernel.Value` stores
one `Region_program.t`; `Region_program.pixel` embeds the existing scalar
program; `Region_execution.lower` classifies the degenerate Pixel form; and
`Region_eval` is the correct reference executor for a non-degenerate program.
The completed baseline comparison and Pixel no-regression evidence are recorded
in [`region-foundation-plan-todo.md`](region-foundation-plan-todo.md).  This
guide starts the next implementation boundary; it does not claim that a
Region-native operation or production Region lowerer has landed.

This document defines the next implementation boundary.  Selected operations
may construct non-degenerate Region programs and execute through a dedicated
Region lowering path.  It also separates that work from ordinary tiling for
data locality, so that making more kernels tileable does not create a second
semantic language or put scheduling policy in operation modules.

## The two kinds of future participation

There are two distinct reasons a kernel can use the Region execution stack.

| Kind | Example | Program representation | Who chooses tiles |
|---|---|---|---|
| Region-native sharing | RMSNorm, LayerNorm, Softmax, attention | Non-degenerate `Region_program`: explicit partition, ordered locals, and emitter | Region lowering/schedule |
| Region-scheduled locality | Pointwise chains, Conv, bounded stencils | Pixel form plus an access-footprint description; optionally a derived block-local Region plan | Schedule, subject to resource checks |

The first changes the semantic lifetime of a computed value: a softmax maximum
or denominator is calculated once per logical slice and reused by every output
in that slice.  It therefore needs a Region local and an ordered Region phase.

The second normally changes only loop blocking, storage placement, and reuse
of loaded data.  It must preserve the same values, reduction order, and
observable result as its Pixel program, so it can remain Pixel-form and receive
an optimized tiled schedule without changing the stored Kernel program.  If a
selected schedule needs cache/accumulator lifetime to be inspectable or shared
with another Region phase, it may instead derive a non-degenerate, block-local
Region program and prove the same relationship to the Pixel source.

Do not use a `Block` `Region_partition` merely to record an arbitrary physical
tile.  Introduce it only when the block has Region-owned locals or phases whose
lifetime must be represented and checked.  Otherwise tile sizes, vector widths,
cache buffers, workgroups, and loop order belong to execution IR and can vary
per backend and machine.

## Operation-facing contract

Keep the present `Compute(S).pixel` implementation as the operation's
authoritative scalar semantics.  Do not add a mandatory second `Compute(S)`
method and do not put regions into `Semantics.SEMANTICS`: that interface is a
scalar value/index domain and cannot state output ownership, local lifetime,
or stores.

Instead, an operation family that has a proven sharing opportunity may expose
an optional *declarative regionizer* at a boundary where its typed operation,
validated parameters, operand signatures, and output shape are available.  In
outline:

```ocaml
module type REGIONIZER = sig
  type context
  type reject

  val try_regionize :
    context ->
    pixel:Expr.Value.t ->
    (Region_program.t, reject) result
end
```

This is intentionally not a second executable algorithm.  It constructs
shared `Expr`/`Region_program` syntax and declares the semantic partition;
the common Region lowering owns loops, buffers, and backend execution.  The
converter must prove its result reconstructs the supplied Pixel expression
with `Region_program.reconstructs`.  A rejection retains that exact Pixel
program and uses its current execution path.

The regionizer must run before operation provenance is discarded.  The current
Foundation Kernel stores only the Region computation, which is correct: it
must not regain an operation tag merely for an optimization.  A future pass
can either run at the `Graph_ir`/`Eval_op` boundary, or carry a private typed
regionization candidate alongside symbolic construction until it is accepted.
After acceptance, only `Region_program.t` crosses the Kernel boundary.

## Dedicated execution path

`Region_eval` remains the reference oracle; it is not the production design
for a Region-native kernel.  Add a lowering stage that takes a validated
non-degenerate `Region_program` and produces a backend-neutral execution plan
with:

1. a logical-region/key traversal;
2. once-per-key local allocation/evaluation in declaration order;
3. emitter loops for the outputs owned by that key;
4. a schedule-selected physical tiling, vectorization, placement, and loop
   order; and
5. stores applying `Kernel.Result_conversion` exactly once at the existing
   logical-value boundary.

The dispatch decision is once per logical value or admitted fused node:

```text
Pixel-form Region_program  -> existing zero-overhead Pixel loop
non-degenerate Region      -> Region lowerer -> dedicated Region executor
```

The Region executor should compile/specialize the local and emitter syntax
ahead of its output loops.  It must not call an opaque per-output closure,
allocate a local map per output, or recover sharing by repeatedly running
`Region_eval.value_at`.  The latter remains useful only for diagnostics,
differential tests, and Pixel specialization checks.

## Locality scheduling for Pixel-form kernels

Build a separate access-footprint analysis for Pixel-form computations.
For each output tile it should describe the required input, weight, and
intermediate windows and whether the operation has a bounded, analyzable
footprint.  The schedule may then choose a tile and cache/placement strategy.
It may reject unknown or unbounded footprints without changing ordinary graph
execution.

This analysis is complementary to regionization:

- Regionization identifies values safe and useful to compute once for a
  semantic output region.
- Footprint analysis identifies data reusable while executing a physical tile.
- A fused Region-native kernel can use both: semantic locals for, say, a
  normalization slice and a physical tile for its input/output locality.

Keep numerical restrictions explicit.  Reblocking an ordered reduction, using
a tree reduction, or changing an intermediate rounding boundary is not an
`Identical` optimization.  It needs a separately specified numerical mode and
test oracle.

## Effect on the Foundation layer

No Foundation representation change is required for this direction.  The
existing contracts are the enabling layer:

- one `Region_program.t` per Kernel value means Pixel and Region-native work
  already meet at one semantic boundary;
- `Region_program.pixel_expression` preserves the existing optimized Pixel
  lowering for unconverted operations;
- local scope checking, structural analysis, and `reconstructs` provide the
  correctness contract for an optional regionizer;
- `Region_execution.lower` provides the existing one-time classification;
- `Region_eval` supplies a concrete oracle for each accepted program.

Foundation must remain conservative.  Do not change `Compute(S).pixel`,
`Semantics.SEMANTICS`, Graph IR, or the Kernel value representation merely to
anticipate tiling.  Add concrete hooks only with the first converter/lowerer:
the regionizer boundary, a typed rejection/result type, backend-neutral Region
execution IR, and footprint metadata/analysis.  If shaped locals, block-owned
locals, multi-output emission, or a fused-node scope become necessary, extend
`Region_program` deliberately and preserve the current scalar-local programs
as a compatible subset.

The completed Foundation closeout is the prerequisite for a production handoff.
Any prototype must still preserve the existing Pixel path and must carry its
own Region-native validation; it is not additional Foundation evidence.

## First implementation slice

Use one operation with unambiguous sharing (RMSNorm or Softmax) before general
tiling or multi-node fusion:

1. Preserve its current Pixel expression as the source oracle.
2. Implement an optional regionizer that selects its normalized axes, declares
   `Whole` partitions, and hoists the repeated reductions into ordered locals.
3. Check the program and require `Region_program.reconstructs` before
   admission; retain Pixel on every rejection.
4. Lower the accepted program through the dedicated Region executor, initially
   with a simple deterministic region/key and emitter traversal.
5. Differential-test against Pixel/reference results, including NaNs, signed
   zero, f32 conversion, degenerate dimensions, and multiple independent
   region keys.
6. Benchmark once-per-slice work and ensure unconverted Pixel kernels retain
   the Foundation no-regression path.

Only then introduce footprint-driven tiling for a bounded pointwise/stencil
case.  This demonstrates that locality scheduling can benefit a Pixel-form
kernel without abusing the semantic Region partition.
