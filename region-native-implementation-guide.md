# Region-native execution and locality implementation guide

## Purpose and status

The Foundation implementation is complete: every logical `Kernel.Value` stores
one `Region_program.t`; `Region_program.pixel` embeds the existing scalar
program; `Region_execution.lower` classifies the degenerate Pixel form; and
`Region_eval` is the correct reference executor for a non-degenerate program.
The completed baseline comparison and Pixel no-regression evidence are recorded
in [`region-foundation-plan-todo.md`](region-foundation-plan-todo.md). The later
scalar-Region implementation and its production evidence are recorded in
[`region-compute-implementation-todo.md`](region-compute-implementation-todo.md).

The scalar-Region implementation described by the original version of this
guide has since landed. It used an optional regionizer and retained Pixel as the
operation authority. The adopted follow-up direction now treats computation
form as operation-specific: Pixel remains natural for many operations, while
RMSNorm, LayerNorm, and Softmax author Region programs used by Native and
Native4D Direct and Symbolic. See
[`region-compute-follow-up.md`](region-compute-follow-up.md) and the migration
plan in
[`region-operation-computation-implementation-plan.md`](region-operation-computation-implementation-plan.md).

The separation from ordinary tiling remains unchanged: making more kernels
tileable must not create a second semantic language or put scheduling policy in
operation modules.

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

Do not put regions into `Semantics.SEMANTICS`: that interface is a scalar
value/index domain and cannot state output ownership, local lifetime, or
stores. Instead, let each operation expose its natural authored form:

- Pixel-authored operations retain `Compute(S).pixel`;
- Region-authored operations expose an operation-owned declarative program;
- Native4D counterparts expose the same form after checked Axis4/parameter
  mapping and delegate to the shared numeric definition.

In outline, a Region-authored operation provides:

```ocaml
module type REGION_COMPUTATION = sig
  type context
  type reject

  val program : context -> (Region_program.t, reject) result
end
```

This constructs shared `Expr`/`Region_program` syntax and declares the semantic
partition; the common Region lowering owns loops, buffers, and backend
execution. A Region-authored program is authoritative and does not reconstruct
against a second handwritten Pixel definition. Its fresh scalar projection is
the compatibility meaning at one output coordinate. `reconstructs` remains the
right proof for a transformation of a Pixel-authored computation.

Program construction runs before operation provenance is discarded. Kernel
stores only the resulting Region computation and does not regain an operation
tag. The dialect dispatcher owns typed operand resolution and calls the
operation-owned computation; it owns no arithmetic.

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

For every Region-authored operation, deterministic trace tests must enumerate
the complete logical output domain by key, record ordered locals and owned
outputs, and independently prove exactly one visit per output. The trace prints
the stable program and open local/emitter expressions without expanding the
scalar projection. Cover Native and Native4D configurations separately; the
latter must make the Axis4 mapping and singleton T/D axes visible.

## Locality scheduling for Pixel-form kernels

Build a separate access-footprint analysis for Pixel-form computations.
For each output tile it should describe the required input, weight, and
intermediate windows and whether the operation has a bounded, analyzable
footprint.  The schedule may then choose a tile and cache/placement strategy.
It may reject unknown or unbounded footprints without changing ordinary graph
execution.

This analysis is complementary to Region computation:

- An operation-authored or transformed Region program identifies values safe
  and useful to compute once for a semantic output region.
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
- local scope checking and structural analysis validate every Region program,
  while `reconstructs` proves Pixel-to-Region transformations;
- `Region_execution.lower` provides the existing one-time classification;
- `Region_eval` supplies a concrete oracle for each accepted program.

Foundation remains conservative. Do not change `Semantics.SEMANTICS`, Graph IR,
or the Kernel value representation merely to anticipate tiling. The
post-Foundation ownership migration may replace `Compute(S).pixel` for an
intrinsically Region-authored operation, but Pixel-authored operations and the
Pixel fast path remain unchanged. If shaped locals, block-owned locals,
multi-output emission, or a fused-node scope become necessary, extend
`Region_program` deliberately and preserve the current scalar-local programs
as a compatible subset.

The completed Foundation closeout is the prerequisite for a production handoff.
Any prototype must still preserve the existing Pixel path and must carry its
own Region-native validation; it is not additional Foundation evidence.

## Completed first implementation slice

The landed stepping-stone implementation used the following sequence before
general tiling or multi-node fusion:

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

The next slice does not redo that work. It promotes the proven Region programs
to operation-authored computations, routes Native and Native4D Direct and
Symbolic through them, derives scalar projection where required, and adds
operation-level whole-domain coverage/disjointness traces.
