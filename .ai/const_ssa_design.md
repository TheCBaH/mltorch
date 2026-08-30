# Deferred constant SSA

## Problem

The current constant-folding pass has two jobs at once:

1. prove that a node depends only on constants; and
2. evaluate the node and attach its resulting `Tensor.packed` payload.

The second job requires archive payloads to have been loaded.  This makes a
purely structural transformation unable to show the graph that would result
from folding.  It is particularly visible in the CHW-to-HWC relayout pipeline:
an imported convolution weight is behind a `Permute`, so it remains a node
output until its bytes are materialised.  The later batch-norm fold consequently
sees it as dynamic; Native4D then inherits a non-canonical Native graph.

The conversion needs a statically known weight, shape, format, and layout.  It
does not inherently need the weight's bytes at graph-analysis time.

## Proposal

Split a constant's *static identity* from its *materialised payload*.  A
constant may be captured data, a literal, or a pure expression over other
constants.  Folding a constant-only node then removes the node from the main
graph even if none of its inputs has been read from the archive.

For example, folding a relaid weight would make the main graph contain:

```text
t2: constant
conv(x, t2)
```

with the sidecar definition:

```text
t2 = permute [H <- W, W <- H] (captured p_conv1_weight)
```

At an explicit materialization stage the sidecar expression is resolved in
dependency order, materialised once, and cached as an ordinary
`Tensor.packed`.  Direct evaluation never resolves sidecar expressions.

This should be called **deferred constant evaluation** (or deferred constant
folding).  It is not ordinary constant folding until the expression has been
evaluated into bytes.

## Representation

Do not embed expressions in `Tensor.packed`.  That type means a physical tensor
and its callers reasonably assume it has data.  Instead, make a sidecar
constant program that is part of Native rewrite state.  It is produced and
consumed by Native transformations before any conversion to another dialect.

Conceptually:

```ocaml
type const_value =
  | Captured of archive_key
  | Literal of Tensor.packed
  | Apply of {
      op : Const_op.t;
      args : const_id list;
      output : Tensor_sig.t;
    }

type constant_plan = const_value Const_id.Map.t
```

The exact types should use the project's versioned recipe and identity machinery
rather than these sketch types.  The key properties are:

- It is a shared SSA DAG, not a recursive expression attached independently to
  every tensor.  Shared transformed weights remain shared and each value has a
  natural cache key.
- Each value has a validated output signature before any payload is loaded:
  shape, format, quantization, and all static operation parameters.
- `Captured` refers directly to a stable archive key (or the equivalent lens
  resolution), not merely to graph provenance.
- It is acyclic and only permits pure operations with constant arguments.
- A main-graph `Input.Constant` can name either a materialised payload or a
  deferred plan value.  The graph-facing classification remains `Constant`.

There are two deliberately distinct states:

```text
symbolic:      main graph + constant plan
materialized:  executable main graph + Tensor.packed constant table
```

Keeping the materialised cache separate from the immutable plan is clearer:
the plan is reproducible transform output; the cache is an execution detail.
The existing direct evaluators accept only the materialized state.

## Constant dialect

The constant dialect should be deliberately smaller than `Graph_ir`, but it
must be rich enough to represent normal constant folding for the supported
canonical pipeline.  Symbolic folding is the default; materializing a folded
value is an explicit fallback, not the normal representation of a constant.

`Permute` is sufficient to make CHW-to-HWC relaid weights structurally
constant.  Batch-norm folding then emits constant arithmetic, including scalar
literals, broadcasted `Add`/`Sub`/`Mul`/`Div`, `Sqrt`, and permutations.  Those
operations must also be representable before the second fold can turn the
resulting convolution parameters into deferred constants.

The initial permitted set must cover the single-output, pure operations that
the existing fold encounters in canonicalization:

- literals and captured tensors;
- `Permute` and any required reshape/reindex operation;
- pointwise arithmetic and scalar variants;
- `Sqrt` and the other unary operations emitted by existing folds;
- broadcast semantics and format/quantization rules identical to Native.

Multi-output operations remain out of scope until the plan can define all
outputs atomically.  Dynamic-shape, stateful, random, or input-dependent
operations are never constant-plan operations.

Avoid independently reimplementing their numerical behavior.  `Const_op` may
be a restricted representation, but it should reuse Native operation payloads,
shape inference, and evaluation semantics where possible.  This preserves
rounding, dtype, quantization, broadcast, and error behavior.  An explicit
capability predicate should say which Native operations are legal in Const-SSA;
the representation must not become a second unconstrained `Graph_ir`.

## Transformation behavior

The foldability predicate changes from:

```text
all operands are graph constants with bound Tensor.packed payloads
```

to:

```text
all operands are effective constants with a valid constant-plan definition
```

For a foldable single-output node, the rewrite:

1. removes the main-graph node;
2. retains its output id as a main-graph constant input;
3. adds an `Apply` value to the sidecar plan;
4. records the same value correspondence and diagnostic provenance as ordinary
   folding; and
5. drops now-unused main-graph operands, while retaining any captured-plan
   dependencies needed to resolve the new value later.

The dependency in step 5 is important.  If folding removes graph tensor `w`, a
recipe referring only to `w`'s old graph id is invalid after the rewrite.  The
new plan must instead retain `Captured archive_key` or an earlier plan value.

This lets the structural pipeline report the same main graph irrespective of
whether payloads were preloaded.  Preloading can warm a later materialization
cache, but it must not enable a different symbolic graph rewrite.

If a constant-only operation is not expressible in Const-SSA, the symbolic fold
does not remove it.  A later explicit materialization stage may evaluate that
operation if all required payloads are available and replace it with a literal
materialized constant.  If they are unavailable, the stage reports its missing
payloads or leaves the operation unfolded according to its requested policy.

## Lowering and evaluation

Const-SSA is a Native facility.  Native constant folding produces the canonical
Native graph and its constant plan; it is not a Native4D legalization pass.
All of the following happen in, or are defined against, Native:

- `is_effective_constant`;
- symbolic folding of constant-only Native operations;
- the constant-operation capability set and its shape/format validation; and
- the explicit materialization pass.

Before direct Native execution, materialization resolves every used constant
through:

```text
materialised cache -> deferred constant plan -> archive/literal source
```

and writes the resulting `Tensor.packed` values into the executable constant
table.  Resolution must be memoized per plan value for one materialization or
loaded model; otherwise a shared transformed weight could be recomputed for
every use.  `Eval_direct` then binds those tensors exactly as it does today; it
has neither an archive handle nor a Const-SSA evaluator.

### Native4D conversion

Native4D is a consumer of the canonical Native result.  It does not need to
understand an unfused Native constant-producing subgraph.  A plan value used as
a Native4D graph constant is valid when its *exported Native signature* has
`T = D = 1`; its captured leaves and intermediate plan values need not have
that shape.

For example, a captured Native six-axis convolution weight may be an input to
a Const-SSA `Permute` whose output has `T = D = 1`.  The Native4D graph names
only that output constant.  Its 6D captured source remains an implementation
detail of Native Const-SSA materialization and does not become a Native4D graph
input.

Consequently, Native4D domain checking only needs to verify the signatures of
the post-fold Native graph and its visible constant-plan outputs.  After the
explicit Native materialization boundary, `Eval_direct4` receives ordinary
four-axis `Tensor.packed` constants and remains unchanged.

A Native4D-specific lowering step that reads constant elements to synthesize a
new tensor is separate from this design.  It must either consume the already
materialized Native constants or explicitly construct a Native Const-SSA value
before conversion.  It must not introduce an independent Native4D constant
language.

## Batch-norm folding

Batch-norm folding is a required Const-SSA use case, not an exception that
forces materialization.  Its existing rewrite already emits parameter
arithmetic as ordinary graph nodes.  The second constant fold should represent
those nodes as Const-SSA values:

```text
scale  = gamma / sqrt(variance + epsilon)
weight = conv_weight * broadcast(permute(scale))
bias   = (conv_bias - mean) * scale + beta
```

Here `epsilon` and identities for absent optional parameters are typed literal
values in the constant plan.  `permute` and broadcasting use their normal
Native meaning.  The resulting convolution therefore has direct,
effective-constant `weight` and `bias` edges without requiring archive data.

This is why the permitted operation set must include the arithmetic and
broadcast forms above before deferred constant folding replaces the canonical
fold.  A `Permute`-only plan may be a useful prototype for structural relayout
inspection, but it is not sufficient for canonical batch-norm folding.

## Provenance and archive identity

Provenance must not be used as a payload lookup mechanism.  Provenance from an
unpermuted captured tensor to a permuted tensor explains derivation, but the
captured bytes still have the unpermuted layout.  Treating provenance as the
payload would silently read incorrect coordinates.

A deferred value therefore needs both:

- a direct captured source capable of loading the original bytes; and
- the exact operations that convert those bytes to the output layout.

The existing provenance and correspondence records remain valuable for
diagnostics, graph maps, and source reporting; they are not substitute data
definitions.

## Benefits

- Graph inspection, cost analysis, and dialect-domain checks no longer require
  model weights to be loaded.
- The canonical graph structure is independent of preload policy.
- CHW-to-HWC conversion can treat relaid weights as static immediately.
- Payload IO becomes demand-driven and unsupported archive tensors that are
  never used need not be decoded.
- The plan gives a precise, serializable explanation of how each derived
  constant is obtained.

## Costs and risks

- This adds a second SSA data structure, lifetime rules, serialization, cache
  invalidation policy, and a resolver.
- Exact numerical behavior is non-negotiable.  A convenient but separate
  evaluator could introduce rounding or quantization mismatches.
- The plan must be validated as carefully as the main graph: signatures,
  operand existence, purity, topological order, and archive identity.
- Plans can retain large dependency graphs.  Common-subexpression sharing,
  dead-plan elimination, and bounded diagnostics are needed.
- An unrestricted embedded operation graph would duplicate most of `Graph_ir`.
  The dialect must stay tied to genuine compile-time constant operations.

## Recommended implementation order

1. Introduce `Constant_plan`, `is_effective_constant`, and distinct symbolic
   and materialized transform states.
2. Define the Const-SSA capability set for captures, literals, `Permute`, and
   the arithmetic/broadcast operations used by batch-norm folding.
3. Add structural deferred folding for single-output constant nodes, make graph
   dumps show plan definitions, and make both canonical fold rounds symbolic.
4. Add an explicit, memoized Native `materialize_constants` boundary that delegates
   operation semantics to existing Native evaluation.  Direct evaluators remain
   materialized-only.
5. Make Native4D conversion consume this canonical Native result.  It checks
   only exported plan values for the four-axis invariant; captured plan leaves
   may remain six-axis Native tensors.
6. Keep any Native4D-specific lowering that reads elements after the Native
   materialization boundary, or have it create Native Const-SSA values before
   conversion.

This sequencing solves the immediate graph-analysis problem early while keeping
the executable, numerically exact path incremental and testable.
