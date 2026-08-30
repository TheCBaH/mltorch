# Scalar dot product over 6D tensor sections

## Decision

Add dot product as a **`SEMANTICS` primitive** when the purpose is to improve
evaluation of long, computationally intensive contractions.  It still returns
one `SEMANTICS.t` scalar, not a `Graph_ir` tensor edge and not a tensor with
shape `[1,1,1,1,1,1]`.

This is the same scalar result layer as `S.mul` and `S.sqrt`: a compute
functor may use it to build one value.  The primitive boundary is justified
not by new mathematical semantics, but by a better implementation opportunity
than the generic composition `sum (fun i -> load ... i * load ... i)`.  In
particular, Direct can make the entire contraction one tight, stride-aware
payload loop.  A graph-level `Pointwise.Mul` or `Pointwise.Sqrt` is
tensor-valued only because the graph schedules its scalar expression once per
output coordinate.  Dot itself has one scalar result; making it a singleton
tensor would add scheduling and storage work without helping the direct loop.

## Contract

Each operand chooses a logical, axis-contiguous one-dimensional section of a
Native 6D tensor.  It is physically contiguous only when the selected axis is
the channels-last `C` axis; the other axes have fixed dense strides.

```ocaml
type section = {
  tensor : Tensor_ref.t;
  start : Vec6.coord;
  axis : Axis.t;
}

type dot = {
  lhs : section;
  rhs : section;
  count : Op_config.Pos.t;
}
```

For the **semantic** operation, starts must be expressed in the semantic index
domain, rather than as only fixed `Vec6.coord` values:

```ocaml
val dot :
  input -> start:position index Vec6.t -> axis:Axis.t ->
  input -> start:position index Vec6.t -> axis:Axis.t ->
  count:Op_config.Pos.t ->
  t
```

Each `start` is still a full 6D coordinate and each operand still supplies one
axis.  The difference is that a start may be formed from the consuming
operation's output coordinate.  That capability is required for Bmm and
attention, where the row/column/key being contracted varies per output value.
It does **not** make Dot tensor-valued or give it an output coordinate: it
merely lets its scalar expression depend on the enclosing operation's
coordinate, just as `S.load` does.

The `section` record above remains the useful shape-level contract for a
future standalone, fixed-section Dot operation; there its `Vec6.coord` starts
are validated parameters.  A fixed-coordinate-only `SEMANTICS.dot` would be
too narrow to accelerate the existing hot contractions.

`start` gives the coordinate of the first element.  `axis` says which one
coordinate advances through the sequence.  The two operands may name the same
tensor, use different starts, and use different axes.  `count` belongs to the
operation so both sequences have exactly the same length.

Its scalar value is:

```text
sum i = 0 .. count - 1:
  load(lhs.tensor, lhs.start with lhs.axis := lhs.start[lhs.axis] + i)
  *
  load(rhs.tensor, rhs.start with rhs.axis := rhs.start[rhs.axis] + i)
```

This is deliberately not a per-section or broadcasted reduction.  The full
6D start coordinate makes each argument name one exact sequence.  The result
is one scalar with no coordinate of its own; in the semantic form it may still
depend on an enclosing operation's output coordinate through its starts.

## Validation

Validation needs the actual operand shapes, and therefore belongs at the
caller's graph/shape boundary rather than in the scalar compute functor.  For
semantic starts derived from an output coordinate, the compute functor must
prove the same bounds from the validated input/output shape relation.

For each section:

1. Every start component must be in range for its tensor.
2. Its selected-axis interval must fit: `count <= extent(axis) - start(axis)`.
   Use this subtraction form rather than evaluating `start + count`, which can
   overflow before the comparison.
3. `count` is positive in the initial contract.  An empty dot product could
   mathematically return zero, but it adds an avoidable special case and is not
   needed while Native tensors themselves have positive extents.

These checks prove that every load is in bounds.  The compute implementation
may therefore build the selected index from the validated start plus the
reduction index and use the existing encapsulated index conversion.

## Semantic implementations and performance

The Direct implementation is the reason to add the primitive.  Native tensors
are dense, channels-last 6D payloads, not arbitrary-stride views.  Once the
two validated starts and axes are known, Direct can calculate each section's
physical increment once and iterate `count` times with two scalar offsets. It
avoids the generic reduction callback, per-term coordinate construction, and
the repeated six-axis offset calculation performed by generic `load`/`load6`.
It must preserve the existing left-to-right floating-point accumulation order
unless a separately specified numerical-mode change permits reassociation.

The gains are plausible only when `count` is sufficiently large or dot is in a
hot nested computation.  The work remains O(`count`) and is normally limited
by payload reads.  For very short sections, a semantic call plus setup may be
as cheap as, or slightly dearer than, the existing composition; benchmark the
intended workloads before committing to this interface.

Symbolic has two valid lowering choices:

1. **Initially recommended:** implement `Symbolic.dot` by constructing the
   existing `Sum` reduction over a product of two loads.  This keeps expression
   size O(1) in `count` and requires no `Expr`/grounder/kernel changes.  The
   Direct interpreter still receives its optimized primitive path.
2. **Later, only when compiled-kernel profiling justifies it:** add a symbolic
   dot intrinsic.  That lets kernel elaboration emit an explicit optimized
   contraction, but requires a new expression/intrinsic form and handling in
   evaluation, printing, comparison, traversal, grounding, and kernel code.

The first choice deliberately makes the semantic interface richer than the
initial symbolic IR.  That is already established by fixed-window pooling:
the semantic layer is the place to select a Direct fast path, whereas the
symbolic representation should gain a new primitive only when it also needs a
distinct optimized representation.

## Existing `Sum` compared with `Dot`

`S.sum` is a general **unary reduction control form**:

```ocaml
S.sum ~lo ~hi (fun i -> term i)
```

It owns only the reduction interval and accumulator.  The caller supplies an
arbitrary scalar expression for every `i`: it may be a load, a square, an
exponential, a masked expression, a nested reduction, or a product of any
number of values.  `Sum` neither knows nor can exploit how many source tensors
the body reads, which axes their indices advance on, or whether they are dense.
It must keep the callback and generic expression construction.

`S.dot` is a specialised **binary reduction over two tensor sections**:

```ocaml
S.dot lhs ~start:lhs_start ~axis:lhs_axis
  rhs ~start:rhs_start ~axis:rhs_axis ~count
```

Mathematically, it is the restricted case
`Sum(i, load(lhs, advance(lhs_start, lhs_axis, i)) * load(rhs,
advance(rhs_start, rhs_axis, i)))`.  Operationally it owns both loads, both
advancing axes, and the common count.  Direct can consequently precompute two
payload offsets and increments, then perform a simple multiply-accumulate
loop.  `Sum` cannot safely infer that pattern from an arbitrary callback.

The restriction is also why Dot must **not** replace Sum generally:

| Reduction | Can use `Dot`? | Reason |
| --- | --- | --- |
| `sum x[i]` | No | Only one sequence; no second multiplicand. |
| `sum x[i] * x[i]` over one axis | Yes | Dot with the same input/start/axis twice. |
| `sum (x[i] - mean)^2` | No | The per-term subtraction is not a section load; a transformed-input or fused norm primitive would be required. |
| `sum exp(score[i])` | No | Unary nonlinear reduction. |
| Multi-axis convolution/window sum | Usually no as one Dot | It walks several axes and may use padding/dilation; individual one-axis inner contractions may be candidates only if the accumulation order remains acceptable. |
| Matrix row/column contraction | Yes | Exactly two single-axis sequences with a shared count. |

Thus Dot supplements, rather than generalises or replaces, `Sum`.  Its payoff
comes from choosing a much narrower contract that exposes memory traversal to
the Direct backend.

## Projected effect on existing operations

The semantic primitive is deliberately narrow: it does not turn every existing
operation into a scalar operation, and it does not add a graph-level tensor
operator.  It makes one optimized scalar contraction available to operations
that need it.

### Candidate consuming operations

| Existing operation | Current reduction | Dot applicability | Projected computation effect |
| --- | --- | --- | --- |
| `Bmm.Compute.pixel` (`lib/native/ops/matmul.ml`) | `sum_k input[out with C=k] * mat2[H=out.H, W=k, C=out.C]` | **Exact first consumer.** The starts are formed from `out`; lhs advances `C`, rhs advances `W`, and the count is `input_shape.C`. | **Simplification and likely Direct speedup.** Replaces the generic callback and two generic loads per `k` with one semantic dot. This is the clearest benchmark target. |
| scaled dot-product attention score (`Attention.Compute.score_at`, `lib/native/ops/attention.ml`) | `sum_e (query[e] * scale) * (key[k,e] * scale)` | **Near match, but not an automatic replacement.** Query and key are one-axis sections (`C`), but starts depend on `out` and `k`. | Dot could optimize the unscaled query/key contraction. Moving the two scale multiplications outside changes floating-point rounding, so preserve current evaluation order or obtain an explicit numerical-equivalence decision first. The softmax and value-weighted sums remain ordinary `Sum`s. |
| `RmsNorm` (`lib/native/ops/norm.ml`) | multi-axis `sum x^2` | **Conditional self-dot.** A normalization over exactly one axis can use `dot x x` with identical sections. For several axes, use an inner self-Dot on the selected axis plus outer `Sum`s. The current general implementation supports an arbitrary list of axes. | A one-axis specialization has one Dot per output and can load each value once for `x[i] * x[i]`. For several axes, the number of Dots is the product of the other normalized extents; retain the generic reducer unless that specialization is profiled. |
| `LayerNorm` / `GroupNorm` (`lib/native/ops/norm.ml`) | sums of `x` and `(x - mean)^2` over one or more axes | **No direct use.** The centred square is not a product of two raw section loads; their general reductions can cover several axes. | Keep `Sum`. Rewriting variance as `E[x^2] - E[x]^2` merely to use Dot changes numerical stability and is not justified by this proposal. |
| forward / transposed convolution (`lib/native/ops/conv.ml`) | nested channel and spatial/kernel sums of input × weight | **Convolution-specific conditional use.** Contract along `C` at each fixed kernel position: 2D needs `kernel_h × kernel_w` Dots per output; 3D needs `kernel_d × kernel_h × kernel_w`. Group offsets simply become the two `C` starts. A 1×1 convolution needs exactly one C-Dot. | **1×1 is a strong candidate.** For ordinary kernels, C-Dot gives the fewest generally eligible Dot calls because `C` is normally the largest contraction axis, is physically contiguous, and remains valid under dilation. It changes the current channel/spatial accumulation nesting, so it needs an explicit numerical-order decision. A single Dot for a whole receptive field requires a new flattened/strided multi-axis Dot or a convolution-specific fused primitive; materialised `im2col` would add substantial memory traffic. Transposed convolution remains deferred because its contribution/bounds logic is more complex. |
| pooling, pointwise ops, reshape/split/concat | max, unary/pointwise, or data movement | **Not applicable.** | No change. |

Consequently, the expected initial use is **Bmm**, followed by grouped or
ungrouped **1×1 convolution** if its benchmark wins.  Attention and one-axis
RMSNorm are follow-up candidates after numerical testing.  General convolution
is not a one-Dot operation under the single-axis contract: its low-call-count
form is C-Dot at each kernel position, and its single-Dot form requires broader
semantics.

For any convolution decomposition, choosing Dot on contraction axis `a`
requires one Dot per combination of the *other* contraction axes:

```text
dot_calls_per_output = product(extent(b) for b in contraction_axes, b <> a)
```

Thus the fewest calls come from the largest **eligible** axis.  `C` is the
default choice because it is physically contiguous in Native and advances by
one even with spatial dilation.  A width-Dot can use fewer calls only when
kernel width exceeds channels per group and the implementation can guarantee
unit dilation and a fixed valid window.  Fewer Dot calls do not reduce the
total multiply-add count; they reduce semantic-call/setup overhead and expose
larger contiguous inner loops.

| Affected area | Projected change | Computation effect |
| --- | --- | --- |
| `Semantics.SEMANTICS` | Add one `dot` signature. | **Small interface increase, intentional.** Every semantics implementation must provide it, making the direct fast path available to all compute functors. |
| `Direct` | Add a section-dot loop, ideally in a tensor-level helper that precomputes the two dense-layout increments. | **Simplification and performance improvement in hot uses:** one loop, no reduction closure, no per-term 6D coordinate update, and no repeated full offset calculation. Work remains O(`count`) and memory-read bound. |
| `Symbolic` / `Expr` | Initially lower `Symbolic.dot` to the existing `Sum(load * load)` expression. | **Small localized increase, no IR complexity increase:** the new semantic method has an adapter implementation, while expression size remains constant in `count`. A future symbolic intrinsic is a cross-cutting complexity increase. |
| Operation compute functors | `Bmm` adopts Dot first; 1×1 convolution is the next exact contraction; other operations remain unchanged unless they meet the exact contract. | **Simplification for Bmm and likely 1×1 convolution.** Attention/RMSNorm need targeted follow-up; general convolution, pooling, pointwise ops, and general norms do not gain a forced rewrite. |
| Shape/build validation | A call site validates two selected sections against their source signatures. | **Small, explicit increase:** two independent bounds checks per operand; no broadcast, output-shape inference, or rank-repacking rules. |
| Scheduling and tensor storage | None for scalar-expression use. | **Simplification:** no singleton output tensor, no output schedule, and no intermediate `Mul` tensor to materialize. |
| Symbolic grounding, kernel elaboration, and analysis | Initially consume the ordinary existing reduction expression. | **No new operation-specific complexity initially.** A symbolic dot intrinsic would make this row a material increase. |
| `Bmm` / general `matmul` | Bmm becomes the first consumer; general `matmul` remains unchanged. | **Simplification for Bmm:** it is exactly a single-axis contraction. **Avoided complexity for general matmul:** Dot does not inherit rank-polymorphic matrix rules or batch broadcasting. |
| Native4D | No change in the initial proposal. | **No increase now;** a later Native4D scalar-dot feature would be a separate binary arbitrary-axis reduction design. |
| Tests | Add focused Direct-vs-Symbolic and bounds fixtures, plus Bmm regression/performance coverage. | **Small increase:** coverage is limited to the semantic-dot contract and its first consumer, rather than a new graph/operator matrix. |

The largest practical gain is not merely avoiding a graph decomposition such
as `Slice -> Mul -> Reduce.Sum`; it is avoiding the generic scalar composition
inside Direct's innermost loop.  The semantic primitive keeps the contraction
local and lets a consumer's output expression remain fused, while allowing the
Direct backend to choose a data-layout-specific implementation.

### Deferred alternative: scalar graph values

If a dot result later has to cross a graph edge—be an externally visible graph
output, be cached independently, or be consumed by another graph node as a
scalar—the effect changes materially.  Native would need scalar edge types and
their signatures, graph JSON, direct and symbolic environments, constant
handling, graph transforms, and scalar/tensor operand rules.  That is a
cross-cutting **complexity increase**, so it should be justified by a concrete
graph-level use case rather than introduced for this semantic primitive.

## Non-goals and consequences

This does not replace `Bmm` as a graph operation or provide general `matmul`.
It is an implementation primitive that `Bmm` can use for its inner
contraction.

- Existing `Bmm` is a rank-3 layout specialization with `H=batch`, `W=row` or
  contract, and `C=contract` or column.  It is inappropriate for arbitrary 6D
  sections: its RHS load deliberately fixes `N`, `T`, and `D` at zero.
- General `matmul` needs rank-sensitive vector/matrix cases and batch
  broadcasting.  Those semantics cannot be recovered solely from Native's
  fixed 6D shape once original ATen rank has been erased.
- The semantic dot primitive does not by itself provide a standalone scalar
  graph value.
  If it must be a graph output or feed a later graph node as a scalar,
  `Graph_ir` would need typed scalar edges, scalar serialization, evaluation,
  and tensor/scalar operand rules.  A singleton tensor wrapper is a possible
  compatibility boundary, but should not define the semantic API.
- Native4D gains no automatic legalization.  A scalar dot is a binary,
  arbitrary-axis reduction and would need an explicit Native4D decision.

## Recommended tests if implemented

- Direct and Symbolic agreement for same-tensor and two-tensor sections.
- Distinct axes, nonzero starts, and sections ending exactly at an extent.
- Rejection of an out-of-range non-axis coordinate, a too-long selected-axis
  window, and a non-positive count.
- Symbolic-print test showing the existing `sum` lowering; use fixed starts to
  confirm no coordinate variables and Bmm-derived starts to confirm the
  expected enclosing-output dependencies.
- Hand-computed values that distinguish an incorrect axis, start, or operand
  ordering.
- Direct benchmark against the former generic `sum`/`load`/`mul` composition,
  covering a short section and representative hot long sections; record time
  and allocation.  The primitive should be retained only where it wins on the
  intended evaluation workloads.
