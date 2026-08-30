# `index.Tensor`: why it stays deferred

**Status: investigated, not implemented, 2026-08-29.** This is a scoping
record for `todo.md` §2 item, not a design for landed code — the opposite of
`.ai/matmul_softmax_design.md`, which records a decision that *was* acted on.

## 1. The one occurrence

CSATv2 (945 nodes) has exactly one `torch.ops.aten.index.Tensor` node, and it
is the only occurrence anywhere in the 100-model sweep. Its `indices` operand
is `[None, None, <tensor>]` against a rank-3 `self` (`[784, 3, 64]`) — matches
`ops.md`'s framing, "an optional-tensor index list with only the final index
live": a single-axis gather over the last axis, broadcast over the two
leading axes. ATen's own promotion rule for this shape
(`self.shape[:p] + index.shape + self.shape[p+1:]`) gives `[784, 3, 64]`,
matching the archive's declared output shape exactly.

Tracing the index tensor back through the graph (`clone_1 <-
clone.default <- c_stem_dct_lifted_tensor_0`) shows it is a **lifted tensor
constant** — `model_constants_config.json` names it `stem_dct.lifted_tensor_0`,
dtype `LONG`, shape `[64]`, backed by real bytes in `data/constants/tensor_0`.
It is not derived from the model's runtime input; every run of this model
uses the same fixed reindexing (almost certainly the model's DCT
coefficient/zig-zag reorder, given the module name `LearnableDct2d`).

## 2. Why "bake the constant permutation into the graph" doesn't work for both importers

The obvious narrow scope — decode the index tensor's concrete `int64` values
at import time and store them as a static lookup table in a new op's params,
the same way `Split_with_sizes`' sizes or `Slice`'s offsets are compile-time
data — runs into an asymmetry between the two importers this repo requires to
stay in sync (`.ai/native_add_op.md`):

- **`Op_bridge`** is ATen-linked and reached with live `Aten_tensor.t` values
  already materialized, so it genuinely could read the constant's concrete
  bytes at dispatch time.
- **`Native_interp`** is metadata-only during lowering. A constant input is
  imported with only its shape/dtype (`Input.Constant` kind); its concrete
  bytes are not available until `Eval_direct.run ~constants` runs, well after
  the graph structure is fixed (`native_interp_exec.ml`'s `transform_lowered`/
  `constants_for`). There is no channel back from "the eval-time constant
  store" into "the graph-build-time op params" — building it would mean
  `Native_interp`'s import step stops being metadata-only for this one op,
  which is a boundary the rest of the importer (and the jsoo backend it
  feeds) currently relies on.

So a build-time-baked lookup table is only implementable for one of the two
importers required to land any op (`ops.md`'s five-layer "supported"
definition demands both).

## 3. Why a genuinely runtime gather is a bigger architectural unit than it looks

The alternative is a real two-tensor-input `Graph_ir` node (`self`, `index`)
evaluated at `Eval_direct`/`Eval_symbolic` time like any other op, with the
gathered axis's position read out of the `index` tensor's *value* rather than
known at compile time. `lib/native/semantics.ml`'s `SEMANTICS` interface has
no primitive for that direction: `value_of_index : delta index -> t` exists
(used by `max_pool2d_with_indices`, whose *output* is an index reported as a
value), but nothing goes the other way, `t -> position index`.

Adding one is not a small extension:

- **`Direct`**: trivial — truncate/round the float and bounds-check it into a
  `Dim.index`.
- **`Symbolic`**: every existing index expression in this engine is affine in
  the enumerated loop coordinates (`n,t,d,h,w,c`) — that invariant is what
  lets `Kernel.Bounds`, `output_transfer`'s claim classification, and the
  kernel-DSL lowering reason about a node's footprint without evaluating it.
  A value read out of another tensor is not affine in anything; it would need
  a wholly new "opaque, data-dependent index" AST case that those consumers
  do not know how to interpret, or an explicit refusal to run this op under
  `Symbolic` at all (breaking the "every op gets both walk layers" cross-cutting
  rule in `todo.md`, unless justified the way `Group_norm`/`Select`'s Native4D
  gaps already are — but this would be a *Native*, not Native4D, gap, which
  has no existing precedent for a partial-semantics op).

Either path is a unit of work comparable to (or larger than) `Sdpa` itself,
for a single real-world occurrence.

## 4. Decision: leave deferred

`index.Tensor` stays out of scope for now. CSATv2 stays a graph-only CI
fixture regardless — `todo.md` §2's own header already says so until the
*whole* section (including the deliberately-deferred batched `matmul.default`
family) is substantially complete — so landing this op alone would not move
CSATv2 to a native-verified target, and it blocks no other model in the
100-model sweep. If a future model needs `index.Tensor` on a genuinely
runtime-dependent index (not a lifted constant), the design would have to
start from §3 above, not from the narrower "bake a constant table" idea this
investigation ruled out.
