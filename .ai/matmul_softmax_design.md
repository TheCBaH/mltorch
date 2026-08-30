# `matmul.default` / `softmax.int` — matrix/attention dialect decision

Resolves `ops.md`'s §3 row, the prerequisite `ops.md`
flags before landing either op ("do not add only one half"). Grounded against
three real downloaded archives inspected directly (`unzip -p <model>.pt2
<model>/models/model.json`, walked with a scratch Python script — no repo
tooling exists yet to do this, since neither op is bound): `csatv2` (the
`ops.md` row's original evidence), `mobilevitv2_175`, and `mvitv2_tiny` (both
named as stopping on this pair in the `clone.default` landing note).
The three models turn out to exercise **three structurally unrelated
computations that happen to share ATen op names** — the finding that drives
the whole decision below.

## 1. Decision

**(a) independent Native ops, not (b) a recognized attention subgraph** — and
narrower than `ops.md`'s own phrasing of (a) ("Native `Matmul` + `Softmax` and
a richer Native4D matrix dialect"), because the corpus evidence does not
support a single general `Matmul`:

1. `softmax.int` becomes its own standalone Native reduction op, `Softmax`
   — general, not attention-specific, sharing the `Dims_keepdim` reduction
   family's shape rule (single axis, not a list) and Sdpa's already-proven
   `exp`/`max_reduce`/`sum` primitives. See §3.
2. `matmul.default` is bound at the **import boundary**, not as one new
   Graph_ir node, because its corpus occurrences split into two shape
   families with no common representation:
   - batch-less (every leading axis extent 1) binds to the **existing**
     `Bmm` node — zero new Native surface, and Native4D's existing
     batch=1→`Conv2D` legalization (`native4d_design.md` §7.4) already
     covers it. See §4.
   - batched/multi-head (`D` and/or `H` > 1, exactly Sdpa's `[D,H,W,C]`
     frame) needs a genuinely new Native op. **Not undertaken by this
     decision** — scoped as its own follow-up row (§5), because it is
     comparable in size to Sdpa's own landing, not a small extension of
     `Bmm`.
3. Native4D gains **no new dialect richness** for either op. `Softmax` gets
   a typed rejection (no reduction primitive beyond `MeanKeepDims` exists in
   the reduced CNN dialect — the same absence §9 of `attention_design.md`
   already used to reject Sdpa). The batched-matmul case gets a typed
   rejection for the same reason `attention_design.md` §9 already proved for
   Sdpa: it would name Native's `D` axis, and `Axis4.of_axis Axis.D = None`
   by construction — no amount of dialect work makes that representable.
   This means `ops.md` row 204's question ("decide whether Native4D remains
   a CNN dialect") is answered the same way for this pair as it already was
   for Sdpa: **yes**, and this decision adds no exception to that answer.

Why not (b): a "recognize and legalize the complete attention subgraph"
approach needs a subgraph shape to recognize. §2 shows there are at least
three, unrelated at the graph-structure level (one has no matmul at all; the
other two use `matmul.default` for arithmetically different jobs). Pattern-
matching three unrelated decompositions under one legalizer is the same
mistake `select.int`/`stack.default`'s own history already corrected
(each now has its own node rather than being recognized as a
`Slice`+`Reshape` / N-`Reshape`+`Concat` pattern) — a shared
op *name* is not evidence of a shared subgraph *shape*.

## 2. The three corpus patterns

### 2a. `csatv2` — channel attention, rank 2, no batch axis at all

`ops.md`'s original evidence: 28 `matmul.default` + 14 `softmax.int`, exactly
one shape pair, all 28/14 occurrences:

```text
matmul:   [1,49] x [49,1] -> [1,1]      (score: a single scalar, not a row)
softmax:  [1,1] -> [1,1], dim=-1        (over exactly one element)
matmul:   [1,1] x [1,49] -> [1,49]      (rescale the value slice by that scalar)
```

Traced to its producer: `unbind.int(linear_4, dim=-1)` where `linear_4` has
shape `[1,49,3]` — the three `matmul` operands per block are a 3-way
per-channel split (`timm.models.csatv2.SpatialAttention` /
`SpatialTransformerBlock`, per `nn_module_stack`), not Q/K/V. Every softmax
is over a **single-element axis** — its output is a proof of the shape
family the op appears in, not evidence that a real multi-key reduction is
needed here. The matmul itself is ordinary 2D-by-2D matrix multiplication
(ATen's own `matmul.default` contract for two rank-2 operands): no leading
batch dimension exists to broadcast or preserve.

### 2b. `mobilevitv2_175` — separable self-attention, softmax alone, no matmul

`matmul.default` count: **0**. `bmm.default` count: **0**. 9 `softmax.int`
occurrences, shapes `(1,1,4,16)`, `(1,1,4,49)`, `(1,1,4,196)`, all `dim=-1`.
Traced to its producer: `split_with_sizes.default(conv2d_15, [1,224,224],
dim=1)` on a `[1,449,4,196]` tensor, softmax applied directly to the
size-1 "query score" split. This is MobileViT-v2's actual published
mechanism (linear/separable self-attention: `context_scores =
softmax(query)`, then `context = sum(context_scores * key)`, no `QKᵀ`
matmul at all) — `softmax.int` here is a genuine, general reduction over a
real axis (up to 196 elements), structurally unconnected to any matmul.

### 2c. `mvitv2_tiny` — real batched multi-head attention, Sdpa's own frame

`matmul.default` count: 20; `softmax.int` count: 10; `bmm.default`: 0.
Sample shapes (all confirmed `D`=1, only `H` varies):

```text
[1,1,3136,196] x [1,1,196,96]  -> [1,1,3136,96]     (H=1)
[1,2,784,196]  x [1,2,196,96]  -> [1,2,784,96]       (H=2)
[1,4,196,196]  x [1,4,196,96]  -> [1,4,196,96]       (H=4)
[1,8,49,196]   x [1,8,196,96]  -> [1,8,49,96]        (H=8)
```

softmax occurrences (all `dim=-1`): `(1,1,3136,196)`, `(1,2,784,196)`,
`(1,2,784,784)`, `(1,4,196,196)`, `(1,4,196,784)`, `(1,8,49,196)`,
`(1,8,49,49)`. Right-aligned by `Aten_shape.of_aten`, a rank-4 tensor lands
on `[D,H,W,C]` (`native_aten_bridge_layout.md`:40) — **exactly**
`attention_design.md`'s Sdpa frame (`D`=batch, `H`=heads, `W`=sequence,
`C`=feature), not `Bmm`'s rank-3 `[H=batch,W,C]` frame. This is real
multi-head QKᵀ / softmax / ·V attention, decomposed rather than exported as
one `scaled_dot_product_attention.default` call. No sample has `D > 1` —
only `H` (1/2/4/8) varies — so there is no evidence in this corpus for a
distinct outer batch dimension on top of heads, only for heads themselves.

## 3. `Softmax`: a standalone reduction, not attention-specific

`Graph_ir.Softmax`, params `{ axis : Axis.t }` — a single axis, not a list
(ATen's `softmax.int` takes exactly one `dim`; there is no multi-axis
`Dims_keepdim.t` to reuse for the *params* shape, only for the *style* of a
tiny wrapper module: `output_shape` is the identity on the input shape,
since softmax never changes rank or extent, unlike `Mean`/`Amax`/
`Vector_norm` which can drop the reduced axes).

`Compute` needs the ordinary numerically-stable formula — max-subtract,
exponentiate, sum, divide — **not** `Sdpa`'s `_safe_softmax` all-`-inf`
zero-row guard: that guard is specific to `_safe_softmax`, the internal
helper ATen's SDPA math backend calls, not to plain `torch.softmax` /
`aten::_softmax`, which is free to produce `nan` on an all-`-inf` row (and
is expected to). Copying Sdpa's guard here would be a silent semantic
change, not a simplification. No new `SEMANTICS` primitive is needed:
`exp`, `max_reduce`, `sum` already exist and are already exercised
(`attention_design.md` §6 confirms all three).

Import boundary: both `Op_bridge` and `Native_interp` decode `dim`, resolve
it to a Native `Axis.t` via the existing `axis_of_dim ~rank` helper
(`native_aten_bridge_layout.md`:39 cites the same helper for `unbind.int`'s
`dim`), and reject `half_to_float=true` (the corpus shows f32 throughout;
ATen's `_softmax` decomposition of `softmax.int` always passes
`half_to_float=false` for an already-f32 input, so this is expected to be a
dead branch, not a live gap — confirm against the real dispatcher before
relying on that, per `.ai/native_add_op.md`'s "decode every argument" rule).

Native4D: typed rejection, `Domain.check_node`'s new `Softmax` arm following
`Sdpa`'s precedent (`` `Softmax_no_dialect_primitive `` or similar —
operation-specific reason, not the generic `Unsupported_op`, per
`native4d_design.md` §7.9's own naming rule) — the reduced CNN dialect has
no reduction primitive beyond `MeanKeepDims`, and nothing in the corpus
needs a Native4D-converted softmax: every model that reaches this op is
transformer-shaped and already stops at another Native4D domain limit
(§7.9's `D`-axis argument, or the batched-matmul case below) regardless of
whether `Softmax` itself converts.

## 4. `matmul.default`, batch-less case: bind to the existing `Bmm`

**No new Graph_ir node.** ATen's own `matmul.default` contract promotes two
rank-2 operands to a rank-3 batched matmul with batch=1 before dispatching
(this is not a Native-side choice — it is what the op already means); this
decision only recognizes that promoted form as identical to what `Bmm`
already computes. `Op_bridge` and `Native_interp` both gain a `matmul`
binding that:

1. reads the **raw ATen rank** of both operands before `Tensor_bridge.of_aten`
   erases it (the same `attention_design.md` §5 rule: rank lives only at the
   import boundary, "nothing downstream of `of_aten` has anywhere to keep
   it");
2. accepts exactly: both operands rank 2, or both rank ≥ 3 with every
   leading (non-matrix) axis extent exactly 1 — the only shapes §2a's
   evidence covers;
3. builds a `Graph_ir.Bmm` node against the two operands' already-`of_aten`
   rank-3-equivalent shapes (`H=1,W,C`) unchanged;
4. rejects everything else (rank 1, or any leading axis > 1) with a typed
   diagnostic naming which condition failed, not a bare "unsupported" —
   §5 explains why the >1 case is deliberately out of scope here rather
   than silently mishandled.

Native4D needs **no change at all**: the resulting `Bmm` node is
indistinguishable from any other `Bmm` node, and `native4d_design.md` §7.4's
existing batch=1→`Conv2D` legalization already covers it (mat2 permuted to
convolution-weight layout, 1x1 `Conv2D` applied to input, `Identical`).

## 5. `matmul.default`, batched/multi-head case: out of scope, its own row

§2c's evidence (`mvitv2_tiny`) is real batched multi-head attention in
exactly `attention_design.md`'s `[D,H,W,C]` frame, decomposed into explicit
`matmul.default`(QKᵀ) → `softmax.int` → `matmul.default`(·V) rather than one
`scaled_dot_product_attention.default` call. Landing this needs:

- a new Native op (batched matmul over the `H`-and-inward frame, `D`
  included per the observed shape even though no sample has `D > 1` —
  restricting to `D = 1` would be exactly the kind of unevidenced
  narrowing `todo.md`'s "remove-unnecessary-specialization audit" already
  flagged elsewhere; `D` costs nothing extra here since the frame already
  carries it and `Bmm`'s own `H=batch` shape rule generalizes directly by
  adding one more batch-like axis, the same generalization
  `attention_design.md` §2 made for Sdpa itself);
- its own shape rule, `Compute`, ATen binding, both importer arms, and
  verification (ATen-vs-Native walk + Direct-vs-Symbolic fuzz walk) — a
  unit of work comparable to `Bmm` itself, not a small extension;
- Native4D: **no work at all**, typed rejection by construction — same
  argument `attention_design.md` §9 already proved for Sdpa (`D` has no
  `Axis4` name), reachable here with zero new reasoning since the frame is
  the same one.

This is deliberately **not** landed by this decision. `ops.md`'s row and
`todo.md` §2 should carry it as its own row once scheduled, separate from
§3's `Softmax` and §4's batch-less `matmul` binding — those two are
comparatively small (an existing reduction family / an existing node,
respectively) and do not need to wait on this larger op.

## 6. What stays a typed rejection, no corpus evidence either way

- `matmul.default` with unequal-but-broadcastable leading axes (e.g. one
  operand's batch axis is 1 and the other's is not) — ATen's contract
  allows this via broadcasting; no downloaded archive exercises it.
- `matmul.default` with a rank-1 operand (vector-matrix / matrix-vector /
  vector-vector forms) — a different output-rank rule ATen applies only for
  rank-1 operands; no evidence either.
- `softmax.int` with `half_to_float=true`.

Each should fail with a named diagnostic, not silently produce a wrong
answer or a bare "unsupported".

## 7. Verification note

Two of the three grounding models (`mobilevitv2_175`, `mvitv2_tiny`) are not
in `PT2_MODELS_CRAM`/`PT2_MODELS_ALL` at the time of this decision; they were
fetched ad hoc (`make pt2.download PT2_MODEL=<name>`) purely to inspect
`model.json` for this decision and are not otherwise wired into any test
target. `csatv2` remains the graph-only CI fixture per `.ai/testing_strategy.md`
regardless of how §3/§4 land — this decision does not change that (§3/§4
alone do not clear CSATv2's other blockers: `index.Tensor`, `addcmul.default`,
and the N/H/W/C-incompatible-attention Native4D gap `ops.md`:109 already
documents).
