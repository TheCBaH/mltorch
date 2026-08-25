# Operations TODO

Derived from `ops.md` (status 2026-08-23, including its "Design-goal audit"
section), cross-checked against the source tree as of this checkout (branch
`unbind-int`). Completed items with no open follow-up have been removed from
this file — GELU/Sigmoid/scalar `mul.Tensor`/`add.Tensor` (`ops.md`'s "Low"
table), the six-model walk/regression closeout, the SDPA/Native4D decision,
and the design-goal fix giving `select.int`/`stack.default` their own single
`Select`/`Stack` `Graph_ir` nodes (each previously decomposed into a
`Slice`+`Reshape` pair, resp. N `Reshape`s + `Concat` — `Select`/`Stack`
reuse `Slice`'s/`Concat`'s shape rule and `Compute` functor, not their node;
JSON codec, dispatch audit, and Direct-vs-Symbolic coverage all landed;
`me_visualize_frontier_cram.t` and the PT2_DATA-gated model-support snapshot
both confirmed unaffected — capability/blocker status and node counts are
unchanged, since the fix is purely internal to how each op's node is shaped;
Native4D `Select4`/`Stack4` deliberately deferred, same `unsupported()`
treatment they still get in `lib/native4d/domain.ml`/`lower.ml`; and **item 1
below, `Concat4`** — domain rule (`Domain.check_node`'s `Concat` arm now runs
`check_dims` on the joined axis, the same shape [`Slice`]/[`Unbind`] get,
instead of `unsupported()`), closed-op registration (`Ops4.Concat4`, alphabetized
into `Op.op` between `Clamp` and `Conv2d`), builder (`Builder.concat4`), shape
(delegates to `Concat.Concat.output_shape`, never restated), direct/symbolic
eval (delegates to `Concat.Concat.Compute`), JSON round-trip, output-transfer
claim (`Reindexing`, alongside `Permute4`/`Reshape4`/`Slice4`/`Unbind`), and
the `Concat -> Concat4` lowering arm (`Identical`, axis-only conversion) —
are all landed and documented in `ops.md` and git history; nothing
outstanding remains against them. Regenerating the 100-model sweep
(`make pt2.json-model-support`) after landing `Concat4` moved 9 more models
across the `native4d_converts` line — `rdnet_tiny` and the 8
`rexnet*`/`rexnetr*` variants, all previously blocked on "no legalization for
concat" — confirming the cross-model leverage this item predicted. **And
`split_with_sizes.default`** (formerly item 1 below) — own `Graph_ir` node
(`lib/native/ops/split.ml`'s `Split_with_sizes`, alongside `Unbind`),
`output_shapes` checking positivity and the exact-sum-to-extent contract,
ATen binding (`bin/aten_ops_gen.ml`), both bridge arms (`Op_bridge` and
`Native_interp`), and dtype preservation through a new `Tensor.split_with_sizes`
`Eval_direct` bypass sharing `copy_cells` with `Tensor.unbind`'s — verified
against real ATen (`Interp_verify`) for f32 and int64. Native4D `Split4` is
the deliberately-not-done half, same `unsupported()` treatment `Select4`/
`Stack4` get. Regenerating the sweep after this landing moved 7 more models
from `native_builds:false` to `true` (`inception_next_atto`,
`inception_next_tiny`, `mixnet_l`, `mixnet_xl`, `mixnet_xxl`, `tf_mixnet_l`,
`tf_mixnet_s`; the 8th predicted model, `lambda_resnet26t`, still fails to
build, now on `softmax.int`) — all 7 now stop at Native4D's domain check
instead. Neither `Concat4` nor `Split_with_sizes` has an ATen-vs-Native
ops-walk or a Native Direct-vs-Symbolic fuzz-walk registration yet (same gap
`cat.default`/`stack.default`/`select.int` already have — see the
cross-cutting note below); not fixed in either of these changes. See
`.ai/pt2_model_support.md`'s 2026-08-25 update for both.

Ordered: (1) the rest of CSATv2's medium-complexity structural set; then
(2) the high-complexity matrix/attention/indexing set.

## 1. Medium complexity — remaining CSATv2 structural set

Independent vertical slices, in this order (unchanged from `ops.md`, verified
still unimplemented in `Graph_ir` — no `Group_norm`, `Amax`, `Pow_scalar`,
`Vector_norm`, or `Upsample` node exists today):

1. **`group_norm.default`** — not in CSATv2 either, blocks 3 models in the
   sweep. A distinct normalization primitive (grouped-channel statistics),
   not a legalization to existing `Batch_norm`/`Layer_norm` — needs its own
   shape rule (channel groups must divide channel count) and Native4D axis
   check.
2. **`amax.default` (14)**, restricted to the observed `dim=[1]`,
   `keepdim=true` configuration — general empty/`None` dims deferred.
   `Max_keepdims` in Native4D only after axis-domain checking.
3. **`pow.Tensor_Scalar` (1, exponent 0.5) + `linalg_vector_norm.default`
   (14, ord 2, dims `[1,2]`, keepdim)** — prefer a small explicit numeric
   kernel only if it matches ATen tolerance; otherwise fused `Pow_scalar` +
   `Vector_norm`. Needs reduction primitives beyond `Mean_keepdims`. The
   sweep hits both independently outside CSATv2 too (`mobilenetv5_base` on
   `pow.Tensor_Scalar`, `swiftformer_xs` on `linalg_vector_norm.default`) —
   corroborating evidence, not extra scope.
4. **`upsample_bilinear2d.vec` (14)**, explicit output sizes,
   `align_corners=true` — real bilinear-resize op (shape inference,
   coordinate transform, boundary handling); Native4D `ResizeBilinear4` must
   reject/diagnose unsupported axes rather than reinterpret them. Do not
   conflate with `upsample_bicubic2d.vec`, which the sweep shows blocking
   `sam2_hiera_tiny` separately — a different coordinate-transform/overload,
   not covered by this row.
5. **`clone` with a supplied memory format (2 in CSATv2)** — decode the enum,
   prove identity for the observed layout or represent the relayout; only
   admit a no-op proof in Native4D, never assume a requested layout
   conversion is a plain `Clone`. The sweep shows this same case hard-rejects
   3 more models (`hiera_tiny_224`, `mobilevitv2_175`, `mvitv2_tiny`) as
   `malformed` — i.e. the importer's refuse-rather-than-ignore choice is
   already correct there too; no separate design question, just more
   coverage once this row lands.
6. **`max_pool2d.default` with `ceil_mode=true`** — not in CSATv2 at all; new
   from the sweep, blocks 2 models (`hgnetv2_b0`,
   `legacy_seresnext26_32x4d`), also currently a hard `malformed` reject.
   Same shape-computation family as the existing pooling ops: ceiling instead
   of floor division when computing output extent.

Each row: ATen binding in `bin/aten_ops_gen.ml` if absent, `Aten_op_config`
decoder/spec fixture, bridge lowering, Native implementation, dispatch audit.
Add an ATen-vs-Native walk only where a non-vacuous recipe exists; otherwise a
table-driven boundary suite. Keep CSATv2 graph-only in CI until a complete
end-to-end execution test passes (`ops.md` explicit instruction) — items 1 and
6 have no bearing on that CSATv2 gate since they don't appear in its graph,
but should not be deferred behind it either given their independent
cross-model leverage. Every new bridge arm here must build exactly one node
per ATen op (the design-goal rule `select`/`stack`/`concat4`/
`split_with_sizes` were fixed against or built to) — none of these targets
has an obvious decomposition temptation the way `select`/`stack` did, but
check before landing.

## 2. High complexity — matrix/attention and indexing semantics

Blocked on a design decision this repo has *not yet* made for the general
case (SDPA-as-literal-target is already decided — Native4D rejects SDPA by
construction, `.ai/attention_design.md` — but general `matmul`/`softmax` is
not):

1. **Matrix/attention dialect decision** (prerequisite for everything below):
   choose (a) Native `Matmul` + `Softmax` and a richer Native4D matrix
   dialect, or (b) recognize/legalize the complete attention subgraph as a
   unit. Write this decision down in `.ai/` before adding either half —
   `ops.md` explicitly warns against landing only one.
2. **`matmul.default` (28) + `softmax.int` (14, dim -1)** — CSATv2's
   decomposed attention. Native only has `Bmm` (confirmed: no `Matmul` node
   in `Graph_ir`), Native4D only legalizes single-batch BMM to convolution
   and has no softmax. Depends on (1).
3. **`index.Tensor` (1)** — gather semantics via an optional-tensor index
   list with only the final index live. Start with a narrowly typed
   single-axis gather contract only if it unblocks the model; don't bundle
   with the scalar/activation work.
4. **`addcmul.default` (14)** — prefer bridge decomposition to `Mul`+`Add`
   for the verified default `value=1`; use `Mul_scalar` (already implemented)
   for non-unit `value`. Scheduled last because it must preserve
   rounding/order claims. Note: this is the same kind of multi-node
   decomposition `select`/`stack` were fixed against — only acceptable here
   because `addcmul`'s ATen semantics *is* literally `add` composed with `mul`
   (no shape/axis logic of its own to lose), unlike `select`/`stack`, which
   had their own shape rule a decomposition would have hidden — now each has
   its own `Select`/`Stack` node instead. If that distinction stops holding
   for `addcmul` too (e.g. a fused-kernel/precision requirement
   surfaces), give it its own `Addcmul` op instead.

CSATv2 (945 nodes) stays a deliberately graph-only CI fixture until this
section is substantially complete — do not attempt to make it a native
verification target piecemeal.

## Cross-cutting, applies to every item above

- Follow `ops.md`'s five-layer "supported" definition for every new row: ATen
  surface, import/bridge, Native, Native4D, verification — and record which
  layers are deliberately out of domain rather than silently skipping them.
- **One Native graph node per ATen op** — share implementation between
  similar ops (a `Compute` functor or shape rule one op module calls from
  another), never decompose one ATen target into a chain of other ops' nodes.
  See `ops.md`'s "Design-goal audit" and `.ai/native_add_op.md`.
- Every new operation/configuration needs both walk layers (ATen-vs-Native
  ops-walk *and* Native Direct-vs-Symbolic fuzz walk), not just a happy-path
  fixture — this is what tripped up GELU/Sigmoid initially per `ops.md`'s
  "Current-source audit" section.
- `make format` before every commit; use `fixup!` commits per finding when
  responding to review feedback, one per defect (see `CLAUDE.md`).

## Explicitly not actionable from this file

- **Kernel/Fusion `over_limit` ("evaluation depth exceeds 2048")** — 8 models
  in the sweep fail the Stage-Program/Kernel/Fusion branch this way
  (`efficientnet_b1/b2/b3_pruned`, `fbnetv3_g`, `regnetz_005/040/d8`,
  `repvit_m1_0`), disjoint from every operator/domain gap above (`ops.md`'s
  "Cross-model signal" section). It's a `Kernel_adapt` evaluation-depth
  ceiling, not a missing operation or a Native4D domain limit, so none of §1-2
  touches it. Noted here only so it isn't mistaken for one of these items if
  it shows up again; no owner assigned yet.
- **Grouped convolution and `requires_payloads`** — the 100-model sweep
  confirms both are widespread (13 and 11 models respectively) but both are
  already-closed design decisions (`.ai/native4d_design.md` for grouped
  convolution; the payload-precondition framing in `ops.md`'s Executive
  status for `requires_payloads`), not backlog items. Do not open work against
  either from this file.
