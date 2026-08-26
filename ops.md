# Operations support report

Status at 2026-08-23. This report covers the six release assets deliberately
tracked by this repository: `test_convnext2`, `mobilenetv2_050`, `csatv2`,
`regnetx_002`, `efficientnet_b0`, and `fastvit_sa12`. They are the union of
`PT2_MODELS_CRAM` and the graph-only pool fixture in `Makefile`; they are also
the six entries in `data/pt2-functional-manifest.json`.

The source of the inventory is each pinned producer `models/model.json`.
`Makefile` checks that this graph agrees with the graph in a downloaded release
archive, so it is a useful reproducible planning baseline. Counts below are
node occurrences, not a claim that a payload-free `model.json` can execute.

Priority calls below are now cross-checked against a second, broader data
source: `test/data/pt2_json_model_support.jsonl`, a payload-free sweep of all
~100 models in the pinned producer submodule (`bin/pt2_json_model_support.ml`,
`make pt2.json-model-support`; prose in `.ai/pt2_model_support.md`). It
answers a question the six-model table cannot: how many *distinct models*,
not just how many nodes in one graph, does each missing operator or domain
limit actually block. See "Cross-model signal from the 100-model payload-free
sweep" below; it is folded into the complexity tables and delivery order that
follow.

## Executive status

Native lowering already covers the normal convolutional core: convolution,
inference batch norm, adaptive average pool, linear, reshape/view, pointwise
tensor arithmetic, relayout, mean, ReLU/Hardtanh/SiLU, clone, unbind, layer
norm, and SDPA. This is enough for the *target sets* of MobileNetV2 and
RegNetX. Their payload-free Model Explorer sessions still cannot cross to
Native4D because batch-norm folding needs archive payloads; that is a payload
precondition, not an operation gap. The existing native verification cohort
therefore correctly stays `mobilenetv2_050` and `regnetx_002`.

Three small pointwise/configuration additions unlock disproportionately useful
coverage:

- `gelu.default` unlocks the first Native-lowering blocker in ConvNeXt2 and
  FastViT (4 and 19 nodes respectively; CSATv2 has 32).
- `sigmoid.default` unlocks the first blocker in EfficientNet-B0 (16 nodes) and
  is also used twice by FastViT.
- Scalar `mul.Tensor` needs a Native scalar-multiply form. CSATv2 has six
  scalar multiplications. Scalar `add.Tensor` is already legalised to
  `Add_scalar`, but must be regression-tested for CSATv2's 17 and FastViT's 17
  scalar adds.

After those changes, EfficientNet-B0 should be Native-lowerable; ConvNeXt2
should be Native-lowerable after GELU; FastViT still requires its attention
path to be representable in Native4D. CSATv2 remains a deliberately graph-only
fixture until the larger transformer/structural set is implemented.

## Design-goal audit: Native stays one node per ATen op

The Native dialect is meant to stay as close to ATen as the six-axis frame
allows: one Native `Graph_ir` node per ATen op it represents, never a fixed
subgraph of several nodes assembled to reuse another op's code. Where two ATen
targets are genuinely similar, the right move is to share the
*implementation* (a `Compute` functor or shape rule one op module calls from
another) while still giving each its own graph constructor — not to legalize
one target into a combination of other ops' nodes. This is now stated
explicitly in `.ai/native_add_op.md` ("Design goal: Native stays one node per
ATen op, no decomposition"), `.ai/native_aten_bridge_design.md`'s "Extending
coverage", and `.ai/native_graph_design.md`'s design-decisions list; this
report previously recommended the opposite for two of the rows below (§2,
"Legalize stack to reshape/unsqueeze plus concat"), which is corrected here.

Checked against the current tree (`lib/native_aten_bridge/op_bridge.ml`,
mirrored in `lib/native_interp/native_interp.ml`):

- **Compliant.** `unsqueeze.default` lowers to a single `Reshape` node and
  nothing else. This is not a decomposition: inserting a size-1 axis never
  changes the linearized data order, so in the always-6D frame it is the same
  no-op shape change `Reshape` already performs for `view.default` — one ATen
  target, one node, the sanctioned "delegate after translating params" case.
  Likewise `sub.Tensor`'s scalar form → `Add_scalar` with a negated scalar.
- **Deviation: `select.int`.** Lowers to *two* nodes, `Slice` then `Reshape`
  (`op_bridge.ml:1752-1792`). No node in the resulting graph names
  `select.int`; a graph consumer sees an ordinary slice-then-reshape pair.
  Fix: add a `Select` `Graph_ir` op whose `output_shape`/`Compute` functor
  calls `Slice`'s with a one-wide window on the selected axis and folds away
  that axis, the same computation the bridge currently spells out across two
  nodes, but owned by one.
- **Deviation: `stack.default`.** Lowers to *N+1* nodes, one `Reshape` per
  operand then `Concat` (`op_bridge.ml:1687-1738`). Same defect as above, at
  variadic arity. Fix: add a `Stack` `Graph_ir` op whose `Compute` functor
  calls `Concat`'s after deriving each operand's unsqueezed shape, exactly as
  the bridge does today, but as one node rather than `N+1`.

Both deviations landed under this report's own prior guidance (§2's `select`/
`unsqueeze`/`stack`/`cat` row, now corrected below) and are tracked as
follow-up work in `todo.md` rather than re-opening the whole CSATv2 slice —
the Native4D-side consequence is that neither op can be given its own
Native4D counterpart (§7 of `.ai/native4d_design.md` maps Native ops
op-for-op) until it exists as its own Native op.

## Per-model operation inventory and blocking frontier

| Model | Nodes | Native-covered targets/configurations | Missing or incomplete Native requirements | Native4D outlook |
| --- | ---: | --- | --- | --- |
| `mobilenetv2_050` | 152 | `conv2d`, inference BN, add, Hardtanh, adaptive pool, linear, view | None in this graph | Achievable from the weighted archive after canonical folding; depthwise convolution and Hardtanh are already in the dialect. |
| `regnetx_002` | 145 | `conv2d`, inference BN, add, clone, ReLU, adaptive pool, linear, view | None in this graph | Same payload/folding qualification as MobileNetV2. |
| `efficientnet_b0` | 239 | all other 223 nodes, including SiLU, mean, tensor multiply | `sigmoid.default` (16) | Add a direct Sigmoid4 counterpart; then the convolutional graph should be in scope after BN folding. |
| `test_convnext2` | 71 | conv, add, tensor multiply, clone, layer norm, linear, permute, view, adaptive pool | `gelu.default` (4) | GELU needs a direct Native4D counterpart; validate LayerNorm axes and all reshapes against the N/H/W/C domain. |
| `fastvit_sa12` | 332 | conv, BN, tensor arithmetic, relayout, mean, ReLU, linear, unbind, SDPA in Native | `gelu.default` (19), `sigmoid.default` (2), scalar `add.Tensor` (17) | Even after the small additions, SDPA is intentionally rejected by Native4D. It needs a dialect extension/legalization, not just an importer arm. |
| `csatv2` | 945 | conv, tensor arithmetic, div, clone (without memory-format), layer norm, linear, mean, relayout, SDPA, unbind, view | scalar add/mul configurations; GELU; plus the structural, reduction, indexing, resize, and matmul/softmax set below | Not a near-term Native4D conversion candidate. It combines N/H/W/C-incompatible attention with operations missing from both Native and Native4D. |

The current payload-free Model Explorer probe corroborates the first three
frontiers: ConvNeXt2 and FastViT stop at `gelu.default`, EfficientNet-B0 at
`sigmoid.default`, while MobileNetV2 and RegNetX import successfully. CSATv2
currently stops still earlier on a scalar `mul.Tensor` argument, hence the
importance of treating overload *configuration* as part of support.

## Cross-model signal from the 100-model payload-free sweep

`test/data/pt2_json_model_support.jsonl` (100 models, generated 2026-08-23).
50 models fail Native import outright (`native_builds:false`); of the 50 that
build, the Native4D and Stage-Program/Kernel/Fusion branches each reject 25%
and 16% respectively, on largely disjoint grounds. Ranked by how many
*distinct models* each cause blocks — the number that should drive sequencing
ahead of any single model's node count:

| Cause | Models blocked | Note |
| --- | ---: | --- |
| `cat.default` unsupported | 15 | Already planned in §2 below as a CSATv2 requirement (15 node occurrences in that one graph); this is independent evidence it is the single highest-leverage missing operator across the corpus, not a CSATv2-specific need. |
| `split_with_sizes.default` unsupported | 8 | Not previously tracked anywhere in this report. The natural inverse of `cat`/`stack`; a variadic-output legalization, not a new arithmetic kernel. |
| Grouped convolution (`groups` neither 1 nor depthwise) | 13 | `outside_dialect_domain`, all in Native4D. Confirms `regnetx_002`'s blocker (§ per-model table) at scale — the whole RegNetX/RegNetY/RegNetZ family plus `efficientnet_b0_g16_evos` and both `mobilenet_edgetpu_v2_*` variants share it. Still a deliberate four-axis domain limit per `.ai/native4d_design.md`, not a bug; not an action item. |
| Constant payload required at conversion time (`requires_payloads`) | 11 | Confirms the payload-precondition framing already used for MobileNetV2/RegNetX in the Executive status above, now at corpus scale (`convmixer_1024_20_ks9_p14`, `fastvit_t8`, both `mobileone_s*`, all four `repghostnet_*`, both `starnet_s*`, `xception41p`). |
| `group_norm.default` unsupported | 3 | Not previously tracked. A distinct normalization primitive from the existing (batch/layer) norms — grouped-channel statistics, not a legalization to either. |
| `clone.default` with `memory_format` | 3 (`malformed`) | Corroborates §2's existing "clone with a supplied memory format" row (CSATv2, 2 occurrences): the importer's refusal-over-silent-ignore choice there is exactly the "malformed" (hard reject, not "unsupported") classification these three additional models hit. |
| `max_pool2d.default` with `ceil_mode=true` | 2 (`malformed`) | Not previously tracked. Same shape-computation family as the already-covered pooling ops; ceiling instead of floor division when computing output extent. |
| `expand.default`, `split.Tensor`, `avg_pool2d.default` (non-adaptive), `_native_batch_norm_legit.no_stats` | 2 each | Not previously tracked. |
| `adaptive_max_pool2d.default`, `select.int`, `leaky_relu.default`, `conv1d.default`, `pow.Tensor_Scalar`, `sum.dim_IntList`, `upsample_bicubic2d.vec`, `zeros.default`, `linalg_vector_norm.default`, `arange.start`, `im2col.default` | 1 each | Singleton blockers; `select.int`, `pow.Tensor_Scalar`, and `linalg_vector_norm.default` already appear in §2/§3 below via CSATv2 evidence — these are corroborating, independent hits, not new work. `upsample_bicubic2d.vec` is a *different* overload from §2's planned `upsample_bilinear2d.vec` and would need its own coordinate-transform review, not an automatic extension of that slice. |

A separate, previously untracked failure mode: 8 of the 50 models that build
and pass Native4D (or don't need it) still fail the Stage-Program/Kernel/
Fusion branch with `over_limit`/"evaluation depth exceeds 2048"
(`efficientnet_b1_pruned`, `efficientnet_b2_pruned`, `efficientnet_b3_pruned`,
`fbnetv3_g`, `regnetz_005`, `regnetz_040`, `regnetz_d8`, `repvit_m1_0`). This
is a resource-limit question in `Kernel_adapt` — larger/deeper graphs
exceeding a fixed evaluation-depth ceiling — not a missing operator or a
Native4D domain limit; it is disjoint from every row above (e.g. `regnetz_005`
fails Native4D on grouped convolution *and* Kernel on depth, for unrelated
reasons). It is out of scope for the operator-coverage work below and is
tracked here only so it isn't mistaken for one of these rows if it recurs.

## Work grouped by implementation complexity

### 1. Low: direct pointwise and existing-IR completion

These operations have no new topology and should be added end-to-end as a
single vertical slice per operation:

| Requirement | Models / occurrences | Native work | Native4D and walk work |
| --- | --- | --- | --- |
| `gelu.default` | ConvNeXt2 4; FastViT 19; CSATv2 32 | The ATen binding already exists. Add a `Gelu` payload/IR op, builder, shape rule, direct and symbolic evaluation, kernel elaboration, serialization/printing, output-transfer classification, bridge arm, and operation tests. | Add `Gelu` to `Native4d.Op`, shape/eval/codec/transfer and lowering. A generated ATen walk already exists; add Native Direct-vs-Symbolic and Native4D conversion/eval cases. |
| `sigmoid.default` | EfficientNet 16; FastViT 2 | Binding already exists; same vertical slice as GELU, with a `Sigmoid` op. | Same direct counterpart in Native4D. The generated default walk already exists; add native and conversion coverage. |
| scalar `mul.Tensor` | CSATv2 5 float + 1 int configuration | Reuse `tensor_or_scalar` in the bridge and add `Mul_scalar`, mirroring `Add_scalar`/`Div_scalar`; do not silently materialize a scalar tensor. | Add `Mul_scalar` as a direct Native4D counterpart. Add a scalar-bearing walk recipe because the generated multi-argument target is currently `needs_meta`. |
| scalar `add.Tensor` regression coverage | CSATv2 15 float + 2 int; FastViT 17 int | No new operation: the bridge already maps it to `Add_scalar`. Exercise both integer and float literals at full graph and node level. | Already has `Add_scalar`; conversion and walk tests should prove the configuration remains supported. |

Exit criterion: run the `native_graph visualize` probe over all six source
graphs; EfficientNet must no longer be rejected for Sigmoid, and ConvNeXt/FastViT
must no longer be rejected for GELU. The target-specific node test must compare
ATen against Native and Native Direct against Symbolic; a successful importer
alone is insufficient. The same slice must also meet the ops-walk and fuzz gate
below; a hand-written happy-path fixture is not a replacement for varying the
supported configuration space.

### 2. Medium: legalizations and bounded new operations

These additions are feasible without changing the basic six-axis Native frame,
but each introduces a new shape/dataflow case or a non-pointwise algorithm.
They should be independent vertical slices, in this order.

| Requirement | Model evidence | Recommended implementation | Native4D implication |
| --- | --- | --- | --- |
| `select.int` (6) and `unsqueeze.default` (14) | CSATv2 | `unsqueeze` legalizes to `Reshape` alone (one node, no decomposition — see the design-goal audit above). `select` needs its own `Select` op, one node, whose shape/`Compute` calls into `Slice`'s after folding away the selected axis — **not** a bridge-level `Slice`-then-`Reshape` pair. Validate negative dimensions and rank restoration against ATen. | No new primitive if every generated reshape satisfies `Shape4`; otherwise produce a domain diagnostic. `Select` gets a direct `Select4` counterpart once it exists in Native. |
| `cat.default` (15 nodes in CSATv2; also the single most-blocking missing operator across the 100-model sweep — see above, 15 *distinct models*) and `stack.default` (1) | CSATv2 plus 15 other models corpus-wide | Add a variadic `Concat` Native op with axis-aware shape/evaluation (done). `stack` needs its own `Stack` op, one node, whose `Compute` calls into `Concat`'s after deriving each operand's unsqueezed shape — **not** a bridge-level per-operand `Reshape` chain into `Concat` (see the design-goal audit above; this row previously recommended exactly that decomposition) (done — `select.int`/`stack.default` landed 2026-08-25, `d915b62`). | `Concat4` landed (2026-08-25): domain rule (`Domain.check_node`'s `Concat` arm gates the joined axis via `check_dims`, the same shape `Slice`/`Unbind` get), closed-op registration, builder, shape/eval delegating to Native's `Concat.Concat`, JSON, `Reindexing` output-transfer claim, and an `Identical` `Concat -> Concat4` lowering arm — see `test/native4d/{op_json_test,fixtures4,compute_test,domain_test,lower_test,verify_test}.ml`. Confirmed by the regenerated 100-model sweep: 9 more models now have `native4d_converts:true` (`rdnet_tiny`, 8 `rexnet*`/`rexnetr*`). `Stack` still needs the analogous `Stack4`; not started. |
| `split_with_sizes.default` — not present in CSATv2, but blocks 8 other models corpus-wide (see sweep above) | 8 models, none currently tracked here | The variadic inverse of `cat`: split a tensor into pieces of given sizes along an axis. Natural to land alongside the `cat`/`stack` slice since both are axis-aware variadic reshapes of the same data-movement kind. **Landed 2026-08-25.** `Graph_ir.Split_with_sizes` (own module in `lib/native/ops/split.ml`, alongside `Unbind`): `params = {axis; sizes}`, `output_shapes` checks every size is positive and the sizes sum exactly to the axis's extent (own `Shape_error.Split_with_sizes` fault, reusing `Output_count_over_limit` for the list-length ceiling); `Compute.pixel` is `Slice`'s per-window read with a caller-supplied `offset`. ATen binding added (`bin/aten_ops_gen.ml: op "split_with_sizes"`), bridge arms in both `Op_bridge` and `Native_interp`, dtype preserved through a `Tensor.split_with_sizes` bypass in `Eval_direct` (same treatment `Unbind`/`Tensor.unbind` get, now sharing a `copy_cells` helper) rather than routing through the f32-only generic `Compute` path. Verified against real ATen via `Interp_verify` for both f32 and int64. Confirmed by the regenerated 100-model sweep: 7 more models now have `native_builds:true` (`inception_next_atto`, `inception_next_tiny`, `mixnet_l`, `mixnet_xl`, `mixnet_xxl`, `tf_mixnet_l`, `tf_mixnet_s`; the 8th, `lambda_resnet26t`, now stops on `softmax.int`). | Add a `Split4` counterpart now that `Concat4`'s axis/data-movement handling exists as the template; same boundary-op caveat as `Concat4`. Not started — all 7 newly-building models above stop at Native4D's domain check (`outside_dialect_domain`, "no legalization for split_with_sizes"). |
| `group_norm.default` — not present in CSATv2; blocks 3 models corpus-wide | 3 models, none currently tracked here | A distinct normalization primitive (grouped-channel statistics) from the existing batch/layer norms — not a legalization to either. Needs its own shape rule (channel groups must divide channel count) and reduction. **Landed 2026-08-25.** `Norm.GroupNorm` (`lib/native/ops/norm.ml`): windowed `S.sum` over one group's channel slice nested inside a full-extent sum over every other non-batch axis (the engine's first non-compile-time-constant `S.sum` lower bound, derived via `index_floor_div_pos`/`index_scale` from the output pixel's own channel index); `output_shape` checks `num_groups` divides the channel count and bounds the reduction count the same way `LayerNorm`/`RmsNorm` do; per-channel affine (not per-normalized-shape like `LayerNorm`'s) checked via `check_affine`, reusing `normalized_shape` with a single-axis `dims`. ATen binding, bridge (`Op_bridge`) and importer (`Native_interp`) arms all landed, both permuting NCHW⇄NHWC around the op like `Batch_norm`'s. **No real ATen C binding** (`bin/aten_ops_gen.ml` does not select `group_norm`): `at::native::group_norm` sits outside the hand-curated source closure `lib/aten/build_archive.sh` compiles, and adding it undefined-symbol'd every binary linking `aten`; Native import needs no ATen binding, so `native_builds` is unaffected, but `Interp_verify` real-ATen verification is unavailable for this op — correctness is instead pinned by hand-computed values (`test/native/compute_test.ml`) cross-checked against an independent calculation, plus the mandatory Direct-vs-Symbolic bitwise-agreement test. Confirmed by the regenerated 100-model sweep: the 3 predicted models (`efficientnet_b0_g8_gn`, `efficientnet_b3_g8_gn`, `test_efficientnet_gn`) all now have `native_builds:true`. | Needs a direct Native4D counterpart with its own axis-domain check; do not conflate with `Batch_norm`/`Layer_norm`. Not started — all 3 newly-building models above stop at Native4D's domain check (`outside_dialect_domain`, "no legalization for group_norm"). |
| `amax.default` (14), restricted to dim `[1]`, `keepdim=true` | CSATv2 | Add a maximum keep-dim reduction. Its exact observed configuration is a good first contract; general empty/None dims can wait. **Landed 2026-08-26.** `Reduce.Amax` (`lib/native/ops/reduce.ml`), factored beside `Mean` behind a new `Dims_keepdim` helper (params/JSON/`kept_map`/`output_shape` shared, per the design-goal audit's "share the implementation, not the node") rather than each op restating its own copy; `Amax.Compute` nests one `S.max_reduce` per reduced axis, the same shape `Mean`'s `S.sum` nest uses, with no divisor. The refactor also closed a latent gap in `Mean.output_shape`, which had never bounded its reduction count via `Vec6.numel_bounded` (CLAUDE.md's 32-bit-aggregate rule, already followed by `Norm.normalized_count`) — `Dims_keepdim.output_shape` now does, for both ops. ATen binding (`bin/aten_ops_gen.ml: op "amax"`), bridge (`Op_bridge`) and importer (`Native_interp`) arms reuse `mean.dim`'s own general `dims_arg`/`axes_for_rank` machinery, so general `dims`/`keepdim` — not just the single observed configuration — is supported for the same cost. | Needs a direct Native4D counterpart with its own axis-domain check; do not conflate with `Mean_keepdims`. **Landed 2026-08-26.** `Ops4.Max_keepdims`, the elementwise-maximum twin of `Ops4.Mean_keepdims` (keep-dimensions-only, per `.ai/native4d_design.md` §1). `Domain.check_node`'s new `Amax` arm runs `check_dims` on the reduced axes, the same treatment `Mean`/`Slice`/`Unbind` get; `lower.ml`'s `Amax` arm legalizes `keepdim=true` directly and `keepdim=false` to `Max_keepdims` + `Reshape4`, textually mirroring `Mean`'s own correction-C1 legalization. Verified against real ATen via the generated `Amax_walk` (`lib/aten_gen/walk_meta.ml`, reusing `Recipe_reduce` — "mean.dim and friends" — unchanged, confirming the recipe's own reuse premise) and a Native Direct-vs-Symbolic fuzz walk (`lib/native_op_walk/amax_nwalk.ml`); hand-derived `dispatch:` fixtures (`test/native_bridge_test.ml`) and pinned `Compute` values (`test/native/compute_test.ml`) round it out, plus the mandatory per-op Native4D coverage (`test/native4d/{op_json_test,fixtures4,compute_test}.ml`). The regenerated 100-model sweep shows no movement: `amax.default` blocks no model in that corpus — its only known occurrence remains CSATv2's own graph. |
| `pow.Tensor_Scalar` (1), exponent `0.5`; `linalg_vector_norm.default` (14), ord 2/dims `[1,2]`/keepdim | CSATv2 | Prefer a small, explicit numeric kernel (`Sqrt` plus multiply/reduction where the order is demonstrably compatible) only if it matches ATen tolerance. Otherwise add fused `Pow_scalar` and `Vector_norm` to preserve numerical semantics. **Landed 2026-08-26, `pow.Tensor_Scalar` in two stages.** First, exponent-0.5-only, legalized directly to the existing `Sqrt` node (`pow(x,0.5) = sqrt(x)` exactly, once translated — the same "map onto an existing op" legalization `sub.Tensor`'s scalar form gets). The 100-model sweep immediately showed this was too narrow — `mobilenetv5_base` has a *second*, independent `pow.Tensor_Scalar` occurrence at exponent 2 — so the same session generalized it to a real `Pow_scalar`: one Native op (`Pointwise.Pow`) whose `Compute` special-cases the exact six exponents ATen's own `PowKernel.cpp` does (`0.5`/`-0.5`/`-1`/`2`/`3`/`-2`, via sqrt/reciprocal/`mul` combinations — an OCaml-level `if` evaluated once per graph node, since the exponent is a decode-verified compile-time constant) and falls back to `exp(exponent * log x)` otherwise (only `Equivalent` to ATen, not bit-identical, and valid only for `x > 0`, both documented rather than assumed). This needed one new `SEMANTICS` primitive, `log`, threaded through `Expr.Value`/`Direct`/`Symbolic` the same four sites `erf` required. `linalg_vector_norm.default` (ord=2 only) got its own `Reduce.Vector_norm` (`lib/native/ops/reduce.ml`), the third op sharing `Dims_keepdim` alongside `Mean`/`Amax`: no new `SEMANTICS` primitive needed there, since the leaf squares the loaded value (`S.mul v v`) under one `S.sum` nest per axis with a final `S.sqrt`, in place of `Mean`'s division. Both `ord` (rejected unless 2) and `dtype` (rejected unless absent) are decoded and rejected rather than dropped, via a new `reject_dtype` importer helper mirroring `reject_memory_format`. Verified against real ATen via generated walks — `Pow_tensor_scalar_walk` now draws from a candidate list covering the whole special-cased exponent set plus one generic exponent (1.5, exercising the fallback), over a POSITIVE-only base (`Walk.tensor_spec_positive`) since several candidates are singular/NaN at a non-positive base; `Linalg_vector_norm_walk` reuses `Recipe_reduce` and pins `ord=2` — both "matched" on every candidate. | Needs reduction primitives beyond `Mean_keepdims`; do not call this a simple pointwise extension. **Landed 2026-08-26.** `Ops4.Vector_norm_keepdims`, the third op (with `Mean_keepdims`/`Max_keepdims`) sharing the keep-dimensions-only reduction shape; same axis-domain check and keepdim=false-to-`+Reshape4` legalization. The general `Pow` is reused directly in Native4D (`Op.Pow of Pointwise.Pow.t`, alongside `Add_scalar`/`Mul_scalar`/`Div_scalar` — no axis semantics, no wrapper payload) with matching domain/shape/eval/lower arms; a new Native fuzz walk (`pow_nwalk.ml`, via a new `Native_tensor.synth_positive`) reuses `Pointwise.Scalar_bin.Walk`'s shared config space, and `Pow` was added to `Sink_permute.elementwise`. Confirmed by the regenerated 100-model sweep at each stage: against exponent-0.5-only, `mobilenetv5_base`'s diagnostic sharpened from "unsupported PT2 operator" to a `malformed`-graph rejection naming the actual value (`exponent=2 is not supported (only 0.5)`); after generalizing, that model clears `pow.Tensor_Scalar` entirely and now stops on `rsqrt.default` — a distinct, currently-unbound op (notably the same computation `Pow`'s own `-0.5` case already does internally) and a natural next lead. `swiftformer_xs` advances past `linalg_vector_norm.default` entirely and now stops on `clamp_min.default` — a previously-untracked, currently-unbound op and a natural next lead. |
| `upsample_bilinear2d.vec` (14), explicit output sizes and `align_corners=true` | CSATv2 | Add a real bilinear-resize op: shape inference, coordinate transform, boundary handling, direct/symbolic kernel, and targeted oracle tests. | A natural `ResizeBilinear4` can operate on H/W/C, but must reject/diagnose unsupported axes rather than reinterpret them. |
| clone with a supplied memory format (2) | CSATv2 | Decode the exact memory-format enum and either prove it is an identity for the observed layout or represent the required relayout. The current bridge correctly refuses a supplied value rather than ignoring it. | Only admit a no-op proof; a requested layout conversion is not automatically a `Clone` in four-axis form. |

Each row needs an ATen binding in `bin/aten_ops_gen.ml` if it is absent
(`select` and `unsqueeze` are already bound; the other new targets are not),
an `Aten_op_config` decoder/spec fixture, bridge lowering, Native implementation,
and a complete dispatch audit. Add an ATen-vs-Native walk only when a
non-vacuous parameter recipe is available; otherwise keep a table-driven
boundary suite until such a recipe is written.

### 3. High: matrix/attention and indexing semantics

These are the CSATv2 requirements that alter the computation model or make
Native4D's intentionally reduced domain insufficient.

| Requirement | Model evidence | Why it is high complexity | Decision / prerequisite |
| --- | --- | --- | --- |
| `matmul.default` (28) plus `softmax.int` (14, dim -1) | CSATv2's decomposed attention blocks | Native has `Bmm`, not general broadcasted/rank-polymorphic `Matmul`; Native4D only legalizes a single-batch BMM to convolution and has no softmax. The source uses this pair as attention, so independent simplistic implementations risk a later redesign. | First specify a restricted rank/layout subset from the captured graphs. Then choose either (a) Native `Matmul` + `Softmax` and a richer Native4D matrix dialect, or (b) recognize and legalize the complete attention subgraph. Do not add only one half. |
| `scaled_dot_product_attention.default` (CSATv2 4; FastViT 2) | Native already has a fused operation | Native4D deliberately rejects SDPA: it names Native's `D` batch axis and the dialect has neither BMM/MatMul nor softmax. This is an explicit domain decision, not an unimplemented bridge arm. | Decide whether Native4D remains a CNN dialect. If conversion of FastViT/CSATv2 is required, add a first-class attention/matrix sub-dialect (with its axis vocabulary and verification), or a proven lowering to it. |
| `index.Tensor` (1) | CSATv2 uses an optional-tensor index list with only the final index live | Advanced indexing is gather semantics, not a view/slice. It brings index dtype, bounds, broadcasting, and output-shape rules that the current tensor/frame model avoids. | Start with a narrowly typed single-axis gather contract only if it unblocks the model; otherwise keep CSATv2 graph-only. It should not be bundled with the scalar/activation work. |
| `addcmul.default` (14) | CSATv2 | Semantically simple (`self + value * tensor1 * tensor2`), but adding it as a fused op creates another numeric/serialization/walk surface. | Prefer a bridge decomposition to existing `Mul` + `Add` when the default `value=1` is verified; use `Mul_scalar` for a non-unit value. This is low in arithmetic but scheduled here because it depends on the scalar completion and must preserve rounding/order claims. |

## Whole-stack definition of “supported”

An operation is not supported merely because its target is recognized. For every new
row, complete the applicable layers below, then record which levels are
deliberately out of domain:

1. **ATen surface:** curated binding (`bin/aten_ops_gen.ml`), generated
   operation description/config/spec, and an exact PT2 argument decoder.
2. **Import:** `Native_aten_bridge.Op_bridge` plus the archive importer
   (`Native_interp`) must lower the same target and configuration to the same
   Native graph. Check scalar literals, optionals, defaults, tuple/list arity,
   dtype, and layout before construction.
3. **Native:** typed payload and `Graph_ir` constructor; builder; shape;
   direct, symbolic, and kernel paths; JSON/printer; operand mapping;
   transform/output-transfer exhaustiveness; and graph-level payload execution.
4. **Native4D:** domain rule first, then closed-op registration, builder/shape,
   direct/symbolic evaluation, JSON, conversion map and output-transfer claim.
   A rejection should remain a typed `outside_dialect_domain` result when the
   issue is the dialect boundary.
5. **Verification:** one real-node fixture or captured-model test, one focused
   ATen-vs-Native property/walk where a meaningful recipe exists, Native
   Direct-vs-Symbolic coverage, and Native-to-Native4D map verification where
   conversion is promised. Add a negative test for every rejected option.

The generated ATen walk has recipes today for only a subset of the curated
surface (the ResNet-oriented cram exercises convolution, ReLU,
max-pool-with-indices, and mean). “No walk recipe” is therefore a coverage
backlog, not evidence that a target is unsupported. Conversely, do not add a
default-only walk for multi-input operations such as matmul or concat: it would
not exercise their defining relationship.

## Ops-walk and fuzz coverage

Every newly supported operation and overload configuration must be accounted for
by the two existing, seedable walk systems. These are complementary to the
focused oracle and serialized-import tests above:

| Layer | Harness and oracle | Required coverage |
| --- | --- | --- |
| ATen bridge | Generated `Aten_op_walk` recipe, run by `native_walk_run` / `pt2_op_native_walk_cram.t`; oracle is ATen versus Native. | Fuzz the arguments, shapes, and values that the bridge claims to accept. A recipe must retain operand relationships by construction and use a cascade for dependent parameters; invalid draws and skipped results do not count as coverage. |
| Native implementation | Hand-written `Native_op_walk` module under `lib/native/ops`, bounded by `lib/native_op_walk/walk_limits.ml`; oracle is Native Direct versus Symbolic. | Fuzz each new Native primitive over valid shapes and parameters, including its scalar/value form where applicable. This catches disagreements between the two evaluation paths even where both have passed an ATen fixture. |

The CI walks use fixed PCG seeds and modest limits so failures reproduce in a
golden or CRAM test. A deliberate heavier-fuzz run must use additional recorded
seeds and/or more steps while retaining the same validity cascade; its tensor
budget is controlled centrally by `Walk_core.Limits` and
`lib/native_op_walk/walk_limits.ml`. Never turn fuzzing into an unbounded CI
job, and always print the seed, target, step, and final configuration (the
current walkers already print the latter three) so a mismatch becomes a focused
regression fixture.

For this delivery, the minimum ops-walk matrix is:

| Support claim | ATen-vs-Native ops-walk | Native Direct-vs-Symbolic fuzz walk |
| --- | --- | --- |
| `gelu.default`, `approximate="none"` | Generated one-tensor/default recipe; vary shape and tensor values. Keep table-driven negative coverage for omitted, wrong-kind, and `"tanh"` `approximate` values. | Add the `Gelu` walk and include near-zero, signed-zero, and large finite inputs in the deterministic boundary companion. |
| `sigmoid.default` | Generated one-tensor/default recipe; vary shape and values. | Add the `Sigmoid` walk, with saturation boundary examples in its focused companion. |
| scalar `mul.Tensor` and scalar `add.Tensor` | Add an explicit scalar-bearing `walk_meta` recipe rather than leaving the targets in `Aten_op_walk.needs_meta`. It must vary both integer and float literals, scalar position/spelling accepted by the decoder, and broadcast-valid tensor shapes. If `mul.Scalar` is supported, give that distinct generated target the same treatment. | Add `Mul_scalar` and exercise the existing `Add_scalar` path; vary integer and non-f32-exact float literals as well as tensor shape. |

The all-target coverage sweep must continue to print `Aten_op_walk.needs_meta`.
Treat every new entry as an explicit backlog item with an owner and reason. A
target may remain there only when its support claim is limited to table-driven
fixtures; it must not be described as fuzz-covered or end-to-end verified until
a non-vacuous, correlated recipe is added. For multi-operand operations such as
concat, matmul, attention, and resize, the recipe must vary the relationship
that defines the operation (axis/list arity, compatible matrix dimensions,
mask/layout, or coordinate policy), not merely input fill values.

### Current-source audit and coverage holes

Audit of the current source registries (`Aten_op_walk`, `Native_op_walk`, and
`Native_aten_bridge.Op_bridge`) finds that the walk framework is sound, but its
coverage is incomplete. “Walk emitted” means only that a generated spec can be
run; it is not proof that the bridge lowers it. In particular, generated GELU
and Sigmoid walks currently exercise an absent Native implementation and cannot
be counted as support coverage until their bridge and Native operations land.

| Scope | Covered today | Hole that must remain explicit |
| --- | --- | --- |
| ATen-vs-Native ops-walk | Single-tensor and curated-meta targets include the convolution/pool families; ReLU, SiLU, Hardtanh, Hardsigmoid, and Hardswish; clamp; views/transposes; norms; SDPA; pad/slice/unbind; mean; clone; and tensor `sub`. | `Aten_op_walk.needs_meta` currently contains 25 selected targets. Of the targets the bridge already lowers, `add.Tensor`/`add_.Tensor`, `mul.Tensor`, `div.Tensor`, `permute.default`, `_native_batch_norm_legit_no_training.default`, `addmm.default`, and `bmm.default` have no ATen-vs-Native fuzz recipe. Scalar forms accepted through `add.Tensor`, `sub.Tensor`, and `div.Tensor` are also not varied by the existing tensor-only/default recipes. |
| Native Direct-vs-Symbolic fuzz | Registered Native walks cover the existing convolution, pooling (except adaptive average), norms, SDPA, reshape/permute, clamp/clone, add/sub/mul, scalar add/div, activations, mean, pad/slice, and unbind primitives. | The current Native registry has no walk for `Adaptive_avg_pool2d`, `Bmm`, tensor `Div`, or `Sqrt`, despite bridge arms for those primitives. `Discard` is an internal graph-plumbing node and is deliberately excluded; it needs graph-level rather than standalone-op coverage. |
| Binding versus implementation state | `gelu.default`, `sigmoid.default`, and `mul.Scalar` are present in the curated ATen binding/walk selection, making their absence visible. | They are not currently bridge-lowered as supported Native operations (`Graph_ir` has no GELU, Sigmoid, or scalar-Mul primitive). Their generated walk results are therefore a missing-implementation signal, not a green fuzz gate. `sqrt.default` has a bridge arm but is absent from the curated ATen walk selection, so it has neither an ATen-vs-Native recipe nor a Native fuzz walk. |

Close the holes in this order: first add correlated ATen recipes for each
already-lowered target in `needs_meta` (starting with scalar add/sub/div and
tensor add/mul/div); then register Direct-vs-Symbolic walks for adaptive
average pool, BMM, tensor division, and sqrt; finally add the planned GELU,
Sigmoid, and scalar-Mul vertical slices together with both walk layers. Keep
focused table-driven tests for rejected options and boundary values: fuzzing
valid configurations complements those tests and cannot replace them.

## Recommended delivery order

1. Land GELU, Sigmoid, and scalar multiply as separate vertical slices, while
   pinning scalar-add configuration coverage. This makes EfficientNet-B0
   Native-ready and removes the early false frontiers from ConvNeXt2/FastViT.
2. Run weighted-archive Native/Native4D verification for MobileNetV2, RegNetX,
   EfficientNet-B0, and ConvNeXt2. Use failures to distinguish missing
   operations from the four-axis domain.
3. Implement CSATv2's structural set (`select`/`unsqueeze`, then concat/stack —
   land `split_with_sizes` and `group_norm` alongside concat/stack even though
   neither appears in CSATv2, since the 100-model sweep shows they are
   independently high-leverage — see "Cross-model signal" above), followed by
   the constrained reductions and bilinear resize. Maintain the model's
   graph-only CI role until a complete end-to-end execution test passes.
4. Write the matrix/attention dialect decision before adding general matmul,
   softmax, or Native4D SDPA. That decision determines whether FastViT and
   CSATv2 are Native4D goals or intentionally Native-only/graph-explorer models.
5. Before calling any slice complete, run its focused bridge/import tests, its
   ATen-vs-Native ops-walk CRAM coverage, and its bounded Native
   Direct-vs-Symbolic fuzz walk. Promote a minimized failing seed/configuration
   to a deterministic regression fixture, then retain the seed in the fuzz
   corpus.

This order prioritizes broad model coverage without turning Native4D into an
accidental, unreviewed general tensor dialect.
