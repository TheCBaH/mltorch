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
cross-cutting note below); not fixed in either of these changes. **And
`group_norm.default`** (formerly item 1 below) — `Norm.GroupNorm` in
`lib/native/ops/norm.ml`: its own reduction, a windowed `S.sum` over one
group's channel slice (the engine's first non-compile-time-constant `S.sum`
lower bound) nested inside a full-extent sum over every other non-batch
axis, and its own shape rule (channel count must divide `num_groups`,
bounded the same way `LayerNorm`/`RmsNorm`'s reduction counts are);
per-channel (not per-normalized-shape) affine, checked via `check_affine`.
ATen binding, `Op_bridge` and `Native_interp` arms landed, both permuting
NCHW⇄NHWC around the op like `Batch_norm`'s. **No real ATen C binding**:
`at::native::group_norm` sits outside `lib/aten/build_archive.sh`'s
hand-curated source closure, and adding it undefined-symbol'd every binary
linking `aten` — `bin/aten_ops_gen.ml` does not select it, so
`Interp_verify` real-ATen verification is unavailable for this op; Native
import needs no ATen binding at all, so `native_builds` is unaffected, and
correctness is instead pinned by hand-computed values in
`test/native/compute_test.ml`, cross-checked against an independent
calculation, plus the Direct-vs-Symbolic bitwise-agreement test. Native4D
`Group_norm4` is the deliberately-not-done half, same `unsupported()`
treatment `Select4`/`Split4`/`Stack4` get. Regenerating the sweep after this
landing moved the last 3 models with this blocker to `native_builds:true`
(`efficientnet_b0_g8_gn`, `efficientnet_b3_g8_gn`, `test_efficientnet_gn`),
all now stopping at Native4D's domain check instead. See
`.ai/pt2_model_support.md`'s 2026-08-25 update for all three landings.

**And `amax.default`** (formerly item 1 below) — `Reduce.Amax` in
`lib/native/ops/reduce.ml`, alongside `Mean`: both now share their
axis-list/`keepdim` shape rule through a new `Dims_keepdim` helper (params
record, JSON codec, `kept_map`, `output_shape`) rather than each restating it,
per the design goal's "share the implementation, not the node". Factoring
`Dims_keepdim.output_shape` also closed a latent gap in `Mean`'s own rule: it
now bounds the reduction count via `Vec6.numel_bounded` before folding it (the
same 32-bit-aggregate rule `Norm.normalized_count` already followed —
CLAUDE.md's aggregate rule — which `Mean.output_shape` had never applied).
`Amax.Compute` nests one `S.max_reduce` per reduced axis (mirroring `Mean`'s
`S.sum` nest) with no divisor. ATen binding (`bin/aten_ops_gen.ml: op "amax"`),
bridge (`Op_bridge`) and importer (`Native_interp`) arms both reuse the same
general `dims_arg`/`axes_for_rank` machinery `mean.dim` already has — general
`dims`/`keepdim` is supported directly (not artificially restricted to the
single observed configuration), since it costs nothing beyond what `Mean`
already does. Native4D gets `Max_keepdims` (`Ops4.Max_keepdims`), the
elementwise-maximum twin of `Mean_keepdims`: axis-domain checking in
`Domain.check_node`'s new `Amax` arm (`check_dims`, same treatment
`Mean`/`Slice`/`Unbind` get), and a keepdim=false legalization to
`Max_keepdims` + `Reshape4` in `lower.ml`, textually mirroring `Mean`'s own
correction-C1 legalization. Verified against real ATen via the generated
`Amax_walk` (`lib/aten_gen/walk_meta.ml`, reusing `Recipe_reduce` — "mean.dim
and friends" — unchanged); a Native Direct-vs-Symbolic fuzz walk
(`lib/native_op_walk/amax_nwalk.ml`); hand-derived `dispatch:` fixtures in
`test/native_bridge_test.ml` (keepdim true/false, empty/omitted `dim`
reducing over all axes, out-of-range `dim` rejection); pinned values in
`test/native/compute_test.ml`; and the mandatory per-op Native4D coverage
(`test/native4d/{op_json_test,fixtures4,compute_test}.ml`). Regenerating the
100-model sweep after this landing shows no movement: `amax.default` blocks no
model in that corpus (its only known occurrence remains CSATv2's own 945-node
graph, which stays graph-only for unrelated reasons — see §2 below).

**And `pow.Tensor_Scalar` + `linalg_vector_norm.default`** (formerly item 1
below) — two independent landings sharing one commit since both were the
same todo row. `pow.Tensor_Scalar` shipped twice in the same session:
initially restricted to the observed exponent 0.5, legalized to the existing
`Sqrt` node (`pow(x, 0.5) = sqrt(x)` exactly once translated — the "map onto
an existing op" legalization `sub.Tensor`'s scalar form already gets), then
**generalized to every exponent** the same session, once the 100-model sweep
showed a second real model (`mobilenetv5_base`) needing exponent 2 — a
`Validation_failure` for anything but 0.5 would have been a second artificial
restriction discovered the moment the first one was tested against the
corpus, not a real scope boundary. The general form is one Native op,
`Pointwise.Pow` (`lib/native/ops/pointwise.ml`), whose `Compute` special-cases
exactly the six exponents ATen's own `PowKernel.cpp` special-cases —
`0.5`/`-0.5`/`-1`/`2`/`3`/`-2` via `sqrt`/reciprocal-of-sqrt/reciprocal/`mul`
combinations, matched in an OCaml `if` chain evaluated once per graph node
(the exponent is a compile-time constant: `D.scalar_arg`/`required_scalar_arg`
only decode `Argument.Int`/`Argument.Float`, rejecting `Argument.Sym_float`
and everything else as a typed decode error, and a tensor-derived exponent
would trace to the different `pow.Tensor_Tensor` target this arm does not
handle) — and falls back to `exp(exponent * log x)` for anything else, adding
one new `SEMANTICS` primitive (`log`, threaded through `Expr.Value`/`Direct`/
`Symbolic` the same four sites `erf` required). The fallback is only
`Equivalent` to ATen's `std::pow`, not bit-identical, and (like `Sqrt`)
correct only for `x > 0` — both documented in `Pow`'s own comment rather than
asserted implicitly. Reused directly in Native4D (`Ops4.op.ml`'s `Pow of
Pointwise.Pow.t`, alongside `Add_scalar`/`Mul_scalar`/`Div_scalar` — no axis
semantics, so no wrapper payload needed) with matching domain/shape/eval/
lower/output-transfer arms. The ATen walk (`Pow_tensor_scalar_walk`) now
draws from a candidate list covering the whole special-cased set plus one
generic exponent (1.5, exercising the fallback), over a POSITIVE-only base
(`Walk.tensor_spec_positive`, not `tensor_spec`) since several candidates are
singular or NaN-producing at a non-positive base — mirroring
`Recipe_reduce`'s siblings and `Sqrt`'s own walk's existing domain-hazard
avoidance; a new Native fuzz walk (`lib/native_op_walk/pow_nwalk.ml`, via a
new `Native_tensor.synth_positive`) reuses `Pointwise.Scalar_bin.Walk`'s
shared shape/scalar config space, the same one `Add_scalar`/`Div_scalar`/
`Mul_scalar` already reuse. Both walks are "matched"/"direct==symbolic" for
every candidate, including the generic-exponent fallback case. Added to
`Sink_permute.elementwise` (a fixed-exponent `Pow` reads and writes each
output slot independently, like `Sqrt`/`Add_scalar`), confirmed by extending
the `sink_permute_allowlist` fixture. The now-obsolete `Unsupported_option`
`` `Pow_exponent `` rejection tag (native_interp.ml) was deleted rather than
left dead, per CLAUDE.md.
`linalg_vector_norm.default` (ord=2 only — the schema default; a general
`ord` needs a `pow` of its own and stays out of scope) is the genuinely new
op: `Reduce.Vector_norm` in `lib/native/ops/reduce.ml`, the third op sharing
`Dims_keepdim` alongside `Mean`/`Amax`. Its `Compute` needs no new `SEMANTICS`
primitive — the leaf squares the loaded value (`S.mul v v`) under one `S.sum`
nest per reduced axis (mirroring `Mean`'s nest), with a final `S.sqrt` in
place of a division. ATen binding (`bin/aten_ops_gen.ml: op "linalg_vector_norm"`),
bridge and importer arms decode-and-reject a non-2 `ord` and a present
`dtype` (the latter via a new `reject_dtype` importer helper mirroring
`reject_memory_format`, and `Interp_decode.scalar_type_opt_arg_result` on the
bridge side) rather than silently dropping either, per
`.ai/native_add_op.md`'s "decode every argument" rule. Native4D gets
`Vector_norm_keepdims` (`Ops4.Vector_norm_keepdims`), the third op sharing the
`Mean_keepdims`/`Max_keepdims` keep-dimensions-only shape, with the same
axis-domain check and keepdim=false-to-`+Reshape4` legalization. Verified
against real ATen via generated walks (`Pow_tensor_scalar_walk`, pinning the
scalar to exactly 0.5 rather than drawing it, since any other value is a
rejection not a config to explore; `Linalg_vector_norm_walk`, reusing
`Recipe_reduce` and pinning `ord=2` the same way) — both "matched", confirming
the straightforward sum-of-squares-then-sqrt implementation is within ATen's
tolerance without needing a scaled/stable reformulation; a Native
Direct-vs-Symbolic fuzz walk (`lib/native_op_walk/vector_norm_nwalk.ml`);
hand-derived `dispatch:` fixtures (`test/native_bridge_test.ml`, covering
`pow`'s special-cased exponents and its generic fallback, plus the `ord`/
`dtype` rejections); pinned `Compute` values (`test/native/compute_test.ml`);
and the mandatory per-op Native4D coverage. Regenerating the 100-model sweep
after each stage shows real, moving evidence, not a vacuous restriction:
against the exponent-0.5-only `pow.Tensor_Scalar`, `mobilenetv5_base`'s
diagnostic sharpened from "unsupported PT2 operator" to a `malformed`-graph
rejection naming the actual value — `pow.Tensor_Scalar: exponent=2 is not
supported (only 0.5)` — confirming its occurrence is a genuine `x^2`, not a
disguised sqrt; after generalizing `pow.Tensor_Scalar` to every exponent in
the same session, `mobilenetv5_base` clears that blocker entirely and now
stops on `rsqrt.default` — a distinct, currently-unbound ATen op (notably the
same operation `Pow`'s own `-0.5` special case computes internally, just not
yet reachable as its own bound target) and a natural next lead.
`swiftformer_xs` advances past `linalg_vector_norm.default` entirely and now
stops on `clamp_min.default` — a previously-untracked op, not currently bound
anywhere in this repo, and a natural next candidate (no existing `.ai`/todo
mention; noted here only as a discovered lead, not scoped as a row).

Ordered: (1) the rest of CSATv2's medium-complexity structural set; then
(2) the high-complexity matrix/attention/indexing set.

## 1. Medium complexity — remaining CSATv2 structural set

Independent vertical slices, in this order (unchanged from `ops.md`, verified
still unimplemented in `Graph_ir` — no `Upsample` node exists today; `Amax`
and `Pow_scalar`/`Vector_norm` landed, see above):

1. **`upsample_bilinear2d.vec` (14)**, explicit output sizes,
   `align_corners=true` — real bilinear-resize op (shape inference,
   coordinate transform, boundary handling); Native4D `ResizeBilinear4` must
   reject/diagnose unsupported axes rather than reinterpret them. Do not
   conflate with `upsample_bicubic2d.vec`, which the sweep shows blocking
   `sam2_hiera_tiny` separately — a different coordinate-transform/overload,
   not covered by this row.
2. **`clone` with a supplied memory format (2 in CSATv2)** — decode the enum,
   prove identity for the observed layout or represent the relayout; only
   admit a no-op proof in Native4D, never assume a requested layout
   conversion is a plain `Clone`. The sweep shows this same case hard-rejects
   3 more models (`hiera_tiny_224`, `mobilevitv2_175`, `mvitv2_tiny`) as
   `malformed` — i.e. the importer's refuse-rather-than-ignore choice is
   already correct there too; no separate design question, just more
   coverage once this row lands.
3. **`max_pool2d.default` with `ceil_mode=true`** — not in CSATv2 at all; new
   from the sweep, blocks 2 models (`hgnetv2_b0`,
   `legacy_seresnext26_32x4d`), also currently a hard `malformed` reject.
   Same shape-computation family as the existing pooling ops: ceiling instead
   of floor division when computing output extent.

Each row: ATen binding in `bin/aten_ops_gen.ml` if absent, `Aten_op_config`
decoder/spec fixture, bridge lowering, Native implementation, dispatch audit.
Add an ATen-vs-Native walk only where a non-vacuous recipe exists; otherwise a
table-driven boundary suite. Keep CSATv2 graph-only in CI until a complete
end-to-end execution test passes (`ops.md` explicit instruction) — item 3 has
no bearing on that CSATv2 gate since it does not appear in its graph, but
should not be deferred behind it either given its independent cross-model
leverage. Every new bridge arm here must build exactly one node
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
