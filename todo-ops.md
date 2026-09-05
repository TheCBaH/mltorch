# ATen operation coverage: breadth-first TODO

This is the current operation backlog for the checked-in tracked-model
corpus.  Its primary goal is **breadth**: make a small, useful operation or
configuration work through every applicable surface of this project before
adding another Native-only arm.  **Depth**—the number of target names, node
occurrences, and graphs that get farther through the pipeline—matters, but is
a secondary outcome and must not be reported as whole-project support.

It supersedes neither [`todo.md`](todo.md) nor [`ops.md`](ops.md): the former
records detailed implementation decisions and the latter defines the support
contract.  This file is the compact, corpus-derived queue that connects that
contract to the next work items.

Status: 2026-09-05 (`cumsum.default` landed full-stack, Native AND Native4D
— the "Matrix / reduction" backlog row; see `ops-progress.md`'s landing
record and this file's own note below. No corpus model moves: EdgeNeXt, the
row's only occurrence, still fails one node earlier at `bitwise_not.default`
— scoreboard unchanged. `lstm.input` was investigated and deliberately NOT
landed this session: it needs a genuine new ordered-SCAN primitive in
`Expr`/`SEMANTICS` (a fold whose body sees the running accumulator), which
neither `sum` nor `max_reduce` provide — see this file's own note below for
the full scoping argument, including why `cumsum` looked similar but was not;
preceded the same day by `IndexTensor4` landed, the Native4D counterpart to the
pre-existing `Index_tensor` Native op — no corpus model gated on it, so the
scoreboard is unchanged; leaves live max-pool indices as the sole remaining
"Remaining Native4D counterpart backlog" row; preceded the same day by
`squeeze.dim` landed full-stack Native AND Native4D —
binds to the pre-existing `Reshape`/`Reshape4` pair, no new `Graph_ir`
constructor needed, closing the "Shape / sequence" backlog's first row;
preceded the same day by `conv3d.default` full-stack Native, Native4D
unconditionally rejected — the deferred "Higher-rank convolution" slice,
requiring a fix to the minimal static-dispatch ATen archive, which had
`at::native::slow_conv3d` stubbed as unimplemented; preceded by
`matmul.default`'s unequal-rank operand case, importer-only, reusing
`Batched_matmul`'s existing broadcast unchanged; preceded 2026-09-04 by
`unfold.default` full-stack Native, closing the `conv1d`/`unfold` P1 slice
entirely, `conv1d.default` full-stack, `Attention.Sdpa` head/batch
broadcasting closing the matmul/SDPA broadcasting gap, `Batched_matmul`'s own
real ATen broadcasting, and `Matmul.Batched_matmul` Native4D `D = 1`
admission + Region computation landing). `eca_halonext26ts` reaches
`native_builds:true` and `kernel_converts:true`, stopping at
`unfold.default`'s own deliberate Native4D rejection; `lambda_resnet26t` now
reaches `native_builds:true` and `kernel_converts:true` too, stopping at a
later, unrelated intrinsic `axis D` boundary.

## Priority model: breadth first, then depth

An ATen target name is not a unit of support on its own.  A configuration is
**full-stack supported** only when it has an accountable path through all of
the relevant surfaces below:

| Surface | What must be present for a breadth claim |
| --- | --- |
| ATen boundary | A curated binding where it can link, or a documented binding-closure exception with hand-derived bridge fixtures. |
| Both imports | `Native_aten_bridge.Op_bridge` and `Native_interp` decode the same accepted domain and construct equivalent Native graphs. |
| Native execution | A first-class `Graph_ir` operation that preserves the ATen operation's semantics, with its own shape checks, Direct evaluation, and Symbolic evaluation. |
| Native4D | A first-class `Native4d.Op` counterpart for the Native operation, with its own builder/shape/evaluation/domain coverage and four-axis tests.  The Native-to-Native4D map should map that operation to its counterpart, not use legalization as a substitute for missing operation support. |
| Kernel / explorer path | A legal Native graph reaches the Kernel/Stage-Program path; a representative graph exercises that path without an operation-specific failure. This branch is independent of Native4D conversion — `kernel_converts` and `native4d_converts` resolve as separate branches off the same `native_builds` graph (`bin/pt2_json_model_support.ml`'s `resolve_branch`), confirmed by `convit_tiny` (`native4d_converts:false` on an intrinsic `axis T` boundary, `kernel_converts:true`). |
| Evidence | ATen-vs-Native coverage when linkable, Native Direct-vs-Symbolic coverage, Native-to-Native4D map verification, and importer/domain regressions for every accepted or intentionally rejected configuration. |

The preferred route into Native is therefore **addition, not legalization**:
give the ATen operation a Native operation that keeps its semantics, rather
than lowering it to a different operation or a replacement subgraph.  The
Native4D path follows it with a corresponding Native4D operation.  Sharing a
compute functor, scalar primitive, parameter type, or test generator is
desirable; erasing the operation's identity is not.

An exception needs a written proof that the ATen operation is a true identity
or representation-only normalization in the accepted domain.  Such an
exception is intentionally narrow: it must not change value semantics, lose an
observable output, or use a decomposition merely because a direct operation is
not yet implemented.  It is recorded as an exception in the landing note, not
used as the default coverage route.

The default for any Native operation with meaningful N/H/W/C semantics is a
Native4D counterpart.  Reusing its parameter record or compute functor is
encouraged; re-expressing it as a different operation or a small legalized
subgraph is not a substitute for that counterpart.  Structural normalization
such as dropping a proven pure identity is a separate representation concern
and does not count as Native4D support for a missing semantic operation.

Call a target **Native-only, deliberately bounded** only when an explicit
design decision establishes that the operation inherently names a `D`/`T` axis
or requires a new kernel semantic outside the current N/H/W/C dialect.  This
is an exceptional, honest outcome, but it does not increment the full-stack
breadth tally.  A missing counterpart, bridge arm, evaluator, or untested
importer is simply incomplete—not an intentional boundary.

Depth is measured separately and always labelled with its stage: distinct
target/configuration count, graph-node count, `native_builds`,
`native4d_converts`, and `kernel_converts`.  In particular, a high node count
behind an earlier import failure is neither demonstrated support nor a reason
to skip the breadth gates above.

## Scope and method

The corpus is the 100 selected `pytorch-image-models` exports under
`modules/devcontainer.pytorch-image-models/models/*/models/model.json`, not a
cross-modality benchmark.  `models-selected.yaml` selects 40 of 89 architecture
families and calls out 70 release models; all are image-classification-style
graphs.  “Modalities” in this checkout therefore means CNN, hybrid, ViT,
attention, sequence-like, and resize/indexing *architectures*, not NLP, audio,
or multimodal model suites.  Do not make general ATen or multimodal coverage
claims from this data alone.

The census was made by:

1. counting every `.graph_module.graph.nodes[].target` in all 100 JSON files;
2. comparing those names with the target lists in
   `lib/native_interp/native_interp_lower_{compute,shape}.ml`; and
3. reading `test/data/pt2_json_model_support.jsonl`, generated by
   `bin/pt2_json_model_support.ml`, for the earliest actual import/conversion
   frontier.

This deliberately distinguishes a target-name census from support for every
ATen configuration.  A target listed by an importer can still reject a shape
or option; a target occurring later than a model's first failure is not yet a
demonstrated end-to-end blocker.

## Findings and scorecard

| Measure | Result |
| --- | ---: |
| Tracked graphs / architecture families | 100 / 40 |
| Raw ATen graph nodes | 30,717 |
| Distinct ATen target names | 85 |
| Target names lowered by Native and present in corpus | 53 (30,182 nodes) |
| Present target names without a Native lowering arm | 32 (535 nodes) |
| `native_builds:true` in the support sweep | 92 / 100 |
| Native4D conversions / kernel conversions | 58 / 100; 66 / 100 |
| Graphs successful in Native, Native4D, and Kernel | 44 / 100 |

The first five rows are target/node **depth** measures.  The last three are
graph-stage depth measures, not an operation breadth measure: a graph may
reach a later stage while another operation in the same graph is only
Native-only, and the kernel branch has limits independent of ATen support.
Do not derive a count of full-stack operations from either number.  Until the
per-operation matrix below is recorded for a landing, the only defensible
whole-project claim is the verified graph-stage result.

Every landing should leave a small matrix in its landing note (or in this
file's refresh) with these columns: ATen boundary, `Op_bridge`,
`Native_interp`, Native Direct/Symbolic, Native4D, Kernel, and a representative
graph.  This makes a support gap visible even when the corpus count rises.

The checked-in selection manifest currently says `total_nodes: 30217`, while
the checked-in JSON graphs contain 30,717 target-bearing nodes.  Treat that
500-node difference as selection/export metadata drift until the producer
regenerates or explains it; do not use the manifest total as the operation
census denominator.

The 8 Native failures (recensus 2026-09-05, after the `squeeze.dim`
landing, `test/data/pt2_json_model_support.jsonl`) are not 8 unsupported
target names.  Their earliest frontiers are: `bitwise_not.default`
(`edgenext_xx_small`), `im2col.default` (`volo_d1_224`), `lstm.input`
(`sequencer2d_s`), `meshgrid.indexing` (`vit_small_patch16_dinov3_qkvb`), and
`upsample_bicubic2d.vec` (`sam2_hiera_tiny`) — one model each, all already
tracked as a P1 slice or in the deferred backlog below; `index.Tensor`'s
multi-entry case (2 models: `maxxvitv2_nano_rw_256`, `mvitv2_tiny`); and a
rank-5 `getitem` importer gap (`hiera_tiny_224`, tracked under "Configuration
and transform gaps"). `lambda_resnet26t` — gated on `squeeze.dim` at the
previous recensus — is no longer in this list at all: it now reaches
`native_builds:true` (and `kernel_converts:true`), stopping at a later,
unrelated intrinsic `axis D` boundary at the Native4D stage (see the
`squeeze.dim` landing note below). This list has fully turned over since the
file's original 19-model census — every target named in that original count
(`zeros.default`, `_native_batch_norm_legit.no_stats`,
`adaptive_max_pool2d.default`, `arange.default`/`arange.start`,
`leaky_relu.default`) is landed; keep this paragraph refreshed at each
recensus rather than letting it drift the way it
did here.

## Ordered breadth-first work

Work is ordered by the number of graphs whose *first* frontier it removes,
then by whether it can gain a direct Native4D counterpart and reach the kernel
path.  Do not start a second operation in a family merely to increase the
distinct-target count while the first one lacks an importer, evaluator,
Native4D counterpart, or verification surface.  A Native-only outcome is
allowed only after an exceptional boundary has been made explicit and
regression-tested.

### P0A — complete shared Native vertical slices

**All three rows landed** (commits `2c4fc9c` "add full-stack leaky ReLU,
zeros, and arange factory support" and `56516bf` "add full-stack batch norm
no-stats support"), including their Native4D counterparts (`Zeros4`,
`Arange4`, `BatchNormNoStats` all present in `lib/native4d/ops4.ml`) — despite
the table below predating the strikethrough convention used in P0B.  Recensus
(2026-08-30) confirms each op is resolved at its old frontier and the graph
moves to a *new* first failure rather than reaching `native4d_converts`/
`kernel_converts` (no corpus model exercises these ops' Native4D path yet, so
that stage is unverified against real graphs — synthetic fixture coverage
lives in the commits' `test/native4d/*` additions):

| Work | Old frontier (3 models) | New frontier after landing |
| --- | --- | --- |
| `zeros.default` | `convit_tiny`, `edgenext_xx_small`, `sequencer2d_s` | `repeat.default` (ConViT), `_to_copy.default` (EdgeNeXt), `lstm.input` (Sequencer, as predicted) |
| `arange.default` / `arange.start` | ConViT, EdgeNeXt, MViTv2 (`.default`); DINOv3 ViT (`.start`, model id `vit_small_patch16_dinov3_qkvb`) | Neither target is any tracked model's current frontier any more (MViTv2's is now `_to_copy.default`; DINOv3 ViT's is `meshgrid.indexing`, already in the deferred backlog) |
| `_native_batch_norm_legit.no_stats` | `nf_regnet_b0`, `vit_tiny_r_s16_p8_224` | **New:** both now fail Native *construction* itself (not an unsupported operator) with `"Reshape is not a Const-SSA operation"` — see the Const-SSA gap below. This is a `native4d_reason: malformed` / `native_builds: false` failure, one stage earlier than an importer gap. |

None of the six affected models reach `native_builds:true` yet, so the
scorecard's `native_builds` 83/100 is unchanged by this landing alone — the
83 already includes these commits (regenerated at `0f766f1`). Do not re-add
these three rows as open work; promote their *new* frontiers instead per the
deferred-backlog promotion rule.

### P0B — clear present Native4D counterpart frontiers

These graphs already reach Native.  They are therefore immediate breadth work:
adding their Native4D counterparts improves full-pipeline graph support without
waiting for another ATen-import target.

| Native4D work | Corpus signal | Counterpart scope |
| --- | --- | --- |
| ~~`GroupedConv4`~~ | **Landed** (commit 61120ec): `GroupedConv2D`, groups carried as a genuine field, reusing Native's Conv2d compute unchanged. Moved 13 of the 14 frontier models to `native4d_converts:true` (`native4d_converts` 39→53); the 14th, `efficientvit_b0`, now stops at `split_with_sizes` instead — see `SplitWithSizes4` below. | Done. |
| ~~`GroupNorm4`~~ | **Landed** (commit 4ec447b): reuses `Norm.GroupNorm`'s shape/check_affine/compute unchanged; `groups` is an ordinary field, channel restricted to C matching `Batch_norm_no_stats`'s precedent. All 3 frontier models (`efficientnet_b0_g8_gn`, `efficientnet_b3_g8_gn`, `test_efficientnet_gn`) reach `native4d_converts:true` (53→56); two of the three still hit the kernel depth ceiling independently, so only `test_efficientnet_gn` joins the all-three-stages count (42→43). | Done. |
| ~~`SplitWithSizes4`~~ | **Landed** (commit 861855f): reuses `Split.Split_with_sizes`'s shape rule/offset unchanged; no N=1 precondition since the axis is kept, not dropped. All 9 frontier models moved to a LATER frontier rather than to `native4d_converts:true` -- `requires_payloads` for `inception_next_*`/`mixnet_*`/`tf_mixnet_*`, a batched-matmul D-axis rejection for `efficientvit_b0`, a `Stack` rejection for `skresnet18` -- so `native4d_converts` is unchanged at 53. Depth moved; the stage-completion count did not. | Done. |
| ~~`Expand4`~~ | **Landed** (commit dc92db8): target typed `Shape4.t`, the same trigger `Reshape4` has; `Domain.check_node` admits every `Expand` unconditionally, the real gate is the `four` wrap plus the lowerer's `Shape4.of_vec6`. `swiftformer_xs` moved to a LATER frontier (`requires_payloads`) rather than to `native4d_converts:true` -- Expand was one of several blockers. `native4d_converts`/`kernel_converts`/all-three-stages unchanged at 56/61/43. | Done. |

**P0B is now fully landed** (all four rows above).  Cumulative since P0B started: `native4d_converts` 39→56, all-three-stages 31→43 (see the P0A row history for `native_builds` 81→83, from the P0A landings that preceded this session).

**2026-08-30 recensus.** P0A's three rows and P1's `leaky_relu.default` row
are also already landed (see their tables above); the headline scorecard was
already regenerated at `0f766f1` and does not change further from this
recensus alone. The recensus surfaced one new frontier — the
`_native_batch_norm_legit.no_stats` fold tripping `nf_regnet_b0` and
`vit_tiny_r_s16_p8_224` on `"Reshape is not a Const-SSA operation"` — which
turned out to be the first of five (Reshape, Expand, Mul_scalar, Pow,
Add_scalar), now all landed; see "Configuration and transform gaps" below.
That work is done: zero Const-SSA registry gaps remain in the 100-model
corpus, and `native_builds` moved 83→87.

**2026-08-30, `Select4` landed** (full-stack; see `ops-progress.md`'s landing
record). `csatv2` — the one model `Select4` was gating — moves from
`no legalization for select x=t405 params={axis=H index=0}` to
`no legalization for stack xs=[t413, t416, t419] params={axis=H}`: the next
row of the same "Remaining Native4D counterpart backlog" table (`Stack4`).
`native_builds`/`native4d_converts`/`kernel_converts`/all-three-stages are
unchanged (87/56/63/43) — depth moved, stage-completion did not, since
`csatv2` was and remains `native4d_converts:false` and a graph-only CI target
(`kernel_reason: over_limit`, unrelated to this change).

**2026-08-30, `Stack4` landed** (full-stack; see `ops-progress.md`'s landing
record). `csatv2` — one of the two models `Stack4` was gating — moves from
`no legalization for stack xs=[t413, t416, t419] params={axis=H}` to a later
node in the same graph: a `permute.default` mixing a non-unit `T` extent into
`H`, an intrinsic Native-only boundary (see "Ordered breadth-first work"
below), not further Native4D work. `skresnet18` — the other — now reports the
dialect's own axis-domain diagnostic (`axis D is outside the N/H/W/C
dialect`) at the same node instead of the lowerer's generic catch-all; it was
already `native4d_converts:false` and stays so.
`native_builds`/`native4d_converts`/`kernel_converts`/all-three-stages are
unchanged (87/56/63/43) — depth moved, stage-completion did not.

**Next priority**: of the four models the Const-SSA work unblocked
(`nf_regnet_b0`, `vit_tiny_r_s16_p8_224`, `test_vit4`, `csatv2`), all four now
stop on an intrinsic `D`/`T` axis — Native-only, deliberately bounded
territory, not further work here (`csatv2`'s own case is the `permute.default`
above). `Softmax4` is the one remaining row of the Native4D counterpart
backlog table, with no corpus model currently gated on it. Otherwise: P1's
remaining one-model slices (`adaptive_max_pool2d`, `conv1d`/`unfold`,
`im2col`/`col2im`, `upsample_bicubic2d`) and promoting
`_to_copy.default`/`lstm.input` from the deferred backlog (first frontiers
for EdgeNeXt/MViTv2 and Sequencer respectively, per the P0A table).

**2026-08-30, `repeat.default` landed** (full-stack Native; see
`ops-progress.md`'s landing record). `repeat` gets a first-class `Repeat`
Native op (`lib/native/ops/repeat.ml`) reusing `Reshape.delinearize`'s
per-axis "mod x d = x - d*(x/d)" idiom rather than its
flatten-then-redistribute strategy, since every axis keeps its own identity
under tiling. Both importers (`Op_bridge`, `Native_interp`) resolve
`repeats` against `self`'s own ATen rank via a new
`Aten_shape.resolve_repeat_size`, mirroring `resolve_expand_size`'s split
without its `-1` convention. No Native4D counterpart yet (tracked the same
"dialect does not have it at all" way `Stack`/`Softmax` were before their
own counterparts existed) — deliberately deferred, per this session's own
scoping decision, rather than an oversight. `convit_tiny` — the one corpus
model `repeat.default` was gating — moves to `repeat_interleave.self_int`,
exactly the deferred-backlog row anticipated. `native_builds`/
`native4d_converts`/`kernel_converts`/all-three-stages are unchanged
(87/56/63/43) — depth moved, stage-completion did not, since `convit_tiny`
was and remains `native_builds:false`.

**2026-08-30, `repeat_interleave.self_int` landed** (full-stack Native,
`dim` required; see `ops-progress.md`'s landing record). Shares
`lib/native/ops/repeat.ml` with `Repeat`: a new `RepeatInterleave` module
duplicates each element `repeats` times along ONE named axis
(`out / repeats`, floor division) rather than tiling every axis modulo its
own extent — the opposite composition `Repeat` performs. `dim=None`
(flatten-first) is out of scope, rejected with a typed diagnostic
(`Unsupported_option` / `Validation_failure`), since no corpus occurrence
uses it. No Native4D counterpart yet, same deliberate-deferral reasoning as
`Repeat`. `convit_tiny` moves to `copy.default` — the next row of the same
"Factories, indexing, and copies" deferred-backlog family.
`native_builds`/`native4d_converts`/`kernel_converts`/all-three-stages
unchanged (87/56/63/43) — depth moved, stage-completion did not.

**2026-08-30, `copy.default` landed**. Binds directly to the EXISTING
`Pointwise.Expand` node rather than a new Graph_ir op — see
`op_bridge_shape.ml`'s own comment: real ATen's `copy_`/`copy` fully
overwrites `self` with `src` broadcast to `self`'s shape, so the result
depends only on `src`'s values and `self`'s declared shape, never on what
`self` held before, which is exactly `Expand{size=self's shape}` applied to
`src`. The corpus's own 40 occurrences (verified) are the functionalized
form of `tensor[idx] = value` — always equal-shape, zero dtype casts — but
the translation is the fully general broadcast regardless, not an overfit to
that case. No new Graph_ir constructor, so no Native4D/domain.ml changes
needed at all — `Expand4` already exists, so this is full-stack including
Native4D wherever `Expand4` already reaches. `convit_tiny` moves to
`select_scatter.default` — the predicted reconstruction node immediately
downstream of each `copy.default` (writes the computed slice back into the
base tensor `select`/`copy` only produced a view/value for). `mvitv2_tiny`'s
frontier is unchanged (`_to_copy.default`, already earlier in its graph than
its own `copy.default` occurrences).
`native_builds`/`native4d_converts`/`kernel_converts`/all-three-stages
unchanged (87/56/63/43) — depth moved, stage-completion did not.

**2026-08-30, `select_scatter.default` landed** (full-stack Native; see
`ops-progress.md`'s landing record). New `Split.Select_scatter` Graph_ir op:
`self` with `src` written at one position of one axis, every other position
carried through from `self` unchanged — reuses `Select`'s own `output_shape`
to validate `src`'s shape rather than restating the drop-and-repack rule,
and needs no new `SEMANTICS` primitive (`S.index_eq`/`S.select`, already
added for `Pad`'s reflect mirror, are the whole compute basis for the
branch). `select_scatter` is now a curated ATen binding too (`bin/aten_ops_gen.ml`),
so its `Op_bridge` dispatch test runs against real ATen as the oracle rather
than only hand-derived values. `convit_tiny` moves to `rsub.Scalar` — the
next row of the "Pointwise / type" deferred-backlog family.
`native_builds`/`native4d_converts`/`kernel_converts`/all-three-stages
unchanged (87/56/63/43) — depth moved, stage-completion did not, since
`convit_tiny` was and remains `native_builds:false`. No `Select_scatter4`
Native4D counterpart yet — deliberately deferred, same scoping as
`Repeat`/`RepeatInterleave`.

**2026-08-30, `rsub.Scalar` landed** (full-stack Native + Native4D; see
`ops-progress.md`'s landing record). New `Pointwise.Rsub_scalar` computes
`other - alpha * self` — genuinely its own op, not a legalization onto
`Add_scalar`/`Mul_scalar` (composing those would decompose one ATen node
into two Native ones). Since it carries no axis parameter it reuses Native's
own payload directly in the Native4D dialect, the same treatment
`Add_scalar`/`Mul_scalar`/`Pow` already get — no new `Ops4` module needed.
`rsub.Scalar` is now a curated ATen binding too, so both the `Op_bridge`
dispatch test and the generated `Interp_dispatch` arm run against real ATen.

Landing this exposed the next link in ConViT's own Const-SSA fold chain
(the same whack-a-mole pattern the five-op admission series hit earlier):
`Sigmoid` first, then `Rsub_scalar` itself once `Sigmoid` was admitted. Both
are now in `Const_ssa.allows` with matching symbolic-grounding arms. This
clears ConViT's chain entirely: it now reaches `native_builds:true` (was
`false`), stopping instead at `Repeat`'s already-tracked missing
`Repeat4` Native4D counterpart (deliberately deferred at `repeat.default`'s
own earlier landing). `native_builds` moves 87→88;
`native4d_converts`/`kernel_converts`/all-three-stages unchanged (56/63/43).

**2026-08-30, `_to_copy.default` landed** (full-stack Native + Native4D; see
`ops-progress.md`'s landing record). New `Pointwise.To_copy` op, restricted
to the three-way value-domain target the corpus actually uses: `Bool` (a
genuine `x != 0` test), `Float` (identity), `Long` (truncation toward zero,
matching ATen's `static_cast<int64_t>`). The `Long` case needed a genuinely
new `SEMANTICS` primitive (`trunc`) — added as a new `unary_op` case
(`Erf`/`Exp`/`Log`/`Sqrt`/`Trunc`) rather than a bespoke `Expr.Value`
constructor like `Round_f32`, since every existing site pattern-matches
`Unary (_, a)` generically and needed no change. `Bool` needed no new
primitive: `select`/`lt` already express a nonzero test. Since the op
carries no axis parameter, Native4D reuses Native's own payload directly
(the `Add_scalar`/`Rsub_scalar` treatment), so this landed full-stack in one
session including a curated `_to_copy` ATen binding for oracle verification.
`edgenext_xx_small` moves to `bitwise_not.default` (next row of the
"Pointwise / type" deferred-backlog family); `mvitv2_tiny` moves to
`index.Tensor`'s already-tracked multi-entry case (`indices is not an
optional tensor list`, the open MaxxViTv2 case this file's own "Factories,
indexing, and copies" row already flagged). `native_builds`/
`native4d_converts`/`kernel_converts`/all-three-stages unchanged
(88/56/63/43) — depth moved, stage-completion did not, since neither model
reaches `native_builds:true` yet.

**2026-08-30, `Repeat4`/`RepeatInterleave4` landed** (full-stack Native4D;
see `ops-progress.md`'s landing record). Both reuse `Shape4`'s own
representation rather than adding an axis-domain check of their own:
`Repeat4.params.repeats : Shape4.t` IS the "T/D multiplier is 1" condition
by construction (the same treatment `Expand4.params.size` gets), so
`Domain.check_node` admits `Repeat` unconditionally, the real gate being
the lowerer's own `Shape4.of_vec6`. `RepeatInterleave4` gets the
`check_dims`-style single-axis rejection `Select4`/`Slice4` get instead,
for one consistent diagnostic across the named-axis op family, even though
(unlike those two) it is not load-bearing here: `RepeatInterleave`
multiplies its named axis rather than collapsing it, so a T/D target would
already be caught by the blanket four-axis shape check. `convit_tiny` — the
one corpus model whose Native4D frontier was `Repeat`'s missing
counterpart — moves to `select_scatter.default`'s own already-tracked
missing `Select_scatter4` counterpart (the next row of the Native4D
counterpart backlog table), confirming both new ops against a real corpus
graph rather than only synthetic fixtures.
`native_builds`/`native4d_converts`/`kernel_converts`/all-three-stages
unchanged (88/56/63/43) — depth moved, stage-completion did not, since
`convit_tiny` remains `native4d_converts:false` either way.

**2026-08-30, `Select_scatter4` landed** (full-stack Native4D; see
`ops-progress.md`'s landing record). Reuses `Split.Select_scatter`'s shape
rule/pixel map unchanged, the same delegation `Select4` makes to
`Split.Select`. Gets the same `check_dims`-style single-axis rejection
`Select4` gets — genuinely load-bearing here, unlike `RepeatInterleave4`'s:
the WRITTEN axis is real op-level data even though (unlike `Select`)
`Select_scatter`'s own output shape (`self_shape`, unchanged) never
depends on it, so there is no separate shape-consequence rejection to
demonstrate. `convit_tiny` — the one corpus model gated on it — moves to an
intrinsic `axis T is outside the N/H/W/C dialect` boundary, a genuine
Native-only wall rather than a missing counterpart, confirming the corpus
model's own Native4D story is otherwise complete up to that point.
`native_builds`/`native4d_converts`/`kernel_converts`/all-three-stages
unchanged (88/56/63/43) — depth moved, stage-completion did not.
`test/native4d/lower_test.ml` crossed the 1000-line file-size cap in this
landing and was split into `lower_test.ml` (factory/norm/precondition rows)
+ `lower_shape_test.ml` (reshape/expand/repeat/transpose family), sharing
rendering helpers via a new `lower_fixtures.ml`.

**Next priority**: every row of the "Remaining Native4D counterpart
backlog" table is now landed except `Softmax4` and live max-pool
indices/`IndexTensor4` — both open-ended designs, not one-session slices.
Otherwise: `lstm.input` from the deferred backlog (36 occurrences,
Sequencer2D's own first frontier), or P1's remaining one-model slices
(`conv1d`/`unfold`, `im2col`/`col2im`, `upsample_bicubic2d`).

**2026-08-30, `adaptive_max_pool2d.default` landed** (full-stack Native +
a full Native4D counterpart for the value-only op; see `ops-progress.md`'s
landing record and the P1 table's own row above). `bat_resnext26ts` moves
to `eye.m` — the next row of the "Factories, indexing, and copies" deferred
backlog table below.
`native_builds`/`native4d_converts`/`kernel_converts`/all-three-stages
unchanged (88/56/63/43) — depth moved, stage-completion did not, since
`bat_resnext26ts` was and remains `native_builds:false`.

**Next priority**: `eye.m` (12 occurrences, now `bat_resnext26ts`'s own
first frontier) from the deferred backlog below; `Softmax4` and live
max-pool indices/`IndexTensor4` remain the two open-ended Native4D
counterpart-backlog rows; otherwise `lstm.input` from the deferred backlog
(36 occurrences, Sequencer2D's own first frontier), or P1's remaining
one-model slices (`conv1d`/`unfold`, `im2col`/`col2im`,
`upsample_bicubic2d`).

**2026-08-30, `eye.m` landed** (full-stack Native + a full `Eye4` Native4D
counterpart; see `ops-progress.md`'s landing record). Follows `zeros.
default`/`arange`'s own precedent directly (a genuine no-operand factory,
not a decomposition), including their `eval_direct.ml` dtype-fidelity
bypass; `Aten_shape.of_aten` right-aligns eye.m's rank-2 `[n; m]` onto the
frame's `[W; C]` axes unconditionally, so `Eye4` needs no axis-domain
rejection the way `Zeros4` doesn't either. `bat_resnext26ts` now reaches
`native_builds:true` (was `false`), stopping at a genuine intrinsic `axis
D is outside the N/H/W/C dialect` wall later in the same graph — Native-
only, deliberately bounded territory, not further Native4D work.
`native_builds` moves 88→89; `native4d_converts`/`kernel_converts`/
all-three-stages unchanged (56/63/43), since `bat_resnext26ts` stays
`native4d_converts:false`/`kernel_converts:false` either way (the kernel
stage is separately capped by the pre-existing evaluation-depth ceiling).

**Next priority**: `Softmax4` and live max-pool indices/`IndexTensor4`
remain the two open-ended Native4D counterpart-backlog rows; otherwise
`lstm.input` from the deferred backlog (36 occurrences, Sequencer2D's own
first frontier), or P1's remaining one-model slices (`conv1d`/`unfold`,
`im2col`/`col2im`, `upsample_bicubic2d`).

**2026-08-31, `Softmax4` landed** (full-stack Native4D counterpart; see
`ops-progress.md`'s landing record). Reuses `Reduce.Softmax`'s own
`output_shape`/`Compute` unchanged, softmax's `Axis.t` narrowed to
`Axis4.t` the same way `RepeatInterleave4`'s single named axis is — softmax
never drops rank, so unlike `Amax`/`Mean`/`Vector_norm` there is no
keepdim-vs-drop distinction to restate, and unlike `RepeatInterleave4` the
axis check is the WHOLE domain gate rather than a diagnostic-consistency
extra, since the output shape (`x`'s own, unchanged) never depends on which
axis is reduced. No corpus model is currently gated on `Softmax4` (per this
file's own note at the row's original addition), so `native_builds`/
`native4d_converts`/`kernel_converts`/all-three-stages are unchanged
(89/56/63/43) and the corpus signal (`make pt2.json-model-support`) did not
change. Landing this also retired the `Sdpa_batch_axis` diagnostic's stale
"(Native has no Bmm or softmax in Native4D)" parenthetical — updated to
name the actual remaining blocker, Native4D's `Bmm` legalization admitting
only a single batch.

**Next priority**: live max-pool indices/`IndexTensor4` is now the only
remaining open-ended Native4D counterpart-backlog row; otherwise
`lstm.input` from the deferred backlog (36 occurrences, Sequencer2D's own
first frontier), or P1's remaining one-model slices (`conv1d`/`unfold`,
`im2col`/`col2im`, `upsample_bicubic2d`).

**2026-09-02, Region computation landed, then `Matmul.Batched_matmul`'s
Native4D counterpart at `D = 1` (full-stack; see `.ai/region_compute_design.md`
and `.ai/native4d_design.md` §7.4/§7.9 for the design record — this branch's
own `_ai_/ops-progress.md` copy predates this recensus and was not kept in
sync commit-by-commit).** Region computation gives RMSNorm, LayerNorm,
Softmax, and SDPA an operation-authored `Region_program.t` shared by Native
and Native4D Direct/Symbolic evaluation, replacing ad hoc per-pixel
expressions with cached scalar/vector locals. Two corpus effects, both
confirmed against `test/data/pt2_json_model_support.jsonl`:
- `convit_tiny`: `kernel_converts` flips `over_limit`→`true` (the shallower
  Region-cached evaluation clears the evaluation-depth ceiling); its
  `native4d_converts` stays `false` on its own unrelated `axis T` boundary.
- `efficientvit_b0`: `Batched_matmul` (Native, landed 2026-08-29 — see the
  "Configuration and transform gaps" table above) gets a Native4D
  counterpart admitted at `D = 1` — the earlier reading that `Batched_matmul`
  named `D` unconditionally was wrong; `N`/`T`/`D`/`H` all agree by
  `output_shape`, `T = D = 1` is forced by `check_shapes`, and the corpus's
  real batch (`mvitv2_tiny`, `efficientvit_b0`) is carried on `N`/`H`, both
  dialect-nameable. `efficientvit_b0` newly reaches ALL THREE stages
  (`native4d_converts` and `kernel_converts` both `true`).
`native4d_converts` 56→57, `kernel_converts` 63→64, all-three-stages 43→44;
`native_builds` unchanged at 89 (neither model's frontier was at Native
construction). `Bmm` itself was later simplified to legalize unconditionally
onto this same `Batched_matmul` route at Native level (any batch extent,
commit `b65600a`), retiring the old batch=1-only 1x1-convolution
legalization — a representation simplification with no corpus-count effect
(`Bmm`'s importer domain was already batch-less only). **Still open**: real
ATen broadcasting in both `Batched_matmul` and `Sdpa` (unequal, not merely
`>1`, leading extents) — now with direct corpus evidence
(`lambda_resnet26t`, `mobilenetv5_base`), tracked in the "Configuration and
transform gaps" table above rather than as a fresh row here.

**Next priority**: the matmul/SDPA broadcasting gap above now has corpus
evidence and is the natural next slice; otherwise unchanged from the note
above (live max-pool indices/`IndexTensor4`, `lstm.input`, or P1's remaining
one-model slices).

**2026-09-04, `Batched_matmul` real ATen broadcasting landed** (full-stack;
see the "Configuration and transform gaps" table's own row above for the
complete record). `lambda_resnet26t` clears this frontier and moves to
`conv3d.default` (deferred backlog, "Higher-rank convolution"). SDPA's own
head-broadcasting half of this gap is intentionally NOT started this
session — its scope (the Region-authored `Computation` path, `output_shape`,
`score_shape`, and mask broadcasting all need re-deriving against a genuine
broadcast output shape, not `query_shape` as today) is closer to
`region-sdpa-computation-plan.md`'s own multi-session precedent than to the
matmul fix just landed. `native_builds`/`native4d_converts`/`kernel_converts`/
all-three-stages unchanged (89/57/64/44).

**Next priority**: SDPA head broadcasting (`mobilenetv5_base`) is the one
piece of this gap still open, and needs its own design pass rather than a
quick extension. Otherwise unchanged: live max-pool indices/`IndexTensor4`,
`lstm.input`, `conv3d.default` (now `lambda_resnet26t`'s frontier), or P1's
remaining one-model slices (`conv1d`/`unfold`, `im2col`/`col2im`,
`upsample_bicubic2d`).

**2026-09-04, SDPA head/batch broadcasting landed** (full-stack; see
`.ai/attention_design.md` §12 for the complete record). `Attention.Sdpa`
gains real ATen `N`/`T`/`D`/`H` broadcasting (equal, or one side 1) across
query/key/value, reusing `Matmul.Batched_matmul`'s own per-axis rule and
`broadcast_coord` treatment in both `Compute` paths (`Legacy_pixel` and the
authoritative Region-authored `Computation`); `output_shape`/
`total_work_bounded`/the output-numel bound now derive the REAL broadcast
batch shape rather than assuming it equals `query_shape`'s own, and
`Region_computation.program`'s SDPA sanity check was fixed to match (it
used to assume `output_shape = query.shape`, no longer true once query can
be the SMALLER operand on a broadcast axis). Grounded against real ATen at
the flash-oracle's own admission gate (`sdp_utils_cpp.h`'s
`check_batch_size_and_num_heads_dense`): a broadcastable-but-unequal head or
batch count without `enable_gqa` is real ATen behavior that simply falls to
the `math` backend (not GQA's `repeat_interleave`, which only runs when
`enable_gqa=true`) — two hand-derived `verify_print` fixtures (one per
broadcast axis family) confirm real ATen agrees with this row's
`math`-backend-shaped arithmetic at the EXISTING `1e-5` tolerance, not a
newly widened one. `mobilenetv5_base` moves from `native_builds:false` to
`native_builds:true` AND `native4d_converts:true` in the same landing — its
SDPA occurrences are all `D = 1` (broadcasting only on `H`), so the
already-landed Native4D `Sdpa` `D = 1` counterpart (2026-09-02) now has its
first real corpus model, stopping only at the kernel stage's pre-existing,
unrelated evaluation-depth ceiling. `native_builds` 89→90,
`native4d_converts` 57→58; `kernel_converts`/all-three-stages unchanged
(64/44). **Not done**: `Recipe_sdpa` (the ATen-oracle fuzz walk) was not
widened to sweep broadcast configurations — same scoping precedent as
`Batched_matmul`'s own broadcast landing, `.ai/matmul_softmax_design.md`
§6's "not done" note.

**Next priority**: this closes the matmul/SDPA broadcasting gap entirely.
Remaining work: live max-pool indices/`IndexTensor4`, `lstm.input`,
`conv3d.default` (`lambda_resnet26t`'s frontier), or P1's remaining
one-model slices (`conv1d`/`unfold`, `im2col`/`col2im`,
`upsample_bicubic2d`).

**2026-09-04, `conv1d.default` landed** (full-stack; see `ops-progress.md`'s
landing record). New `Conv.Conv1d` Native op (`lib/native/ops/conv_conv1d.ml`):
its own graph node (one node per ATen op, per this file's design goal), but
its `output_shape`/`Compute` delegate whole to `Conv2d` with the H axis
pinned to a new public `Conv2d.unit_window` (kernel=1, stride=1, no padding,
dilation=1) — the same delegation shape `Conv2d_padding` already uses for
`Conv2d`, and the same "H stays real but pinned" treatment
`lower_engine.ml`'s own former private `unit_window` copy used for `Linear`'s
1x1-convolution legalization (now a single shared definition). Both
importers relayout the ATen rank-3 `[N,C,L]`/weight `[Cout,Cin/groups,K]`
operands through one new self-inverse permutation, `perm_conv1d` (swaps
`N<->H` and `W<->C`): both ATen shapes share the same positional pattern
(`role0, channel, spatial`) under `Aten_shape.of_aten`'s mechanical
right-alignment, so one constant relayouts `x`/`weight` in and the raw
output back out. At Native4D, `Conv1d` needs no new op or payload at all: its
`to_conv2d_params` translation plugs directly into the EXISTING `forward_conv`
helper (`lib/native4d/lower_engine.ml`) that already picks `Conv2D`/
`DepthwiseConv2D`/`GroupedConv2D` for `Conv2d`/`Conv2d_padding`/`Convolution`
by `groups` — the same "map onto an existing op after translating
parameters" legalization `Bmm`→`Batched_matmul` and `Linear`→`Conv2D4` use,
just at the Native4D layer instead of Native's own. Real-ATen grounded: a new
curated `conv1d` ATen binding (`bin/aten_ops_gen.ml`) backs a
`test/native_bridge/conv_test.ml` oracle test (defaults, non-default
stride/padding/dilation together, depthwise, and general grouping, all `Ok`
against real ATen) — `conv1d.default` lands in the generated ATen walk's own
`needs_meta` backlog (no schema default for a required `int[1]` window, the
same reason `eye.m`/`rsub.Scalar`/`select_scatter.default` are there too),
so this hand-derived coverage is the oracle, not a placeholder for one.
`eca_halonext26ts` — the one corpus model gating this row — moves to
`unfold.default`, exactly the P1 table's own prediction.
`native_builds`/`native4d_converts`/`kernel_converts`/all-three-stages
unchanged (90/58/64/44) — depth moved, stage-completion did not, since
`eca_halonext26ts` was and remains `native_builds:false`.

**2026-09-04, `unfold.default` landed** (full-stack Native; see
`ops-progress.md`'s landing record — closes the conv1d/unfold P1 slice
entirely). New `Unfold` Native op (`lib/native/ops/unfold.ml`): a
sliding-window view along one named axis, appending a genuinely new
window-offset axis. Native has no rank — every tensor is always six axes,
right-aligned by `Aten_shape.of_aten` — so appending an ATen axis RELABELS
every axis already in the frame; the op expresses this as one fixed
rotation over `Axis.t` (`source_of`/`dest_of`: each carried-through axis
shifts one step toward N, the fresh window axis always lands on C), with a
single precondition (`x_shape`'s own N must already be unit) standing in
for "ATen rank <= 5". No permute wrap needed in either importer — unlike
`Conv1d`, `Unfold`'s own axis parameter is resolved directly via the same
`Aten_shape.axis_of_dim`/`dim_axis` machinery `Select`/`RepeatInterleave`
use, then translated through `dest_of` to name the OUTPUT axis. Grounded
against real ATen: a new curated `unfold` binding plus a `verify_print`
oracle in `test/native_bridge/shape_ops_test.ml` covering a plain vector,
both of the corpus's own HaloAttn spatial-axis cases, and — the most
demanding case — a rank-5 input producing a rank-6 output (using every
frame axis), all agreeing with real ATen on the first run. **Native4D**:
unconditionally rejected (not a missing counterpart) — `Unfold`'s own shift
moves real (non-unit) content onto D/T for any input whose H/C axes
actually hold data, which is the ordinary case, so this is the same
intrinsic-axis boundary `Batched_matmul`'s multi-batch form and `Sdpa`'s own
D axis are (`.ai/native4d_design.md` §8 territory).

`eca_halonext26ts` — the one corpus model gating this row — clears BOTH
`conv1d.default` and every `unfold.default` occurrence (6, across all four
HaloAttn blocks) and moves to a genuinely different kind of frontier:
`matmul.default: both operands must be rank>=2 and of equal rank, got
self=[32, 8, 8, 16] other=[16, 23]` — an IMPORTER decode-time rank check
(both `Op_bridge` and `Native_interp`'s `matmul.default` arms currently
require equal operand rank), not a missing Native operation. This is a new,
unscoped gap — worth its own investigation given today's earlier
`Batched_matmul`/`Sdpa` real-ATen-broadcasting landings — not a continuation
of the conv1d/unfold slice.
`native_builds`/`native4d_converts`/`kernel_converts`/all-three-stages
unchanged (90/58/64/44) — depth moved, stage-completion did not, since
`eca_halonext26ts` was and remains `native_builds:false`.

**2026-09-05, `matmul.default`'s unequal-rank case landed** (importer-only;
see `.ai/matmul_softmax_design.md` §7.1 for the complete record). Real ATen
accepts two `matmul.default` operands of DIFFERENT rank (both rank>=2) via
an implicit prepend of `1`s to the lower-rank operand's batch prefix — and
`Aten_shape.of_aten`'s existing right-alignment already does exactly this,
independently per operand, so `Batched_matmul`'s pre-existing (2026-09-04)
per-axis `N`/`T`/`D`/`H` broadcast computes the correct output with no
operand-rank comparison in the shape rule at all. The only change is at the
importer boundary (`Op_bridge_linalg.ml`, `Native_interp_lower_compute.ml`):
the `Batched_matmul` route's guard widened from `rank_a = rank_b && rank_a
>= 3` to `rank_a >= 2 && rank_b >= 2`; `Matmul_unsupported_shape`'s message
and doc comments dropped "and of equal rank" since rank<2 is now the only
live rejection. No new Native op, no Native4D change — this is purely an
addition at the boundary, reusing 100% pre-existing Native machinery.
`eca_halonext26ts` — the corpus model exposing this gap — moves from
`native_builds:false` to `native_builds:true` AND `kernel_converts:true` in
one landing (`native_builds` 90→91, `kernel_converts` 64→65), stopping at
its own next node, an `unfold.default` occurrence Native4D already rejects
unconditionally and deliberately (this week's own P1 landing) —
`native4d_converts`/all-three-stages unchanged (58/44). Verified by hand
(`test/native_bridge/dispatch_test.ml`'s new "accepts unequal operand ranks"
fixture, two independent `2x3 @ 3x2` products against a shared rank-2
`other`) and by graph structure (`test/native_interp/matmul_test.ml`).
**Not done**: `Recipe_matmul`'s ATen-oracle fuzz walk was not widened to
sweep an unequal-rank configuration, same scoping precedent as the
2026-09-04 broadcast landing.

**Next priority**: live max-pool indices/`IndexTensor4`, `lstm.input`,
`conv3d.default` (`lambda_resnet26t`'s frontier), or P1's remaining
one-model slices (`im2col`/`col2im`, `upsample_bicubic2d`).

**2026-09-05, `conv3d.default` landed** (full-stack Native, Native4D
unconditionally rejected; see `ops-progress.md`'s landing record — closes the
deferred "Higher-rank convolution" backlog row). New `Conv.Conv3d` Native op
with its own genuine triple-nested `Compute` (NOT a `Conv2d` delegation,
since ATen's schema names three independent spatial axes): one new
permutation, `perm_conv3d` (a 4-cycle on `D,H,W,C`, not its own inverse),
lands ATen's own `D`/`H`/`W` spatial axes on native `D`/`H`/`W` directly and
batch on native `N`, reused identically for `x` and `weight` the same way
`perm_conv1d` is. Required a fix to the minimal static-dispatch ATen archive:
`at::native::slow_conv3d` (real `conv3d`'s non-dilated CPU forward path) was
stubbed as unimplemented and reliably ABORTED THE PROCESS rather than
erroring — `lib/aten/build_archive.sh` gained `ConvolutionMM3d.cpp` +
`Unfold3d.cpp` and the dead stub was removed. `lambda_resnet26t` — the one
corpus model gating this row — clears all 3 occurrences and moves to
`squeeze.dim` (the "Shape / sequence" deferred-backlog row).
`native_builds`/`native4d_converts`/`kernel_converts`/all-three-stages
unchanged (91/58/65/44) — depth moved, stage-completion did not, since
`lambda_resnet26t` was and remains `native_builds:false`.

**2026-09-05, `squeeze.dim` landed** (full-stack Native AND Native4D; see
`ops-progress.md`'s landing record — closes the "Shape / sequence"
deferred-backlog row's first entry). Real ATen's `squeeze.dim` drops the
named axis when its extent is 1 and leaves `self` unchanged (same rank)
otherwise — a genuine per-shape branch, not a rejection — and either outcome
keeps the linearized data order, so both importers legalize it to the
EXISTING `Reshape` node alone (`Op_bridge` decides the branch against the
live tensor, `Native_interp` against the serialized shape). No new
`Graph_ir` constructor, so no Native4D work was needed either: `Reshape`
already lowers unconditionally to `Reshape4`, so this landed full-stack,
including Native4D, in one importer-only change — the same shape
`copy.default`'s binding onto `Expand`/`Expand4` took. `lambda_resnet26t` —
the one corpus model gating this row — clears all 3 occurrences and moves
from `native_builds:false` to `native_builds:true` AND `kernel_converts:true`
in one landing, stopping at a LATER, unrelated intrinsic `axis D` boundary
at the Native4D stage (the same family `Batched_matmul`'s multi-batch form
and `Unfold` already occupy). `native_builds` 91→92, `kernel_converts`
65→66; `native4d_converts`/all-three-stages unchanged (58/44).

**Next priority**: live max-pool indices/`IndexTensor4`, `lstm.input`
(Sequencer2D's own first frontier), or P1's remaining one-model slices
(`im2col`/`col2im`, `upsample_bicubic2d`).

**2026-09-05, `IndexTensor4` landed** (full-stack Native4D counterpart; see
`ops-progress.md`'s landing record). Reuses `Index_tensor.Index_tensor`'s own
`output_shape`/`Compute` unchanged: the gathered axis narrows to `Axis4.t`
with the same `check_dims`-style rejection `Select4`/`RepeatInterleave4` get.
Unlike `Select4`, no post-hoc `Shape4.of_vec6` re-check is needed — `self`'s
axis extent is overwritten by `index`'s own length (no drop, no repack), the
same shape story `Select_scatter4` already has. Classified `Discontinuous` in
`Output_transfer4`, matching Native's own `Index_tensor`: the gathered
position is data-dependent on `index`'s stored value, not the output
coordinate alone. No corpus model is currently gated on it (`index.Tensor`'s
own single-entry Native case, `csatv2`, was already past this point before
Native4D had a counterpart at all — its own frontier is elsewhere), so
`make pt2.json-model-support` produced no diff and all four scoreboard
numbers are unchanged (92/58/66/44). This closes the `IndexTensor4` half of
the "Live max-pool indices and `IndexTensor4`" backlog row; live max-pool
indices (a genuine two-output `ArgMaxPool`/`MaxPoolWithIndices4`, needing the
multi-output plumbing `Unbind` established) remains open and is the only row
left in the "Remaining Native4D counterpart backlog" table.

**Next priority**: live max-pool indices is now the only remaining Native4D
counterpart-backlog row; otherwise `lstm.input` (Sequencer2D's own first
frontier), or P1's remaining one-model slices (`im2col`/`col2im`,
`upsample_bicubic2d`).

**2026-09-05, `cumsum.default` landed** (full-stack Native AND Native4D; see
`ops-progress.md`'s landing record — closes the "Matrix / reduction"
deferred-backlog row's `cumsum.default` entry). Real ATen's `cumsum.default`
walks one axis, and each output position is the SUM of every input position
at or before it on that axis — a genuine ordered dependency, but not the kind
that needs a new primitive: `output[i] = sum(input[0..i])` is just
`SEMANTICS.sum` with `hi` bound to the OUTPUT's own position on the axis
(`+1`) instead of a fixed extent, exactly the coordinate-derived-bound
pattern `Adaptive_axis.Compute.bin` (this same file, `pool.ml`) already
established for adaptive pooling's per-output bin edges. So `Cumsum`'s
`Compute` needed no new `SEMANTICS`/`Expr.Reduction` primitive, unlike
`lstm.input` below. `dtype` is decoded and restricted to the schema default
(`None`) or `FLOAT` — Native's compute domain is always float
(`semantics.ml`), so `FLOAT` is the identity, exactly `_to_copy.default`'s
own `Float` case — rather than rejected outright the way `softmax.int`/
`sum.dim_IntList`/`linalg_vector_norm.default` reject ANY supplied `dtype`:
EdgeNeXt's own occurrence passes `dtype=FLOAT` casting a bool mask, so the
permissive form is load-bearing, not speculative. Grounded against real ATen
via a new curated `cumsum` binding (`bin/aten_ops_gen.ml`, needing no
`Override` — an ordinary `(Tensor, int, ScalarType?)` schema shape codegen
already emits unmodified) and `verify_print` fixtures in the new
`test/native_bridge/cumsum_test.ml` (dim=0/1/-1, an out-of-range rejection,
and the FLOAT-dtype accept path — the last via `dispatch_print` rather than
`verify_print`, since the schema-driven real-ATen decode `Interp_dispatch`
uses for every `ScalarType?` argument still raises on ANY present value
regardless of what `Op_bridge` itself accepts). **Native4D**: a direct
`Cumsum4` counterpart, reusing `Reduce.Cumsum`'s own `output_shape`/`Compute`
unchanged with the axis narrowed to `Axis4.t`, the same `check_dims`-style
rejection `Softmax4` has — landed in the same session, full-stack from the
start. No corpus model is currently gated on it (EdgeNeXt's own occurrence is
one node past `bitwise_not.default`, still unlanded — see the "Pointwise /
type" backlog row — so the graph never reaches its `cumsum.default` node),
so `make pt2.json-model-support` produced no diff and all four scoreboard
numbers are unchanged (92/58/66/44).

**`lstm.input` investigated 2026-09-05, deliberately NOT landed.** Sequencer2D's
36 occurrences are all one shape family (single-layer, `bidirectional=True`,
`batch_first=True`, biased; only the `output` tensor is live, `h_n`/`c_n` are
dead) — the corpus-scoping part is easy. The blocker is architectural: an
LSTM step is `h_t, c_t = gate(x_t, h_{t-1}, c_t-1)`, a genuinely
NON-associative recurrence whose gate computation at step `t` must consume
the actual (nonlinear) VALUE of step `t-1`'s own output — unlike `cumsum`
above, this cannot be expressed as `SEMANTICS.sum`/`max_reduce` with a
cleverer bound, because those reductions combine independently-computable
terms with a fixed associative op and never expose the running accumulator to
the body. `Compute`'s own contract (`.ai/native_compute_design.md`) requires
building ONE expression generic over a SYMBOLIC output coordinate (so it can
be printed, proved, and grounded later, not just evaluated once per concrete
pixel), which rules out unrolling the recursion in the host language even
though `T` (sequence length) is a compile-time constant here. Landing
`lstm.input` for real therefore needs a new ordered-SCAN primitive threaded
through `Expr.Reduction`/`SEMANTICS` (both `Direct` and `Symbolic`
semantics, the printer, the comparator, well-formedness) — foundational DSL
work, not a vertical op slice. No other backlog row currently needs it
(`einsum.default`, `max.dim`, `tile.default` are all ordinary associative
reductions or gathers); `cumsum.default` above looked like a candidate
motivator but turned out not to need it either. Treat `lstm.input` as its own
scoping/design task before implementation, not as "next priority" filler.

### P1 — small vertical slices with one-model proof

| Work | Corpus evidence | Breadth-first acceptance condition |
| --- | --- | --- |
| ~~`leaky_relu.default`~~ | **Landed** (commit `2c4fc9c`, bundled with the P0A `zeros`/`arange` slice). `darknet17` now shows `native_builds:true`, `native4d_converts:true`, `kernel_converts:true` — the first P1/P0A row to clear all three graph stages end to end. | Done. |
| ~~`adaptive_max_pool2d.default`~~ | **Landed** (2026-08-30; see `ops-progress.md`'s landing record). Full-stack Native (`Adaptive_max_pool2d`/`Adaptive_max_pool2d_with_indices`, both importers, a curated ATen binding + real-ATen walk oracle covering the argmax index too) plus a full Native4D counterpart for the value-only op via `Drop_pool_indices`'s dead-index narrowing — the exact `max_pool2d.default`/`max_pool2d_with_indices.default` relationship, one level inside Native since ATen itself has no value-only adaptive overload. The two-output ATen-facing form's Native4D story is **not** closed: it joins `Max_pool2d_with_indices` in the dialect's existing "no argmax-pool operation" rejection, the same tracked "Live max-pool indices and `IndexTensor4`" backlog row below — the original acceptance condition's "Add an `AdaptiveMaxPool2d4` counterpart" undersold this: a *value-only* counterpart is exactly what landed, not a two-output one, and no corpus model needs the two-output counterpart today. `bat_resnext26ts` moves to `eye.m`, the next row of the "Factories, indexing, and copies" deferred backlog. | Done for the value-only op; the two-output form's Native4D counterpart folds into the live-index backlog row, not a fresh gap. |
| ~~`conv1d.default`~~ + ~~`unfold.default`~~ | **Both landed** (2026-09-04; see `ops-progress.md`'s landing records). `Conv1d` legalizes directly onto the existing `Conv2D`/`DepthwiseConv2D`/`GroupedConv2D` Native4D ops via `forward_conv`, H pinned to the unit window — no `Conv1d4`. `Unfold` is full-stack Native but unconditionally REJECTED at Native4D: its own axis-shift moves real content onto D/T for any ordinary (non-unit H/C) input, the same intrinsic-axis boundary `Batched_matmul`/`Sdpa` have, not a missing `Unfold4` counterpart. `eca_halonext26ts` clears both and moves to `matmul.default`'s unequal-operand-rank case — an importer gap, not a missing Native op. | Done. |
| `im2col.default` + `col2im.default` | 4 + 4 nodes in `volo_d1_224`; `im2col` is the first failure | These are a paired layout/gather-scatter problem, not an isolated patch.  Add paired `Im2col4`/`Col2im4` counterparts and design their import, evaluation, and kernel story together; the observed `im2col` is 3x3, dilation 1, padding 1, stride 2. |
| `upsample_bicubic2d.vec` | 1 node in `sam2_hiera_tiny`; its first failure | Separate coordinate-transform review from existing bilinear/nearest support, then add a `Bicubic2d4` counterpart rather than a bridge-only arm.  The occurrence has output `[56,56]`, `align_corners=false`, no scales. |

## Configuration and transform gaps in already named operations

These should be tracked as operation coverage, even though their target names
are in the lowering lists.  They are particularly important breadth work: a
target that works only in one importer, one evaluator, or one rank/axis domain
does not become full-stack supported merely because its name is present.  If a
new configuration has distinct ATen semantics, add it as a Native operation
and a Native4D counterpart; do not conceal it in a bridge legalization.

| Gap | Evidence | Breadth-first TODO |
| --- | --- | --- |
| Broadcasted batch dimensions in `matmul.default` | **Landed 2026-09-04.** `Matmul.Batched_matmul.output_shape` now applies real per-axis ATen broadcasting (equal, or one side extent 1) on all four batch axes (`N`/`T`/`D`/`H`), reusing the same per-axis rule `Pointwise_binary.broadcast_output_shape` uses elsewhere; `Compute` reads each operand through `Pointwise_binary.broadcast_coord` (its own shape's extent-1 axes clamped to index 0) instead of reading both operands straight off the output coordinate. Both importers needed no change — they already delegated the shape check to `output_shape` itself. `lib/native4d/domain.ml`'s `check_batched_matmul` was also fixed to read the BROADCAST (output) `D`, not just `input`'s own `D` — a regression risk the old code had once broadcasting existed, since `input` could show `D=1` while `mat2`'s `D>1` made the real output `D>1`. Verified: hand-computed Direct-eval fixtures in `test/native/linear_test.ml` (mat2's `H` broadcasting against input's `H=2`) and a Native4D domain fixture in `test/native4d/{fixtures,domain_test}.ml` proving the check reads mat2's `D`, not input's. Corpus effect: `lambda_resnet26t`'s `[1,4,64,16] @ [1,1,16,64]` (`H`: 4 vs 1) now passes this check and moves to a NEW frontier, `conv3d.default` (already tracked in the deferred backlog's "Higher-rank convolution" row) — `native_builds`/`native4d_converts`/`kernel_converts`/all-three-stages unchanged (89/57/64/44) since `lambda_resnet26t` was and remains `native_builds:false`. **Not done as part of this landing**: the ATen-oracle fuzz recipe (`lib/aten_walk_recipes/recipe_matmul.ml`) still shares one `d`/`h` pair between both operands (no broadcast configuration exercised against real ATen) — consistent with existing precedent (`Add`/`Mul`'s own `*_nwalk.ml` walks also use one shared shape, no broadcast sweep either), but worth widening if this area gets more work. |
| Head broadcasting in `scaled_dot_product_attention.default` | **Landed 2026-09-04.** `Attention.Sdpa.output_shape` now broadcasts `N`/`T`/`D`/`H` per axis (equal, or one side 1) across query/key/value, chained the same way real ATen's two matmuls chain it (`qk_batch` then `full_batch`); both `Compute` paths (`Legacy_pixel`, `Computation`) read every operand through `broadcast_coord` against its own shape. Grounded against real ATen (two `verify_print` fixtures, `math`-backend since the flash gate requires equal heads/batch without GQA) at the existing `1e-5` tolerance — see `.ai/attention_design.md` §12 for the full record. `mobilenetv5_base` moves to `native_builds:true` AND `native4d_converts:true` (its SDPA occurrences are all `D = 1`, so the existing Native4D `D = 1` counterpart, `.ai/native4d_design.md` §7.9, now has its first real corpus model), stopping only at the kernel stage's pre-existing evaluation-depth ceiling. `native_builds` 89→90, `native4d_converts` 57→58; `kernel_converts`/all-three-stages unchanged (64/44). `Recipe_sdpa` itself was not widened to sweep broadcast configurations, same scoping as `Batched_matmul`'s own landing. |
| Rank-5 conv-weight ingestion | `hiera_tiny_224` fails before a node count: `getitem is rank 5, expected 4` | Find and repair or explicitly bound the `sizes_rank_4` assumption exposed after clone memory-format support.  Exercise the same accepted representation through both importers and downstream conversion; this is an importer/representation breadth gap, not evidence for a new ATen target. |
| ~~Constant-SSA registry gaps~~ | **Landed** (five commits: `2f180d9` Reshape, `9b97429` Expand, `035772c` Mul_scalar, `5cf86d9` Pow, `19f33d2` Add_scalar, each paired with a census-regen commit). `Const_ssa.allows` now admits `{Add, Sub, Mul, Div, Sqrt, Permute, Reshape, Expand, Mul_scalar, Pow, Add_scalar}`. Whack-a-mole recensus after each landing (real graph failures, not a full-corpus fold-trace rerun — see below) surfaced the next op in the same two models' fold chains until none remained: `nf_regnet_b0`/`vit_tiny_r_s16_p8_224` needed Reshape then Expand then Mul_scalar; `test_vit4` needed Expand; `csatv2` needed Pow then Add_scalar. **2026-08-30 recensus: zero `"is not a Const-SSA operation"` blockers remain anywhere in the 100-model corpus.** All four models now reach `native_builds:true` and stop at a genuine, pre-existing Native4D boundary instead (an intrinsic `D`/`T` axis for the first three; a missing `Select4` legalization for csatv2) — real further work, but not a Const-SSA gap. `native_builds` 83→87, `kernel_converts` 61→63; `native4d_converts`/all-three-stages unchanged (56/43) since none of the four clear their new Native4D-stage blocker yet. | Done. Pow's grounding (`const_ssa_symbolic.ml`'s `pow_expr`) deliberately mirrors `Pointwise_binary.Pow.Compute.pixel`'s exact six ATen-special-cased exponents plus its `exp(scalar*log x)` fallback, numerically verified against all seven branches in `const_ssa_test.ml`. `Select4` (csatv2's new blocker) is already tracked in "Remaining Native4D counterpart backlog" below. |

## Deferred target-name depth backlog

The following 23 target names occur behind the present frontiers.  This is a
depth inventory, not a delivery queue.  Keep it visible, but do not prioritize
by raw node count alone: several belong to one architecture and will only be
reached after its earlier P0/P1 work.  Promote an item only with (1) evidence
that it is an exposed first frontier and (2) a proposed cross-surface slice or
an explicit, tested Native-only boundary.

| Family | Targets (occurrences; models) |
| --- | --- |
| Factories, indexing, and copies | `index.Tensor` (28; single-entry case landed for CSATv2/MViTv2 in `3392dc0`; multi-entry case now confirmed as MViTv2's OWN first frontier too, not just MaxxViTv2's, after `_to_copy.default` landed — `"index.Tensor.indices is not an optional tensor list"`), ~~`eye.m`~~ (12; **landed** 2026-08-30 — see `ops-progress.md`'s landing record; Bat-ResNeXt's frontier moved past it to an intrinsic `axis D` boundary), `meshgrid.indexing` (1; DINOv3 ViT — now its first frontier) |
| Pointwise / type | `neg.default` (24; DINOv3 ViT), `type_as.default` (24; DINOv3 ViT), `pow.Scalar` (1; EdgeNeXt), `sin.default` and `cos.default` (3 each; EdgeNeXt, DINOv3 ViT), `bitwise_not.default` (1; EdgeNeXt — now its first frontier, after `_to_copy.default` landed), `div.Tensor_mode` (1; EdgeNeXt) |
| Shape / sequence | ~~`squeeze.dim`~~ (3; Lambda-ResNet — **landed** 2026-09-05, full-stack Native AND Native4D via the existing `Reshape`/`Reshape4` pair; see `ops-progress.md`'s landing record. Lambda-ResNet now reaches `native_builds:true`/`kernel_converts:true`, stopping at a later, unrelated intrinsic `axis D` boundary), `tile.default` (2; SAM2 Hiera, DINOv3 ViT), `lstm.input` (36; Sequencer2D — now its first frontier) |
| Matrix / reduction | `einsum.default` (20; MViTv2), ~~`cumsum.default`~~ (2; EdgeNeXt — **landed** 2026-09-05, full-stack Native AND Native4D; see `ops-progress.md`'s landing record. EdgeNeXt's own frontier does not move: it still fails one node earlier at `bitwise_not.default`, the "Pointwise / type" backlog row), `max.dim` (1; VOLO) |
| Higher-rank convolution | ~~`conv3d.default`~~ (3; Lambda-ResNet — **landed** 2026-09-05, full-stack Native, Native4D unconditionally rejected; see `ops-progress.md`'s landing record. Lambda-ResNet's frontier moved on to `squeeze.dim`, also now landed above) |

**`index.Tensor`'s single-live-entry, raw-`Long` case landed** (commit
`3392dc0`: `Graph_ir.Index_tensor`, both importers, Native shape/Compute,
Native4D rejection). Recensus (2026-08-30): CSATv2 is past it (its new
frontier is the `Pow` Const-SSA gap above); MViTv2 is also past it (new
frontier `_to_copy.default`, in the table above). **MaxxViTv2 is not** — its
`indices` operand is a genuine multi-entry list, and the importer now reports
this distinctly: `"index.Tensor.indices is not an optional tensor list"`
(`malformed`, at Native-build time) rather than an unsupported-operator
census miss. That is the real runtime-gather case `todo.md` deferred; treat it
as its own slice — locked to the single-entry domain today by
`.ai/index_tensor_design.md`'s acceptance rule — not as a reason to touch the
already-landed single-entry path.

**`conv3d.default` scoped 2026-09-05, landed the same day** (see
`ops-progress.md`'s landing record for the implementation — this note's own
scoping rationale below held up unchanged, including the "real `Conv3d`
Native op, not a delegation" conclusion). All 3 corpus
occurrences (`lambda_resnet26t`, the LambdaLayer relative-position conv) have
a genuine `[N,C,D,H,W]` rank-5 ATen shape (e.g. input `[1,1,8,8,64]`, weight
`[16,1,9,9,1]`, `stride=[1,1,1]`, `padding=[4,4,0]`) but the THIRD ATen
spatial axis (`W`, last position) always carries `kernel=1, stride=1,
padding=0` in every occurrence — i.e. a genuine two-axis (`D`,`H`)
convolution repeated independently across 64/128 planes, not a load-bearing
third spatial axis. This makes a `Conv1d`-style delegation (pin the unit
axis to `Conv2d.unit_window`, reuse `Conv2d`'s compute unchanged) tempting,
but per this file's own "addition, not legalization" rule that delegation
would be a narrowing, not a faithful `conv3d.default` — unlike `Conv1d`
(whose ATen schema genuinely has only one spatial axis, so pinning the other
IS the whole of conv1d's semantics), `conv3d.default`'s schema genuinely
names three independent spatial axes, and a future corpus model (or a
`Recipe_conv3d` oracle sweep) could easily need all three non-unit. **The
scoped-correct route is a real `Conv3d` Native op**: genuine triple-nested
convolution over three spatial axes, fitting ATen's `[N,C,D,H,W]` onto
Native's six-axis frame via `N`→`N`, `C`→`C`, and the three ATen spatial
axes onto three of Native's remaining `{T,D,H,W}` (this project's own
`Batched_matmul`/`Sdpa` precedent already established that Native's raw
6-axis representation carries no fixed per-axis semantic — only Native4D's
N/H/W/C dialect restricts axis meaning). Native4D itself should very likely
REJECT `Conv3d` unconditionally, the same "Native-only, deliberately
bounded" treatment `Unfold`/multi-batch `Batched_matmul` get: a real 3-axis
spatial convolution needs `T`+`D`+`H` (or similar) to simultaneously carry
non-unit spatial content, which is outside the 4-axis N/H/W/C dialect by
construction, not merely unimplemented. This is a materially bigger lift
than `Conv1d`'s delegation was — new `Compute` semantics (a nested `S.sum`
over three window offsets instead of `Conv2d`'s two), a new rank-5 ATen
relayout permutation, and its own curated-binding oracle test (`conv3d` has
no schema default for its required `int[3]` window, the same `needs_meta`
reason `conv1d`/`eye.m`/`rsub.Scalar` needed hand-derived coverage) — sized
closer to a `Conv2d`-from-scratch session than to `Conv1d`'s reuse. Landed
the same session after all, including the exact `N`→`N`/`C`→`C`/spatial-onto-
`{D,H,W}` layout and the unconditional Native4D rejection predicted above —
see `ops-progress.md`'s landing record.

## Remaining Native4D counterpart backlog

An operation rejected solely because `Native4d.Op` has no name for it is a
missing Native4D implementation, not a dialect exclusion.  The default work
is to add a direct counterpart with the same operation-level semantics and to
constrain only parameters that genuinely leave N/H/W/C.  Do not treat a
replacement `Reshape`/`Concat`/other operation sequence as completing this
work.

| Native4D work | Corpus signal | Counterpart scope |
| --- | --- | --- |
| ~~`Select4`~~ | **Landed** (2026-08-30; see `ops-progress.md`'s landing record). Native's `Select` gets a first-class `Select4` counterpart, the same `check_dims`-style axis-domain rejection `Slice`/`Concat`/`Split_with_sizes`/`Unbind` already had, and a real `Lower.convert` arm. `csatv2` is the one corpus model it was gating; its frontier moves to `Stack4` below rather than to `native4d_converts:true` (a later blocker in the same graph), so the scoreboard counts are unchanged — depth moved, stage-completion did not. | Done. |
| ~~`Stack4`~~ | **Landed** (2026-08-30; see `ops-progress.md`'s landing record). Native's `Stack` gets a first-class `Stack4` counterpart, the same `check_dims`-style axis-domain rejection `Concat`/`Select`/`Slice`/`Split_with_sizes`/`Unbind` already had, and a real `Lower.convert` arm. Two corpus models were gated on it: `csatv2`'s frontier (`axis=H`) moves to a later node in the same graph — a `permute.default` mixing a non-unit `T` extent into `H`, an intrinsic Native-only boundary, not further Native4D work; `skresnet18`'s frontier (`axis=D`) now reports the dialect's own axis-domain diagnostic at the same node instead of the lowerer's generic catch-all. Neither reaches `native4d_converts:true`, so the scoreboard counts are unchanged — depth moved, stage-completion did not. | Done. |
| ~~`Repeat4` / `RepeatInterleave4`~~ | **Landed** (2026-08-30; see `ops-progress.md`'s landing record). Both reuse `Shape4`'s own representation for `repeats` rather than a bespoke axis check; `RepeatInterleave4` also gets the `check_dims`-style single-axis rejection for diagnostic consistency, though it is not load-bearing for this particular op (see the landing note). `convit_tiny` is the one corpus model `Repeat4` was gating; its frontier moves to `select_scatter.default` below rather than to `native4d_converts:true`, so the scoreboard counts are unchanged — depth moved, stage-completion did not. | Done. |
| ~~`Select_scatter4`~~ | **Landed** (2026-08-30; see `ops-progress.md`'s landing record). Reuses `Split.Select_scatter`'s shape rule/pixel map unchanged, the same delegation `Select4` makes to `Split.Select`, plus the same `check_dims`-style single-axis rejection. `convit_tiny` — the one corpus model gated on it — moves to an intrinsic `axis T is outside the N/H/W/C dialect` boundary, so the scoreboard counts are unchanged — depth moved, stage-completion did not. | Done. |
| ~~`Softmax4`~~ | **Landed** (2026-08-31; see `ops-progress.md`'s landing record). Reuses `Reduce.Softmax`'s own `output_shape`/`Compute` unchanged; the axis narrows to `Axis4.t` and gets the same `check_dims`-style rejection `Select4`/`Slice4`/`RepeatInterleave4` get. No corpus model was gated on it, so the scoreboard counts are unchanged. | Done. |
| ~~`IndexTensor4`~~ | **Landed** (2026-09-05; see `ops-progress.md`'s landing record). Reuses `Index_tensor.Index_tensor`'s own `output_shape`/`Compute` unchanged, the gathered axis narrowed to `Axis4.t` with the same `check_dims`-style rejection `Select4`/`RepeatInterleave4` get. No corpus model is currently gated on it (`index.Tensor`'s own single-entry Native case, `csatv2`, was already past this point before Native4D had a counterpart at all), so the scoreboard is unchanged. | Done. |
| Live max-pool indices | `Max_pool2d_with_indices`/`Adaptive_max_pool2d_with_indices` are still rejected outright when the index output is live -- a genuine two-output ArgMaxPool/MaxPoolWithIndices4 counterpart, the multi-output plumbing `Unbind` established (`Builder.opN`, `Graph_shape4`'s list form, `Eval_op4`'s per-ordinal dispatch), is the remaining piece. No corpus model is currently gated on it either. | Treat as a full gather/indexing design requiring a counterpart and kernel semantics, not as a reason to declare the operation impossible. |

The exceptional Native4D boundaries are narrower than “not currently in
`Ops4`”: an operation whose semantics intrinsically require `D` or `T` in the
fixed N/H/W/C frame (for example an admitted multi-batch `BatchedMatmul` or
SDPA form) needs an explicit design choice to extend the frame.  Until then it
is Native-only and must carry a typed diagnostic.  This exception does not
apply merely because the direct counterpart has not been written yet.

Native4D also needs payloads for constant folding on 13 models in the
payload-free sweep.  Supplying release payloads is a verification precondition,
not an operation implementation task.  The kernel branch's largest frontier is
the evaluation-depth ceiling (22 models in the checked-in JSONL); it is
independent of operation support and must be reported separately.

## Progress tracking

[`ops-progress.md`](ops-progress.md) is the checked-in, mutable progress
ledger.  It contains the per-operation/configuration breadth matrix, the
stage-qualified corpus scoreboard, and timestamped landing records.  Keep this
file focused on priorities and acceptance rules; do not duplicate volatile
counts or landing history here.

The ledger answers two independent questions: “which operation/configuration
works everywhere?” and “how much of this corpus gets farther?”  A landing must
update both.  If a depth metric does not move, record the newly exposed frontier
instead of treating the landing as having no progress.

## Completion rule and refresh procedure

For each row, follow the priority model above and `ops.md`'s whole-stack
definition.  The landing note must classify the result rather than using the
unqualified word “supported”:

| Landing classification | Required evidence | May count toward |
| --- | --- | --- |
| **Full-stack** | Every surface in the breadth table, including a first-class Native operation, its first-class Native4D counterpart, and a Kernel/Stage-Program representative graph, plus the required oracle and Direct-vs-Symbolic walks. | Full-stack operation breadth and every successful graph-stage metric. |
| **Native-only, deliberately bounded** | ATen boundary where possible, both importers, a first-class Native operation with shape/evaluation/oracle evidence, and a tested Native4D rejection naming an approved intrinsic `D`/`T` or new-kernel-semantic boundary. | Native target/configuration and `native_builds` depth only. |
| **Incomplete** | Any missing importer, operation/counterpart, evaluator, verification path, or unclassified failure. | No support count; retain only as a backlog/frontier observation. |

Binding-closure exceptions need hand-derived bridge/import fixtures and must
be stated in the landing note.  A Kernel depth ceiling or missing payload is
not an operation failure, but it must be recorded so an operation landing does
not overclaim graph breadth.

After every landing, regenerate the corpus signal with
`make pt2.json-model-support`, recensus the 100 JSON graphs, and update both
ledgers in [`ops-progress.md`](ops-progress.md) with:

1. the newly exposed first frontier and its landing classification;
2. the change in `native_builds`, `native4d_converts`, and `kernel_converts`;
3. the number of selected graphs that now complete all three stages, separately
   from graphs that only import to Native; and
4. the per-operation surface matrix described in “Findings and scorecard.”

Refresh `models-selected.yaml` as well if its node-count metadata still
differs from the exported JSON, so the next depth prioritization has one
denominator.
