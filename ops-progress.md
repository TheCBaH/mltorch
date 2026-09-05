# Ops coverage: progress ledger

Newly initialized at this landing — no prior copy of this file was found in
the working tree (`_ai_/` is gitignored, so this ledger does not survive a
fresh clone; see [`todo-ops.md`](todo-ops.md)'s "Progress tracking" section
for what it is supposed to hold going forward). Landing history before this
entry lives in `git log`, not here.

## Stage-qualified corpus scoreboard

Recomputed from `test/data/pt2_json_model_support.jsonl`, recensus
2026-09-04 (after the `unfold.default` landing below), 100 tracked graphs:

| Measure | Result |
| --- | ---: |
| `native_builds:true` | 90 / 100 |
| `native4d_converts:true` | 58 / 100 |
| `kernel_converts:true` | 64 / 100 |
| All three stages succeed | 44 / 100 |

**2026-09-04, `unfold.default`** (full-stack Native; see
[`todo-ops.md`](todo-ops.md)'s own landing note for the complete record).
Closes the `conv1d`/`unfold` P1 slice entirely. `eca_halonext26ts` clears
every one of its 6 `unfold.default` occurrences (all four HaloAttn blocks)
and moves from `unsupported PT2 operator: torch.ops.aten.unfold.default` to
a genuinely different kind of frontier: `matmul.default: both operands must
be rank>=2 and of equal rank, got self=[32, 8, 8, 16] other=[16, 23]` — an
importer decode-time restriction (both `Op_bridge` and `Native_interp`
currently require equal operand rank for `matmul.default`), not a missing
Native operation. All four scoreboard numbers are unchanged (90/58/64/44) —
depth moved, stage-completion did not, since `eca_halonext26ts` was and
remains `native_builds:false`.

**2026-09-04, `conv1d.default`** (full-stack; see
[`todo-ops.md`](todo-ops.md)'s own landing note for the complete record).
`eca_halonext26ts` — the one corpus model gating this P1 row — moves from
`native_builds:false` (blocked on `unsupported PT2 operator: torch.ops.aten.
conv1d.default`) to a NEW frontier, `unfold.default` (also
`native_builds:false`): the model's next node the corpus census doesn't yet
support, exactly the P1 table's own prediction. All four scoreboard numbers
are unchanged (90/58/64/44) — depth moved, stage-completion did not.

**2026-09-04, `Attention.Sdpa` head/batch broadcasting** (full-stack; see
[`todo-ops.md`](todo-ops.md)'s own landing note and
`.ai/attention_design.md` §12 for the complete record). `native_builds`
90 and `native4d_converts` 58 both move for the SAME model,
`mobilenetv5_base`: it was blocked at Native construction on query `H=8`
vs key `H=1` (`native_builds:false`); real ATen `N`/`T`/`D`/`H`
broadcasting (equal, or one side 1) in `Attention.Sdpa.output_shape`/
`Compute` now admits it, and since every one of its SDPA occurrences has
`D = 1` (broadcasting only on `H`), it ALSO clears the pre-existing
Native4D `Sdpa` `D = 1` counterpart (landed 2026-09-02, previously with no
corpus model reaching it at all). It stops at the kernel stage on the
pre-existing, unrelated evaluation-depth ceiling (`kernel_reason:
over_limit`), so `kernel_converts`/all-three-stages are unchanged (64/44).

**2026-09-02, Region computation + `Batched_matmul`'s Native4D `D = 1`
counterpart** (see [`todo-ops.md`](todo-ops.md)'s own landing note for the
full record; this ledger fell behind commit-by-commit and is reconciled here
in one recensus). `native4d_converts` 56→57 and `kernel_converts` 63→64 for
two DIFFERENT reasons, not one model clearing both: `efficientvit_b0`'s
`native4d_converts` flips true (the new `D = 1` counterpart) and it already
had `kernel_converts:true`, so it newly joins all-three-stages (43→44);
`convit_tiny`'s `kernel_converts` flips true independently (Region
computation's cached locals cleared its evaluation-depth ceiling), while its
`native4d_converts` stays false on an unrelated `axis T` boundary — proof
that `kernel_converts` does not require `native4d_converts` (see
`bin/pt2_json_model_support.ml`'s `resolve_branch`, called once per branch
against the shared `native` result, not chained). `lambda_resnet26t` and
`mobilenetv5_base` are unchanged (`native_builds:false`): both were and
remain blocked by the still-open half of this same gap — real ATen
broadcasting (unequal, not merely `>1`, leading extents) in `Batched_matmul`
and `Sdpa` respectively, which this landing did not add.

`Softmax4` (2026-08-31) left all four numbers unchanged — no corpus model
was gated on it, and `make pt2.json-model-support` produced no diff. Before
that, `native_builds` moved 88→89 with the `eye.m` landing: `bat_resnext26ts`
now builds at Native, stopping at an intrinsic `axis D is outside the N/H/W/C
dialect` wall later in the same graph (`native4d_converts`/
`kernel_converts`/all-three-stages unchanged, since that model was and
remains false on all three). Before that, `native_builds` moved 87→88 with
the `Rsub_scalar` + Const-SSA `Sigmoid`/`Rsub_scalar` landing: `convit_tiny`
now builds at Native, stopping at `Repeat`'s then-missing `Repeat4`
counterpart. Unchanged by every other landing below, including
`_to_copy.default`, `Repeat4`/`RepeatInterleave4`, `Select_scatter4`, and
`adaptive_max_pool2d` (most recent before `eye.m`): `Select4`, `Stack4`,
`repeat.default`, `repeat_interleave.self_int`, `copy.default`,
`select_scatter.default`, `_to_copy.default`,
`Repeat4`/`RepeatInterleave4`, `Select_scatter4`, and
`adaptive_max_pool2d.default` each moved a model's *frontier* only —
`edgenext_xx_small` to `bitwise_not.default`, `mvitv2_tiny` to
`index.Tensor`'s multi-entry case, `convit_tiny` through
`select_scatter.default`'s own `Select_scatter4` counterpart to a genuine
intrinsic `axis T is outside the N/H/W/C dialect` wall, and
`bat_resnext26ts` from `adaptive_max_pool2d.default` to `eye.m`.

## Per-operation breadth matrix

| Op | ATen boundary | `Op_bridge` | `Native_interp` | Native Direct/Symbolic | Native4D | Kernel | Representative graph |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `unfold.default` / `Unfold` | **landed**: curated binding (`bin/aten_ops_gen.ml`), `test/native_bridge/shape_ops_test.ml`'s hand-derived oracle (a plain vector, both of the corpus's own HaloAttn spatial-axis cases, and a rank-5-input/rank-6-output case) — `unfold.default` lands in the generated walk's `needs_meta` backlog (no schema default for its required int args), so this hand-derived coverage is the oracle | **landed**: `lib/native_aten_bridge/op_bridge_shape.ml`, no permute wrap needed (`Unfold`'s own axis resolves directly via the shared `dim_axis`/`Aten_shape.axis_of_dim` machinery, then `Unfold.Unfold.dest_of`) | **landed**: `lib/native_interp/native_interp_lower_shape.ml`, sharing the same `dest_of` translation and `axes_for_rank`; `test/native_interp/malformed_test.ml` pins `dest_of`'s one raising case as a typed `` `Rank_over_six `` row, not an escaping exception | **landed**: `lib/native/ops/unfold.ml` (`Unfold`, a fixed one-step-toward-N axis rotation — `source_of`/`dest_of` — expressing how appending an ATen axis relabels Native's fixed six-axis frame), `test/native/unfold_test.ml` + `graph_direct_shape_test.ml` + `graph_symbolic_shape_test.ml` | **rejected, not a gap**: `lib/native4d/domain.ml` + `lower_engine.ml` reject unconditionally — the shift moves real content onto D/T for any ordinary (non-unit H/C) input, the same intrinsic-axis boundary `Batched_matmul`/`Sdpa` have | reaches Native only; Native4D rejects it outright, so no kernel path | `eca_halonext26ts` (native-import frontier; graph does not reach kernel path from here) |
| `conv1d.default` / `Conv1d` | **landed**: curated binding (`bin/aten_ops_gen.ml`), `test/native_bridge/conv_test.ml`'s hand-derived oracle (defaults, non-default stride/padding/dilation, depthwise, general grouping) — `conv1d.default` lands in the generated walk's `needs_meta` backlog (no schema default for a required `int[1]` window), so this is the oracle, not a placeholder | **landed**: `lib/native_aten_bridge/op_bridge_conv.ml`, sharing the new `perm_conv1d` relayout and `make_conv1d_params`/`conv_in_channels` with `op_bridge_decode.ml` | **landed**: `lib/native_interp/native_interp_lower_compute.ml` + `native_interp_decode_conv.ml`'s `conv1d_params`, `test/native_interp/conv_test.ml` pins both the decomposed graph structure and the Native4D legalization | **landed**: `lib/native/ops/conv_conv1d.ml` (`Conv1d`, delegates `output_shape`/`Compute` whole to `Conv2d` with H pinned to the new public `Conv2d.unit_window`), `test/native/conv_test.ml` + `graph_direct_conv_test.ml` + `graph_symbolic_conv_test.ml` | **landed, no new op**: `Conv1d`'s `to_conv2d_params` plugs directly into the pre-existing `forward_conv` (picks `Conv2D`/`DepthwiseConv2D`/`GroupedConv2D` by `groups`, same as `Conv2d`/`Conv2d_padding`), `test/native4d/lower_test.ml` | reaches wherever a legal `Conv2D`/`DepthwiseConv2D`/`GroupedConv2D` graph's own kernel path already reaches | `eca_halonext26ts` (native-import frontier; graph does not reach kernel path from here) |
| `eye.m` / `Eye` + `Eye4` | **landed**: curated binding (`bin/aten_ops_gen.ml`), `test/native_bridge/activation_test.ml`'s `verify_print` runs against it as the oracle (no walk recipe — `n`/`m` are required SymInts with no schema default, the same reason `zeros`/`arange` have none) | **landed**: `lib/native_aten_bridge/op_bridge_shape.ml`, reusing `zeros.default`'s layout/device/pin_memory rejection and FLOAT/DOUBLE dtype dispatch verbatim | **landed**: `lib/native_interp/native_interp_lower_shape.ml` | **landed**: `lib/native/ops/factory.ml` (new `Factory.Eye`, one `S.index_eq` comparing the `w`/`c` coordinates — no new `SEMANTICS` primitive), `test/native/factory_test.ml` | **landed**: `Eye4`, typed `Shape4.t` like `Zeros4` — rows/columns land on `W`/`C` unconditionally (both real dialect axes), so no axis-domain rejection is needed, `test/native4d/{op_json_test,compute_test,fixtures4}.ml` | reaches wherever the reused Native4D factory path already reaches | `bat_resnext26ts` (native-import frontier; graph does not reach kernel path from here) |
| `adaptive_max_pool2d.default` / `Adaptive_max_pool2d` + `Adaptive_max_pool2d_with_indices` | **landed**: curated binding + `Recipe_adaptive`-reused walk oracle (`lib/aten_gen/walk_meta_pool.ml`), `lib/aten/build_archive.sh` gained the structured kernel's source files | **landed**: `lib/native_aten_bridge/op_bridge_pool.ml` (always the two-output form + `discard`) | **landed**: `lib/native_interp/native_interp_lower_compute.ml` | **landed**: `lib/native/ops/pool.ml`, value via `S.max_reduce` (no new primitive), index derived from `max_reduce`/`select`/negation (no new primitive), `test/native/{pool_test,graph_direct_pool_test,graph_symbolic_pool_test,drop_pool_indices_test}.ml` | **landed for the value-only op** (`Adaptive_max_pool2d`, direct reuse like `Adaptive_avg_pool2d`); the two-output op joins `Max_pool2d_with_indices`'s existing "no argmax-pool operation" rejection — the tracked live-index/`IndexTensor4` backlog, not a fresh gap | reaches wherever the reused Native4D pointwise/pool path already reaches for the value-only op; the two-output op is graph-import-only until `Drop_pool_indices` narrows it | `bat_resnext26ts` (native-import frontier; graph does not reach kernel path from here) |
| `select.int` / `Select` | landed pre-existing (`select.int` decomposed into `Select`) | landed | landed | landed (Native `Split.Select`, pre-existing) | **landed 2026-08-30**: `Select4`, `test/native4d/{op_json,fixtures,fixtures4,domain,lower,compute,verify}_test.ml` | reaches `Snapshot4`/lowering; no corpus graph exercises `Select4` end to end into the kernel path yet (csatv2, the only corpus model whose frontier `Select4` clears, is a graph-only CI target — see below) | `csatv2` (native4d frontier only; not a kernel-path graph) |
| `stack.default` / `Stack` | landed pre-existing (own `Stack` node, not a `Concat` decomposition) | landed | landed | landed (Native `Concat.Stack`, pre-existing) | **landed 2026-08-30**: `Stack4`, `test/native4d/{op_json,domain}_test.ml`, `test/native4d/fixtures{,4}.ml` | reaches `Snapshot4`/lowering; no corpus graph exercises `Stack4` end to end into the kernel path yet (csatv2 and skresnet18, the two corpus models whose frontier `Stack4` clears, stay blocked at a later stage — see below) | `csatv2` (native4d frontier only; not a kernel-path graph) |
| `repeat.default` / `Repeat` | landed 2026-08-30: read via `Aten_tensor.shape` only (no ATen kernel call needed), `test/native_bridge/repeat_test.ml` | landed 2026-08-30: `lib/native_aten_bridge/op_bridge_shape.ml` | landed 2026-08-30: `lib/native_interp/native_interp_lower_shape.ml` | landed 2026-08-30: `lib/native/ops/repeat.ml`, `test/native/{repeat_test,graph_direct_shape_test,graph_symbolic_shape_test}.ml` | none yet — no `Repeat4` counterpart; tracked as backlog, not an intrinsic-axis boundary (see landing note) | not reachable without a Native4D counterpart | `convit_tiny` (native-import frontier only; graph does not reach kernel path from here) |
| `repeat_interleave.self_int` / `RepeatInterleave` | landed 2026-08-30, `dim` required (`dim=None` rejected): same shape-only read, `test/native_bridge/repeat_interleave_test.ml` | landed 2026-08-30: `lib/native_aten_bridge/op_bridge_shape.ml` | landed 2026-08-30: `lib/native_interp/native_interp_lower_shape.ml` | landed 2026-08-30: `lib/native/ops/repeat.ml` (shares the file with `Repeat`), `test/native/{repeat_test,graph_direct_shape_test,graph_symbolic_shape_test}.ml` | none yet — no `RepeatInterleave4` counterpart; same backlog status as `Repeat4` | not reachable without a Native4D counterpart | `convit_tiny` (native-import frontier only; graph does not reach kernel path from here) |
| `copy.default` / `Expand` (direct bind, no new op) | **landed this entry**: `self`'s live shape read only, no ATen kernel call, `test/native_bridge/copy_test.ml` | **landed this entry**: `lib/native_aten_bridge/op_bridge_shape.ml` | **landed this entry**: `lib/native_interp/native_interp_lower_shape.ml` | reuses `Pointwise.Expand`'s own pre-existing Direct/Symbolic — no new Compute, `test/native_interp/copy_test.ml` pins the built graph structure | **full-stack already**: `Expand4` pre-exists, admitted unconditionally by `Domain.check_node` | reaches wherever `Expand4`'s own kernel path already reaches | `convit_tiny` (native-import frontier only; graph does not reach kernel path from here) |
| `select_scatter.default` / `Select_scatter` | **landed this entry**: real `at::select_scatter` now a curated binding (`bin/aten_ops_gen.ml`), `test/native_bridge/shape_ops_test.ml`'s `verify_print` runs against it as the oracle | **landed this entry**: `lib/native_aten_bridge/op_bridge_shape.ml` | **landed this entry**: `lib/native_interp/native_interp_lower_shape.ml` | **landed this entry**: `lib/native/ops/split.ml` (new `Split.Select_scatter`, reuses `Select.output_shape` for the `src`-shape check, `S.index_eq`/`S.select` for the branch — no new `SEMANTICS` primitive), `test/native/{slice_select_test,graph_direct_pad_slice_test,graph_symbolic_pad_slice_test}.ml` | none yet — no `Select_scatter4` counterpart; tracked as backlog the same deliberate way `Repeat`/`RepeatInterleave` were at their own landing | not reachable without a Native4D counterpart | `convit_tiny` (native-import frontier only; graph does not reach kernel path from here) |
| `rsub.Scalar` / `Rsub_scalar` | **landed this entry**: real `at::rsub.Scalar` now a curated binding, `test/native_bridge/dispatch_test.ml`'s `verify_print` runs against it as the oracle (default and non-default `alpha`) | **landed this entry**: `lib/native_aten_bridge/op_bridge_pointwise.ml` | **landed this entry**: `lib/native_interp/native_interp_lower_compute.ml` | **landed this entry**: `lib/native/ops/pointwise_binary.ml` (new `Rsub_scalar`, `other - alpha * x`, no new `SEMANTICS` primitive), `test/native/pointwise_test.ml` + `graph_direct_pointwise_test.ml` + `graph_symbolic_pointwise_test.ml` | **full-stack this entry**: reuses Native's own payload directly (no axis param, same treatment as `Add_scalar`/`Mul_scalar`/`Pow`), `test/native4d/{op_json_test,compute_test,fixtures4}.ml` | reaches wherever the reused Native4D pointwise path already reaches | `convit_tiny` (native-import frontier; also the corpus's own `1 - sigmoid(...)` use) |
| `_to_copy.default` / `To_copy` | **landed this entry**: real `at::_to_copy` now a curated binding (`bin/aten_ops_gen.ml`), `test/native_bridge/to_copy_test.ml`'s `verify_print` runs against it as the oracle for the `dtype`-omitted case (the only case `Interp_decode.scalar_type_opt_arg` can decode) | **landed this entry**: `lib/native_aten_bridge/op_bridge_shape.ml` | **landed this entry**: `lib/native_interp/native_interp_lower_shape.ml` | **landed this entry**: `lib/native/ops/pointwise_unary.ml` (new `Pointwise.To_copy`, three-way `Bool`/`Float`/`Long` target domain; `Long` needed a genuinely new `SEMANTICS` primitive, `trunc`), `test/native/pointwise_test.ml` + `graph_direct_pointwise_test.ml` + `graph_symbolic_pointwise_test.ml` | **full-stack this entry**: reuses Native's own payload directly (no axis param, same treatment as `Rsub_scalar`), `test/native4d/{op_json_test,compute_test,fixtures4}.ml` | reaches wherever the reused Native4D pointwise path already reaches | `edgenext_xx_small` and `mvitv2_tiny` (native-import frontier for both; neither reaches the kernel path from here) |
| `repeat.default` / `Repeat4` (Native4D counterpart) | pre-existing (this entry adds only the Native4D counterpart) | pre-existing | pre-existing | pre-existing | **landed this entry**: `Repeat4`, reusing `Shape4.t` for `repeats` directly (the same treatment `Expand4.size` gets) — `Domain.check_node` admits `Repeat` unconditionally, `test/native4d/{op_json_test,compute_test,fixtures4,domain_test,lower_test}.ml` | reaches wherever a legal `Repeat4` graph's own kernel path already reaches | `convit_tiny` (native4d frontier only; graph does not reach kernel path from here) |
| `repeat_interleave.self_int` / `RepeatInterleave4` (Native4D counterpart) | pre-existing | pre-existing | pre-existing | pre-existing | **landed this entry**: `RepeatInterleave4`, `Axis4.t`-typed axis with the same `check_dims`-style rejection `Select4`/`Slice4` get, `test/native4d/{op_json_test,compute_test,fixtures4,domain_test,lower_test}.ml` | reaches wherever a legal `RepeatInterleave4` graph's own kernel path already reaches | no corpus model currently reaches this boundary (ConViT's own `repeat_interleave.self_int` occurrences sit behind `Repeat4`'s now-cleared frontier, in `native_builds`, not blocked at Native4D) |
| `select_scatter.default` / `Select_scatter4` (Native4D counterpart) | pre-existing | pre-existing | pre-existing | pre-existing | **landed this entry**: `Select_scatter4`, reusing `Split.Select_scatter`'s shape rule/pixel map unchanged, `Axis4.t`-typed axis with the same `check_dims`-style rejection `Select4` gets, `test/native4d/{op_json_test,compute_test,fixtures4,domain_test,lower_test}.ml` | reaches wherever a legal `Select_scatter4` graph's own kernel path already reaches | `convit_tiny` (native4d frontier only; graph does not reach kernel path from here) |
| `softmax.int` / `Softmax4` (Native4D counterpart) | pre-existing | pre-existing | pre-existing | pre-existing | **landed 2026-08-31**: `Softmax4`, reusing `Reduce.Softmax`'s own `output_shape`/`Compute` unchanged, `Axis4.t`-typed axis with the same `check_dims`-style rejection `Select4`/`Slice4`/`RepeatInterleave4` get — the whole domain gate here, since the output shape never depends on which axis is reduced, `test/native4d/{op_json_test,compute_test,fixtures4,domain_test,lower_shape_test}.ml` | reaches wherever a legal `Softmax4` graph's own kernel path already reaches | no corpus model currently reaches this boundary (every model that reaches `softmax.int` is transformer-shaped and already stops at an earlier `D`-axis/batched-matmul limit, per `matmul_softmax_design.md` §3) |
| `matmul.default` / `Batched_matmul4` (Native4D counterpart) | pre-existing (`Batched_matmul` itself landed 2026-08-29, see its own earlier row) | pre-existing | pre-existing | pre-existing | **landed 2026-09-02**: admitted at `D = 1` (no `Ops4` payload, no `4`-suffixed variant — the op names no axis and carries no shape); `Domain.check_node`'s arm is conditional on `D > 1`, `.ai/native4d_design.md` §7.4 | reaches wherever a legal `Batched_matmul` graph's own kernel path already reaches | `efficientvit_b0` — newly reaches ALL THREE stages (`native_builds`/`native4d_converts`/`kernel_converts`). **Still incomplete**: real broadcasting (unequal leading extents) stays a Native-level rejection, corpus evidence `lambda_resnet26t` (`native_builds:false`) |
| `scaled_dot_product_attention.default` / `Sdpa4` (Native4D counterpart) | pre-existing | pre-existing | pre-existing | pre-existing (now Region-authored, see the Region-computation note below) | **landed 2026-09-02**: admitted at `D = 1`, reusing `Attention.Sdpa`'s payload verbatim through `Region_computation4`'s `native_op`, `.ai/native4d_design.md` §7.9 | reaches wherever a legal `Sdpa` graph's own kernel path already reaches | **landed corpus evidence 2026-09-04** (below): `mobilenetv5_base`, once the Native-level `H`-broadcast rejection was lifted — every occurrence is `D = 1`. |
| `matmul.default` / `Batched_matmul` real broadcasting | pre-existing binding/importers (only the shape rule changed) | pre-existing | pre-existing | **landed 2026-09-04**: `output_shape` broadcasts `N`/`T`/`D`/`H` per axis (equal, or one side 1), reusing `Pointwise_binary.broadcast_output_shape`'s per-axis rule restricted to the four batch axes; `Compute` reads each operand through `Pointwise_binary.broadcast_coord` against ITS OWN shape rather than the output coordinate directly. `test/native/linear_test.ml` (hand-computed mat2-broadcasts-H fixture) | `lib/native4d/domain.ml`'s `check_batched_matmul` fixed to read the BROADCAST (output) `D` via both operands, not `input`'s own `D` alone — `test/native4d/{fixtures,domain_test}.ml`'s new broadcast-domain fixture pins this | reaches wherever a legal `Batched_matmul` graph's own kernel path already reaches | `lambda_resnet26t` — moves off this frontier to `conv3d.default` (deferred backlog). **Not done**: the ATen-oracle walk (`lib/aten_walk_recipes/recipe_matmul.ml`) does not sweep a broadcast configuration against real ATen (same scoping as `Add`/`Mul`'s own walks) |
| `scaled_dot_product_attention.default` / `Sdpa` real broadcasting | pre-existing binding/importers (neither checks head/batch equality; only Native's own shape rule changed) | pre-existing | pre-existing | **landed 2026-09-04**: `output_shape` broadcasts `N`/`T`/`D`/`H` per axis across query/key/value, chained the same way real ATen's two matmuls chain it (`batch_shape` returns `(qk_batch, full_batch)`); both `Compute` paths (`Legacy_pixel`, `Computation`) read every operand through `broadcast_coord` against its own shape. Real-ATen grounded: two `verify_print` fixtures in `test/native_bridge/sdpa_test.ml` (`H` and `D` broadcast, `enable_gqa=false`, dispatching to ATen's `math` backend since the flash gate requires equal heads/batch without GQA) agree at the EXISTING `1e-5` tolerance. `test/native/sdpa_test.ml` adds shape-level broadcast/rejection coverage plus a bit-for-bit two-heads-broadcast-against-one-shared-kv correctness fixture. `.ai/attention_design.md` §12 | `lib/native/region_computation.ml`'s SDPA sanity check fixed the same way (recomputes the real broadcast shape via a new `Attention.Sdpa.batch_shape` instead of assuming `output_shape = query.shape`) | reaches wherever a legal `Sdpa` graph's own kernel path already reaches | `mobilenetv5_base` — moves from `native_builds:false` to `native_builds:true` AND `native4d_converts:true` (its `D = 1` occurrences newly clear the pre-existing Native4D `Sdpa` counterpart above), stopping at the kernel stage's unrelated evaluation-depth ceiling. **Not done**: `Recipe_sdpa` was not widened to sweep broadcast configurations (same scoping as `Batched_matmul`'s own broadcast landing) |

## Landing records

### 2026-09-04 — `unfold.default`, closing the conv1d/unfold P1 slice

Closes the `unfold.default` half of [`todo-ops.md`](todo-ops.md)'s P1 row
for `eca_halonext26ts` (6 occurrences, its frontier after `conv1d.default`
landed). ATen's `unfold(Tensor(a) self, int dimension, int size, int step)
-> Tensor(a)` is a sliding, possibly-overlapping window view: the named
axis shrinks to the window COUNT and a genuinely new axis (extent `size`)
is appended holding each window's own elements — real data movement when
windows overlap, not a reshape, and RANK-INCREASING, which Native's
always-six-axis frame has no direct precedent for.

**What landed**

- `lib/native/ops/unfold.ml`: new `Unfold` Native op. Native has no rank —
  every tensor is always six axes, right-aligned per ATen rank by
  `Aten_shape.of_aten` — so appending an ATen axis RELABELS every axis
  already in the frame: `of_aten`'s own rule right-aligns a rank-(r+1)
  tensor one slot further left than a rank-r one, which works out to a
  fixed rotation over `Axis.t` per `Axis.all = [N;T;D;H;W;C]` — content on
  `T` moves to `N`, `D` to `T`, `H` to `D`, `W` to `H`, `C` to `W`, and the
  fresh window axis always lands on `C`. `source_of`/`dest_of` express this
  rotation directly (total on `{N,T,D,H,W}` / `{T,D,H,W,C}` respectively,
  raising on the one axis each has no mapping for); `params.axis` names the
  OUTPUT axis holding the window COUNT (never `C`, which the op always
  reserves for the window offset). The single precondition —
  `x_shape`'s own `N` must already be unit — stands in for "ATen rank <=
  5" without the op needing to know a rank at all. `output_shape` reuses
  `Window_axis.output_extent` (kernel=`size`, stride=`step`, no
  padding/dilation) for the window-count arithmetic rather than
  restating it; `Compute.pixel` needs no new `SEMANTICS` primitive, just
  `index_add`/`index_scale`/`assume_index`, the same idiom `Conv2d`'s own
  channel-group index uses — every window is fully interior by
  construction (proved algebraically in the file's own comment), so
  unlike `Conv2d`'s clipped kernel range there is no padding case to
  clamp against at all.
- `lib/native/shape_error.{ml,mli}`: new `Unfold` module, two faults
  (`Reserved_axis`, `Rank_exceeded`) for the two typed rejections above.
- `lib/native/{graph_ir,graph_shape,eval_op,graph_builder}.{ml,mli}`: the
  usual per-op wiring (alphabetical, between `Unbind` and
  `Upsample_bilinear2d`). `lib/native/transform/output_transfer.ml`:
  `Reindexing` — the window/offset pair selects exactly one source element
  per output, a gather, the same argument `Unbind`'s own selection makes.
- **No permute wrap needed in either importer** — unlike `Conv1d`, whose
  bridge/importer arms insert an explicit relayout because `Conv2d`'s
  compute has fixed semantic axis requirements, `Unfold`'s own `axis`
  parameter resolves directly through the SAME `dim_axis`/
  `Aten_shape.axis_of_dim` (`Op_bridge`) and `axes_for_rank`
  (`Native_interp`) machinery `Select`/`RepeatInterleave`/`Unbind` already
  use, translated through `Unfold.Unfold.dest_of` to name the OUTPUT axis
  instead of `dimension`'s own. `dest_of` raises only when `dimension`
  resolves to the frame's `N` (self already occupies all six axes at its
  outermost position, so the appended axis has no room at all regardless
  of `x_shape`'s actual values) — `Op_bridge`'s arm catches it as
  `` `Validation_failure `` (the existing `try...with Invalid_argument`
  idiom conv arms use), `Native_interp`'s arm catches it and reports the
  SAME `` `Rank_over_six `` row a genuinely too-large declared rank already
  gets (`Bad_dimension`), not a new variant — both are "this tensor does
  not fit the six-axis frame," so one row covers both origins.
- **Native4D**: unconditionally rejected, not a missing counterpart —
  `lib/native4d/domain.ml` (the "dialect does not have it at all" bucket,
  alongside `Index_tensor`) and `lower_engine.ml`'s matching fallback.
  `Unfold`'s own shift moves whatever real (non-unit) content sits on the
  input's `H`/`C` axes onto the output's `D`/`W`, and Native4D forces `T`/`D`
  unit always — a real dialect-incompatible move for any input whose `H`/`C`
  actually hold data, which is the ordinary case (an unfolded spatial or
  channel axis is rarely unit). Same intrinsic-axis-boundary classification
  `Batched_matmul`'s multi-batch form and `Sdpa`'s own `D` axis get
  (`.ai/native4d_design.md` §8).
- **ATen curated binding**: `bin/aten_ops_gen.ml` gains `op "unfold"` — a
  `variants: method` op (like `select.int`/`transpose.int`/`permute`,
  already curated the same way), whose kernel is already linked.
- **Real-ATen oracle**: `unfold.default` lands in the generated ATen walk's
  own `needs_meta` backlog (`test/native_walk_coverage_test.ml`) — required
  int args with no schema default, the same reason `conv1d.default`/
  `eye.m`/`rsub.Scalar` are there too — so real-ATen agreement is
  hand-verified instead: `test/native_bridge/shape_ops_test.ml`'s new
  `unfold_verify` harness (`verify_print`, comparing `Op_bridge` +
  `Eval_direct` against real ATen dispatch on the SAME decoded node) checks
  a plain vector, both of the corpus's own HaloAttn spatial-axis cases
  (`dimension=2` and `dimension=3` on a rank-4 tensor, `size=12 step=8`, the
  exact configuration `eca_halonext26ts` uses), a negative `dimension`, and
  — the most demanding case — a rank-5 input producing a rank-6 output
  (every frame axis in use, the shape the corpus's own SECOND chained
  `unfold` call actually produces). All six agreed with real ATen on the
  first run.
- Tests: `test/native/unfold_test.ml` (Direct compute: a plain-vector case
  and an H-onto-D case matching the corpus's own axis choice, plus both
  typed rejections), `test/native/graph_direct_shape_test.ml` +
  `graph_symbolic_shape_test.ml` (one-node graph build/eval, Direct-vs-
  Symbolic agreement), `test/native_bridge/shape_ops_test.ml` (the real-ATen
  oracle above), `test/native_interp/malformed_test.ml` (`dest_of`'s one
  raising case reaches a typed row, not an escaping exception).

**Classification**: Full-stack for the NATIVE operation, per
[`todo-ops.md`](todo-ops.md)'s completion rule — ATen boundary, both
importers, Native Direct/Symbolic, and real-ATen verification evidence, all
enumerated above. Native4D is a deliberate, tested, typed rejection
(`` `Unsupported_op ``) rather than a missing counterpart — an intrinsic
`D`/`T`-axis boundary in the same sense `Batched_matmul`'s multi-batch form
and `Sdpa`'s own `D` axis are, per the completion table's "Native-only,
deliberately bounded" tier.

**Corpus effect**: `eca_halonext26ts` — the one corpus model gating this
row — clears every one of its 6 `unfold.default` occurrences (all four
HaloAttn blocks) and moves to a genuinely different kind of frontier:
`matmul.default: both operands must be rank>=2 and of equal rank, got
self=[32, 8, 8, 16] other=[16, 23]` — an IMPORTER decode-time restriction
(`Op_bridge`'s and `Native_interp`'s `matmul.default` arms currently both
require equal operand rank), not a missing Native operation.
`native_builds`/`native4d_converts`/`kernel_converts`/all-three-stages
unchanged (90/58/64/44) — depth moved, stage-completion did not, since
`eca_halonext26ts` was and remains `native_builds:false`.

**Next**: `matmul.default`'s unequal-operand-rank case is a new,
not-yet-scoped gap — worth investigating given today's earlier
`Batched_matmul`/`Sdpa` real-ATen-broadcasting landings, but a different
piece of work from this P1 slice, not a continuation of it. Otherwise
unchanged: live max-pool indices/`IndexTensor4`, `lstm.input`,
`conv3d.default` (`lambda_resnet26t`'s frontier), or P1's remaining
one-model slices (`im2col`/`col2im`, `upsample_bicubic2d`).

### 2026-09-04 — `conv1d.default`, the first half of the conv1d/unfold P1 slice

Closes the `conv1d.default` half of [`todo-ops.md`](todo-ops.md)'s P1 row
for `eca_halonext26ts` (5 occurrences, its first frontier). ATen's
`conv1d(Tensor input, Tensor weight, Tensor? bias=None, int[1] stride=1,
int[1] padding=0, int[1] dilation=1, int groups=1) -> Tensor` is the one
kernel/stride/dilation-per-window family with a single spatial axis, so the
design question was where that axis lands in Native's always-6D frame.

**What landed**

- `lib/native/ops/conv_conv2d.ml`: a new public `Conv2d.unit_window`
  (kernel=1, stride=1, no padding, dilation=1) — an axis this small always
  keeps its extent unchanged. Replaces `lib/native4d/lower_engine.ml`'s own
  former PRIVATE copy of the same record (used for `Linear`'s 1x1-convolution
  legalization), now `Conv.Conv2d.unit_window` in both places: one
  definition instead of two that could drift.
- `lib/native/ops/conv_conv1d.ml`: new `Conv1d` module. Its own `params`
  carries only a `w : Conv2d.axis_window` (no `h` field at all — this op
  names no second spatial axis, so unlike `Conv2d_padding`'s own params
  there is nothing for a stray value to misrepresent), plus `in_channels`
  and `groups`. `output_shape`/`Compute` translate to a full `Conv2d.params`
  (`to_conv2d_params`, `h = Conv2d.unit_window`) and delegate whole to
  `Conv2d.output_shape`/`Conv2d.Compute` — the same delegation shape
  `Conv2d_padding` already uses for `Conv2d` (`.ai/native_add_op.md`'s own
  worked exception: reuse the *implementation*, not the *node*). `Conv1d`
  is still its own distinct `Graph_ir` constructor/JSON tag/bridge arm — one
  node per ATen op, no decomposition.
- `lib/native/graph_ir.{ml,mli}`, `graph_shape.ml`, `eval_op.ml`,
  `transform/output_transfer.ml` (`Continuous`, alphabetical next to
  `Conv2d`), `graph_builder.{ml,mli}`: the usual per-op wiring, mirroring
  `Conv2d`'s own arms with `Conv2d.unit_window`'s `h` folded in through
  `to_conv2d_params` rather than restated.
- **Layout**: both ATen operands (`input` rank-3 `[N,C,L]`, `weight` rank-3
  `[Cout,Cin/groups,K]`) share the same positional pattern under
  `Aten_shape.of_aten`'s mechanical right-alignment — `[role0, channel,
  spatial]` — so ONE new permutation, `perm_conv1d` (swaps `N<->H` and
  `W<->C`; a product of two disjoint transpositions, hence its own inverse),
  relayouts `x`/`weight` in and the raw `Conv1d` output back to the generic
  `[N,C,L]` convention, in both `lib/native_aten_bridge/op_bridge_decode.ml`
  and `lib/native_interp/native_interp_decode.ml`. Verified against real
  ATen (below) rather than only reasoned about.
- `lib/native_aten_bridge/op_bridge_conv.ml` + `op_bridge_decode.ml`: the
  `conv1d.default` dispatch arm; `make_conv1d_params` and a new shared
  `conv_in_channels` helper factored out of `make_conv2d_params` (both now
  call it) rather than a second copy of the bounded-product rule.
  `op_bridge_error.ml` gains `` `Conv1d_invalid_weight_rank `` and
  `` `Invalid_w_arg `` (the single-int twin of `` `Invalid_hw_arg ``, since a
  1-D op has no second axis to broadcast a scalar list into).
- `lib/native_interp/native_interp_decode_conv.ml`'s `conv1d_params` +
  `native_interp_decode.ml`'s `sizes_rank_3`/`w1`/`` `Bad_w_arity ``: the
  mirror import arm. `native_interp_decode.ml`'s `is_nontrivial_node` gains
  `conv1d.default` too, so the importer groups its 4-node decomposition
  under one label the same way `conv2d.default`'s already is.
- **Native4D**: no new op, payload, or domain arm beyond admitting `Conv1d`
  unconditionally into the "legalizations that constrain nothing here"
  bucket (`lib/native4d/domain.ml`, alongside `Conv2d`/`Conv2d_padding`,
  which get no `groups` check there either — that decision is
  `forward_conv`'s). `lib/native4d/lower_engine.ml`'s `Conv1d` arm calls
  `Conv.Conv1d.to_conv2d_params` and hands the result to the EXISTING
  `forward_conv` helper unchanged — the same "map onto an existing op after
  translating parameters" legalization `Bmm`→`Batched_matmul` and
  `Linear`→`Conv2D4` already use, just at this layer instead of Native's
  own. `Conv1d` therefore reaches `Conv2D`/`DepthwiseConv2D`/`GroupedConv2D`
  depending on `groups`, exactly as `Conv2d` does.
- **Real-ATen oracle**: `bin/aten_ops_gen.ml` gains `op "conv1d"` (a curated
  binding; its kernel is already linked, sharing `conv2d`'s translation
  unit). `conv1d.default` lands in the generated ATen walk's own
  `needs_meta` backlog (`test/native_walk_coverage_test.ml`) — a required
  `int[1]` window with no schema default, the same reason `eye.m`/
  `rsub.Scalar`/`select_scatter.default` are there too — so real-ATen
  agreement is hand-verified instead:
  `test/native_bridge/conv_test.ml`'s new `importer_vs_aten_1d` harness (the
  rank-3 twin of the existing `conv2d.default` one, sharing its
  `Interp_dispatch`-vs-`Native_interp`-plus-`Eval_direct` comparison shape)
  checks defaults, non-default stride/padding/dilation together, depthwise,
  and general grouping — all `Ok` against real ATen on the first run, which
  is the strongest evidence the `perm_conv1d` layout derivation is right,
  not merely plausible.
- Tests: `test/native/conv_test.ml` (box-sum + grouped Direct fixtures,
  proving delegation through the pinned H window), `test/native/
  graph_direct_conv_test.ml` + `graph_symbolic_conv_test.ml` (one-node graph
  build/eval, Direct-vs-Symbolic agreement), `test/native4d/{fixtures,
  lower_test}.ml` (`Conv1d` -> `Conv2D` with the pinned unit window visible
  on `h`), `test/native_bridge/conv_test.ml` (the real-ATen oracle above),
  `test/native_interp/conv_test.ml` (decomposed-graph structure dump +
  `Conv1d` -> `Conv2D`/`DepthwiseConv2D` through the real Native4D
  legalization path, not a synthetic fixture).

**Classification**: Full-stack, per [`todo-ops.md`](todo-ops.md)'s
completion rule — ATen boundary, both importers, Native Direct/Symbolic, a
full Native4D legalization (onto pre-existing ops, the same route `Bmm`/
`Linear` already take), and real-ATen verification evidence, each with test
coverage enumerated above.

**Corpus effect**: `eca_halonext26ts` — the one corpus model gating this row
— moves from `unsupported PT2 operator: torch.ops.aten.conv1d.default` to
`unsupported PT2 operator: torch.ops.aten.unfold.default`, exactly the P1
table's own prediction for what comes next. `native_builds`/
`native4d_converts`/`kernel_converts`/all-three-stages unchanged
(90/58/64/44) — depth moved, stage-completion did not, since
`eca_halonext26ts` was and remains `native_builds:false`.

**Next**: `unfold.default` (6 occurrences, now `eca_halonext26ts`'s own
frontier) closes the rest of this P1 row. Otherwise unchanged: live max-pool
indices/`IndexTensor4`, `lstm.input`, `conv3d.default`
(`lambda_resnet26t`'s frontier), or P1's remaining one-model slices
(`im2col`/`col2im`, `upsample_bicubic2d`).

### 2026-09-04 — `Attention.Sdpa` head and batch broadcasting

Closes the "Configuration and transform gaps" table's own SDPA row in
[`todo-ops.md`](todo-ops.md) — `mobilenetv5_base`'s corpus evidence
(query `H=8` vs key `H=1`), the one piece of the matmul/SDPA broadcasting
gap left open after `Batched_matmul`'s own broadcasting landing earlier the
same day. Full record: `.ai/attention_design.md` §12.

**What landed**

- `lib/native/ops/attention.ml`: `batch_axes = [N;T;D;H]`,
  `broadcast_extent`/`broadcast_batch` (per-axis "equal, or one side 1",
  reusing `Matmul.Batched_matmul`'s own rule), and `batch_shape ~query_shape
  ~key_shape ~value_shape : (qk_batch, full_batch) Err.t` — `qk_batch` is
  `query @ key^T`'s own broadcast batch (what the mask broadcasts against,
  added BEFORE the second matmul), `full_batch` is `qk_batch` broadcast
  again with `value` (the real output batch, the second matmul). `Ev = E`
  (`C`) and key/value's shared sequence extent (`W`) stay STRICT equality —
  neither is a batch axis. `output_shape` now returns `full_batch`, not
  `query_shape` (a real behavior change whenever query's own extent on a
  batch axis is 1 while the broadcast output is larger); `total_work_bounded`
  and the output-numel bound both now read `full_batch`, not `query_shape`'s
  own — reading `query_shape` directly would under-count exactly the way
  `Batched_matmul`'s own Native4D domain-check bug once did for `D`.
- `Legacy_pixel.pixel` (test-only oracle) and `Computation.program`
  (authoritative, Region-authored): every operand read (query/key/value/mask)
  now reduces the output coordinate against ITS OWN shape via
  `broadcast_coord` before `load`, mirroring `Matmul.Batched_matmul.Compute`
  — a no-op when every operand agrees, so every pre-existing fixture stays
  bit-identical (confirmed: only cosmetic `Symbolic` print-golden changes,
  literal `0` in place of the general output-index variable on already-unit
  axes, promoted in `test/native/symbolic_test.ml`). `Legacy_pixel.pixel`
  gained a genuinely new `~value_shape` parameter — it used to always equal
  `~key_shape` (implied by the old strict equality) and no longer does.
- `lib/native/shape_error.ml`/`.mli`: new `Batch_mismatch` variant (mirrors
  `Matmul.Batched_matmul`'s `Batch_mismatch`/`Contract_mismatch` split) for
  the four broadcastable axes' genuine-incompatibility case ("or one must be
  1" in the message); `Extent_mismatch`'s STRICT wording is unchanged for
  `C`/`W`, which are never batch axes.
- `lib/native/region_computation.ml`: `check_output`'s signature widened
  from "compare against one fixed operand's shape" to "compare against an
  explicit `~expected` shape" — the SDPA arm can no longer assume
  `output_shape = query.shape` (true before this landing, not after), so it
  recomputes the real expected shape via the new `Attention.Sdpa.batch_shape`.
  `Rms_norm`/`Layer_norm`/`Softmax`'s call sites are mechanically updated to
  pass `~expected:x.Tensor_sig.shape` (identical behavior).
- **Real-ATen grounding**: two hand-derived `verify_print` fixtures in
  `test/native_bridge/sdpa_test.ml` — query `H=2` vs key/value `H=1`, and
  query batch=2 vs key/value batch=1, both `enable_gqa=false` — confirm real
  ATen agrees with this row's `math`-backend-shaped computation at the
  EXISTING `1e-5` tolerance (`Verify.atol_for_target`'s one entry). Grounded
  in real ATen source (`attention.cpp`'s `_scaled_dot_product_attention_math`
  calls plain `at::matmul`, which broadcasts, whenever `enable_gqa=false`;
  `sdp_utils_cpp.h`'s `check_batch_size_and_num_heads_dense` requires equal
  heads/batch for the fused/flash kernel unless GQA is enabled, so this
  configuration was already off the flash kernel in real ATen too — landing
  on `math`, which is structurally what this op already computes, per
  attention_design.md §6). `Recipe_sdpa` (the ATen-oracle fuzz walk) was
  deliberately NOT widened — same scoping precedent as `Batched_matmul`'s
  own broadcast landing (`.ai/matmul_softmax_design.md` §6's "not done"
  note).
- `test/native/sdpa_test.ml`: rewrote the "N and T must agree" test (no
  longer true — replaced with an "N/T/D/H broadcast" test proving both the
  new-legal broadcast cases AND a still-rejected genuine mismatch, `N=2` vs
  `N=3`), and added a bit-for-bit correctness fixture (query with two
  DIFFERENT per-head queries against one shared, broadcast key/value pair)
  proving broadcasting COMPUTES the right per-head answer, not merely that
  the shape check now accepts the input.
- `.ai/attention_design.md`: new §12 (this landing's full record); §9
  updated (was still describing Native4D as an "unconditional typed
  rejection", stale since the 2026-09-02 `D = 1` counterpart landed — fixed
  in the same change since this landing gives that counterpart its first
  corpus model); §10's tolerance-policy note extended to distinguish what
  changed (Native now admits broadcast/off-flash configurations, hand-
  verified) from what did not (`Recipe_sdpa` itself, still flash-only); §11's
  "no model coverage" / "no Native4D execution path" bullets corrected (both
  were already stale before this landing, from the 2026-09-02 counterpart,
  and are fixed here since this is the first landing to touch this file
  since). `.ai/native4d_design.md` §7.9's own "no model coverage" line
  updated to match.

**Classification**: Full-stack, per [`todo-ops.md`](todo-ops.md)'s
completion rule — real ATen semantics (not merely a relaxed shape check),
both `Compute` paths, a Native4D corpus win as a side effect (the
already-landed `D = 1` counterpart), and real-ATen verification evidence,
each enumerated above.

**Corpus effect**: `mobilenetv5_base` moves from `native_builds:false`
(blocked at Native construction on this exact `H` mismatch) to
`native_builds:true` AND `native4d_converts:true` in the same landing — its
SDPA occurrences are all `D = 1` (broadcasting only on `H`), so the
already-landed Native4D `Sdpa` `D = 1` counterpart (2026-09-02) has its
first real corpus model. It stops at the kernel stage on the pre-existing,
unrelated evaluation-depth ceiling (`kernel_reason: over_limit`).
`native_builds` 89→90, `native4d_converts` 57→58; `kernel_converts`/
all-three-stages unchanged (64/44).

**Next**: this closes the matmul/SDPA broadcasting gap entirely (both
halves now landed). Remaining work: live max-pool indices/`IndexTensor4`,
`lstm.input` (36 occurrences, Sequencer2D's own first frontier),
`conv3d.default` (`lambda_resnet26t`'s own new frontier since the matmul
broadcasting landing), or P1's remaining one-model slices
(`conv1d`/`unfold`, `im2col`/`col2im`, `upsample_bicubic2d`).

### 2026-08-31 — `Softmax4`, the Native4D counterpart to Native's `Softmax`

Closes the last remaining row of [`todo-ops.md`](todo-ops.md)'s "Remaining
Native4D counterpart backlog" table with a landed counterpart (live max-pool
indices/`IndexTensor4` is now the only row left, and it is scoped as an
open-ended design, not a one-session slice).

**What landed**

- `lib/native4d/ops4.ml`: `Softmax4`, appended to the "reduction" section
  after `Vector_norm_keepdims`. `params = { axis : Axis4.t }` — ONE named
  axis, the same shape `RepeatInterleave4`'s payload takes. Unlike the four
  keep-dimensions ops in that section, softmax never drops rank at all, so
  there is no repack to defer to at this layer — the shape arm simply
  carries `x`'s own shape through unchanged.
- `lib/native4d/op.ml`: `Softmax4` constructor (between `Slice4` and
  `Split_with_sizes4`, global alphabetical order) + registry entry.
- `lib/native4d/domain.ml`: `Softmax` moved out of the "dialect does not
  have it at all" bucket (shared with `Index_tensor`) into its own new
  `check_dims node [params.axis]` arm, the `Select`/`Select_scatter`/
  `Slice`/`Stack`/`RepeatInterleave` treatment. Unlike `RepeatInterleave`'s
  arm (diagnostic consistency only, not load-bearing there), this one IS
  the whole domain gate: softmax's output shape never depends on which axis
  is reduced, so there is no separate shape-consequence rejection the way
  `Select`'s pair has.
- `lib/native4d/lower_engine.ml`: the conversion arm, converting only the
  axis KEY (`dims4`, `Select_scatter`'s own pattern) — no post-hoc output
  re-check, because the output is `x`'s shape unchanged: if `x` is already
  four-axis so is the output, whichever axis is reduced. Removed from the
  "rejected by `Domain.check` before the walk starts" bucket (now only
  `Index_tensor` plus the intrinsically-unrepresentable ops).
- `lib/native4d/graph_shape4.ml`: a new `softmax_params` adapter (mirroring
  `repeat_interleave_params`) + the output-shape arm, delegating whole to
  `Reduce.Softmax.output_shape`.
- `lib/native4d/eval_op4.ml`: the matching compute arm, through
  `Reduce.Softmax.Compute`, using the shared adapter so the shape rule and
  the compute cannot disagree about which axis is reduced.
- `lib/native4d/output_transfer4.ml`: classified `Continuous`, not
  `Reindexing` — a genuine reduction, not a copy: every output element
  depends on the whole reduced axis (both the max and the sum range over
  it), the same answer Native's own `Output_transfer` gives `Softmax`.
- `lib/native4d/builder.ml`: `softmax4` builder function.
- `lib/native4d/error.ml` + `domain.ml`'s `Sdpa` arm: the
  `` `Sdpa_batch_axis `` diagnostic's parenthetical ("Native has no Bmm or
  softmax in Native4D") was stale the moment `Softmax4` landed — `Sdpa`'s
  rejection was never actually about softmax's absence (its batch axis is
  `D` unconditionally, regardless of configuration), so the message now
  names the real remaining blocker: Native4D's `Bmm` legalization admits
  only a single batch. `.ai/attention_design.md` §9 and
  `.ai/matmul_softmax_design.md` §3 updated to match.
- Tests: `test/native4d/op_json_test.ml` (JSON round-trip sample, axis
  distinct from `Slice4`'s own), `test/native4d/fixtures4.ml` (`softmax4`
  per-op Direct-vs-Symbolic fixture), `test/native4d/compute_test.ml` (a
  hand-computed Direct value: `[0, ln 3]` over a two-element axis gives
  `exp` values `[1, 3]` summing to 4, so softmax is exactly `[1/4, 3/4]` —
  exactly representable, so the bitwise comparison is about the computation,
  not float printing), `test/native4d/domain_test.ml` (replaced the old "no
  counterpart at any axis" test with the axis-gated one: C converts, D is
  refused — reusing the existing `Fixtures.softmax_over` Native fixture,
  whose own doc comment also needed updating), `test/native4d/lower_shape_test.ml`
  (one conversion golden plus the D-axis domain-check refusal, the
  `repeat_interleave4` pair's own precedent immediately above it in the
  same file). `test/me_group8_cram.t` and `test/native4d/domain_test.ml`'s
  `sdpa` test both needed their expected diagnostic text updated for the
  reworded `` `Sdpa_batch_axis `` message.

**Classification**: Full-stack, per [`todo-ops.md`](todo-ops.md)'s
completion rule — a Native4D counterpart to an already full-stack Native
op, with shape/eval coverage and the Native-to-Native4D lowering arm, each
with its own test evidence enumerated above.

**Corpus effect**: none. No corpus model is currently gated on `Softmax4`
(per `matmul_softmax_design.md` §3, confirmed by `make pt2.json-model-support`
producing no diff on `test/data/pt2_json_model_support.jsonl`): every model
that reaches `softmax.int` is transformer-shaped and already stops at
another Native4D domain limit (an intrinsic `D`-axis argument, or the
batched-matmul case) regardless of whether `Softmax` itself converts.
`native_builds`/`native4d_converts`/`kernel_converts`/all-three-stages are
unchanged (89/56/63/43).

**Next**: live max-pool indices/`IndexTensor4` is the only remaining
open-ended Native4D counterpart-backlog row — a full gather/indexing design,
not a one-session slice. Otherwise: `lstm.input` from the deferred backlog
(36 occurrences, Sequencer2D's own first frontier), or P1's remaining
one-model slices (`conv1d`/`unfold`, `im2col`/`col2im`,
`upsample_bicubic2d`).

### 2026-08-30 — `eye.m`, the rank-2 identity-matrix factory

Closes the "Factories, indexing, and copies" deferred-backlog row
[`todo-ops.md`](todo-ops.md) tracked for `bat_resnext26ts` (12 occurrences,
its first frontier after `adaptive_max_pool2d.default` landed). ATen's
`eye.m(SymInt n, SymInt m, *, ScalarType? dtype=None, Layout? layout=None,
Device? device=None, bool? pin_memory=None) -> Tensor` is a genuine
no-operand factory, the same shape `zeros.default`/`arange` already are —
so it follows their precedent directly rather than composing from an
existing op.

**What landed**

- `lib/native/ops/factory.ml`: new `Factory.Eye` module, alongside `Zeros`/
  `Arange`. `Aten_shape.of_aten` right-aligns a rank-2 ATen shape onto the
  frame's innermost two axes ([W; C]), so `n`'s rows land on `W` and `m`'s
  columns on `C` unconditionally — the same fixed-axis assumption
  `Arange`'s own pixel already makes for its single `C` axis. `Compute`
  needs no new `SEMANTICS` primitive: the diagonal test is one `index_eq`
  comparing the `w`/`c` coordinates, the same "compare via the index
  domain" idiom `Pad`'s reflect region test established.
- `lib/native/graph_ir.ml`/`.mli`: `Eye` constructor + registry entry
  (alphabetical, between `Expand` and `Gelu`).
- `lib/native/graph_shape.ml`, `lib/native/eval_op.ml`,
  `lib/native/eval_direct.ml`, `lib/native/graph_builder.ml`/`.mli`: the
  usual per-op shape/compute/builder wiring. `eval_direct.ml` gets the same
  `materialize_fmt` dtype-fidelity bypass `Zeros`/`Arange` get, alongside
  them, rather than the generic (always-F32) `Schedule.evaluate` path.
- `lib/native/transform/output_transfer.ml`: `Eye` joins the bulk
  `Continuous` bucket (no operand at all, so trivially deterministic and
  input-independent).
- `lib/native_aten_bridge/op_bridge_shape.ml`: the `eye.m` arm, reusing
  `zeros.default`'s layout/device/pin_memory rejection and FLOAT/DOUBLE
  dtype dispatch verbatim — the trailing argument list is identical. `n`/
  `m` are read via `int_arg` (two scalar SymInts, not a SymInt[] list) and
  passed through `Aten_shape.of_aten` the same way `zeros`'s `size` is.
- `lib/native_interp/native_interp_lower_shape.ml`: the mirror import arm,
  reusing `zeros.default`'s own rejection/dtype-dispatch code shape; `n`/
  `m` read with `int_arg` and packed into `shape_of_sizes`'s list form.
- **ATen curated binding**: `bin/aten_ops_gen.ml` gains `op "eye"
  ~overload:"m"`. Its dispatch kernel (`TensorFactories.cpp`'s `eye`/
  `eye_out_cpu`) is already linked — `zeros`/`arange` share the same
  translation unit — so no `build_archive.sh` change was needed, unlike
  `adaptive_max_pool2d`'s separate kernel file.
- **Real-ATen oracle**: no walk recipe — `n`/`m` are required SymInts with
  no schema default, so `eye.m` lands in the walk generator's `needs_meta`
  backlog the same way `zeros.default`/`arange`'s two overloads already do
  (`test/native_walk_coverage_test.ml`). Real-ATen agreement is instead
  hand-verified: `test/native_bridge/activation_test.ml`'s `verify_print`
  runs the `eye.m` dispatch against real ATen on a non-square `n <> m`
  (3x2) shape, so a transposed row/column comparison would be visible.
- Tests: `test/native/factory_test.ml` (Direct dtype-fidelity + diagonal
  placement on a 2x3 shape, Symbolic reaching the kernel adapter — mirrors
  `Zeros`/`Arange`'s own two-test pattern each), `test/native_bridge/
  activation_test.ml` (dispatch default/DOUBLE dtype + real-ATen verify),
  `test/native_interp/activation_test.ml` (serialized-path lowering for
  both dtypes), `test/native4d/{op_json_test,compute_test,fixtures4}.ml`
  (JSON round-trip sample, Direct-vs-Symbolic fixture).

**Native4D**: `Eye` gets a full `Eye4` counterpart in the same change —
`lib/native4d/{ops4,op,domain,graph_shape4,eval_op4,eval_direct4,
output_transfer4,lower_engine,builder}.ml` — typed `Shape4.t` like `Zeros4`
rather than reusing Native's own payload directly, since (unlike
`Add_scalar`/`Rsub_scalar`/`_to_copy`) `Eye` carries a shape parameter, not
just a dtype. `Domain.check_node` admits `Eye` unconditionally alongside
`Arange`/`Zeros`: rows/columns land on `W`/`C` by construction, both real
dialect axes, so there is no axis-domain rejection to add. The actual
`Eye -> Eye4` lowering arm in `lower_engine.ml` mirrors `Zeros`'s exactly
(`sig_of` the built shape, `shape4` to validate/convert it into the
dialect, `Eye4` params carry it directly).

**Classification**: Full-stack, per [`todo-ops.md`](todo-ops.md)'s
completion rule — ATen boundary, both importers, Native Direct/Symbolic,
a full Native4D counterpart, and a Kernel-path-reaching representative
graph shape are all present, each with its own test evidence enumerated
above.

**Corpus effect**: `bat_resnext26ts` — the one corpus model gating this
row — now reaches `native_builds:true` (was `false`), stopping at a new,
genuine intrinsic boundary later in the same graph: `node n924: axis D is
outside the N/H/W/C dialect` (`native4d_reason: outside_dialect_domain`).
Its kernel stage stays blocked independently at the pre-existing
evaluation-depth ceiling (`kernel_reason: over_limit`, unrelated to this
change — `native4d_converts`/`kernel_converts` were already `false`
either way). `native_builds` moves 88→89; `native4d_converts`/
`kernel_converts`/all-three-stages are unchanged (56/63/43).

**Next**: `Softmax4` and live max-pool indices/`IndexTensor4` remain the
two open-ended Native4D counterpart-backlog rows; otherwise `lstm.input`
(36 occurrences, Sequencer2D's first frontier) or P1's remaining
one-model slices (`conv1d`/`unfold`, `im2col`/`col2im`,
`upsample_bicubic2d`).

### 2026-08-30 — `adaptive_max_pool2d.default`, the adaptive counterpart to `max_pool2d`/`max_pool2d_with_indices`

Closes the P1 row [`todo-ops.md`](todo-ops.md) tracked for
`bat_resnext26ts` (16 nodes, its first frontier). ATen's own
`adaptive_max_pool2d(Tensor self, int[2] output_size) -> (Tensor, Tensor)`
has no value-only overload the way `max_pool2d.default`/
`max_pool2d_with_indices.default` split into two ATen ops — it always
returns (values, indices) — so the design mirrors that pair's relationship
one level down inside Native itself rather than at the ATen boundary.

**What landed**

- `lib/native/ops/pool.ml`: two new modules. `AdaptiveMaxPool2d`
  (value-only, one output) reuses `Adaptive_axis.check`/`Adaptive_axis.
  Compute.bin` verbatim from `AdaptiveAvgPool2d`, with the reduction swapped
  from `S.sum`/`S.div` to `S.max_reduce` — exactly `Reduce.Amax`'s
  relationship to `Reduce.Mean`. No new `SEMANTICS` primitive: `max_reduce`
  already has the same `lo`/`hi`/`f` shape as `sum`.
  `AdaptiveMaxPool2dWithIndices` (ATen-facing, two outputs) reuses
  `AdaptiveMaxPool2d`'s own `Compute.pixel` for `value_pixel` and derives
  the flat argmax index for `index_pixel` purely from existing primitives —
  a two-pass max-reduce/select/negation idiom (find the max, then find the
  smallest flat index whose value equals it via `select (lt v m) sentinel
  flat` and a max-reduce-of-negated-candidates for the min) — rather than a
  bespoke intrinsic like `max_pool2d_index`, the same "`max`/`min`/`relu`
  are derived, not primitive" rule `semantics.ml`'s own module doc states.
  Verified against real `at::adaptive_max_pool2d`'s tie-break convention by
  the walk oracle below, not just assumed from `MaxPool2dWithIndices`'s own
  comment.
- `lib/native/graph_ir.ml`/`.mli`: `Adaptive_max_pool2d` and
  `Adaptive_max_pool2d_with_indices` constructors + registry entries
  (alphabetical, between `Adaptive_avg_pool2d` and `Amax`).
- `lib/native/graph_shape.ml`, `lib/native/eval_op.ml`,
  `lib/native/graph_builder.ml`/`.mli`: the usual per-op shape/compute/
  builder wiring, `Adaptive_max_pool2d_with_indices`'s builder and
  `eval_op.ml`'s `output = 0 then value_pixel else index_pixel` dispatch
  mirroring `Max_pool2d_with_indices`'s own exactly.
- `lib/native/transform/output_transfer.ml`: `Adaptive_max_pool2d`
  `Continuous` (joins the bulk bucket); `Adaptive_max_pool2d_with_indices`
  joins `Max_pool2d_with_indices`'s `output = 0 then Continuous else
  Discontinuous` arm (the index is data-dependent/argmax-shaped).
- `lib/native/transform/passes/drop_pool_indices.ml`: a second `on_node`
  arm narrowing `Adaptive_max_pool2d_with_indices` to `Adaptive_max_pool2d`
  once its index is dead, exactly `Max_pool2d_with_indices` → `Max_pool2d`.
  Since ATen has no value-only adaptive overload, this pass is the ONLY
  route by which `Adaptive_max_pool2d` is ever produced — both importers
  always build the two-output op first. The `Identical` narrowing claim is
  **proved structurally**, not declined
  (`test/native/drop_pool_indices_test.ml`: `4 clusters: 3 proved
  (structural), 1 vacuous`, matching the fixed-window pair's own result),
  confirming the generic-primitive-built `AdaptiveMaxPool2dWithIndices.
  Compute.value_pixel` really is the same term as `AdaptiveMaxPool2d.
  Compute.pixel` (it's defined as that module's own `Compute.pixel`).
- `lib/native_aten_bridge/op_bridge_pool.ml` + `op_bridge_error.ml`: the
  `adaptive_max_pool2d.default` arm, combining `adaptive_avg_pool2d.
  default`'s rank check (reusing the now-generalized `` `Adaptive_pool_rank
  ``, its message no longer hardcoded to "adaptive_avg_pool2d") with
  `max_pool2d_with_indices.default`'s two-output/`discard`/relayout
  pattern.
- `lib/native_interp/native_interp_lower_compute.ml`,
  `native_interp_decode.ml`, `native_interp_error.ml`/`.mli`: the mirror
  import arm (new `` `Adaptive_max_pool2d_input `` metadata role, alongside
  the existing avg one), `is_nontrivial_node` and `materialized_output_names`
  both gaining the new target (the latter drops its serialized indices name
  from tracking, the same treatment `max_pool2d_with_indices.default`/
  `native_layer_norm.default` already get, since the built op returns only
  `[values]`).
- **ATen curated binding**: `bin/aten_ops_gen.ml` gains
  `op "adaptive_max_pool2d"`. Building it exposed a genuine linker gap —
  `adaptive_max_pool2d`'s structured CPU kernel wasn't in the curated
  archive's source list at all — fixed in `lib/aten/build_archive.sh` by
  adding `AdaptiveMaxPooling2d.cpp` (the structured meta+impl, the same
  shape `DilatedMaxPool2d.cpp` is for `max_pool2d_with_indices`) and
  `cpu/AdaptiveMaxPoolKernel.cpp` to the CAP list, alongside
  `AdaptiveAveragePooling.cpp`/`cpu/AdaptiveAvgPoolKernel.cpp`.
- **Real-ATen oracle**: `lib/aten_gen/walk_meta_pool.ml` gains
  `adaptive_max_pool2d`, reusing `Recipe_adaptive`'s walk config UNCHANGED
  (the shape/output_size axes don't depend on which reduction runs) —
  registered in `lib/aten_gen/walk_meta.ml`. The generated walk agrees with
  real ATen across randomized shapes/output sizes in
  `test/native_walk_coverage_test.ml`'s bridge-coverage sweep, including the
  argmax index output (not just the value) — the first per-model evidence
  that the tie-break derivation above actually matches ATen, not merely the
  comment it was modeled on.
- Tests: `test/native/pool_test.ml` (hand-computed values+argmax over a
  5x5→3x3 non-divisible-bin case, and a ties-choose-smallest-flat-index
  case), `test/native/graph_direct_pool_test.ml` +
  `graph_symbolic_pool_test.ml` (end-to-end graph builds, Symbolic staging
  the whole thing as an ordinary nested-expression tree rather than an
  intrinsic node — ground matches Direct bit-for-bit), `test/native/
  drop_pool_indices_test.ml` (the adaptive discarded/live pair + the
  structural `Identical` proof), `test/native_bridge/pool_dispatch_test.ml`
  and `test/native_interp/pool_test.ml` (relayout + Discard routing +
  rank/size rejection, mirroring each importer's own `adaptive_avg_pool2d`/
  `max_pool2d_with_indices` coverage), `test/native4d/{op_json_test,
  fixtures4,compute_test}.ml` (the value-only op's JSON/Direct-vs-Symbolic
  coverage), `test/native4d/{fixtures,domain_test}.ml` (the two-output op's
  discarded/live rejection, mirroring `Max_pool2d_with_indices`'s own case
  in `domain_test.ml` — which, like this op, previously had no dedicated
  unit-test coverage there at all).

**Native4D**: `Adaptive_max_pool2d` (value-only, no axis parameter) gets a
full counterpart by direct reuse — `lib/native4d/{op,domain,graph_shape4,
eval_op4,output_transfer4,lower_engine,builder}.ml` — exactly
`Adaptive_avg_pool2d`'s own treatment, admitted unconditionally by
`Domain.check_node`. `Adaptive_max_pool2d_with_indices` joins
`Max_pool2d_with_indices` in the dialect's existing "no argmax-pool
operation" rejection (`` `Live_max_pool_indices `` when the index is live,
plain `unsupported` when dead but not yet narrowed) — this is the SAME
open design point `todo-ops.md`'s "Live max-pool indices and `IndexTensor4`"
backlog row already tracks, not a fresh gap this landing introduces.

**Classification**: Full-stack for `adaptive_max_pool2d.default`'s NATIVE
semantics (ATen boundary, both importers, Native Direct/Symbolic, and a
full Native4D counterpart for the value-only op, each with oracle and
Direct-vs-Symbolic evidence above) — but the two-output ATen-facing form's
Native4D story is deliberately incomplete pending the tracked live-index
design, the same way `Max_pool2d_with_indices` itself has been all along.
This doesn't fit the completion table's "Native-only, deliberately bounded"
tier cleanly either — that tier is for an intrinsic `D`/`T`-axis or
new-kernel-semantic boundary, and this is neither — so it's recorded
plainly as: full-stack up to and including a real Native4D counterpart,
with the multi-output live-index case left exactly where
`Max_pool2d_with_indices` already leaves it.

**Corpus effect**: `bat_resnext26ts` — the one corpus model gating this
row — moves from `unsupported PT2 operator: torch.ops.aten.
adaptive_max_pool2d.default` to `unsupported PT2 operator: torch.ops.aten.
eye.m`, the next row of the "Factories, indexing, and copies" deferred
backlog. `native_builds`/`native4d_converts`/`kernel_converts`/
all-three-stages are unchanged (88/56/63/43) — depth moved, stage-completion
did not, since `bat_resnext26ts` was and remains `native_builds:false`.

**Next**: `eye.m` (12 occurrences, now `bat_resnext26ts`'s own first
frontier) from the deferred backlog; `Softmax4` and live max-pool
indices/`IndexTensor4` remain the two open-ended Native4D counterpart-backlog
rows; otherwise `lstm.input` (36 occurrences, Sequencer2D's first frontier)
or P1's remaining one-model slices (`conv1d`/`unfold`, `im2col`/`col2im`,
`upsample_bicubic2d`).

### 2026-08-30 — `Select_scatter4`, the Native4D counterpart to `Select_scatter`

Closes the row [`todo-ops.md`](todo-ops.md)'s "Remaining Native4D
counterpart backlog" table added for it in the previous entry —
`convit_tiny`'s own Native4D frontier, exposed by the `Repeat4`/
`RepeatInterleave4` landing moving it past `Repeat`'s own blocker.

**What landed**

- `lib/native4d/ops4_split.ml`: `Select_scatter4`, right after `Select4`.
  `params = { axis : Axis4.t; index : int }` — the same shape `Select4`'s
  own payload takes — plus `self`/`src` operands (two, unlike `Select4`'s
  one). No shape rule or pixel map of its own: both delegate whole to
  `Split.Select_scatter`, the same way `Select4` delegates to
  `Split.Select`.
- `lib/native4d/ops4.ml`: `module Select_scatter4 = Ops4_split.Select_scatter4`
  re-export.
- `lib/native4d/op.ml`: `Select_scatter4` constructor (between `Select4`
  and `Sigmoid`) + registry entry.
- `lib/native4d/domain.ml`: `Select_scatter` moved out of the "dialect does
  not have it at all" bucket into its own new `check_dims node
  [params.axis]` arm, the `Select`/`Slice`/`Stack`/`RepeatInterleave`
  treatment — genuinely load-bearing here, unlike `RepeatInterleave`'s own
  arm: the WRITTEN axis is real op-level data regardless of shape
  consequence, but UNLIKE `Select`, this op's own output shape
  (`self_shape`, unmodified — no drop, no repack) never depends on which
  axis is named, so there is no separate shape-consequence rejection the
  way `Select`'s own comment describes.
- `lib/native4d/lower_engine.ml`: the conversion arm, converting only the
  axis KEY (`dims4`, `Select`'s own pattern) — no post-hoc output re-check
  the way `Select`'s own arm needs, because the output is `self`'s shape
  unchanged: if `self` is already four-axis so is the output, whichever
  axis this op names. Removed from the "rejected by `Domain.check` before
  the walk starts" bucket.
- `lib/native4d/graph_shape4.ml`: a new `select_scatter_params` adapter
  (mirroring `select_params`) + the output-shape arm, delegating whole to
  `Split.Select_scatter.output_shape`.
- `lib/native4d/eval_op4.ml`: the matching compute arm, through
  `Split.Select_scatter.Compute`, using the shared adapter so the shape
  rule and the compute cannot disagree about which position is written.
- `lib/native4d/output_transfer4.ml`: classified `Reindexing` — every
  output is copied from ONE of `self`/`src` with no arithmetic, the same
  argument Native's own `Output_transfer` already makes for
  `Select_scatter`.
- `lib/native4d/builder.ml`: `select_scatter4` builder function.
- Tests: `test/native4d/op_json_test.ml` (JSON round-trip sample, axis/index
  distinct from `Select4`'s own and two distinct operands),
  `test/native4d/fixtures4.ml` (`select_scatter4` per-op Direct-vs-Symbolic
  fixture — `self`/`src` at genuinely DIFFERENT shapes, `src` computed as
  the actual repacked shape `Select4` would produce there, not a same-shape
  pair the way `binary`'s helper assumes), `test/native4d/fixtures.ml` (new
  `select_scatter_w`/`select_scatter_d` NATIVE-graph fixtures),
  `test/native4d/domain_test.ml` (`select_scatter`'s own W/D axis-gate
  pair, the `slice`/`concat` shape — not the `select`/`stack`/`unbind`
  "axis rule and its shape consequence" shape, since there is no separate
  shape consequence to demonstrate here), `test/native4d/lower_test.ml`
  (one conversion golden, `select4`'s own single-test precedent, since no
  other op in this directory duplicates a domain-level rejection as a
  lowering-level test too).

`test/native4d/lower_test.ml` crossed the tracked 1000-line file-size
ceiling (`scripts/check-file-size.sh`) with this landing's addition. Split
into `lower_test.ml` (factory/norm/precondition rows, including this
entry's own new test) and a new `lower_shape_test.ml` (the
reshape/expand/repeat/transpose family, op3-impl.md commit 8's own scope —
unrelated to this entry, moved only to make room), sharing the file's
`build`/`described`/`show`/`outcome` rendering helpers via a new
`lower_fixtures.ml`. Mirrors the precedent `ops4_split.ml`/`ops4_conv.ml`
set for `lib/native4d/ops4.ml` itself.

**Classification**: Full-stack, per [`todo-ops.md`](todo-ops.md)'s
completion rule — a Native4D counterpart to an already full-stack Native
op, with shape/eval coverage and the Native-to-Native4D lowering arm, each
with its own test evidence enumerated above.

**Corpus effect**: `convit_tiny` is the one corpus model whose Native4D
frontier was `Select_scatter`'s missing counterpart (`node n25: no
legalization for select_scatter self=t191 src=t205
params={axis=C index=2}` — axis C, confirming `Select_scatter4`'s design
against a real graph, not only synthetic fixtures). Its frontier moves to
`node n38: axis T is outside the N/H/W/C dialect` — a genuine intrinsic
`T`/`D`-axis boundary, Native-only and deliberately bounded territory per
`todo-ops.md`'s own classification table, not a missing counterpart.
`native_builds`/`native4d_converts`/`kernel_converts`/all-three-stages are
unchanged (88/56/63/43) — depth moved, stage-completion did not, since
`convit_tiny` was and remains `native4d_converts:false`.

**Next**: every row of the "Remaining Native4D counterpart backlog" table
is now landed except `Softmax4` and live max-pool indices/`IndexTensor4` —
both open-ended designs the table itself scopes as such, not one-session
slices. Otherwise: `lstm.input` from the deferred backlog, or P1's
remaining one-model slices (`adaptive_max_pool2d`, `conv1d`/`unfold`,
`im2col`/`col2im`, `upsample_bicubic2d`).

### 2026-08-30 — `Repeat4` and `RepeatInterleave4`, the Native4D counterparts to `Repeat`/`RepeatInterleave`

Closes the row [`todo-ops.md`](todo-ops.md)'s "Remaining Native4D
counterpart backlog" table did not yet have (added and immediately marked
landed in this entry) — `convit_tiny`'s own Native4D frontier was `Repeat`'s
missing counterpart, ever since the `repeat.default` landing deliberately
deferred it.

**What landed**

- `lib/native4d/ops4.ml`: two new modules next to `Expand4`.
  `Repeat4.params = { repeats : Shape4.t }` — reuses `Shape4.t`'s own
  representation (a `Vec6.shape` with T/D pinned to the unit extent)
  directly for the repeat multiplier, since "repeats.T = 1, repeats.D = 1"
  IS the representability condition a multiplier needs, the same
  `Expand4.params.size` treatment. `RepeatInterleave4.params = { axis :
  Axis4.t; repeats : Op_config.Pos.t }` — one named axis, the `Select4`/
  `Slice4` treatment.
- `lib/native4d/op.ml`: `Repeat4`/`RepeatInterleave4` constructors (between
  `Relu` and `Reshape4`) + registry entries.
- `lib/native4d/domain.ml`: `Repeat` moved into the "constrain nothing"
  unconditional-admit bucket (`Expand`'s own bucket) — dropped out of the
  "dialect does not have it at all" list it shared with `RepeatInterleave`/
  `Select_scatter`/`Softmax`/`Index_tensor`. `RepeatInterleave` gets its own
  new `check_dims node [params.axis]` arm, the same `Select`/`Slice`/`Stack`
  treatment — not load-bearing here (unlike those three, `RepeatInterleave`
  MULTIPLIES its named axis rather than collapsing it to 1, so a T/D target
  would already be caught by the blanket four-axis shape check the same way
  `Repeat`'s is), added anyway so every named-axis op in the `Split`/
  `Repeat` family gets one consistent diagnostic rather than a silent
  per-op difference in how good the error message is.
- `lib/native4d/lower_engine.ml`: the two conversion arms. `Repeat`'s
  converts `repeats` through the existing `shape4 ~id` helper (`Expand`'s
  own pattern) and needs no post-hoc output re-check the way `Select`/
  `Stack` do, because `Repeat` neither drops nor inserts an axis — every
  axis keeps its own identity (repeat.ml's own doc comment), so a T/D-unit
  `repeats` composed with a T/D-unit input `x` is always T/D-unit on
  output, automatically. `RepeatInterleave`'s converts only the axis KEY
  (`dims4`, `Select`'s own pattern) for the same reason: multiplying one
  axis in place cannot shift a different axis into T/D. Both removed from
  the file's "rejected by `Domain.check` before the walk starts" bucket.
- `lib/native4d/graph_shape4.ml`: a new `repeat_interleave_params` adapter
  (mirroring `select_params`) + two output-shape arms, delegating whole to
  `Repeat.Repeat.output_shape`/`Repeat.RepeatInterleave.output_shape`.
- `lib/native4d/eval_op4.ml`: the matching compute arms, through
  `Repeat.Repeat.Compute`/`Repeat.RepeatInterleave.Compute` — `Repeat4`'s
  needs no params argument at all (repeat.ml's own `Compute.pixel` reads
  only `x_shape`), `RepeatInterleave4`'s reuses the shared
  `Graph_shape4.repeat_interleave_params` adapter so the shape rule and the
  compute cannot disagree about which axis is multiplied.
- `lib/native4d/output_transfer4.ml`: both classified `Reindexing` — pure
  data movement, no arithmetic, the same argument `Expand4` makes (and the
  argument Native's own `Output_transfer` already makes for `Repeat`/
  `RepeatInterleave`).
- `lib/native4d/builder.ml`: `repeat4`/`repeat_interleave4` builder
  functions.
- Tests: `test/native4d/op_json_test.ml` (JSON round-trip samples, distinct
  multipliers/axis so a codec bug is visible in the printed output, not
  just the count), `test/native4d/fixtures4.ml` (`repeat4`/
  `repeat_interleave4` per-op Direct-vs-Symbolic fixtures),
  `test/native4d/fixtures.ml` (new `repeat`/`repeat_interleave_w`/
  `repeat_interleave_d` NATIVE-graph fixtures, for the two tests below),
  `test/native4d/domain_test.ml` (`Repeat`'s own "always admits; its
  multiplier is checked as a shape" test, the same single-row shape
  `Expand`'s gets; `RepeatInterleave`'s own W/D axis-gate pair, the same
  shape `slice`'s gets), `test/native4d/lower_test.ml` (four new tests:
  `Repeat4` converting and being refused on a D-tiling multiplier via
  `Shape4.of_vec6`'s `Non_four_dimensional_tensor`, `RepeatInterleave4`
  converting and being refused on a T-named axis via the domain check's
  `Axis_outside_dialect` — proving the two failure PATHS are actually
  different, not just asserting both eventually reject).

**Classification**: Full-stack, per [`todo-ops.md`](todo-ops.md)'s
completion rule — both are Native4D counterparts to already full-stack
Native ops, with shape/eval coverage and the Native-to-Native4D lowering
arm, each with its own test evidence enumerated above.

**Corpus effect**: `convit_tiny` is the one corpus model whose Native4D
frontier was `Repeat`'s missing counterpart (`node n16: no legalization for
repeat x=t196 params={repeats=[W=14 C=14]}` — a genuine two-axis multiplier,
both H/W/C-legal, confirming `Repeat4`'s design against a real graph rather
than only synthetic fixtures). Its frontier moves to `select_scatter.default`
(`node n25: ... params={axis=C index=2}`) — the next row of the same
"Remaining Native4D counterpart backlog" table, `Select_scatter4`, now added
there — rather than to `native4d_converts:true`, since it is a later blocker
in the same graph. No corpus model reaches `RepeatInterleave4`'s own
frontier: ConViT's `repeat_interleave.self_int` occurrences sit behind
`Repeat4`'s now-cleared blocker in the SAME graph, at `native_builds` time,
not at Native4D. `native_builds`/`native4d_converts`/`kernel_converts`/
all-three-stages are unchanged (88/56/63/43) — depth moved, stage-completion
did not, since `convit_tiny` was and remains `native4d_converts:false`.

**Next**: `Select_scatter4` (convit_tiny's new frontier, and the last
remaining row of the Native4D counterpart backlog alongside `Softmax4`/live
max-pool indices). Otherwise: `lstm.input` from the deferred backlog, or
P1's remaining one-model slices (`adaptive_max_pool2d`, `conv1d`/`unfold`,
`im2col`/`col2im`, `upsample_bicubic2d`).

### 2026-08-30 — `_to_copy.default`, the ATen dtype cast, plus a new `SEMANTICS` `trunc` primitive

Closes `_to_copy.default`'s row in [`todo-ops.md`](todo-ops.md)'s "Next
priority" note — the highest-leverage remaining target (EdgeNeXt and
MViTv2 both gated on it as their first frontier), flagged since the
`repeat.default` landing as needing a new value-domain trunc/round
`SEMANTICS` primitive first.

**What landed (`trunc`, the new `SEMANTICS` primitive)**

- `lib/expr_internal/expr_repr.ml`: `Trunc` added to `unary_op` (now
  `Erf | Exp | Log | Sqrt | Trunc`) — the representation's single source,
  per that module's own comment.
- `lib/expr_internal/value.ml`: `apply_unary`/`unary_name` arms (`Float.trunc`
  / `"trunc"`) and the smart constructor `let trunc a = Unary (Trunc, a)`.
  Every other site in `lib/expr_internal` (`fold.ml`, `check.ml`, `pp.ml`,
  `rewrite.ml`, `eval.ml`) already pattern-matches `Unary (_, a)` generically,
  so none of them needed a change — unlike `Round_f32`, which is a distinct
  `Value.t` constructor (for the kernel elaborator's f32-storage-precision
  concern, an unrelated need) and so touches all of those sites individually.
  Adding to the existing `unary_op` variant, not a new `Value.t` case, is
  what keeps this a ~10-line addition.
- `lib/expr/expr_api.ml`: `Trunc` added to the public `unary_op` mirror,
  `val trunc : t -> t` added to `Value`'s signature.
- `lib/native/semantics.ml`: `val trunc : t -> t` added to `SEMANTICS`,
  documented as genuinely primitive (unlike `max`/`min`/`relu`, no finite
  `add`/`mul`/`select` composition discards a fractional part).
- `lib/native/direct.ml`: `let trunc = Float.trunc`.
- `lib/native/symbolic.ml`: `let trunc a = Expr.Builder.map Expr.Value.trunc a`.

**What landed (`_to_copy.default` / `To_copy`)**

- `lib/native/ops/pointwise_unary.ml`: a new `To_copy` module, `params =
  { target : target }` where `target = Bool | Float | Long` — a closed
  vocabulary scoped to this op (not `Payload.packed_fmt`, which has no Bool
  case and whose `Real`/`Quant` tags describe storage formats this op never
  materializes: `graph_builder.ml`'s own "op-output edges are F32" note means
  every ordinary compute edge, this one included, stays F32 regardless of
  the ATen-visible target dtype). `output_shape` is the identity, like
  `Clone`. `Compute.pixel`: `Float` is the loaded value unchanged; `Long` is
  `S.trunc`; `Bool` is `select (lt 0 v) 1 (select (lt v 0) 1 0)` — a genuine
  nonzero test using only pre-existing primitives, not an overfit to the
  corpus's own all-zero (`zeros`-sourced) BOOL-target operand.
- `lib/native/ops/pointwise.ml`: `To_copy` facade re-export.
- `lib/native/graph_ir.ml`/`.mli`: `To_copy` constructor (between `Sum` and
  `Unbind`) + registry entry.
- `lib/native/graph_shape.ml`, `lib/native/eval_op.ml`: shape/compute
  dispatch arms.
- `lib/native/transform/output_transfer.ml`: classified `Discontinuous`,
  per-OP not per-target — the `Float` target is a true identity, but `Long`
  (an integer boundary) and `Bool` (a zero boundary) can each flip their
  result from an arbitrarily small input change, so the whole op answers
  conservatively rather than reading the payload to special-case `Float`.
- `lib/native/transform/passes/sink_permute.ml`: added to the `elementwise`
  allowlist (a genuine per-slot unary transform, output shape equals input
  shape); not added to `reuse_permute.ml`'s `binary_elementwise` (one tensor
  operand).
- `lib/native/graph_builder.ml`/`.mli`: `to_copy` builder function.
- `lib/native4d/{op,domain,lower_engine,graph_shape4,eval_op4,
  output_transfer4,builder}.ml`: full Native4D counterpart, the same six
  trivial mirror sites `Rsub_scalar`'s own landing used — since the op
  carries no axis parameter, Native4D reuses Native's own
  `Pointwise.To_copy.t` payload directly (no `Ops4` module, no axis-domain
  check). `output_transfer4.ml` needed its own `Discontinuous` arm (the
  dialect's file previously had none — its own header comment, "there is no
  argmax-pool, so nothing here is Discontinuous," needed updating to scope
  that claim to pooling specifically). Full-stack in this single landing,
  the same reasoning `Rsub_scalar` and `Add_scalar`/`Mul_scalar`/`Pow`
  already get: no axis to design a dialect boundary around.
- `bin/aten_ops_gen.ml`: `op "_to_copy"` added to the curated ATen binding
  selection (right after `to.dtype`, which was already curated but is a
  different overload — the public `Tensor.to` method, not the internal op
  the exporter actually emits). The schema's `MemoryFormat?` last argument is
  handled generically by the C-type generator (`Optional (Base
  MemoryFormat)` in `aten_c_type.ml`), so no `Override`/hand-written
  signature was needed — the generator emitted both the `atg__to_copy` C shim
  and the matching `Interp_dispatch` arm unmodified.
- `lib/native_aten_bridge/op_bridge_shape.ml`: `_to_copy.default` dispatch
  arm, right after `clone.default`'s (both are single-operand,
  identity-shape unary transforms of `self`). Decodes `dtype` into the
  three-way `target` domain (`None`/`FLOAT` → `Float`, `BOOL` → `Bool`,
  `LONG` → `Long`, anything else a typed `Validation_failure`); rejects
  non-default `layout`/`device`/`pin_memory`/`memory_format` the same way
  `zeros.default`/`clone.default` already do; reads-and-discards
  `non_blocking`.
- `lib/native_interp/native_interp_lower_shape.ml`: the matching
  payload-free arm (the one `test/data/pt2_json_model_support.jsonl`
  actually measures) — same dtype/reject/discard logic, using the existing
  `` `Unsupported_option `` / `` `Dtype `` typed diagnostic
  `zeros.default`/`arange.default` already share.
- Tests: `test/native/pointwise_test.ml` (Direct compute, all three
  targets, including the negative-fraction truncation case),
  `graph_direct_pointwise_test.ml` + `graph_symbolic_pointwise_test.ml`
  (one-node graph, Direct-vs-Symbolic agreement — the `Long` case is the one
  that actually exercises `trunc`'s staged form), `test/native_bridge/
  to_copy_test.ml` (**new file**: `dispatch_print` hand-derived values for
  all three targets plus the dtype-omitted/rejection/non_blocking cases,
  and a `verify_print` oracle test against real ATen for the dtype-omitted
  case — the only configuration `Interp_decode.scalar_type_opt_arg` can
  decode, since it refuses to decode ANY present `ScalarType` argument by
  design; the `Long` truncation fact is independently cross-checked against
  real ATen by the pre-existing `test/aten_tensor_test.ml` "to.dtype casts
  float -> int64 (truncating)" test, which calls `Aten_tensor.O.to_dtype`
  directly, bypassing that decoder), `test/native_interp/to_copy_test.ml`
  (**new file**, payload-free, asserts the built graph structure for all
  three targets, the dtype-omitted default, the rejection diagnostic, and
  `non_blocking`'s no-effect proof — mirroring `copy_test.ml`'s own `dump`
  pattern), `test/native4d/{op_json_test,compute_test,fixtures4}.ml` (JSON
  round-trip sample, per-op Direct-vs-Symbolic fixture).

**Classification**: Full-stack across every surface in
[`todo-ops.md`](todo-ops.md)'s completion-rule table — ATen boundary (now a
curated binding), both importers, Native Direct/Symbolic, and a full
Native4D counterpart, each with its own test evidence.

**Corpus effect**: both corpus models gated on `_to_copy.default` as their
first frontier move to a new one. `edgenext_xx_small`'s BOOL-target
occurrence (a `zeros`-sourced mask, immediately followed by
`bitwise_not.default`) moves to `bitwise_not.default` itself — the next row
of the "Pointwise / type" deferred-backlog family. `mvitv2_tiny`'s
LONG-target occurrences (18 of them, `add` results cast for indexing) move
to `index.Tensor`'s already-tracked multi-entry case
(`"index.Tensor.indices is not an optional tensor list"`) — confirming
MViTv2, not only MaxxViTv2, needs that open slice.
`native_builds`/`native4d_converts`/`kernel_converts`/all-three-stages are
unchanged (88/56/63/43) — depth moved, stage-completion did not, since
neither model reaches `native_builds:true` yet.

**Next**: `lstm.input` from the deferred backlog (Sequencer2D's own first
frontier), P1's remaining one-model slices (`adaptive_max_pool2d`,
`conv1d`/`unfold`, `im2col`/`col2im`, `upsample_bicubic2d`), or a
`Repeat4`/`RepeatInterleave4` Native4D counterpart now that `convit_tiny`
gives them a real corpus frontier to verify against.

### 2026-08-30 — `rsub.Scalar`, the reverse of `sub.Tensor`'s scalar form, plus Const-SSA `Sigmoid`/`Rsub_scalar`

Closes `rsub.Scalar`'s row in [`todo-ops.md`](todo-ops.md)'s deferred
target-name depth backlog (exposed as ConViT's new first frontier by this
session's `select_scatter.default` landing), and — as a direct consequence —
two rows of the Const-SSA registry gap (`Sigmoid` then `Rsub_scalar` itself).

**What landed (`rsub.Scalar` / `Rsub_scalar`)**

- `lib/native/ops/pointwise_binary.ml`: a new `Rsub_scalar` module, `params =
  { other : float; alpha : float }` + `x`, computing `other - alpha * x`.
  Genuinely its own op rather than a legalization onto
  `Add_scalar`/`Mul_scalar`: composing those two would decompose one ATen
  node into two Native ones, the exact failure mode
  `.ai/native_add_op.md`'s design goal rules out (unlike `sub.Tensor`'s own
  scalar form, which legalizes to `Add_scalar` alone with a negated scalar —
  one node, not two). No new `SEMANTICS` primitive: `S.sub`/`S.mul`/`S.const`
  already exist.
- `lib/native/graph_ir.ml`/`.mli`: `Rsub_scalar` constructor (between
  `Rms_norm` and `Sdpa`) + registry entry.
- `lib/native/graph_shape.ml`, `lib/native/eval_op.ml`: shape/compute
  dispatch arms, the same shape `Pow`'s own arms take.
- `lib/native/transform/output_transfer.ml`: classified `Continuous`, same
  bucket as every other scalar-parameter arithmetic op.
- `lib/native/transform/passes/sink_permute.ml`: added to the `elementwise`
  allowlist (its scalar params are per-op constants, not per-axis data, the
  same reasoning `Add_scalar`/`Pow`/etc. already have there); not added to
  `reuse_permute.ml`'s `binary_elementwise`, which is unary-operand ops'
  own exclusion (only one tensor operand).
- `lib/native/graph_builder.ml`/`.mli`: `rsub_scalar` builder function.
- `lib/native4d/{op,domain,lower_engine,graph_shape4,eval_op4,
  output_transfer4,builder}.ml`: full Native4D counterpart, six trivial
  mirror sites — since the op carries no axis parameter, Native4D reuses
  Native's own `Pointwise.Rsub_scalar.t` payload directly (no `Ops4` module,
  no axis-domain check), the exact treatment `Add_scalar`/`Mul_scalar`/`Pow`
  already get. Full-stack in this single landing, unlike `Select_scatter`'s
  deliberately-deferred counterpart, precisely because there is no axis to
  design a dialect boundary around.
- `bin/aten_ops_gen.ml`: `op "rsub" ~overload:"Scalar"` added to the curated
  ATen binding selection (schema `rsub.Scalar(Tensor self, Scalar other,
  Scalar alpha=1) -> Tensor`, two `Scalar` args, one defaulted — the
  generator emitted the `atg_rsub_Scalar` C shim and the matching
  `Interp_dispatch` arm with no hand-written `Override`/`Walk_meta` needed,
  the same clean case `select_scatter.default`'s own curated addition was).
- `lib/native_aten_bridge/op_bridge_pointwise.ml`: `rsub.Scalar` dispatch
  arm (between `rsqrt.default` and `sigmoid.default`), decoding `other`
  (required) and `alpha` (`scalar_arg ~default:(Aten_scalar.Float 1.)`, the
  ATen-side helper — note its `~default` is an `Aten_scalar.t`, not a bare
  `float`, unlike the payload-free `Native_interp`-side `scalar_arg`).
- `lib/native_interp/native_interp_lower_compute.ml`: the payload-free
  dispatch arm (the one `test/data/pt2_json_model_support.jsonl` actually
  measures), right after `rsqrt.default`'s.
- Tests: `test/native/pointwise_test.ml` (Direct compute, default and
  non-default `alpha`), `graph_direct_pointwise_test.ml` +
  `graph_symbolic_pointwise_test.ml` (one-node graph, Direct-vs-Symbolic
  agreement — ConViT's own `1 - sigmoid(...)` values),
  `test/native_bridge/dispatch_test.ml` (**new section**: `verify_print`
  against real ATen for default and non-default `alpha`, a single-node
  graph-shape pin), `test/native4d/{op_json_test,compute_test,
  fixtures4}.ml` (JSON round-trip sample, per-op Direct-vs-Symbolic fixture).

**What landed (Const-SSA `Sigmoid` and `Rsub_scalar`)**

Regenerating the corpus census after `rsub.Scalar` landed showed
`convit_tiny` moving from `native_builds:false` (unsupported operator) to
`native_builds:false` again, but for a different reason: `"Sigmoid is not a
Const-SSA operation"` — the same whack-a-mole pattern the five-op admission
series (`Reshape`/`Expand`/`Mul_scalar`/`Pow`/`Add_scalar`) hit earlier this
session. Admitting `Sigmoid` then exposed `Rsub_scalar` itself as the next
link in the same fold chain, so both are landed together here rather than
across two sessions.

- `lib/native/transform/const_ssa.ml`: `Const_ssa.allows` now also accepts
  `Graph_ir.Sigmoid` and `Graph_ir.Rsub_scalar`.
- `lib/native/transform/const_ssa_symbolic.ml`: a new `sigmoid_expr` helper
  mirroring `Pointwise_activation.Sigmoid.Compute.pixel` exactly (`1 / (1 +
  exp(0 - v))`, not a generic sigmoid primitive — the same reason `pow_expr`
  stays `PowKernel.cpp`'s special-cased expression, since map verification
  compares this grounding against materialization, which calls the same
  `Compute` functor via `Eval_direct`); its `ground` arm broadcasts the
  operand coordinate like `Sqrt`'s (always a no-op for this unary op, but
  keeps the shape-safety helper used uniformly). `Rsub_scalar`'s own arm
  grounds to `other - alpha * x`, wrapped in one `Round` like the other
  arithmetic cases, with no broadcast (same shape as its operand, the
  choice `Add_scalar`/`Mul_scalar`/`Pow` already make).
- Tests: `test/native/const_ssa_test.ml` — for each op, an export/validate
  test, a grounding-to-expression test (pinning the exact algebraic
  expression against a hand-picked input so a wrong formula is visible, not
  just a wrong shape), and a full materialize test (captured input through
  `Const_ssa_materialize.materialize`, comparing the resulting `Tensor.t`).

**Classification**: Full-stack across every surface in
[`todo-ops.md`](todo-ops.md)'s completion-rule table for `rsub.Scalar` —
ATen boundary (curated binding), both importers, Native Direct/Symbolic,
and a full Native4D counterpart, each with its own test evidence. The
Const-SSA admissions are a orthogonal verification-path extension (constant
folding), not a new operation; they are what let `convit_tiny` actually
reach `native_builds:true` rather than stopping one gate short of it.

**Corpus effect**: `convit_tiny` — the one corpus model gated on
`rsub.Scalar` (10 occurrences, ConViT's own GPSA gating expression) — now
reaches `native_builds:true` (was `false`), stopping instead at `Repeat`'s
already-tracked missing `Repeat4` Native4D counterpart (`no legalization for
repeat x=t196 params={repeats=[W=14 C=14]}`), the deliberate deferral
`repeat.default`'s own earlier landing recorded. `native_builds` moves
87→88; `native4d_converts`/`kernel_converts`/all-three-stages are unchanged
(56/63/43) — `convit_tiny` was and remains `native4d_converts:false`.

**Next**: `_to_copy.default` is now the highest-leverage remaining target (2
models: EdgeNeXt, MViTv2) but still needs the value-domain trunc/round
`SEMANTICS` primitive flagged earlier this session. `Repeat4`/
`RepeatInterleave4` now have a real corpus frontier (`convit_tiny`) to
verify a Native4D counterpart against, promoting them from "no corpus model
reaches this boundary yet" backlog to ordinary next-priority work. Otherwise,
P1's remaining one-model slices (`adaptive_max_pool2d`, `conv1d`/`unfold`,
`im2col`/`col2im`, `upsample_bicubic2d`), or `lstm.input` from the deferred
backlog.

### 2026-08-30 — `select_scatter.default`, the write-back counterpart to `select.int`

Closes `select_scatter.default`'s row in [`todo-ops.md`](todo-ops.md)'s
deferred target-name depth backlog (exposed as ConViT's new first frontier by
this session's `copy.default` landing).

**What landed**

- `lib/native/ops/split.ml`: a new `Select_scatter` module in the same file
  as `Select`/`Slice`/`Unbind`/`Split_with_sizes`. `params = { axis : Axis.t;
  index : int }`, its own record (not a type alias of `Select.params`, so
  field access at every call site stays unambiguous) with a `select_params`
  conversion function to reuse `Select`'s own `output_shape`/params where
  needed. `output_shape ~self_shape ~src_shape` returns `self_shape`
  unchanged after checking `src_shape` against exactly the shape `Select`
  itself would produce at this `axis`/`index` (`Select.output_shape
  (select_params p)`) — the same "operand shape not trusted, checked against
  the derived expectation" discipline `Affine_bias`/`Norm.check_affine` use,
  generalised past their per-channel case; a mismatch is a new
  `Shape_error.Select_scatter` row (own module, not `Operand_shape`, since
  that module's own doc comment scopes it to an OPTIONAL operand and `src`
  is required). `Compute.pixel` builds `src`'s read coordinate by running
  `Aten_shape.repack_dropped`'s pairing in the OPPOSITE direction from
  `Select.Compute.pixel`'s own `base` — `out` here ranges over `self`'s
  UNDROPPED shape, unlike `Select`'s `out`, which ranges over the packed
  (dropped) shape, so the fold reads `out` at each KEPT axis and writes to
  the corresponding PACKED axis rather than the reverse. Getting this
  direction backwards was caught by the Direct compute test before this
  landed (an out-of-bounds `Tensor.read`) — see "Getting a real ATen
  oracle" in `.ai/native_add_op.md`'s sibling doc on why the compute test is
  not optional. The branch itself (`out[axis] = index ? src : self`) needs
  no new `SEMANTICS` primitive: `S.index_eq`/`S.select`, added for `Pad`'s
  reflect mirror, are the whole basis, per `semantics.ml`'s own comment.
- `lib/native/graph_ir.ml`/`.mli`: `Select_scatter` constructor (between
  `Select` and `Sigmoid`) + registry entry.
- `lib/native/graph_shape.ml`, `lib/native/eval_op.ml`: two-operand shape and
  compute dispatch arms, the same shape `Index_tensor`'s `~self_shape ~self
  ~index` arms take.
- `lib/native/transform/output_transfer.ml`: classified `Reindexing` — every
  output is copied from ONE of `self`/`src` with no arithmetic, and the
  choice is a structural fact about the output coordinate (never about
  either operand's stored value), so it is exactly as sound as `Concat`'s
  N-input selection, of which this is the two-input, axis-narrowed case
  (distinguished in the comment from `Index_tensor`'s data-dependent branch,
  which is `Discontinuous`).
- `lib/native/graph_builder.ml`/`.mli`: `select_scatter` builder function.
- `lib/native4d/domain.ml`, `lib/native4d/lower_engine.ml`: `Select_scatter`
  added to the existing "dialect does not have it at all" bucket alongside
  `Index_tensor`/`Repeat`/`RepeatInterleave`/`Softmax` — no `Select_scatter4`
  counterpart exists yet, so there is no axis-domain distinction to draw.
- `bin/aten_ops_gen.ml`: `op "select_scatter"` added to the curated ATen
  binding selection, right after `select`'s own entry. Unlike
  `Repeat`/`RepeatInterleave`/`copy.default` (which read only shapes/values
  already materialized by an earlier node, needing no ATen kernel call at
  all), `select_scatter`'s own values are exactly what real ATen computes,
  so this landing added the curated binding rather than settling for
  hand-derived dispatch-only evidence — the generator emitted both the
  `atg_select_scatter` C shim and the matching `Interp_dispatch` arm with no
  further hand-written code, confirming the schema (with its `SymInt index`)
  needed no `Override`/`Walk_meta` entry.
- `lib/native_aten_bridge/op_bridge_shape.ml`: `select_scatter.default`
  dispatch arm, immediately after `select.int`'s, sharing its `norm_dim`/
  `Aten_shape.axis_of_dim`/`Aten_shape.resolve_index` canonicalisation.
- `lib/native_interp/{native_interp_error,native_interp,
  native_interp_lower_shape}.{ml,mli}`: the payload-free dispatch arm (the
  one `test/data/pt2_json_model_support.jsonl` actually measures), sharing
  the existing `resolve_select_index` helper, and a new `` `Select_scatter_input ``
  metadata role (own role per op, matching `Select_input`/`Stack_input`'s
  own precedent, even though the rank-resolution purpose is identical).
- Tests: `test/native/slice_select_test.ml` (Direct compute: writes `src` at
  the index and keeps `self` elsewhere, and the src-shape-mismatch
  rejection), `test/native/graph_direct_pad_slice_test.ml` +
  `graph_symbolic_pad_slice_test.ml` (two-input graph, Direct-vs-Symbolic
  agreement — the staged term shows the `select((H = 1), src[...], self[...])`
  branch grounding identically to Direct), `test/native_bridge/
  shape_ops_test.ml` (**new section**: `verify_print` against real ATen
  across axes/negative dim+index, `dispatch_print` for the out-of-range-index
  and wrong-src-shape rejections, and a single-node graph-shape pin, mirroring
  `select.int`'s own section immediately above it).

**Classification**: Full-stack for the Native surfaces (ATen boundary — now
a curated binding, not merely shape/value reads — both importers, Native
Direct/Symbolic with Direct-vs-Symbolic agreement) per `todo-ops.md`'s
completion rule; **incomplete** for Native4D/Kernel, since no counterpart
exists. May count toward Native target/configuration depth and
`native_builds`, not toward `native4d_converts`/`kernel_converts`/
all-three-stages.

**Corpus effect**: `convit_tiny` is the one corpus model gated on
`select_scatter.default` (30 occurrences, the write-back half of every
`select`/`copy.default` pair this session already landed). Its frontier
moves to `rsub.Scalar` — the next row of the "Pointwise / type"
deferred-backlog family (10 occurrences) — rather than to
`native_builds:true`. `native_builds`/`native4d_converts`/`kernel_converts`/
all-three-stages are unchanged (87/56/63/43) — depth moved, stage-completion
did not.

**Next**: `rsub.Scalar` (ConViT's new frontier) is the natural next target.
`_to_copy.default` remains the other 2-model target but still needs the
value-domain trunc/round `SEMANTICS` primitive flagged earlier this session.
Otherwise, P1's remaining one-model slices (`adaptive_max_pool2d`,
`conv1d`/`unfold`, `im2col`/`col2im`, `upsample_bicubic2d`), or `lstm.input`
from the deferred backlog.

### 2026-08-30 — `copy.default`, bound directly to the existing `Expand` node

Closes `copy.default`'s row in [`todo-ops.md`](todo-ops.md)'s deferred
target-name depth backlog (exposed as ConViT's new first frontier by this
session's `repeat_interleave.self_int` landing).

**What landed**

No new Graph_ir operation. `copy(Tensor self, Tensor src, bool
non_blocking=False) -> Tensor` binds directly to the pre-existing
`Pointwise.Expand` node: real ATen's `copy_`/`copy` fully OVERWRITES `self`
with `src` broadcast to `self`'s shape, so the result depends only on
`src`'s values and `self`'s declared shape/dtype, never on what `self` held
before — exactly `Expand{size=self's shape}` applied to `src`. This is the
same "map onto an existing op after translating parameters" route
`sub.Tensor`'s scalar form takes to `Add_scalar`
(`.ai/native_add_op.md`'s sanctioned case, not a decomposition: one node,
`copy.default`'s own identity not preserved in the graph, same as that
precedent). `non_blocking` is a device-transfer hint, read-and-discarded the
same way `implicit` is in `expand.default`'s own arm.

Verified against the whole 100-model corpus (not assumed): every
`copy.default` occurrence (40 total, ConViT and MViTv2) has IDENTICAL
`self`/`src` shape and dtype — zero broadcasts, zero casts observed. They
are all the functionalized form of `tensor[idx] = value`
(`__setitem__` → `select.int` view + `copy_`, functionalized to `select` +
`copy.default`, with a `select_scatter.default` immediately downstream
writing the computed value back into the base tensor). The landed
translation is nonetheless the fully general broadcast, not an overfit to
the observed equal-shape case — proved in both test suites with a genuine
`self` != `src` shape.

- `lib/native_aten_bridge/op_bridge_shape.ml`: `copy.default` dispatch arm,
  reading `self`'s shape via `Aten_tensor.shape` + `Aten_shape.of_aten`
  (no `-1` convention to resolve, unlike `expand.default`'s own `size`), and
  building `src`'s Native operand via `native_tensor_arg`. `self`'s own
  Native operand is never built — its VALUE is provably irrelevant.
- `lib/native_interp/native_interp_lower_shape.ml`: the matching
  payload-free arm — `self`'s declared shape is already a right-aligned
  `Vec6.shape` via `tensor_shape`, needing no rank/`-1` resolution at all.
- No changes to `lib/native/graph_ir.ml`, `graph_shape.ml`, `eval_op.ml`,
  `output_transfer.ml`, `graph_builder.ml`, or `lib/native4d/{domain,
  lower_engine}.ml` — binding to an EXISTING op touches none of the
  exhaustive-match sites a new `Graph_ir` constructor would force. Native4D
  is consequently full-stack already: `Expand4` (landed earlier this
  session's predecessors) is admitted unconditionally by
  `Domain.check_node`.
- Tests: `test/native_bridge/copy_test.ml` (**new file**, ATen-linked
  `Op_bridge.dispatch`, hand-derived values: equal-shape identity,
  `non_blocking` ignored, a genuine broadcast, and the non-broadcastable
  rejection reusing `Pointwise.Expand.output_shape`'s own check),
  `test/native_interp/copy_test.ml` (**new file**, payload-free, asserts the
  built graph structure — an `Expand` node, not a `Copy` node — mirroring
  `expand_test.ml`'s own `dump` pattern).

**Classification**: Full-stack across every surface in
[`todo-ops.md`](todo-ops.md)'s completion-rule table, Native4D included,
since the landing reuses an already-full-stack op rather than adding a new
one.

**Corpus effect**: `convit_tiny` — the one corpus model actually gated on
`copy.default` as its first frontier (MViTv2 also has 10 occurrences, but
they sit behind its already-earlier `_to_copy.default` blocker, so its
frontier is unchanged) — moves to `select_scatter.default`: the predicted
write-back node immediately downstream of each `copy.default`, 30
occurrences, same "Factories, indexing, and copies" deferred-backlog
family. `native_builds`/`native4d_converts`/`kernel_converts`/
all-three-stages are unchanged (87/56/63/43) — depth moved, stage-completion
did not.

**Next**: `select_scatter.default` (ConViT's new frontier) is the natural
next target, being the write-back counterpart of `select`/`copy` just
landed. `_to_copy.default` remains the 2-model target (EdgeNeXt, MViTv2)
needing the value-domain trunc/round `SEMANTICS` primitive flagged earlier
this session. Otherwise: `lstm.input` from the deferred backlog, or P1's
remaining one-model slices (`adaptive_max_pool2d`, `conv1d`/`unfold`,
`im2col`/`col2im`, `upsample_bicubic2d`).

### 2026-08-30 — `repeat_interleave.self_int`, the Native counterpart to ATen's `Tensor.repeat_interleave` (explicit `dim`)

Closes `repeat_interleave.self_int`'s row in
[`todo-ops.md`](todo-ops.md)'s deferred target-name depth backlog (exposed
as ConViT's new first frontier by this session's own `repeat.default`
landing).

**What landed**

- `lib/native/ops/repeat.ml`: a new `RepeatInterleave` module (the file's
  header comment retitled to cover ATen's whole `repeat` family), sharing
  the file's `bounded_axis_extent` helper (factored out of `Repeat`'s own
  `axis_extent` in this entry) for the same int64-bounded-product-before-
  narrowing guard. `RepeatInterleave.params = { axis : Axis.t; repeats :
  Op_config.Pos.t }` — ONE named axis and a scalar multiplier, unlike
  `Repeat`'s six-axis vector, matching the real ATen argument shape
  (`repeat_interleave(self, repeats: SymInt, dim: int?)`).
  `Compute.pixel` reads `out / repeats` (floor division, via the existing
  `S.index_floor_div_pos` primitive — no new `SEMANTICS` primitive needed,
  same as `Repeat`) on the named axis only; every other axis passes through
  unchanged. This is the exact *opposite* composition from `Repeat`'s
  wraparound modulo: contiguous duplication (`[x,y]` repeat-interleaved by 2
  is `[x,x,y,y]`) vs. tiling (`[x,y]` repeated by 2 is `[x,y,x,y]`).
- `lib/native/op_config.ml`/`.mli`: `` `Repeats `` added to
  `Op_config.Bad.param` — the shared closed vocabulary both importers use
  to reject a non-positive op-configuration value with one detection
  origin, the same mechanism `` `Stride ``/`` `Kernel_size `` already use.
- `lib/native/graph_ir.ml`/`.mli`, `graph_shape.ml`, `eval_op.ml`,
  `transform/output_transfer.ml` (classified `Reindexing`, the same
  "wraparound-gather" argument `Repeat`'s own entry makes, just with floor
  division standing in for modulo), `graph_builder.ml`/`.mli`: the usual
  registry/dispatch/builder wiring.
- `lib/native_aten_bridge/op_bridge_shape.ml`: `repeat_interleave.self_int`
  dispatch arm, restricted to an explicit `dim` (the `dim=None`
  flatten-first form is a genuinely different reshape-then-repeat
  composition with no corpus occurrence, rejected via
  `` `Validation_failure ``). `output_size` is read-and-discarded, the same
  treatment `implicit` gets in `expand.default`'s arm.
- `lib/native_interp/{native_interp_error,native_interp,
  native_interp_lower_shape}.{ml,mli}`: the payload-free dispatch arm (the
  one `test/data/pt2_json_model_support.jsonl` actually measures), a new
  `` `Repeat_interleave_input `` metadata role, and a new
  `` `Repeat_interleave_dim `` row on `unsupported_option` for the same
  `dim=None` rejection, reported through the existing `Unsupported_option`
  mechanism rather than a bespoke error type.
- Tests: `test/native/repeat_test.ml` (Direct compute, three cases: two
  axes' duplication + the by-1 identity), `test/native/
  graph_direct_shape_test.ml` + `graph_symbolic_shape_test.ml` (one-node
  graph, Direct-vs-Symbolic agreement), `test/native_bridge/
  repeat_interleave_test.ml` (**new file**, ATen-linked `Op_bridge.dispatch`,
  hand-derived values, covering both axes of a rank-2 tensor, the identity
  case, the absent-`dim` rejection, and the non-positive-`repeats`
  rejection).

**Classification**: Full-stack for the Native surfaces (ATen boundary, both
importers, Native Direct/Symbolic) per `todo-ops.md`'s completion rule, for
the explicit-`dim` domain only; **incomplete** for Native4D/Kernel (no
counterpart) and for the `dim=None` configuration (typed rejection, not
implemented). May count toward Native target/configuration depth and
`native_builds`, not toward `native4d_converts`/`kernel_converts`/
all-three-stages.

**Corpus effect**: `convit_tiny` is the one corpus model gated on
`repeat_interleave.self_int` (20 occurrences, all `dim` explicit — 0 or 1 —
on ConViT's GPSA relative-position grid, immediately downstream of the
`repeat.default` work this session already landed). Its frontier moves to
`copy.default` — the next row of the same "Factories, indexing, and copies"
deferred-backlog family (40 occurrences, also MViTv2's current frontier) —
rather than to `native_builds:true`. `native_builds`/`native4d_converts`/
`kernel_converts`/all-three-stages are unchanged (87/56/63/43) — depth
moved, stage-completion did not.

**Next**: `copy.default` is now the highest-leverage target (ConViT and
MViTv2 both gated on it). `_to_copy.default` remains the other 2-model
target but still needs the value-domain trunc/round `SEMANTICS` primitive
flagged earlier this session. Otherwise, P1's remaining one-model slices
(`adaptive_max_pool2d`, `conv1d`/`unfold`, `im2col`/`col2im`,
`upsample_bicubic2d`).

### 2026-08-30 — `repeat.default`, the Native counterpart to ATen's `Tensor.repeat`

Closes `repeat.default`'s row in [`todo-ops.md`](todo-ops.md)'s deferred
target-name depth backlog (previously exposed as ConViT's first frontier by
the P0A `zeros.default` landing).

**What landed**

- `lib/native/ops/repeat.ml` (**new file**): `Repeat.params = { repeats :
  Vec6.shape }` (right-aligned multiplier vector, reusing `Vec6.shape`'s
  positive-`Dim.extent` representation since a repeat count, like an extent,
  is always >= 1). `output_shape` computes `x_extent * repeats` per axis,
  bounded in `int64` against `Kernel.Limits.Hard.extent` before narrowing —
  the same "bound the product before multiplying" rule
  `Resize.Bilinear_axis.check` follows, reusing `Window_over_limit`'s
  `` `Output_extent `` row rather than a new fault (the same reuse
  `Pad`/`Concat` make). `Compute.pixel` reads each axis at `out mod
  x_shape`, the exact "mod x d = x - d*(x/d)" idiom `Reshape.delinearize`
  already uses per axis — reused directly, not re-derived — but WITHOUT
  `Reshape`'s flatten-then-redistribute: repeat keeps every axis's own
  identity, so no cross-axis linearization is needed, and `Compute.pixel`
  needs only `x_shape`, not `repeats` itself.
- `lib/native/graph_ir.ml`/`.mli`: `Repeat` constructor + registry entry
  (alphabetical, between `Relu` and `Reshape`).
- `lib/native/graph_shape.ml`, `lib/native/eval_op.ml`: shape and compute
  dispatch arms.
- `lib/native/transform/output_transfer.ml`: classified `Reindexing` — "the
  wraparound gather `Expand`'s broadcast is the degenerate case of": every
  output reads exactly one input element with no arithmetic, taken modulo
  rather than clamped to 0.
- `lib/native/aten_shape.ml`/`.mli`: new `Repeat_size` module +
  `resolve_repeat_size ~self_dims ~repeats`, mirroring `resolve_expand_size`'s
  self-rank check without its `-1` substitution (repeat has only one fault:
  `repeats` shorter than `self`'s own rank). Shared by both importers, so
  they cannot drift on the rank check. `pp_ints`/`pp_dims` hoisted out of
  `Expand_size` to top-level helpers so `Repeat_size` doesn't restate them.
- `lib/native_aten_bridge/op_bridge_shape.ml`: `repeat.default` dispatch arm
  — reads `self`'s shape only (no real ATen kernel call, the same as
  `Reshape`/`Expand`/`Select`'s own arms), so no `lib/aten` C++ binding for
  `repeat` was needed to satisfy the "ATen boundary" surface.
- `lib/native_interp/{native_interp_decode,native_interp_error,
  native_interp_lower_shape,native_interp}.{ml,mli}`: `resolve_repeat`
  (mirroring `resolve_expand`), `Bad_repeat`/`` `Repeat_input `` rows, and
  the payload-free dispatch arm — this is the importer
  `test/data/pt2_json_model_support.jsonl` (and therefore `native_builds`)
  actually measures.
- Tests: `test/native/repeat_test.ml` (Direct compute, hand-derived, mirrors
  `reshape_test.ml`), `test/native/graph_direct_shape_test.ml` +
  `graph_symbolic_shape_test.ml` (one-node graph, Direct-vs-Symbolic
  agreement through `Eval_symbolic`/`Stage_program.ground`),
  `test/native_bridge/repeat_test.ml` (ATen-linked `Op_bridge.dispatch`,
  hand-derived expected values against real `Aten_tensor.t` fixtures,
  covering the rank-extension and rank-too-small cases).

**Deliberately not done**: a `Repeat4` Native4D counterpart. This session
scoped `repeat.default` as the smaller, well-understood slice specifically
*because* it needed no new `SEMANTICS` primitive (unlike the corpus's other
high-leverage frontier, `_to_copy.default`'s float-to-int cast, which does);
extending into Native4D was out of scope for this landing and is tracked as
ordinary backlog (`Repeat` joins `Index_tensor`/`Softmax` in
`Domain.check_node`'s "dialect does not have it at all" bucket — an
**incomplete** state per `todo-ops.md`'s classification table, not a
declared intrinsic-axis exception). No corpus model's frontier reaches this
boundary yet, so there is nothing to verify it against today.

Also **not done**: the optional `lib/native_op_walk` random-walk fuzz
harness (native_add_op.md step 8) — skipped as the doc marks it, in favor of
the required ATen-linked `test/native_bridge/repeat_test.ml` dispatch test,
which already exercises `Op_bridge.dispatch` against real `Aten_tensor.t`
fixtures with hand-derived values.

**Classification**: Full-stack for the Native surfaces (ATen boundary, both
importers, Native Direct/Symbolic) per `todo-ops.md`'s completion rule;
**incomplete** for Native4D/Kernel, since no counterpart exists. May count
toward Native target/configuration depth and `native_builds`, not toward
`native4d_converts`/`kernel_converts`/all-three-stages.

**Corpus effect**: `convit_tiny` is the one corpus model gated on
`repeat.default` (10 occurrences, ConViT's GPSA relative-position grid).
Its frontier moves to `repeat_interleave.self_int` — the next row of the
same "Shape / sequence" deferred-backlog family in `todo-ops.md` — rather
than to `native_builds:true`, since `repeat_interleave.self_int` (20
occurrences) is a later blocker in the same graph. `native_builds`/
`native4d_converts`/`kernel_converts`/all-three-stages are unchanged
(87/56/63/43) — depth moved, stage-completion did not.

**Next**: `repeat_interleave.self_int` (ConViT's new frontier) or one of
P1's remaining one-model slices (`adaptive_max_pool2d`, `conv1d`/`unfold`,
`im2col`/`col2im`, `upsample_bicubic2d`); `_to_copy.default` remains the
higher-leverage (2-model) target but needs a new value-domain trunc/round
`SEMANTICS` primitive first — see `todo-ops.md`'s "Next priority" note.

### 2026-08-30 — `Stack4`, the Native4D counterpart to Native's `Stack`

### 2026-08-30 — `Stack4`, the Native4D counterpart to Native's `Stack`

Closes the `Stack4` row of [`todo-ops.md`](todo-ops.md)'s "Remaining Native4D
counterpart backlog" table.

**What landed**

- `lib/native4d/ops4_split.ml`: `Stack4.params = { axis : Axis4.t }`, `xs :
  Tensor_ref.t list`, delegating its shape rule and pixel map to Native's
  `Concat.Stack` (itself built on `Concat.Concat` over each operand's
  unsqueezed shape) rather than restating either.
- `lib/native4d/ops4.ml`: `Stack4` re-export; `lib/native4d/op.ml`: `Stack4`
  constructor + registry entry (alphabetical); `lib/native4d/builder.ml`:
  `stack4` builder function.
- `lib/native4d/domain.ml`: `Stack` now gets the same `check_dims`-style
  axis-domain rejection `Concat`/`Select`/`Slice`/`Split_with_sizes`/`Unbind`
  already had, replacing its prior blanket `unsupported`.
- `lib/native4d/graph_shape4.ml`: `stack_params` adapter (shared by
  `Eval_op4` and the output-shape rule) + `Stack4` output-shape arm, gathering
  every operand's shape and delegating whole to `Concat.Stack.output_shape`.
- `lib/native4d/eval_op4.ml`: `Stack4` compute arm, through
  `Concat.Stack.Compute` — selects an operand by index, not by a
  within-segment offset, unlike `Concat4`'s arm.
- `lib/native4d/output_transfer4.ml`: `Stack4` classified `Reindexing` (pure
  data movement, no arithmetic — the same argument `Concat4`/`Select4` make).
- `lib/native4d/lower_engine.ml`: the actual `Stack -> Stack4` conversion arm,
  mirroring `Select4`'s: converts the axis, then re-validates the single
  output via `Shape4.of_vec6` — `Stack` *inserts* an axis rather than
  dropping one, so the packed result re-enters the dialect only when `H` is
  unit (the precondition always lands on `H`, not on whichever of `H`/`W`/`C`
  the new axis itself names — see `Ops4_split.Stack4`'s own comment).
- Tests: `test/native4d/op_json_test.ml` (JSON round-trip sample),
  `test/native4d/fixtures.ml` (`stack_n`/`stack_h_batch1`/`stack_h_nonunit`/
  `stack_rank5_t`, the `Stack` analogue of the existing `Unbind` fixture
  family), `test/native4d/domain_test.ml` (the axis-rule/shape-consequence
  pair those fixtures pin), `test/native4d/fixtures4.ml` (a `stack4` per-op
  Direct-vs-Symbolic fixture, H=1 so the result stays four-axis).
- `test/me_visualize_frontier_cram.t` and
  `test/data/pt2_json_model_support.jsonl`: regenerated after
  `make pt2.json-model-support`.

**Classification**: Full-stack, per
[`todo-ops.md`](todo-ops.md)'s completion rule — ATen boundary, both
importers, Native Direct/Symbolic, Native4D counterpart with shape/eval
coverage, and the Native-to-Native4D lowering arm are all present, each with
its own test evidence enumerated above.

**Corpus effect**: two corpus models had `Stack` as their `native4d` frontier.
`csatv2` (`axis=H`, node n17) moves from
`no legalization for stack xs=[t413, t416, t419] params={axis=H}` to a new
frontier one op later: `n19`, a `permute.default` that swaps a non-unit `T`
extent into `H` — an intrinsic multi-axis transpose the N/H/W/C dialect
cannot represent, not a missing counterpart, so this is a genuine Native-only
boundary rather than further Native4D work. `skresnet18` (`axis=D`, node n28)
moves from the same generic `no legalization for stack ...` message to the
dialect's own axis-domain diagnostic, `node n28: axis D is outside the
N/H/W/C dialect` — the same node and the same (already-intentional) rejection,
now reported through `check_dims` instead of falling through to the lowerer's
catch-all. Both models keep `native4d_converts:false`; `csatv2` remains a
graph-only CI target (`kernel_reason: over_limit`) and `skresnet18` remains
`kernel_converts:true` independent of this change. None of the four scoreboard
counts move — "depth moved; stage-completion did not", per
[`todo-ops.md`](todo-ops.md)'s recording rule.

**Next**: `Softmax4` is the one remaining row of the Native4D counterpart
backlog table, with no corpus model currently gated on it (Softmax has no
Native4D counterpart at any axis today, per `.ai/matmul_softmax_design.md`
§3). Otherwise, P1's remaining one-model slices
(`adaptive_max_pool2d`, `conv1d`/`unfold`, `im2col`/`col2im`,
`upsample_bicubic2d`) and promoting
`repeat.default`/`_to_copy.default`/`lstm.input` from the deferred backlog
are the next-ordered breadth-first work per `todo-ops.md`.

### 2026-08-30 — `Select4`, the Native4D counterpart to Native's `Select`

### 2026-08-30 — `Select4`, the Native4D counterpart to Native's `Select`

Closes the `Select4` row of [`todo-ops.md`](todo-ops.md)'s "Remaining Native4D
counterpart backlog" table.

**What landed**

- `lib/native4d/ops4.ml` (payload moved to new `lib/native4d/ops4_split.ml`,
  see below): `Select4.params = { axis : Axis4.t; index : int }`, delegating
  its shape rule and pixel map to Native's `Split.Select` (which itself
  delegates to `Split.Slice`) rather than restating either.
- `lib/native4d/op.ml`: `Select4` constructor + registry entry (alphabetical).
- `lib/native4d/builder.ml`: `select4` builder function.
- `lib/native4d/domain.ml`: `Select` now gets the same `check_dims`-style
  axis-domain rejection `Slice`/`Concat`/`Split_with_sizes`/`Unbind` already
  had, replacing its prior blanket `unsupported`.
- `lib/native4d/graph_shape4.ml`: `select_params` adapter (shared by
  `Eval_op4` and the output-shape rule, so they cannot disagree about which
  axis is dropped) + `Select4` output-shape arm.
- `lib/native4d/eval_op4.ml`: `Select4` compute arm, through
  `Split.Select.Compute`.
- `lib/native4d/output_transfer4.ml`: `Select4` classified `Reindexing` (pure
  data movement, no arithmetic — the same argument `Slice4`/`Unbind` make).
- `lib/native4d/lower_engine.ml`: the actual `Select -> Select4` conversion
  arm (this was the missing piece found mid-session — `domain.ml` already
  admitted `Select` through the axis-domain check, but no lowerer arm existed
  yet, so a legal graph would have hit the lowerer's own
  domain-check/match-disagreement invariant). Converts the axis, then
  re-validates the single output via `Shape4.of_vec6` for the same reason
  `Unbind`'s multi-output arm does: `Select` drops its axis, so the result
  re-enters the dialect only when the packed shape stays four-axis.
- `lib/native4d/ops4_split.ml` (**new file**): `Slice4`/`Select4`/`Concat4`/
  `Unbind`/`Split_with_sizes4` moved out of `ops4.ml` verbatim, re-exported by
  the same names, to keep `ops4.ml` under the tracked 1000-line file-size
  ceiling (`scripts/check-file-size.sh`) after `Select4`'s addition pushed it
  to 1026 lines. Mirrors the precedent `ops4_conv.ml` set for the convolution
  family.
- Tests: `test/native4d/{op_json_test,fixtures,fixtures4,domain_test,
  lower_test,compute_test,verify_test}.ml` — JSON round-trip sample,
  axis-rule/shape-consequence fixtures (`select_c_batch1`/`select_n`/
  `select_c_batch2`/`select_rank5_t`, the `Select` analogue of the existing
  `Unbind` fixture family), a per-op Direct-vs-Symbolic fixture, a hand-value
  Direct oracle test (checked against `Split.Select.Compute`, not against
  Native — proving the arithmetic, not just self-consistency), a lowering
  golden, and an end-to-end Native-vs-Native4D numeric verify test.
- `test/me_visualize_frontier_cram.t` and
  `test/data/pt2_json_model_support.jsonl`: regenerated after
  `make pt2.json-model-support`.

**Classification**: Full-stack, per
[`todo-ops.md`](todo-ops.md)'s completion rule — ATen boundary, both
importers, Native Direct/Symbolic, Native4D counterpart with shape/eval/oracle
coverage, and the Native-to-Native4D lowering arm are all present, each with
its own test evidence enumerated above.

**Corpus effect**: `csatv2` is the only corpus model whose frontier `Select4`
was gating (`.ai/`-tracked design intent aside, `Select` occurs at `axis=H`
in that graph's `n3`). Its frontier moves from
`no legalization for select x=t405 params={axis=H index=0}` to
`no legalization for stack xs=[t413, t416, t419] params={axis=H}` — i.e. to
the next item in the same "Remaining Native4D counterpart backlog" table
(`Stack4`). `csatv2` stays `native4d_converts:false` either way and remains a
graph-only CI target (`kernel_reason: over_limit`, independent of this
change), so none of the four scoreboard counts above move. This is exactly
the "depth moved; stage-completion did not" outcome
[`todo-ops.md`](todo-ops.md) asks landings to record rather than treat as no
progress.

**Next**: `Stack4` (csatv2's new frontier) and `Softmax4` are the two
remaining rows of the Native4D counterpart backlog table; `Stack4` now has a
concrete corpus model to verify against, the same way `Select4` did coming
into this session.
