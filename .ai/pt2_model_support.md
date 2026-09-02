# PT2 functional-release model support

A snapshot of what the six models in the PT2-functional-ATen release archive
(`data/pt2-functional/`, fetched via `make pt2.download-cram` +
`make pt2.download PT2_MODEL=csatv2`; see `PT2_MODELS_CRAM`/`PT2_MODELS_ALL`
in the Makefile) actually do at each layer of the pipeline. This is prose over
`test/pt2_model_support_cram.t`, the pinned, parseable snapshot: that cram
test's JSON-Lines golden is the source of truth for the "Native graph builds"/
"Native4D converts" columns and is checked (and re-diffed on drift) by
`make pt2.runtest`, so read this table as commentary, not as the thing to
trust if the two disagree.

| Model | Nodes | Native graph builds | Native4D converts (`to4d --fold`) | Native-vs-ATen numeric match (`native-infer-verify`) |
|---|---|---|---|---|
| `mobilenetv2_050` | 152 | yes | yes, fully (100 canonical nodes) | yes, wired into `PT2_MODELS_NATIVE_VERIFY` |
| `regnetx_002` | 145 | yes | **no** — stops at one 3-group `convolution` node (see below) | yes, wired into `PT2_MODELS_NATIVE_VERIFY` |
| `efficientnet_b0` | 240 | yes (16 `Sigmoid`, 49 `Silu`) | yes, fully (254 canonical nodes) | yes, wired into `PT2_MODELS_NATIVE_VERIFY` |
| `test_convnext2` | 71 | yes (4 `Gelu`) | yes, fully (49 canonical nodes) | yes, wired into `PT2_MODELS_NATIVE_VERIFY` |
| `fastvit_sa12` | 332 | yes (19 `Gelu`, SDPA) | no as of this table (2026-08-23) — SDPA was unconditionally rejected by Native4D then; that changed 2026-09-02 (`.ai/native4d_design.md` §7.9, SDPA now converts at `D = 1`), so this row needs re-running before citing it — the row's OWN `outside_dialect_domain` diagnostic (axis T) is unrelated to SDPA and may still block this model regardless | `native-infer-verify.fastvit_sa12` passes (max_abs 1.4e-5, cosine_similarity 1), doesn't need Native4D at all; `native-transform-verify.fastvit_sa12`'s third pass (`--verify-symbolic standard`) not yet timed on this model's 1218 real nodes, so not wired into `PT2_MODELS_NATIVE_VERIFY` pending that — the other two passes of that target would very likely also pass |
| `csatv2` | 945 | **no** — deliberately graph-only CI fixture (see `.ai/testing_strategy.md`) | no | no |

## How this was produced (2026-08-23)

```sh
make pt2.download-cram && make pt2.download PT2_MODEL=csatv2

# "Native graph builds" + per-op node counts:
dune build bin/native_graph.exe
_build/default/bin/native_graph.exe visualize --pt2 data/pt2-functional/<model>/<model>.pt2 --output /tmp/<model>.json
jq '.diagnostics' /tmp/<model>.json                                                    # hard rejections, if any
jq '.graphCollections[0].graphs[] | select(.id=="g/native/000") | .nodes | length' /tmp/<model>.json
jq -r '.graphCollections[0].graphs[] | select(.id=="g/native/000") | .nodes[].label' /tmp/<model>.json | sort | uniq -c

# "Native4D converts" -- distinguishes a missing operation from a genuine
# four-axis domain limit (the reason this command exists at all):
_build/default/bin/native_graph.exe to4d --pt2 data/pt2-functional/<model>/<model>.pt2 --fold

# "Native-vs-ATen numeric match" -- real weights, real ATen reference:
make native-infer-verify.<model>
```

## Notable findings

- **GELU and Sigmoid no longer block anything.** Before `e775783`/`6cfd2ac`
  (this repo's GELU/Sigmoid/scalar-mul commits), ConvNeXt2 and FastViT
  stopped at `gelu.default` and EfficientNet-B0 at `sigmoid.default` in the
  payload-free probe `ops.md` describes. All three now build a complete
  Native graph with those nodes present, and (for the two with a Native4D
  path) convert fully.
- **RegNetX's blocker is a real domain limit, not a bug.** `to4d` stops at a
  3-group `convolution.default` node. `.ai/native4d_design.md` (lines
  42/78/483-486) already documents this as deliberate: Native4D recognizes
  only `groups=1` and full depthwise (`groups=in_channels`); "any other
  grouping → reject" is the stated rule, and adding general grouped
  convolution "merely to make the conversion total" is explicitly the wrong
  move. No action item follows from this row.
- **FastViT's remaining blocker was SDPA as of this table (2026-08-23), but
  SDPA's `D = 1` admission does not move it.** `.ai/native4d_design.md` §7.9
  (2026-09-02) narrows the closed decision: `D` is not a Native4D axis at any
  extent *above 1*, but SDPA now converts at `D = 1`. Re-running
  `make pt2.json-model-support` (2026-09-03) shows `fastvit_sa12`'s blocker
  is, and already was before this change, `node n604: axis T is outside the
  N/H/W/C dialect` — an earlier, unrelated node than any SDPA node in the
  graph. `Domain.check_node` stops at the first violation, so SDPA's own
  admission was never on the reported path; this model shows no observable
  movement from it. `efficientvit_b0` is the model that actually moves, and
  it moves on `Batched_matmul`, not SDPA — see "Updated 2026-09-03" below.
  `mvitv2_tiny`, this table's original `Batched_matmul` corpus example, does
  not move: it still fails Native import outright, on an unrelated
  `index.Tensor` defect (same section).
- **CSATv2 stays graph-only** by deliberate policy: it needs a matrix/
  attention dialect decision this repo hasn't made yet, plus a structural
  op set (`select`/`unsqueeze`, `cat`/`stack`, `amax`, `pow`/`vector_norm`,
  `upsample_bilinear2d`) it doesn't have. Do not try to make it a native
  verification target piecemeal.

## The broader, payload-free sweep

The six models above are the ones with real weights (downloaded, gated). The
producer submodule (`modules/devcontainer.pytorch-image-models/models/`)
carries a `model.json` for ~100 models with no download at all -- loading,
Native import and Native4D conversion don't need payloads (only `--fold` and
real inference do), so `bin/pt2_json_model_support.ml` sweeps all of them,
writing `test/data/pt2_json_model_support.jsonl`.

That binary calls `Me_export.session` directly -- the same library
`native_graph visualize` calls -- rather than shelling out to the CLI and
re-parsing its JSON/stderr with jq and shell prefix-matching, so there is no
message format for this tool and the CLI to drift apart on; the schema is
just the OCaml `row` type it encodes (see the file's header comment). Each
line carries `model`, `nodes` (from the always-available `stage:source`
capability -- populated even when Native import fails, since decoding is a
strictly earlier step) and `native_builds` (is `stage:initial_native`
`Available`?).

Past Canonical the pipeline FORKS into two independent branches -- this is
also what the Model Explorer web UI's four graph options are: Native4D is one
branch, and Stage Program / Kernel / Kernel Fusion is the other, entirely
unrelated to Native4D's four-axis dialect check (`me_export_shape.ml`'s "the OTHER
branch from canonical Native"). A model can succeed on one and fail the
other, so the file carries two independent, self-contained
`(converts, stage, reason, blocker)` triples rather than one flat set of
fields:
- `native4d_*`: `stage:native4d`'s own outcome.
- `kernel_*`: `stage:kernel`'s outcome, which is a complete read of the whole
  Stage Program -> Kernel -> Fusion branch -- `stage:stage_program` has no
  `Unavailable` case of its own in `Me_export` (a failure there is Fatal, not
  a gradeable capability), and `stage:fusion` always exactly mirrors
  `stage:kernel`'s Ready/Refused, since Fusion is a plan carried by the same
  `Kernel_adapt.t` the kernel graph was built from.

Each triple's `stage`/`reason`/`blocker` name WHERE that branch stopped and
WHY, resolved to the deepest real cause (all four fields `null` on that
branch's success):
- `"source"`: `model.json` itself didn't decode (`reason:
  "model_json_decode"`) -- shared by both triples.
- `"native"`: the payload-free Native import failed -- recoverably (`reason:
  "unsupported_operator"`/`"unsupported_input"`/`"over_limit"`, a session
  still builds) or as a defect the importer flags outright (`reason:
  "malformed"`: an argument value the importer rejects rather than merely
  not implementing, e.g. `clone`'s `memory_format=channels_last` (as of
  2026-08-26; `ContiguousFormat`/`PreserveFormat` are accepted, and
  `max_pool2d`'s `ceil_mode=true` was a third example here until both
  landed, see below)). Shared by both triples too -- a `prerequisite_unavailable`
  reason on either branch's own capability (which carries no detail of its
  own) is ALWAYS resolved here, since neither branch can run at all without
  Native import.
- `"native4d"` (`native4d_*` only): Native4D itself rejected the graph on its
  own terms (`reason: "outside_dialect_domain"` or `"requires_payloads"`).
- `"kernel"` (`kernel_*` only): `Kernel_adapt` itself rejected the stage
  program (`reason: "unsupported_graph_shape"` or `"over_limit"`).

Run `make pt2.json-model-support` to regenerate; `make verify.pristine`
(already in CI, see `.github/workflows/build.yml`'s "pt2 json model support"
+ "verify pristine" steps) catches drift, the same as any other checked-in
generated file in this repo -- no separate diff/check target needed.

As of 2026-08-25: 100 models, 69 have `native_builds:true` (Native import
itself succeeded) -- up from the 50 this section originally reported: Native's
own `cat.default`/`stack.default`/`select.int` support moved 9 models across
that line, `split_with_sizes.default` moved 7 more (an 8th, `lambda_resnet26t`,
still fails to build, now on `softmax.int` instead), then `group_norm.default`
(below) moved the last 3. Of the 31 that don't build, 25 stop on one missing
ATen op each (`reason: "unsupported_operator"`, e.g. `avg_pool2d.default`
non-adaptive (4), `split.Tensor` (3), `alias.default` (3) -- a session still
builds, so `nodes` is populated) and 6 are `"malformed"` (a real importer
defect: `clone.default`'s `memory_format` or `max_pool2d`'s `ceil_mode=true`
-- Fatal, so no session and `nodes` is `null`). This inventory has drifted
since 2026-08-23 as other work landed; it is not re-audited op-by-op here,
only re-totaled.

**Updated 2026-08-26** after landing `max_pool2d.default`'s `ceil_mode=true`:
71 have `native_builds:true`, up 2. Both predicted
models move: `hgnetv2_b0` clears every branch outright (`native4d_converts`
and `kernel_converts` both `true`, all four `reason`/`blocker` fields `null`);
`legacy_seresnext26_32x4d` now builds and passes `kernel_converts`, but stops
at Native4D's pre-existing, already-tracked grouped-convolution domain limit
(`reason: "outside_dialect_domain"`, "convolution has 32 groups, which is
neither 1 nor depthwise" -- the same limit `regnetx_002`'s own frontier hits,
not a new gap). Of the 29 that still don't build, the `"malformed"` bucket
drops from 6 to 4 -- every remaining instance is `clone.default`'s
`memory_format`, next in this row's own list.

**Updated again, same day,** after landing `clone.default`'s
`memory_format`: `native_builds:true` stays at
71 -- this row changes WHICH models are `"malformed"`, not how many build,
since `ContiguousFormat`/`PreserveFormat`/absent were already accepted and
only `ChannelsLast`/`ChannelsLast3d` remain refused. The `"malformed"`
bucket drops from 4 to 1: `csatv2`, `mobilevitv2_175` and `mvitv2_tiny` all
clear the `clone` blocker and move to `"unsupported_operator"` instead
(`index.Tensor`, `softmax.int`, `matmul.default` respectively -- all three
already tracked gaps, not new ones). `hiera_tiny_224` is the lone
remaining `"malformed"` model, but on a DIFFERENT, previously-hidden defect
now exposed by clearing `clone`: `getitem is rank 5, expected 4` (a conv
weight tensor, named `getitem` because it is itself the result of an earlier
multi-output op's `__getitem__`, declared at a rank `Native_interp`'s
`sizes_rank_4` refuses) -- noted here only as a discovered lead, not yet
investigated or scoped as a row.

Of the 69 that build (71 as of 2026-08-26, see above), the two branches
diverge for 35 of them -- proof the fork is real, not just a modeling nicety.
`native4d_converts` is `true` for 34 and `false` for the other 35 (24
`outside_dialect_domain`, 11 `requires_payloads`, e.g. `regnetx_002` and
`convmixer_1024_20_ks9_p14`). `kernel_converts` is `true` for 53 and `false`
for 16, all `over_limit` -- a disjoint failure set from Native4D's:
`regnetx_002` and `convmixer_1024_20_ks9_p14` both have `kernel_converts:true`
despite failing Native4D, and conversely some models pass Native4D while
`kernel_converts` is `false`. (As of 2026-08-26: 35/36 and 55/16
respectively, `hgnetv2_b0` and `legacy_seresnext26_32x4d` both added to the
"build" denominator above.)

Landing a Native4D `Concat4` op (`Domain.check_node` gating its joined axis
the same way `Slice`/`Unbind` gate theirs) moved 9 more
models across the `native4d_converts` line in this run: `rdnet_tiny` and the 8
`rexnet*`/`rexnetr*` variants, all previously blocked on "no legalization for
concat". `rdnet_tiny` still fails `kernel_converts` (`over_limit`, disjoint as
above); the 8 `rexnet*`/`rexnetr*` variants now pass both branches in full.

Landing Native's own `split_with_sizes.default` (a `Graph_ir.Split_with_sizes`
node reusing `Slice`'s per-window shape/pixel
logic and dtype-preserving `Eval_direct` bypass, the same treatment `Unbind`
gets) moved 7 models from `native_builds:false` to `true`:
`inception_next_atto`, `inception_next_tiny`, `mixnet_l`, `mixnet_xl`,
`mixnet_xxl`, `tf_mixnet_l`, `tf_mixnet_s`. All 7 now reach Native4D and stop
there (`outside_dialect_domain`, "no legalization for split_with_sizes" --
Native4D has no `Split4` yet, the same "dialect does not have it at all"
answer `Concat4` used to get); `mixnet_l`/`mixnet_xl`/`mixnet_xxl`/
`tf_mixnet_l`/`tf_mixnet_s` also fail `kernel_converts` (`over_limit`,
disjoint as above; pre-existing for these models, not caused by this
change).

Landing Native's own `group_norm.default` (`Norm.GroupNorm` in
`lib/native/ops/norm.ml`: its own reduction -- a WINDOWED sum over one
group's channel slice nested inside a full-extent sum over every other
non-batch axis, `S.sum`'s first windowed, non-compile-time-constant `lo` in
the engine -- and its own shape rule, channel count must divide `num_groups`)
moved the last 3 models with this blocker from `native_builds:false` to
`true`: `efficientnet_b0_g8_gn`, `efficientnet_b3_g8_gn`,
`test_efficientnet_gn`. All 3 now reach Native4D and stop there
(`outside_dialect_domain`, "no legalization for group_norm" -- Native4D has
no `Group_norm4` yet, the same "dialect does not have it at all" answer
`Concat4`/`Split_with_sizes` used to get). `efficientnet_b0_g8_gn`/
`efficientnet_b3_g8_gn` also fail `kernel_converts` (`over_limit`, disjoint as
above; pre-existing, not caused by this change); `test_efficientnet_gn` passes
it. Unlike `Concat4`/`Split_with_sizes`, `group_norm.default` has no real ATen
C binding (`bin/aten_ops_gen.ml` does not select it): `at::native::group_norm`
sits outside the hand-curated source closure `lib/aten/build_archive.sh`
compiles, and adding it there undefined-symbol'd every binary linking `aten`.
Native import and the payload-free sweep need no ATen binding at all, so this
does not affect `native_builds`; only real-ATen verification
(`Interp_verify`) is unavailable for this op, and its Native correctness is
instead pinned by hand-computed values in `test/native/norm_test.ml`,
cross-checked against an independent calculation.

**Updated again, same day,** after landing `alias.default` (binds to the
*existing* `Graph_ir.Clone` node -- paramless, shape-preserving, already this
repo's "same value, no change" node -- since ATen's `alias(self) -> Tensor`
schema takes no argument beyond `self` and is a pure identity view; Native4D
already elides `Clone` entirely, so this costs nothing there either):
`native_builds:true` stays at 72 -- this landing changes WHICH models are
blocked, not how many build. All 5 models `alias.default` blocked
(`ghostnetv2_100`, `ghostnetv2_130`, `ghostnetv2_160`, `ghostnetv3_050`,
`ghostnetv3_130`) clear it and move to a new, previously-undiscovered
blocker: `upsample_nearest2d.vec` -- a distinct overload from the already-
landed `upsample_bilinear2d.vec`, noted here only as a discovered lead, not
yet investigated or scoped as a row.

**Updated again, same day,** after landing `sum.dim_IntList` (`Reduce.Sum` in
`lib/native/ops/reduce.ml`, the fourth op sharing `Dims_keepdim` alongside
`Mean`/`Amax`/`Vector_norm` -- `Compute` is `Mean`'s numerator with no final
division; the header comment in `reduce.ml` had literally anticipated this
exact op ("a category file so a future `aten.sum`/… lands beside them"). Also
factored the four ops' near-identical Native4D `keepdim=false` correction-C1
legalization (`Mean`/`Amax`/`Sum`/`Vector_norm` each had their own ~30-line
copy) into one shared `lower_keepdim_reduction` helper in
`lib/native4d/lower.ml`, which is what kept that file under the tracked
1000-line ceiling once `Sum`'s own copy was added -- confirmed
behavior-preserving by the full test suite passing unchanged before promoting
any new goldens): `native_builds:true` moves from 77 to 79.
`mobilevitv2_175` clears this blocker and moves to the already-tracked
`expand.default` (3 models). `resnest14d` and `skresnet18` -- the other two
models this blocker predicted to unblock -- now build in full and reach
already-tracked, pre-existing Native4D domain limits (grouped convolution and
"no legalization for split_with_sizes" respectively, both `native4d_converts:
false` but `native_builds:true`) -- neither a new gap.

**Updated again, same day,** after landing `split.Tensor` (the equal-chunk-
size sibling of the already-supported `split_with_sizes.default`, legalized
onto that *existing* `Split_with_sizes` node by deriving its sizes list from
`split_size`/the axis extent -- no new Native surface): `native_builds:true`
stays at 77 -- this landing changes WHICH op blocks each of the 4 models it
was predicted to unblock, not how many build. All 4 (`edgenext_xx_small`,
`efficientvit_b0`, `maxxvitv2_nano_rw_256`, `skresnet18`) clear `split.Tensor`
cleanly and move to independent, already-tracked blockers
(`zeros.default`, `_assert_tensor_metadata.default`, `index.Tensor`,
`sum.dim_IntList` respectively) -- none a new gap.

**Updated again, same day,** after landing `upsample_nearest2d.vec` (a new
Native op, `Resize.Nearest2d` in `lib/native/ops/resize.ml`, alongside
`Bilinear2d`: no `align_corners` at all, since nearest-neighbor has nothing
to interpolate between -- ATen's own `nearest_neighbor_compute_source_index`
is an EXACT integer floor division over the same static graph-construction-
time extents `Bilinear_axis`/`Window_axis`/`Adaptive_axis` already divide, so
this is `Identical` to ATen, not merely `Equivalent`): `native_builds:true`
moves from 72 to 77. All 5 models `alias.default` unblocked above now clear
this new blocker too and build in full: the `ghostnetv2_100`/`_130`/`_160`
trio reaches `native4d_converts:true` and stops only on the already-tracked,
out-of-scope `kernel` `over_limit` ("evaluation depth exceeds 2048" --
`.ai/native4d_design.md`'s Kernel/Fusion note, unrelated to this landing);
the `ghostnetv3_050`/`_130` pair stops on the already-tracked, already-closed
`requires_payloads` domain limit, then the same `over_limit`. Neither is a
new gap.

**Updated 2026-08-29** after landing `avg_pool2d.default` (`Pool.AvgPool2d`
already existed as a Native op -- and already had an ATen binding and a
generated ATen-vs-Native walk, both dormant -- but no `Op_bridge`/
`Native_interp` arm ever built it): `native_builds:true` moves
from 71 to 72. `inception_v3` is the only model in the corpus whose FIRST
blocker was `avg_pool2d.default`, and it clears every branch outright
(`native4d_converts` and `kernel_converts` both `true`). The other 3 models
this op was predicted to unblock (`ghostnetv3_050`, `ghostnetv3_130`,
`maxxvitv2_nano_rw_256`) still don't build: the first two now stop on
`alias.default`, the third on `split.Tensor` -- both already-tracked,
independent gaps (see the "Cross-model signal" list in `ops.md`), not new
ones. 72/100 now build; of those, `native4d_converts` is `true` for 36 and
`false` for 36, and `kernel_converts` is `true` for 56 and `false` for 16.

**Updated again, same day,** after landing `alias.default`, `upsample_nearest2d.vec`,
`split.Tensor`, `sum.dim_IntList`, and finally `expand.default`:
`native_builds:true` moves from 72 to 81 across
the five. `expand.default` (`Pointwise.Expand`, broadcasting `self` to a
resolved target `size` via the existing `Pointwise_binary.broadcast_coord`)
was the last of the five and unblocks all 3 models it was predicted to:
`convit_tiny` builds past this stage and now stops on `zeros.default`
(already tracked, not new); `swiftformer_xs` builds in full and reaches
Native4D's own `no legalization for expand` rejection (`Expand` has no
`Ops4` counterpart at any axis -- a broadcast that may ADD leading axes has
no four-axis shape to check against, so it joins the `Group_norm`/`Select`/
`Softmax`/`Split_with_sizes`/`Stack` bucket); `mobilevitv2_175` (already
unblocked once by `sum.dim_IntList`, independently blocked again by this op)
now builds in full and reaches Native4D's pre-existing `D`-axis domain
limit, then the already-tracked Kernel `over_limit`. `test_vit4` surfaces a
genuinely new gap one stage later: `Const_ssa.allows`
(`lib/native/transform/const_ssa.ml`), the small curated whitelist of ops a
CONSTANT subgraph may fold through (`Add`/`Sub`/`Mul`/`Div`/`Sqrt`/`Permute`
only), does not include `Expand` -- a distinct unit of work, not opened as a
row anywhere yet.

**Updated again, same day,** after landing `_assert_tensor_metadata.default`
(no new `Graph_ir` op at all -- ATen's own schema returns no `Tensor`, so `a`
is routed to the *existing* `Discard` sink `max_pool2d_with_indices.default`'s
dead indices edge already uses, in both `Op_bridge` and `Native_interp`):
`native_builds:true` stays at 81 -- `efficientvit_b0` was already
`native_builds:false` (blocked here since the `split.Tensor` landing above),
and this landing moves which op blocks it rather than clearing it outright.
It now stops on the already-scoped, deliberately-deferred `matmul.default`
batched/multi-head shape family -- not a new gap.

A previous version of this sweep reported `native_builds:true` for all but 5
models -- 95, not 50 -- because it read the payload-free `visualize` CLI's
exit code (which stays 0 whenever `stage:initial_native` degrades gracefully
to `Unavailable` rather than aborting) instead of the `stage:initial_native`
capability itself. That conflated "the CLI didn't crash" with "Native import
succeeded" and undercounted real import failures by 45 models; the current
`bin/pt2_json_model_support.ml` reads the capability directly, matching what
the doc has always claimed `native_builds` means (`to4d`'s first stage
succeeding, per the cram's own field description at the top of
`test/pt2_model_support_cram.t`).

An earlier run of this sweep also reported 14 models failing with
`` `not a well-formed ExportedProgram` `` (`eca_halonext26ts`,
`edgenext_xx_small`, `efficientvit_b0`, the 8 `rexnet*`/`rexnetr*` variants,
`test_vit4`, `vit_small_patch16_dinov3_qkvb`, `volo_d1_224`). That message was
misleading: all 14 are well-formed programs. The real cause, recovered by
running `bin/pt2_census.exe` (which surfaces the raw Jsont decode error
instead of `me_export.ml`'s `Result.to_option`-flattened one) against each
model.json, was `Number 9.2233720368547758e+18 not in OCaml int range` --
every one of the 14 encodes `aten.slice.Tensor`'s `end` argument as Python's
`sys.maxsize` (2^63-1, the "unbounded/slice-to-the-end" sentinel), which
exceeds even a native OCaml `int` (63 bits). Fixed by
`Schema_runtime.python_int_jsont` (`lib/schema_runtime/schema_runtime.ml`),
which `schema_codegen.ml` now emits for union `int` arms (`Argument.as_int`,
`ConstantValue.as_int`, `SymInt.as_int`, `SymExprHint.as_int`) in place of
stock `Jsont.int`: it clamps an out-of-range value to `max_int`/`min_int`
instead of rejecting it, which preserves the sentinel's "unbounded" meaning
losslessly for any real tensor (libtorch's own slice kernel already clamps
`end` to the dimension size). Ordinary struct `int` fields (counts, indices,
versions) are untouched and still reject an out-of-range value as invalid.

**Updated 2026-09-03**, verifying `.ai/native4d-sdpa-compatibility-plan.md`
§3.2's "check before claiming model movement" after `Sdpa`/`Batched_matmul`
were admitted into Native4D at `D = 1` (`97163cf`): re-running
`make pt2.json-model-support` moves exactly one model,
`efficientvit_b0` (`native4d_converts` `false` -> `true`, `native4d_stage`/
`native4d_reason`/`native4d_blocker` all `null`). This is the real payoff:
`efficientvit_b0` was already `native_builds:true`, already known to stop at
`matmul.default`'s "batched/multi-head shape family" (this file's own
2026-08-26 entry above), and the blocker text before this run named exactly
`Batched_matmul`'s rejected batch axis (`"batched matmul's batch axis is D,
which the N/H/W/C dialect has no name for"`) -- the argument §2.1 of the
compatibility plan rebuts. `fastvit_sa12` and `mvitv2_tiny`, this table's two
running SDPA/`Batched_matmul` examples, do **not** move (see their bullets
above) -- neither is evidence against the change, since neither ever reached
the node that changed.

## Keeping this current

- `PT2_MODELS_NATIVE_VERIFY` (Makefile) wires `mobilenetv2_050`,
  `regnetx_002`, `efficientnet_b0` and `test_convnext2` into
  `make native-infer-verify`/`native-transform-verify`, which
  `.github/workflows/build.yml` already runs. Adding a model there is a
  Makefile-only change once it passes both by hand.
- `test/pt2_model_support_cram.t` is the machine-checked half of this
  table: `make pt2.runtest` runs it (locally and in CI, see
  `.github/workflows/build.yml`'s "pt2 load test" step) and fails on any
  drift, promoted the same way as any other cram golden
  (`dune promote test/pt2_model_support_cram.t`). Re-run it after any change
  that adds a Native or Native4D operation, and update this table's prose
  to match rather than letting the two drift apart.
- `test/data/pt2_json_model_support.jsonl` is the payload-free sweep's
  checked-in snapshot; `make pt2.json-model-support` regenerates it and
  `make verify.pristine` (CI) is what actually enforces it stays current.
