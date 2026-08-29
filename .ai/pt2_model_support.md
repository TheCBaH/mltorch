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
| `fastvit_sa12` | 332 | yes (19 `Gelu`, SDPA) | no — SDPA is intentionally rejected by Native4D (see `.ai/attention_design.md`); also one `outside_dialect_domain` diagnostic (axis T) unrelated to activations | `native-infer-verify.fastvit_sa12` passes (max_abs 1.4e-5, cosine_similarity 1), doesn't need Native4D at all; `native-transform-verify.fastvit_sa12`'s third pass (`--verify-symbolic standard`) not yet timed on this model's 1218 real nodes, so not wired into `PT2_MODELS_NATIVE_VERIFY` pending that — the other two passes of that target would very likely also pass |
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
- **FastViT's remaining blocker is SDPA**, which `.ai/attention_design.md`
  already settles as a closed decision (Native4D rejects SDPA by
  construction — `D` is not a Native4D axis at any extent), not an open
  question.
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
