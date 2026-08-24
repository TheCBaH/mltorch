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
