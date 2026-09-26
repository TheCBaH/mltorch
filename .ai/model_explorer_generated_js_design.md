# Model Explorer: generated JavaScript per operator

Status: implemented. This extends the operator detail graphs of
`model_explorer_design.md` §15 (`Me_detail`) with the JavaScript that
`Loop_js` emits for each output of a canonical Native operator. The text is
the same kernel the generated-JS executor runs (`Loop_node_executor`, used by
`loop_js_pt2 --nodes`).

## What the user sees

When a session is requested with **Generated JavaScript** on, each output
node of an operator's detail graph (`out<i>`, labelled `output <i>: t<N>`,
one per ordered output, built by `Me_detail.of_operator`) carries an
attribute `js` holding the emitted program. Model Explorer's node side panel
shows it when that node is selected. An output that cannot be lowered
carries `js_unavailable` with the reason instead, never silence. Text over
the attribute ceiling is cut at `max_attr_chars`, and the node gets
`js_truncated: true`, exactly as `body` and `body_truncated` on Kernel value
nodes (`Me_kernel`).

The detail graph is the attach point for the same reason §15 gives:
canonical Native is where operator-to-computation correspondence is
structural and exact. A Loop program is per canonical node and output, so it
inherits that correspondence and needs no PT2 provenance claim.

**Raw vs optimized is a toggle, not two attributes.** Only one `js` family
is ever attached to a node: the caller picks which optimization passes ran
(`Loop_opt.Pass.t list`, `[]` for raw, `Pass.all` for the default optimized
pipeline, or any named subset) before asking `Me_detail.generated_js` for
the text. The web UI mirrors this as two nested controls: "Generated
JavaScript" (off by default) reveals "Optimized" (on by default); unchecking
"Optimized" reveals one checkbox per pass, all unchecked (i.e. raw), letting
a reader build an arbitrary subset back up. This was a mid-implementation
correction — the first cut attached both `js` and `js_raw` to every node
unconditionally, on the theory that a second row costs nothing when the side
panel already lets either be expanded independently. Superseded once the
selectable-subset ask arrived: a bare boolean couldn't express "raw" and
"a chosen subset" as the same mechanism, and showing both textified programs
side by side stopped being the point once there could be more than two.

## Why an option, and why in the session

- **The web app never requests a detail on demand.** Every operator's detail
  graph is already in the initial session (§15, "initial and referentially
  complete"). `Coordinator.openDetail` exists, but no UI path calls it. So
  the JS has to travel inside the session to be visible in the browser at all.
- **Off by default.** It is a few KB per operator. Always on, it would press
  on the session byte and aggregate ceilings for every user and churn every
  session golden (`test/me_group*_cram.t`). As an option it changes nothing
  for a request that does not ask.
- **Budgets already cover it.** The attribute is bounded per node by
  `max_attr_chars`. The session aggregates are checked on the merged result, so
  a model whose details no longer all fit loses its later details to the
  existing `Over_limit` diagnostic (§15). No new ceiling is added.

## Shape of the change

### Options (`Me_request.Options`, `Me_export.Options`)

Both `Options.t` types (the wire-decoded request one and the export
library's own, separate record every entry point converts it into) gain
`generated_js : Loop_ir.Loop_opt.Pass.t list option`: `None` is off,
`Some []` is raw, `Some Loop_opt.Pass.all` is the optimized default, and
anything else names a custom subset. The `generatedJs` wire member (a JSON
array of pass names, or absent) defaults to `None` when absent, so older
requests keep decoding. Normalized like `stages` — deduped and put in
`Pass.all`'s pipeline order — so two requests differing only in checkbox
click order compare equal; unlike `stages`, an empty list is a legitimate
request (raw) rather than `` `Invalid_options ``.
`Me_session.Capability` gains `feature:generated_js` (timeline slot right
after `feature:expression_detail`), `Available Present` when requested,
matching `Fold`'s shape. The web app's effective-options readout renders
"generated JS off/optimized/raw/custom (...)" alongside the existing note.

### Producing the text (`Me_detail`)

The new function is `Me_detail.generated_js : ?limits:Kernel.Limits.t ->
?passes:Loop_ir.Loop_opt.pass list -> Graph_ir.graph -> Graph_ir.node ->
output:Output_ordinal.t -> Js_attr.t`, where
`type Js_attr.t = Emitted of string | Unavailable of Loop_node_program.error`.
It runs `Loop_node_program.lower` and then `Loop_js.emit`. `passes` defaults
to the full pipeline (`Loop_node_program.lower`'s own default); `limits` is
the *Kernel* budget the adapter admits against, distinct from
`Me_limits.Limits.t` (which only bounds the rendered attribute's length).
Both call sites that build an operator detail (`me_export_shape.ml`, the
initial session, and `me_export.ml`, the on-demand delta) already hold the
rebuilt canonical graph and node. When `generated_js` is `Some passes`, they
convert `passes` to `Loop_opt.pass` functions via `Loop_opt.select` and pass
the resulting `Js_attr.t list` to `Me_detail.of_operator ~generated_js:[...]`.
`of_operator` decorates the output nodes, so id derivation and ordering stay
in one place.

The payload is typed (an `Unavailable` carries the lowering error, not a
string). It is rendered to text only at the attribute, through
`Loop_node_program.pp_error`.

### Selecting which optimizations ran (`Loop_opt.Pass`)

`Loop_opt` gains a named `Pass.t` — one constructor per pipeline stage, in
the *fixed pipeline order* (the existing "not alphabetical, order is
behavior" rationale on `Loop_opt.passes` itself) — plus `Pass.all`,
`Pass.name`, `Pass.of_name`, and `select : Pass.t list -> Loop_opt.pass list`
(the named subset, still filtered to pipeline order regardless of the
argument's order or duplicates). `Loop_node_program.lower` gains `?passes`
(default: the full pipeline, so every existing caller is unaffected).
**This changes only what a reader sees.** Every executed path — the Loop
interpreter, the jsoo executor, `Kernel_eval` — still goes through
`Loop_lower.lower`'s own fixed pipeline; `Loop_opt.select`'s result never
reaches any of them.

### Library dependency

`model_explorer_export` gains `loop_ir` (native) and `loop_ir_js` (the jsoo
mirror, over `native_js`). Both are unwrapped-friendly: `loop_ir_js` carries
its own `Loop_ir` alias module (mirroring the wrapped native library's
namespace), so `me_detail.ml` writes `Loop_ir.Loop_node_program`/
`Loop_ir.Loop_js`/`Loop_ir.Loop_opt` unmodified in both builds, no shim
needed. The worker bundle grew by 672,881 bytes (+5.0%: 13,554,396 →
14,227,277) from `loop_ir` plus `js_ast` joining the closure.
`loop_js_exec`, which is jsoo-only and compiles the code, is **not** needed:
only the text is shown, and nothing is executed in the explorer.

### Surfaces

- **CLI:** `native_graph visualize --generated-js[=raw|optimized|PASS,...]`
  (bare flag, via Cmdliner's `~vopt`, means `optimized`). **Not** on
  `native_graph detail`: that command's key is always a Kernel VALUE key
  (`Detail_key.create`, never `create_operator`), so `Me_export.detail`
  always takes the `Me_detail.of_value` branch, which `generated_js` (an
  `of_operator`-only decoration) cannot reach — the operator detail graphs it
  decorates are already an eager part of the session `visualize` exports.
- **Web:** two nested controls in the request options form, next to "Fold
  constants and payloads" — see "What the user sees" above. The worker
  bridge (`webapp_bridge.ml`) reads/echoes `generatedJs` as an array of pass
  names or `null`; `web/app/presentation.js` carries it through
  controls ↔ URL (`?generatedjs=optimized|raw|PASS,...`) ↔ wire, mirroring
  `verifySymbolic`'s existing round trip exactly.

## Considered and dropped

- **A code pane in `web/app`.** Stage 0 hand-edited a session to put a
  41-line, ~2.9KB synthetic program on one Kernel-stage value node's `body`
  attribute and drove the pinned visualizer directly. Collapsed, the
  attribute is genuinely unreadable (`white-space: nowrap`, clipped at the
  panel's ~200px width against a ~13000px scroll width) — but every
  attribute row already carries its own `unfold_more` expand toggle, which
  switches to `white-space: pre-wrap` with a scrollable box, preserving
  newlines and indentation legibly (confirmed by screenshot). A custom
  `<pre>` pane would duplicate that for no readability gain, so it was
  dropped rather than built.
- **Syntax highlighting.** Not without a vendored highlighter, and not needed
  to read the code.
- **A web control for an arbitrary pass subset, beyond raw/optimized.**
  Requested and built (see "Selecting which optimizations ran" above) — kept
  here as a historical note that the first response to "consider… select a
  set of applied optimizations" was to add only the library mechanism
  (`Loop_opt.Pass`/`select`) without a UI, reasoning that a 7-checkbox
  control was disproportionate to "consider." A later message asked for the
  UI explicitly, with the layering (`Generated JavaScript` → `Optimized` →
  per-pass) left to this design to decide.

## Verification

- Unit tests (`test/model_explorer/me_detail_test.ml`) over
  `Graph_fixtures.multi_output`'s `max_pool2d_with_indices` node (its second
  output, the pooled index converted to int64, gives an unrelated-enough
  kernel from the first that the mutation proof below has two naturally
  distinct texts to compare, no fabricated `Unavailable` needed):
  - `Me_detail.generated_js` (both the default pipeline and `~passes:[]`)
    equals a direct `Loop_node_program.lower` + `Loop_js.emit`, per output.
  - Mutation: forcing `of_operator`'s per-output index lookup to always read
    index 0 makes `out0`'s and `out1`'s `js` compare equal — proven to
    actually go red (reverted afterward), for both the optimized and raw
    toggle.
  - A tight `Kernel.Limits.t` (`max_size:1`, distinct from
    `Me_limits.Limits.t`) forces a genuine `` `Adapt `` failure, rendered as
    `js_unavailable` by `of_operator` — proven red/right the same way.
  - A tight `Me_limits.Limits.t` (`max_attr_chars:16`) forces `js_truncated`
    at exactly the configured cut length.
- `test/me_generated_js_cram.t`, over the committed mobilenetv2_050
  model.json: off/default-optimized/`=raw`/a named subset (`=unit_loops`,
  chosen empirically — the first canonical operator is a Permute simple
  enough that fold/simplify/guards/cse/hoist are each individually no-ops on
  it) each produce a present, distinct `js` text; an unknown pass name is
  rejected; and the whole document is unchanged by the flag once the `js`
  attributes and the one capability row are stripped before diffing. This is
  the CLI/session integration proof, not a second proof of the JS's own
  correctness (the unit tests above own that).
- Existing `me_group*_cram.t`/`me_visualize_*_cram.t` goldens: unchanged
  other than the new `feature:generated_js` capability row every session now
  always carries (`not_requested` when off) — the same kind of golden churn
  adding `Fold`/`Expression_detail` caused historically, not a regression.
- Browser gate (Playwright, `web/test/webapp.spec.ts`): navigates straight
  to `expr/g/native/001/n0`'s `out0` via the element's own `selectNode`, and
  reads `js` off the rendered side-panel attribute table (not through our
  own validator) at four points — off (absent), default (optimized), raw,
  and a custom subset (`unit_loops` alone) — asserting presence/absence and
  that the text changes each time. `web/test/presentation-unit.test.mjs`
  covers the pure `presentation.js` round trip (controls/URL/wire) with 6
  new cases.
