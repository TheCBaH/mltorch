# Model Explorer: generated JavaScript per operator

Status: design, not yet implemented. This extends the operator detail graphs of
`model_explorer_design.md` §15 (`Me_detail`) with the JavaScript that
`Loop_js` emits for each output of a canonical Native operator. The text is
the same kernel the generated-JS executor runs (`Loop_node_executor`, used by
`loop_js_pt2 --nodes`).

## What the user sees

When a session is requested with **Generated JavaScript** on, each output
node of an operator's detail graph (`out<i>`, labelled `output <i>: t<N>`,
one per ordered output, built by `Me_detail.of_operator`) carries an
attribute `js` holding the emitted program. Model Explorer's node side panel
shows it when that node is selected. An output
that cannot be lowered carries `js_unavailable` with the reason instead,
never silence. Text over the attribute ceiling is cut at `max_attr_chars`, and
the node gets `js_truncated: true`, exactly as `body` and `body_truncated` on
Kernel value nodes (`Me_kernel`).

The detail graph is the attach point for the same reason §15 gives:
canonical Native is where operator-to-computation correspondence is
structural and exact. A Loop program is per canonical node and output, so it
inherits that correspondence and needs no PT2 provenance claim.

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

### Options (`Me_request.Options`)

`Options.t` gains `generated_js : bool`, carried on the wire like `fold`,
with default `false` when the member is absent, so older requests keep
decoding. It is part of request identity like every other option.
`Me_session.Capability` records whether the session was built with it, so
the shell can say "requested: … · generated JS on/off" alongside the
existing options note.

### Producing the text (`Me_detail`)

The new function is `Me_detail.generated_js : Graph_ir.graph -> Graph_ir.node
-> output:Output_ordinal.t -> Js_attr.t`, where
`type Js_attr.t = Emitted of string | Unavailable of Loop_node_program.error`.
It runs `Loop_node_program.lower` and then `Loop_js.emit`. Both call sites that
build an operator detail (`me_export_shape.ml`, the initial session, and
`me_export.ml`, the on-demand delta) already hold the rebuilt canonical
graph and node. When `generated_js` is set, they pass the results to
`Me_detail.of_operator ~generated_js:[...]`. `of_operator` decorates the output
nodes, so id derivation and ordering stay in one place.

The payload is typed (an `Unavailable` carries the lowering error, not a
string). It is rendered to text only at the attribute, through
`Loop_node_program.pp_error`.

### Library dependency

`model_explorer_export` gains `loop_ir`. That library already builds under
js_of_ocaml (`js/jsoo/loop_ir_js`), and its dependencies (`js_ast`,
`native`, `expr`, `core`) are already inside the export's closure except
`js_ast`. The worker bundle grows by `loop_ir` plus `js_ast`, so record the
bundle size before and after. `loop_js_exec`, which is jsoo-only and
compiles the code, is **not** needed: only the text is shown, and nothing is
executed in the explorer.

### Surfaces

- **CLI:** `native_graph visualize --generated-js` and the same flag on
  `native_graph detail`.
- **Web:** a checkbox in the request options form, next to "Fold constants and
  payloads". The worker bridge passes it through the `Options` it already
  normalizes.

## Later, not in the first cut

- **A code pane in `web/app`.** Model Explorer's side panel may flatten
  newlines or wrap a 100-column line badly. That is unverified, and the
  first browser check decides it. If it reads poorly, `panels.js` gets a
  `<pre>` pane filled with `textContent` (never `innerHTML`, as the rest of
  the panels) from the selected node's `js` attribute. The adapter's
  select-then-act event (§20.4) is the hook.
- **Syntax highlighting.** Not without a vendored highlighter, and not needed
  to read the code.
- **Optimized vs raw.** Once the Loop IR optimization passes land
  (`loop_ir_optimization_design.md`), a second attribute could hold the
  unoptimized program for comparison.

## Verification

- A cram test beside the existing detail crams: an operator detail with
  `--generated-js` holds a `js` attribute per `out<i>` node, and its text
  equals `Loop_js.emit` of `Loop_node_program.lower` for that node. Run the
  lowering and emit, and diff the two texts.
- The same cram without the flag produces a session byte-identical to
  today's. The option must be invisible when off.
- A truncation case at a small `max_attr_chars` profile: `js_truncated`
  present, and the text cut at the ceiling.
- An unavailable case: a node that does not lower carries `js_unavailable`,
  and the session still exports.
- Browser gate (Playwright): export with the checkbox on, select an `out<i>`
  node, and read the `js` attribute through the element's own node data, not
  through our validator.
- Mutation: attach the JS of output 0 to every `out<i>` node of a
  multi-output operator (`split_with_sizes`, `max_pool2d_with_indices`). The
  cram's per-output equality must go red.
