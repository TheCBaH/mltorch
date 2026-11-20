# Model Explorer: generated JavaScript per operator

Status: **superseded by the lowering inspector** (retired 2026-11-20). The
session option `generatedJs` / `Me_request.Options.generated_js`, the attribute
producer `Me_detail.generated_js` (and `Js_attr`), the `feature:generated_js`
capability row, the `native_graph visualize --generated-js` flag, the web
controls and the `?generatedjs=` URL key no longer exist. The text that used to
travel inside the session as `js` / `js_truncated` / `js_unavailable` attributes
on an operator detail's `out<i>` nodes is now requested per node, on demand, by
the lowering pane (worker `lowering_worker`, bridge `mltorch.lowering`), which
also covers Loop, C, WebAssembly and SSA and a side-by-side comparison. The
browser gate for generated JavaScript is the pane's Playwright coverage in
`web/test/webapp.spec.ts`; the exact-equality proof (the text is
`Loop_js.emit` of `Loop_node_program.lower`) lives with `Lowering_view`'s tests.
`model_explorer_design.md` §15 keeps a pointer.

What this record keeps is what is still true: why the text is built from the
canonical node (below), the `Loop_opt.Pass` naming mechanism the inspector's
variant keys reuse, what was considered and dropped, and the CLI section.

## Why it was a session option, and why that was retired

The web app never requested a detail on demand, so the JavaScript had to travel
inside the initial session to be visible; it was off by default because it cost
a few KB per operator and would have churned every session golden. Both reasons
disappeared once a per-node request path (the lowering worker) existed: the text
no longer presses on session ceilings, and the capability row had no meaning
beyond "the request asked for it".

The attach point was always the canonical Native operator, where
operator-to-computation correspondence is structural and exact; a Loop program
is per canonical node and output, so it needs no PT2 provenance claim. That
rationale carries over to the inspector.

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

## Considered and dropped

- **A code pane in `web/app`.** (Dropped for the side-panel attribute; the lowering
  inspector later built one for a different reason: per-node, per-family text, with
  compare.) Stage 0 hand-edited a session to put a
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

## Inspecting a node's lowering from the command line

The same per-node lowerings are available without a session: `native_graph lower
--model M --node nN --artifact family[:variant]` prints one family's text (Loop, JS,
C, WAT, SSA) for one node of the canonical bundle, `--compare SPEC` prints a unified
diff of two, and `--list` prints the node's invocation with each family's default
(emits, or the typed refusal). Spellings are `Lowering_key`'s; the node id is spelled as
the canonical view prints it. The command grants no node (the canonical bundle is built
granting none), exits 1 on a refusal, and is covered by
`test/native_graph_lower_cram.t`. The inspector's own design is the lowering-inspector
doc in this directory.
