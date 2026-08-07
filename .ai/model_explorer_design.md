# Model Explorer export — design record

The durable half of the Model Explorer graph-visualization work: what is in the tree,
why it has the shape it has, and which decisions later stages may not quietly reverse.
`.ai/model_explorer_plan.md` is the research-backed high-level plan; the round-by-round
implementation plan is a gitignored working file. **This document is the source of
truth** — when a design here changes materially, it changes here in the same commit.

Sections appear as their code lands. What is not below is not yet written.

---

## 1. Layout

| Path | Dune name | Role |
|---|---|---|
| `lib/model_explorer_export/` | `model_explorer_export` | The export library. Pure, `(wrapped false)`, whole thing inside js_of_ocaml's closure |
| `test/model_explorer/` | per-suite | Expect suites, `(modes best js)` |

`(wrapped false)` because three shells — the cmdliner CLI, a node golden, a browser
bundle — refer to these modules by their `Me_*` names; the prefix *is* the namespace.
No core library depends on this one.

**Everything here runs under js_of_ocaml.** That is not a portability aspiration, it is
the point: the browser compiles this same library rather than a second implementation of
it, so a divergence between what the CLI exports and what the page renders is not
expressible. CLAUDE.md's 32-bit rule therefore applies to every module in this directory
without exception, and the suites carry `(modes best js)` for the same reason.

---

## 2. Archive limits — `Pt2_zip.Limits`

Every archive is untrusted, whether a browser or the CLI opened it. There is no
`default` profile for that reason: `untrusted` is the default and the ceiling, `trusted`
is for internal callers holding data they produced, and `--limits` may tighten and never
weaken.

**Five ordered checkpoints, and the order is the mitigation** — each runs before the step
that would make it too late:

1. **Before the OCaml string exists.** `Pt2_archive.read_file_limited`, so `open_pt2` no
   longer calls a bare `In_channel.input_all`. The requested bound is *clamped*, not
   trusted: the length is narrowed to the `int` `really_input_string` takes, and a bound
   above that domain is not a bound (`Pt2_archive.effective_max_bytes`, 512MB).
2. **Before `Zipc.of_binary_string`.** The raw EOCD member count, preflighted. A
   `Zipc.fold` count is taken *after* decoding, over a path-keyed map in which duplicates
   overwrite.
3. **Immediately after decoding.** `Zipc.member_count zip = raw_eocd_count`. This is the
   only place a duplicate path below `max_entries` is detectable.
4. **Over the central directory,** before materializing: path length, depth, `..`,
   absolute paths, control bytes, encryption; *both* `compressed_size` and
   `decompressed_size` against `max_entry_bytes`, with a running `int64` aggregate
   **checked before each addition**. `decompressed_size : int` is 32-bit under jsoo, so
   it widens first.
5. **At materialization, per method.** `Stored` requires `compressed_size =
   decompressed_size` and enforces the per-entry ceiling *before* calling zipc, which
   otherwise reaches a `String.sub` with no size comparison. `Deflate` gets a declared
   pre-check, zipc's bounded inflate, and a produced-length re-check. Any other method is
   refused.

`allow_encrypted` is a field rather than an inline constant so the rejection has a name in
the record; zipc cannot decrypt at all, so a permissive value would only defer the failure
to a worse place.

**Ceilings come from the format and the platform, never from measuring well-behaved
models.** zipc rejects ZIP64, so the classic 16-bit fields *are* the format's ceilings.
`entry_bytes` is 2^30: `Zipc.File.max_size` is `min Int.max_int 4294967295`, which is
4294967295 natively and 2147483647 under jsoo (measured: `Sys.int_size = 32`,
`max_int = 2147483647`, `Sys.max_string_length = 2147483643`), so a per-backend limit
would make the same archive open in one shell and fail in the other.

**Tests** (`test/pt2_zip_limits_test.ml`, `(modes best js)`) are hand-built archives, byte
by byte, because zipc's writer cannot produce any of the things being tested — two
members with one path, an encrypted entry, an unsupported method, headers whose declared
sizes disagree with the bytes beside them. Each case pins the **exact** error rather than
"it failed": for four of them the point is *which* checkpoint caught it, and an aggregate
detected at materialization instead of in the central directory means the ordering has
silently changed while every "rejects" assertion stayed green.

---

## 3. Ceilings and profiles — `Me_limits`

Three layers, deliberately not one.

**`Hard`** — non-negotiable ceilings, fixed from reasoning: the ZIP format, the jsoo
32-bit `int`, the `Bytesrw.Bytes.Writer` limit type (`limit : ?action -> int -> filter`,
whose bound is an `int` and therefore 32-bit under jsoo) and `Kernel.Limits.Hard`'s
precedent. Never from measuring well-behaved models.

**`Limits.t`** — a profile: one tunable ceiling per field, plus `response_live_bytes`,
which is **derived** by `create` and stored. Derived-and-stored rather than recomputed,
so the peak a profile implies is computed once, checked once, and no call site can arrive
at a different figure.

**`Wire_limits.t`** — `private Limits.t`, the subset that may cross the worker boundary.

### The bound everything answers to

`Hard.jsoo_safe_bytes = 2^30`. Under js_of_ocaml `Sys.int_size = 32` and
`max_int = 2147483647`; a live total that is ever narrowed to `int` must stay inside that
domain **with room for the arithmetic that precedes the narrowing**, so the ceiling is
half of `max_int` and the sum of any two totals bounded by it is still representable.

### Aggregates are bounded separately from fields

CLAUDE.md's rule, applied where it bites: a sum of individually-in-range factors can still
overflow, and bounding each document separately does not bound what is live at once. Four
relations between the constants, checked once at module initialisation in `int64` — a
wrapped total passes a `<=` test:

```
3 * max_request_json_bytes                        <= jsoo_safe_bytes
max_request_live_bytes                            <= jsoo_safe_bytes
max_response_live_bytes                           <= jsoo_safe_bytes
max_diagnostics * escape * (msg + id) bytes       <= max_response_meta_bytes
```

Placing them on the *constants* rather than on a request keeps them out of the hot path
while still being what makes the per-field ceilings sound: with all four true, any set of
fields the builder accepts fits, and no aggregate arithmetic runs on attacker-influenced
values at all.

`response_live_bytes` is the calculator behind two of them — the maximum of four response
phase sums (`R_decode`, `R_build`, `R_commit`, `R_install`), in checked `int64`, every
product and every addition tested **before** it is performed so an overflow is reported
rather than inspected after the fact. Its phases are built from lifetimes the code can
*establish*: the JS document string is conservatively live across the whole synchronous
call, because `prepareOpen(metaText, documentText)` holds both arguments for its
duration and a release the API cannot perform is not a release. `R_install` is the peak
and it is a **sum**, not a maximum of its two largest members — both sessions, both
processed graphs, both render states and both elements coexist there by construction.

**What is derived and what is calibrated.** The expansion factors — heap bytes per byte
of document for a decoded session, a processed graph, a parsed JS value — are frozen
*before* Stage 1, conservatively, from reasoning. They are **inputs to a hard
derivation** and therefore cannot be calibrated later: a factor that changes after
measurement changes the value it was used to derive. Measurement may only tighten a
**profile**.

Enforcement is consequently **before input acquisition**: `create` rejects a profile
whose peak does not fit, so no document is ever read under one.

### Which checks can fail, and why that was designed for

Two ceilings are deliberately *not* collapsed into each other:

- `max_response_document_bytes` (8MB) sits **below** the `max_session_bytes` /
  `max_detail_bytes` field ceilings (32MB). Collapsing them would make
  "no profile admits a document the browser would refuse" true by construction and its
  test vacuous. With the gap, `trusted` is a profile that violates it, and the check has
  a witness.
- `trusted` is **not** every field at `Hard`. The phase sums at the full document ceiling
  come to roughly 1.6GB, so an all-at-`Hard` profile is not constructible — which is the
  ceilings doing their job — and the two document fields sit at half of theirs. That
  leaves `create`'s peak rejection reachable *from above*, through the public API.

`Limits.within_hard_response` holds the relations a profile must satisfy before the
browser may read a document produced under it. It is **not** part of `create`: a profile
is valid without it (`trusted` is), and folding it in would delete the only profiles able
to show the check is live. Its third relation — derived peak at or below
`Hard.max_response_live_bytes` — is *implied* by the first two under monotonicity and is
documented as such rather than presented as independent; it is retained as the direct
statement of what the other two are a means to, and becomes live if a phase ever gains a
term the document ceilings do not drive. The equality it rests on is pinned by a test.

### One field table, two traversals

`create` (ceiling = the `Hard` figures) and `Wire_limits.of_limits` (ceiling = another
profile) run the **same traversal over the same field list**, reading through a `get`
function rather than by name. A field added to the record and not to the list is a field
neither of them checks, and the list is the only place that can happen. Widths unify at
`int64` — `Int64.of_int` is total, and comparing there means the 32-bit backend never
narrows a ceiling before testing it. The nested `Pt2_zip.Limits.t` is traversed the same
way, against `Pt2_zip.Limits.trusted`, so there is no second copy of that module's
ceilings here.

### Which profiles cross the wire

`untrusted` is the default **and the ceiling** — not `Hard`. A file input is untrusted and
a caller may tighten `untrusted`, never weaken it, and a value below `Hard` can be far
looser than `untrusted`; since the request *is* the wire, that check cannot be delegated
to the page that built it. `small` is fieldwise no looser and is wire-selectable;
`large` and `trusted` are programmatic-only and `of_limits` rejects them. That is what
makes `--limits untrusted|small` a guarantee rather than a UI convention.

`Wire_limits.t` being `private Limits.t` behind that check is the point: a type nobody can
inhabit except through it makes "the worker and the page disagree about the profile"
unconstructable, rather than something rejected somewhere downstream.

**The wire codec is not here.** `of_json` and `jsont` land with the rest of the worker
protocol, whose shape freezes at Stage 2 and whose fixtures live beside it. What this
module owns now is the domain closure — the property that codec will encode.

### Request identity is protocol, not policy

`max_epoch_bytes = 36` (a UUID) and `max_request_id_bytes = 47` (`36 + 1 + 10`, the widest
`<epoch>-<seq>` the grammar can produce) are `Hard` constants and **not** profile fields.
As profile fields they were two authorities for one number: a tightened profile could
make the production builder issue an identity its own worker rejects, and they were
checked before the profile they supposedly belonged to had been decoded. They are
consequences of a grammar that `Request_id.of_string` checks directly, and they exist here
for the JS bridge and for readers rather than as a second, weaker rule. For the same
reason they are not on the calibration table: a constant nobody may tune is not a default
anybody can measure.

`max_seq_exclusive` is `int64` and exclusive. As `int (* 2^32 *)` it would not be a
representable value under jsoo at all — the literal truncates before it is ever compared
against, and the exported JS constant would then be a different number from the one the
OCaml grammar enforces.

### Open obligations

- `Hard.max_request_stages` states the cardinality of `Me_session.Capability.graph_stage`,
  which does not exist yet. When it does, its test asserts the two agree.
- The `Bytesrw` writer's growth behaviour is currently given a factor-of-two allowance in
  `P_encode`. Confirm and record the actual behaviour when the encoder lands.
- Every profile figure is provisional. The release profile is calibrated after Stages 2–4,
  using temporary instrumentation rather than a second exporter.

### Diagnostics — `Me_limits.Diagnostic`

The one type that crosses *every* boundary in this design — session, worker response,
delta — and therefore the one place an unbounded exception string could enter the wire.
Two consequences, both structural.

**The code vocabulary is closed.** `Code.t` is a variant with a wire tag, so the browser
side is a `switch` rather than a string match, and `of_string` is total in its failure so
an unknown tag from a future producer decodes as a named error instead of being silently
dropped. Two pairs in it are separated deliberately: `Malformed_request` versus
`Malformed_response` are **directional** — bytes the worker received versus bytes the
coordinator received — and reporting both as one turns a worker defect and a page defect
into the same telemetry; `Request_in_flight` is separate from `Malformed_request` because
a second concurrent submission is well-formed and refused by an invariant, not malformed.
`Inconsistent_mount` is separate from `Internal` because it is the one condition in which
the page and the bridge may disagree and neither can repair it.

`Code.all` is built by walking a successor chain rather than written out beside the type.
A hand-written list compiles perfectly while `of_string` quietly stops recognising a newly
added tag — which, for a closed vocabulary meant to be switched on, turns a named condition
into an unknown one. The chain makes the compiler demand an arm.

**`create` is total.** No result, so there is no path by which a diagnostic *about* a
rejection becomes itself a rejection. Both variable-length fields are bounded there:

- `message` is **sanitised then truncated**. Sanitisation is replacement, not only
  truncation, and the reason is a property of the dependency: `Jsont_bytesrw`'s encoder
  **does not validate UTF-8** — it emits the bytes it is given, producing a document that
  is not JSON. Nothing in OCaml on that path notices; the first thing to notice is the
  browser's parser, which turns a diagnostic about some other condition into a malformed
  response. That behaviour is pinned by a test rather than assumed, so a jsont upgrade that
  starts validating fails the suite instead of silently invalidating the argument.
  Replacement can *widen* — one stray byte becomes three — so the ceiling is measured
  against what is about to be written, not against what was read, and the cut lands on a
  scalar-value boundary.
- `graph` is checked against `max_id_bytes` and **dropped, never truncated**, and likewise
  dropped when it is not clean UTF-8. Truncating or sanitising an identifier yields a
  *different* identifier, which either resolves to the wrong graph or fails to resolve
  while looking authoritative; a missing graph is honestly missing. The drop sets
  `truncated` too, so it stays visible.

`of_exn` is the one bridge from an escaping exception and is deliberately lossy:
`Code.Internal` plus `Printexc.to_string` through `create`'s sanitiser. Nothing else may
put an exception's text on the wire.

`jsont` decodes through the **same constructor as the producer**, under `Limits.trusted` —
hard invariants only, since a `Jsont.t` carries no profile and the profile-level bound
belongs to whoever holds one. A `truncated` the wire claimed is kept beside what
re-sanitising found: a producer that already cut something knows that and this decoder
cannot.

---

## 4. Identifiers — `Me_ids`

Model Explorer splits `namespace` on `/`, honours `\/`, and treats `;` specially, while
group labels are arbitrary PT2 strings. So there is **one** encoder, applied to **dynamic
components only** — a group label, a model name — and never to the structural separators
of an assembled id.

**Injectivity is the property, and it is a one-pass property.** `component` escapes `%`
*alongside* `/`, `;`, `\`, `#` and `:` rather than after them. A two-pass scheme that
escaped `%` last maps the label `a/b` and the literal label `a%2Fb` to the same id, while
every individual rendering still looks correct — which is why the suite drives a corpus
chosen for near-collisions and asserts the distinctness of the outputs rather than reading
them. Removing `%` from the escaped set collapses 17 inputs to 12 outputs; that red was
observed before the guard was restored.

**Control bytes are rejected, not encoded.** They survive percent-encoding into an id the
renderer displays verbatim, and there is no correct rendering of them. Consequently
`component` returns a `Core.result`: the plan sketched `string -> string`, and a signature
that cannot express its own rejection is not a contract.

**Two ceilings, on two sides of the encoding.** `max_label_bytes` bounds the raw input,
because that is the figure a caller can reason about; `max_id_bytes` and
`max_namespace_component_bytes` bound the *result*, because that is what the renderer sees
and encoding can expand a component threefold. They are separate checks and a label inside
the first can still produce an id past the second.

**`Layer` is the dialect, and is not `graph_stage`.** It names which of `Pt2`, `Native`,
`Native4d`, `Symbolic` or `Kernel` an id belongs to, and is owned by the flow spine — which
is why the counters behind `g/<layer>/<NNN>` are per-layer and dense: a `Pass.Exec_id.t` is
dense per dialect, so `t/native/007` and `t/native4d/000` may carry equal executions and
still not collide. `Me_session.Capability.graph_stage` is a *different* vocabulary naming
which exported graph a capability is about; initial and canonical Native are two stages of
one layer, and fusion is a stage of the Kernel layer. Collapsing the two would leave
`g/<layer>/<NNN>` unable to name the two Native graphs that share a layer. Both use the
same successor-chain `all` as `Diagnostic.Code`.

**A detail id does not re-encode its parents.** They are ids this module already produced,
and encoding an encoded id is not idempotent — the `%` of an existing escape escapes again
— so the result would no longer name the graph it is derived from.

Stable node ids plus `MATCH_NODE_ID` fallback cover untouched nodes and explicit mapping
entries cover changed ones. **That is sound only because of the id-identity rule** — a
changed value always means a new id (`native_transform_design.md` §4) — and the comparison
view rests on it entirely.

---

## 5. The flow spine — `Me_flow`

`Pass.Exec_id.t` names a **transition**, so it cannot name the `N + 1` states that `N`
changed steps produce, and it does not exist at all for an import, a cross-dialect
conversion, an adaptation, or a run in which no pass changed anything. The spine is
therefore its own structure rather than a projection of execution identities — which is
also what keeps the zero-changed-pass run showing a Native state.

**It is a rooted DAG, not a chain.** The dataflow branches at canonical Native — to
Native4D by conversion, and to the symbolic stages by adaptation — so "states =
transitions + 1" describes nothing here, and `legal_triples` names each crossing
individually rather than deriving "the next layer" from a rule that cannot express a
branch.

**`Pass_execution` qualifies an execution by its dialect.** `Pass.Exec_id.t`'s ordinal is
dense *per specialization*, so a Native and a Native4D execution are routinely equal —
`fold #0` in both. Only the `(layer, exec)` pair is unique session-wide, and the
"every execution occurs at most once" rule is meaningless without it. `Pass` itself does
not depend on `Me_ids.Layer`: the exporter adds the layer while consuming each branch's
trace, which is the only place that knows which branch it is reading.

**Bipartite, because an edge cannot carry navigation.** Model Explorer has no edge event
and `subgraphIds` is a node field, so the rendered spine has one node per state *and one
per transition*, with edges `state → transition → state`. Selecting a transition node is
what opens its comparison.

**What `validate` covers, and what it deliberately does not.** Unique ids; exactly one
`Pt2` root; endpoints resolve; reachability; at most one producer per state agreeing with
`produced_by`; only legal layer triples; for `Pass e`, `e.layer` equal to the transition's
layer *and both endpoints'*; every `Pass_execution.t` at most once. `max_states` and
`max_transitions` are checked **first**, before the walks that are linear in them — a
bound checked after the work it bounds is not a bound. `Transition.comparison` is *not*
checked here: it resolves against a table this module cannot see, so `Me_session` does it,
with the pane-graph equality that makes it meaningful.

**Acyclicity is implied, and the code says so.** One root, one producer per state and
reachability make the reachable subgraph a tree, so a cycle can only sit in an unreachable
component and is reported as `Unreachable_state`. The walk's grey/black colouring stays as
a **termination guard** rather than a validation rule: depending on an earlier check for
termination is fragile in the one direction that matters, since reordering it turns a
defect into a hung process instead of a wrong answer. `Cycle` is the name that guard
reports under, and the suite records the reasoning rather than pretending the rule has a
witness.

---

## 6. The session document — `Me_session`

**Deterministic interchange.** No wall time, no nonce, no host state: two loads of the
same bytes with the same provenance and options produce identical JSON, natively and under
node. That is why `epoch` lives in `Runtime` and not in `Session` — a field cannot be
forgotten out of a document it was never in — and the suite asserts two runtimes with
different epochs encode identically.

**Provenance is part of the statement, not noise.** `source_sha256` is populated *only*
when the caller supplied an expected digest: the catalog path, where it was computed at
build time and the browser verified it. A CLI opening an arbitrary local file has nothing
to verify against and records `None`. There is no SHA-256 in this repository or its switch
and `Digest` is MD5; this rule is why none is needed, and both shells agreeing on `None`
for local input is what keeps the parity gate unaffected.

**Import is out of scope for v1.** No CLI or browser import mode; `Session.jsont`'s decode
half exists for controlled round-trip and conformance tests only, and the signature says
so — a decoder that exists reads as a decoder that is trusted, and this one is not.

### The capability table is the specification

`compatible` is one exhaustive match, so a key added to the type stops it compiling. Each
key admits exactly one payload shape, and the suite enumerates the whole key × payload
grid rather than sampling it: a table that admitted everything would show as a row of
all-yes.

`Loop_ir` and `Codegen` are permanently `Unavailable Not_implemented` — **not**
`Not_requested`, which would read as "you did not ask" for something that cannot be asked
for, and not `Unavailable` with any other reason either.

**Verification splits from `Pass_audits`** because `Native_interp.transformed.composed`
survives per-pass audit truncation. Mapping that truncation to one
`Verification → Unavailable Over_limit` hides a result that is still available, while
`Available` would conceal that the detail is incomplete. Two keys say both things. Units
differ and are stated: `Verification_summary` counts verifier **clusters**;
`Pass_audit_status` counts audit **reports** in its scalars and clusters in
`omitted_counts`. Both count payloads encode *and decode* through
`Pass.Outcome_counts.jsont`, so a malformed binding cannot become an unchecked map.

Every key occurs exactly once and `all_keys` is complete, so **absence is a producer
defect rather than an ambiguous silence**. `all_keys` comes off a successor chain for the
reason `Diagnostic.Code.all` does.

### What `validate` resolves

Unique graph ids; every `View.graph`, `Pane_state.graph`, node-data graph and node key, and
`subGraphIds` entry; `default_view`. Every `IncomingEdge.sourceNodeId` exists **and its
`sourceNodeOutputId` names a slot the node actually has** — resolving the node is not
enough, since an edge pointing at output 3 of a single-output node renders as a connection
carrying nothing.

**The mapping rule is scoped to the comparison, and that scope is load-bearing.** Within
each `Comparison.sync` a node id appears in at most one `Mapping_entry` per side — but
*not* globally, because canonical Native is the left pane of both comparisons hanging off
that state, so its nodes legitimately appear on the left of two mapping sets. A global
check would reject the exact branch the flow spine exists to show.

**A transition's comparison must be over its own two graphs.** Resolving is not enough: a
comparison naming two unrelated graphs resolves perfectly and then shows the wrong diff,
which is a rendering nobody can tell is wrong. `Me_session` performs this check because
`Me_flow` cannot see the comparison table.

Aggregates are checked **before** the walks that are linear in them.

### The flow is data on the wire

`Session.jsont` carries the spine, not only the bipartite graph that renders it:
`flow_nav.js` maps a selected node id → `Transition.id` → its `Comparison`, and
`Transition.comparison` exists nowhere else. Encoding the graph alone would leave every
transition node clickable-looking and resolving to nothing.

`Exec_id.index` is `int64_as_string`, as are every byte count and report count in the
document. The adaptive `Jsont.int64` emits a JSON number, which `JSON.parse` rounds past
2^53 — and the ordinal is `int64` precisely because it may exceed what a jsoo `int` holds.

Tagged unions rather than sets of optional members, for `Capability.payload`,
`Capability.status` and `Transition.kind`: a payload that names no field, or a `Pass` that
carries no execution, is then unrepresentable on the wire rather than rejected after the
fact.

---

## 7. PT2 → Native navigation — `Me_pt2`

Built from `Pt2_native_graph.node_origins` **only**, and root-scoped: the lowerer is
root-only and writes `graph_path = []`, so a non-root origin names a node in a nested graph
with no native counterpart, and pairing one would send the right pane somewhere unrelated.
Nested graphs stay source views; tensor origins stay metadata.

**Entries are connected components, not pairs.** The sidecar is many-to-many in both
directions — the lowerer decomposes, relayout folds — so a pairwise encoding puts one node
in several entries, which `sync_navigation.ts` requires never happens. Components are
disjoint *by construction*, which discharges the requirement rather than checking it. The
case that makes this concrete is a decomposition and a fold meeting at one node: PT2 #0
lowers to `n0` and `n1`, and `n1` also covers PT2 #1, so all four ids are one component.
Emitted pairwise, each pair reads as perfectly correct while `p0` and `n1` each appear
twice. The suite asserts disjointness over the output rather than arguing it from the
algorithm.

**`deleted` needs the source universe, so it is a parameter.** The sidecar is indexed by
*native* node, so a PT2 node with no counterpart never appears in it and a `deleted`
derived from it alone could only ever be empty. A field that cannot be populated is not a
field. `created` and `deleted` both come off the same relation the entries do, so a node
cannot be in a component and reported created.

**Determinism is load-bearing.** The union-find takes the lexicographically smaller root as
representative, components are ordered by their native side and ids are sorted within each
side, so the output does not depend on insertion order — which the session's "two loads
produce identical JSON" claim quantifies over, and which a representative chosen by
insertion order would break while every entry stayed correct.

---

## 8. Stage 0A — the contract spike, and its answers

A hand-authored fixture driving the **real pinned element**
(`ai-edge-model-explorer-visualizer@0.1.2`) under Chromium. It uses no exporter — there is
none yet, and the questions are about the renderer rather than about our projection of a
model — but it builds its `GraphCollection` through the real `Model_explorer` binding, so
the JSON it feeds the element is the JSON the export path will emit rather than a second
description of the same schema.

`make spike.setup` once, then `make spike.runtest`. The spec **records** answers and fails
only if the driver could not run: a gate that fails when the answer is inconvenient
teaches nothing, and the design's job is to react to the answer.

### The answers

| Question | Answer | What it settles |
|---|---|---|
| Can an element be built **off-DOM** and swapped in? | **Yes** — properties set while detached, `modelGraphProcessed` fires after `appendChild` | §6.1.1 keeps its off-DOM staging. The specified snapshot/restore fallback is **not** needed |
| Does `modelGraphProcessed` fire reliably, and can it be bounded? | **Yes** to both — a `setTimeout` race around a `{once: true}` listener settles cleanly | The install transaction may await it with a timeout rather than unconditionally |
| Is `config` **assignable on a live element**? | **No** — the assignment does not take | Replacement requires a **new element**, so two elements and two processed graphs coexist during an install. This is what makes `R_install` a sum rather than a maximum |
| Install a **new graph plus a parent `subgraphIds` link after** `modelGraphProcessed`, then enter it? | **Yes** | The expression-detail delta path is viable: a graph nobody has seen can be linked from a node already on screen |
| `subgraphIds` navigation | **Yes** — `selectNode(id, graphId, collectionLabel)` navigates | |
| Node data through **both** provider entry points | **Yes** — `addNodeDataProviderData` (graph-id keyed) and `addNodeDataProviderDataWithGraphIndex` | Session v1's graph-addressed `Node_data_set` maps to either |
| `uiStateChanged` → `initialUiState` round trip | **Yes** — the emitted state is accepted by a fresh element | "Back to flow" can restore the pane the user left |

### The finding the spike existed to catch

**`window.modelExplorer` must be configured *after* the bundle loads, not before.**
The bundle assigns its own `window.modelExplorer` over whatever is there, so the natural
order — set the global, then load the script — leaves the object empty and every asset
falling back to `static_files/*` and `worker.js` at the **server root**. Under a deployment
with a path prefix that is a page that renders nothing, with nine 404s and no error that
names the cause.

The page therefore mutates the object the bundle installed:

```html
<script src="vendor/main_browser.js"></script>
<script>
  window.modelExplorer.assetFilesBaseUrl = 'vendor/static_files';
  window.modelExplorer.workerScriptPath  = 'vendor/worker.js';
</script>
```

Mutation rather than replacement, so nothing that captured a reference keeps the old
values. Both paths resolve **relative to the page**, which is what makes base-URL
resolution under a path prefix a real question rather than a formality. The
before-the-bundle assignment is kept in the fixture as the probe that proves the ordering
rule, rather than deleted once the rule was known.

The exact asset set, pinned: `static_files/{FontBold,FontMedium,FontRegular}.{json,png}`,
`static_files/icons_20240521.{json,png}`, `static_files/styles.css`, and `worker.js`.

---

## 9. The projection — `Me_build` / `Me_native`

**Functorised over the dialect side.** Native4D is projected through the same body with a
different op type; the alternative is two projections that have to be kept saying the same
thing, which is the drift the `.ai/` record exists to prevent. `Me_native` is the single
Native instantiation, so the program holds one Native projection rather than a functor
application per call site.

**Bounded rendering stops the printer.** `bounded ~max pp v` installs an output function
that refuses the write crossing the cap and unwinds, so a deeply nested op stops being
printed at the ceiling. `Format.asprintf` followed by `String.sub` has already paid for
everything it then discards, and for the values this bounds that is the whole cost. The
suite proves the difference with an instrumented printer: at a 16-byte cap it reaches 30 of
1000 items, not 1000.

**Namespaces come off the `Group` tree**, which is the authoritative structural hierarchy,
and each level goes through `Me_ids.namespace_component`. The renderer splits `namespace`
on `/`, so an unencoded group label containing a slash would put a node one level deeper
than the hierarchy says — a defect invisible in any rendering that happens not to use such
a label, and one the suite covers directly. The root group contributes no level: the extra
level would name the graph, which the graph id already does.

**Boundary nodes are ordinary nodes.** Graph inputs, constants and outputs become nodes
with pinned ids (`in:t<k>` / `const:t<k>` / `out:t<k>`), so an edge from a graph input is
an ordinary edge rather than a case every consumer must special-case, and a comparison
matches them by id like anything else.

A tensor with no producer is **named**, not dropped: `Unknown_producer` carries the id.
Node and edge ceilings are checked before the walks linear in them.

**The two halves are tested together.** `Me_build`'s output is fed through
`Me_session.validate` in the same fixture, because neither test alone would show a
disagreement — the projection would happily emit an output-slot index the session validator
rejects.

---

## 10. Partiality versus defect — `Me_classify`

One distinction, drawn twice: between a graph a partial conversion does not **claim** to
handle, and a **defect**. A partiality becomes a capability the session reports unavailable
with a reason the user can act on; a defect becomes a session error and is never downgraded.
Reporting an internal invariant failure as "outside the dialect" tells the user to change
their model to work around our bug.

Both classifiers are **total matches** over the upstream error rows — no wildcard — so a row
added upstream stops this compiling instead of falling into whichever answer was written
last. The suite enumerates every row rather than sampling, because the matrix and the code
drifting apart is exactly what happened once before.

**Native4D.** `Missing_constant_payload` → `Requires_payloads`. The ten domain rejections →
`Outside_dialect_domain`. `Bad_constant_payload`, `Map` and `View` → **fatal**: a payload
that *was* supplied and is wrong, or a map/view invariant failure, is ours.

Having payloads removes **exactly one** failure mode. It does not put a graph inside the
dialect — which is why Native4D is conditional for a `.pt2` and a bare `model.json` alike,
and why the suite covers both a payload case and a domain case.

**Kernel.** `Passthrough_output` → `Unsupported_graph_shape`, **recoverable**: a graph input
used directly as a graph output has no stage to name, and the kernel suite builds one its
own comment calls legal. `Kernel`'s limit rows → `Over_limit`, since a real model can be too
big and that is a bound doing its job. Everything else is fatal: the stage program is
repository-generated, so a structural failure in it is ours, and the two selection rows are
reachable only through `?select`, which whole-program export never passes.

**Prerequisite propagation.** With no Native graph every dependent key becomes
`Unavailable Prerequisite_unavailable` — except where it is already `Not_requested`, which
**takes precedence**, since a feature nobody asked for was not blocked by anything and
reporting it as blocked invents a dependency the caller never exercised.

`Feature Flow` is in the dependent set: with no Native state the spine holds only
`s/pt2/000` and no transitions, and a one-node flow graph asserts a navigability that does
not exist. `Source`, `Expression_detail`, `Loop_ir` and `Codegen` are not.

The suite pins the **complete vector**, not the failing key — a test checking only
`Initial_native` would let every downstream row drift — and starts from a vector the
capability table accepts, so it cannot show `propagate` carrying forward something
`Session.validate` would reject.

---

## 11. Testing posture

Two commands catch different things, and both are required (CLAUDE.md): `make
jsoo.runtest` diffs the backends and so sees only divergence; `make jsoo.inline-runtest`
runs the expect suites under node against their committed goldens and so sees a wrong
answer both backends agree on. Suites in this area carry `(modes best js)` and share one
golden, so a figure that differed per backend fails the build rather than being averaged
over.

**A check that has never failed is not evidence.** Where a test asserts that a guard
rejects something, the guard is neutered once and the test watched to go red before being
restored. Two guards in `Me_limits` have that history: the nested-`zip` traversal in the
fieldwise comparison, and `create`'s peak rejection. Where a relation *cannot* fail, it
says so in the code rather than being presented as a check.
