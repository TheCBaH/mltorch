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
   above that domain is not a bound (`Pt2_archive.effective_max_bytes`, 0x7000_0000 /
   ~1.75GB — bounded by jsoo's `Sys.max_string_length`, like `entry_bytes` below, not by
   any catalog model; an earlier 512MB version *was* picked by measuring models, wrongly,
   and broke `make inference` on vit_l_16/vit_l_32 (~1.15GB each) once nothing upstream of
   it was failing first. Unlike `entry_bytes`, this ceiling doesn't need to halve the
   domain for "an entry and a copy" — `Zipc.of_binary_string` keeps a reference to the
   whole string rather than copying it; only the later per-entry `String.sub` copies, and
   that is what `entry_bytes` already bounds).
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
and it is a **sum**, not a maximum of its largest members — the sessions, the processed
graphs, the render states and the elements coexist there by construction.

**How many elements coexist, and why it is not two.** It was two while a replaced
visualizer could be removed as soon as its successor was ready. It cannot be: the pinned
custom element has no abort or dispose API, and disconnecting one that is still processing
destroys a live Angular component mid-flight (NG0953). A cancelled or timed-out install is
therefore *quarantined* — connected, hidden, stripped of authority — until its expected
`modelGraphProcessed` proves removal is safe. `Hard.max_quarantined_elements` (3) bounds
how many may be awaiting that proof, and `R_install` budgets `2 + that`, one more than the
reachable population of `1 + that`: a candidate is admitted only while fewer than the
ceiling are quarantined, so a current, an active and a full quarantine cannot coexist. The
same deliberate over-reservation the queued-buffer terms make, for the same reason.

Three is the largest ceiling the peak admits — at four, `trusted` reaches 1 126 170 624
and stops being constructible — and `test/model_explorer/me_limits_test.ml` recomputes
that from the coefficients so raising the constant fails there, with the reason, rather
than failing as a library that will not load.

**The JavaScript-held terms are budgeted at `min doc max_response_document_bytes`, not at
`doc`.** No document above that ceiling reaches the browser: `within_hard_response` decides
which profiles are wire-selectable and the page rejects an over-size payload before
decoding it. `trusted` and `large` sit above it precisely because they are native-only, so
budgeting their JS-side objects at their own 16MB ceiling reserves for objects that cannot
exist. That was harmless at coefficient 2 and is decisive at five elements: without the
clamp those two profiles would exceed `jsoo_safe_bytes` and the module would raise at
initialisation.

The ceiling is exported to the page as `mltorch.hard.maxQuarantinedElements`. The renderer
takes it as a separate, mandatory input from any tighter policy value, so a test may narrow
it and no caller can widen it past the figure this calculator budgeted against.

**What is derived and what is calibrated.** The expansion factors — heap bytes per byte
of document for a decoded session, a processed graph, a parsed JS value — are frozen
*before* Stage 1, conservatively, from reasoning. They are **inputs to a hard
derivation** and therefore cannot be calibrated later: a factor that changes after
measurement changes the value it was used to derive. Measurement may only tighten a
**profile**.

Enforcement is consequently **before input acquisition**: `create` rejects a profile
whose peak does not fit, so no document is ever read under one.

### One over-limit row, owned by `Me_limits`

Nine validators count against these ceilings — `Me_build`, `Me_kernel`, `Me_source`,
`Me_verify`, `Me_detail`, `Me_pt2`, `Me_fusion`, `Me_flow`, `Me_session`. Each used to
declare its own `` `Over_limit of string * int `` and print it its own way, which made the
aggregate name a stringly-typed enum with nine independent spellings, and made "which
validator rejected" a fact that existed only inside a format string.

Both halves now live in `Me_limits` and are closed vocabularies:

- **`Me_limits.Field.t`** — the aggregate, in the renderer's own spelling
  (`Field.to_string Node_data_results = "nodeDataResults"`). This is the **single
  authority** for those wire names; nothing else may spell one. `Field.all` is built by
  walking a successor chain, as `Me_ids.Layer.all` is, so a constructor cannot be added
  without reaching the list.
- **`Me_limits.Scope.t`** — which validator was counting. Carried in the payload rather
  than supplied by the printer, because three different scopes count `Nodes` and a
  caller reading only the rendered string cannot tell them apart.

`Me_limits.over_limit_error` is **flat-included** by all nine — it is a shared base
domain, not a crossed seam, so there is nothing to wrap and `scope` already records where
the count came from. It is narrower than `Me_limits.error` for the same reason
`live_error` is: a module that only counts has no limit *field* to reject.

`` `Over_limit_64 `` is gone. `Over_limit.count` is `int64` whichever width the field has —
`Invalid.t`'s decision, for `Invalid.t`'s reason, so the two `int64` aggregates
(`max_total_nodes`, `max_total_edges`) need no tag of their own.

`Me_limits.check` / `check64` are the only place the `>` comparison is written. Nine
copies of it used to agree about everything and could stop.

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

### One field table, three traversals

`create` (ceiling = the `Hard` figures) and `Wire_limits.of_limits` (ceiling = another
profile) run the **same traversal over the same field list**, reading through a `get`
function rather than by name. A field added to the record and not to the list is a field
neither of them checks, and the list is the only place that can happen. Widths unify at
`int64` — `Int64.of_int` is total, and comparing there means the 32-bit backend never
narrows a ceiling before testing it. The nested `Pt2_zip.Limits.t` is traversed the same
way, against `Pt2_zip.Limits.trusted`, so there is no second copy of that module's
ceilings here.

The **third** traversal is the wire codec, and it is why the record also carries an
`assign`. Decoding a profile means *writing* fields, and writing them through the same
table that reads them is what keeps "a field the checks do not see" impossible in both
directions.

**`assign` narrows, so it is unreachable outside `assign_checked`.** An `int` field's wire
value arrives as `int64`, and `Int64.to_int` under js_of_ocaml truncates to 32 bits — so a
member of `4294967297` becomes `1`, and a check on that is a check on a wrapped value,
which CLAUDE.md says is not a bound. `assign_checked` tests against the ceiling profile
first; a ceiling is itself a validated profile, so bounding against it bounds against
`Hard` too.

That guard's red was **observed, and only on one backend**: with the bound removed,
`dune runtest` stays green — natively the value does not wrap, so `create`'s own hard check
rejects it with the same message — while `dune build @runtest-js` reports `accepted`. It is
the exact split CLAUDE.md names, and here only the node expect run can see it.

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

**The wire form is a tightening, not a profile.** `Wire_limits.jsont` emits a flat object of
the fields this profile tightens `untrusted` by, and nothing else — so `untrusted` encodes
as `{}`, and two equal profiles encode to equal bytes, which is what the session's
determinism claim quantifies over. Keys are the field's own **diagnostic** name, dotted for
a nested one (`zip.max_entries`), so the member a rejection names and the member that
carried it are literally the same string rather than two tables that agree today. Values
are JSON *strings* through `int64_as_string`: two fields hold values past 2^31, and a JSON
number would be read back through a 32-bit `int` under node. An unknown member is
**rejected by name**, never ignored — a profile the worker silently did not apply is the
disagreement this type exists to prevent — and decoding finishes through `of_limits`, so
every decodable value is one the encoder could have produced.

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
`component` returns a `Err.t`: the plan sketched `string -> string`, and a signature
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

**The source graph's ids share the Native graph's prefixes on purpose.** `pt2_boundary`
yields `in:`/`const:`/`out:` over an *encoded SSA name* where `boundary` yields the same
prefixes over a tensor id; the two graphs are different, and a reader who learns one grammar
has learnt both. `pt2_namespace` checks each level against `max_namespace_component_bytes`
and the joined string against `max_id_bytes`, for the same two-sided reason as above.

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

`legal_triples` ranges over `Transition.Kind_tag.t` — the kind *without* the execution a
`Pass` carries — and not over kind names. The table used to match string literals, so a
typo in one made a legal transition silently illegal, in a table whose entire job is to
say which are legal. `Kind_tag` is also the single authority for the wire spelling
(`Transition.kind_name` is `Kind_tag.to_string` of `Transition.tag`), so the encoder and
the table cannot drift apart.

**The rejection carries the triple, not the transition id.** `` `Illegal_transition ``
holds `(transition, before, kind, after)` and `` `Pass_layer_disagrees `` holds the
execution's layer beside *both* endpoints', because the check is that the execution
equals both and one endpoint would not say which equality failed. With the id alone,
four structurally different rejections rendered as one identical sentence and a reader
had to go back to the flow to learn anything — see the four cases in
`test/model_explorer/me_flow_test.ml`, which now print four distinct triples.
`` `Duplicate_pass_execution `` carries the `Pass_execution.t`; the table that detects it
is still keyed by the rendering, which `Pass_execution.pp` makes injective over the pair.

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

A hand-authored fixture driving the **real element** — built from the submodule, not
installed from npm; see §17 — under Chromium. It uses no exporter — there is
none yet, and the questions are about the renderer rather than about our projection of a
model — but it builds its `GraphCollection` through the real `Model_explorer` binding, so
the JSON it feeds the element is the JSON the export path will emit rather than a second
description of the same schema.

`make spike.setup` once, then `make spike.runtest`. The spec **records** answers and fails
only if the driver could not run: a gate that fails when the answer is inconvenient
teaches nothing, and the design's job is to react to the answer.

The Playwright browser path is `web/.playwright-browsers`, explicitly passed to both
commands. In CI both setup and the gate run in the devcontainer as `vscode`; a browser
installed in the runner user's default cache is invisible there. Keeping it in the
workspace also gives the Actions cache and the container the same concrete directory.

### The answers

| Question | Answer | What it settles |
|---|---|---|
| Can an element be built **off-DOM** and swapped in? | **Yes** — properties set while detached, `modelGraphProcessed` fires after `appendChild` | §6.1.1 keeps its off-DOM staging. The specified snapshot/restore fallback is **not** needed |
| Does `modelGraphProcessed` fire reliably, and can it be bounded? | **Yes** to both — a `setTimeout` race around a `{once: true}` listener settles cleanly | The install transaction may await it with a timeout rather than unconditionally |
| Is `config` **assignable on a live element**? | **No** — the assignment does not take | Replacement requires a **new element**, so elements and processed graphs coexist during an install. This is what makes `R_install` a sum rather than a maximum. Read two here; §3 explains why the budget is now `2 + max_quarantined_elements` |
| Install a **new graph plus a parent `subgraphIds` link after** `modelGraphProcessed`, then enter it? | **Yes** | The expression-detail delta path is viable: a graph nobody has seen can be linked from a node already on screen |
| `subgraphIds` navigation | **Yes** — `selectNode(id, graphId, collectionLabel)` navigates | |
| Node data through **both** provider entry points | **Yes**, but see the correction below — `addNodeDataProviderData` is **not** graph-id keyed | Session v1's graph-addressed `Node_data_set` maps to either, once keyed correctly |
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

### The Stage 1 gate, and the correction it forced

`web/test/session.spec.ts` is a **second** spec, and the difference from the spike is the
point. The spike drives a hand-authored fixture to settle questions about the **renderer**;
the gate drives the session `native_graph visualize` actually produced from the committed
resnet18 `model.json` — three graphs, node data and group attributes — to settle a question
about **us**: whether a document `Me_session.validate` accepts is a document the renderer
accepts. Our validator is our own reading of the schema, and a projection can satisfy it and
still be rejected. One hard assertion, and it is about our side: every graph finishes
processing.

It immediately corrected a spike answer. **`addNodeDataProviderData(name, data)` is not
keyed by graph id**: the element wraps `data` in `{[paneGraph.id]: data}` itself, so a caller
that keys it too lands one level too deep and the run reads `.results` off a graph id,
throwing `TypeError: Cannot convert undefined or null to object`. The spike recorded that
call as accepted because nothing threw *at the time* — it never asked whether the data
arrived. That is the difference between a spike that observes an API and a gate that uses
it, and it is why the gate exists as well as the spike rather than instead of it.

### What a node datum can actually say

Two further facts about the pinned element, read out of
`processNodeDataProviderDataForGraph` in the vendored bundle. Both contradict the natural
reading of `Node_data_set.value`, and the browser shell depends on them.

**The element renders `value`, and reads no `label` key at all.** The text it shows
(`strValue`) is derived from `value`, with `typeof value === 'string'` explicitly handled
alongside numbers and booleans. There is no `label` anywhere in that path. So
`Node_data_set.value.label` — the one place the `"<verdict> [sampled n]"` spelling is
written — cannot be passed *as* a label and be displayed; a caller that tried would show the
bare rank digit and silently lose the strength and coverage the label exists to carry. The
shell therefore installs the verification set with the **label as the value**.

**A per-result `bgColor` overrides the gradient.** The gradient is consulted only when
`bgColor == null`, so named buckets need no API the element does not already have, and
`textColor` can be left out because the element derives a contrasting one by luminance.
This is what lets `Map_verify.Verdict.rank`'s 0–6 scale become five *named* buckets —
neutral Vacuous, positive Proved, distinct Tested, warning Unproved, strongest-contrast
Refuted — instead of a position on an anonymous green-to-red ramp, where rank 0 would read
as the best outcome rather than as no claim at all. Bucketing keys on the rank and never on
the label text: the rank states the category exactly, and re-deriving it from prose is the
one place a shortened "proved" could creep back in.

A consequence worth stating because it stayed hidden for a long time: a `Node_data_set`
result is `{ nodeId, value }`, an **object**, not a pair. The shell read it as a pair and
would have thrown "object is not iterable" — but the only set addressed to the default view
is the verification one, and the browser did not request verification until the request
options landed, so nothing ever reached the code. Requesting verification and navigating to
the Kernel view both reach it.

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

### The source view — `Me_source`

The exported program's **own** graph, and the one unconditional row of §5.4's matrix: it
claims that the `model.json` parsed and nothing more. Everything past it is conditional on
lowering, and lowering is where operator support enters.

**Deliberately not a `Me_build.Make` instantiation.** That functor is parameterised over a
dialect whose operands are `Graph_ir.Tensor_id.t` and whose hierarchy is a `Graph_ir.Group.t`
tree. This side has neither: its values are **SSA names** and its hierarchy is
`nn_module_stack`. Sharing the body would mean generalising over both, and the two
projections would then have to be read together to be understood — the functor buys nothing
where the parameters are the whole projection.

**The input/constant split comes from `GraphSignature`.** A `Parameter`, `Buffer`,
`Tensor_constant` or `Custom_obj` spec becomes `const:`, a `User_input` or `Token` spec
`in:`. Reading the exporter's `p_`/`b_`/`c_` name prefixes instead would invent a rule the
schema already states, and would misfile any parameter the exporter happened to name
otherwise.

**Edges follow tensor-valued arguments only.** A `sym_int` operand names an entry in
`sym_int_values`, not a node output; drawing an edge for it would assert a dataflow
dependency the graph does not have, and would then fail as an unknown producer on a graph
that is entirely correct. `tensor_names` is a **total** match over `Argument.t` for the same
reason the classifiers are: a constructor added upstream that carried a tensor would
otherwise lose its edges silently, and a missing edge is a picture that is *wrong* rather
than one that is incomplete — the reader cannot tell the difference by looking. An SSA name
nothing defines is `Unknown_producer`, carrying the name.

**A module level is relative to its parent.** `nn_module_stack` is `;`-separated
`path,name,class` entries whose `name` is the *full* dotted path, so the naive join yields
`layer1/layer1.0/layer1.0.conv1` — every level repeating its whole prefix. Levels are taken
relative: `layer1/0/conv1`. The outermost entry names the whole module and carries an empty
`name`, so it contributes no level, exactly as `Me_build` drops the root group's own
component. A level that does not extend its parent is kept whole; dropping a prefix that is
not there would be guessing at a hierarchy the exporter did not write.

**Every separator in a rendered argument is a literal.** `Fmt.comma` carries a break hint,
and an attribute value is a single line the renderer displays verbatim — a wrapped shape
puts a newline inside it, and the cap then measures a string that also had to pay for the
break. Truncation is reported by a separate `<name>_truncated` attribute rather than an
appended ellipsis, which would put the value past the very bound it reports.

**Root-scoped, like `Me_pt2`.** The node ids here are exactly the ones the sidecar's origins
render to, which is what lets the import comparison pair the two panes; `node_ids` is total
and cheap, so a projection failure does not take the mapping with it. A nested graph has no
signature of its own and no native counterpart to pair with.

### The four-axis branch — `Me_native4d`

`Me_build` is a functor for exactly this: Native4D is projected through the **same body**
with a different op type, so `Me_native4d` is nine lines. Two projections kept in step by
hand would let a divergence read as a dialect difference rather than as the exporter defect
it would be.

**The dataflow branches at canonical Native**, and this is one branch. A graph outside the
dialect's domain is the partiality `Me_classify.native4d` exists to classify, not a failure:
it becomes a capability carrying the reason, exactly as an unsupported operator does one
stage earlier. When it is outside, there is **no Native4D graph, no view and no flow state**
— the same rule the unlowered session follows, since a spine node for a graph nobody can
open asserts a navigability that is not there.

**`--fold` had to start actually folding for this branch to exist at all.** The importer
emits every conv weight right-aligned onto D/H/W/C, so an unfolded graph has non-unit D on
every weight and is outside the dialect entirely; folding is what relays them. But a
structural caller reaches `transform_lowered` with **no constants**, and `Fold_const`
declines every node whose payload is not bound — so `--fold` on an archive was reporting the
capability available while folding nothing. `Native_interp.preload` is now exposed and the
exporter calls it, which is one implementation of "which payloads a fold gets" rather than
two. On resnet18 the canonical graph goes from 194 nodes to 93, and only then does the
conversion succeed.

That also fixes the payload-free half honestly: a `model.json` has nothing to preload, so it
stays outside the dialect and says so. `Outside_dialect_domain` rather than
`Requires_payloads` because that is the first node that actually fails — the matrix admits
either, and the classifier reports which.

### The value graphs — `Me_kernel`

The stage program and the kernel, and they are **not** `Me_build.Make`. That functor
projects an op graph — nodes carrying an operator, edges carrying tensor ids — and neither
of these is one. A `Stage_program.t` and a `Kernel.t` are lists of **values**, each an
`Expr.Value.t` whose dependencies are *recovered by folding the expression* rather than read
off a field. Parameterising the shared body over "how do I get a node's operands" would have
made both call sites harder to read than two projections that each say what they do.

One body serves both here, though, because they really are the same shape: an id, a
signature, and a set of sources. What differs is where the values came from and which
boundary each has.

**Edges come from `Expr.Fold.sources`, which is loads *and* intrinsic descriptors.** Taking
`loads` alone would drop a real dependency — a max-pool names its input only in its
descriptor — and the picture would be *wrong* rather than incomplete.

**`consts` are boundaries.** A synthetic constant-filled operand is a source a stage loads
from and no stage produces; omitting it is the one thing that made a real model fail here,
and its fill value goes on the node because it appears nowhere else. Kernel inputs carry
their **binding** — caller, captured constant, filled — for the same reason: that is the
distinction the adaptation exists to make and it is invisible in the graph shape.

**`Me_ids.value_node` is `v<k>`, a different letter from `op_node`.** Those graphs are
node-indexed and these are value-indexed; a stage and the kernel value adapted from it share
this id precisely so the two panes pair without an explicit mapping — the same argument the
Native comparison rests on.

**Fusion says which unavailability it is.** Where the kernel is available, fusion is
`Not_implemented` — it is a view over an unchanged kernel that has not landed. Where the
kernel is not, fusion is `Prerequisite_unavailable`. Two different facts, and reporting both
as one would lose which.

### Fusion — `Me_fusion`

An **overlay and a decision list over an unchanged kernel**, never a fabricated rewrite.
`Fusion_plan` changes where a value is computed, not what it means, and it carries the kernel
it planned over precisely so a plan cannot be applied to a different one. A "fused kernel
graph" would be a graph nothing produced, and a reader comparing it against the kernel would
be comparing a rendering with the thing it renders — so the capability names the **kernel
graph**, and the fusion view is a second view over it.

**Placement is two facts, not one.** Which dependency *edges* are virtual, and which *values*
need stores. An externally live producer is both — virtual for its consumer and still stored
for whoever else needs it — and a single materialized/virtual per value cannot say that. So
the edges are an `EdgeOverlay` and the values are a `Node_data_set`, and neither is derivable
from the other.

**The overlay rides on the kernel graph**, in `tasksData`, rather than on a comparison: it is
a fact about one plan, and a comparison's overlay slot would make it a fact about two panes.

**A rejection is a fact about a value, so it lives on that value's datum.** One diagnostic
per rejection is seventy on resnet18 alone against a `max_diagnostics` of 64 — a list the
ceiling truncates rather than a report. What reaches the diagnostics is one **summary**,
bounded by construction.

**`Multiple_uses` renders `">= 2"`, never a figure.** The planner's counter saturates at two
because the cross-body total is an `int` and the per-value limits admit a mathematical
aggregate past 2^31 — which wraps negative under js_of_ocaml and would read as unique-use.
Legality only asks one versus more than one, so a count would claim precision the value does
not have; upstream names the field `at_least` for the same reason.

### Verification, placed and counted — `Me_verify`

**The exporter places clusters itself, and the reason is visible in every real report.**
`Map_verify.Group_path` appends a group's **label alone**, while the importer labels every
group with the PT2 node's target — so the twenty sibling `_native_batch_norm_legit_no_training`
groups of a ResNet collapse into one path, and that path cannot key the ID-qualified
namespace the projection emits. `Map_verify`'s own attribution also takes the *first*
destination edge that resolves, which is set order rather than a common group.

So placement is recomputed over the namespaces the projection actually used: a cluster's
**destination** members map to their producing nodes, **producerless members are ignored**,
and the placement is the longest common namespace prefix of the rest, root when none remain.
That is `Graph_view.common_group`'s rule expressed in namespace space, which is the space the
answer has to be in. Ignoring producerless members is not a detail — a fold's cluster names
the folded weight, which is a graph constant belonging to no group, and counting it as "root"
would lift most clusters on a real model out of their groups.

**`namespace_of` is shared, not recomputed.** `Me_build` hands out the same function its own
projection uses. Two answers about one hierarchy, computed two ways, is how they come to
disagree.

**Two destinations, saying different things.** `groupNodeAttributes` is the only part of the
wire format keyed by **namespace** rather than by node id, so it is where a per-group tally
belongs; the node-keyed `Node_data_set` is the only form that can colour a node, and carries
one node's `Outcome.join` over its outputs. Neither substitutes for the other.

**The rank is how weak a verdict is, not how good it is.** A gradient over it runs proved
(1, 2) through tested (3, 4) to unproved (5) and refuted (6). `Vacuous` is **0, below
proved**: a creation or a deletion claims nothing, and the ordering exists so a claimless
cluster cannot mask a real verdict when a node's outputs are joined.

**Two capability keys, not one.** `Native_interp.transformed.composed` survives per-pass audit
truncation, so mapping that truncation to one `Verification → Over_limit` hides a result that
is still available, while `Available` alone conceals that the per-pass detail is incomplete.

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

**Lowering.** The row every other capability's fate follows from, since it decides whether
there is a Native graph at all. `Unsupported_operator` and `Unsupported_input` are
recoverable — and recoverable for a `.pt2` and a bare `model.json` alike, because
`Native_interp.lower` takes an `ExportedProgram.t` and gains no operator support from the
archive payload. Everything else is fatal: a malformed graph is a decoder that accepted
what it should not have, the builder/provenance/transform/verify/lens rows are internal
invariants, and `Eval`/`Tensor_bridge` belong to execution, which export never performs.
`` `Bad_view `` (op3-impl.md commit 1's row for an invalid `view`/`_unsafe_view` target —
two inferred dims, a numel mismatch, a non-divisible inference, or a source past the numel
ceiling) needs no arm of its own: it is one more member of `#Native_interp.malformed`, this
rule already matches wholesale, and it reads Fatal for the same reason every other row here
does — a decoder that accepted an invalid reshape target is our bug, not the model's.
Confirmed rather than assumed in `test/me_group3_cram.t`.

**`diagnostic_code` is where the two closed vocabularies meet, once.** Every
`Capability.reason` has a `Diagnostic.Code` of the same name, so the map is an identity in
spelling — which is exactly why it is written out rather than assumed. They are separate
types on purpose, and a code added for the worker protocol must not silently become a
capability reason.

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

## 11. The CLI — `native_graph visualize`

```
native_graph visualize --model MODEL.json|MODEL.pt2   (--pt2 is an alias)
    [--output SESSION.json] [--format session|collections]
    [--limits untrusted|small] [--fold]
```

**Format detection is content plus declaration, never extension alone.** `{` means a
`model.json`, `PK\x03\x04` a zip; a mislabelled file is the ordinary case, not the
adversarial one, and the two loaders fail very differently on each other's input. Size is
rejected from `stat` **before reading**, so an oversized file never becomes an OCaml string.

Both paths end at a `Pt2_native_graph.t`, which is what `transform_lowered` takes — the
payload-free JSON path exists precisely because the archive is a prerequisite of binding
payloads and of nothing else.

**Only the two wire-selectable profiles are offered.** `large` and `trusted` exist for
callers holding data they produced and cannot be named on the command line, which is what
makes `Wire_limits` a guarantee rather than a UI convention. `small` accepts a real vision
model, which the cram pins.

**`--fold` on a payload-free `model.json` is a successful structural session** carrying
`Feature Fold → Unavailable Requires_payloads`, not a usage error: the browser cannot
surface one and the same code path serves both shells.

**`--format collections` warns on stderr** and names what it discarded rather than silently
emitting half a document.

**Two comparisons, and only one of them can rely on `MATCH_NODE_ID` — which each one
DECLARES.** `c/canonical` (initial → canonical Native) ships with **no** explicit mapping
entries: stable node ids plus the id-identity rule (a changed value is a new id) already
pair every node a pass did not touch. Entries for the changed nodes come from the pass lens;
until that lands, an empty set is *correct* rather than approximate. `c/import` (exported
program → initial Native) is the opposite case: the two panes speak different id languages,
equal ids pair nothing across them, and every correspondence has to be stated — which is
what `Me_pt2` computes, as connected components rather than pairs.

`Sync_navigation.match_node_id_fallback` is how each says which it is, and it is not a
rendering preference the browser could infer. An empty `entries` means "the ids already pair
the untouched nodes" for one comparison and "no mapping was computed" for another, and
nothing else on the wire separates them. It cannot be left to the renderer's default either,
because the fallback and the highlight are the same fact seen twice: a consumer that
disabled the fallback on `c/canonical` would give every node an empty mapped set, which
`renderDiffHighlights` reads as "all mapped nodes are missing" — turning
`show_diff_highlights` from a report of the handful of touched nodes into a claim that the
pass rewrote the whole graph. The browser translates the field verbatim and infers nothing;
`web/test/compare.spec.ts` proves both directions against the real element, and
`me_visualize_json_cram.t` pins entry count and flag together, since neither alone
distinguishes the three configurations.

**`Transition.comparison` is populated on both transitions**, which is also what makes the
Pt2 flow state name a real graph. Before the source view existed it pointed at the *initial
Native* graph, which was a placeholder the session validator could not object to.

### Two session shapes, one code path

A model this repository cannot lower is an **ordinary outcome the matrix has a row for**,
not a usage error: the exported program decoded, which is the whole of what `Source` claims,
so it becomes a *successful* session carrying a source view and a capability vector that
says what is missing and why. The browser shell cannot surface a usage error and the same
code path serves both, which is the same argument `--fold` on a payload-free `model.json`
rests on.

`Initial_native` carries the reason the classifier gave — `Unsupported_operator` or
`Unsupported_input` — rather than `Prerequisite_unavailable`, which would be circular for
the row that *is* the lowering, and would drop the only detail a user can act on. Everything
downstream is `Prerequisite_unavailable`, enumerated through `Me_classify`'s
`depends_on_lowering` rather than restated at the call site. There is **no flow**: with no
Native state the spine would hold `s/pt2/000` and no transitions, and a one-node flow graph
asserts a navigability that does not exist.

**Free text lands in a diagnostic, not in the capability.** The vector says *which* rows are
missing; `Me_limits.Diagnostic` is the bounded type that carries the message, and it is the
one type crossing every boundary of this design. A `Me_classify.Fatal` row still exits —
reporting an internal invariant failure as "this model is outside what we support" tells the
user to change their model to work around our bug.

`model.op_targets` counts **distinct PT2 targets**, which is what the field names and the one
figure both shapes can produce; it previously counted native nodes, which the unlowered shape
has none of.

### Three crams, and what each reaches

`test/me_visualize_json_cram.t` is **ungated** — the committed resnet18 `model.json` is a
submodule file, not a downloaded weight, so the whole lower/transform/project/validate path
runs in every build, and its payload-free nature is the point rather than a limitation.
`test/me_visualize_unsupported_cram.t` is ungated too and builds its graphs **in the cram**:
what it has to exercise is a graph the lowerer rejects, and no released model is one. It
pins the complete vector for both recoverable rows and that a fatal row still exits.
`test/me_visualize_cram.t` is `PT2_DATA`-gated and adds what neither can reach: the archive
path, and `--fold` actually available. It is named in `pt2.runtest` — the only target that
sets `PT2_DATA` — because a gated cram no target names never runs at all.

Both summarise rather than print the document. A golden holding a megabyte of node JSON
changes on every unrelated projection tweak and gets re-promoted without being read, which
is the opposite of what a golden is for. What is pinned is the shape, the counts, and the
**complete capability vector** — a test asserting only the interesting key would let every
downstream row drift.

---

## 12. The worker request — `Me_request`

**Self-contained means the request carries the profile.** Leaving the worker to pick its
own default would let a page build a session under one profile and a worker validate it
under another — the disagreement `Wire_limits` exists to prevent — so `limits` is a field
and there is no fallback.

**Closed by construction, in both directions.** Every record here is `private` behind a
checked constructor. A checked *decoder* does not make a public encoder safe: with a public
record the encoder can emit JSON its own decoder rejects, and the round-trip property then
holds only of whichever values a fixture happened to pick. Closing the type is what gives
the closure test something to be true about.

**Provenance only, never the bytes.** Those travel beside this as a transferable
`ArrayBuffer`; a pure OCaml type cannot hold a js_of_ocaml value, and putting the bytes in
the JSON would defeat the transfer. What is here is exactly the set the determinism claim
quantifies over.

### Request identity is a grammar

`<uuid>-<seq>`: 36 characters of canonical **lowercase** hyphenated UUID, a `-`, then 1–10
decimal digits with no leading zero (bare `0` allowed) whose value is below
`Hard.max_seq_exclusive`. Exact rather than merely bounded, because the builder is the sole
producer and an exact check costs one fixed-width parse — which makes `Hard`'s two byte
constants *consequences* of this grammar rather than a second rule that can drift from it.
A length ceiling alone admits `9999999999`, a sequence no producer bounded by 2^32 can
issue, so "consumer and producer accept the same strings" would be false in the direction
that matters. Uppercase is refused for the same reason a leading zero is: two spellings of
one id make `epoch` a non-injective projection, and the epoch is what a delta binds against.

The sequence is compared through `Int64.of_string` and **never** through `int`, which wraps
under node before the comparison.

**There is no `epoch` field anywhere.** It is `Request_id.epoch id`, so no message can name
one epoch in its id and another beside it.

### One identity, not two

`Detail_key` carries `{ parent_graph; value }` and *derives* `parent_node`. A Kernel value
has exactly one tensor id and one body, so a separate `parent_node` field would admit a
mismatched pair that validation would then have to reject. The visible consequence is that
the derived id's last two components are the same tensor id — that is what the derivation
looks like once the parent stopped being a field.

`create` checks `parent_graph` **and** the derived id, because a component inside the
ceiling can still render an id past it. `validate ~limits` is separate from construction:
a key carries no witness of the profile it was built under, and only the enclosing request
knows which profile applies.

### The digest lives inside `Catalog`

A sibling `verified_sha256 option` would permit a `Local` source claiming a verified digest
— a false verification claim entering the deterministic session — and a `Catalog` source
with none. Neither is constructible, so the session's digest rule needs no validation.

### Options are normalised, not merely validated

`stages` is filtered *through* `Capability.all_stages`, which removes duplicates and imposes
constructor order in one pass. The list is therefore bounded by that type's cardinality by
construction — no separate ceiling — and two requests differing only in the order a user
typed them are the **same value**, which is what makes "same options ⇒ identical JSON" a
statement about a canonical value rather than about a spelling. An empty list is an error:
a request asking for nothing is a caller defect, not an empty session.

### The constructors revalidate; they do not check a witness

`build_session`/`build_detail` re-run every profile-dependent check under **this request's**
profile — `Source`'s `name`/`url`/`ref_`, and `Detail_key.validate`. A `Source.t` keeps no
record of the profile it was checked against, so "was it built under this one" is
unobtainable; revalidation is obtainable and is what the worker will enforce. So the
`Invalid_source` and `Invalid_detail_key` rows are **reachable** for a caller using the
checked constructors, and that is the point: a source built under `untrusted` and combined
with a `small` wire profile is valid alone and invalid here. The suite drives exactly that
combination.

### Decoding is staged, and finishes through the constructors

Jsont decodes an object's members independently, but `Source.create ~limits` needs the
*decoded* limits — so the members arrive as raw pieces and are validated in a fixed order:
id, profile, options, source, key. The last two take the profile the step before them
produced, which is why the order is part of the design and not an implementation detail.

The decoder then finishes through `build_session`/`build_detail` rather than assembling the
record itself. That is what makes encoder and decoder one domain rather than two that happen
to agree, and the suite asserts it as encode → decode → encode returning equal bytes.

Note what this ordering does *not* buy: a member's own codec (an unknown stage tag, a
non-string integer) fails during member decoding, before any staged step runs. The staged
order governs the checks that need the profile, which are the ones that could otherwise be
run under the wrong one.

---

## 13. The worker response — `Me_response`

**The document lives inside the constructor that has one.** Returning a `(t * string)` pair
and letting the shell send any constructor makes every combination the envelope forbids
constructible — a session with no document, a failure carrying one, a byte count disagreeing
with the bytes beside it — and a JS test that decodes only the metadata cannot see any of
it. There is no `bytes` field on a payload at all: it is **derived** in `Wire` from the
document it describes, in one place, from the same string.

**Three types, narrowing by one constructor each time, and the narrowing is the design.**
`Handle_result` is what a function holding a validated request may return. `Final` adds
`Protocol_failure`, the pre-decode failure only the shell can produce — every check before
the request exists (length, the request codec, the profile, the source, the buffer) can fail
with no validated id in hand, and a `Failed` could then be built only by echoing untrusted
fields. A handler that received a validated request cannot honestly produce that
constructor, and `Handle_result` says so as a type rather than as a rule in prose that
something can contradict.

**`Meta` is what actually crosses**, and it is neither of the others: the final constructors
carry a document the metadata must not contain, and the metadata carries a byte count they do
not have. It is `private`, produced only by `Wire`, which is what keeps the derivation true
now that the invariant lives on the type that is serialized. A compiled OCaml variant through
`postMessage` would hand the page jsoo's private runtime representation, with no stable
discriminant and no stable field layout — so a `"kind"` discriminant over five constructors,
with every member optional on the wire and the discriminant saying which are required.

Every counter is `int64` through `int64_as_string`; `bytes` is an `int` and the width rule
does not bite, because it is the length of a string that already exists, bounded by
`max_session_bytes` — itself an `int` because the writer limit is one.

**The metadata ceiling is an aggregate bound applied by the writer**, not an inference from
individually-bounded fields: JSON escaping expands and three bounded fields still sum. So
`of_final` is result-valued — a writer limit is a property of the writer, and a function
whose implementation can fail should say so regardless of which inputs reach it today. The
ceiling is injectable so the branch can be driven; a bound nobody can reach is a bound taken
on trust.

**The terminal response is precomputed.** `constant_protocol_failure` is already encoded, so
the fallback path never runs the encoder — "it cannot itself fail" is a property of a value
rather than an argument about a function, and the suite asserts the ceiling over exactly the
bytes that path posts (126 of 1048576).

---

## 14. Model bytes in, session out — `Me_export`

**This is the body the CLI used to hold, and it moved because the browser compiles this
library rather than a second implementation of it.** A session builder living in `bin/`
would have made "what the CLI exports" and "what the page renders" two things again the
moment the worker gained one. What stays in the CLI is the shell: the filesystem, the
profile flag, the two output formats — and the `stat` size check, which only a filesystem
caller can perform and which keeps an oversized file from ever becoming an OCaml string.

**In memory throughout.** The worker receives a transferred `ArrayBuffer`, not a path, so
detection, archive opening and decoding all take a string (`Pt2_archive.of_string`).

**Declared is a hint; the content decides.** `detect` reads `PK\x03\x04` or `{`, and the
session records the *detected* kind — which is what makes `--fold` on a payload-free
`model.json` a decision about the bytes rather than about a label. `handle` additionally
checks the request's *declared* format against it, and a disagreement is `Invalid_source`:
a fact about the source, not an internal error.

**`handle` is total.** Every failure a valid request can produce is already a constructor of
the return type, so there is no error row left to widen and no second place where a failure
could become a message. It holds a validated request, so it always knows which request it is
failing about — the condition under which the answer is `Failed` — and `Handle_result` makes
the other constructor unwritable there.

**A model this repository cannot lower is a session, not a failure**, in the worker exactly
as in the CLI: the exported program decoded, which is the whole of what `Graph_stage Source`
claims. Only a `Me_classify.Fatal` row becomes `Failed`.

`diagnostic_code` draws the same line the classifiers draw: a model that is not what it
claimed is `Invalid_source`, a ceiling reached inside the projection is `Over_limit` — a
real model can be too big and that is a bound doing its job — and an invariant of ours is
`Internal`.

---

## 15. Expression detail — `Me_detail`

**On demand, which is the whole reason it is a delta rather than a stage.** A kernel value's
expression can be far larger than the graph node that produced it, and a session carrying
every one of them up front would pay for every expression to show one.

**The delta carries no epoch and no key.** Three identities take part in a detail response —
the pending request's, the metadata's, and the payload's — and comparing only the first two
lets a shell announce key A with a payload for key B, which is then validated on its own
terms and installed while the coordinator completes A. *Removing* the field beats comparing
it: the validated key arrives as an **argument** to `apply`, so metadata and payload cannot
name different values at all. The epoch belongs to the browser runtime and only the bridge
holds it, so staleness is checked there and this stays pure in the session and the delta.

**The initial session stays referentially valid.** A kernel value node carries no
`subGraphIds` until its detail exists, so nothing in a fresh session points at a graph that
is not there. The graph, the view and the link are installed **together**, because a link to
an uninstalled graph is a dangling reference the validator rejects and a graph nobody links
to is unreachable — a detail commits all three or none.

**Re-requesting replaces.** Only the committed result counts toward the aggregates, so asking
twice cannot inflate them. The graph and the view carry the same id, so one predicate removes
both. The aggregates are checked on the **merged** session, over every installed detail; the
node ceiling is checked on the **delta alone**, because a merged check would let an
over-ceiling delta through whenever the session it joins is small.

**The size ceiling runs before the walk.** `Fold.size` is an unmetered traversal but
allocates nothing, while building nodes allocates per node — and an expression is exactly
the shape whose size is not apparent from the thing that names it.

**Every node carries the subtree rooted there, rendered and bounded — one attribute, not
per-constructor ones.** `Expr.Pp.index` takes a `names` function precisely so its output
cannot depend on allocation history, and this walk holds no scoped naming environment to
give it; `Expr.Pp.value` builds its own. So a `Select`'s condition and a `Reduce`'s bounds
are visible in their node's rendering rather than as index-language nodes, which would put
two languages in one graph.

**The worker's detail path is a smaller pipeline than its session path**, and re-lowers
rather than caching. That is what a self-contained request means: the worker holds no session
between requests, so there is nothing that could be stale.

**The measure is `Expr.Fold.size`, which counts index trees too**, so it exceeds the number
of value nodes produced — a convolution is 88 by that measure and 8 nodes here. The bound is
conservative deliberately, because the index trees are what the bounded per-node rendering
pays for, and the rejection is named for what was *measured* rather than for what would have
been built.

**`native_graph detail` is its own command, not a flag on `visualize`.** The two produce
different documents with different lifetimes — a session is a whole model, a delta is one
value's expression merged into a session someone already has — and one command emitting
either would have to be told which.

**The browser gate drives the whole path.** 0A proved a late graph plus a parent
`subGraphIds` link works, with a hand-authored fixture; the gate now does it with a real
delta. It selects the first `v<N>` kernel-value node from the session it just exported, then
requests that value's expression, so the fixture stays valid when the pinned model changes.
That is the only place the merge is checked against the element rather than against our own
validator.

"Entered" is `selectNode` called TWICE, not once: the first call selects the linking node
inside its *parent* graph -- `selectNode`'s own implementation only reveals a node in an
already-named graph, it never activates a node's linked subgraph -- so it proves the link is
findable and nothing more. The second call passes the delta graph's own id as the target
graph instead, which is the same primitive a real click on a `subgraphIds` node drives
(`selectGraphInPane`).

The proof that it actually switched is `modelGraphProcessed` firing **for that graph's id
specifically**, not merely firing again. It measurably does fire again for a same-graph
reselect too -- caught by a first version of this gate that waited for "the next event" and
passed even when the second call was sabotaged to reselect the still-active parent graph,
because a node reveal queued by the FIRST call can deliver its own event after the second
call's listener is already attached. `ModelGraphProcessedEvent.modelGraph.id` (Angular
Elements' own bridging of the `@Output()`, not a guess reconstructed from timing) is what
makes the two distinguishable.

---

## 16. Testing posture

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

## Group-2 coverage: `test/me_group2_cram.t`

`me_visualize_json_cram.t` runs over the committed resnet18 `model.json` and
reaches none of the five Group-2 functional overloads — no downloadable model
serialises them (see `.ai/native_aten_bridge_layout.md`). `me_group2_cram.t` is
the only place they reach a session at all: a payload-free program built in the
cram itself, carrying `conv2d.default`, `conv2d.padding`, `max_pool2d.default`,
`rms_norm.default` and `linear.default` in one chain, with **every parameter
non-default**.

Defaults would prove nothing here. A projection that carries a field and one
that drops it and re-derives the default are indistinguishable on a default
value, and that is precisely the failure this cram exists to catch. What it
pins per node: the source view's labels and namespaces, the imported native
graph's `params` attribute in full, each node's provenance namespace (the
serialized target, group-qualified, so a relayout permute is attributable to the
node that caused it), the output shape, and every incoming edge's
(source, output slot, input position) triple.

## Group-6 coverage: `test/me_group6_cram.t`

`pad.default` and `slice.Tensor` reach a session nowhere else — no model this
repository can download serialises either (op6-impl F1) — so this cram is the
whole of their Model Explorer evidence, and it is built payload-free in the cram
itself like Group 2/3/5's.

What it pins beyond the usual labels/namespaces/shapes/edges: **the two
conversions that only exist at import**, both visible in the `params` attribute
and nowhere else in the session.

- Pad's serialised list is **innermost-first**, so on a rank-4 input
  (right-aligned onto `D, H, W, C`) the list `[1, 2, -1, 0]` becomes
  `pads=[W:-1,0, C:1,2]`. A reversal prints `W:1,2 C:-1,0`; an unsigned decode
  loses the crop.
- Slice's bounds are **canonical**: `dim=-1` has become `C`, an `as_sym_int`
  start has resolved to a value, and `end=99` has been *clamped* to the extent
  rather than refused. What is stored is `[1, 7)` — the point of resolving at
  import rather than at evaluation, and invisible in the source view.

It also pins the two ways a Group-6 graph can fail, which land at **different
stages**:

| graph | outcome |
|---|---|
| `slice` on `dim=0` of a rank-5 tensor (the frame's `T`) | Native imports it; `stage:native4d` is `unavailable outside_dialect_domain`, naming the axis |
| a crop or a slice that empties an axis | the import itself fails; `native_graph` exits `123` with the `Shape_error` row and **no session is written** |

That asymmetry is worth stating because it is easy to expect one shape for both:
an axis outside the dialect still has a Native graph to show, and a shape with
no Native form does not.

## Group-7 coverage: `test/me_group7_cram.t`

The first group cram whose reason **splits**. `layer_norm.default` is the usual
case — every ExportedProgram in the corpus lowers it before writing
`model.json`, so this is the only place it reaches a session at all.
`native_layer_norm.default` is not: four ViT models serialise it 148 times and
`vit_b_32` is downloadable. It is here anyway, because a payload-free fixture
pins the rendering deterministically and because its two dead outputs are the
thing to see — a real model shows them too, but only behind a download and an
839-node graph.

Both targets render as the **same** `Layer_norm` node. They differ in their
argument list and their return arity, not in the arithmetic, and the only
visible difference in the session is the epsilon each carried (`1e-05` against
`1e-06`).

What it pins beyond the usual labels/namespaces/shapes/edges:

- **The dropped outputs.** The source node declares three; the native node has
  one, and the `mean`/`rstd` edges do not exist in the native graph at all. That
  is sound only because nothing reads them, and `Live_layer_norm_stats` is what
  refuses a graph that does — so the rendering is a fact about the graph rather
  than a hope about it. See `native_multi_output_design.md` §3 for why this is a
  third dead-output shape and not the batch-norm one.
- **The operand order and the normalized axis**, through `params`, which is the
  op's own `pp` verbatim: `layer_norm x=t0 weight=t1 bias=t2 params={dims=[C];
  eps=1e-05}`. Reading the trailing extents as leading ones prints `dims=[H]`;
  a swapped affine pair prints the operands the other way round. Both are
  shape-preserving, so this is the only place in the session they show.
- **Shared affine operands.** Both nodes read the same `const:t1` / `const:t2`
  at slots 1 and 2. An importer materialising a ones/zeros tensor per node would
  show four extra constants here — and would build a structurally different
  graph from the other importer's for the same node.

The capability-unavailable fixture is a normalization over **all four** axes of
a rank-4 input, which right-aligns onto `D, H, W, C`: Native imports it and
`stage:native4d` is `unavailable outside_dialect_domain`, naming `D`. Same shape
as Group 6's `T` case, and for the same reason — the axis is gated on the Native
`Axis.t` before conversion, so the diagnostic can name it.

## Group-8 coverage: `test/me_group8_cram.t`

A stronger version of Group 7's split reason: `scaled_dot_product_attention`
is not merely absent from a downloadable model's graph, its NAME is *inside*
one — 600-1200 occurrences in the four `vit_*` graphs' `from_node`
provenance — while its actual TARGET appears zero times, because every
exporter decomposes it into twelve primitives before writing `model.json`
(op8-impl.md F1). This hand-built, payload-free fixture is therefore the
row's only Model Explorer evidence, and stays that way until the
decomposition (out of scope for this row) is separately supported.

Two nodes chained (`attn1 -> attn2`), pinning both axes `op8.md:253-259`
asks for in one fixture: `attn1` has no mask and the default scale, `attn2`
has both an explicit mask and an explicit scale (`0.1`).

What it pins beyond the usual labels/namespaces/shapes:

- **The three mandatory operand edges in order, plus the optional mask
  edge.** `params` (the op's own `pp`) shows `query=... key=... value=...
  mask=none|t..`, and the edge list shows the same order at stable slot ids.
  `attn1`'s mask stays absent rather than a materialized ones-shaped
  constant — `Graph_ir`'s `Sdpa` carries `mask : Tensor_ref.t option`, and a
  fourth constant edge there would mean this importer and `Native_interp`
  built structurally different graphs for the same absent-mask node.
- **`Default` vs `Explicit` scale**, through `params` alone:
  `params={scale=default}` against `params={scale=explicit(0.1)}`. The two
  are not the same computation — `Default` is `1/sqrt(head_dim)`, resolved
  only inside `Compute` — so the JSON/session distinction has to survive
  intact rather than being pre-folded into a number.
- **The output shape**, which is exactly `query_shape` (`Ev = E`, F4's
  flash-oracle constraint, not a mathematical one): both nodes keep
  query's `[Wq=2, C=4]` even though `attn2`'s query is `attn1`'s output.

The capability-unavailable half needs no separate fixture: **both** nodes
already fail `stage:native4d` unconditionally (F8 — the batch axis is `D`,
which the N/H/W/C dialect has no name for, and there is no legalization to
fall back to, unlike `Bmm`'s single-batch escape hatch). `Domain.check` is
not an accumulator, so the session reports the FIRST failure (`n0`) and
stops there — the same fail-fast contract every other capability-unavailable
fixture in this file relies on.

---

## 17. The visualizer bundle — built from the submodule, not installed from npm

`web/src/vendor` and `webapp-dist/vendor` are produced by `make visualizer.build`, which
runs upstream's own release script (`src/ui/scripts/build_npm.sh`) inside
`vendored/ocaml-model-explorer/model-explorer`. The npm dependency on
`ai-edge-model-explorer-visualizer` is gone.

**One commit pins both halves of the integration.** The OCaml types in
`vendored/ocaml-model-explorer/lib/model_explorer.ml` are hand-written against upstream's
TypeScript schema, and the runtime is now built from the very tree those types describe.
Previously they were pinned separately — the types against upstream's current shape, the
runtime against a published release — and the gap was not academic. Every one of these is
emitted by `lib/model_explorer_export` and was **silently dropped** by
`ai-edge-model-explorer-visualizer@0.1.2`:

| Field we emit | 0.1.2 | upstream |
| --- | --- | --- |
| `Graph.tasksData` — `me_export.ml:383-388` | absent from `src` *and* `dist` | `split_pane.ts:108-117` |
| `EdgeOverlaysData.graphName` — `me_fusion.ml:121` | absent | `edge_overlays.ts:35` |
| `EdgeOverlay.showEdgesConnectedToSelectedNodeOnly`, `.visibleEdgeHops` | absent | `edge_overlays.ts:69,88` |
| `SyncNavigationData.showDiffHighlights` — `me_pt2.ml:135`, `me_export.ml:428` | absent | `sync_navigation.ts:103` |

`tasksData` was the one with a visible cost: `Me_fusion`'s overlay rides on the kernel
graph and rendered nothing at all. `test/session.spec.ts` now asserts the loaded bundle
reads the field, with a control alongside so a zero reads as "unsupported" rather than "the
fetch lost it". That assertion is the tripwire: it fails if the npm build is ever vendored
back in.

**Why there was no published release to move to.** npm carries only 0.1.0/0.1.1/0.1.2, and
`latest` is 0.1.2 (2025-06-23). Upstream's own `custom_element_npm/package.json` is still
0.1.2 at `main` despite a year of commits to `src/ui/src`, and all four of upstream's
`custom_element_demos` consume the npm package with lockfiles resolving to 0.1.0/0.1.1 —
older than the pin this repository had. Building is the only way to get the schema the
OCaml layer already targets. The Python distribution is not an alternative: it ships the
full Angular app, no custom-element bundle.

**The patch, and why it is not optional.** `angular.json` builds `custom_element` with a
`fileReplacements` entry swapping the real worker service for a stand-in that reads
`window.modelExplorer.workerScriptPath` — which is exactly what lets the bundle be served
from a path prefix, as the Pages deployment does. On 2026-04-06 (`49aa6a8`) the real
service moved worker construction into `init(onPortal)`; the stand-in has not been touched
since 2025-02-19, so `ng build custom_element` fails on `TS2339: Property 'init' does not
exist`. Nothing upstream catches it: `build_ui_code.yaml` runs `ng build model_explorer`
only, and `custom_element` is built solely at release time. That is the most likely reason
the npm package has not moved in a year.

`patches/model-explorer-custom-element-worker.patch` restores `init`/`ngOnDestroy` on the
stand-in, and is the **only** drift — with it applied, both Angular builds succeed and all
three browser gates pass unmodified, with no `NG0953`. `make visualizer.patch` is
idempotent and **refuses rather than skips** when it no longer applies: a silently
unpatched tree fails later, in a type error that says nothing about the submodule having
moved.

A consequence worth knowing before it alarms someone: once built, `git status` reports
`vendored/ocaml-model-explorer` as modified, and `git diff` shows the commit with a
`-dirty` suffix. That is the patched working tree of the nested submodule, not a moved
pointer — the recorded SHA is unchanged, so there is nothing to commit and nothing to
clean up. `git submodule update --force` would revert the patch, and the next
`visualizer.patch` reapplies it.

**Build shape.** `$(VISUALIZER_DIST)/main_browser.js` is a file target with every step
inside the recipe, so an existing bundle costs nothing — no `npm ci`, no patch, no Angular
build. That is what makes the CI cache worth having: keyed on the nested submodule commit
and the patch hash, a hit skips a 1191-package install and two Angular builds outright. A
prerequisite on `node_modules` would defeat it, since a cached bundle with no
`node_modules` would reinstall and then rebuild against the newer timestamp. There is no
dep-tracking on the submodule source — the same caveat `data/dune` carries — so after
bumping the submodule, `rm -rf $(VISUALIZER_DIST)`.

`build_npm.sh` builds `model_explorer` as well as `custom_element`, and must:
`worker_service.ts` constructs its worker from a runtime string rather than
`new URL(..., import.meta.url)`, so the custom-element target emits no worker chunk of its
own and `worker.js` can only come from the app build.

CI checks submodules out top-level only, by the policy `build.yml` documents, so both the
`js.yml` browser job and `pages.yml` name the nested checkout explicitly — before the cache
step, which is what makes the key computable.

---

## 18. The comparison surface, as the element actually implements it

`web/src/compare.js` + `web/test/compare.spec.ts` are `web-ui-3.md`'s Prerequisite 0: the
two-pane adapter may use only what is proven here, in Chromium, against the bundle §17
builds. Hand-authored and tiny, so every observation is about the visualizer rather than
about resnet18. Like the Path B gate it **asserts** nearly everything it records — these
are the premises the adapter is built on, not accommodations to a third party.

**Opening a comparison is two phases, and both need their own event.**
`selectNode(nodeId, graphId, collectionLabel, 1)` is what creates the second pane:
`selectGraphInPane(g, 1)` delegates to `openGraphInSplitPane` when only one pane exists,
adding the pane synchronously before `panes()[1].id` is read. `modelGraphProcessed` carries
`{modelGraph, paneIndex}`, and a candidate is ready only when both `(left, 0)` and
`(right, 1)` have arrived.

**Wait on the pane index as well as the graph id.** `Session.validate` does not require a
comparison's two panes to name different graphs, and when they coincide
`detail.modelGraph.id` is identical for both events — so a wait keyed on the id alone
resolves the right-pane wait on the left pane's event and the adapter finalizes a
comparison whose second pane never processed. The gate covers that case explicitly, and
records which pane the *resolving* event carried: `fired` alone cannot police this, because
the wrong event resolves the promise just as well as the right one. Deleting the filter
turns the recorded pane into a mismatch, which is what makes the assertion able to fail.

**Sync navigation is armed a beat AFTER the second pane event.** The component that reads
`config.syncNavigationData` and selects `VISUALIZER_CONFIG` mode is
`sync-navigation-button`, and `split_panes_container.ng.html` constructs it under
`@if (hasSplitPane && allPanesLoaded())`. So a probe run in the same tick as the pane-1
event reports "nothing paired" for a pairing that is about to work — measured at ~1.2 s
here. This costs the adapter nothing, since nobody clicks a node in the tick the pane
appears, but a gate that did not know it would pin a false negative as the contract. Every
candidate is therefore armed on a pairing known to exist before any observation is trusted.

**Cross-pane navigation is observable through `uiStateChanged`,** whose
`paneStates[i].selectedNodeId` is positional; `selectedNodeChanged` carries only an opaque
`paneId`. Reading the other pane's *current* selection proves nothing, though — both panes
carry a selection from the moment the comparison opens, so the observation has to be the
**change** away from a value deliberately primed to be the wrong answer.

Proven: 1:N, N:M, and N:M read right-to-left all follow the supplied `mappingEntries`;
`disableMappingFallback: true` is honoured; and the *same* document with the flag unset
pairs an id present in both graphs and named in no entry. One direction alone cannot tell an
honoured flag from a mapping that never loaded, which is why both are asserted — and it is
what `Sync_navigation.match_node_id_fallback` decides per comparison. With an **empty**
mapping and the fallback off, nothing corresponds to anything, not even that id; upstream's
`renderDiffHighlights` reads that same empty mapped set as "all mapped nodes are missing",
which is why `c/canonical` must declare its fallback rather than have one assumed.

**What the gate does NOT claim.** Diff highlights and overlay edges are drawn in WebGL, and
the element renders into a shadow root (`ViewEncapsulation.ShadowDom`), so "the node is
outlined" and "the overlay is in the left pane" have no public observable — the overlay
dropdown is gated on config, not on loaded overlays, so even its presence says nothing.
Configs carrying `showDiffHighlights` with its two border colours, and per-pane overlay
lists, are asserted **accepted**: both panes open and nothing throws, which is meaningful
because `SplitPane` provides its own `EdgeOverlaysService` and loads the matching list in
`ngOnInit`, where a malformed payload would throw. Nothing beyond that is claimed. Per-pane
routing rests on `split_pane.ts:73-93` being read, not on a pixel.

`addNodeDataProviderData(name, data, paneIndex)` is accepted per pane, at the point the
adapter installs it: after both pane events.

## 19. The two-pane adapter — `web/app/`

§18 says what the element does; this says what the browser app makes of it. Phase 3 of
`web-ui-3.md`: the two comparisons `Me_session` already exports (`c/import`, `c/canonical`)
become openable presentations. Nothing about the wire changed except §20's one field.

**A presentation is a closed sum, and it replaces "the current view id" at every boundary.**
`presentation.js` declares `{kind:'single', view}` and `{kind:'comparison', comparison}`;
Phase 4's `flow` is the third arm, which is why every consumer switches on `kind` rather
than testing whether a field is present. A comparison is *not* a view and cannot be encoded
as one — `View` names one graph and a comparison names two — so `Coordinator.view` became a
**derived** accessor (`null` while a comparison is on screen) over a retained
`#presentation`. Two stored fields could disagree; one cannot.

The URL follows the same shape. `encodeUrl` writes exactly one branch key, *from the
branch*, so exclusivity is structural rather than a rule each caller remembers; `decodeUrl`
treats a URL carrying both `view` and `comparison` as naming **neither**, because no
non-arbitrary rule picks a winner and guessing opens a presentation the link did not ask
for. `requestKey` is untouched: a presentation is not part of a request, which is what keeps
"same model, different pane pair" a `present()` rather than a re-export.

**A load resolves a view; a comparison is opened on top of it.** `Coordinator.load` takes an
ordered `prefer` list of *view ids*, so `?comparison=c/import` still loads a single view
first and then switches. That is one extra presentation change on startup, and it is the
honest shape: the comparison rests on a document that does not exist until the load lands,
so it cannot be resolved before then. The switch replaces its URL rather than pushing —
one navigation, one entry.

**Two-phase readiness, and the `safe` invariant that pays for it.** `Renderer` resolves a
comparison completely before a slot, timer, deferred or element exists — one record with
that id, each pane's graph found *inside the collection its `Pane_state` names* (a graph in
some other collection is `Wrong_collection` at the exporter and a refusal here), a first
node on both (a node-less graph cannot be navigated to, since `selectNode` is the only way
into a pane), well-formed mapping entries, array overlays. There is no fall-through from an
unresolvable comparison to `defaultView`.

The candidate then waits on `(graph, paneIndex)` twice: connect hidden with the left graph
and the full config, await `(left, 0)`, issue `selectNode(right.firstNodeId, right.graph,
right.collectionLabel, 1)`, await `(right, 1)`, install node data per pane, resolve.
`entry.safe` means *no processing this renderer asked for is outstanding* — the
precondition `#release` asserts and `#park` branches on — so it is true at the pane-0 event
and **false again immediately before** the pane-1 request. That single toggle is what makes
the two cancellation windows behave differently:

- cancelled *between* the two events ⇒ unsafe ⇒ quarantined, and the late pane-1 event
  releases it;
- cancelled *before* the pane-0 event ⇒ the event finds the entry cancelled, releases it,
  and never asks for a second pane — no quarantine slot is spent on an event nobody awaits.

Deleting `entry.safe = false` turns the first into a removal of an element that is still
processing; both are pinned in `renderer-unit`, and both were watched to fail without it.
The processing deadline **restarts** for the second pane (each is its own layout round) while
`startedAt` does not, so the heartbeat keeps reporting total elapsed. Heartbeat, capacity,
supersession, quarantine, `abortReady`, `finalize` and `markInconsistent` are untouched.

**Config translation is translation, never inference.** `disableMappingFallback` is
`sync.matchNodeIdFallback !== true` and nothing else (§20); `showDiffHighlights` is emitted
only when declared; mapping entries cross as **arrays** (`{leftNodeIds, rightNodeIds}`),
because expanding a correspondence *component* into pairs would turn one claim into a
Cartesian product of claims.

**The overlay wrapper is the adapter's, and that asymmetry is deliberate.**
`Comparison.overlays_left/right` are bare `Model_explorer.EdgeOverlay.t list`, but
`edgeOverlaysDataListLeftPane` takes `EdgeOverlaysData[]` — `{type, name, graphName,
overlays}`. The browser wraps, per pane, and omits the key entirely for an empty list —
which is every session produced today, since the one real overlay rides on the kernel
graph's own `tasksData`. Moving the wrapper onto the wire is the principled shape and is
**not** taken while no comparison populates one; it is the obvious follow-up if one ever
does.

**What the chrome may say.** `panels.js` renders the comparison selector from
`Session.comparisons` alone — no capability is consulted — with `Comparison.label` verbatim,
plus one leading "Single view" entry that exists only because a control able to enter
compare mode must be able to leave it. Pane headings come from `viewByGraph`: a comparison
names only `{collection, graph}` and its own label is prose, so a heading is a **lookup** of
the stage view declaring the same graph (falling back to the bare graph id), never a graph
id built from a stage name. They are rendered only for a comparison that is on screen — a
heading written while a replacement was still being prepared would label the graph still
showing with the name of one it may never be replaced by. The summary states the exact
component count and never claims completeness, and for an empty mapping says so outright
before saying which of §20's two regimes applies.

## 20. The flow spine as a destination — `Me_flow.State.view`

Phase 4 of `web-ui-4.md` makes the exported spine navigable. The first half of that
is a wire contract: **a flow state names the view it opens, explicitly.**

`Me_flow.State.t` carries `view : string`, required rather than derived, and
`Me_session.validate` checks that it resolves, that it is a `View.Stage` (never
`Flow` or `Compare`), and that the view's graph is the state's own.

**Why not look the view up by graph.** It is ambiguous *by contract*, not merely in
some hypothetical export. Nothing in `Session.validate` forbids two stage views over
one graph, and `View.kind`'s own docstring contemplates exactly that — canonical is
"a VIEW naming the final Native state's graph, not a separate projection, and when
every pass and the pack are identities that graph is the initial one". A lookup that
happens to be unique in today's output would be a rule the document does not
guarantee. Matching on label or layer would be inference of a cruder kind. So the
producer says which view, and the validator checks it meant it.

**Why the check lives in `Me_session` and not `Me_flow`.** Identical reasoning to
`Transition.comparison`: `Me_flow.validate` establishes the rooted-DAG contract and
cannot see the view table, so a rule about views cannot be stated there. `Me_flow`
owns DAG validity; `Me_session` owns every rule that crosses two tables.

Three tags, each carrying data rather than prose, with the multi-field payload in
its own module so `state`/`view` labels cannot collide (CLAUDE.md's
record-namespace rule, not warning 30 silencing):

| Tag | Payload | Means |
| --- | --- | --- |
| `` `Flow_view_unknown `` | `Flow_state_view.t` | the id resolves to no declared view |
| `` `Flow_view_not_stage `` | `Flow_state_view.t` | it resolves to a `Flow` or `Compare` view |
| `` `Flow_view_graph_disagrees `` | `Flow_view_graph.t` | a stage view, but over another graph |

The third carries **both** graphs, because the check is that they are equal —
naming one would not say which side was wrong, and the ids alone would have to be
looked back up in two tables to mean anything.

**No fallback for older documents.** A client-side graph-to-view map as a
compatibility shim would reintroduce exactly the ambiguity this field exists to
remove. A session predating the field simply has no navigable flow.

The exporter populates it from the stage view it built for the same graph, and the
pairing is pinned end-to-end in `test/me_visualize_json_cram.t` — the five states
against `v/source`, `v/initial`, `v/canonical`, `v/stage_program` and `v/kernel`,
each with its graph agreeing — so a renaming on either side is a test failure rather
than a silently unnavigable spine.

### 20.1 The flow graph — `Me_flow_graph`

The spine renders as a **bipartite** graph: one node per state, one per transition,
edges `before -> transition -> after` and nothing else. Bipartite because Model
Explorer has no edge-selection event and `subgraphIds` is a node field — a
transition has to *be* a node for selecting it to offer its comparison.

`Me_flow_graph` is its own module rather than a case inside a dialect projector:
this record shape has nothing in common with a `Graph_ir` walk, and `Me_build` is
functorised over exactly that walk.

**Every node carries one output slot, leaves included.** `Session.validate` reads a
node's slot count as `List.length outputsMetadata` and rejects an edge into a slot
that does not exist, so a leaf without one does not merely render oddly — it makes
the whole session invalid.

**Two ceilings are checked before anything is allocated, and that ordering is the
bound.** `Me_flow.validate` runs first, because `max_states`/`max_transitions` live
there and every walk below is linear in them. Then the projected counts —
`states + transitions` against `max_nodes_per_graph`, `2 * transitions` against
`max_edges_per_graph` — because those two are **wire-selectable independently** and
*nothing downstream enforces them*: `Session.validate` counts graphs (`Field.Graphs`)
and the session-wide `int64` `Total_nodes`/`Total_edges`, never a single graph's.
Skipping this would make the flow graph the one graph in the session on which
`max_nodes_per_graph` means nothing. Every other projector (`Me_build`, `Me_source`,
`Me_kernel`) owns the same two checks for the same reason.

A transition names its comparison in an attribute **only when it has one**. An
absent comparison is honest absence, never an identity claim, so it gets no
attribute rather than one reading "none".

### 20.2 A flow destination is three agreeing facts

`Session.validate` binds together what nothing bound before:

- the spine's own `flow.graph`;
- exactly one `View.Flow`, naming that graph — resolved **in the collection that
  view declares**, so a graph living in another collection is `Wrong_collection`
  exactly as it is for a view or a pane;
- `feature:flow` `Available (Graph flow.graph)`.

and, symmetrically, `flow = None` admits **neither** a `View.Flow` nor an
`Available` `feature:flow`. *A capability alone never creates a destination* —
that was already the browser's rule, and it is now the document's.

Before this, a session could carry a spine whose graph no collection held, a flow
view pointing elsewhere, or a capability advertising a third graph, and the browser
would have been the first thing to notice. The client's defensive index remains, but
as defence in depth rather than as the boundary.

`Flow_destination.part` is a closed variant (`Flow_view` / `Flow_capability`) rather
than prose in a message, and `named` is `None` when that part is absent entirely
versus `Some g` when it named the wrong graph — the two are different defects and
the payload keeps them apart. Each of the six rules was individually neutered and
watched to turn its test green before being restored.

The exporter adds the graph to the same collection as the stage graphs and declares
`v/flow` "Export flow". **It is never the default view**: the browser stays
source-first and the CLI keeps canonical Native.

### 20.3 The selected-node event, as measured

`web/src/flow.js` + `web/test/flow.spec.ts` are `web-ui-4.md`'s browser
prerequisite, against the bundle §17 builds. The flow adapter routes a node
selection to a declared destination, so it must separate a user's click from the
shell's own setup `selectNode`. **`NodeInfo` carries no origin bit** —
`{nodeId, graphId, collectionLabel, node?, paneId?}` — and both paths reach the same
`AppService.selectNode`, so identity and order are the only separators that exist.

Three suppression designs were wrong before this fixture was written, which is why
it *enumerates* the sequence instead of confirming an assumed one:

1. **a barrier armed at finalize on the setup node id** — unsound: a setup event
   delivered *before* finalize leaves the barrier armed, so
   `setup A · user B · user A` discards the user's genuine A;
2. **a count of issued `selectNode` calls** — unsound for the reason below;
3. anything that assumes the sequence without measuring it.

What the element actually does, measured in Chromium:

| Observation | Result |
| --- | --- |
| events per `selectNode` call | **two** — a clear (`nodeId: ""`), then the target |
| ordering | the clear always precedes its target |
| auto-select on graph load | **none** |
| a replaced candidate's later selection | still emitted |
| `removeEventListener` | stops delivery |
| a foreign graph's selection | names its own `graphId`/`collectionLabel` |

The two-events-per-call result is why a count cannot work: one call, one budget, and
the budget is spent on the clear while the target escapes. It comes from
`selectNode` clearing before revealing —
`appService.selectNode(paneId, undefined)` at `model_graph_visualizer_base.ts:443`,
then `setNodeToReveal` → `revealNode` → `handleSelectNode`
(`webgl_renderer.ts:3582-3606`) — each feeding the same signal
(`model_graph_visualizer_base.ts:179`).

So the adapter's rule is **identity-based**: an empty `nodeId` is not a selection at
all and is never routed by either listener, and the target is suppressed by matching
the `{nodeId, graphId}` the renderer itself asked for, once. "No auto-select on load"
retires one of [D2]'s two residual cases outright; the other — `revealNode`'s
un-rendered branch completing without selecting — costs at most one missing action
row under select-then-act, never a wrong navigation.

"A replaced candidate still emits" is the measurement that makes listener removal
**mandatory** rather than tidy: the element does not go quiet on its own, so
authority has to be taken away explicitly.

### 20.4 The adapter — select-then-act

**Selection and action are two steps, and that is a correctness choice as much as
a usability one.** `AppService.selectNode` is reached by an ordinary single click
(`app_service.ts:491-543`), so routing on selection alone would make a mis-click
destructive and put a flow node's own attributes permanently out of reach. It also
lets the unpaired-transition case be a *visible statement* — "No exported comparison
is available for this transition." — rather than a click that silently does nothing.
`web-ui-4.md`'s routing table was amended to match.

The consequences run through the whole design: a spurious or missing selection costs
a stale or absent action row and **never a navigation**, which is what makes [D2]'s
two residual cases tolerable rather than blocking.

A state offers the stage view its `State.view` names; a transition offers its
`Transition.comparison` by that comparison's own label, or nothing at all. No graph,
label, layer or endpoint is ever consulted to invent one.

**Three sites in `app.js` knew only two branches**, and each failed *silently* for a
flow rather than loudly:

| Site | Failure without the third arm |
| --- | --- |
| `changePresentation` | sends `{view: 'v/flow'}`, which the strict stage route refuses — the page reports a failure for a destination it declares |
| `onSession` | `?flow=v/flow` loads, resolves, and leaves the **source stage** on screen under a flow URL |
| `preferFor` | unchanged, but the source-stage load is the *bootstrap* presentation, never the destination |

A load resolves a **view**, so neither a comparison nor a flow can be reached by
`prefer`; both are opened on top of the installed document, replacing the URL just
written rather than pushing — one navigation, one history entry. Two browser tests
exist purely for this boundary, because clicking the control on a live session never
reaches it: a direct `?flow=` load, and back/forward into and out of a flow.

One CSS rule earns a mention because it bit here: **`[hidden]` is a UA rule, so any
author `display` on the same element silently beats it.** `.panel` and the grid rows
were all un-hiding themselves; an empty container merely *looked* hidden because a
zero-height box reads as invisible. `[hidden] { display: none !important; }` is
stated once rather than remembered at each new container.
