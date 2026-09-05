# Project design ideas from the LSTM review

These proposals come from reviewing the LSTM implementation plan, its historical notes,
`lstm-review.md`, `CLAUDE.md`, the relevant `.ai/` design records, and the current source.
They describe changes that would have reduced the work or prevented defects exposed by
LSTM. They are proposals, not implemented capabilities or additional LSTM prerequisites.
The [implementation plan](lstm-implementation-plan.md) defines the required landing scope.

The existing separation is useful: `Expr` owns a tensor-independent language, operation
modules own arithmetic, and Region programs express computations shared across coordinates.
Keep those boundaries. Most of the friction comes from contracts that are documented but
not enforced by types or shared implementation, and from treating a compact expression as
evidence that execution or verification is cheap.

## 1. Make each Region local's definition determine its shape

Evidence: [region_local.ml](../lib/native/region_local.ml) stores `{ id; shape; value }`,
where every RHS is an `Expr.Value.t`. [region_eval.ml](../lib/native/region_eval.ml)'s
`evaluate_locals` dispatches on slot count, so a vector of extent one takes the scalar
branch without its reducer bound. [region_execution.ml](../lib/native/region_execution.ml)
already dispatches on declared shape, but that fix did not reach the reference path.
A scan also cannot be represented faithfully by the current expression-only RHS.

Replace independent shape/value fields with a closed RHS variant: scalar expression,
vector body with its extent and binder, or scan descriptor. Derive the read kind and
storage extent from that variant. Compile it into a typed slot layout whose cases retain
the distinction between scalar, vector and trace; do not infer semantic kind from an
offset/count pair. Render the RHS itself, exposing scoped `init` and `update` children for
a scan. [me_detail.ml](../lib/model_explorer_export/me_detail.ml) and
[region_trace.ml](../lib/native/region_trace.ml) currently assume an expression root.

This would have made the extent-one bug harder to express and given scans a clear place
in checking, execution, substitution, and rendering. Start with the RHS and layout
representation; preserve independent reference and production evaluation loops, but share
the small layout/read contract. Verify scalar versus extent-one vector behavior through
both `materialize` and `value_at`, plus an unspecialized trace rendered in the explorer.

## 2. Make validation produce the artifact that execution consumes

Evidence: [region_execution.mli](../lib/native/region_execution.mli) calls `lowered`
already validated, yet both `lower` and `lower_region` accept only a program. Shape is
supplied later to execution. [stage_program.ml](../lib/native/stage_program.ml)'s `ground`
materializes stages without a complete preflight. This leaves no place to guarantee a
shape-dependent per-key cost before execution starts.

Use one result-valued preparation boundary that takes the program, output shape and
immutable limits, then returns an abstract executable artifact. Store its validated
layout, shape and resource measures. Every public constructor, including a convenience
constructor for a known Region program, must establish the same invariant. Pixel and
Region execution both need the selected limits. Derive fresh mutable meters at the
execution scope rather than storing them in the artifact.

Perform any cost-changing result conversion before validation, or revalidate afterward;
[kernel_eval.ml](../lib/native/kernel_eval.ml)'s `converted` currently changes the emitter
before lowering. Thread the same configuration through symbolic construction:
[eval_symbolic.ml](../lib/native/eval_symbolic.ml) currently selects default limits even
when the surrounding Region Kernel path receives custom ones.

This would have turned much of LSTM's entry-point audit into a constructor migration.
Test that raw programs cannot manufacture the executable token, every constructor rejects
the same over-limit case, wider custom limits reach construction and execution, and two
independent `value_at` calls each receive a fresh allowance.

## 3. Treat scope as part of the traversal interface

Evidence: the recursive language boundary is already isolated in
[expr_repr.ml](../lib/expr_internal/expr_repr.ml), but
[fold.ml](../lib/expr_internal/fold.ml), [rewrite.ml](../lib/expr_internal/rewrite.ml),
[value.ml](../lib/expr_internal/value.ml) and the printer each encode binder behavior.
Existing binders are reducers; a scan adds a locally bound previous-state reader and
different scopes for initialization and update. A global list of allowed binders cannot
express those per-child rules.

Introduce a small internal scoped-child interface that exposes each child with the
reducer and local binders in scope there. Build free-variable analysis and structural
rewrites on that interface, while keeping compare/hash/printing's lexical environments
explicit. Extend freshening to both namespaces. Keep the existing recursive representation
boundary and tensor-independent public library; a generic compiler framework is unnecessary.

For composition, provide a clearly named fragment-import operation that freshens bound
identities and avoids free identities using the receiving builder's supply. Document that
import happens before insertion. The existing `Rewrite.freshen` documentation already
explains why freshening after accidental capture cannot recover the original intent.
Longer term, generative fragment namespaces could make unsafe independent composition
unrepresentable, but they would require a broader API migration.

This would have reduced the number of independent places where LSTM must teach the
project about `lane`, `step`, and `prev`. Tests should cover valid captures of earlier
locals, rejected init-only scope violations, independent fragments with colliding ordinals,
and alpha-equivalent compare/hash behavior across both binder namespaces.

## 4. Give every resource limit an explicit unit and owner

Evidence: [region_program.ml](../lib/native/region_program.ml)'s `checked_slot_total`
bills storage against the expression `max_size` limit. Conversely,
[region_eval.ml](../lib/native/region_eval.ml)'s materializer retains a slot array for
every key, so a per-key storage limit does not bound its total retained scratch.
[kernel.ml](../lib/native/kernel.ml) already supplies useful precedents: checked limits,
exclusive hard ceilings, and ceilings justified against native and JavaScript execution.

Separate syntax nodes/depth, local storage slots, peak live scan state, per-key updates,
Kernel-wide updates, retained proof nodes, and cumulative grounding construction. For each,
state the unit, aggregate formula, owner/reset scope, boundary rule, payload and default.
Use one checked arithmetic utility for bounded sums/products before narrowing to `int`.
Share immutable profiles and charge operations, rather than merely using matching field
names in separate components.

Static admission should report conservative cost before execution; runtime meters should
bound raw expressions and dynamically composed contexts that admission cannot fully see.
Stream reference execution by key when its memory contract assumes one live slot array.
Keep update counts distinct from elapsed time: an update can contain an expensive reduction.
General reduction-work metering would be a separate future task.

This would have prevented slot/state double-counting, missed emitter/vector multipliers,
and reset-scope ambiguity in the LSTM plan. Validate exact-limit/next-charge behavior,
scan-free programs with scan limits disabled, nested row-zero state use, and observed
charges across all locals and emitters. A preflight rejection alone cannot prove that
runtime calls shared one meter; instrumentation must observe a successfully executed case.

## 5. Preserve sharing when transforming recurrent computations

Evidence: [region_program.ml](../lib/native/region_program.ml)'s `specialize_pixel`
substitutes local definitions into an expression. This preserves scalar semantics, but a
cached trace read replaced by inline scan evaluation can replay an earlier scan inside
every later update. Compact AST size does not expose that execution multiplier.
Separately, [ground_expr.ml](../lib/native/transform/ground_expr.ml) is a tree whose size,
cell collection and projection recurse through repeated pointers. A coupled recurrence
can therefore have small allocation count and exponentially large logical tree size.

The near-term change is bounded specialization with post-rewrite cost measurement, plus
grounding budgets covering both roots and all construction. This is required in the LSTM
plan. A further improvement is an explicit let/DAG representation for shared ground terms,
or a verifier interface that consumes Region definitions without erasing their sharing.
Represent scan projections against one definition rather than copying its recurrence.

That larger change must make every relevant consumer sharing-aware: comparison, hashing,
normalization, projection, cell collection, evaluation and rendering. Node identity must
not become proof of equality across graphs. Preserve source/destination provenance and
explicit rounding/materialization boundaries. Simply memoizing construction or adding a
pointer cache would not establish either bounded work or sound verification.

This would have made LSTM verification useful on larger cases and reduced repeated
specialization. Keep it separate from the landing until it has dense recurrence benchmarks
and checks against small unrolled reference terms, including differing constants and rounds.

## 6. Preserve typed failures through every subsystem boundary

Evidence: [region_context.ml](../lib/native/region_context.ml)'s `program` discards the
entire `Region_program.error` payload in favor of `Invalid_program`, which
[region_computation.ml](../lib/native/region_computation.ml) propagates without detail.
Existing scalar/vector callbacks in [eval.ml](../lib/expr_internal/eval.ml) return options,
which cannot distinguish unknown ids from invalid projections. The review also found an
unnamed new grounding limit with a contradictory verdict mapping.

Use result-valued lookup contracts where failure categories matter. Define named payload
records and closed field/kind variants once, then wrap or widen those errors with
`Err.map_error` at subsystem seams. Keep the originating payload and error wrapper until
the deliberate reporting boundary. A limit error should identify its resource and carry
the configured ceiling; an invalid coordinate should retain the coordinate and extent.
Use `Err.Escape` when a result must cross a callback that cannot return one.

For public budget changes, make the migration part of the contract: exact record field,
profile values, reset semantics, error constructor, verdict mapping, printers and label
vocabulary. Derive shared defaults from the lower-level owner. A future abstract verifier
budget with constructors/profile updates could reduce record-literal migrations, but is
not necessary to resolve the current review.

This would have let LSTM integration preserve precise failures without reopening each
error boundary. Verify payloads through the operation API, not only at the inner helper;
exercise all verdict labels and the distinction between pair and construction exhaustion.

## 7. Distinguish output contracts from computation sharing

Evidence: the graph already represents multiple outputs and evaluates each ordinal, as
implemented in [graph_ir.ml](../lib/native/graph_ir.ml) and
[eval_direct.ml](../lib/native/eval_direct.ml). Yet Region's `check_output` accepts only
ordinal zero. The ATen verifier's fixed-tuple policy deliberately permits fewer outputs,
so it cannot by itself establish that an LSTM bridge exposes all three results.

First, make output shape/arity validation explicit per operation and ordinal. Add a
strict full-output oracle mode or operation-level assertion for contracts such as LSTM's
`output`, `h_n`, and `c_n`, while retaining the existing legitimate dropped-output policy
where needed. Derive deadness from use analysis, not exported names.

Then consider a separate, larger extension: an operation computation group with shared
locals and multiple named emitters, each with its own output shape/projection. All LSTM
outputs derive from the same traces. The planned one-program-per-output approach is
correct, but can compute those traces repeatedly; a computation group could share them
and calculate only requested outputs. Its scheduler, liveness analysis and resource
accounting must agree on what executes once and what executes per output.

This would reduce both three-output integration risk and redundant recurrence work.
Do not require the computation-group extension for LSTM. Establish full arity first;
evaluate the larger proposal with all outputs live, one live, and discarded state outputs,
checking both results and scan-start counts.

## 8. Make binding feasibility an executable check

Evidence: the minimal ATen archive deliberately includes throwing stubs in
[atg_stubs.cpp](../lib/aten/atg_stubs.cpp). Binding generation and successful linking do
not prove that an operation's runtime path is available.
[aten_walk_gen.ml](../lib/aten_gen/aten_walk_gen.ml)'s default recipe also excludes
`Tensor[]` inputs, so LSTM requires a deliberate parameter-list generator.

Give a new binding an executable smoke fixture before building its Native implementation:
construct valid operands, execute it against this archive, and inspect every output's
count, shape and representative values. Reuse that fixture as the smallest oracle/walk
configuration. Keep tests requiring ATen in the existing ATen test layer, preserving the
pure Expr/Native and browser build boundaries.

This would catch a missing runtime dependency or tensor-list recipe while the investigation
is still small. For LSTM, include nonzero initial states and unequal batch/sequence/hidden
dimensions so layout errors are observable, then expand coverage to layers, directions,
biases and inactive inference dropout.

## 9. Turn extension inventories into repeatable checks

The historical notes repeatedly corrected textual call counts, missed public constructors,
and compile-boundary assumptions. Those are process failures that project tooling can
reduce; preserving the sequence of mistaken counts does not help implementation.

Maintain a short extension checklist by responsibility: recursive representation, public
construction, binder-aware traversals, evaluators, rewrites, validation, budget profiles,
diagnostics, rendering, importers and backend tests. Generate or re-enumerate concrete sites
when starting a change. Use exact application searches and complete compiler feedback;
do not treat a truncated text search as a caller census. Prefer symbol-aware reports if
they become available within the existing development tooling.

Add a small cross-library compile fixture using only public `Expr` APIs to construct,
check, print and execute a Region program. The existing namespace tests and Dune library
boundary are good foundations. Include malformed/error cases and cost instrumentation,
since numeric agreement does not establish sharing or budget enforcement. Keep both
JavaScript checks: native-versus-JS agreement and inline tests against committed goldens
catch different failures.

Working plans should state the current contract, unresolved measurements and exit criteria.
Durable decisions belong in `.ai/` with their implementation. Remove superseded proposals
and commit-status notes from active instructions, retaining only rationale that still
explains a constraint. This would have made the LSTM plan shorter and harder to misread as
already implemented.

## Suggested priority

| Priority | Work | Relation to LSTM |
|---|---|---|
| Required with the scan foundation | Typed RHS/read errors, validated preparation, scope handling, separate budgets, bounded specialization/grounding, public API and runtime checks | Already reflected in the implementation plan |
| Required with the operation | Three-output validation, executable ATen probe/recipe, full configuration fixtures | Establishes the actual supported domain |
| Incremental project improvements | Shared layout/scoped traversal helpers, fragment import, repeatable inventories and extension fixtures | Reduce future integration work; keep abstractions small |
| Separate design work | Sharing-aware ground IR and multi-output computation groups | Improve proof scalability and avoid repeated traces; need broader consumer changes |
