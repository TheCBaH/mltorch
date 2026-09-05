# Project execution checklist

Status: planned. This document orders the work from
[lstm-implementation-plan.md](lstm-implementation-plan.md) and
[project_design_ideas.md](project_design_ideas.md). The implementation plan owns the exact
scan, error, budget and LSTM contracts; the design ideas explain the broader changes.
Use [lstm-plan.md](lstm-plan.md) for the detailed tensor/arithmetic contract where the
implementation plan refers to it. Follow `CLAUDE.md` throughout.

Execute the steps in order. Steps 1–11 deliver the reusable scan foundation; steps 12–16
deliver LSTM; steps 17–20 cover subsequent project improvements. The later improvements
are not LSTM prerequisites. A step is complete only when its changes and acceptance
evidence exist; writing this checklist does not complete any implementation item.

The order deliberately puts reusable project improvements before scan implementation:
step 5 validates better representation, scope, error and execution boundaries on existing
scalar/vector operations. Steps 6–10 extend those foundations with concrete scan requirements.
This costs an initial refactor and a second extension of some interfaces, but isolates
behavior-preserving changes from new recurrence semantics. Keep that preparatory work small.
Sharing-aware verification and multi-output scheduling follow the LSTM landing because their
larger design choices benefit from an executable workload and measured costs.

Each step may span several commits. Keep public signatures, exhaustive matches and their
caller migrations together so intermediate commits build. Steps 5–10 form one foundation
deliverable: do not claim scan support until step 11 passes. Record durable decisions in
the tracked `.ai/` design record with the implementation; keep this execution ledger in
the separate `_ai_/` repository.

## Foundation

### 1. Establish the baseline and current migration inventory

- [x] Inspect both repositories' status and preserve existing edits. Re-read relevant `.ai/`
  records and verify that the source assumptions in the two planning documents still hold.
- [x] Establish the current build/test baseline and record any pre-existing failures separately.
- [x] Enumerate application sites for `Expr.Eval.value`, `Region_execution.lower`/
  `lower_region`, `Stage_program.ground`, `Ground_eval.at`/`expand`, and limit constructors.
  Include public signatures, full budget literals, printers, labels and explorer consumers.
- [x] Locate existing Expr, Region, verifier, ATen and JavaScript fixtures that subsequent
  steps can extend. Check file-size constraints before choosing new module boundaries.

Completion: a current inventory by responsibility and a reproducible baseline. Use complete
searches and compiler feedback; historical caller counts are not acceptance criteria.

Evidence (2026-09-05): the root worktree was clean and the ledger repository's pre-existing
untracked `lstm-review.md` was left untouched. `make precommit` passed with no pre-existing
failures. Reviewed the expression, Region execution, error, Native compute, and JavaScript
design records alongside both LSTM plans. The current migration inventory is rooted in
`region_eval`/`region_execution`, `eval_direct`, `kernel_eval`, `schedule`, `stage_program`,
`ground_eval`, `map_verify`, `kernel`/`kernel_adapt`, their public interfaces, and the Native4D
direct path; the complete source search also identified the focused Expr, Region, grounding,
Native, Native4D, ATen, and JS fixture directories. Existing files remain below the repository
file-size ceiling, so later modules should continue the existing split-by-responsibility pattern.

### 2. Fix extent-one vector evaluation

- [x] Change `Region_eval.evaluate_locals` to dispatch on the declared local shape, matching
  the rule already used in `Region_execution`.
- [x] Add an extent-one vector fixture whose body actually reads its reducer, plus an SDPA
  `Wk = 1` regression through `value_at` and materialization.
- [x] Verify both reference and production execution. Observe the regression fail with the
  defective dispatch restored, then restore the fix.

Completion: scalar and extent-one vector locals remain distinguishable and the regression
reaches the previously failing path. Keep this correction separate from the structural refactor.

Evidence (2026-09-05): `c1f05f4` changes the reference loop to select by
`Region_local.Shape.t`, retaining the production loop's rule. The Wk=1 SDPA test now runs the
production graph, direct reference materialization, and direct reference `value_at`, each
bitwise against the legacy oracle. With the former slot-count dispatch restored temporarily,
the focused test failed at the reference materializer with `Unbound_reducer`; the declared-shape
dispatch was restored. `NO_COLOR=1 opam exec -- dune runtest test/native/region_compute_test.exe`
and `make precommit` pass.

### 3. Prove ATen LSTM feasibility

- [ ] Build a small probe for `aten.lstm.input` against the project's minimal ATen archive.
  Execute the binding to expose any reachable throwing stub or missing kernel dependency.
- [ ] Supply nonzero initial states and unequal batch, sequence and hidden dimensions.
  Assert exactly three outputs and inspect their shapes and numeric results.
- [ ] Record the binding/build requirements and retain the probe as the seed for the later
  oracle and tensor-list walk recipe.

Completion: an executable three-output oracle exists. Resolve any runtime dependency gap
before beginning LSTM arithmetic; successful linking alone is insufficient.

### 4. Finalize the foundation contracts and resource measurements

- [ ] Record the recursive scan representation, both projections, per-child scopes,
  freshening obligation, exact callback/error declarations and meter reset rules from the plan.
- [ ] Specify the typed Region RHS/layout and validated execution artifact, including all
  public constructors and error propagation through `Region_context`/`Region_computation`.
- [ ] Preserve dependency direction: `Expr` cannot depend on Native. `Kernel` already owns
  `Region_program.t`, so Region preflight must accept lower-level limits or explicit fields,
  rather than introducing a `Region_program` → `Kernel` cycle.
- [ ] Produce the three resource estimates: maximum updates per key, summed Kernel work
  after output liveness/selection, and Direct work across every materialized output.
- [ ] Measure slot/state allocation and practical ceilings natively and under Node using
  synthetic programs. Choose scan defaults and hard ceilings with stated headroom; the
  proposed local-slot default is 8192. Recheck estimates against actual LSTM programs in step 16.
- [ ] Adopt the plan's separate retained-pair and cumulative grounding accounts, including
  exact profile values, term/meter ownership, verdict mapping and public record migration.

Completion: no unnamed limit, unit, reset scope or error payload remains in the foundation
contract. Label estimates and initial policy values honestly; do not present them as benchmarks.

### 5. Strengthen the existing project foundations

- [ ] Replace independent Region `shape`/`value` fields with scalar and vector RHS cases.
  Derive read kind and storage extent from the RHS; add the scan case only in step 6.
- [ ] Introduce shared typed slot-layout/read helpers while retaining independent reference
  and production evaluation loops. Update Region folds, printers and explorer RHS rendering.
- [ ] Stream reference materialization by key; verify existing normalization/SDPA behavior
  and the extent-one regression before introducing recurrence.
- [ ] Preserve full Region construction errors through `Region_context` and
  `Region_computation`, using the existing `Err` boundary conventions.
- [ ] Establish result-valued preparation for current programs, retaining validated shape,
  existing limits and layout. Migrate both public lowering constructors and their callers;
  step 7 extends that invariant to the new resource dimensions.
- [ ] Factor a small scoped-child helper from current Expr traversals where it removes
  duplication. Keep lexical environments explicit. Provide a named fragment-import helper
  using the receiving builder supply and freshening before insertion.
- [ ] Extend external public-API and existing-operation fixtures to pin these contracts.
  Keep the pure library dependencies and numerical/materialization behavior unchanged.

Completion: existing scalar/vector operations validate the new structure and preparation
boundary without any scan arithmetic. Avoid adding an unused generic IR or speculative
capabilities; the next step supplies the concrete new constructor and scope requirements.

### 6. Extend those foundations with bounded scans

- [ ] Extend `expr_repr`, the internal scan implementation and public recursive signature.
  Export `Scan`, `Scan_limits`, `Scan_meter`, `Scan_admission` and both projection constructors.
- [ ] Add `Builder.scan` using the shared supply. Extend scope/import helpers, folds,
  freshening, substitution, checking, comparison, hashing, normalization and printing to
  local binders and the distinct initializer/update scopes.
- [ ] Implement admission and inline evaluation: immutable previous rows, ordered lane
  updates, projection bounds, shared update charging and internal state reservation/unwinding.
- [ ] Implement the exact `scan_reader`, projection errors, closed invalid-limit fields,
  missing-meter error and shared error conversions specified in the plan.
- [ ] Add the scan RHS/layout case and continuation-based `Region_program.Builder.scan`.
  Extend dependency order, local-kind agreement and region invariance checks to trace reads.
- [ ] Render unspecialized scan definitions with scoped `init` and `update` children.
- [ ] Cover freshening/capture, init scope, zero steps, row-zero projections, nested state,
  exact-limit/next-charge behavior, and raw scans under constant/dynamic reductions.
- [ ] Compile a consumer outside Expr that constructs and inspects a scan through public
  APIs; carry dependent signature and exhaustive-match migrations with the new constructors.

Completion: Expr builds independently, and scan semantics, scope and runtime bounds pass
focused tests. Region can construct/check/render trace definitions. Wider construction and
narrower evaluation limits exercise state rejection even when no updates occur.

### 7. Separate resource dimensions and validate executable artifacts

- [ ] Add `max_local_slots`, `max_scan_state`, `max_scan_updates_per_key` and
  `max_scan_updates_total` with checked construction, hard ceilings and consistent defaults.
- [ ] Use bounded arithmetic for all aggregates. Implement the plan's occurrence-based
  measures, including vector extents, emitter multiplicity, enclosing reductions and key counts.
- [ ] Stop charging storage against syntax size. Count resident slots once in scan peak
  state, omit nonexistent trace row buffers, and give scan-free programs zero scan-state cost.
- [ ] Extend step 5's result-valued `lower`/`lower_region` preparation to validate and retain
  the new limits and measures. Cover both Pixel and Region branches.
- [ ] Revalidate converted emitters, expose no unchecked constructor for a validated token,
  and run preflight from the operation, Stage and Kernel boundaries.
- [ ] Preserve the full typed program/limit failure through operation-facing errors.
  Apply total-update aggregation at Kernel scope only.

Completion: every preparation entry rejects the same relevant invalid program before scan
execution. Tests distinguish storage, state, per-key and Kernel-total failures, including
scan-free programs with scan limits disabled and exact error payloads at the operation boundary.

### 8. Execute Region traces and propagate configured meters

- [ ] Run each trace descriptor once per key, writing directly into its validated slot range.
  Charge each lane update before evaluating its body; cached projections spend no updates.
- [ ] Share one meter across all locals and emitters for a Region key. Allocate a fresh
  meter per standalone value, `value_at` invocation, and Pixel output coordinate as specified.
- [ ] Extend the streaming reference materializer from step 5 to traces, retaining one
  slot array at a time and verifying the larger storage workload.
- [ ] Thread limits through `Eval_symbolic.run`, `Region_kernel.of_graph`, Direct execution,
  Stage grounding and both Kernel Pixel paths. Remove silent default substitution.
- [ ] Make `Stage_program.ground` preflight all stages before materializing the first and
  return `Err.t`. Use `Err.Escape` across `Schedule.ground`'s materialization callback.
- [ ] Complete all signature/caller migrations, including Native4D's Direct path and tests.
- [ ] Extend optional counters to observe scan starts and exact combined update charges
  in successfully executed programs, separately from preflight-rejection tests.

Completion: reference and production results agree; counters establish sharing and reset
scope. Repeated same-key `value_at` calls remain independent, custom limits survive every
path, and many-key materialization respects the one-array scratch assumption.

### 9. Bound specialization and reject unsupported fusion

- [ ] Re-measure the final expression after `specialize_pixel`, `substitute_loads` and
  `substitute_locals`; propagate scan limits and typed failures to callers.
- [ ] Keep scope-preserving renaming and simple index/source rewrites cheap as specified.
- [ ] Add a chained-scan case whose cached Region execution fits but whose specialized
  replay exceeds its update allowance; reject before replay begins.
- [ ] Add a recurrent-computation summary to shared fusion admission. Exercise both planner
  and direct `Kernel_elab.admit` entry points, including a singleton inline scan.

Completion: compact syntax cannot bypass execution-cost admission after substitution, and
both fusion entry points reject recurrent computations consistently.

### 10. Bound grounding and migrate verifier budgets

- [ ] Implement `Ground_eval.Budget`, `Meter` and sized `Term`, threading one meter through
  roots, `body_at`, ordinary reductions, max-pool, scan unrolling and expansion.
- [ ] Charge retained roots/replacement deltas against `max_nodes`; charge cumulative
  construction, discarded state and repeated cached embeddings against `max_ground_nodes`.
- [ ] Process left then right explicitly. Share accounts within an attempt and reset both
  for each Structural/constant-bound attempt. Stop before a crossing insertion and prohibit
  probing partial frontiers after exhaustion.
- [ ] Migrate all `Ground_eval.at`/`expand` callers, verifier budget records/profiles, the
  external full record literal, printers and goldens using the exact plan values/contracts.
- [ ] Map pair exhaustion to `Max_nodes limit` and construction exhaustion to
  `Max_ground_nodes limit`. Update the complete verdict/label vocabulary and its tests.
- [ ] Cover oversized roots, later expansion, shared-pair exhaustion, replacement at the
  exact pair cap, discarded intermediates, repeated embeddings and fresh-attempt allowances.

Completion: each account is independently exercised with the other sufficiently large;
both verdicts carry the correct configured limit. Existing non-scan grounding remains
correct under the newly explicit budgets.

### 11. Accept and land the reusable scan foundation

- [ ] Extend the external public-API fixture to build, check, print and execute a complete
  scan-backed Region program through Stage and Kernel paths.
- [ ] Run every Stage 1 exit condition in the implementation plan, including trace ownership,
  numerical agreement, scope, limits, specialization, grounding, fusion and rendering.
- [ ] Run the full native checks and both JavaScript checks. Review golden changes and
  confirm that the pure libraries gained no ATen/ctypes dependency.
- [ ] Publish the landed contracts in `.ai/` and update this checklist with evidence.

Completion: the scan foundation is independently usable and green before LSTM arithmetic
is added. No LSTM support claim is made by this milestone.

## LSTM implementation

### 12. Integrate the ATen binding and full-output oracle

- [ ] Turn the feasibility probe into the registered `lstm.input` binding and add its
  `Tensor[]` walk metadata/recipe.
- [ ] Require exactly three oracle outputs with ordinal mapping `0=output`, `1=h_n`,
  `2=c_n`. Keep legitimate dropped-output verification policies for other operations.
- [ ] Extend fixtures across layer counts, biases, directions, input layouts, nonzero
  runtime initial states and inference dropout values `0`, `0.5`, and `1`.

Completion: the real binding and fixture generator exercise all three outputs, including
configurations beyond the corpus's single-layer, bidirectional case.

### 13. Implement the full Native LSTM operation

- [ ] Add the validated layer/direction payload, optional bias pairs, layout tag, shape
  checks, codec, printing and operation walk configuration.
- [ ] Register graph operands, output shapes/arity, builder support, evaluation dispatch
  and output transfer. Keep LSTM out of `Const_ssa.allows`.
- [ ] Add one shared checked `divmod` helper using the existing index language; verify
  remainder bounds before relying on `Clamp_low`.
- [ ] Author one Region computation per output ordinal using width-`2K` state, ordered
  direction/layer scans, correct initial-state selection and completed prior-layer traces.
- [ ] Implement stacked layers, both direction counts and both layouts in this step.
  Preserve gate order, floating-point association and required inter-layer/result rounding.
- [ ] Validate positive static dimensions, matching states/parameters and all aggregate
  products. Reject unsupported training, projection, packed/unbatched and dtype cases.
- [ ] Compare all outputs against the oracle; include unequal dimensions, live final states,
  and configurations with and without biases.

Completion: Native supports the entire accepted inference domain. No single-layer-only
landing or corpus-only shape assumption satisfies this step.

### 14. Implement both import paths and output validation

- [ ] Add the recurrent ATen bridge and Native interpreter lowering modules.
- [ ] Decode tensor lists and validate raw ranks before right-alignment loses rank evidence;
  reuse shared semantic shape/configuration checks after decoding.
- [ ] Preserve all three real outputs, and use actual use analysis to identify dead ones.
  Reject malformed lists/flags with typed diagnostics at the importing boundary.
- [ ] Verify live state outputs individually and together, plus discarded-state fixtures.
  Check explicit output cardinality rather than relying on fixed-tuple verifier leniency.

Completion: both importers produce the same accepted semantics and informative rejection
behavior, with no silent truncation or name-based assumption about dead outputs.

### 15. Add the Native4D counterpart

- [ ] Reuse the axis-independent Native payload where possible; implement the Native4D
  registration, conversion and Region computation routing.
- [ ] Respect rank-3 placement on `H/W/C` with `N=T=D=1`, including layout-dependent sequence
  placement and layout-independent initial/final state indexing.
- [ ] Ensure dead-output cleanup occurs before Native4D conversion, which has no `Discard`.
  Preserve any live state results.
- [ ] Compare Native4D with Native and ATen across the accepted configuration matrix.

Completion: Native4D has the same supported LSTM domain, with explicit layout and live/dead
output coverage rather than a blanket recurrence rejection.

### 16. Verify the complete landing and record actual support

- [ ] Run operation walks, malformed-input cases and both Sequencer2D shape families.
- [ ] Reconcile measured trace slots, peak state, per-key counts and Kernel/Direct totals
  with step 4's estimates. Verify default admission and rejection under tighter limits.
- [ ] Record numerical and execution-count evidence separately from wall-clock benchmarks.
  Account for repeated computation across the initial one-program-per-output implementation.
- [ ] Run repository checks, both JavaScript checks, and the relevant ATen/Native4D suites.
- [ ] Regenerate `make pt2.json-model-support`; report the actual next frontier separately
  for Native, Native4D and Kernel. Update support ledgers only from that evidence.
- [ ] Update tracked design records and mark completed checklist items with their validation.

Completion: all accepted LSTM configurations are implemented and checked across the intended
paths. Removing the 36 LSTM blockers does not by itself establish whole-model support.

## Subsequent project improvements

### 17. Consolidate the reusable extension workflow

- [ ] Extract the proven small layout/read, scoped-child and bounded-arithmetic helpers
  where duplicate implementations remain; preserve public boundaries and independent oracles.
- [ ] Audit use of the fragment-import helper from steps 5–6; consolidate remaining unsafe
  composition sites and document the demonstrated rules for both binder namespaces.
- [ ] Turn the migration inventory into a reusable extension checklist, using complete
  searches and compiler checks rather than fixed call counts.
- [ ] Retain the external public-API fixture, observable resource tests and executable ATen
  smoke fixture as reusable extension examples.
- [ ] Consolidate durable contracts in `.ai/`; retire superseded instructions in working
  plans without deleting rationale that still explains a constraint.

Completion: a future expression/operation extension has a concrete path through construction,
scope, errors, resources, execution, rendering and backends without duplicating LSTM-specific code.

### 18. Design and implement sharing-aware verification

- [ ] Benchmark dense recurrence grounding and specialization after the bounded-tree landing;
  identify which repeated work or logical growth prevents useful verification.
- [ ] Compare explicit let/DAG ground terms with verification directly over Region definitions.
  Record the chosen representation, ownership, budget units and numerical/provenance contract.
- [ ] Migrate all consumers together: comparison, hashing, normalization, projection, cell
  collection, evaluation and rendering. Make sharing explicit rather than relying on pointers.
- [ ] Preserve source/destination identity and rounding boundaries. Check against small
  unrolled terms, including differing constants and materialization rounds.
- [ ] Retain bounded failure behavior and publish before/after proof coverage, memory and
  work measurements.

Completion: sharing reduces verified work without changing what constitutes evidence or
allowing node identity to prove equality. If measurements do not justify the migration,
record that decision and retain the bounded-tree implementation.

### 19. Design and implement shared multi-output computation

- [ ] Measure duplicate trace execution when all LSTM outputs are live after step 16.
- [ ] Specify computation groups with shared locals and multiple emitters, including per-output
  shapes, liveness, scheduling, conversion and resource-accounting rules.
- [ ] Implement shared recurrence execution and selected-output materialization through the
  relevant Native, Stage, Kernel and Native4D paths without changing graph output semantics.
- [ ] Verify all outputs live, one live and discarded state outputs. Assert scan-start counts
  as well as values, and preserve rounding at output materialization boundaries.

Completion: requested outputs reuse traces with correct scheduling and budgets. This step
does not depend on the ground-IR representation chosen in step 18; it follows it here as the
default priority order. Retain separate output programs if evidence does not justify the change.

### 20. Reassess the remaining broader proposals

- [ ] Evaluate abstract verifier-budget constructors/profile updates against actual migration
  costs; change the public API only if the benefit warrants the caller migration.
- [ ] Evaluate generative fragment namespaces if the import helper still permits recurring
  capture defects; specify compatibility before changing binder identity types.
- [ ] Assess general reduction-work metering separately from scan-update accounting, with
  its own workload evidence, unit and enforcement scope.

Completion: each proposal has an evidence-backed adopt/defer decision and, if adopted, a
separate concrete implementation plan. None silently expands the original LSTM domain into
training, packed sequences, projections or a general compiler rewrite.

## Verification and completion records

Use focused existing suites while implementing; run the required repository checks before
each commit as directed by `CLAUDE.md`. At the foundation and LSTM acceptance steps, include:

```sh
opam exec -- dune build lib/expr
NO_COLOR=1 opam exec -- dune runtest test/expr test/native
make precommit
make jsoo.runtest
make jsoo.inline-runtest
```

Run the relevant ATen, importer, Native4D and explorer tests for their changes. Regenerate
model support at step 16; downloaded-model inference remains a separately identified check.
Run non-promoting tests first and inspect golden changes before promotion. For a claimed
regression fix, demonstrate that the regression fails without the fix.

When marking a step complete, record the implementation commit(s), relevant test command(s),
result, and any remaining limitation in a short note under that step. Missing toolchains,
model data or unrun checks remain explicit limitations, not passing evidence.
