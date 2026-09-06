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

- [x] Build a small probe for `aten.lstm.input` against the project's minimal ATen archive.
  Execute the binding to expose any reachable throwing stub or missing kernel dependency.
- [x] Supply nonzero initial states and unequal batch, sequence and hidden dimensions.
  Assert exactly three outputs and inspect their shapes and numeric results.
- [x] Record the binding/build requirements and retain the probe as the seed for the later
  oracle and tensor-list walk recipe.

Completion: an executable three-output oracle exists. Resolve any runtime dependency gap
before beginning LSTM arithmetic; successful linking alone is insufficient.

Evidence (2026-09-05): `fb5da79` adds a binding-only `lstm.input` probe and executes
it through the minimal static-dispatch archive. It uses `T=2`, `B=3`, input width 4,
hidden width 5, nonzero `h0`/`c0`, four zero parameter tensors, and receives status 0
with exactly three tensors of shapes `[2,3,5]`, `[1,3,5]`, and `[1,3,5]`; the committed
golden records every value. Whole `RNN.cpp` pulled quantized static registrations (FBGEMM
and JIT schema parsing), so the archive now derives a sentinel-checked floating recurrence
projection from named upstream sections. The binding stays out of public importer/config/walk
generation until its operation-specific recipe lands. `NO_COLOR=1 opam exec -- dune runtest
test/aten_ops_test.exe` and `make precommit` pass.

### 4. Finalize the foundation contracts and resource measurements

- [x] Record the recursive scan representation, both projections, per-child scopes,
  freshening obligation, exact callback/error declarations and meter reset rules from the plan.
- [x] Specify the typed Region RHS/layout and validated execution artifact, including all
  public constructors and error propagation through `Region_context`/`Region_computation`.
- [x] Preserve dependency direction: `Expr` cannot depend on Native. `Kernel` already owns
  `Region_program.t`, so Region preflight must accept lower-level limits or explicit fields,
  rather than introducing a `Region_program` → `Kernel` cycle.
- [x] Produce the three resource estimates: maximum updates per key, summed Kernel work
  after output liveness/selection, and Direct work across every materialized output.
- [x] Measure slot/state allocation and practical ceilings natively and under Node using
  synthetic programs. Choose scan defaults and hard ceilings with stated headroom; the
  proposed local-slot default is 8192. Recheck estimates against actual LSTM programs in step 16.
- [x] Adopt the plan's separate retained-pair and cumulative grounding accounts, including
  exact profile values, term/meter ownership, verdict mapping and public record migration.

Completion: no unnamed limit, unit, reset scope or error payload remains in the foundation
contract. Label estimates and initial policy values honestly; do not present them as benchmarks.

Evidence (2026-09-05): the tracked design record adds the recursive-group placement, both
`Value.scan_at`/`local_scan_at` constructors, the `Scan`/`Scan_limits`/`Scan_meter` error and
API surface, the two-namespace scope rules, the Region `Builder.scan` continuation contract,
the `preflight`/`lower`/`lower_region` validated-artifact signatures, the static-measure
formulas, the grounding `Budget`/`Meter`/`Term` contract with its verdict mapping, the fusion
and rendering requirements, and the `divmod` encoding for Stage 2. A source re-survey against
the current tree (post-`c1f05f4`/`fb5da79`) found and corrected several line-number and scope
drifts from the working plan, most substantively: the `Region_context`/`Region_computation`
error-erasure the plan cites once actually recurs at five sites (one in `region_context.ml`,
four per-operation sites in `region_computation.ml`), and `Ground_eval` lives under
`lib/native/transform/`, not `lib/native/`. The three censuses are computed directly from the
corpus's two LSTM shapes: max per-key updates `2*16*192 = 2*32*96 = 6144`; summed worst-case
Kernel total across all 36 occurrences and up to 3 live outputs each, `12,976,128`; the Direct
path's total is recorded as not independently boundable (unbounded by downstream fan-out),
which is why `max_scan_updates_total` stays Kernel-scoped. Slot/state ceilings are backed by a
standalone array-allocation probe run both natively (`Sys.max_array_length =
18,014,398,509,481,983`) and under `js_of_ocaml`/node (`Sys.max_array_length = 536,870,911`),
confirming allocation at 8192 through 50,000,000 float elements succeeds on both backends with
no stack-overflow-style frontier to discover, unlike `Hard.depth`/`eval_recursion`; defaults
(`max_local_slots`/`max_scan_state` = 8192, `max_scan_updates_per_key` = 8192,
`max_scan_updates_total` = 16,000,000) are chosen with stated headroom over the censuses, not
presented as measured capacity. No production code changed. `make precommit` passes.

### 5. Strengthen the existing project foundations

- [x] Replace independent Region `shape`/`value` fields with scalar and vector RHS cases.
  Derive read kind and storage extent from the RHS; add the scan case only in step 6.
- [x] Introduce shared typed slot-layout/read helpers while retaining independent reference
  and production evaluation loops. Update Region folds, printers and explorer RHS rendering.
- [x] Stream reference materialization by key; verify existing normalization/SDPA behavior
  and the extent-one regression before introducing recurrence.
- [x] Preserve full Region construction errors through `Region_context` and
  `Region_computation`, using the existing `Err` boundary conventions.
- [x] Establish result-valued preparation for current programs, retaining validated shape,
  existing limits and layout. Migrate both public lowering constructors and their callers;
  step 7 extends that invariant to the new resource dimensions.
- [ ] Factor a small scoped-child helper from current Expr traversals where it removes
  duplication. Keep lexical environments explicit. Provide a named fragment-import helper
  using the receiving builder supply and freshening before insertion.
- [x] Extend external public-API and existing-operation fixtures to pin these contracts.
  Keep the pure library dependencies and numerical/materialization behavior unchanged.

Completion: existing scalar/vector operations validate the new structure and preparation
boundary without any scan arithmetic. Avoid adding an unused generic IR or speculative
capabilities; the next step supplies the concrete new constructor and scope requirements.

Evidence (2026-09-05): six commits land the six checked items. `5824787` folds
`Region_local`'s independent `shape`/`value` fields into `Region_local.Rhs.t` (`Scalar of
Value.t | Vector of {extent; var; body}`), with `Shape.t` demoted to a value-free projection
kept only for the `Shape_mismatch` payload and printing. `8c16c5a` gives `Region_context`'s and
all four `Region_computation` dispatch sites' `Invalid_program` the real `Region_program.error`
payload instead of erasing it; the graph-boundary regression's golden now shows the actual
cause. `fd8a802` restructures `Region_eval.materialize` from an unbounded per-key `Hashtbl` to
the same `fold_keys`/`fold_outputs` streaming shape `Region_execution.materialize` already
uses -- same reference `evaluate_locals`/`emit`, so results are unaffected; the whole tree's
tests confirm no golden changed. `3e79906` factors the byte-for-byte-duplicated slot-layout
fold and reader out of both evaluators into `Region_slots`, leaving each evaluator's own
`evaluate_locals`/`emit` independent. `384f8e5` makes `Region_execution.lower`/`lower_region`
take `~max_size`/`~max_depth` and re-run `Region_program.check` before lowering, closing a real
gap: `Region_program.with_output` is a raw record update, and `Kernel_eval.converted` uses
exactly that to splice in a result conversion after construction, so a rewritten emitter
previously reached `lower_region` unchecked; migrated all six call sites (`eval_direct`,
`eval_direct4`, `stage_program`, `kernel_eval`, two tests). `c7cc819` adds the regression this
check's fix demands: lowering the same program before and after an oversized `with_output`
rewrite, at the same limits, shows the rewritten one now fails (`depth exceeds limit 16`)
where it previously could not have been rejected at all.

The scoped-child/fragment-import helper is deliberately left undone. `lib/expr_internal/fold.ml`
already states its design intent against a generic scope-aware visitor ("binder behaviour stays
visible in each signature, and a new constructor breaks the ones that must handle it instead of
silently falling through a default"), and `rewrite.ml`'s `rebuild` already threads a caller-
supplied scope environment through `Reduce`'s binder for every rewrite built on it (`freshen`,
`alpha_normalize`) -- there is no current, unforced duplication left to factor. The only concrete
consumer named by the plan is `Builder.scan`'s own two-namespace construction and freshening
obligation, which does not exist until step 6. Building the helper now, disconnected from that
consumer, is exactly the "unused generic IR or speculative capabilities" this step's own
completion note warns against; land it as part of step 6 instead, against the real `Scan`
binders. `make precommit` and the whole `NO_COLOR=1 opam exec -- dune runtest` tree pass after
every commit above.

### 6. Extend those foundations with bounded scans

- [x] Extend `expr_repr`, the internal scan implementation and public recursive signature.
  Export `Scan`, `Scan_limits`, `Scan_meter`, `Scan_admission` and both projection constructors.
- [x] Add `Builder.scan` using the shared supply. Extend scope/import helpers, folds,
  freshening, substitution, checking, comparison, hashing, normalization and printing to
  local binders and the distinct initializer/update scopes.
- [x] Implement admission and inline evaluation: immutable previous rows, ordered lane
  updates, projection bounds, shared update charging and internal state reservation/unwinding.
- [x] Implement the exact `scan_reader`, projection errors, closed invalid-limit fields,
  missing-meter error and shared error conversions specified in the plan.
- [x] Add the scan RHS/layout case and continuation-based `Region_program.Builder.scan`.
  Extend dependency order, local-kind agreement and region invariance checks to trace reads.
- [x] Render unspecialized scan definitions with scoped `init` and `update` children.
- [x] Cover freshening/capture, init scope, zero steps, row-zero projections, nested state,
  exact-limit/next-charge behavior, and raw scans under constant/dynamic reductions.
- [x] Compile a consumer outside Expr that constructs and inspects a scan through public
  APIs; carry dependent signature and exhaustive-match migrations with the new constructors.

Completion: Expr builds independently, and scan semantics, scope and runtime bounds pass
focused tests. Region can construct/check/render trace definitions. Wider construction and
narrower evaluation limits exercise state rejection even when no updates occur.

Evidence (2026-09-05), Expr-level half (`03d7162`, `219737b`): `Scan`/`Scan_limits`/
`Scan_meter`/`Scan_admission` land in `lib/expr_internal/` (new files `scan.ml`,
`scan_limits.ml`, `scan_meter.ml`, `scan_admission.ml`), aliased through `lib/expr/expr.ml`.
`Scan` joins the existing recursive group in `expr_repr.ml`; `Value.t` gains `Local_scan_at`
(cached trace read) and `Scan_at` (inline descriptor), tags 11/12, append-only. `value.ml`'s
`compare`/`hash` gain a second, LOCAL alpha-equivalence namespace alongside the existing
reducer one -- `prev` is the first local binder this language has -- built so that every
non-scan comparison/hash takes the exact code path it did before (empty namespace, falls
through to the original branch). `fold.ml`'s locals-family queries (`locals`/`scalar_locals`/
`vector_locals`, plus new `scan_locals`) move off unscoped `walk` onto a shared hand-rolled
scope-aware recursion; `free_reducers`/`binders` gain scan's two sibling scopes; new
`local_binders`. `rewrite.ml`'s `rebuild` gains `on_local_scan_at`/`on_local_bind` channels;
`freshen` mints all three scan binders and gains a `free_locals` avoidance set;
`freshen_scan` lets a whole descriptor be freshened before splicing (used by
`substitute_locals`'s new `Scan of Scan.t` case, which is what lets `specialize_pixel`
eventually turn a cached read into a re-executing one). `check.ml` splits `Duplicate_binder`
into `Duplicate_local_binder`/`Duplicate_reducer_binder`. `Eval.value` gains `?scan`/
`?scan_meter` and evaluates an inline `Scan_at` directly over two row buffers, charging one
update per lane per step, reserving `2*width` state for the nesting peak
(`Fun.protect`-released on every exit including an `Err.Escape` unwind).

`dune build lib/expr` is green; the whole repository's existing test suite passes
UNCHANGED (no golden moved), confirming the additions are behavior-preserving for non-scan
expressions. `test/expr/scan_test.ml` covers the recurrence itself, zero steps, row-wins-on-
simultaneous-failure, the exact-limit/next-charge meter boundary, construction-time rejection
of the descriptor's worst case (including both zero-is-meaningful cases), freshening
(structural/interpretive equivalence plus a captured free local surviving untouched), a
cached `Local_scan_at` read delegating to a supplied reader, and `Scan_admission.check`
under no/constant/dynamic enclosing reductions. Runs under both JS backends
(`make js.runtest` green).

A real regression surfaced and was fixed in the same commit: the wider `Value.t`/`Eval.value`
measurably deepened `Eval.value`'s compiled stack frame under node, dropping
`Hard.eval_depth`'s survivable frontier from 2048 to somewhere between 1820 and 1850. Root
cause was NOT a tail-call issue (confirmed by testing -- an indirect ref-cell dispatch for the
new `Scan_at` case made it no better, and reverting to a plain `and`-bound sibling of `go`
changed nothing), just interpreter growth, matching this file's own documented precedent for
`Hard.eval_recursion`. Re-measured and lowered `Hard.eval_depth` to 1536 (confirmed stable
over repeated runs, ~2x resnet18's combined-depth requirement, same margin the original
ceiling had) and re-measured every dependent sample point in `depth_probe.ml`.

Evidence (2026-09-05), Region-level half (`4c4e2de`): `Region_local.Rhs` gains `Scan of
Expr.Scan.t` alongside `Scalar`/`Vector`; `Region_local.Shape` gains a matching `Scan of
{width; steps}`. `Region_program.Builder.scan` matches the plan's exact continuation-based
signature, converting `Expr.Builder.scan`'s failure with `Err.map_error` (never a raw
`Err.fail` re-wrap, which would double the detection trace) and never invoking the
continuation on failure; `Region_program.error` gains `` `Scan of Expr.Scan.error ``.
Dependency order, shape agreement and region-invariance needed no new traversal code:
`Region_local.Rhs.value` wraps a scan as `Expr.Value.scan_at` at closed placeholder indices
(`Expr.Index.zero`, never evaluated), so every existing `Region_program.check` call into
`Expr.Fold`/`Expr.Check.fragment` already applies its per-child `lane`/`step`/`prev` masking
unchanged -- a scan's `allowed_free` is empty, exactly like a scalar's, because
`Fold.free_reducers` on the wrapper already excludes `lane`/`step` per child before `check`
sees the result. `Shape_mismatch` gains a `read` field (three declared shapes need to know
which read triggered the mismatch, not just the old binary inference); `shape_error` checks
all three of `Fold.scalar_locals`/`vector_locals`/`scan_locals` against all three
declared-as-something-else predicates.

Rendering does NOT reuse that wrapper -- its placeholder row/lane would print as a
fabricated projection. `Expr.Pp` gains `scan`/`scan_open`, factored out of the existing
`Scan_at` case (`pp.ml`'s `at`/`scan_body`/`guard_at` became mutually recursive top-level
functions taking `~names` explicitly rather than closures private to `value_open`); it
renders `init`/`update` with real `lane`/`step`/`prev` naming but no trailing `@[row,lane]`.
`Region_program.pp`/`Region_trace.pp_local` (text) call it directly on the descriptor;
`Me_detail.of_value` (the explorer's graph-node view) instead splits a scan local into two
named roots (`init`/`update`) holding the raw `Expr.Value.t` children directly -- no wrapper
needed, since a root has no enclosing projection to fabricate -- and extends its
local-naming map with `prev -> "<name>_prev"` so `update`'s otherwise-unwrapped `prev`
occurrences still render readably.

`test/native/region_scan_construction_test.ml` is the external consumer fixture: builds a
counter scan through `Region_program.Builder.scan`, checks/prints it (confirming no
fabricated projection on the declaration line and a real one on the emitter's read),
confirms `Builder.scan` short-circuits without invoking its continuation on an invalid
descriptor, and exercises all three `Shape_mismatch` directions plus a scan-to-later-local
forward reference. Executing a scan-backed program is explicitly OUT of scope for this
step: `Region_execution`/`Region_eval`'s evaluators dispatch on a Scan RHS with a new,
explicit `` `Scan_execution_not_implemented `` case (mirrored into `Kernel_eval.error`)
rather than an inexhaustive match, documented in the tracked design record as a temporary
boundary expected to disappear once metered execution (a later step) lands.
`make precommit`, `NO_COLOR=1 opam exec -- dune runtest` (whole tree) and `make js.runtest`
all pass.

### 7. Separate resource dimensions and validate executable artifacts

- [x] Add `max_local_slots`, `max_scan_state`, `max_scan_updates_per_key` and
  `max_scan_updates_total` with checked construction, hard ceilings and consistent defaults.
- [x] Use bounded arithmetic for all aggregates. Implement the plan's occurrence-based
  measures, including vector extents, emitter multiplicity, enclosing reductions and key counts.
- [x] Stop charging storage against syntax size. Count resident slots once in scan peak
  state, omit nonexistent trace row buffers, and give scan-free programs zero scan-state cost.
- [x] Extend step 5's result-valued `lower`/`lower_region` preparation to validate and retain
  the new limits and measures. Cover both Pixel and Region branches.
- [x] Revalidate converted emitters, expose no unchecked constructor for a validated token,
  and run preflight from the operation, Stage and Kernel boundaries.
- [x] Preserve the full typed program/limit failure through operation-facing errors.
  Apply total-update aggregation at Kernel scope only.

Completion: every preparation entry rejects the same relevant invalid program before scan
execution. Tests distinguish storage, state, per-key and Kernel-total failures, including
scan-free programs with scan limits disabled and exact error payloads at the operation boundary.

Evidence (2026-09-06): the tracked design record's "Static measures" and "Validated execution
artifact" sections carry the full landed contract; summarized here. `Kernel.Limits.t` gains
`max_local_slots`/`max_scan_state` (`int`) and `max_scan_updates_per_key`/`max_scan_updates_total`
(`int64`), each checked against a new `Hard` ceiling by the same exclusive `v >= hard` rule as the
existing fields; `scan_limits : t -> Expr.Scan_limits.t` narrows the two `Expr`-shared fields, and
`default` derives them from `Expr.Scan_limits.default` rather than restating the constants.
`Expr.Fold.scan_cost : Value.t -> int64 * int` is the one traversal both static measures are built
from (lane-update count and peak live state a single evaluation costs), aggregating via saturating
`Int64` arithmetic; it deliberately does not multiply through an enclosing `Reduce`'s extent, which
is `Scan_admission.check`'s complementary job at a different layer (raw `Expr` construction/
rewrite) -- a documented split, not an oversight, since every scan built through
`Region_program.Builder.scan` sits flat at Region-local top level. `Region_program.preflight
~max_local_slots ~max_scan_state ~max_scan_updates ~output_shape` composes the renamed-in-effect
`checked_slot_total` (now billed against `max_local_slots`, never `max_size`) with a `scan_peak`
check (reusing `Scan.error`'s existing `State_over_limit` tag) and a `per_key` check (the new
`` `Scan_updates_over_limit ``); both formulas collapse to `0` with no special case for a scan-free
program, regression-tested directly. `Region_program.scan_updates_total` (`keys * per_key`) is
exposed separately and summed only at `Kernel.create`, the one Kernel-scoped
`` `Scan_updates_total_over_limit `` check.

`Region_execution.lower`/`lower_region` take the four new fields plus `~output_shape`, share one
`validate` helper running `check` then `preflight`, and `lowered` now retains `output_shape` so
`materialize`/`value_at` drop that parameter entirely. A real pre-existing gap surfaced and was
fixed in the same change: `lower`'s Pixel branch previously returned `Pixel_loop` with no
validation at all (not even `max_size`/`max_depth`); it now preflights too, which is sound because
a Pixel program's `outputs_per_key` is always 1. `Schedule.ground` itself became `Err.t`-returning
(crossing its `Tensor.materialize` callback with `Err.Escape` instead of `Err.or_raise`, per the
design record's own wording), and `Stage_program.ground` gained `?limits`, a `type error =
[Region_program.error | Region_eval.error]`, and a preflight-every-stage-first pass inside one
`Err.Escape` scope. `Kernel_eval.converted`, `eval_direct.ml`, `native4d/eval_direct4.ml`,
`lib/native_op_walk/native_verify.ml`, and roughly forty test call sites across
`test/native/graph_symbolic_*_test.ml`, `stage_program_test.ml`, `kernel_eval_test.ml`,
`symbolic_test.ml`, and `test/native4d/compute_test.ml` are migrated; none exercises a resource
limit, so every existing assertion is unchanged. The operation boundary
(`Region_computation.program`) is one funnel wrapping the existing four-op dispatch (renamed
`built`) with a single `preflight` call, not four; the Kernel boundary (`Kernel.create`) preflights
each value and separately aggregates `max_scan_updates_total`; `Me_classify.kernel` classifies the
new tag `Over_limit`, alongside the kernel's other size limits.

New tests: `test/native/region_preflight_test.ml` (storage/state/per-key/`outputs_per_key`/
scan-free-with-scans-disabled, all against a real scan-backed `Region_program.t`),
`test/native/kernel_scan_test.ml` (`max_scan_updates_total`, split into its own file to keep
`kernel_test.ml` under the file-size cap), plus dedicated cases in `region_compute_test.ml`
("Region construction preflights local/trace storage too") and `kernel_test.ml` ("the four scan
fields are checked and derived"). `opam exec -- dune build lib/expr`, `NO_COLOR=1 opam exec --
dune runtest test/expr test/native`, the whole-tree `NO_COLOR=1 opam exec -- dune runtest`,
`make jsoo.runtest`, `make jsoo.inline-runtest`, and `make melange.runtest` all pass.

Limitation: `make precommit`'s `format` step could not be verified end to end -- `dune fmt` fails
on `experiments/tailcall/backend_driver.ml`, untracked, cppo-preprocessed content unrelated to this
step that was already present in the working tree (alongside an untracked `.ai/expr_tailcall_design.md`
and an uncommitted `Makefile` addition wiring `tailcall.runtest`/`tailcall.benchmark`) before this
session's edits began and is left untouched, per this file's own instruction to preserve existing
edits. Every file this step touches was independently confirmed already correctly formatted
(auto-promoted cleanly before the aggregate `dune fmt` command reached the unrelated file), and
`build`/`runtest`/`check.file-size`/`check.whitespace` all pass directly against exactly the files
this step changed.

### 8. Execute Region traces and propagate configured meters

- [x] Run each trace descriptor once per key, writing directly into its validated slot range.
  Charge each lane update before evaluating its body; cached projections spend no updates.
- [x] Share one meter across all locals and emitters for a Region key. Allocate a fresh
  meter per standalone value, `value_at` invocation, and Pixel output coordinate as specified.
- [x] Extend the streaming reference materializer from step 5 to traces, retaining one
  slot array at a time and verifying the larger storage workload.
- [x] Thread limits through `Eval_symbolic.run`, `Region_kernel.of_graph`, Direct execution,
  Stage grounding and both Kernel Pixel paths. Remove silent default substitution.
- [x] Make `Stage_program.ground` preflight all stages before materializing the first and
  return `Err.t`. Use `Err.Escape` across `Schedule.ground`'s materialization callback.
- [x] Complete all signature/caller migrations, including Native4D's Direct path and tests.
- [x] Extend optional counters to observe scan starts and exact combined update charges
  in successfully executed programs, separately from preflight-rejection tests.

Completion: reference and production results agree; counters establish sharing and reset
scope. Repeated same-key `value_at` calls remain independent, custom limits survive every
path, and many-key materialization respects the one-array scratch assumption.

Evidence (2026-09-06): `0eb6fcf` implements execution; `34e8cd8` adds the larger-storage
regression. `Region_execution.evaluate_locals`/`Region_eval.evaluate_locals` both fill a
`Rhs.Scan s` local's whole preflighted slot range directly, row-major: row 0 is one
evaluation of `s.init` per lane (`~reducer:[(s.lane, l)]`, no update charge); row `r`
(`1<=r<=steps`) is one evaluation of `s.update` per lane
(`~reducer:[(s.lane, l); (s.step, r-1)]`), charging `Expr.Scan_meter.charge_update` once
per lane BEFORE evaluating its body, with `prev` answered by an ordinary `local_at`
resolver override reading row `r-1` from the slots just written -- deliberately NOT
`Eval.value`'s own inline `Scan_at` machinery (no `reserve`/`release` of `2*width` state),
since a trace local's storage was already accounted for by `max_local_slots`/
`max_scan_state` at preflight and only ever needs the public `charge_update` operation.
`Expr.Eval.value`'s `?reducer` widened from one optional `(Reduce_var.t * int)` pair to a
list, since a scan row's `update` needs `lane` and `step` bound simultaneously; the two
production call sites (both evaluators' `Vector` case) and one test call site pass a
singleton list, unaffected otherwise. `Region_slots.scan_reader : t -> float array ->
Expr.Eval.scan_reader` factors the cached-`Local_scan_at`-read half (bounds-check row then
lane against a trace local's own `(width, steps)`, now recorded in `Region_slots.t`
alongside the existing `(offset, count)` map, so lookup stays O(1) setup like `reader`),
shared by both evaluators the same way `reader` already is.

Meter allocation matches the design record's four-row reset-scope table exactly:
`Region_execution.materialize`/`Region_eval.materialize` create one fresh
`Expr.Scan_meter.t` per Region key (shared by every local, including a scan's own trace
fill, and the emitter for that key); `value_at` on either module creates one per
invocation; `Schedule.ground` and both of `Kernel_eval.machine`'s Pixel arms (the
on-demand `eval_value` path and the whole-tensor `materialize` path) create one per
output coordinate. `Region_execution.lower`/`lower_region` take `~scan_limits:
Expr.Scan_limits.t` directly (replacing the separate `~max_scan_state:int
~max_scan_updates:int64` pair) since `lowered` needs exactly that type to build a meter
later; `Region_program.preflight` keeps its original two-field signature unchanged,
narrowed from `scan_limits` at the one call site inside `Region_execution.validate`. Every
caller already held a `Kernel.Limits.t` and narrows it once via the existing
`Kernel.Limits.scan_limits` accessor: `kernel_eval.ml`, `eval_direct.ml`,
`eval_direct4.ml` (Native4D), `stage_program.ml`, and two test call sites
(`region_program_test.ml`, `region_compute_test.ml`, both migrated to
`Expr.Scan_limits.default` since their literals already matched it exactly).
`Eval_symbolic.run` gained `?limits` (default `Kernel.Limits.default`), threaded into
`Region_computation.program`, replacing a hardcoded default; `Region_kernel.of_graph`
resolves its own `?limits` once and passes that SAME value to both `Eval_symbolic.run`
and `Kernel_adapt.of_stage_program` rather than only the latter.
`Stage_program.ground`'s preflight-every-stage-first pass and `Schedule.ground`'s
`Err.Escape` crossing were already landed in step 7 (`46a5a6b`); this step additionally
threads `~scan_limits:Expr.Scan_limits.t` through `Schedule.ground` (one fresh meter per
Pixel output coordinate) and its `stage_program.ml`/test call sites.
`Scan_execution_not_implemented` is removed from `Region_eval.error`/`Kernel_eval.error`
(no longer reachable) along with its `pp_error` arms.

New test `test/native/region_scan_execution_test.ml`: production/reference agreement on a
counter-scan oracle (`trace[row,lane]=row`) across an 8-key partition; `counters.scans`/
`scan_updates`/`locals`/`emitters`/`keys` match hand-derived expected counts exactly;
`value_at` called twice at the same output succeeds both times (proving a fresh
per-invocation meter, not a shared one); a `max_state:0` limit still succeeds (proving a
trace local never reserves inline state); a tight `max_updates:5L` limit against
`Region_eval.materialize` directly (which never preflights) genuinely raises
`Updates_exhausted` -- confirmed non-vacuous by temporarily bypassing the `charge_update`
call and watching this same test fail, then restoring it; an exactly-one-key-sized limit
still succeeds across 8 keys (proving per-key reset, not a shared budget); and a 306-slot
single-key trace (`width=6, steps=50`) agrees bitwise between both evaluators, covering
"verify the larger storage workload". `NO_COLOR=1 opam exec -- dune runtest` (whole tree),
`make jsoo.runtest`, `make jsoo.inline-runtest`, and `make melange.runtest` all pass;
`make check.file-size`/`check.whitespace` pass; every file this step touches was
independently confirmed correctly formatted (`dune build @lib/fmt @test/fmt`).

Limitation: `make precommit`'s aggregate `format` step still cannot be verified end to
end in this session, for the same pre-existing reason recorded at step 7 -- `dune fmt`
fails on unrelated, untracked, cppo-preprocessed `experiments/tailcall/backend_driver.ml`,
present before this step's edits began and left untouched. Separately, and unrelated to
this step's own edits, `.ai/native_kernel_dsl_design.md`, `lib/native/kernel.ml`, and
`test/native/depth_probe.ml` were found modified on disk (an `Eval.value` frame-depth
comment rewording, no `Hard.eval_depth` value change) partway through this session by a
concurrent process outside this session's control; per this file's own instruction to
preserve existing edits, they were left uncommitted and excluded from both commits above.

### 9. Bound specialization and reject unsupported fusion

- [x] Re-measure the final expression after `specialize_pixel`, `substitute_loads` and
  `substitute_locals`; propagate scan limits and typed failures to callers.
- [x] Keep scope-preserving renaming and simple index/source rewrites cheap as specified.
- [x] Add a chained-scan case whose cached Region execution fits but whose specialized
  replay exceeds its update allowance; reject before replay begins.
- [x] Add a recurrent-computation summary to shared fusion admission. Exercise both planner
  and direct `Kernel_elab.admit` entry points, including a singleton inline scan.

Completion: compact syntax cannot bypass execution-cost admission after substitution, and
both fusion entry points reject recurrent computations consistently.

Evidence (2026-09-06): `Region_program.specialize_pixel`/`reconstructs` take
`~scan_limits:Expr.Scan_limits.t` and, once `max_size`/`max_depth` hold, call
`Expr.Fold.scan_cost` on the fully-inlined result and compare it to
`scan_limits`, failing with the existing `` `Scan (Expr.Scan.State_over_limit _
| Updates_over_limit _) `` -- no new error case, since a chain's multiplied
cost is exactly what `scan_cost`'s own recursive `Scan_at` case already
computes over the substituted tree (`Expr.Rewrite.substitute_locals`'s `Scan`
case reuses the `Scan_at` wrapper unchanged). This made `substitute_loads`/
`substitute_locals` gaining their own `limits` parameter unnecessary: the one
re-measurement at `specialize_pixel`'s own boundary already covers every
value they can produce, so they and the cheap scope-preserving helpers
(`freshen`, `alpha_normalize`, `substitute_output`, `substitute_reducer`,
`map_sources`) are untouched. Scan limits reach `specialize_pixel` from
`Stage_program.Stage.pixel_body` and `Ground_eval.Env.pixel_body`/`body_at`,
narrowed from `Kernel.Limits.t` the same way `Region_execution.lower` already
does.

A real bug surfaced and was fixed in the same change: `Expr.Fold
.measure_with_locals` (the size/depth re-measurement `specialize_pixel`'s
private `preflight` already ran) called its `local` callback for every
`Local`/`Local_at`/`Local_scan_at` node without regard to scope, including a
scan's own `prev` binder occurrences inside `update` -- a bound reference to
the scan's own state row, never a Region local, with no entry in the
caller's id-keyed map. This raised `Not_found` on ANY scan-backed program
reaching `specialize_pixel`, even a single non-chained one, before this
step's fix -- nothing before now had exercised that path. Fixed by threading
a `bound : Local_var.Set.t` scope through `measure_with_locals`'s walk
(adding `s.Scan.prev` to `bound` only for `s.Scan.update`, matching `prev`'s
own scope), the same bound-tracking convention `scoped_locals`/`keep`
already use for the analogous free-locals queries.

`Kernel_elab.admit` gains a `has_scan` check (reusing `Expr.Fold.scan_cost`:
`snd (scan_cost e) > 0` holds exactly when a `Scan_at` node is present, since
every scan reserves nonzero live state) over both resolved pixel expressions,
failing with the new `` `Recurrent_use of Kernel.Use.t `` (inserted
alphabetically). Both `site` and `site_in` call this one `admit`, so both the
direct entry point and the planner (`Fusion_plan.plan`, which routes through
`site_in`) reject consistently with no separate planner-side code --
confirmed by `Fusion_plan`'s own `pointwise` table only screening the
*consumer* side, so a scan-containing *producer* loaded by an
otherwise-pointwise consumer previously reached `site_in` unguarded.

Tests: `test/native/region_program_test.ml` ("specialize_pixel rejects a
chained scan whose replay exceeds updates even though Region execution
fits") builds two chained width-2/steps-5 scans (the second reading the
first's trace via the cheap `Local_scan_at` reader) under
`max_updates=50L`; `Region_program.preflight` succeeds (Region's own
`per_key` is 20) while `specialize_pixel` fails at the same limits (the
inlined replay costs 110 updates) -- demonstrating "efficient materialization
succeeds, specialized replay is rejected before any evaluator replays it".
`test/native/fusion_test.ml` ("both admission entry points reject a scan
producer, including a singleton inline scan") builds a two-value `Kernel.t`
by hand: a producer whose pixel expression is a bare `Scan_at` (no Region
local at all, built directly from `Expr.Builder.scan`) loaded once,
pointwise, by a scan-free consumer -- accepted by every pre-existing check,
rejected only by the new one, at `Kernel_elab.site`, `site_in`, and through
`Fusion_plan.plan`. `NO_COLOR=1 opam exec -- dune runtest` (whole tree),
`make jsoo.runtest`, `make jsoo.inline-runtest`, and `make melange.runtest`
all pass; `make build`/`runtest`/`check.file-size`/`check.whitespace` and
`opam exec -- dune build @lib/fmt @test/fmt` all pass on every file this step
touches.

Limitation: `make precommit`'s aggregate `format` step still cannot be
verified end to end -- `dune fmt` fails on the tracked, cppo-preprocessed
`experiments/tailcall/backend_driver.ml`, unrelated to this step and present
before it began (see steps 7/8's identical limitation note).

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
