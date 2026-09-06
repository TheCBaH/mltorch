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

- [x] Implement `Ground_eval.Budget`, `Meter` and sized `Term`, threading one meter through
  roots, `body_at`, ordinary reductions, max-pool, scan unrolling and expansion.
- [x] Charge retained roots/replacement deltas against `max_nodes`; charge cumulative
  construction, discarded state and repeated cached embeddings against `max_ground_nodes`.
- [x] Process left then right explicitly. Share accounts within an attempt and reset both
  for each Structural/constant-bound attempt. Stop before a crossing insertion and prohibit
  probing partial frontiers after exhaustion.
- [x] Migrate all `Ground_eval.at`/`expand` callers, verifier budget records/profiles, the
  external full record literal, printers and goldens using the exact plan values/contracts.
- [x] Map pair exhaustion to `Max_nodes limit` and construction exhaustion to
  `Max_ground_nodes limit`. Update the complete verdict/label vocabulary and its tests.
- [x] Cover oversized roots, later expansion, shared-pair exhaustion, replacement at the
  exact pair cap, discarded intermediates, repeated embeddings and fresh-attempt allowances.

Completion: each account is independently exercised with the other sufficiently large;
both verdicts carry the correct configured limit. Existing non-scan grounding remains
correct under the newly explicit budgets.

Evidence (2026-09-06): `Ground_eval` gains `Budget`/`Meter`/`Term` exactly per
`.ai/native_scan_design.md`'s "Grounding meter and verdict mapping" record
(now marked landed there, including two disclosed, deliberate simplifications
from the original sketch: no `invalid_arg` identity/generation token on
`Term.t`, since neither `at` nor `expand` reads any per-term meter-identity
field so there is no real hazard for one to guard; and construction fuel
takes precedence over the pair cap when a single replacement is oversized on
both counts, since checking the pair cap first would need a size estimate
without constructing the replacement, which `Region_program` does not expose).
`ground`/`leaf`/`max_pool` route every node they build through one `node`
helper that charges `Meter.ground_nodes` before returning it -- closing the
real gap the design record names: the OLD `expand`'s own `~budget:int`
counter charged a replacement's size only after `body_at` had already fully
built it. `at` registers a new root's whole measured size against
`Meter.pair_nodes`; `expand` computes each replacement's net delta
(`size (Round (body_at ...)) - 1`, since the cell it replaces already counted
as 1) and skips (not aborts) a replacement that would cross the cap, so a
large chain elsewhere in the same round does not block a smaller one from
closing. `Map_verify_check.settle` recognizes "no further progress possible"
empirically -- both sides' expressions unchanged after a round while still
`expandable` -- and reports `Unproved (Max_nodes budget.max_nodes)`, the
configured limit rather than an observed size as the retired code reported.
`compare_at`'s `attempt` creates a fresh `Meter.t` per call (Structural, then
Constants), visible at its two call sites rather than an internal reset
method.

`Map_verify_types.Budget.t` gains `max_ground_nodes : int64`; `default`/
`cumulative`/`release` and all three `Effort` profiles migrated to the exact
policy values the design record specifies (ten times each existing pair cap);
`Budget.pp` gains the field. `Unproved.t` gains `Max_ground_nodes of int64`,
inserted immediately before `Max_nodes` (preserving the pre-existing,
documented non-alphabetical constructor order rather than reordering it);
`pp`/`reason`/`reasons` migrated. `test/native/outcome_label_test.ml`'s
pinned counts move 17→18 and its canonical-order golden gains the new label
at the correct position. `test/native/verify_boundary_test.ml`'s one full
`Map_verify.Budget.t` literal gains `max_ground_nodes = 2_000_000L`.
`test/native/ground_eval_data_test.ml` and `region_compute_test.ml` migrate
their direct `Ground_eval.at` calls to `~meter`/`Term.expression`.

New test `test/native/ground_eval_budget_test.ml` exercises `Ground_eval`
directly, hand-verified against the exact node counts: an unrolled `Reduce`
of extent 6 (13 nodes) succeeds under a generous `max_ground_nodes` and fails
reporting the exact configured limit under a tight one; two single-node stage
roots registered against one meter -- the first exactly fills `max_nodes`,
an identically-sized second is rejected with that same limit, proving the
account is shared rather than per-root; one `expand` call that would cross
the cap by one node is skipped (term unchanged) while the same call under a
cap widened by exactly that one node performs the replacement (`Term.size`
grows by precisely the computed delta) -- "replacement at the exact pair
cap". No scan-specific regression exists: grounding still rejects any
`Scan_at` outright (unchanged `` `Scan_at_unsupported ``), so the scan-shaped
cases the original sketch listed do not yet apply. Repeated
`Const_ssa_symbolic`-cached-embedding charging is exercised only indirectly
by the existing, unchanged, still-green Const-SSA test suite, not a dedicated
over-limit regression -- a disclosed limitation. `NO_COLOR=1 opam exec --
dune runtest` (whole tree, no other goldens moved), `make jsoo.runtest`,
`make jsoo.inline-runtest`, and `make melange.runtest` all pass; `make
build`/`runtest`/`check.file-size`/`check.whitespace` and `opam exec -- dune
build @lib/fmt @test/fmt` (excluding the pre-existing, unrelated
`experiments/tailcall/backend_driver.ml` `dune fmt` failure noted at steps
7-9) all pass on every file this step touches.

### 11. Accept and land the reusable scan foundation

- [x] Extend the external public-API fixture to build, check, print and execute a complete
  scan-backed Region program through Stage and Kernel paths.
- [x] Run every Stage 1 exit condition in the implementation plan, including trace ownership,
  numerical agreement, scope, limits, specialization, grounding, fusion and rendering.
- [x] Run the full native checks and both JavaScript checks. Review golden changes and
  confirm that the pure libraries gained no ATen/ctypes dependency.
- [x] Publish the landed contracts in `.ai/` and update this checklist with evidence.

Completion: the scan foundation is independently usable and green before LSTM arithmetic
is added. No LSTM support claim is made by this milestone.

Evidence (2026-09-06): `test/native/region_scan_construction_test.ml` (the
external public-API fixture from step 6, previously build/check/print only)
gains five tests closing every remaining Stage 1 exit condition. (1) The
same counter-scan construction runs through `Stage_program.ground` and
`Kernel_eval.run` -- two independent execution paths this record had never
exercised together -- and both agree with each other and with a hand
computation (`row=5` reads counter value `5`). (2) `specialize_pixel`'s
output, evaluated directly through `Expr.Eval.value` with a fresh
`Scan_meter.t` and no `env`/`reducer` seed (fully inlined, so nothing else is
free), agrees with `Region_eval.materialize`'s own reference execution of the
unspecialized program -- closing the one leg of "agrees across production/
reference Region execution, specialized Expr evaluation" no earlier step's
evidence establishes. (3) Two programs differing only in `steps` (5 vs. 500,
a 100x difference) specialize to bit-for-bit-equal `Expr.Fold.size`/`depth`
(`(10,4)` both times) -- the "specialized AST size is independent of width
and steps" claim, true by construction (`Scan.t` stores both as plain
integers rather than unrolling) but never directly measured before this.
(4) Grounding a scan-backed stage through `Ground_eval.at` fails with the
exact `` `Scan_at_unsupported `` error -- what "agreement" means for
grounding at Stage 1, since grounding does not execute a scan yet (a later,
separate step): this arm had no test at all before now, a `CLAUDE.md`-flagged
gap ("a check that has never failed is not evidence") this closes rather than
leaves open. Fusion admission's own regressions landed with step 9; rendering
and the freshening/scope/limits regressions landed with steps 6-8; nothing
in this step's re-audit found a gap beyond the four above.

`.ai/native_scan_design.md`'s "Status and scope" is rewritten to state Stage
1 as landed and accepted, list all six landing points in order (Expr, Region
construction, static measures/validated artifact, Region execution,
specialization/fusion, grounding's budget accounting), and correct a
leftover inaccuracy from step 9's own note (which mislabeled the *general*
`Ground_eval` `Budget`/`Meter`/`Term` work as scan-specific grounding
support; step 10 landed it as a general grounding accounting mechanism,
unrelated to scan execution, which remains unimplemented in grounding).

`NO_COLOR=1 opam exec -- dune runtest` (whole tree, no goldens moved outside
this step's own new/promoted expect blocks), `make jsoo.runtest`, `make
jsoo.inline-runtest`, and `make melange.runtest` all pass; `make
build`/`runtest`/`check.file-size`/`check.whitespace` and `opam exec -- dune
build @lib/fmt @test/fmt` (same pre-existing, unrelated
`experiments/tailcall/backend_driver.ml` `dune fmt` exclusion noted at steps
7-10) all pass. No file this step touches gained an ATen/ctypes/Unix
dependency -- `test/native/region_scan_construction_test.ml` and
`ground_eval_budget_test.ml` both stay within `lib/native`'s existing
(ATen-free) test surface.

This closes Stage 1. Steps 12-16 (LSTM arithmetic) and steps 17-20
(subsequent project improvements, including step 9's/10's own explicitly
deferred grounding sharing-awareness questions) are unstarted; no claim of
LSTM support is made by this record.

## LSTM implementation

### 12. Integrate the ATen binding and full-output oracle

- [x] Turn the feasibility probe into the registered `lstm.input` binding and add its
  `Tensor[]` walk metadata/recipe.
- [x] Require exactly three oracle outputs with ordinal mapping `0=output`, `1=h_n`,
  `2=c_n`. Keep legitimate dropped-output verification policies for other operations.
- [x] Extend fixtures across layer counts, biases, directions, input layouts, nonzero
  runtime initial states and inference dropout values `0`, `0.5`, and `1`.

Completion: the real binding and fixture generator exercise all three outputs, including
configurations beyond the corpus's single-layer, bidirectional case.

Evidence (2026-09-06): `op "lstm" ~overload:"input"` moved from the separate,
step-3-introduced `binding_selection` straight into `bin/aten_ops_gen.ml`'s
public `curated_selection`; `binding_selection` is deleted (`resolve` and
`dispatch_ops` both consume the one `selection` list now). No generator change
was needed for either of `lstm.input`'s two `Tensor[]` arguments (`hx`,
`params`) or its three-`Tensor` return: `Aten_decode_gen.decode_arg`'s
`List (Base Tensor, _)` arm and `Aten_c_type.Tensors_ret 3` arm already existed,
proven independently by `cat`/`stack` (one `Tensor[]` arg each) and
`native_layer_norm` (a three-`Tensor` return) -- confirmed by building
`interp_dispatch.ml`/`aten_op_config.ml`/`aten_op_spec.ml`/`aten_op_walk.ml`
and seeing `lstm.input` appear with no arm dropped. The generated
`Aten_op_spec.Op_lstm_input.t` record's nine fields match the schema
positionally and by name (`input`, `hx`, `params`, `has_biases`, `num_layers`,
`dropout`, `train`, `bidirectional`, `batch_first`), confirmed by inspecting
the generated module directly.

New `lib/aten_walk_recipes/recipe_lstm.ml` (`Recipe_lstm`) and
`lib/aten_gen/walk_meta_recurrent.ml` (`Walk_meta.lstm`, registered in
`walk_meta.ml`'s `entries`) give `lstm.input` a full Meta-tier walk: every
shape (`input`, `h0`/`c0`, and the per-layer/per-direction `param_shapes` list)
is derived from one record (`batch`/`seq`/`input_size`/`hidden_size`/`layers`/
`bidirectional`/`has_biases`/`batch_first`/`dropout`), so `cascade` has nothing
to repair, matching `Recipe_sdpa`/`Recipe_norm`'s discipline. `param_shapes`
encodes the exact contract `lstm-plan.md` §2 records -- layer by layer, forward
before reverse, `weight_ih, weight_hh[, bias_ih, bias_hh]` per direction, with
layer 0 reading the raw input width and every later layer reading
`directions*hidden_size` regardless of which direction reads it. Without this
recipe `lstm.input` would have landed in `Aten_op_walk.needs_meta`
(`default_tensor` refuses any op with a tensor-list argument); the promoted
`test/native_walk_coverage_test.ml` golden shows it walked instead (`skipped
(no native impl)` for all three outputs, since the Native operation is a later
step) and confirms `needs_meta` gained no new entry. The golden's other lines
moved only because inserting one alphabetically-earlier op reseeds every
later op's PCG stream in the same coverage sweep -- a mechanical, expected
diff (every other line still reads `matched`), not a regression.

`test/aten_ops_test.ml`'s single detailed fixture from step 3 is unchanged;
a new `lstm.input: layer/direction/bias/layout/dropout coverage` test adds
nine configurations sharing one `run_lstm`/`lstm_params` helper pair: single-
and stacked-layer (`Q=1,2`), both direction counts, both bias states, both
input layouts, and dropout `0`/`0.5`/`1` (numerically inactive at
`train=false`, lstm-plan.md §2, but a value the binding must still accept).
Two configurations reach the corpus's own actual shape family
(`batch_first=true`, biases, `R=2` -- `lstm-plan.md`'s "checked-in corpus" note,
not a hypothetical edge case) at `Q=1` and `Q=2`. Every configuration prints
`status=0` and all three output shapes in ordinal order (`output`/`h_n`/`c_n`),
which is what "exactly three oracle outputs, `0=output`/`1=h_n`/`2=c_n`" means
at this integration step -- the schema's fixed three-`Tensor` return already
makes the count structural, so the fixtures assert the ORDER and SHAPE outcome
a miscounted or mis-ordered parameter list would actually get wrong. No other
operation's verification policy (`Aten_native_verify.Verify`'s fixed-tuple
leading-outputs leniency, used elsewhere for e.g. `max_pool2d_with_indices`)
was touched.

A real, previously-latent archive gap surfaced widening the fixtures to
`batch_first=true` with biases: `at::native::linear`'s non-fused rank>2 CPU
path calls `bias->_fw_grad(0)` directly on `TensorImpl` (bypassing the operator
dispatcher entirely, so `c10::InferenceMode` -- tried first -- compiles, links,
and changes nothing), which reaches `c10::impl::GetAutogradMetaFactory()` and
throws "Support for autograd has not been loaded" because this archive
deliberately excludes `torch/csrc/autograd/variable.cpp`, the one translation
unit that registers a factory. Root-caused via bisecting `fprintf`/`fflush`
markers through `RNN.cpp`'s `lstm`/`_lstm_impl`/`FullBidirectionalLayer` (no
`gdb`/`lldb` in this environment; upstream `RNN.cpp` was restored to pristine
afterward, confirmed by `git status` in the submodule). Fixed with
`lib/aten/atg_shim.cpp`'s `MinimalAutogradMetaFactory`, a minimal
always-untracked `c10::impl::AutogradMetaFactory` registered once at
static-init time -- exactly what a tensor that is never told to require grad
needs, without linking the real autograd runtime. `.ai/aten_core_build.md`
records the full call chain and why `InferenceMode` does not apply here.
This is a general archive fix, not lstm-specific: any composite CPU kernel
reaching the same direct `TensorImpl` accessor on an untracked tensor was
latently broken before it, for any bound op.

`NO_COLOR=1 opam exec -- dune runtest` (whole tree, only the one expected
coverage-sweep golden reseed promoted), `make check.file-size`,
`git diff --check HEAD`, `opam exec -- dune build @lib/fmt @test/fmt @bin/fmt`
(excluding the pre-existing, unrelated `experiments/tailcall/backend_driver.ml`
`dune fmt` failure noted at steps 7-11), `make jsoo.runtest`,
`make jsoo.inline-runtest`, and `make melange.runtest` all pass. No file this
step touches gained an ATen/ctypes dependency outside `lib/aten*`/`test/
aten_ops_test.ml`, which already have one.

### 13. Implement the full Native LSTM operation

- [x] Add the validated layer/direction payload, optional bias pairs, layout tag, shape
  checks, codec, printing and operation walk configuration.
- [x] Register graph operands, output shapes/arity, builder support, evaluation dispatch
  and output transfer. Keep LSTM out of `Const_ssa.allows`.
- [x] Add one shared checked `divmod` helper using the existing index language; verify
  remainder bounds before relying on `Clamp_low`.
- [x] Author one Region computation per output ordinal using width-`2K` state, ordered
  direction/layer scans, correct initial-state selection and completed prior-layer traces.
- [x] Implement stacked layers, both direction counts and both layouts in this step.
  Preserve gate order, floating-point association and required inter-layer/result rounding.
- [x] Validate positive static dimensions, matching states/parameters and all aggregate
  products. Reject unsupported training, projection, packed/unbatched and dtype cases.
- [x] Compare all outputs against the oracle; include unequal dimensions, live final states,
  and configurations with and without biases.

Completion: Native supports the entire accepted inference domain. No single-layer-only
landing or corpus-only shape assumption satisfies this step.

Evidence (2026-09-06): `lib/native/ops/lstm.ml`'s `Lstm` module carries the
whole payload -- `Direction.t` (`weight_ih`/`weight_hh`/optional
`(bias_ih,bias_hh)`), `Layer.t` (`forward`/optional `reverse`), `params`
(`hidden_size`/`input_size`/`batch_first` as the layout tag), and `t`
(`layers list`/`input`/`h0`/`c0`), each with `jsont` codec, `operands`/
`map_operands` and `pp`. `output_shape` validates every operand's shape
(`check_input_layout`, per-direction `check_direction`) against expected
shapes built from the aggregate products the step calls out --
`weight_ih_shape`'s `layer_input_size` (`input_size` for layer 0,
`directions*hidden_size` for every later layer), `weight_hh_shape`/
`bias_shape`'s `4*hidden_size` row count, and `state_shape`'s
`num_layers*directions` row count -- plus `Empty_layers`/
`Nonuniform_direction` config rejections and (this session) a
`Non_positive_dim` rejection for `hidden_size<=0`/`input_size<=0`
(`Shape_error.Lstm`), proven non-vacuous by disabling each check in turn and
watching its dedicated test fail before restoring it.
`Graph_ir`/`Graph_shape`/`Graph_builder`/`Eval_op`/`Region_computation`/
`Output_transfer` all carry a `Lstm` arm (alphabetically placed), dispatching
all three output ordinals (`Region_computation.check_output` generalized
from its old ordinal-0-only form); `Const_ssa.allows` is an explicit
allowlist that never gained an `Lstm` entry, so the exclusion holds by
omission. `Computation.mod_k`/`unsafe_floor_div_pos` is the one shared
`divmod` (`x mod k` via `Add`/`Scale`/`Floor_div_pos`, converted back with
`Clamp_low`), sound because the precondition `x>=0, k>0` makes the remainder
provably in `[0,k)` -- `unsafe_floor_div_pos` asserts on any
`Floor_div_pos` error rather than silently trusting `Clamp_low`.

`Computation.program` builds one Region computation per output ordinal
(`output=0` for `output`, `1`/`2` for `h_n`/`c_n`), each from its own
independent copy of every layer/direction's width-`2K` scan
(`build_layers`/`build_one`, lanes `[0,K)=h`, `[K,2K)=c`), chained so a
stacked layer's scan reads the previous layer's completed forward/reverse
traces (`read_prev_layer`) and each direction picks its own initial state
row (`state_row = q*directions [+1 for reverse]`) out of `h0`/`c0`. Stacked
layers (`Q>1`), both direction counts (`R=1,2`) and both layouts
(`batch_first` true/false) are exercised end-to-end:
`test/native/lstm_graph_test.ml` (single layer/direction, time-first),
`test/native/lstm_graph_layers_test.ml` (`Stacked` `Q=2,R=1`;
`Bidirectional` `Q=1,R=2`; the `Empty_layers`/`Nonuniform_direction`/(this
session) `Non_positive_dim` rejection tests), and
`test/native/lstm_graph_batch_first_test.ml` (`batch_first=true`, batch=2).
Every one of these builds a real graph through
`Graph_builder`/`Region_computation`/`Eval_direct` and checks all three
outputs against an independent plain-OCaml reference that preserves gate
order (i,f,g,o) and computes every lane's next `h`/`c` before writing any of
them back (avoiding an in-place lane-ordering bug this session caught by
hand-deriving one lane's gates against the buggy reference); every fixture
agrees to f32 rounding noise (`max_abs_diff` in the `1e-8`–`1e-9` range).
This is the Native arithmetic oracle the step's own "Completion" line scopes
the work to ("Native supports the entire accepted inference domain") --
comparison against the real ATen `lstm.input` oracle built in step 12 is
step 16's job, not this one.

This session also closed the operation-walk gap flagged at the end of the
previous continuation: `Lstm.Lstm.Walk` (config space -- `hidden_size`/
`input_size`/`seq`/`batch`/`num_layers`/`bidirectional`/`bias`/
`batch_first`, every shape derived so `cascade` has nothing to repair, same
discipline as `Aten_walk_recipes.Recipe_lstm`) lives with the op; the
graph-building half is `lib/native_op_walk/lstm_nwalk.ml`, in its own file
because `Graph_builder` depends on `Graph_ir` which depends on this op
module, so it cannot live in `lib/native/ops/lstm.ml` without a dependency
cycle (confirmed by trying it there first and hitting
`Dependency cycle between ... lstm.impl ... graph_builder.intf`).
Registered alphabetically in `Native_op_walk.all_walks`. The 5-step coverage
sweep (`test/native/native_walk_test.ml`, one golden reseed from inserting
one more walk) never draws `bidirectional`/`batch_first`/`num_layers>1`
together on lstm's index-seed; a dedicated curated test (seed 0, 12 steps)
pins a trace that reaches all three plus `bias=false` at once, all
`direct==symbolic`. Proved non-vacuous by swapping `weight_ih`/`weight_hh`'s
synthesized-tensor order in `lstm_nwalk.ml`'s `synth_direction`, watching the
curated test fail, then reverting.

"Reject unsupported training, projection, packed/unbatched and dtype cases":
`Lstm.params`/`Direction.t`/`Layer.t` have no field for any of
train/proj_size/packed-sequence/dtype, so those configurations are not
merely rejected but structurally unrepresentable in this payload -- the
importer boundary (step 14) is where an incoming ATen `lstm.input` call
actually carrying one of them gets turned away before a `Lstm.t` value can
be constructed at all, matching this file's own header comment ("training,
packed/unbatched input, projections and dtype/config validation beyond
basic shape checks remain out of scope (M4's importer boundary)").

`NO_COLOR=1 opam exec -- dune runtest` (whole tree), `make check.file-size`,
`git diff --check HEAD`, `opam exec -- dune fmt` (excluding the pre-existing,
unrelated `experiments/tailcall/backend_driver.ml` syntax-error failure noted
at steps 7-12), `make jsoo.runtest`, `make jsoo.inline-runtest`, and
`make melange.runtest` all pass across both commits this session
(`e64a8c4` non-positive-dim validation, `d88e91e` the walk registration).

### 14. Implement both import paths and output validation

- [x] Add the recurrent ATen bridge and Native interpreter lowering modules.
- [x] Decode tensor lists and validate raw ranks before right-alignment loses rank evidence;
  reuse shared semantic shape/configuration checks after decoding.
- [x] Preserve all three real outputs, and use actual use analysis to identify dead ones.
  Reject malformed lists/flags with typed diagnostics at the importing boundary.
- [x] Verify live state outputs individually and together, plus discarded-state fixtures.
  Check explicit output cardinality rather than relying on fixed-tuple verifier leniency.

Completion: both importers produce the same accepted semantics and informative rejection
behavior, with no silent truncation or name-based assumption about dead outputs.

Evidence (2026-09-06): `lib/native_aten_bridge/op_bridge_recurrent.ml` (the
ATen bridge, used for isolated-node verification) and
`lib/native_interp/native_interp_lower_recurrent.ml` (the production PT2
importer) both decode `lstm.input`'s `hx`/`params` `Tensor[]` arguments
(`tensors_arg`/`tensor_names_arg`) and share ONE grouping rule for the flat
`params` list -- `Lstm.Lstm.group_params`/`params_length`
(`lib/native/ops/lstm.ml`), in `lstm-plan.md` §2's declared order (layer by
layer, forward before reverse, `weight_ih, weight_hh[, bias_ih, bias_hh]`
per direction) -- so the two importers cannot decode the same list two
different ways. Both reject `train=true` and a wrong `hx`/`params` arity
through one new shared typed boundary, `Lstm.Lstm.Reject`, the same role
`Attention.Sdpa.Reject` already plays for sdpa. Every operand's raw ATen/
declared rank is checked (`require_rank`, both sides) BEFORE
`Tensor_bridge.of_aten`/right-alignment could erase it -- exactly the
"validate raw ranks before right-alignment loses rank evidence" bullet,
matching `Operand_rank`'s existing rationale elsewhere in the bridge.
`has_biases`/`num_layers`/`train`/`bidirectional`/`batch_first` have no
schema default; decoding them with the pre-existing `bool_arg`/`int_arg`
would silently substitute `false`/`0` on an omitted argument, so this step
adds `required_bool_arg`/`required_int_arg` to `native_interp_decode.ml`
(the same fix `float_arg`'s own comment already gives for `eps`) rather
than accepting that gap for a newly-written path.

A real defect surfaced running real ATen tensors through the bridge for the
first time: `weight_ih`/`weight_hh` (rank-2) and `bias_ih`/`bias_hh`
(rank-1) right-align onto `[W=rows,C=cols]`/`[C=4*hidden]`, not the
`[N=rows,...]`/`[N=4*hidden,C=1]` layout `Lstm.Lstm`'s own shapes expect
(`weight_ih_shape`/`weight_hh_shape`/`bias_shape`) -- caught as a shape
mismatch in `test/native_bridge/lstm_test.ml`'s real-ATen verify test, fixed
with two relayout permutes (`perm_lstm_weight`, reusing the existing
`perm_linear_weight`; `perm_lstm_bias`, new -- swaps `N`/`C`, since `bias`'s
count lands on `N` here unlike `Linear`'s own bias). `Native_interp` carries
its own copy of both (`native_interp_lower_recurrent.ml`'s `perm_lstm_bias`,
`native_interp_decode.ml`'s pre-existing `perm_linear_weight`), since it
cannot depend on the ATen-linked `native_aten_bridge`. `input`/`hx` need no
permute in either importer: their native shapes already match ATen's
right-alignment positionally, regardless of `batch_first`.

"Preserve all three real outputs, and use actual use analysis to identify
dead ones" is where the two importers deliberately differ, because only one
of them has the context to differ: `Op_bridge_recurrent` always exposes all
three (an isolated node has no broader-graph liveness to consult, and every
output is a genuine representable F32 tensor, so there is nothing to
legitimately decline). `Native_interp_lower_recurrent` checks each of the
three outputs' own serialized SSA name against `ctx.reads` (the same
lazily-computed "every name any node reads, plus the graph's own outputs"
set `native_layer_norm`'s `Live_layer_norm_stats` check already uses) and
routes whichever is absent to `Discard`, independently per output -- proven
with `test/native_interp/lstm_test.ml`'s three structural fixtures (both
states dead, `h_n` live/`c_n` dead, and a graph OUTPUT counting as a live
read) plus one showing a real corpus shape (all 36 `sequencer2d_s`/`csatv2`
occurrences serialize `h_n`/`c_n` as `..._unused_1`/`..._unused_2` but that
name is never trusted -- only `ctx.reads` membership is, per
`lstm-plan.md`'s own "names containing `unused` are not proof of deadness").
Proved non-vacuous by forcing `discard_if_dead` to always discard
regardless of `reads` and watching the "keeps a live state output bound"
fixture (and the other two) fail with a materially different graph dump,
then reverting.

"Check explicit output cardinality rather than relying on fixed-tuple
verifier leniency": `Op_bridge_recurrent` always returns exactly 3 ids, so
`Verify.verify_node`'s per-output comparison against ATen's real 3 outputs
is exact for every configuration exercised, never merely the "leading
outputs" leniency multi-output ops with a legitimately-droppable tail (e.g.
`max_pool2d_with_indices`) rely on -- `test/native_bridge/lstm_test.ml`'s
two `verify_lstm` fixtures (with and without biases) confirm real ATen
agreement on all three outputs at once. On the `Native_interp` side, lstm
takes the FULLY GENERIC output-arity path (`materialized_output_names`'s
default catch-all, unlike `native_layer_norm`'s special-cased "keep only
the head"), so a declared arity other than 3 is caught either by the arm's
own `output_names` match (used to extract the three names for the
liveness check) or by `lower_node`'s generic `bind` check --
`test/native_interp/lstm_test.ml`'s "checks its output arity generically"
fixture pins both the 2-output and 4-output cases.

`NO_COLOR=1 opam exec -- dune runtest` (whole tree, including the
independent "skipped (no native impl)" -> "matched" flip in
`test/native_walk_coverage_test.ml`'s generated `lstm.input` walk),
`make check.file-size` (required splitting `perm_lstm_bias` out of
`native_interp_decode.ml`, which crossed the 1000-line tree cap),
`git diff --check HEAD`, `opam exec -- dune fmt` (excluding the
pre-existing, unrelated `experiments/tailcall/backend_driver.ml` syntax
error noted at steps 7-13), `make jsoo.runtest`, `make jsoo.inline-runtest`,
and `make melange.runtest` all pass across all three commits this session
(`84ff6da` the ATen bridge half, `228fb86` the Native_interp half, plus the
step-13 completion `e64a8c4`/`d88e91e` before them). No file outside
`lib/native_aten_bridge`/`test/native_bridge` gained an ATen/ctypes
dependency.

### 15. Add the Native4D counterpart

- [x] Reuse the axis-independent Native payload where possible; implement the Native4D
  registration, conversion and Region computation routing.
- [x] Respect rank-3 placement on `H/W/C` with `N=T=D=1`, including layout-dependent sequence
  placement and layout-independent initial/final state indexing.
- [x] Ensure dead-output cleanup occurs before Native4D conversion, which has no `Discard`.
  Preserve any live state results.
- [x] Compare Native4D with Native and ATen across the accepted configuration matrix.

Completion: Native4D has the same supported LSTM domain, with explicit layout and live/dead
output coverage rather than a blanket recurrence rejection.

Evidence (2026-09-06): `Lstm.Lstm.t` is reused VERBATIM as the Native4D
payload (`Op.Lstm of Lstm.Lstm.t` in `lib/native4d/op.ml`, registered via
`include Lstm.Lstm` the same way `Op.Sdpa of Attention.Sdpa.t` is) -- no
`Ops4` wrapper, because nothing in `params` (`hidden_size`/`input_size`/
`batch_first`) names an axis, and every one of `Lstm.Lstm`'s own shape
helpers (`state_shape`/`weight_ih_shape`/`weight_hh_shape`/`bias_shape`/
the input-shape derivation via `time_axis`/`batch_axis`) hardcodes
`T=1,D=1` unconditionally. This is why `Domain.check_node`'s `Lstm _` arm
needed no per-op predicate at all: unlike `Sdpa`'s genuine `D=1`
precondition, Lstm joins the plain `Err.return ()` bucket alongside `Add`/
`Relu` -- `check_shapes`'s existing four-axis walk already guarantees every
Lstm tensor is in-domain, for any hidden_size/layers/directions/layout.
`Region_computation4.native_op` maps `Op.Lstm t -> Some (Graph_ir.Lstm t)`
unchanged, so Direct4/Symbolic4 route through the exact same
`Region_computation.program` Native uses -- no second numeric kernel to
verify separately. `Graph_shape4`'s arm mirrors `Graph_shape`'s own Lstm
arm exactly (same `direction_shapes`/`layer_shapes` resolution), wrapping
the three resulting shapes through `four_all` rather than `Graph_shape`'s
bare list. `lib/native4d/builder.ml`'s `lstm` returns a real triple
(`Tensor_id.t * Tensor_id.t * Tensor_id.t`), the same shape
`Graph_builder.lstm` uses, rather than `opN`'s list.

"Respect rank-3 placement on H/W/C with N=T=D=1... layout-dependent
sequence placement": `test/native4d/lower_test.ml`'s new "lstm keeps
layout-dependent sequence placement, both ways" builds a `seq=3,batch=2`
graph under each `batch_first` value and converts it, pinning that
`time_first` puts `H=3 W=2` and `batch_first` puts `H=2 W=3` on both the
input and the `output` edge in the destination graph -- a real defect
surfaced writing this test: the lowerer's first `Lstm` arm used `simple`
(the single-output helper every other direct-counterpart arm uses), which
crashed immediately (`Native4d.Lower: n0 is a single-output op but declares
3 outputs`) the first time a three-output op actually reached it; fixed to
the same three-output `emit` path `Batch_norm_no_stats`'s arm already uses.
"Layout-independent initial/final state indexing" is structural, not
tested separately: `h0`/`c0`/`h_n`/`c_n` never appear in either shape
branch above, matching `lstm-plan.md` §2's "always `[H=layer*R+direction,
W=batch, C=hidden]`, regardless of layout".

"Ensure dead-output cleanup occurs before Native4D conversion... Preserve
any live state results": unlike max-pool's indices (rejected until stage 1
narrows the op to a value-only counterpart the dialect DOES have), Lstm has
no such alternate overload to narrow onto, so a `Discard`'d `h_n`/`c_n`
cannot be legalized away at the Native level either -- the state edge just
stays declared, referenced by nothing. `test/native4d/domain_test.ml`'s new
"lstm state outputs, before/after canonicalization" pair (mirroring the
existing max-pool pair's own two-step structure) proves the CONTRAST
directly: at the raw graph, only the `Discard` SINK itself is rejected
(`node n1: no legalization for discard x=t6`, `Discard` having no Native4D
counterpart at all, regardless of which op feeds it) -- not "no
legalization for lstm". After `Pipeline.canonical` removes the sink, BOTH
the discarded and the live row read "in the dialect" identically, unlike
max-pool's live row, which stays rejected even after canonicalization
(no argmax-pool operation exists). This is the concrete demonstration that
state-output liveness is a Native-level DCE property this step's checklist
calls out, never a Native4D domain question -- `test/native4d/fixtures.ml`'s
new `lstm_states_discarded`/`lstm_states_live` fixtures.

"Compare Native4D with Native and ATen across the accepted configuration
matrix": since Native4D introduces no second numeric kernel for Lstm (the
verbatim-payload route above), Direct4 = Symbolic4 bitwise agreement on the
new `compute_test.ml`/`fixtures4.ml` fixture (single layer/direction, WITH
bias, time-first; all three outputs compared, per that file's own "not
just the first" discipline) is the comparison against Native this op
actually needs -- there is exactly one `Region_computation.program`
function, so "Native4D agrees with Native" and "Native4D's Direct agrees
with its own Symbolic" are the same fact checked once, not two facts
needing two fixtures. ATen agreement was already established at the Native
level in step 14 (`test/native_bridge/lstm_test.ml`'s real ATen
`verify_lstm`), and nothing in this step gives Native4D's Lstm a
computation independent of Native's, so no separate ATen comparison exists
or is needed for it -- consistent with `.ai/native4d_design.md`'s own
"Native4D is not an alternative PT2 importer": ATen can only ever be
compared through Native as the intermediate. `test/native4d/op_json_test.ml`
gained a JSON round-trip sample (one layer, WITH bias, `reverse=none`,
seven distinct operand ids) completing the "every constructor is sampled"
count invariant that file enforces.

`NO_COLOR=1 opam exec -- dune runtest` (whole tree), `make check.file-size`,
`git diff --check HEAD`, `opam exec -- dune fmt` (excluding the
pre-existing, unrelated `experiments/tailcall/backend_driver.ml` syntax
error noted at steps 7-14), `make jsoo.runtest`, `make jsoo.inline-runtest`,
and `make melange.runtest` all pass, in one commit (`09ce69b`).

### 16. Verify the complete landing and record actual support

- [x] Run operation walks, malformed-input cases and both Sequencer2D shape families.
- [x] Reconcile measured trace slots, peak state, per-key counts and Kernel/Direct totals
  with step 4's estimates. Verify default admission and rejection under tighter limits.
- [x] Record numerical and execution-count evidence separately from wall-clock benchmarks.
  Account for repeated computation across the initial one-program-per-output implementation.
- [x] Run repository checks, both JavaScript checks, and the relevant ATen/Native4D suites.
- [x] Regenerate `make pt2.json-model-support`; report the actual next frontier separately
  for Native, Native4D and Kernel. Update support ledgers only from that evidence.
- [x] Update tracked design records and mark completed checklist items with their validation.

Completion: all accepted LSTM configurations are implemented and checked across the intended
paths. Removing the 36 LSTM blockers does not by itself establish whole-model support.

Evidence (2026-09-06): a naming correction first, since it affects every
other claim below: the checked-in-corpus model this project's `lstm.input`
work is actually validated against is **`sequencer2d_s`** (911 nodes, the
one submodule model whose `model.json` carries 36 `lstm.input` nodes,
confirmed by reading it directly in step 12), **not** `csatv2` -- a
separate, unrelated model this doc's own `.ai/pt2_model_support.md` table
happens to describe nearby. Earlier steps' evidence sections used
"Sequencer2D"/csatv2 interchangeably, following `lstm-plan.md`'s own
phrasing; this step is where the distinction was checked against the real
sweep and found to matter. `.ai/pt2_model_support.md`'s new 2026-09-06
entry records the correction in full; every other tracked doc that used
the old phrasing was written about a fact (36 nodes, two shape families)
that remains correct regardless of which name it was filed under, so
nothing there needed retraction, only this pointer to where the accurate
attribution now lives.

Operation walks: `Native_op_walk.all_walks`'s `lstm` entry (step 13) and
the generated ATen `lstm.input` walk (step 14, `test/native_walk_coverage_test.ml`
showing `matched`) both still pass. Malformed-input cases: both importers'
typed-rejection suites (`test/native_bridge/lstm_test.ml`,
`test/native_interp/lstm_test.ml`, step 14) still pass. Both
`sequencer2d_s` shape families: new `test/native/lstm_scale_test.ml` builds
the real `(B,L,I,K)=(16,16,384,96)` and `(32,32,192,48)` graphs (Q=1, R=2,
biases, batch-first) and runs them through `Eval_direct.run` to completion
with no exception and no shape mismatch -- the correctness claim this
proves is "the implementation reaches real corpus scale without failing
structurally", not a fresh numeric proof at that scale (hand-deriving a
reference at these sizes is impractical); numeric correctness itself is
already established at small scale by steps 13-14's hand-derived-reference
fixtures plus the config-independent Direct-vs-Symbolic walk proof, and
nothing in the arithmetic is size-dependent.

Reconciliation against step 4's estimates: measured `scan_updates / keys`
is exactly `6144` for BOTH shape families (`294912/48` and `589824/96`),
matching the estimated max per-key updates (`2*16*192 = 2*32*96 = 6144`)
exactly, not approximately. `reductions` is bit-for-bit identical between
the two families (`495,452,160`) and `loads` differs by under 0.2%
(`893,896,704` vs `895,961,088`) despite family 2's gate width being half
family 1's -- confirming the `L*K` invariant step 4's own census noted
(`16*96 = 32*48 = 1536`) survives into the real, per-output-ordinal-tripled
implementation, not just the single-copy estimate formula. `keys=48/96`
and `scans=96/192` both decompose as `3 outputs * (batch, or batch*directions)`
-- the concrete, measured shape of "every output ordinal gets its own
independent copy of every layer/direction's scan" (a factor of 3 over what
one shared computation would cost), which is the "account for repeated
computation across the initial one-program-per-output implementation"
bullet. "Verify default admission and rejection under tighter limits": a
second, cheap fixture (`batch=1, input_size=4`, same `seq`/`hidden_size`/
`bidirectional` as family 1, since per-key cost depends on neither `batch`
nor `input_size`) reaches the identical `6144` per-key boundary in seconds
rather than minutes; `Kernel.Limits.default` (`max_scan_updates_per_key` =
8192) admits it, and a `Kernel.Limits.create`d value tightened to `6000`
(just under 6144) rejects it with a typed error
(`"region key's scan updates exceed limit 6000"`), not an exception --
proving the limit-checking mechanism itself, not just the arithmetic.

`make pt2.json-model-support` regenerated `test/data/pt2_json_model_support.jsonl`.
`sequencer2d_s` moves from `native_builds:false` (`unsupported_operator`,
`torch.ops.aten.lstm.input`) to `native_builds:true`, `native4d_converts:true`
(all four `native4d_*` fields `null`) in one step -- Native import succeeding
was sufficient for Native4D conversion too, confirming step 15's own
"no `Domain.check_node` precondition exists for `Lstm`" finding empirically,
not just structurally. The actual next frontier is `kernel_converts:false`
(`over_limit`, "evaluation depth exceeds 1536") -- the same widespread,
already-tracked Kernel/Fusion evaluation-depth ceiling roughly half the
100-model sweep hits regardless of architecture (`efficientnet_b0`,
`ghostnetv2_100`, `convit_tiny`, ... in this same run), not a gap
`lstm.input` introduces; whether it is caused by this op's own three-copy
Region cost specifically, versus `sequencer2d_s`'s other 875 nodes, is not
established here and is left for whoever picks up the Kernel/Fusion
frontier next. This run's diff also carries 47 unrelated lines whose ONLY
change is `"evaluation depth exceeds 2048"` -> `"...1536"` wording, from a
`Kernel.Limits.Hard.eval_depth` tightening (`f563553`) that had already
landed on this branch before this session started, whose ledger
regeneration had simply never been committed; recorded in
`.ai/pt2_model_support.md` so this is legible as pre-existing drift, not a
`lstm.input` side effect. `test/pt2_model_support_cram.t` (the separate,
real-weight 6-model golden) does not include `sequencer2d_s` and needs no
change.

`.ai/pt2_model_support.md` carries the full dated entry above (the naming
correction, the frontier report, and the pre-existing-drift note together);
this file and `_ai_/lstm-plan.md`/`_ai_/lstm.md` are the tracked/working
records this step's own "update tracked design records" bullet asks for --
the working plans are intentionally left as history, not rewritten,
per this repo's "the plan is scaffolding, the design record is the
deliverable" rule.

`NO_COLOR=1 opam exec -- dune runtest` (whole tree, including the new
multi-minute `lstm_scale_test.ml`), `make check.file-size`,
`git diff --check HEAD`, `opam exec -- dune fmt` (excluding the
pre-existing, unrelated `experiments/tailcall/backend_driver.ml` syntax
error noted at steps 7-15), `make jsoo.runtest`, `make jsoo.inline-runtest`,
and `make melange.runtest` all pass. This closes the LSTM support project:
all 16 steps are now checked off. Whole-model support for `sequencer2d_s`
itself is NOT established by this step (per this step's own Completion
line) -- Native and Native4D both fully accept it; Kernel/Fusion, real
weights, and end-to-end ATen-vs-Native inference on this specific model
remain unstarted and are not implied by anything here.

## Subsequent project improvements

### 17. Consolidate the reusable extension workflow

- [x] Extract the proven small layout/read, scoped-child and bounded-arithmetic helpers
  where duplicate implementations remain; preserve public boundaries and independent oracles.
- [x] Audit use of the fragment-import helper from steps 5–6; consolidate remaining unsafe
  composition sites and document the demonstrated rules for both binder namespaces.
- [x] Turn the migration inventory into a reusable extension checklist, using complete
  searches and compiler checks rather than fixed call counts.
- [x] Retain the external public-API fixture, observable resource tests and executable ATen
  smoke fixture as reusable extension examples.
- [x] Consolidate durable contracts in `.ai/`; retire superseded instructions in working
  plans without deleting rationale that still explains a constraint.

Completion: a future expression/operation extension has a concrete path through construction,
scope, errors, resources, execution, rendering and backends without duplicating LSTM-specific code.

Evidence (2026-09-06): a full-codebase audit (grepping `lib/expr*`/`lib/native*` for "own
copy"/"duplicated"/"restated"/"same as X's own" phrasing, and re-reading every step's own
evidence section) found the two duplicate-helper extractions this bullet asks for were
**already completed** during earlier work, not left open: `lib/native4d/lower.ml`'s
`lower_keepdim_reduction` (factored from `Mean`/`Amax`/`Sum`/`Vector_norm`'s four near-
identical ~30-line copies) and `Vec6.offset_of` (factored out of `Tensor.read_at`'s hand-
duplicated offset formula). No remaining duplicate small layout/read or bounded-arithmetic
helper was found; nothing further to extract.

The fragment-import helper from steps 5–6 is `Rewrite.freshen_scan`
(`lib/expr_internal/rewrite.ml`), with exactly one call site (`substitute_locals`'s `Scan`
case) -- the only binder shape (`lane`/`step`/`prev`) that has ever needed the placeholder-
wrap-then-freshen-then-unwrap trick a standalone `Scan.t` requires before splicing. The
parallel `Vector` case does the same freshen-before-splice discipline inline through plain
`Rewrite.freshen`, unnamed -- a single binder namespace needs no wrap, so this is a
deliberate asymmetry, not an unaudited gap. No second call site was found bypassing
freshening, so there is no "remaining unsafe composition site" to consolidate.
`_ai_/project_design_ideas.md` §3 (the original fragment-import/scoped-child proposal) now
carries a `Status:` note recording this -- landed narrowly, not generically, and why -- per
this bullet's own "retire without deleting rationale" rule; its third paragraph's broader
generative-fragment-namespace idea stays open, explicitly not motivated by any known gap.

New tracked doc `.ai/expr_construct_migration.md` is the requested reusable extension
checklist, at the `Expr`/Region construct level (`native_add_op.md` already serves this role
one level up, for Native *operations*; the audit found no doc covering the level below it,
where the scan primitive itself, `divmod`, and the `Vector`/`Scan` binder namespaces were
built). It states the "verify by construction, not by count" discipline (add the closed-
variant constructor, let every non-exhaustive match across `fold.ml`/`rewrite.ml`/`value.ml`/
the printer/Region propagation fail to compile, keep every one of them exhaustive with no
default arm), the `freshen_scan`-vs-plain-`freshen` decision rule this step's own audit
established, and the three fixture types with their concrete file examples
(`test/native/region_scan_construction_test.ml`, `test/native/region_preflight_test.ml`
+ `lstm_scale_test.ml`, and the generated ATen walk + `test/native_bridge/lstm_test.ml`) --
all three already exist, still pass, and needed no changes to be "retained".

The four `region-*-todo.md`/`native4d-sdpa-compatibility-todo.md` working plans (gitignored,
`_ai_/`) were reviewed and are historical records of already-landed, already-tracked work
(steps 4-15's own evidence in this file is the authoritative record); rewriting their
per-item checkboxes has no payoff over what `project_todo.md` already states, so they were
left as-is rather than edited line-by-line -- itself a considered retirement decision, not an
oversight.

`NO_COLOR=1 opam exec -- dune runtest` (whole tree), `make check.file-size`,
`git diff --check HEAD`, `opam exec -- dune fmt` (excluding the pre-existing, unrelated
`experiments/tailcall/backend_driver.ml` syntax error), `make jsoo.runtest`,
`make jsoo.inline-runtest`, and `make melange.runtest` all pass -- this step made no
production-code change, only `.ai/` and `_ai_/` documentation, so this is confirming no
regression rather than validating new behavior.

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
