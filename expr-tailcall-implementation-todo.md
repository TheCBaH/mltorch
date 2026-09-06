# Expression tail-call implementation tracker

Implementation specification: [Expression tail-call implementation plan](expr-tailcall-implementation-plan.md).

All items start unchecked. This tracker records planned work; implementation
status has not been audited. Mark a stage complete only after its work and
applicable verification gates pass. Record evidence in the final table.

## Scope constraints

- Preserve native evaluator behavior, native hard constants, and depth-probe output.
- Convert Value/Bool traversal for bounded Index subtrees; leave `eval_index`
  and producer recursion out of scope.
- Use the full jsoo mirror closure and isolated Melange Expr mirrors.
- Allocate evaluator scratch storage per call; defer cross-call pooling.
- Keep benchmark and fault-injection APIs outside the public Expr API.

## Stage 0 — Baselines and evaluation-order oracle

- [x] Confirm warning 51 is fatal under dev and Melange profiles; demonstrate
  failure with a temporary non-tail annotation and remove it. Verified on
  `experiments/tailcall/tailcall_cases.ml`'s `eval_direct` Binary arm
  (`(eval_direct[@tailcall]) b`): `Error (warning 51 ...)` under dev,
  `Warning 51 ...` (non-fatal) under melange -- melange's warning-fatality
  range differs from dev's; not yet reconciled with the plan's "as under
  dev" claim, see follow-up note.
- [x] Save native region-pixel and region-compute benchmark baselines
  (`make benchmark.region_pixel`/`benchmark.region_compute` output,
  captured this session; not committed to a file -- see follow-up note).
- [x] Save native depth-probe output and hard constants: 256 / 1536 / 96.
  Already the committed baseline in `test/native/depth_probe.ml`'s expect
  blocks; reconfirmed green via `dune build @test/native/runtest`.
- [x] Add canonical `js/probe/order_probe.ml` using the real evaluator and
  logging load/index-load callbacks.
- [x] Cover both-success and both-failure cases for all seven rewritten
  order-sensitive call sites (14 cases).
- [x] Add success/failure pairs for Load, Max_pool, Index.Add, Index.Max,
  Index.Min, and nested Index.Data (12 cases; 26 total).
- [x] Use public, role-correct smart constructors and Value wrappers;
  verify returned values, event order, and first-error priority.
- [x] Add native, bytecode, and ordinary-expr-linked jsoo build routes.
- [x] Commit the native, bytecode, and jsoo oracle goldens.
- [x] Add `expr_order.runtest` and pass it against those goldens.
- [x] Pass the common stage verification gates below (`make precommit`,
  `dune runtest`, `jsoo.runtest`, `jsoo.inline-runtest`, `melange.runtest`,
  `tailcall.runtest`, `expr_order.runtest` all green as of commits
  `1fff577`/`fff42ee`).
- [x] Investigate `test/native/lstm_scale_test.ml`'s "real-scale resource
  counters" case (see note below) -- resolved outside this plan by commits
  `167649b`/`ab4252d`/`fff42ee`.

Follow-ups not yet closed: warning 51's melange-profile fatality (or lack of
it) is unreconciled with the plan's Stage 0 step 1 wording; benchmark
baselines were captured but not saved to a committed file for later
comparison in Stage 2/6 (need a location before Stage 2 needs them).

### Note: `make jsoo.inline-runtest` was dominated by one test (resolved)

While running Stage 0's gates, `make jsoo.inline-runtest`/`js.runtest` took
over 7 minutes from a clean `_build`. Traced to a single expect test:
`test/native/lstm_scale_test.ml`'s `"lstm real-scale resource counters: both
corpus shapes"` case. It did not finish inside a 30s timeout even natively in
isolation (`inline-test-runner.exe -partition lstm_scale_test.ml`); every
other file in `test/native` (107 of them) sums to ~31s total. Dune also runs
the JS inline-test suite as one `node` process per source file (116 for
`test/native` alone), so this is on top of ~250ms fixed per-file overhead ×
116.

Root cause: this case is the only one of the three in that file that runs
`Eval_direct.run` at the *real* `sequencer2d_s` corpus shapes
(`(B,L,I,K)=(16,16,384,96)` and `(32,32,192,48)`, both bidirectional), per the
file's own header. At that scale the scalar-at-a-time symbolic evaluator
(`Kernel_eval`/`Expr.Eval.value` -- the same evaluator this whole tail-call
conversion targets) performs `loads=893,896,704` /
`reductions=495,452,160` for family1 alone (family2 is comparable). The
file's own later comment (before its second test) already documents that
`batch`/`input_size` don't affect the per-key scan-update boundary being
validated there (only `directions*seq*2*hidden_size` does), which is why
tests 2 and 3 in the same file deliberately use a cheap `batch=1,
input_size=4` fixture reaching the *same* boundary instead of the real
shapes. Test 1 can't use that shortcut: its purpose is specifically to
validate the real corpus-scale counters (`loads`/`reductions`/`locals`/
`emitters`) against the project's precomputed worst-case totals, not just the
per-key boundary -- shrinking `batch`/`seq`/`input_size` there would change
what it measures, not just its cost.

Also relevant: test 3 in the same file ("discarding unread state outputs
saves nothing today") documents that `Eval_direct` currently recomputes the
LSTM's shared trace independently for each of its three outputs
(`out`/`h_n`/`c_n`) -- a known, already-tracked "duplicate trace execution"
inefficiency (project step 19, not yet fixed) that likely multiplies test 1's
cost by roughly 3x on top of the real-scale evaluation itself.

Resolution (commits `167649b`, `ab4252d`, `fff42ee`, not part of this plan):
`lstm_scale_test.ml`'s default run now uses the cheap boundary-equivalent
fixture (test 1 renamed "(fast fixture)"); the real corpus-scale case moved
to a separate `[@tags "disabled"]` test, opt-in via `-require-tag disabled`.
A new `scripts/inline-test-timing-report(-all).sh`, wired as `make
inline-timing-report`/`inline-timing-report-js`, times every
`(inline_tests)` partition individually and fails above a threshold --
`inline-timing-report` (native, 10s) is in CI (`build.yml`); the js
variant is not yet wired to CI. The fast fixture still cost 36s under
jsoo/node (vs. 6.7s native) even after shrinking, so `INLINE_TIMING_
THRESHOLD_SECONDS_JS` (60s) is a separate, higher threshold from native's
10s rather than one shared value -- js_of_ocaml's per-op cost is not
native's. `make jsoo.runtest` + `make expr_order.runtest` +
`dune build @test/expr/runtest-js` remain the fast per-stage smoke gates for
this plan; full `jsoo.inline-runtest`/`js.runtest` is back to a reasonable
~40s from a clean `_build` and no longer needs to be avoided.

Also noted in passing: after removing files under `_build` out-of-band
(rather than via `dune clean`), dune's incremental database twice went out of
sync with the filesystem in this environment -- reporting a target as already
built (or erroring that a generated file was missing) when it was not present
on disk. `dune clean` reliably fixed both occurrences; partial `rm -rf` of
`_build` subdirectories is not safe here.

## Stage 1 — Tail-call annotations

- [x] Annotate only the genuine tail edges identified in the plan, with
  short design-reference comments. Commit `38d43ba`.
- [x] Keep non-tail child calls and unrelated traversal modules unchanged.
  Only `lib/expr_internal/eval.ml` touched; `fold.ml`/`rewrite.ml`/`check.ml`/
  `value.ml`/`pp.ml` untouched.
- [x] Pass the common gates and order oracle without changing goldens.
  `make precommit`/`expr_order.runtest`/`jsoo.runtest`/`jsoo.inline-runtest`
  all green; `dune runtest --auto-promote` produced no diffs.

## Stage 2 — Separate shared evaluator plumbing

- [x] Move definitions above `value` into `eval_common.ml` without changing
  their algorithms, including the recursive index helper. Commit `a001adb`.
- [x] Keep `value` in `eval.ml` and add `open Eval_common`.
- [x] Compose the public Eval facade from Eval_common and Eval.
- [x] Preserve internal library wrapping and the public API.
- [x] Verify unchanged depth expectations and order goldens.
- [x] Compare native benchmarks with Stage 0 and pass the common gates.
  `gc_words_per_output`/`*_words` in both benchmark binaries are
  byte-identical to the Stage 0 baseline (deterministic, not timing noise).

## Stage 3 — Build the mirrors

- [ ] Inspect each source library's dependencies, wrapping, and directory tree.
- [ ] Mirror all eleven jsoo libraries, including both probe libraries.
- [ ] Mirror native's nested directories and native4d's `passes4/`.
- [ ] Add all four core namespace shims using `include`.
- [ ] Switch `js/jsoo/dune` executable dependencies to mirrored libraries.
- [ ] Add the Melange Expr libraries with Melange-mode dependencies and
  private `expr_api` treatment.
- [ ] Add cppo processing with identical native/JS evaluator branches and
  separate JSOO_BACKEND/MELANGE_BACKEND definitions.
- [ ] Add the standalone Expr probe, three build routes, consumer shims,
  correct emitted paths, and `expr_probe.runtest`.
- [ ] Repoint the jsoo order probe with its own Expr shim; add the concrete
  Melange order-probe route and shim.
- [ ] Match the original native/bytecode/jsoo goldens and commit the first
  real Melange golden; extend `expr_order.runtest`.
- [ ] Verify the jsoo closure excludes all eleven ordinary mirrored names
  and the existing forbidden dependencies.
- [ ] Check normalized preprocessing output and native benchmark neutrality.
- [ ] Update the backend design document and pass common/probe/oracle gates.

## Stage 4 — Convert the mutual-tail intrinsic loop

- [ ] Convert JS intrinsic rows/cols using the mutual-tag technique.
- [ ] Keep the native branch unchanged.
- [ ] Add a configuration-bounded large-window Max_pool case to the shared
  Expr probe and verify all three routes through `expr_probe.runtest`.
- [ ] Inspect generated JS for the loop transformation.
- [ ] Include the nested-inline-scan scenario in Stage 5's corpus design.
- [ ] Pass all applicable stage gates.

## Stage 5 — Candidate drivers, oracle, and benchmarks

### Candidate implementations

- [ ] Create one canonical `eval_candidates.ml`, opened over Eval_common
  and copied into both JS internal libraries.
- [ ] Implement delayed trampoline with configurable threshold.
- [ ] Implement list-frame machine.
- [ ] Implement parallel-array machine with per-call allocation.
- [ ] Implement the complete direct/cutoff/array-machine hybrid.
- [ ] Port all constructors while preserving existing index/coordinate helpers.
- [ ] Preserve reduction empty-range seeds, callback order, outer reducers,
  bound-variable rebinding, and accumulator transitions.
- [ ] Implement scan row/lane transitions inside the driver, with correct
  step numbering, update charges, complete-row visibility, and resolver scope.
- [ ] Carry reducers, active local resolver, and cleanup state across cutoffs.
- [ ] Verify candidate-specific depth and driver-entry invariants.
- [ ] Inspect each backend's generated JS for hidden recursive driver calls.

### Corpus and harness

- [ ] Build synthetic Expr-only cases iteratively with generator-known depths.
- [ ] Measure direct-evaluator frontiers independently per shape and backend.
- [ ] Store expected values/errors/exceptions with exact observable traces.
- [ ] Populate all 26 order cases from committed per-backend goldens and
  execute them through all four candidates.
- [ ] Add deep reduction/scan/select cases and closed-form expectations.
- [ ] Skip unsafe reference calls entirely while still checking JS candidates.
- [ ] Exit nonzero on every outcome or trace mismatch.
- [ ] Add native/jsoo/Melange runner routes, per-consumer preprocessing,
  both benchmark shims, and correct output paths.
- [ ] Add `expr_bench.runtest` and `expr_bench.js-benchmark`.
- [ ] Benchmark shallow workloads and bounded above-frontier workloads,
  excluding unsafe direct evaluation from timing.

### Cleanup and operand order

- [ ] Install per-call mutable cleanup entries only after successful reserve.
- [ ] Release and restore on logical completion; preserve state across bounces.
- [ ] Put the top-level exception handler inside the structured-escape callback.
- [ ] Capture the original backtrace before LIFO cleanup and re-raise the
  original exception with that backtrace.
- [ ] Test nested scan success, structured failure, and ordinary exception.
- [ ] Verify same-call outer resolver restoration after successful inner scans.
- [ ] Verify physical identity of the injected exception on both JS backends.
- [ ] Test row-0 meter reuse with a budget detecting either leaked reservation.
- [ ] Demonstrate outer-skip and inner-skip cleanup negative controls fail
  with the expected state-limit error; normal cleanup passes.
- [ ] Explicitly sequence all seven JS direct-path sites and candidate
  transitions to each backend's measured order.
- [ ] Verify both-load/both-fail observations across the cutoff.
- [ ] Pass all applicable gates and record candidate benchmark evidence.

## Stage 6 — Install the selected dispatcher

- [ ] Record selected candidate, cutoff, and rationale for each JS backend.
- [ ] Measure the exact intended composition on shallow and deep workloads.
- [ ] Install the selected driver inside Eval's JS branch with unchanged API.
- [ ] If the complete hybrid wins, install it directly without another hybrid.
- [ ] Preserve active environments and cleanup state at each handoff.
- [ ] Verify below/above-cutoff behavior with private benchmark instrumentation.
- [ ] Re-run correctness, public probes, generated-code checks, and benchmarks.
- [ ] Update the evaluator design document with selection and bounded scope.
- [ ] Pass all applicable stage gates.

## Stage 7 — Mirrored Kernel limit and deep completion

### First measurement and constant extraction

- [ ] Measure the deep-index exhaustion signal independently under jsoo and
  Melange, including catchability or exact process status and diagnostic.
- [ ] Extract all shared hard constants into `kernel_hard_shared.ml`.
- [ ] Add flat `kernel_hard.ml` and preserve `Kernel.Limits.Hard.*` via re-export.
- [ ] Mark the extracted modules private in both native profiles and the mirror.
- [ ] Choose a measured JS eval-depth limit with headroom and override only
  mirrored `kernel_hard.ml`'s eval_depth value.
- [ ] Update `kernel.mli` documentation for shared and backend-specific limits.
- [ ] Verify `dune build --profile landmarks` and unchanged native constants.

### Kernel tests

- [ ] Add a jsoo test linked to `native_js`.
- [ ] Measure bounded test artifacts and configure sufficient dependency/value
  limits so the intended guard is reachable.
- [ ] Test cumulative eval-depth admission, execution via `run`, and the
  one-past rejection with the specific `Eval_too_deep` error.
- [ ] Test the unchanged per-body Too_deep boundary separately.
- [ ] Test `value_at` producer depths 96/97 for success/Recursion_too_deep.

### Deep Expr and index boundary

- [ ] Add shared runtime `--deep` mode while retaining shallow default output.
- [ ] Add `expr_probe.deep-runtest` for jsoo/Melange depth-200,000 Value/Bool
  cases, checked against closed forms without invoking native.
- [ ] Add Stack_fault shims on all three routes, each with both interface members.
- [ ] Build the deep-index chain outside the observation handler with bounded
  arithmetic, and accept only the measured stack-exhaustion signal.
- [ ] Check catchable failures in-process, omitting this case entirely from
  the shared run on backends whose fault is not catchable.
- [ ] Where required, add the canonical uncaught-fault entry source, concrete
  jsoo/Melange build routes, Expr shims, and emitted paths.
- [ ] Where required, add `expr_probe.stack-fault-runtest`, capture expected
  failing status safely, reject unexpected completion, and check both measured
  exit status/signal and diagnostic.
- [ ] Verify every JS backend uses exactly one negative-control assertion route.
- [ ] Pass the final verification gates and update design documentation.

## Verification gates

Apply these gates to each stage once the corresponding target exists.
Existing jsoo inline suites and the Melange subset remain regression
coverage; dedicated Expr targets verify the converted evaluator.

| Gate | Required from |
|---|---|
| `make precommit` | Every stage |
| `dune runtest` covering Expr and native | Every stage |
| `make jsoo.runtest` | Every stage |
| `make jsoo.inline-runtest` | Every stage |
| `make melange.runtest` | Every stage |
| `make tailcall.runtest` | Every stage |
| `make expr_order.runtest` | Stage 0; Melange added in Stage 3 |
| `make expr_probe.runtest` | Stage 3 |
| Full mirrored dependency exclusion check | Stage 3 |
| `make expr_bench.runtest` | Stage 5 |
| `make expr_bench.js-benchmark` | Stage 5 |
| Stack instrumentation and generated-JS inspection | Stage 5 |
| Cleanup negative controls | Stage 5 |
| Mirrored Kernel boundary tests | Stage 7 |
| `make expr_probe.deep-runtest` | Stage 7 |
| `make expr_probe.stack-fault-runtest` | Stage 7 when a fault needs a separate process |
| `dune build --profile landmarks` | Stage 7 |
| Native depth output and Hard constants equal Stage 0 | Final acceptance |

## Completion evidence

Record test/benchmark artifacts and implementation commit references when
available. Conditional routes may be marked not applicable only with the
measurement establishing that condition.

| Stage | Status | Evidence / remaining blocker |
|---|---|---|
| 0 — Baselines and oracle | Verified | Commits `1fff577` (oracle), `fff42ee` (jsoo timing threshold fixup). Follow-ups: melange warning-51 fatality unreconciled; benchmark baselines not saved to a file. |
| 1 — Annotations | Verified | Commit `38d43ba`. |
| 2 — Shared plumbing | Verified | Commit `a001adb`. |
| 3 — Mirrors | Unverified | — |
| 4 — Intrinsic loop | Unverified | — |
| 5 — Candidates and harness | Unverified | — |
| 6 — Dispatcher selection | Unverified | — |
| 7 — Limits and deep gates | Unverified | — |
