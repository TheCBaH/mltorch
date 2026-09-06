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

- [ ] Confirm warning 51 is fatal under dev and Melange profiles; demonstrate
  failure with a temporary non-tail annotation and remove it.
- [ ] Save native region-pixel and region-compute benchmark baselines.
- [ ] Save native depth-probe output and hard constants: 256 / 1536 / 96.
- [ ] Add canonical `js/probe/order_probe.ml` using the real evaluator and
  logging load/index-load callbacks.
- [ ] Cover both-success and both-failure cases for all seven rewritten
  order-sensitive call sites (14 cases).
- [ ] Add success/failure pairs for Load, Max_pool, Index.Add, Index.Max,
  Index.Min, and nested Index.Data (12 cases; 26 total).
- [ ] Use public, role-correct smart constructors and Value wrappers;
  verify returned values, event order, and first-error priority.
- [ ] Add native, bytecode, and ordinary-expr-linked jsoo build routes.
- [ ] Commit the native, bytecode, and jsoo oracle goldens.
- [ ] Add `expr_order.runtest` and pass it against those goldens.
- [ ] Pass the common stage verification gates below.

## Stage 1 — Tail-call annotations

- [ ] Annotate only the genuine tail edges identified in the plan, with
  short design-reference comments.
- [ ] Keep non-tail child calls and unrelated traversal modules unchanged.
- [ ] Pass the common gates and order oracle without changing goldens.

## Stage 2 — Separate shared evaluator plumbing

- [ ] Move definitions above `value` into `eval_common.ml` without changing
  their algorithms, including the recursive index helper.
- [ ] Keep `value` in `eval.ml` and add `open Eval_common`.
- [ ] Compose the public Eval facade from Eval_common and Eval.
- [ ] Preserve internal library wrapping and the public API.
- [ ] Verify unchanged depth expectations and order goldens.
- [ ] Compare native benchmarks with Stage 0 and pass the common gates.

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
| 0 — Baselines and oracle | Unverified | — |
| 1 — Annotations | Unverified | — |
| 2 — Shared plumbing | Unverified | — |
| 3 — Mirrors | Unverified | — |
| 4 — Intrinsic loop | Unverified | — |
| 5 — Candidates and harness | Unverified | — |
| 6 — Dispatcher selection | Unverified | — |
| 7 — Limits and deep gates | Unverified | — |
