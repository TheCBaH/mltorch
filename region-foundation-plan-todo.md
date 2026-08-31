# Region foundation implementation log

This file is the live execution record for `region-foundation-plan.md`.  Each
completed gate is committed only after its focused validation succeeds.

## Progress

| Gate | Scope | Status | Commit |
|---|---|---|---|
| 0 | Baseline and benchmark census | complete | historical comparison recovered and recorded |
| 1 | `Expr` scalar-local leaf | complete | pending commit |
| 2 | Region structure and validation | complete | pending commit |
| 3 | Kernel representation migration | complete | pending commit |
| 4 | Region reference execution | complete | `ccc7b1d` |
| 5 | Pixel specialization/reconstruction | complete | `83397a7` |
| 6 | Pixel regression closeout | complete | audit, alternating comparison, and full verification pass |

## Post-Foundation handoff (not a Foundation gate)

The work below is specified by
[`region-native-implementation-guide.md`](region-native-implementation-guide.md).
It is deliberately not added as Gates 7+ here: this file records the bounded
Foundation task, and its completed-gate evidence must remain interpretable on
its own.

The Foundation implementation is complete.  Its scalar-local language,
validated Region representation, reference executor, Pixel specialization, and
Pixel no-regression evidence are ready for the separate Region-native and
locality-scheduling work described below.  This completion does not claim a
production Region-native lowering or an operation speedup.

The post-Foundation implementation has two independent tracks:

- **Region-native shared work:** a provenance-aware optional regionizer builds
  a non-degenerate program, proves reconstruction, then uses a dedicated Region
  lowerer/executor.
- **Locality scheduling:** footprint analysis tiles Pixel-form computations
  without changing their semantic Region partition; a lowering may reify a
  block-local Region program only when its local lifetime must be represented
  and checked.

Neither track changes the recorded Foundation APIs, validation results, or
Pixel no-regression acceptance criteria.

## Decisions

- The permanent benchmark driver landed with the Region migration, so it could
  not have recorded the original tree at the time.  For closeout, the exact
  pre-Region commit `261222c` was checked out with its pinned submodules and
  given a temporary, source-identical driver adapted only from
  `Value.computation = Region_program.pixel body` to `Value.body = body`.
  This recovers a direct measurement of the former Pixel evaluator without
  changing either benchmark workload.
- Preserve the plan's one-AST design: `Expr.Value.Local` is a scalar leaf and
  `Region_program` owns its bindings and scope.
- Keep the existing immutable `Expr.Builder` identity supply. Local and reducer
  identity types are distinct, so their ordinals can share that supply safely.
- `Expr.Check.value` remains the closed-expression boundary. `fragment` is the
  only API that admits local references, and requires an explicit allow-list.
- `Rewrite.substitute_locals` freshens every inserted replacement in the
  destination builder namespace, matching the existing capture-safe load
  substitution discipline.
- `Region_partition` uses the engine's existing `Vec6` canonical N/T/D/H/W/C
  traversal. Whole axes are normalized only in derived region keys; output
  enumeration always starts from the original output shape.
- Region source analysis folds local bodies and the emitter structurally. It
  never expands local references merely to discover dependencies or limits.
- `Kernel.Value` now owns only `computation : Region_program.t`. Existing stage
  adaptation applies `Region_program.pixel` directly, preserving the original
  expression object and keeping operation modules unchanged.
- Kernel elaboration and fusion remain Pixel-only. A non-degenerate Region
  value is visible through structural dependencies but cannot enter existing
  expression-substitution paths.
- `Region_eval` caches the dense scalar-local array by canonical region key;
  the cache is created once per materialization and reuses a validated local-id
  to slot map.
- `Kernel_eval` classifies each computation once before output allocation.
  Pixel values retain the pre-existing direct `Expr.Eval` callback; Region
  values apply result conversion only to the emitter, then use `Region_eval`.
  This keeps the Pixel hot path free of Region-key and local-array work while
  allowing stored Region values to read the same prior-value buffers as Pixel
  values.
- Pixel specialization preflights substituted size and depth structurally. The
  metered Expr fold treats each local leaf as its already-computed expanded
  measure, so it stops at the caller's limits without allocating a duplicated
  tree. Only a successful preflight starts the shared-state, capture-safe
  rewrite.
- The recursive evaluator ceiling is 96 producer transitions. Region dispatch
  adds enough fixed stack cost that the former 128-transition frontier can
  overflow under js_of_ocaml for medium-depth bodies; the lower ceiling returns
  the existing typed runtime error on both backends instead.
- Model Explorer detail renders a Region as separate expression roots for each
  declared local and the emitter. Root ids carry stable `lN-e*` / `emit-e*`
  prefixes, and open-expression attributes name local references by declaration
  order instead of expanding them.
- Pixel execution is selected before tensor allocation. Its materialization
  callback closes over the already-converted `Expr.Value.t` and calls only
  `Expr.Eval.value`; Region classification, key derivation, local slots, and
  arrays occur only in the separate Region branch.
- `Kernel_eval.run` exposes an opt-in ordinary-load observer solely for audit
  tests. It is resolved into a load function once at evaluator setup, leaving
  the normal Pixel loop without an observer branch; the test pins C-innermost
  order and one input load per output cell.
- The recovered `261222c` benchmark and the final Foundation executable were
  each built once with Dune, then launched directly in alternating processes.
  Across three 20-sample runs per side, the median-of-medians was 0.425 ms
  versus 0.412 ms for identity and 0.749 ms versus 0.742 ms for add.  Output
  and load counts were unchanged; final GC words per output cell decreased
  from 214.258 to 193.282 (identity) and 341.322 to 303.300 (add).
- The permanent benchmark also runs one 3x3 convolution-shaped coordinate
  traversal (7,200 outputs and 64,800 loads) and one I64 gather/index traversal
  (64 outputs and 64 ordinary value loads) as deterministic scan smoke cases.

## Validation record

| Date | Gate | Command | Result | Notes |
|---|---|---|---|---|
| 2026-08-30 | 1 | `opam exec -- dune runtest test/expr` | pass | Includes local scope, evaluation, printing, measurement, and substitution coverage. |
| 2026-08-30 | 1 | `opam exec -- dune build lib/expr lib/native` | pass | Exhaustive consumers updated, including grounding and Model Explorer detail labels. |
| 2026-08-30 | 2 | `opam exec -- dune runtest test/expr test/native` | pass | Covers partition enumeration, local scope, invariance, and all existing native behavior. |
| 2026-08-30 | 2 | `make build` | pass | Native Region modules compile in the normal repository build. |
| 2026-08-30 | 3 | `opam exec -- dune runtest test/native test/model_explorer` | pass | Kernel, fusion, evaluator, and explorer compatibility coverage passed after the representation migration. |
| 2026-08-30 | 3 | `make precommit` | pass | Ran after formatter promotion; includes build, full native test run, format verification, diff checks, and file-size checks. |
| 2026-08-30 | 4 | `make precommit` | pass | Reference evaluator and its whole-axis sharing test pass with the repository's full pre-commit suite. |
| 2026-08-30 | 4 | `opam exec -- dune runtest test/native` | pass | Kernel-level whole-channel Region program agrees through buffered `run` and on-demand `value_at`. |
| 2026-08-30 | 5 | `opam exec -- dune runtest test/expr test/native` | pass | Dependent-local specialization, reconstruction, and one-node-over-size rejection pass. |
| 2026-08-30 | 4 | `make jsoo.runtest` | pass | Native and JavaScript probe output matches. |
| 2026-08-30 | 4 | `make jsoo.inline-runtest` | pass | Region-dispatch-adjusted recursion guard passes the JS depth frontier without stack overflow. |
| 2026-08-30 | 0 | `make benchmark.region_pixel` | pass | `region_pixel_bench`: 20 samples on OCaml 4.14.3, Linux aarch64; identity median 0.413 ms / 210.246 GC words per output, add 0.671 ms / 337.328; both over 2,048 cells. |
| 2026-08-30 | 4–5 | `opam exec -- dune runtest test/expr test/native` | pass | A Region consumer of a stored Pixel producer agrees across `run`, default `run_plan`, and `value_at`; Pixel specialization retains the original expression and negative reconstruction returns `false`. |
| 2026-08-30 | 6 | `opam exec -- dune runtest test/model_explorer` | pass | Region detail includes a root for each scalar local and a separate emitter root while existing Pixel detail remains unchanged. |
| 2026-08-30 | 6 | `make jsoo.runtest && make jsoo.inline-runtest` | pass | Native/JavaScript probe output matches and all JS inline expectations pass after Region detail rendering. |
| 2026-08-30 | 6 | `make benchmark.region_pixel` | pass | Hot-path audit run: identity 0.378 ms, add 0.679 ms; fixed output/load counts remain 2,048/2,048 and 2,048/4,096 respectively. |
| 2026-08-30 | 0/6 | isolated `261222c` benchmark build | unavailable at the time | The initial worktree lacked the commit's `err_trace` submodule; superseded by the pinned-submodule recovery below. |
| 2026-08-30 | 6 | `opam exec -- dune runtest test/native` | pass | Pixel hot-path audit test observes exactly three loads in canonical C order `0,1,2`. |
| 2026-08-31 | 0/6 | direct executables from `261222c` and final Foundation tree | pass | OCaml 4.14.3 / Dune 3.24.2, Linux 7.0.0-28-generic aarch64; three alternating baseline/final process pairs, three warmups and 20 samples each. Identity: 0.425 -> 0.412 ms (-3.1%); add: 0.749 -> 0.742 ms (-0.9%). Both retain 2,048 output cells and 2,048/4,096 input loads; GC words/output decrease as recorded above. |
| 2026-08-31 | 0/6 | `_build/default/bin/region_pixel_bench.exe` | pass | Final direct executable: identity 0.426 ms and add 0.679 ms; the convolution smoke reports 7,200 outputs / 64,800 loads and the indexing smoke reports 64 outputs / 64 ordinary value loads. |
| 2026-08-31 | 6 | `make format && make build && make runtest && make jsoo.runtest && make jsoo.inline-runtest && make precommit` | pass | Format, native suite, native/JS probe agreement, JS inline expectations, whitespace/file-size checks, and the pre-commit aggregate all pass. |
