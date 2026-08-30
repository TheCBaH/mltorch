# Region foundation implementation log

This file is the live execution record for `region-foundation-plan.md`.  Each
completed gate is committed only after its focused validation succeeds.

## Progress

| Gate | Scope | Status | Commit |
|---|---|---|---|
| 0 | Baseline and benchmark census | in progress | benchmark driver checkpoint pending commit |
| 1 | `Expr` scalar-local leaf | complete | pending commit |
| 2 | Region structure and validation | complete | pending commit |
| 3 | Kernel representation migration | complete | pending commit |
| 4 | Region reference execution | complete | `ccc7b1d` |
| 5 | Pixel specialization/reconstruction | complete | `83397a7` |
| 6 | Pixel regression closeout | pending | |

## Decisions

- Start at Gate 1: the repository has no pre-existing Region implementation;
  the plan's Gate 0 benchmark command was added after the language boundary
  stabilized. A true pre-migration run was not captured, so its final
  before/after comparison remains open.
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
