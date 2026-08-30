# Region foundation implementation log

This file is the live execution record for `region-foundation-plan.md`.  Each
completed gate is committed only after its focused validation succeeds.

## Progress

| Gate | Scope | Status | Commit |
|---|---|---|---|
| 0 | Baseline and benchmark census | pending | |
| 1 | `Expr` scalar-local leaf | complete | pending commit |
| 2 | Region structure and validation | complete | pending commit |
| 3 | Kernel representation migration | complete | pending commit |
| 4 | Region reference execution | in progress | reference evaluator checkpoint pending commit |
| 5 | Pixel specialization/reconstruction | pending | |
| 6 | Pixel regression closeout | pending | |

## Decisions

- Start at Gate 1: the repository has no pre-existing Region implementation;
  the plan's Gate 0 benchmark command will be added once the language boundary
  is stable enough to avoid churn.
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
  to slot map. Kernel evaluator dispatch remains the next Gate 4 change.

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
