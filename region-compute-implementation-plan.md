# Region computation implementation plan

## Status and boundary

The Region Foundation is complete.  This plan implements the next bounded
slice of [`region-compute-design.md`](region-compute-design.md): **Native
scalar-region sharing for RMSNorm, LayerNorm, and plain Softmax**, with a
dedicated Region execution path.

The work is deliberately narrower than the complete design.  It does not add
Loop IR as a general backend, graph fusion, physical tiling, `Block`
partitions, shaped locals, multi-output Region emission, relaxed rounding, or
SDPA.  Those remain separately gated follow-ons after this slice establishes a
correct and measurable Region-native path.

The Foundation contracts are fixed inputs to this work:

- `Compute(S).pixel` remains the sole operation semantic source and fallback.
- `Kernel.Value.computation` remains one `Region_program.t` per logical value.
- `Region_program.pixel` keeps every unselected operation on the existing
  Pixel loop.
- a converter must pass `Region_program.check` and `reconstructs` before it is
  admitted; rejection retains the exact original Pixel expression.
- strict mode preserves current ordered reductions, expression order, and the
  one final `Kernel.Result_conversion` boundary.

## Completion contract

This slice is complete only when all of the following are true:

1. A provenance-aware, optional regionizer runs before operation identity is
   discarded and can replace an eligible Pixel `Kernel.Value` computation.
2. The Region path has a dedicated lowered/executor representation.  It
   traverses keys once, evaluates scalar locals once per key in declaration
   order, and emits each owned output once.  It does not recover sharing by
   repeatedly calling `Region_eval.value_at`.
3. RMSNorm, LayerNorm, and plain Softmax each construct a non-degenerate
   program from their existing Pixel expression, preserve it on every
   rejection, and prove reconstruction before admission.
4. Every converted operation is bitwise equal to the existing Pixel,
   `Eval_direct`, and `Stage_program.ground` oracles in strict mode, including
   degenerate dimensions and adversarial floating-point inputs.
5. Instrumented tests prove `Theta(K)` reduction work per normalized slice or
   Softmax row, rather than the former repeated per-output reductions.
6. Unconverted Pixel values retain the Foundation hot path and its benchmark
   guarantees.  Native and JavaScript validation remain green.

## Non-negotiable design rules

- Do not add a second operation implementation, a Region arm to
  `Semantics.SEMANTICS`, or AST-pattern matching over pretty-printed formulas.
- Do not use scalar locals as f32 storage.  They remain working floats and
  result conversion applies only to the emitter.
- Do not add `Block`, vector/tensor locals, tree/parallel reductions, Welford,
  cached Softmax exponentials, or an online attention algorithm in this plan.
- Do not make Region conversion the default merely because a candidate exists:
  validation, reconstruction, limits, and the lowering admission check all
  have to succeed.
- Keep the Region reference evaluator as an oracle.  The production path must
  be separately lowered and observable in tests.

## Gate 0 — contract census and measurement harness

### Work

1. Re-census every `Kernel` construction and execution entry point, especially
   `Eval_symbolic`, `Kernel_adapt`, `Kernel_eval`, fusion, Model Explorer, and
   the Native/Native4D test helpers.
2. Record the exact graph-operation provenance available at each boundary.
   Select a private regionization seam before writing a converter; the public
   `Stage_program.t` and existing `Kernel_adapt.of_stage_program` callers must
   retain their Pixel-only behavior by default.
3. Add deterministic counters to the test-only Region execution seam for
   region keys, local evaluations, emitter evaluations, input loads, and
   reduction iterations.  Counters are assertions, not wall-time tests.
4. Add representative RMSNorm, LayerNorm, and Softmax fixtures at several
   normalized extents, including multiple independent keys.  Benchmark the
   existing Pixel form separately from the later Region form.

### Acceptance

- Existing suites pass unchanged.
- The census names the selected provenance seam and all direct consumers that
  need Region-awareness.
- Baseline counter expectations demonstrate repeated Pixel reductions and are
  committed as the red-to-green target for the operation gates.

## Gate 1 — provenance-aware optional regionizer boundary

### Work

1. Add a Native-private `Regionizer` interface and typed rejection result.
   Its input includes the typed operation provenance, validated parameter and
   operand/signature context, output shape, and the already-built Pixel
   expression.
2. Add a `Kernel_adapt` entry point or private callback that accepts admitted
   computation overrides while preserving `of_stage_program` as the existing
   all-Pixel adapter.  Build the candidate map from `Graph_ir`/`Eval_op`
   provenance; do not add operation tags to `Kernel.Value`.
3. Centralize admission: validate with the Kernel limits, require
   `Region_program.reconstructs` against the original expression, and retain
   the original `Region_program.pixel` on every typed rejection.
4. Keep source/limit/fusion analysis structural.  No analysis may specialize a
   candidate merely to discover its dependencies.

### Tests

- A candidate can replace exactly its nominated logical value without changing
  unrelated values or output ordering.
- Rejection for unsupported operation, missing provenance, mismatched shape,
  failed reconstruction, and too-tight limits leaves the original expression
  object in the resulting Pixel program.
- Region candidates retain their ordinary `Kernel` dependency, fusion, and
  Model Explorer behavior without entering Pixel-only substitution paths.

### Acceptance

The seam carries no semantic operation implementation: it only admits a
validated `Region_program.t` built from the existing Pixel expression.

## Gate 2 — dedicated scalar-Region lowering and executor

### Work

1. Add backend-neutral Region execution IR and lowering for the existing
   scalar-local/`Singleton`/`Whole` Foundation subset.  Lowering classifies a
   value once: Pixel stays on the current `Expr.Eval` materialization loop;
   non-degenerate Region produces key traversal, local steps, emitter steps,
   and the final store conversion.
2. Compile/specialize each local body and emitter ahead of the loops.  The
   executor may share expression evaluation support, but must not allocate a
   local map per output, call an opaque per-output Region callback, or replay
   `Region_eval.value_at` to obtain sharing.
3. Preserve `Vec6` N/T/D/H/W/C traversal and C-innermost emission.  Enumerate
   scalar locals once per canonical key and use a validated local-id/slot map.
4. Add explicit trace/counter hooks available only to tests.  They must prove
   once-per-key local evaluation and one emitter evaluation/store per output.

### Tests

- Synthetic whole-C and multi-axis Region programs agree bitwise between
  `Region_eval`, the new executor, `Kernel_eval.run`, and `value_at`.
- Local and emitter counters match key/output cardinalities exactly.
- Result conversion, signed zero, infinities, NaNs, and f32 rounding occur at
  the same logical boundary as Pixel execution.
- Pixel audit tests continue to show no Region allocation or branch in the
  per-output callback.

### Acceptance

`Region_eval` is no longer the materialization mechanism for an admitted
non-degenerate program, but remains its differential oracle.

## Gate 3 — RMSNorm regionizer

### Work

1. Regionize only `Norm.RmsNorm` values with valid `params.dims`, using `Whole`
   on those axes and `Singleton` on the others.
2. Reuse the existing Pixel tree to form ordered locals: `sumsq`, then the
   dependent `mean_square`/`inv` arithmetic.  Preserve normalized-axis order,
   epsilon placement, input/weight coordinate projections, and optional-weight
   behavior exactly.
3. Admit only after limits, local invariance, and reconstruction pass.  Keep
   the unmodified Pixel program otherwise.

### Tests and measurements

- Single- and multi-axis normalization, one-extent axes, multiple keys,
  no-weight and weighted forms, several epsilons, signed zero, NaNs, and
  infinities.
- Bitwise comparison with Direct, Symbolic grounding, Pixel Kernel, Region
  reference, and dedicated Region executor.
- For `R` slices of normalized extent `K`, counters show `R*K` ordered square
  terms and `R*K` emissions, not `R*K*K` terms.

### Acceptance

No RmsNorm `Compute.pixel`, importer, Graph IR, shape rule, or parameter type
changes.  The resulting Region form is strict-mode `Identical`.

## Gate 4 — LayerNorm regionizer

### Work

1. Regionize `Norm.LayerNorm` over `params.dims` with ordered locals for sum,
   mean, variance sum, variance, and inverse standard deviation.
2. Keep the existing two-pass stable variance formula.  Do not substitute
   `E[x^2] - E[x]^2`, Welford, or reassociated arithmetic.
3. Preserve optional affine weight/bias semantics and their existing coordinate
   projection.

### Tests and measurements

- The RMSNorm matrix plus weighted/bias variants, negative values, tiny and
  large epsilons, and multi-axis reductions.
- A dependent-local test proves the variance local can read the earlier mean
  without re-expanding it.
- Counters show two ordered `R*K` reduction traversals and `R*K` emissions.

### Acceptance

LayerNorm is bitwise `Identical` under the same strict validation matrix and
falls back to the exact Pixel computation on every rejection.

## Gate 5 — plain Softmax regionizer

### Work

1. Regionize only `Reduce.Softmax` (not `_safe_softmax`, `log_softmax`, or
   SDPA), selecting `Whole` on `params.axis`.
2. Hoist ordered maximum and denominator locals from the existing Pixel
   expression.  The emitter recomputes its exponent from the shared maximum;
   no shaped exponential cache is introduced.
3. Preserve every supported axis and current all-`-inf` behavior.

### Tests and measurements

- Every axis, multiple independent rows, extent one, extreme finite values,
  signed zero, infinities, NaNs, and all-`-inf` rows.
- Bitwise agreement across all established oracles.
- Counters show one max pass, one denominator pass, and one emission pass per
  row: `Theta(K)`, not `Theta(K^2)` repeated reduction work.

### Acceptance

The converter has no effect on SDPA or safe-softmax semantics and rejects them
explicitly rather than approximating their zero-row rule.

## Gate 6 — closeout and production admission

### Work

1. Run the full Native, Model Explorer, Native4D, and JavaScript suites,
   including the current Pixel benchmark and all new operation fixtures.
2. Benchmark Pixel and admitted Region forms in alternating processes.  Record
   wall time, allocation, loads, reduction iterations, key/local/emitter counts,
   machine/compiler data, and sample count.  Treat operation speedup as an
   observed result, not a substitute for counter evidence.
3. Review the lowered executor for per-output allocation/dispatch and confirm
   unconverted Pixel values still take the Foundation fast path.
4. Update the design, guide, and live TODO with final module names, rejection
   policy, performance evidence, and the explicit statement that this is only
   scalar Region-native work.

### Acceptance

All three converters are strict-mode correct and independently measurable; the
dedicated executor is the only materialization path for admitted Regions; no
other operation is silently converted.

## Deferred follow-ons

These need separate plans and must not be pulled into Gates 0–6:

1. general Loop IR and Native4D routing;
2. complete all-or-nothing whole-graph Region DAG lowering with preserved
   stage rounding;
3. `Relax_internal_f32` and an explicit `Equivalent` numerical policy;
4. shaped locals, local placement, physical tiles, halo/footprint analysis,
   and static stencil fusion;
5. efficient SDPA (score/probability rows, vector accumulators, blocked online
   softmax, and contraction specialization);
6. schedule search, vectorization, parallel reductions, and combine/finalize
   reduction splitting.
