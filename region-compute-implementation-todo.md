# Region computation implementation log

This is the live execution record for
[`region-compute-implementation-plan.md`](region-compute-implementation-plan.md).
It tracks the scalar Region-native slice only; it is not a tracker for Loop IR,
tiling, graph fusion, or SDPA follow-ons.

## Scope lock

- Included: a provenance-aware optional regionizer, dedicated scalar-Region
  lowering/execution, and strict-mode RMSNorm, LayerNorm, and plain Softmax.
- Excluded: changes to `Compute(S).pixel`, `Semantics.SEMANTICS`, Graph IR,
  `Block`, shaped locals, multi-output Region programs, relaxed rounding,
  general Loop IR, graph fusion, locality scheduling, and SDPA.
- Fallback rule: every rejection retains the original `Region_program.pixel`
  and its existing Pixel execution path.

## Progress

| Gate | Scope | Status | Commit |
|---|---|---|---|
| Foundation prerequisite | Region language and Pixel closeout | complete | `2a81e7b`; closeout `787f278` / `c50be5b` |
| 0 | Contract census and baseline counters | complete | `c8fe41f` |
| 1 | Provenance-aware optional regionizer boundary | complete | `c8fe41f` |
| 2 | Dedicated scalar-Region lowering and executor | **reopened** — correctness complete, cost contract unmet | `c8fe41f` |
| 3 | RMSNorm regionizer | complete | `c8fe41f` |
| 4 | LayerNorm regionizer | complete | `c8fe41f` |
| 5 | Plain Softmax regionizer | complete | `c8fe41f` |
| 6 | Closeout and production admission | **reopened** — performance evidence insufficient | `c8fe41f` |

### Reopening record — quadratic owned-output enumeration

Found during design review after closeout. Gates 2 and 6 are reopened for
performance only; every numerical, structural, and fallback result they
recorded stands.

**Defect.** `Region_execution.materialize` enumerates a key's owned outputs
through `Region_partition.fold_outputs`, which runs `Vec6.fold_coords` over the
**entire output shape** once per key and keeps the coordinates that match. For
`R` regions of extent `K` the traversal is `R * (R * K)` = `R^2 * K`, against
`R * K^2` for the Pixel path it replaces. The Region executor is therefore
faster only while `R < K`; normalization over `C` with `R = N*T` slices runs at
`R > K`. The slice moved a quadratic rather than removing one.

`Region_eval.materialize` — the reference oracle this executor was built to
replace as a materialization mechanism — does not have the defect: one domain
pass with a key-indexed local cache.

**Why the gates passed.** Neither closed artifact could observe it.

- `bin/region_compute_bench.ml:15` fixes the shape at
  `~n:1 ~t:1 ~d:1 ~h:1 ~w:4 ~c:extent`, so `R = 4` at every measured extent and
  only `K` varies. The recorded extent-64 results (RMSNorm 4.637 → 0.209 ms,
  LayerNorm 299.533 → 0.316 ms, Softmax 152.119 → 0.179 ms) are valid at
  `R = 4` and do not generalize.
- `test/native/region_compute_test.ml` fixtures use `R = 2`.
- The counters (`keys`, `locals`, `emitters`, `loads`, `reductions`) do not
  count a membership test. "Region reductions were 256, 512, and 512, exactly
  linear" was true while total traversal work was quadratic.

**Fix.** Construct owned coordinates from the key plus the `Whole`-axis extents
instead of scanning and filtering. One function; no representation, validation,
or numerical change. `fold_outputs` can also drop its `Err.t` result, whose
`` `Region_key_out_of_bounds `` case `fold_keys` already makes unreachable.

**Re-closure requires.** Enumeration derived from the key; a benchmark axis
sweeping `R` at fixed `K` with at least one `R > K` shape; re-measured
Pixel/Region timings across that sweep; and either a visit counter or wall-time
evidence, since the existing counter set cannot show it.

## Gate checklist

### Gate 0

- [x] Record the selected provenance seam and all affected Kernel consumers.
- [x] Add deterministic counters for keys, local/emitter evaluations, loads,
  and reduction iterations.
- [x] Add baseline fixtures and benchmarks for RMSNorm, LayerNorm, and
  Softmax across several reduction extents.
- [x] Record compiler, machine, sample count, and baseline measurements.

### Gate 1

- [x] Define the private regionizer/candidate/rejection interface.
- [x] Thread typed operation provenance to that boundary without adding it to
  `Kernel.Value` or changing default `of_stage_program` behavior.
- [x] Centralize validation, reconstruction, limit, and Pixel-fallback rules.
- [x] Test accepted replacement and every rejection/fallback case.

### Gate 2

- [x] Define lowered scalar-Region execution IR.
- [x] Lower a non-degenerate Foundation program once per logical value.
- [x] Execute locals once per key and the emitter once per owned output.
- [x] Retain `Region_eval` only as an oracle, not a materialization mechanism.
- [x] Prove traversal, result conversion, counters, and Pixel hot-path
  preservation on Native and JavaScript.
- [ ] Enumerate owned outputs from the key and the `Whole`-axis extents, not by
  filtering a full-domain scan per key.
- [ ] Show materialization cost scales with the output domain rather than with
  `keys * domain`.

### Gate 3

- [x] Implement RMSNorm selection, `Whole params.dims`, and ordered locals.
- [x] Preserve optional weight, epsilon, normalized-axis order, and fallback.
- [x] Prove reconstruction and bitwise oracle agreement.
- [x] Record linear-per-slice reduction counters and benchmark evidence.

### Gate 4

- [x] Implement LayerNorm sum/mean/variance/inverse local sequence.
- [x] Preserve stable two-pass variance and optional affine operands.
- [x] Prove dependent-local scope, reconstruction, and bitwise agreement.
- [x] Record two linear reduction passes per slice.

### Gate 5

- [x] Implement plain Softmax max/denominator local sequence for every axis.
- [x] Preserve all-`-inf` behavior and explicitly reject safe-softmax/SDPA.
- [x] Prove reconstruction and bitwise agreement for adversarial rows.
- [x] Record linear max/denominator/emission work per row.

### Gate 6

- [x] Alternate Pixel/Region benchmarks and record counts, allocation, and
  timing evidence.
- [x] Run full Native, Model Explorer, Native4D, and JavaScript verification.
- [x] Audit the lowered Region and unconverted Pixel hot paths.
- [x] Update design records with final APIs, measurements, and deferrals.
- [ ] Sweep region count `R` at fixed extent `K`, including an `R > K` shape,
  and re-record Pixel/Region timings across that sweep.
- [ ] State, next to each recorded counter set, which costs it does not observe.

## Decisions and evidence

- `Eval_symbolic.run_regionized` is the selected provenance seam. It retains
  each typed `Graph_ir.op`, operands, output ordinal, symbolic Pixel expression,
  and any synthetic optional-operand constants until `Regionizer` dispatches to
  the typed operation's Region constructor. `Norm.RmsNorm.Region`,
  `Norm.LayerNorm.Region`, and `Reduce.Softmax.Region` own the declarative
  operation-specific programs; `Regionizer` owns only that provenance dispatch.
  `Eval_symbolic.run` and `Kernel_adapt.of_stage_program` remain
  Pixel-only by default; `Region_kernel.of_graph` opts into candidate admission.
- `Kernel_adapt.Region_admission` is the sole admission boundary. An
  unsupported, invalid, reconstruction-failing, or limit-rejected candidate
  retains the original `Region_program.pixel` expression object.
- `Region_execution.lower` distinguishes the unchanged Pixel loop from a
  lowered scalar Region program. `Region_execution.materialize` enumerates
  keys, evaluates fixed local slots in declaration order, then writes each
  owned output; it does not materialize through `Region_eval`. `Region_eval`
  remains the on-demand/reference oracle.
- The executor's only mutable state is scoped to its destination tensor,
  fixed scalar-local array, and optional test counters. The first two are the
  required storage for key-first traversal; counters are omitted from ordinary
  execution. `Region_execution` documents these as the explicit mutation
  exceptions; no mutable reference is used for ordinary Region control flow.
- RMSNorm uses `sumsq` then inverse-scale locals; LayerNorm uses ordered sum,
  mean, variance-sum, and inverse-scale locals; Softmax uses max and
  denominator locals. Each preserves the existing `Compute.pixel` ordering,
  synthetic optional operands, affine coordinate projection, all-`-inf`
  behavior, and final result conversion.
- `test/native/region_compute_test.ml` records two independent keys at extent
  three: RMSNorm performs 6 reduction iterations, LayerNorm 12, and Softmax
  12, with 6 emitter evaluations each.
- `make benchmark.region_compute` alternates Pixel and Region processes for
  extents 8, 32, and 64, with 20 samples after 3 warmups. On Linux
  7.0.0-28-generic/aarch64 with OCaml 4.14.3, extent 64 median times were
  RMSNorm 4.637 ms/0.209 ms, LayerNorm 299.533 ms/0.316 ms, and Softmax
  152.119 ms/0.179 ms (Pixel/Region). Region reductions were respectively
  256, 512, and 512, exactly linear in the four rows and extent.
  **These figures hold only at four region keys.** The driver fixes `W = 4`, so
  the sweep varies `K` alone and cannot show the `R^2 * K` traversal recorded in
  the reopening above. They are not evidence of an asymptotic improvement.
- Phase 1 Softmax evaluates `exp(x[k] - m)` twice per output — once in the
  denominator local, once in the emitter — so it performs `2K` transcendentals
  per row where a shaped exponential cache would perform `K`. Reduction, load,
  and key counters are all optimal and none of them shows this. Recorded here so
  the Phase 4 shaped-local evaluation has the baseline.

## Validation record

| Date | Gate | Command | Result | Notes |
|---|---|---|---|---|
| 2026-08-31 | 0-6 | `make precommit` | pass | Native, Native4D, and Model Explorer inline suites passed. |
| 2026-08-31 | 2-6 | `make benchmark.region_compute` | pass | Alternating Pixel/Region timing, allocation, and counters at extents 8/32/64. |
| 2026-08-31 | 2-6 | `make js.runtest` | pass | js_of_ocaml and Melange outputs matched Native. |
| 2026-08-31 | 0-6 | `make precommit`, `make js.runtest`, `make benchmark.region_compute` | pass | Revalidated after operation-owned Region constructor refactor (`e605361`). |

## Post-closeout ownership migration

This completed log records the optional-regionizer stepping stone. The adopted
follow-up makes Region authoritative for intrinsically shared operations,
routes Native and Native4D Direct and Symbolic through the same program, and
adds operation-level whole-domain traversal proofs. That separate task is
specified by
[`region-operation-computation-implementation-plan.md`](region-operation-computation-implementation-plan.md)
and tracked by
[`region-operation-computation-implementation-todo.md`](region-operation-computation-implementation-todo.md).
