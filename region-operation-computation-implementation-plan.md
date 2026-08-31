# Operation-authored Region computation implementation plan

## Status and boundary

The Region Foundation and the optional scalar-regionizer slice are complete.
The current implementation proves that RMSNorm, LayerNorm, and plain Softmax
can construct valid non-degenerate programs, execute with shared scalar locals,
remain bitwise identical to the former Pixel path, and reduce dynamic reduction
work from quadratic to linear per slice.

This plan closes the remaining ownership and execution gap described in
[`region-compute-follow-up.md`](region-compute-follow-up.md):

- operations author computation at their natural Pixel or Region granularity;
- RMSNorm, LayerNorm, and Softmax use Region as their authoritative form;
- Native and Native4D Direct and Symbolic consume the same Region programs;
- fresh scalar projection remains available but is never the production scan;
- operation-level traces prove complete, disjoint output ownership.

The completed optional-regionizer implementation is a migration input, not work
to repeat. Its historical plan and evidence remain in
[`region-compute-implementation-plan.md`](region-compute-implementation-plan.md)
and
[`region-compute-implementation-todo.md`](region-compute-implementation-todo.md).

## Completion contract

This migration is complete only when all of the following are true:

1. Every operation has one authoritative authored computation form. Existing
   Pixel-authored operations retain `Compute(S).pixel`; Native RMSNorm,
   LayerNorm, and Softmax author `Region_program.t` directly.
2. Native Direct materializes those Region-authored operations by key, evaluates
   locals once per key, and emits every owned output once.
3. Native Symbolic carries the same Region program rather than first building an
   independently handwritten Pixel expression and optionally replacing it.
4. `Ops4.Rms_norm`, `Ops4.Layer_norm`, and `Ops4.Softmax4` expose Native4D
   computation entries which perform checked Axis4 mapping and delegate to the
   same numeric Region definitions.
5. Native4D Direct and Symbolic preserve Region sharing and never route these
   operations through `Eval_op4.Make(S).pixel` for production execution.
6. A bounded scalar projection can evaluate any supported Region program at one
   output coordinate by reconstructing its key and locals afresh. It agrees
   bitwise with Region materialization in strict mode.
7. Deterministic operation-level traces print each program, enumerate every
   canonical key and owned output, show ordered local/emitter expressions, and
   independently prove complete coverage with zero duplicates.
8. The permanent handwritten Pixel implementations of RMSNorm, LayerNorm, and
   Softmax are removed. Any temporary migration adapter derives Pixel from the
   Region program.
9. Pixel-authored operations retain the established Direct and Kernel hot paths,
   output/load counts, and no-regression benchmark contract.
10. Native, Native4D, Model Explorer, transformation verification, Native4D
    cross-dialect verification, and JavaScript suites pass.

## Non-negotiable design rules

- Do not add Region traversal or stores to `Semantics.SEMANTICS`.
- Do not add a semantic `Pixel | Region` sum to `Kernel.Value`; normalize both
  authored forms to one `Region_program.t`.
- Do not copy Region arithmetic into `Regionizer`, `Eval_op`, `Eval_op4`,
  `Eval_direct`, or `Eval_direct4`.
- Do not keep a second handwritten Pixel algorithm as a fallback for a
  Region-authored operation.
- Do not materialize a Region by invoking scalar projection once per output.
- Do not search synthetic optional operands by floating-point value. Resolve
  weight, bias, and other optional roles explicitly at the typed operation
  boundary.
- Validate programs under the actual requested limits. Do not construct under
  `Kernel.Limits.default` and later admit under unrelated limits.
- Keep emitter purity, ordered reductions, working-float locals, and the single
  final `Kernel.Result_conversion` boundary.
- Keep Native4D's T/D axes unit and semantically unavailable. Common Vec6
  representation does not legalize an Axis4-invalid operation.
- Do not pull in Loop IR, tiling, `Block`, shaped locals, multi-output Region
  emission, graph fusion, relaxed rounding, GroupNorm regionization, safe
  softmax, or SDPA.

## Gate 0 — contract census and migration baselines

### Work

1. Census every current consumer of:

   - `Norm.RmsNorm.Compute`, `Norm.LayerNorm.Compute`, and
     `Reduce.Softmax.Compute`;
   - the corresponding operation `Region` modules;
   - `Regionizer`, `Region_context`, and
     `Kernel_adapt.Region_admission`;
   - `Stage_program.Stage.body`;
   - `Eval_direct`, `Eval_symbolic`, `Eval_op4`, `Eval_direct4`, and
     `Eval_symbolic4`;
   - Native4D cross-dialect symbolic verification and grounding.
2. Record which consumers need efficient whole-region materialization, which
   need only scalar projection, and which require structural Region analysis.
3. Pin the present Region program printers and operation counters for RMSNorm,
   LayerNorm, and Softmax.
4. Record Native and Native4D Direct baselines showing that the current paths
   still execute the repeated Pixel reductions.
5. Add red expectations for the final computation form at the Native and
   Native4D symbolic boundaries.

### Acceptance

- The census names every migration site and classifies it by required access.
- Baseline counters distinguish production Region execution from fresh scalar
  projection.
- No representation change begins before the Stage/Kernel/verifier impact is
  recorded.

## Gate 1 — scalar projection and whole-domain trace contract

### Work

1. Give scalar projection one named, bounded API. It takes a validated
   `Region_program.t`, output shape, environment, output coordinate, and result
   conversion. It computes the key, evaluates locals afresh in order, evaluates
   one emitter, and applies conversion exactly once.
2. Ensure `Region_execution.value_at`, `Region_eval.value_at`, and
   `Region_program.specialize_pixel` have distinct documented roles:
   concrete scalar projection, reference evaluation, and symbolic expansion.
3. Add a deterministic traversal trace representation containing:
   - the stable `Region_program.pp` form;
   - canonical keys;
   - ordered local ids and open expressions per key;
   - outputs owned by each key;
   - the open emitter expression;
   - optional evaluated local/emitter values behind test-only hooks;
   - total cells, key count, visited cells, duplicate cells, and missing cells.
4. Compute the trace's coverage summary independently from tensor stores. Use a
   bounded coordinate visit table and require exactly one visit per output.
5. For every `(key, output)` entry, assert that `key_of_output output = key`.
6. Avoid expanding locals or unrolling reductions in the trace.

### Tests

- Synthetic singleton, whole-C, whole-W/C, and non-adjacent whole-axis
  partitions over shapes with several independent keys.
- Empty/invalid domain boundaries already rejected by shape and dimension
  types; extent-one axes remain visible in traces.
- Materialization equals fresh projection at every output, bitwise, including
  f32 conversion, signed zero, infinities, and NaNs.
- A mutation fixture in the test trace deliberately duplicates or omits an
  output and proves the independent summary detects it.

### Acceptance

The Region language has a tested scalar observation law and an inspectable
whole-domain ownership proof before any operation loses its old Pixel oracle.

## Gate 2 — operation-authored computation boundary

### Work

1. Introduce the operation-facing Region computation API used by RMSNorm,
   LayerNorm, and Softmax. Its typed input includes:

   - actual limits;
   - output ordinal and shape;
   - validated operation parameters;
   - role-resolved operand signatures/sources.
2. Move generic coordinate, partition, reduction-builder, and program-checking
   helpers behind a narrow construction module. Keep normalization-only affine
   projection helpers with normalization computation rather than growing a
   generic context junk drawer.
3. Make optional defaults explicit role-resolved sources. A missing RMSNorm
   weight is an identity-weight source; LayerNorm weight and bias are distinct
   identity-scale and zero-offset sources.
4. Define the graph boundary's private authored-form dispatch. Pixel results are
   normalized with `Region_program.pixel`; Region results pass unchanged.
5. Keep the exhaustive Native and Native4D closed-operation dispatchers, but
   restrict them to parameter conversion, operand resolution, and calls into
   operation-owned computation.
6. Validate Region-authored programs under the caller's limits and classify
   invalid construction as an implementation/limit error, not an optimization
   rejection.

### Tests

- Each operation builder accepts every currently supported parameter/operand
  form and rejects output ordinal, shape, axis, and limit errors precisely.
- Two optional operands with the same constant value or different shapes cannot
  be confused by lookup.
- Tight and expanded non-default limits are honored consistently at construction
  and Kernel creation.
- The private authored-form normalization preserves Pixel expression identity
  for Pixel-authored operations.

### Acceptance

All arithmetic and Region structure live in operation computation modules.
Dispatch and context modules own no operation formula.

## Gate 3 — Native Region authority and Direct execution

### Work

1. Promote the existing RMSNorm, LayerNorm, and Softmax Region builders to their
   operation modules' authoritative computation entries.
2. Route `Eval_direct.eval_node` for those operations through program
   construction, `Region_execution.lower`, and whole-region materialization
   once per logical output.
3. Bind real and synthetic operand sources without copying tensors beyond the
   existing optional-operand contract.
4. Apply output format/result conversion at the same logical store boundary as
   current Direct and Kernel execution.
5. Retain temporary old Pixel implementations only as test oracles until the
   Symbolic and Native4D gates close. Mark them unreachable from production
   dispatch.
6. Add operation traces for:

   - RMSNorm over every single axis and representative multi-axis/non-adjacent
     dimension lists;
   - LayerNorm with the same partition matrix and all affine combinations;
   - Softmax over every Vec6 axis;
   - multiple keys, extent-one axes, and equal-numel/different-partition cases.

### Tests and measurements

- Native Direct, Region reference, lowered Region, scalar projection, and
  external ATen/reference fixtures agree bitwise where the existing strict
  contract claims `Identical`.
- Direct counters show one local sequence per key and linear reduction work.
- Trace summaries show every output exactly once for every configuration.
- Existing graph hooks and intermediate-edge visibility remain unchanged.

### Acceptance

Native Direct obtains the Region asymptotic improvement and no production call
to the old Pixel computation remains for the three operations.

## Gate 4 — Symbolic and Stage-program Region carriage

### Work

1. Make the symbolic stage boundary capable of carrying
   `Region_program.t` directly. Prefer migrating
   `Stage_program.Stage.body : Expr.Value.t` to a computation field over adding
   a parallel candidate map that recreates dual authority.
2. Mechanically embed every Pixel symbolic result with
   `Region_program.pixel`, retaining the original expression object.
3. Have Native `Eval_symbolic` insert the authoritative operation-owned Region
   programs directly for RMSNorm, LayerNorm, and Softmax.
4. Update Stage-program validation, printing, grounding, source analysis,
   Kernel adaptation, required-output analysis, and tests to use structural
   Region queries.
5. Update transformation grounding and map verification:

   - Pixel stages retain their existing expression path;
   - Region stages use bounded scalar projection where a per-coordinate proof is
     required;
   - structural Region comparison and whole-domain trace evidence remain
     available without expansion.
6. Remove the operation candidate map and reconstruction admission from the
   primary symbolic-to-Kernel route. Preserve `reconstructs` as a general
   transformation helper.
7. Preserve the Pixel Stage/Kernel hot path: one classification per logical
   value and no Region machinery inside the per-output callback.

### Tests

- Existing Stage-program malformed-input and limit tests are ported to Region
  structural analysis.
- Pixel stage printing and grounding remain stable.
- Region stages print locals and emitter, ground through materialization, and
  agree with fresh projection.
- Kernel adaptation passes the exact Region object through rather than
  reconstructing or re-admitting it against Pixel.
- Native transformation verification either supports Region stages explicitly
  or returns a typed unsupported result; it never silently inspects only the
  emitter and misses local dependencies.

### Acceptance

Native Symbolic and Kernel consume the same authoritative Region object, and
the old optional-regionizer candidate path is not involved.

## Gate 5 — Native4D Region authority and execution

### Work

1. Add computation entries to `Ops4.Rms_norm`, `Ops4.Layer_norm`, and
   `Ops4.Softmax4`, or to operation-owned companion modules with equally clear
   ownership.
2. Use one checked parameter adapter per operation for both shape and
   computation. Move or expose the present `Graph_shape4` adapters as needed so
   shape and execution cannot disagree.
3. Map Axis4 axes to the common Vec6 representation, require T/D to remain unit
   and semantically unavailable, and make their singleton partition modes
   visible in traces.
4. Resolve Native4D optional operands by role and delegate to the same Native
   numeric Region builders.
5. Route `Eval_direct4` through whole-region materialization for these
   operations.
6. Route `Eval_symbolic4` through the same Region-capable Stage-program boundary.
7. Remove the three production arms from `Eval_op4.Make(S).pixel`; retain a
   derived scalar adapter only where a legacy per-coordinate interface still
   requires it.
8. Extend cross-dialect verification to compare Native and Native4D Region
   computations after mapping, while retaining the dialect's `Identical`
   claims.

### Tests and measurements

- RMSNorm and LayerNorm cover single/multi-axis Axis4 lists, optional affine
  operands, multiple keys, and extent-one axes.
- Softmax covers N/H/W/C.
- Native4D traversal traces show all outputs once and T/D singleton for every
  key.
- Native and Native4D Region programs have matching mapped partitions, local
  order, emitter structure, scalar projections, and concrete outputs.
- Native4D Direct counters show linear shared work rather than repeated Pixel
  reductions.
- Axis4-invalid Native source operations retain their existing typed domain
  rejection and cannot reach Region construction.

### Acceptance

Native4D remains a first-class dialect while sharing one numeric Region
definition and the same asymptotic execution behavior as Native.

## Gate 6 — remove dual authority and obsolete admission

### Work

1. Remove the handwritten `Compute(S).pixel` implementations for Native
   RMSNorm, LayerNorm, and Softmax.
2. Replace remaining legitimate scalar consumers with bounded projection from
   the authoritative Region program.
3. Remove or narrow:

   - `Regionizer` operation selection;
   - `Region_context.pixel`;
   - value-based synthetic lookup;
   - primary `Kernel_adapt.Region_admission` reconstruction/fallback;
   - candidate maps in symbolic results.
4. Keep generic Pixel-to-Region transformation admission separate if a real
   consumer exists; otherwise retain only the reusable validation and
   `reconstructs` primitives.
5. Audit comments, module names, public interfaces, Model Explorer labels, and
   errors so none describe Region-authored computation as an optional
   optimization.

### Tests

- Compile-time census shows no production reference to the removed Pixel
  computations or optional candidate route.
- A test-only scalar projection still reproduces the former per-output values.
- An invalid authoritative program produces a typed error and never selects a
  second handwritten algorithm.
- Pixel-authored operation tests remain unchanged except for mechanical
  Stage-program field migration.

### Acceptance

There is one permanent semantic definition for each of the three
Region-authored operations.

## Gate 7 — closeout and production admission

### Work

1. Run full Native, Native4D, Model Explorer, transformation, cross-dialect, and
   JavaScript verification.
2. Run operation-level traversal traces for the complete required matrix and
   retain concise golden output with independent coverage summaries.
3. Benchmark Native and Native4D Direct plus Kernel Region execution at several
   reduction extents. Record keys, locals, emitters, loads, reductions,
   allocation, and elapsed time.
4. Re-run Pixel no-regression benchmarks for representative pointwise,
   convolution, and indexing operations.
5. Audit production call chains to prove:

   - Pixel-authored operations take their specialized Pixel path;
   - Region-authored Direct and Symbolic paths materialize by key;
   - scalar projection occurs only at named compatibility/test boundaries.
6. Update the main Region and Native4D design records with final module names,
   APIs, measurements, validation commands, and any explicitly deferred work.

### Acceptance

- All completion-contract items are demonstrated by tests, traces, counters,
  or documented audit evidence.
- Native and Native4D receive the same operation-form policy.
- Whole-domain Region coverage and single ownership are inspectable for
  non-trivial real operations, not inferred only from a synthetic partition
  test.
- The current optional-regionizer architecture has no remaining production
  responsibility for Region-authored operations.

## Expected file-level change map

The exact split may change after Gate 0, but the expected ownership is:

- Native operation computation:
  `lib/native/ops/norm_rms.ml`, `lib/native/ops/norm_layer.ml`,
  `lib/native/ops/reduce.ml`.
- Region construction/projection/tracing:
  `lib/native/region_program*`, `region_partition*`, `region_eval*`,
  `region_execution*`, and a narrow construction/trace module if needed.
- Native dispatch and stage carriage:
  `lib/native/eval_op.ml`, `eval_direct*`, `eval_symbolic*`,
  `stage_program*`, `kernel_adapt*`, `regionizer*`, `region_context*`.
- Verification and structural consumers:
  `lib/native/transform/ground_eval*`, `map_verify*`, Model Explorer, Kernel
  analysis/fusion consumers.
- Native4D operation and dispatch:
  `lib/native4d/ops4.ml`, `graph_shape4.ml`, `eval_op4.ml`,
  `eval_direct4.ml`, `eval_symbolic4.ml`.
- Tests:
  `test/native/region_program_test.ml`,
  `test/native/region_compute_test.ml`, operation/graph/Stage/Kernel tests,
  and corresponding `test/native4d` compute, verify, domain, and trace coverage.
- Benchmarks:
  extend `bin/region_compute_bench.ml` or add a dialect-comparable driver
  without weakening the existing Pixel benchmark.

## Deferred follow-ons

This plan deliberately leaves the following for later:

1. general automatic Pixel-to-Region discovery;
2. Loop IR and schedule search;
3. physical tiling and footprint analysis;
4. shaped locals and local placement;
5. multi-output Region programs;
6. GroupNorm and batch-normalization Region forms;
7. safe/log softmax and efficient SDPA;
8. relaxed numerical policies, tree reductions, and parallel reduction splits;
9. multi-node Region fusion.
