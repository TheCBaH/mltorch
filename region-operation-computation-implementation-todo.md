# Operation-authored Region computation implementation tracker

This is the live execution record for
[`region-operation-computation-implementation-plan.md`](region-operation-computation-implementation-plan.md).
It starts after the completed Foundation and optional scalar-regionizer slices.

## Scope lock

- Included: heterogeneous operation-authored Pixel/Region forms; authoritative
  Region computation for RMSNorm, LayerNorm, and plain Softmax; Native and
  Native4D Direct/Symbolic routing; bounded scalar projection; whole-domain
  coverage/disjointness traces; removal of permanent dual Pixel authority.
- Excluded: Loop IR, physical tiling, `Block`, shaped locals, multi-output
  Region emission, graph fusion, GroupNorm, batch-normalization Region forms,
  safe/log softmax, SDPA, relaxed rounding, and parallel/tree reductions.
- Kernel representation rule: every logical value continues to store one
  `Region_program.t`; authored form does not add a Kernel sum type.
- Failure rule: a malformed authoritative Region program is a typed
  implementation/limit failure, not a request to run a second handwritten
  Pixel algorithm. There is consequently no `try_` anything on this path.
- Cost rule: owned outputs are enumerated from the key and the `Whole`-axis
  extents. Materialization costs one pass over the output domain regardless of
  how that domain is partitioned.
- Compatibility rule: fresh scalar projection is permitted at named scalar
  boundaries but never as production tensor materialization.

## Progress

| Gate | Scope | Status | Commit |
|---|---|---|---|
| Foundation prerequisite | Region language and Pixel fast path | complete | `2a81e7b`; closeout `787f278` / `c50be5b` |
| Scalar-Region prerequisite | Optional regionizer and dedicated executor | complete for correctness; cost contract unmet | `c8fe41f`; ownership refactor `e605361` |
| P | Owned-output enumeration and region-count benchmark | complete | uncommitted |
| 0 | Contract census and migration baselines | complete | documentation-only |
| 1 | Scalar projection and whole-domain trace contract | complete | uncommitted |
| 2 | Operation-authored computation boundary | in progress | — |
| 3 | Native Region authority and Direct execution | in progress | — |
| 4 | Symbolic and Stage-program Region carriage | in progress | — |
| 5 | Native4D Region authority and execution | complete | uncommitted |
| 6 | Remove dual authority and obsolete admission | pending | — |
| 7 | Closeout and production admission | pending | — |

## Gate checklist

### Gate P

- [x] Enumerate a key's owned outputs from the key and the `Whole`-axis extents
      instead of filtering a full-domain scan per key.
- [x] Drop `fold_outputs`'s `Err.t` result if the rewrite makes
      `` `Region_key_out_of_bounds `` unreachable.
- [x] Add a region-count axis to `bin/region_compute_bench.ml`, which currently
      fixes `W = 4`.
- [x] Measure a sweep of `R` at fixed `K`, including an `R > K` shape.
- [x] Confirm unchanged coordinates, order, values, and conversions against the
      existing Foundation partition-enumeration tests.

### Gate 0

- [x] Census all Native and Native4D computation call sites.
- [x] Classify Stage-program, Kernel, grounding, verifier, explorer, and fusion
      consumers by Pixel-only, scalar-projection, or structural-Region need.
- [x] Record current Native and Native4D Direct repeated-reduction baselines.
- [x] Pin current Region program printers, counters, and symbolic form.
- [x] Add red expectations for authoritative Region carriage.

#### Gate 0 census (2026-08-31)

| Consumer | Current form | Required migration form |
|---|---|---|
| Native Direct | `Eval_direct` calls `Eval_op.Make(Direct).pixel`, then `Schedule.evaluate` for RMSNorm, LayerNorm, and Softmax | Region materialization once per key |
| Native Symbolic | `Eval_symbolic.build` constructs every `Stage.body` as Pixel, then `run_regionized` stores optional `Regionizer` candidates | Stage carries Region directly; Pixel body is mechanical only for Pixel-authored operations |
| Kernel adaptation | `Kernel_adapt.Region_admission` accepts/rejects candidate Region programs against a Pixel reconstruction | Structural Region input; no candidate/reconstruction admission |
| Kernel execution | `Kernel.Value.computation` is already `Region_program.t`; `Kernel_eval` lowers it through `Region_execution` | Whole-region execution (already structurally capable) |
| Stage grounding / transform verification | `Stage_program.Stage.body`, `Stage_program.ground`, `Ground_eval`, and source analysis read `Expr.Value.t` | Structural Region stage support or an explicit verifier rejection |
| Explorer / fusion | `Me_detail`, `Me_kernel`, `Fusion_plan`, and `Kernel_elab` inspect `Kernel.Value.computation` | Structural Region (Kernel-side support exists; Stage rendering does not) |
| Native4D Direct / Symbolic | `Eval_op4.Make(S).pixel` maps parameters then delegates to Native Pixel `Compute(S)`; both drivers use that result | Peer computation dispatcher producing the same Native Region program |

The production Direct baselines are therefore the Pixel implementations in
`Norm.RmsNorm.Compute`, `Norm.LayerNorm.Compute`, and
`Reduce.Softmax.Compute`: each evaluates its reductions per output.  The
Region-kernel counters pinned in `test/native/region_compute_test.ml` show the
target sharing for the shape `W=2,C=3`: RMS `keys=2 locals=4 emitters=6
loads=24 reductions=6`; LayerNorm `2,8,6,36,12`; Softmax `2,4,6,18,12`.
The printers are pinned in `region_program_test.ml`; current symbolic stages
are Pixel `Expr.Value.t` bodies, and the final-form expectation is that the
three operation stages instead carry a non-Pixel `Region_program.t` directly.

### Gate 1

- [x] Define one named bounded scalar-projection API.
- [x] Document concrete projection, reference `value_at`, and symbolic
      specialization as distinct operations.
- [x] Add deterministic program/key/local/output/emitter trace representation.
- [x] Compute output visit counts independently from tensor stores.
- [x] Assert complete coverage, zero duplicates, and ownership agreement.
- [x] Test materialization versus fresh projection bitwise at every output.
- [x] Add mutation fixtures proving missing/duplicate visits are detected.

### Gate 2

- [x] Define the operation-facing Region computation API with actual limits.
- [x] Replace each operation's `Region.try_regionize` with a
      `Computation.program` entry returning a typed error, not a rejection.
- [x] Delete `Unsupported_operation`; move `Missing_operand`, `Output_ordinal`,
      and `Output_shape` to the dispatcher boundary; make `Invalid_partition`
      and `Invalid_program` limit/implementation errors.
- [x] Pass output ordinal/shape and role-resolved operands explicitly.
- [x] Separate generic Region construction from normalization-only helpers.
- [x] Eliminate floating-value lookup for synthetic optional operands.
- [x] Route the normalization divisor through the bounded shared helper rather
      than re-folding an unbounded extent product in `Region_context.count`.
- [x] Normalize Pixel and Region authored forms at the graph boundary.
- [ ] Keep dispatchers arithmetic-free and exhaustive per dialect.
- [x] Test tight and expanded non-default limits, together with output ordinal,
      output shape, and missing-operand construction failures.

### Gate 3

- [x] Make Native RMSNorm's Region program authoritative.
- [x] Make Native LayerNorm's Region program authoritative.
- [x] Make Native Softmax's Region program authoritative.
- [x] Route Native Direct through once-per-key Region materialization.
- [ ] Keep old Pixel implementations test-only during migration.
- [x] Add the complete Native operation trace matrix.
- [x] Prove Native Direct linear work with counters and benchmarks.

### Gate 4

- [x] Carry `Region_program.t` directly at the symbolic Stage boundary.
- [ ] Embed Pixel symbolic bodies mechanically and preserve object identity.
- [x] Insert authoritative Region programs directly from Native Symbolic.
- [x] Migrate Stage validation, printing, grounding, and source analysis.
- [x] Migrate Kernel adaptation without candidate/reconstruction admission.
- [x] Support Region stages in transformation verification.
- [ ] Preserve Pixel Stage/Kernel hot-path behavior.

### Gate 5

- [x] Expose computation entries for `Ops4.Rms_norm`.
- [x] Expose computation entries for `Ops4.Layer_norm`.
- [x] Expose computation entries for `Ops4.Softmax4`.
- [x] Share one checked parameter adapter between shape and computation.
- [x] Route Native4D Direct through once-per-key Region materialization.
- [x] Route Native4D Symbolic through Region-capable Stage programs.
- [x] Remove the three Region-authored production arms from
      `Eval_op4.Make(S).pixel`.
- [x] Trace N/H/W/C cases with T/D visibly singleton.
- [x] Prove Native/Native4D mapped program and result agreement.
- [x] Preserve every existing Axis4/domain rejection.

### Gate 6

- [x] Run the gated `.pt2` model-level differential BEFORE any removal.
- [ ] Demote handwritten Pixel RMSNorm computation to test-only, keeping
      `reconstructs` asserting against it.
- [ ] Demote handwritten Pixel LayerNorm computation on the same terms.
- [ ] Demote handwritten Pixel Softmax computation on the same terms.
- [ ] If deleting outright instead, record the decision and that the arithmetic
      then has one in-tree source.
- [ ] Replace legitimate scalar consumers with bounded derived projection.
- [x] Remove `Regionizer` entirely, along with `Regionizer.candidate`, the
      `regionizers` flag on `Eval_symbolic.build`, and the `regionized` result.
- [ ] Remove `Region_context.pixel`, `reject`, and synthetic-value lookup;
      retain the construction helpers under the Gate 2 module.
- [x] Remove candidate maps and primary reconstruction fallback.
- [ ] Retain general `reconstructs` support for real transformations.
- [ ] Audit public names, errors, comments, and explorer output for obsolete
      optional-optimization language.
- [ ] Prove invalid authoritative Region construction cannot select another
      algorithm.

### Gate 7

- [ ] Run full Native and Native4D tests.
- [ ] Run Model Explorer and transformation verification.
- [ ] Run Native4D cross-dialect verification.
- [ ] Run JavaScript verification.
- [ ] Run and record every operation-level coverage trace.
- [ ] Benchmark Native and Native4D Direct plus Kernel Region execution across
      both extent `K` and region count `R`, including an `R > K` shape.
- [ ] State, beside each recorded counter set, which costs it does not observe.
- [ ] Re-run Pixel no-regression benchmarks.
- [ ] Audit production call chains for Pixel, Region, and projection routing.
- [ ] Update final APIs, evidence, measurements, and deferrals in design docs.
- [ ] Promote a consolidated Region design into `.ai/`; there is currently no
      tracked `region_*` design record for landed, tracked code.
- [ ] Repoint `.ai/native4d_design.md` off `../_ai_/region-compute-follow-up.md`.
- [ ] Settle the `Region` name collision with `transform/region.ml`.

## Decisions and current evidence

- The authored computation form is operation-specific, not dialect-specific.
  Pixel remains natural for many operations; RMSNorm, LayerNorm, and Softmax
  are Region-authored.
- Kernel remains uniform because `Region_program.pixel` embeds Pixel
  mechanically.
- Scalar projection is a required semantic observation law, even though
  repeated projection is intentionally inefficient.
- Whole-domain coverage, disjointness, and ownership require explicit
  operation-level trace evidence. A successful dense tensor comparison alone
  cannot detect duplicate stores paired with an unvisited initialized cell.
- Native4D is a peer operation owner. It owns Axis4 legality and mapping while
  delegating numeric Pixel or Region computation to the shared Native operation
  definition.
- Native `Eval_direct` resolves the three Region-authored operations through
  `Regionizer.program` with its actual `?limits`, then materializes each
  lowered Region program once per key. Other operations retain the Pixel path.
- Native `Eval_symbolic.run_regionized` carries successful authored Region
  programs on their Stages and Kernel adaptation passes that exact object
  through directly. The candidate map remains only for the typed-error
  compatibility path while legacy Stage consumers still read the mechanical
  Pixel body.
- Current Native4D `Eval_op4.Make(S).pixel` delegates all operations to Native
  Pixel computation; `Eval_direct4` and `Eval_symbolic4` therefore repeat
  normalization/Softmax reductions per output.
- Current operation Region constructors independently rebuild the formulas;
  `Region_context.pixel` is retained but not used to extract locals or emitter.
- Current operation builders use `Kernel.Limits.default` before later admission
  under caller limits, so construction and admission limits can differ.
- Current Foundation enumeration tests cover a synthetic whole-C partition,
  but do not print whole-domain traces for non-trivial real operation
  configurations or Native4D mappings.

### Added by design review (2026-08-31)

- `Region_partition.fold_outputs` scans the whole output shape once per key and
  filters by membership, so `Region_execution.materialize` costs `R^2 * K`
  against the Pixel path's `R * K^2`. The Region executor is faster only while
  `R < K`, and normalization over `C` runs at `R = N*T > K`. Gate P.
  `Region_eval.materialize` — the oracle it replaced — is asymptotically
  correct, using one domain pass with a key-indexed cache.
- No committed measurement can see that: `bin/region_compute_bench.ml` fixes
  `W = 4`, the counter fixtures use two keys, and no counter observes a
  membership test. Counters are evidence only for what they count.
- `try_regionize` does not survive the migration. Its name and its
  `(_, reject) result` type encode the optional-optimization model being
  retired; the per-operation bodies become `Computation.program` entries with
  typed errors, and `Regionizer` disappears along with the candidate map.
- `Region_context.count` folds an unbounded product of extents where the Pixel
  path routes the same product through the precondition-documented
  `Norm_shared.normalized_count_unchecked`. The 32-bit JavaScript backends make
  this the repository's most-repeated defect shape, and `reconstructs` is
  presently the only thing comparing the two products — which Gate 6 retires.
- The Phase 1 local tables in the design doc listed locals the builders fold at
  construction (RMSNorm 3 vs 2 declared, LayerNorm 5 vs 4). Corrected; the
  counter goldens `locals=4` and `locals=8` over two keys are the authority.
- `Region_program.pixel` is a public constructor carrying no singleton witness;
  classification is recomputed by `pixel_expression`. The related open hole is
  `with_output`, which replaces an admitted program's emitter without
  revalidating, and is how `Kernel_eval.converted` applies result conversion.
- The projection law is a consequence of `Expr` purity plus
  `Non_invariant_local`, not an independent obligation. Gate 1's test samples
  it.
- Native4D's T/D-unit rule is enforced at the parameter adapter only. A mapped
  partition reading `t=singleton d=singleton` is indistinguishable from a Native
  program with unit T/D, so Gate 5's traces display the invariant rather than
  establish it.
- Phase 1 Softmax performs `2K` `exp` evaluations per row where a shaped cache
  would perform `K`. Correct for Phase 1, invisible to every current counter,
  and the baseline for the Phase 4 shaped-local decision.
- `Region` already names a claimed graph-node set in
  `lib/native/transform/region.ml`, with the derived
  inputs/outputs/interior/convex boundary the deferred whole-graph work needs.
  Reuse it there, and settle the collision before that task rather than during
  it.
- These nine documents are not the tracked design record. `.ai/` holds no
  `region_*` doc despite the Foundation and scalar-Region slices being landed,
  tracked code, and `.ai/native4d_design.md` links into this untracked working
  set. Gate 7.

## Validation record

| Date | Gate | Command | Result | Notes |
|---|---|---|---|---|
| 2026-08-31 | P | `opam exec -- dune runtest test/native` | pass | Added a non-adjacent Whole T/W/C partition regression that compares direct owned-output enumeration with the prior filtered full-domain order. |
| 2026-08-31 | P | `opam exec -- dune exec bin/region_compute_bench.exe` | pass | Sweeps `R=1,4,16,64` at `K=32`, including `R > K`, plus `K=8,64` at `R=4`. Region timings scale with the output domain in this run. `keys`, `locals`, `emitters`, `loads`, and `reductions` do not observe membership tests; wall time is the evidence for traversal cost. |
| 2026-08-31 | 0 | source census (`rg` across `lib`, `test`, and `bin`) | complete | Recorded Native/Native4D dispatch, Stage/Kernel adaptation, grounding, verifier, explorer, and fusion migration seams above. |
| 2026-08-31 | 1 | `opam exec -- dune runtest test/native` | pass | `Region_trace` records canonical program/key/local/emitter/output ownership, uses an independent bounded visit table, and rejects incomplete or duplicate coverage. The regression projects every output of a Region materialization, bitwise including NaN, infinities, and signed zero; its deliberately mutated trace reports one duplicate and one missing visit. |
| 2026-08-31 | 2 (partial) | `make precommit` | pass | Region builders now receive resolved role operands and caller limits; the symbolic migration derives its temporary Pixel body from the authoritative Region program, without a mutable synthetic-default hand-off or value-based lookup. The remaining Gate 2 authored-form carriage is completed with Gate 4. |
| 2026-08-31 | 3 (partial) | `make precommit` | pass | Native Direct constructs RMSNorm, LayerNorm, and Softmax with its actual limits and materializes each Region once per key. The Direct counter fixture records `keys/locals/emitters/loads/reductions` for all three operations, and a tight Direct limit returns the typed construction error; broad trace and benchmark work remains. |
| 2026-08-31 | 4 (partial) | `make precommit` | pass | A Native symbolic Stage now holds the authoritative `Region_program.t`; its legacy Pixel body is derived mechanically, Stage grounding can materialize the structural program, and Kernel adaptation passes the same object through directly. Candidate admission remains only for the compatibility error path. |
| 2026-09-01 | 4 (partial) | `make precommit` | pass | `Stage.computation` embeds legacy Pixel bodies mechanically and returns an authored Region object unchanged. Stage grounding, Kernel validation, and source analysis now consume that uniform structural form; transform verification remains explicitly unmigrated. |
| 2026-09-01 | 4 (partial) | `make precommit` | pass | Transform grounding specializes an authored Stage from its structural Region program. Its regression corrupts the legacy Pixel body and still observes the Region's input dependency, proving verification does not read that compatibility body. |
| 2026-09-01 | 3 (partial) | `make precommit` | pass | The Native trace matrix independently proves complete, duplicate-free ownership for RMSNorm over every axis plus non-adjacent multi-axis partitions, LayerNorm over that matrix and all affine combinations, and Softmax over every Vec6 axis. It includes multiple keys and extent-one T/W axes at the same output numel. |
| 2026-09-01 | 5 (partial) | `make precommit` | pass | Native4D's `Regionizer4` maps the three authored operations through their existing `Graph_shape4` parameter adapters and delegates construction to Native. Direct materializes the resulting Region once per key; its three counter goldens match Native. Symbolic carriage, mapped traces, and cross-dialect evidence remain open. |
| 2026-09-01 | 5 (partial) | `make precommit` | pass | `Eval_symbolic4.run_regionized` now carries successful Native4D-authored Region programs on Stages, and Kernel adaptation receives each exact object. The default Pixel-symbolic entry point is retained for compatibility while the temporary `Eval_op4.pixel` arms remain. |
| 2026-09-01 | 5 (partial) | `make precommit` | pass | Native4D symbolic traces independently verify output ownership and print `td_singleton=true` for every key and output. They cover RMSNorm/LayerNorm multi-axis Regions and Softmax4 over N/H/W/C. Full affine and Native-mapping matrices remain open. |
| 2026-09-01 | 5 (partial) | `make precommit` | pass | The Axis4 norm matrix covers six RMSNorm single/multi-axis partitions and 24 LayerNorm partition/affine combinations, including multiple keys and extent-one W. Every independently collected trace has complete ownership and singleton T/D. Native/Native4D structural mapping evidence remains open. |
| 2026-09-01 | 5 | `make precommit` | pass | `Eval_op4.pixel` now rejects the three Region-authored operations, while the default Native4D symbolic boundary carries their Region programs. Native-to-Native4D lowering preserves Region structure, and matching Direct evaluations cover RMSNorm/C, LayerNorm/C with affine inputs, and Softmax/W. The existing Axis4/D rejection suites remain green. |
| 2026-09-01 | 6 (pre-removal) | `make pt2.runtest` | pass | The complete gated functional-model differential, Native graph/transform, symbolic-verification, Native4D lowering, and explorer cram cohort passed before further Pixel-authority removal. |
| 2026-09-01 | 3 | `make precommit`; `opam exec -- dune exec bin/region_compute_bench.exe` | pass | The permanent extent/region-count sweep now times Native Direct alongside Pixel and Kernel Region paths and records its Region counters. At fixed `K=32`, `R=1,4,16,64` (including `R > K`) has direct counters equal to Kernel Region counters and linear in owned outputs. Timing and allocation include executor/environment overhead; the counters do not observe membership tests. |
| 2026-09-01 | 6 (partial) | `make precommit` | pass | Native and Native4D Symbolic always construct the authored Region stages. Kernel adaptation takes each Stage's uniform computation directly; the `regionizers` flags, `regionized` results, candidate maps, and candidate/reconstruction admission fallback are deleted. Pixel dispatch rejects the three Region-authored operations. |
| 2026-09-01 | 6 (partial) | `make precommit`; `make pt2.runtest` | pass | The transitional `Regionizer`/`Regionizer4` modules are removed. `Region_computation` now names the Native graph-boundary contract, and `Region_computation4` owns only Axis4 mapping before delegating to it. The complete precommit and gated model cohort remain green. |
| 2026-09-01 | 6 (partial) | `make precommit` | pass | RMSNorm, LayerNorm, and Softmax's handwritten scalar implementations are now explicitly `Legacy_pixel` and are referenced only by tests. The Region execution regression compares every one against its corresponding legacy scalar oracle bitwise. |
