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
| P | Owned-output enumeration and region-count benchmark | pending | — |
| 0 | Contract census and migration baselines | pending | — |
| 1 | Scalar projection and whole-domain trace contract | pending | — |
| 2 | Operation-authored computation boundary | pending | — |
| 3 | Native Region authority and Direct execution | pending | — |
| 4 | Symbolic and Stage-program Region carriage | pending | — |
| 5 | Native4D Region authority and execution | pending | — |
| 6 | Remove dual authority and obsolete admission | pending | — |
| 7 | Closeout and production admission | pending | — |

## Gate checklist

### Gate P

- [ ] Enumerate a key's owned outputs from the key and the `Whole`-axis extents
      instead of filtering a full-domain scan per key.
- [ ] Drop `fold_outputs`'s `Err.t` result if the rewrite makes
      `` `Region_key_out_of_bounds `` unreachable.
- [ ] Add a region-count axis to `bin/region_compute_bench.ml`, which currently
      fixes `W = 4`.
- [ ] Measure a sweep of `R` at fixed `K`, including an `R > K` shape.
- [ ] Confirm unchanged coordinates, order, values, and conversions against the
      existing Foundation partition-enumeration tests.

### Gate 0

- [ ] Census all Native and Native4D computation call sites.
- [ ] Classify Stage-program, Kernel, grounding, verifier, explorer, and fusion
      consumers by Pixel-only, scalar-projection, or structural-Region need.
- [ ] Record current Native and Native4D Direct repeated-reduction baselines.
- [ ] Pin current Region program printers, counters, and symbolic form.
- [ ] Add red expectations for authoritative Region carriage.

### Gate 1

- [ ] Define one named bounded scalar-projection API.
- [ ] Document concrete projection, reference `value_at`, and symbolic
      specialization as distinct operations.
- [ ] Add deterministic program/key/local/output/emitter trace representation.
- [ ] Compute output visit counts independently from tensor stores.
- [ ] Assert complete coverage, zero duplicates, and ownership agreement.
- [ ] Test materialization versus fresh projection bitwise at every output.
- [ ] Add mutation fixtures proving missing/duplicate visits are detected.

### Gate 2

- [ ] Define the operation-facing Region computation API with actual limits.
- [ ] Replace each operation's `Region.try_regionize` with a
      `Computation.program` entry returning a typed error, not a rejection.
- [ ] Delete `Unsupported_operation`; move `Missing_operand`, `Output_ordinal`,
      and `Output_shape` to the dispatcher boundary; make `Invalid_partition`
      and `Invalid_program` limit/implementation errors.
- [ ] Pass output ordinal/shape and role-resolved operands explicitly.
- [ ] Separate generic Region construction from normalization-only helpers.
- [ ] Eliminate floating-value lookup for synthetic optional operands.
- [ ] Route the normalization divisor through the bounded shared helper rather
      than re-folding an unbounded extent product in `Region_context.count`.
- [ ] Normalize Pixel and Region authored forms at the graph boundary.
- [ ] Keep dispatchers arithmetic-free and exhaustive per dialect.
- [ ] Test tight and expanded non-default limits.

### Gate 3

- [ ] Make Native RMSNorm's Region program authoritative.
- [ ] Make Native LayerNorm's Region program authoritative.
- [ ] Make Native Softmax's Region program authoritative.
- [ ] Route Native Direct through once-per-key Region materialization.
- [ ] Keep old Pixel implementations test-only during migration.
- [ ] Add the complete Native operation trace matrix.
- [ ] Prove Native Direct linear work with counters and benchmarks.

### Gate 4

- [ ] Carry `Region_program.t` directly at the symbolic Stage boundary.
- [ ] Embed Pixel symbolic bodies mechanically and preserve object identity.
- [ ] Insert authoritative Region programs directly from Native Symbolic.
- [ ] Migrate Stage validation, printing, grounding, and source analysis.
- [ ] Migrate Kernel adaptation without candidate/reconstruction admission.
- [ ] Support or explicitly reject Region stages in transformation verification.
- [ ] Preserve Pixel Stage/Kernel hot-path behavior.

### Gate 5

- [ ] Expose computation entries for `Ops4.Rms_norm`.
- [ ] Expose computation entries for `Ops4.Layer_norm`.
- [ ] Expose computation entries for `Ops4.Softmax4`.
- [ ] Share one checked parameter adapter between shape and computation.
- [ ] Route Native4D Direct through once-per-key Region materialization.
- [ ] Route Native4D Symbolic through Region-capable Stage programs.
- [ ] Remove the three Region-authored production arms from
      `Eval_op4.Make(S).pixel`.
- [ ] Trace N/H/W/C cases with T/D visibly singleton.
- [ ] Prove Native/Native4D mapped program and result agreement.
- [ ] Preserve every existing Axis4/domain rejection.

### Gate 6

- [ ] Run the gated `.pt2` model-level differential BEFORE any removal.
- [ ] Demote handwritten Pixel RMSNorm computation to test-only, keeping
      `reconstructs` asserting against it.
- [ ] Demote handwritten Pixel LayerNorm computation on the same terms.
- [ ] Demote handwritten Pixel Softmax computation on the same terms.
- [ ] If deleting outright instead, record the decision and that the arithmetic
      then has one in-tree source.
- [ ] Replace legitimate scalar consumers with bounded derived projection.
- [ ] Remove `Regionizer` entirely, along with `Regionizer.candidate`, the
      `regionizers` flag on `Eval_symbolic.build`, and the `regionized` result.
- [ ] Remove `Region_context.pixel`, `reject`, and synthetic-value lookup;
      retain the construction helpers under the Gate 2 module.
- [ ] Remove candidate maps and primary reconstruction fallback.
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
- Current Native `Eval_direct` reaches the three operations through
  `Eval_op.Make(Direct).pixel` and `Schedule.evaluate`.
- Current Native `Eval_symbolic.run_regionized` builds Pixel first, retains a
  candidate map, and admits Region later through reconstruction.
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
| — | — | — | pending | No implementation gate has started. |
