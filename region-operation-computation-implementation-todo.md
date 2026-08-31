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
  Pixel algorithm.
- Compatibility rule: fresh scalar projection is permitted at named scalar
  boundaries but never as production tensor materialization.

## Progress

| Gate | Scope | Status | Commit |
|---|---|---|---|
| Foundation prerequisite | Region language and Pixel fast path | complete | `2a81e7b`; closeout `787f278` / `c50be5b` |
| Scalar-Region prerequisite | Optional regionizer and dedicated executor | complete | `c8fe41f`; ownership refactor `e605361` |
| 0 | Contract census and migration baselines | pending | — |
| 1 | Scalar projection and whole-domain trace contract | pending | — |
| 2 | Operation-authored computation boundary | pending | — |
| 3 | Native Region authority and Direct execution | pending | — |
| 4 | Symbolic and Stage-program Region carriage | pending | — |
| 5 | Native4D Region authority and execution | pending | — |
| 6 | Remove dual authority and obsolete admission | pending | — |
| 7 | Closeout and production admission | pending | — |

## Gate checklist

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
- [ ] Pass output ordinal/shape and role-resolved operands explicitly.
- [ ] Separate generic Region construction from normalization-only helpers.
- [ ] Eliminate floating-value lookup for synthetic optional operands.
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

- [ ] Remove handwritten Pixel RMSNorm computation.
- [ ] Remove handwritten Pixel LayerNorm computation.
- [ ] Remove handwritten Pixel Softmax computation.
- [ ] Replace legitimate scalar consumers with bounded derived projection.
- [ ] Remove/narrow `Regionizer`, `Region_context.pixel`, synthetic-value lookup,
      candidate maps, and primary reconstruction fallback.
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
- [ ] Benchmark Native and Native4D Direct plus Kernel Region execution.
- [ ] Re-run Pixel no-regression benchmarks.
- [ ] Audit production call chains for Pixel, Region, and projection routing.
- [ ] Update final APIs, evidence, measurements, and deferrals in design docs.

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

## Validation record

| Date | Gate | Command | Result | Notes |
|---|---|---|---|---|
| — | — | — | pending | No implementation gate has started. |
