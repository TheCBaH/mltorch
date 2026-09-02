# Region computation for SDPA — implementation tracker

Live execution record for
[`region-sdpa-computation-plan.md`](region-sdpa-computation-plan.md)'s Stage A
(§2). Stage B (§3, shaped vector locals) is not started — §4's own
recommendation is to land Stage A alone first and measure Stage B against it,
so this tracker stops at Stage A's closeout.

Ordering note: `native4d-sdpa-compatibility-plan.md` names two independent
changes (Sdpa, Batched_matmul); its Sdpa half depends on this plan's Stage A
and is the natural next step once this closes. `Batched_matmul` depends on
nothing here and could land in either order.

## Scope lock

- Included: `Attention.Sdpa.Legacy_pixel` rename (test-only oracle, matching
  the `Reduce.Softmax`/`Norm` migration); `Attention.Sdpa.Computation`, the
  authoritative `Whole [C]` Region program sharing `sf`/`m`/`z` per key;
  `Region_context.broadcast_coord`; `Region_computation`'s `Sdpa` arm and
  `Sdpa_mask` synthetic role; `Eval_op`'s `Sdpa` arm becoming the standard
  Region-authored impossible-arm; `reconstructs`-against-`Legacy_pixel` and a
  Direct materialize/counters test; a `region_compute_bench` row.
- Excluded (Stage B and beyond): shaped vector locals, caching `score_at`
  itself, the `Region_local.Shape.Vector` constructor, `Expr.Value.Local_at`,
  online/blocked softmax, loosening `total_work_bounded`. See the plan's §3
  and §5 for what Stage B actually needs.
- Native4D: untouched, as the plan's §2.6 item 6 says — Sdpa stays outside
  Native4D's domain until `native4d-sdpa-compatibility-plan.md` is picked up.

## Progress

| Step | Scope | Status | Commit |
|---|---|---|---|
| 1 | `Legacy_pixel` rename + `Computation` Region program (attention.ml) | complete | `7ffba06` |
| 2 | `Region_context.broadcast_coord` | complete | `7ffba06` |
| 3 | `Region_computation`/`Eval_op`/`Eval_direct` wiring, `Sdpa_mask` role | complete | `7ffba06` |
| 4 | Test migration (`Compute` -> `Legacy_pixel` call sites) | complete | `7ffba06` |
| 5 | `reconstructs` proof + Direct counters test (region_compute_test.ml) | complete | `7ffba06` |
| 6 | `region_compute_bench` Sdpa row | complete | `7ffba06` |
| 7 | Stray `opN(-impl).md`/`commit N`/bare `F<n>` citations removed from touched blocks in `attention.ml`/`reduce.ml`/`sdpa_test.ml` | complete | `7ffba06` |
| 8 | Fold SDPA into `.ai/region_compute_design.md` (routing, oracle count, operation-count-vs-wall-clock note) | complete | `27edcfd` |

## What was built

`Attention.Sdpa.Computation.program` authors exactly the plan's §2.1 program:
a `Whole [C]` partition with three scalar locals (`sf`, `m`, `z`) and an
emitter that still recomputes `score_at` per use. The two-level reduction
(`score_at`'s inner `E`-dot nested inside the outer `Wk` max/sum) is threaded
through one `Expr.Builder` state per top-level use (`m`, `z`, the emitter),
not through independent `Expr.Builder.run` calls inside the nesting — an
independent-`run`-per-level draft was tried first and correctly rejected by
`Region_program.check`'s `Duplicate_binder` path (both levels would mint
ordinal 0), which is exactly the "genuine new territory" §2.6 item 5 warned
about. `Region_context.broadcast_coord` is a new, deliberately separate
definition from `Pointwise_binary.broadcast_coord`: same per-axis rule, but
over `Expr.Coord.t` rather than `Vec6.t` (no shared parent type to fold the
two into without a larger refactor).

## Verification

- `Region_program.reconstructs` against `Legacy_pixel`, one example graph
  (query/key/value `[W=2,C=3]`, mask present): `true` — the same proof method
  the three landed ops use (`test/native/region_compute_test.ml`, "Authored
  Regions reconstruct their legacy scalar oracles"). This is Stage A's load-
  bearing claim: substituting the locals back reproduces `Legacy_pixel`'s
  expression exactly, so results are bitwise equal by construction, not by a
  numeric comparison that could pass on a lucky input.
- `Eval_direct.run` materializes the new Sdpa Region program end to end
  (query `[W=2,C=3]`, key/value `[W=3,C=3]`, real mask) with `Region_trace`
  counters recorded (`keys=2 locals=6 emitters=6 loads=228 reductions=120`),
  confirming `Region_execution`'s ownership/coverage check accepts the
  two-level-reduction shape.
- `lib/native_op_walk/sdpa_nwalk.ml`'s Direct-vs-Symbolic fuzz walk is
  unchanged and green under `dune runtest` — it now exercises the Region path
  on both sides (Direct materializes the program directly; Symbolic grounds
  it) across the walk's config space (batch/heads/wq/wk/e/mask/scale), not
  just the one hand-built graph above.
- `test/native/sdpa_test.ml`'s hand-computed goldens against `Legacy_pixel`
  are unchanged and green — the oracle itself was not touched, only renamed.
- `make precommit` (build, format, full `dune runtest`, file-size, whitespace)
  green.
- `bin/region_compute_bench.exe`'s new `sdpa` rows (Native only — no Native4D
  counterpart yet): see below.

### Bench: the ~3x claim is an operation-count model, not a wall-clock one

`region-sdpa-computation-plan.md` §2.3 states Stage A's saving as unit
"ops" (`Wk*E*(E+2)` against `3*Wk*E*E`), and §2.4's `Identical` proof is what
that claim actually rests on — correctness, not speed. Measuring wall clock
against `Legacy_pixel` directly (added to the bench specifically to check
this) shows the opposite sign at these sizes:

```
sdpa regions=1  extent=8  kernel_region_ms=0.029 direct_ms=0.035 legacy_pixel_ms=0.009
sdpa regions=4  extent=8  kernel_region_ms=0.448 direct_ms=0.439 legacy_pixel_ms=0.125
sdpa regions=16 extent=8  kernel_region_ms=7.112 direct_ms=7.010 legacy_pixel_ms=1.989
sdpa regions=4  extent=16 kernel_region_ms=1.502 direct_ms=1.476 legacy_pixel_ms=0.439
```

Region execution is ~3.5x *slower* in wall clock than the compiled
`Legacy_pixel(Direct)` closure it reconstructs, at every point measured. This
is not a Stage A regression: `Legacy_pixel` is no longer reachable through the
graph dispatcher for any of the four Region-authored ops (`Eval_op`'s arm is
`invalid_arg`, matching `Rms_norm`/`Layer_norm`/`Softmax`), so there is no
"fast path" Stage A removed — RMSNorm shows the same shape of result (a
throwaway check during this work measured `legacy_ms=0.022` vs
`direct_ms=0.026`, i.e. near parity, not a win either). The difference in
degree — SDPA ~3.5x slower where RMSNorm is roughly even — is explained by
what Stage A explicitly does not touch (§2.8): the emitter still recomputes
`score_at`'s *nested* two-level reduction once per `C` output, and a nested
reduction evaluated through `Expr.Eval`'s general interpreter carries more
per-element overhead than RMSNorm's flat single-level one. `sf`/`m`/`z`
sharing cuts the raw operation count (~2.4x fewer at `E=Wk=8`, matching the
plan's model) without cutting the interpreter's per-node constant, and at
this size the constant dominates. `region-compute-design.md` already names
the fix as future work — a specialized/compiled `Lowered.t` execution form,
not yet implemented — rather than something Stage A was scoped to provide.
Fold this measurement into `.ai/region_compute_design.md` if Stage B is
planned, since Stage B's own payoff (§3.1: `~1.5*E` per plan's cost model)
will be even further from wall-clock reality until that lowering exists.

## Status: Stage A complete

Both milestones landed (`7ffba06` implementation, `27edcfd` design-record
fold). Nothing further is planned under this plan document.

## Next steps (separate plans)

1. `native4d-sdpa-compatibility-plan.md`'s Sdpa half is unblocked now that its
   stated dependency (this Stage A) is met — pick that up next if the Native4D
   gap is wanted.
2. Stage B (`region-sdpa-computation-plan.md` §3) is explicitly deferred; do
   not start it without re-reading §3.3's build list and §5's open questions,
   and without re-measuring against this Stage A baseline (not against
   `Legacy_pixel`) per §4.
