# Region computation for SDPA — implementation tracker

Live execution record for
[`region-sdpa-computation-plan.md`](region-sdpa-computation-plan.md)'s Stage A
(§2) and Stage B (§3, shaped vector locals). Both are now complete — see
"Stage B" below for the second milestone, landed in a follow-up session after
Stage A's baseline was measured, per §4's recommendation.

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
fold). See "Stage B" below for the plan's remaining section.

## Next steps (separate plans)

1. `native4d-sdpa-compatibility-plan.md`'s Sdpa half is unblocked now that its
   stated dependency (this Stage A) is met — pick that up next if the Native4D
   gap is wanted. **Done**, separately: see that plan's own tracker.

## Stage B (§3): shaped vector locals — complete

Landed in two commits: `0584bca` (the language/Region-program foundation —
`Expr.Value.Local_at`, `Region_local.Shape.Vector`, `Region_program`'s shape
agreement and bounds-checked aggregate slot-count checks, `Region_execution`/
`Region_eval`'s offset+count slot ranges) and `0a0bc41`
(`Attention.Sdpa.Computation` actually building the two vector locals, `s`
and `p`, plus the `Expr.Check.fragment ~allowed_free` fix that landing `s`
immediately exposed).

### Scope lock

- Included: §3.3's steps 1 (`Shape.Vector`), 2 (`Value.Local_at`), 4 (shape
  agreement in `Region_program.check`), 5 (`Region_execution`/`Region_eval`'s
  vector slot ranges, `Eval.value`'s `~local_at`/`~reducer`), 6
  (`Rewrite.substitute_locals`'s `local_binding` variant and the new
  `substitute_reducer` beta-reduction), 7 (a bounds-checked aggregate
  slot-count limit), 8 (`Region_execution`'s `locals` counter now counts
  vector elements, not just local declarations — the "count elements" choice
  §3.3 item 8 left open). §3.1's program: `s` (the score row) and `p` (the
  normalized softmax weight row), both cached — the "better constant" variant
  §3.1 names, not the footprint-only one-vector alternative.
- **Step 3 turned out already done.** The plan's step 3 ("Expr.Builder must
  be able to mint a binder outside a reduction... a Builder-only minter,
  never a public constructor") describes exactly `Builder.fresh_reduce`,
  which was already public (landed with Stage A, for `Rewrite.freshen`'s own
  minting and for scope-violation test fixtures). No new `Expr.Builder` API
  was needed; `Region_program.Builder.vector` and `Attention.Sdpa.Computation`
  both just call it directly, the same way `Builder.reduction` already does
  internally.
- **One gap the plan's own step list did not name**: `Expr.Check.fragment`'s
  free-reducer check has no notion of "this identity is deliberately free by
  design" — it exists specifically to catch composition defects, and a vector
  local's own binder is, by construction, free in its own stored value. This
  needed a targeted exemption (`~allowed_free`), scoped to exactly one
  identity per vector local, in `Region_program.check`'s per-local call.
  Found immediately by `reconstructs`'s own test the moment a real vector
  local existed — see "Verification" below.
- Excluded, as the plan itself scopes: online/blocked softmax (§3.2's
  numerical point — caching is a value substitution, not a reassociation, so
  it needed no tolerance policy or second oracle; folded into
  `.ai/region_compute_design.md`), a `V`-element output accumulator,
  loosening `total_work_bounded` (§3.4's explicit "do not do this in the same
  change").

### What was built

`Region_program.Builder.vector` mirrors `scalar`'s calling convention as
closely as the shape allows: it self-mints both the local's `id` and its own
per-element binder `var` (via `Expr.Builder.fresh_local`/`fresh_reduce`), then
hands the CALLER a *reader* — `Role.Position.t Index.t -> Expr.Value.t` — not
a value, since a vector local has no single value to hand back. The body
callback it takes has exactly `Expr.Builder.reduction`'s own shape (a
symbolic index in, a `Value.t Builder.t` out), so `Attention.Sdpa`'s existing
`score_at` function — unchanged from Stage A — plugs in directly as `vector`'s
first argument with no adaptation.

`p`'s own vector body reads `s` (the first vector local) at `p`'s own
per-element index via `s i` (`Value.local_at s_id (Index.reduce i_var)`) — a
vector local reading ANOTHER vector local at its own binder, which is well
within the mechanism's design (the plan's §3.1 pseudocode does the same:
`p[i] = exp(s[i] - m) / z`) and needed no special-casing anywhere: `s i` is
just an ordinary `Value.t`, indistinguishable at that point from any other
subterm.

`Region_program.check`'s new aggregate slot-count bound reuses `max_size`
(the program's existing general resource budget) as the ceiling, rather than
inventing a new, separately-tuned `Kernel.Limits` field — CLAUDE.md's
32-bit-aggregate rule is about the SUM of individually-bounded local extents
overflowing, and folding that into the budget already threaded through
`create`/`check`/`Builder.finish` was simpler than adding a parameter no
current op needs to tune independently.

### Verification

- **`Region_program.reconstructs` against `Legacy_pixel`** — the same test,
  same fixture, as Stage A's proof (`test/native/region_compute_test.ml`,
  "Authored Regions reconstruct their legacy scalar oracles"). This is the
  load-bearing claim: substituting `s`/`p` back through `specialize_pixel`
  beta-reduces each `Local_at` to exactly `Legacy_pixel`'s expression, so
  results are bitwise equal by construction. First run FAILED with
  `Free_reducer` (the `~allowed_free` gap above) — the test caught the defect
  immediately, before any hand-inspection would have.
- **`lib/native_op_walk/sdpa_nwalk.ml`'s Direct-vs-Symbolic fuzz walk** —
  unchanged, green across the whole config space (batch/heads/wq/wk/e/
  mask/scale), the same evidence Stage A relied on, now covering Stage B's
  program shape too (Direct evaluates the two vector locals via `~reducer`
  per position; Symbolic grounds the beta-reduced form).
- **`Region_execution`'s own counters**, same fixture as Stage A's tracker
  (query `[W=2,C=3]`, key/value `[W=3,C=3]`, real mask, wk=3 e=3):
  `locals` 6 -> 18 (now counting vector ELEMENTS, per §3.3 item 8's "count
  elements" choice — `sf(1)+s(3)+m(1)+z(1)+p(3) = 9` per key × 2 keys),
  `loads` 228 -> 60, `reductions` 120 -> 48. Fewer loads/reductions is the
  direct evidence `score_at`'s dot product is no longer recomputed three
  times per key.
- **`make precommit`** (build, format, full `dune runtest`, file-size,
  whitespace) green throughout, including `test/expr/value_test.ml`'s new
  inline coverage of `Value.local_at`/`Eval.value`'s `~local_at`/`~reducer`/
  `Rewrite.substitute_locals`'s `Vector` case at the library level, ahead of
  and independent from the native-level SDPA proof above.
- Native4D: untouched by Stage B's own commits, and inherits it automatically
  — `Region_computation4.native_op` already maps `Op.Sdpa` straight onto
  `Graph_ir.Sdpa`, so Native4D's Sdpa route (landed by
  `native4d-sdpa-compatibility-plan.md`, `97163cf`) runs the SAME
  `Attention.Sdpa.Computation.program`, now Stage-B-shaped, with no code
  change of its own. The full `dune runtest` run above includes Native4D's
  own tests and passed unchanged.

### Wall-clock measurement against Stage A — now done, and reverses the caveat above

Re-measured (`opam exec -- dune exec bin/region_compute_bench.exe`, median of
20 samples, stable across three repeated runs). Stage B's `kernel_region_ms`/
`direct_ms` are 4.8x-11x faster than Stage A's own recorded numbers at
identical `report_sdpa` shapes, and now BEAT `legacy_pixel_ms` (the uncached
oracle) at every point except the smallest — a reversal of the "operation
count, not wall clock" caveat this section used to defer to. Folded into
`.ai/region_compute_design.md`'s "Design headroom" and "Cost model and
evidence" sections, including the comparison table; that tracked doc's
previous claim ("`Legacy_pixel` is faster in absolute terms at every size
measured") was corrected in place, not merely appended to.

**A real bug surfaced by taking this measurement, now fixed**:
`Region_execution.evaluate_locals`'s `fill` dispatched on a local's numeric
slot count rather than its declared `Shape.t`, so a `Vector` local of extent
1 (SDPA's `s`/`p` at `Wk = 1` — `report_sdpa`'s own first sweep point) was
silently evaluated as a scalar, with no `~reducer` bound, and raised
`Unbound_reducer` — through `Eval_direct.run` itself, not just the
benchmark's `Kernel_eval` path. Neither the `sdpa_nwalk.ml` fuzz walk (whose
`wk` axis includes `1`) nor any existing unit test happened to land on it.
Fixed in `lib/native/region_execution.ml` (dispatch on `Region_local.shape`);
regression test added at `Wk = 1` in `test/native/region_compute_test.ml`,
bitwise against `Legacy_pixel`, confirmed to fail without the fix (reverted
the fix, watched it fail with the exact `Unbound_reducer` backtrace, restored
it — CLAUDE.md's "prove the check can fail"). Both changes are uncommitted as
of this note; land as their own commit(s), separate from any §3.4 work.

**Known cosmetic gap, not fixed**: `Region_program.pp` (used by
`Region_trace`, `Kernel.pp`, `Stage_program.pp`, and Model Explorer's
`me_kernel.ml`) has no way to give a vector local's own binder a display
name — `Expr.Pp.value_open`'s `names` parameter only names `Local_var.t`
occurrences, not free `Reduce_var.t` ones, so `s`/`p`'s own per-element index
prints as the generic free-reducer fallback `?#N` rather than something like
`i0`. Not wrong (it IS free relative to the printed local's own scope), just
less legible than the plan's own `score_at(i)` pseudocode. No test currently
pins this rendering. Fixing it means widening `Expr.Pp.value_open`'s public
signature with a free-reducer naming override — small, but its own change.
