# Statically planned tensor arena — design record

Covers `lib/interval_alloc` (the standalone allocator), `lib/native/alloc_script.ml`,
`arena_plan.ml`, `arena.ml` and `arena_run.ml`, and `Eval_direct`'s destination
passing (`lib/native/tensor.ml`'s `write_*`/`create_of_sig`/`*_into` family,
`Node_executor`/`Region_executor`'s `~dst`/`~dsts`). Scope is Native
(`Eval_direct`) only — Native4D (`Eval_direct4`) has no inference caller and
stays release-only (see `native_tensor_release_design.md`). Builds on the
landed release work there.

Implemented per an untracked implementation plan, tracked in an untracked
implementation tracker (both hold working-file plans/censuses, not the design
record; see CLAUDE.md's exploration-and-planning section for why they are not
named here).

## 1. Pipeline

1. A **dry run** of `Eval_direct` (`Eval_direct.dry_run`) emits an alloc/free
   **script** (`Alloc_script.t`) without computing anything: the same fold
   skeleton (`Eval_direct.walk`) that a real run uses decides which outputs are
   allocated and when each edge is released, so the two can never disagree by
   construction.
2. `Arena_plan.create` splits the script's eligible events by Bigarray kind
   into one `Interval_alloc.Script.t` per kind, solves each with
   `Interval_alloc.solve_best`, and keeps only witnessed placements
   (`Interval_alloc.check`).
3. `Arena.create` allocates one pool per kind from the plan and hands out typed
   slots (`Array1.sub` views).
4. `Eval_direct.run ?arena` binds each eligible output to its slot instead of a
   fresh allocation.

## 2. Eligibility

An edge is **eligible** — the arena may back it — iff:

- some node releases it (`Release_schedule.after` names it, under the run's
  effective `retain`), **and**
- its declared format is not quantized (`Tensor_sig.quant = None`).

A graph output and a `retain`-kept edge are never released, hence never
eligible: the arena backs only intermediates, and the result of a run holds no
arena memory. A quantized edge (I8/I16) is never eligible either — the pool
holds one Bigarray kind, and `Kind.of_fmt` has no case for a quantized element
that would tell two differently-scaled quantized tensors apart at the storage
level, so quantized edges always come from a fresh `Tensor.create_of_sig`.

`Alloc_script.Kind.t` is one constructor per Bigarray element kind an eligible
format uses: `Bool` and the unsigned-8-bit formats share `Int8_unsigned`,
`F16`/`BF16` share `Int16_unsigned`. `I16`/`I8` (quantized) have no `Kind` at
all — `Kind.of_fmt` only classifies formats that can be eligible.

## 3. The witness contract

`Interval_alloc.Witness.t` is abstract: only `Interval_alloc.check` produces
one, from a `Script.t` and a `Solution.t`. It requires every block placed
exactly once, at a non-negative offset inside the pool, with no two live
blocks overlapping. `Arena_plan.t` keeps the whole witnessed script, not just
the offsets, because that script is what the run-time binding (§4) compares
against.

`Solution.Unsafe.make` lets tests build a bad claim, so the checker's
non-vacuity (does it actually reject a bad placement) is provable rather than
assumed — see `interval_alloc`'s own test suite (`test/interval_alloc/`) and
its rejection-mutation tests.

## 4. Script binding — a plan runs only on its own run

Node ids, order and per-edge signatures are not enough to prove a plan matches
a run: two different `retain` settings, or two structurally different graphs,
can agree on all three while disagreeing on which edges are actually released
(and thus eligible). So `Eval_direct.run ?arena` **dry-runs the graph under its
effective `retain`** (the argument, or `All`) at entry and requires the result
to equal the plan's own witnessed script
(`Alloc_script.first_difference`), before any node executes. A difference is
`` `Arena_script_mismatch ``, naming the first differing event (an `Alloc`, a
`Free`, or a `Node` marker — scripts are compared as complete, ordered event
sequences, node markers included, so a swapped node pair is caught even when
every alloc/free by itself still matches).

This is strictly more than an id/order/format check: an omitted `retain`
(defaulting to `All`, under which nothing is ever eligible) against an
`Only empty` plan differs at the very first `Alloc`'s eligibility flag, not at
any id or order.

## 5. Ownership contract

- **A view is valid for the run that acquired it, no longer.** The plan reuses
  a slot's cells once its edge is released within that run; nothing may hold a
  reference to a view past its release. `Eval_direct.run` never returns one to
  its caller (§2: outputs and retained edges are never eligible).
- **Poison on acquire**, not once per run: `Arena.create ?poison` fills a
  slot's cells with a poison value every time `Arena.view` hands it out, not
  only at the start of the run. This is required because a slot within one run
  can be acquired by more than one edge in sequence; poisoning only at
  `create` time cannot catch a writer that leaves a stale value in a slot
  reused later in the *same* run (proven by a non-vacuity test: entry-only
  poison stays green on exactly that case). Two poison values, `A` and `B`,
  are both finite and representable in every kind; a paired run under each
  must agree bit for bit, or some cell went unwritten. This is a diagnostic,
  not a proof — downstream compute can still mask a missed cell.
- **Reentrancy.** `Arena.t` carries an in-use flag (`Arena.with_run`, built on
  `Fun.protect`), so entering a run with a busy arena is `` `Arena_busy ``
  rather than corrupting an in-flight run's views, and the flag clears on
  every exit path, including a mid-graph error.
- **Mixed mode.** A node executor's `~dst` is the arena slot for an eligible
  output, but an executor is free to ignore it and return a tensor of its own
  (a legacy Loop executor does, until A6). `Arena.settle` decides what to bind:
  descriptor identity (`==` against the exact `dst` the executor was handed) —
  if the executor returned that object, its cells are already in place; any
  other tensor is copied in (`Tensor.blit_into`, which requires matching
  format, quantization and shape) and counted as a mixed-mode copy. Descriptor
  identity is the right question here (did the arm return what it was
  given); it says nothing about whether two *different* descriptors share
  backing memory — an `Array1.sub` view over a pool is a distinct descriptor
  from the pool array itself, which is why the "does the returned tensor
  alias the arena" test (§7) cannot use `==` and is behavioral instead.

## 6. Admission

`Arena.Admission.t = Best_effort | Required of budget_bytes`.

- **`Best_effort`:** a plan the arena cannot take — over a per-kind ceiling
  (`` `Arena_over_limit ``, checked against `Kernel.Limits.Hard.numel` and
  `max_bytes` before any pool is allocated: a partial arena is never built) or
  a pool allocation failure — is not an error. The caller (`Arena_run.acquire`)
  reports the reason and the run proceeds release-only.
- **`Required budget`:** the run's scoped footprint (§8) must fit `budget`
  *before* any node runs, or the run is rejected (`` `Over_budget `` or the
  arena's own row) with no fallback. `Arena_run.with_arena` is the entry point
  that enforces this: under `Required`, `f` (the caller's evaluation) is never
  invoked if acquisition fails.

## 7. Non-vacuity — what a mutation must break

Each proof reverts one fix, confirms the relevant test goes red, then restores
it (`test/native/eval_direct_arena_test.ml`, `test/interval_alloc/`):

- the checker rejects a shifted offset (`Overlap`), a dropped placement
  (`Unplaced`), a pool one byte short (`Out_of_pool`), and `int64` extremes
  (overflow rows) — `interval_alloc`'s own suite;
- entry-only poisoning (once per run rather than once per acquire) stays green
  on a writer that skips the last cell of a slot reused within one run — only
  poison-on-acquire catches it;
- comparing only the node-id sequence (not the whole script) lets a
  changed-operand or an omitted-`retain` mismatch through — the whole-event
  comparison (§4) is required;
- disabling the mixed-mode copy (`Arena.settle`) breaks equivalence for an
  executor that returns a tensor other than its `dst`;
- forcing two overlapping edges past the checker (bypassing `Interval_alloc.check`
  in a scratch patch) breaks arena-vs-release equivalence — proving the arena
  path actually depends on the witness, not merely on the plan's own bookkeeping.

## 8. Reported figures — the scope of the memory guarantee

`Arena_plan.Stats` and `Arena_run.Report.t` keep three figures separate, and
none of them is claimed to be the process's total memory:

- **pool bytes** — what the arena's pools actually hold;
- **out-of-arena payload bytes** — the script's *ineligible* allocations
  (outputs, retained edges, quantized edges): these plus the pool bytes are
  the two terms a `Required` budget is checked against;
- **mixed-mode copy count and bytes** (`Arena.Copies.t`) — visible so a
  regression (a formerly-`dst`-honoring executor that stops) shows up as a
  number rather than silently costing time.

Constants, the graph's own inputs, and anything a caller retains are never
counted as arena savings.

## 9. Measured fragmentation (A2 baseline)

On every `PT2_MODELS_CRAM` model (lowered and canonical, `Only empty`), the
best constructive strategy already equals the lower bound (0% gap, "optimal
for this node order") at 0 search iterations — see the tracker's fragmentation
table. The default search budget (50 iterations per kind) is a headroom
figure, not one the cram models currently need; A2r ("a stronger placement
method") was not needed. The node order itself is out of scope (design's own
§7, this document's home in the untracked design predates this record); if a
future model needs a smaller arena than its order allows, the next lever is
scheduling, not the allocator.

### 9.1 Corpus evaluation (the 100 tracked model.json graphs)

The cram models turned out to be the easy cases. `make arena.eval`
(`bin/arena_alloc_eval.ml`) measures every strategy, the order search from each,
the portfolio and the reference minimum (`Interval_alloc.Reference`) on the
payload-free normalized graphs of the whole pinned corpus. Both dialects' scripts
come from dry runs of their own graphs (`Eval_direct.dry_run`,
`Native4d.Eval_direct4.dry_run`, `Only empty`), projected per kind by
`Arena_problem`, the extraction `Arena_plan` itself uses. Classification of
refusals is `Me_classify`'s, so it agrees with the committed model-support
report.

- **What is committed.** `make arena.eval` is the gated cram
  `test/arena_eval_corpus_cram.t`. It runs in CI's build job, never in
  `runtest`. Its expected output is each model's and dialect's proven bounds,
  and the arena of every strategy at every budget and seed, summed over kinds,
  in bytes. It carries no timings.
- **Why it is deterministic.** The search is seeded, the reference is bounded
  by states rather than by a clock, and the grid is fixed inside the `.t`.
- **What a diff means.** A change there is an allocator producing a different
  arena. Review it before promoting, as for any cram. That table is also the
  evidence for choosing the default strategy and budget.
- **The full run.** `make arena.eval.report` is the same evaluation with its
  timings, markdown summary and placements, for a local look.

Figures below are from the first run (budgets 0/25/100/400, seed 1, reference
100k states, depth 256):

- **Coverage.** Native evaluated 100/100. Native4D evaluated 97/100; 3 are
  refused at conversion as `outside_dialect_domain` (`bat_resnext26ts`,
  `eca_halonext26ts`, `lambda_resnet26t`). There were no failures. Every pool
  the production planner chose passes admission.
- **Every proven minimum is the live bound.** That is 100 of 102 Native kinds
  and 98 of 99 Native4D kinds, closed either by the reference search or by a
  strategy's checked placement reaching the bound. No problem needed an
  optimum above the live bound. Three stay open, each with a gap under 1.4%:
  `convit_tiny` (Native) and `nf_regnet_b0` (both dialects).
- **The constructive strategies alone are not enough.** The portfolio with no
  search reaches the live bound on only 81 of 102 kinds. It reaches 88 with 25
  iterations, 90 with 100 and 99 with 400. Single strategies can be far off:
  `greedy_by_area` on `regnetz_d8` is 55% above the bound.
  - The search, not the portfolio, is what closes these.
  - The portfolio's own search, which keeps its winner's decoder, beats the
    generic first-fit `improve` from any single strategy.
  - The default budget of 50 is therefore a real trade-off on this corpus, not
    just headroom.
- **Dialect makes no difference to the allocator.** On the 97 paired models
  the Native4D and Native best pools and bounds are within 0.1 MiB of each other
  out of about 927 MiB in total.
- **Planning cost.** About 1 s per model for the whole matrix per dialect, with
  400 iterations the dominant term. Import and normalization, which are shared,
  take about the same again.

On this evidence, an SMT backend would add proof power where it isn't needed.
The only open gaps are three sub-1.4% cases. Search budget, and scheduling for
anything below a node order's live bound, are the levers that matter.

## 10. Default, prepared runs, Native4D

**`Native_interp_exec` stays opt-in (`?arena`), not default-on.** A one-shot
process (`native_graph eval`, native, `mobilenetv2_050`/`regnetx_002`/
`fastvit_sa12`) shows no memory win from the arena — flat to slightly worse
observed RSS than release-only, with wall time unaffected either way. This
matches the release design's own finding that a released Bigarray is already
reclaimed within 15–27 MiB of `peak_bytes` without an arena: a single cold
run's GC already does this job, so a statically sized pool has nothing to win
against there. Repeated in-process inference (native, 10 samples,
`mobilenetv2_050`) shows no RSS growth either, so nothing in this design
motivates flipping the default for that case either, on the measurements
taken.

The arena's value is destination passing's permanent format/quantization
guarantee (§7 above, unconditional on `?arena`) and the *option* of amortizing
planning across repeated runs — which is exactly what prepared runs (plan
stage A7) would add, and were **not built**: the gate they depend on (G0
showing memory growth or GC pauses under repetition) did not fire on the
measurements taken (native: no growth; node: inconclusive — a repeated run
under node slowed sharply on one sample in a way neither confirmed nor ruled
out a GC pause, in an environment with other contention). A future pass
revisiting node's own repeated-inference behavior, on an uncontended machine,
is the natural next step if prepared runs are wanted.

Native4D (`Eval_direct4`) has no inference caller today and is out of scope;
this record does not claim anything about it.

## 11. CI coverage

Every other check of §7's non-vacuity table runs the arena over synthetic
`Graph_fixtures` graphs, in-process. Two CI checks run it over a real model
instead, one per backend:

- **Native**: `make native-infer-verify-arena` (`.github/workflows/build.yml`,
  alongside `native-infer-verify-direct`). `native_graph eval --arena` forces
  `Admission.Best_effort` for `mobilenetv2_050` and compares the output
  against the real ATen reference, the same way `native-infer-verify-direct`
  does without it.
- **jsoo/node**: `make loop_js.node.pt2.runtest` (`.github/workflows/js.yml`'s
  "jsoo pt2 node coverage" step) runs the same model, under node, through
  `Loop_js_exec`-compiled JavaScript, with `--arena` added to its existing
  `--nodes --shadow --strict` invocation — one run, not a second pass,
  since outputs are bit-identical either way. `js/jsoo/loop_js_pt2.ml`'s
  `--arena` is the jsoo counterpart of `native_graph eval --arena`.

Both flags print whether the plan was used or declined (one line: `arena:
used ...` / `arena: declined: ...`), and both targets fail on a decline as
well as on a wrong answer, so a regression that makes the plan inadmissible
for this model is caught here too, not only in the RSS table above.
`Native_interp_exec` itself is still opt-in on both backends — these are
permanent CI checks of the path, not a change to the default.
