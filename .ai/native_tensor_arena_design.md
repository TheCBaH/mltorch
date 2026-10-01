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
   into one `Interval_alloc.Script.t` per kind, in bytes with each block at
   the alignment its script `Alloc` records (§2.1; never below its cell width,
   so every offset is a whole number of cells, and slots and pools are reported
   in cells), solves each with
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

### 2.1 Alignment

`Alloc_script.alloc` records each edge's `bytes` (its `Element_count` times its
kind's `Element_bytes`, both `Core.Storage_units` types) and an `alignment`
from `Alignment_policy.default`: a 64-byte cache line up to and including one
4096-byte page, a page above it, and never less than the element's own width.
A host may raise that floor: `Alignment_policy.t` is the standard policy or
the standard policy with a host request (`with_host`), and the alignment is
the largest of the size default, the element width and the request, so a
request can strengthen the default but never weaken it. The policy is part of
the plan's identity. `Eval_direct.dry_run ?alignment` records the policy's
alignments in every `Alloc`, `Arena_plan` keeps the policy it was made under,
and a run dry-runs under the plan's policy, so the whole-script comparison (§4)
covers it: a plan labelled with a policy its script was not made under is an
`` `Arena_script_mismatch ``. Reusing a plan under another policy goes through
`Arena_plan.revalidate`, which keeps sizes, lifetimes and offsets (so
disjointness and bounds still hold) and rechecks only each start against the
new alignment, refusing the first misaligned slot. Offsets are relative to a
pool: they are physically aligned only if the pool's base is.

A slot is placed at its exact payload size, only its start aligned: start
alignment does not reserve the rest of a cache line or page, so a smaller slot
may use the tail of a page-aligned one. Padding enters only the minimum.
`Arena_problem` poses each kind twice, as an exact script and as a padded one
(every block `Byte_alignment.pad`ded to its alignment). The padded script's live
bound counts the alignment gaps, so its optimum is provable where the exact
script's live bound, which leaves them out, is often not closed; the reported
per-kind and combined bounds are the padded ones. Any placement of the padded
script is also a placement of the exact one, so the exact optimum lies between
the exact live bound and the padded minimum.

`Arena_plan.place` runs the portfolio on both scripts, checks each placement
against the exact script, cuts it to its actual length (`Interval_alloc.fit`),
and keeps the shorter. Placing exact sizes alone lost on six corpus models —
the strategies order differently once sizes are no longer multiples of their
alignment, up to 47% on `tf_efficientnet_lite3` — and keeping the padded
placement as a candidate means a plan never holds more than padding would.

Each `Arena_plan.Pool` records the strictest alignment of its slots, the base
alignment its storage would need, and `Arena_plan.base_alignment` is the
strictest over pools. No backend meets it: a Bigarray is malloc-backed and a
JavaScript ArrayBuffer guarantees about 8 bytes, and pure OCaml cannot observe
an address. So `Arena.physical_alignment` is `Logical_only` on every backend,
and the report says so rather than declining every arena. A caller that needs
the physical guarantee says so: `Arena_run.acquire ~physical:Physical_required`
refuses (under `Required`) or declines (under `Best_effort`) a plan whose base
alignment the backend cannot give, as `` `Physical_alignment_unsupported ``,
before any node runs — the logical alignment is never passed off as physical.
A physically aligned native pool (an aligned allocation behind a managed
Bigarray) is future work.

## 3. The witness contract

`Interval_alloc.Witness.t` is abstract: only `Interval_alloc.check` produces
one, from a `Script.t` and a `Solution.t`. It requires every block placed
exactly once, at a multiple of its alignment inside the pool, with no two live
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

`Arena.Admission.t = Best_effort | Required of Byte_size.t`.

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

Physical alignment (§2.1) is a separate, explicit requirement
(`Arena.Physical_requirement`), checked with the same refuse-or-decline split.

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

Sizes and offsets throughout (`Arena_plan.Slot`/`Pool`/`Stats`, the admission
budget, the report) are `Core.Storage_units` types; a slot's byte offset
becomes an element offset only in `Arena.view`, by an exact conversion.
`Arena_run.Report` also carries the base and physical alignment (§2.1).

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
`Arena_problem`, the extraction `Arena_plan` itself uses. Every figure is of the
padded script (§2.1), where a minimum is provable, except `placed_bytes`: the
pool `Arena_plan.place` actually holds at the largest budget, exact sizes. Classification of
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

The figures above predate alignment (§2.1), when blocks were aligned only to
their cell width. The committed cram now carries the aligned, padded figures.
With the default portfolio (50 iterations, seed 1), alignment adds 0.03% to
Native's summed arena and 0.07% to Native4D's; the largest single increase is
`nf_regnet_b0` (Native4D), 2.9%. Padding the sizes costs 0.003% on top of
aligning the starts alone, and it is what keeps the minimums provable: with
aligned starts but unpadded sizes, only 88 of 100 Native and 86 of 97 Native4D
models were proven, since the live bound left the padding out. Padded, 99 and
97 are, and `convit_tiny` (Native, 1.36%) is the only open gap. That is why
the minimum is still computed padded while placement is exact (§2.1): the
placed pool (`placed_bytes`) is never above the padded production pool, below it
on 53 of 197 model/dialect pairs, and 0.013% smaller in total.

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
used ...`, from `Arena_run.Report.pp`, / `arena: declined: ...`), and both targets fail on a decline as
well as on a wrong answer, so a regression that makes the plan inadmissible
for this model is caught here too, not only in the RSS table above.
`Native_interp_exec` itself is still opt-in on both backends — these are
permanent CI checks of the path, not a change to the default.

## 12. Storage roles, layouts and ownership

§1–§11 place only intermediates: `Alloc_script` omits constants and graph
inputs and makes outputs ineligible, and that stays the default path. Arenas for
every storage role build on a second script, not on changing an eligibility
flag.

### 12.1 The storage script

`Eval_direct.storage_script ?alignment ?retain config g` is `dry_run`'s fold
with the rest of a run's storage added (`Storage_script`): the used constants
after a `Model_init` boundary, the graph inputs after `Input_population`, each
node's outputs and releases, then `Result_publication` and `Result_release`,
after which every block the run still owns is freed. Each block carries its
`Role` (`Constant`, `Input`, `Intermediate`, `Output`) and the logical arena it
lives in (`Arena_id`), or none.

- **Config.** `Layout.Separate` (constants, copied inputs, intermediates and
  outputs each in their own arena) or `Layout.Shared_execution` (constants
  alone, one `Execution` arena for the rest), and an `Ownership` for constants
  and for inputs: `Borrowed` (the caller's payload, used in place, never
  reclaimed) or `Copied` (copied into a slot, which only the copy occupies).
  A runtime-owned input slot filled through a scoped interface is not built:
  the current tensor type cannot revoke an escaped view.
- **Lifetimes.** Constants are never freed: they are persistent model state,
  not scratch between runs. A borrowed payload is never freed either. A graph
  output or retained edge lives to `Result_release` whatever its role, so an
  input or constant forwarded as an output keeps the longest lifetime and is
  one block, counted once. A retained intermediate is an `Output`. A copied
  input is freed after its last reader like any intermediate.
- **Outside every arena.** A quantized block (a pool holds one element kind and
  no quantization parameters) and a borrowed one.
- **Identity.** Config and alignment policy are part of the script: two scripts
  differ at position 0 when either does.

`Storage_script.arena_script` projects one arena's blocks into an ordinary
`Alloc_script` (eligible iff in that arena, boundaries dropped), so each
logical arena is planned by the unchanged `Arena_problem`/`Arena_plan` of
§1–§3, with its own per-kind pools.

### 12.2 Plans, constants and runs

- **`Storage_plan.create ?limits ?budget script`** plans each logical arena that
  holds a block as its own `Arena_plan`, under the script's policy. Its
  `Footprint` keeps apart what must not be summed blindly: `constants` (held
  once per resident model), `execution` (every other arena's pools: once per
  run slot, and the result arena once per outstanding result), `borrowed`
  (caller payloads the run only reads) and `outside` (quantized edges). An
  output is counted in its arena, never again as a payload.
- **`Constant_arena.create plan ~constants`** prepares one version of the
  model's constants: copied into the `Constants` arena (never poisoned), or
  bound in place when borrowed. Every run of the model binds the same slot
  views read-only. A new model is a new `Constant_arena` in new storage with a
  new `Generation`, so it cannot overwrite what an older lease still reads: the
  old version is alive while anything references it. Load copies are
  preparation cost, reported apart from any run's.
- **`Storage_run.run`** acquires a result arena and every scratch arena
  (`Inputs`/`Intermediates` under `Separate`; none under `Shared_execution`,
  whose `Execution` arena is the result arena), copies each copied input into
  its slot, and runs `Eval_direct.run_storage`, which first requires the run's
  own storage script to equal the plan's (`` `Storage_script_mismatch ``).
  Scratch is released on every exit (`Fun.protect`); the result arena only on
  an error or exception. Poison (a test option) fills each slot as the run
  takes it, never a constant's.

### 12.3 Result leases

A successful run returns a `Result_lease`, not a map: its outputs are views
into the result arena, which stays pinned (`Arena.busy`) until
`Result_lease.release`. Returning from a run never licenses overwriting its
results. `Storage_run` keeps at most `max_outstanding` result arenas (default
one), created lazily; a run while all are pinned fails with `` `Arena_busy ``
before touching anything. `with_outputs` is scoped access and fails with
`` `Lease_released `` after release; `copy_out` copies every output into fresh
storage and releases. Release is idempotent. A lease also holds its constant
version, so a forwarded constant reads its own model's values.

An escaped raw view cannot be revoked: the tensor type has no such notion. The
guarantee is therefore that the arena behind a live lease is never reused, not
that a view taken through `with_outputs` stops working after release — reading
one then is the caller's error, and the next run may overwrite it.

Non-vacuity: letting `Storage_run` pick a pinned result arena makes every case
of the "live result is never overwritten" test report `OVERWRITTEN`.

### 12.4 Layout figures

On the `roles` fixture and every graph fixture, every layout and ownership
combination is equal bit for bit to a release run, and two poisons agree. The
layout is a measurement choice, not a free win: on `chain` shared execution
halves the execution pools (472 → 236 B), but on `residual` it grows them
(112 → 144 B), because three co-live 16 B blocks at 64 B alignment leave gaps
in one pool that separate pools each absorb at their end. Under
`Shared_execution` a lease pins the whole execution arena (144 B there, against
16 B of outputs under `Separate`), so the smaller figure is not the smaller
working set once results are held.

Measured on `mobilenetv2_050` (`native_graph eval --arena-layout
separate|shared`, one sample; `--arena` for the intermediates-only arena of
§1–§11). All three runs print the same outputs, bit for bit.

| run | constants | execution pools | result arena (pinned while leased) |
|---|---|---|---|
| `--arena` (intermediates only) | not arena-backed | 5,125,120 B + 4,000 B outside | fresh outputs |
| `separate` | 7,949,024 B | 5,731,232 B | 4,000 B |
| `shared` | 7,949,024 B | 5,129,120 B | 5,129,120 B |

The constant arena holds 7,948,896 B of payload (262 copies) in 7,949,024 B:
the rest is alignment. `separate` adds the 602,112 B input copy and the
outputs as arenas of their own; `shared` absorbs both into the same pool,
4,000 B above the intermediates-only one. These are pool sizes, not a
working set: peak RSS was the same (~57 MB) in all three runs, and nothing here
measures what a copied-in constant saves against the archive's own payload.
