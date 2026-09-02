# Region computation — post-review action plan

## Purpose and boundary

Working plan for the defects and cleanups found reviewing `origin/main..HEAD`
on `unbind-int` (25 commits, +5017/-294).  Scope is limited to that branch's
Region-computation work plus the design-record migration it left outstanding.

Out of scope: the deferred follow-ons already listed in the operation-computation
implementation plan (Pixel-to-Region discovery, Loop IR, tiling, shaped locals,
multi-output programs, GroupNorm/batch-norm Region forms, safe/log softmax,
SDPA, relaxed rounding, multi-node fusion).

Each unit lands as one `fixup!` commit naming the commit that introduced the
defect, grouped by defect and not by file.  Never amend an existing commit; a
second `fixup!` onto the same target is the right move.

Baseline at review time: `dune build`, `dune runtest`, `dune build @fmt`,
`git diff --check`, `make check.file-size` all green, and
`bin/region_compute_bench.exe` reproduces the recorded table within noise.
Every unit below must leave that true.

---

## F1 — Hoist `fresh_synthetic_ids` out of the per-node loop

**Severity: high.  Fixup target: `e6eb7ff` (Native) and `2406e1d` (Native4D);
one commit, since it is one defect spanning two files.**

### Defect

`lib/native/eval_direct.ml:215` and `lib/native4d/eval_direct4.ml:197` compute
synthetic operand ids unconditionally for every node, folding the whole
`g.Graph.tensors` map into a fresh `Tensor_id.Set` each time.  The result
depends only on `g`, and is needed only by the three Region-authored
operations.  Evaluation is quadratic in graph size for graphs containing no
Region op at all.

### Work

- Compute the ids once in `run` and thread them into `eval_node`, or bind
  `lazy (fresh_synthetic_ids g)` and force at the single use site.  Threading
  from `run` is preferred: the value does not change per node, and the `lazy`
  still allocates a thunk per node.
- Apply the same change to both dialects.

### Acceptance

- A scaling regression in `test/native/region_compute_test.ml`: evaluate a
  Region-free chain of N `relu` nodes at two sizes and assert the time ratio is
  linear rather than quadratic.  Nothing in the suite can see this class of
  regression today, which is why it shipped.
- `make precommit` green.

### Evidence to reproduce

800-node `relu` chain through `Eval_direct.run`, 20 iterations:

| nodes | as landed | with the call hoisted |
|---|---|---|
| 50 | 0.412 ms | 0.187 ms |
| 200 | 5.455 ms | 0.704 ms |
| 800 | 109.085 ms | 2.858 ms |

Note why the permanent benchmark missed it: `bin/region_compute_bench.ml:25-30`
builds single-node graphs, so its cost model is per-operation and no per-graph
regression is observable by it.

---

## F2 — Export the whole Region program to Model Explorer

**Severity: medium.  Fixup target: `d549fb3`.**

### Defect

`lib/model_explorer_export/me_kernel.ml:182` and `:217` fall back to
`Region_program.output` — the emitter alone, whose `Local` nodes have no
definitions in scope.  Measured on the landed programs:

| op | exported size | actual | exported `body` |
|---|---|---|---|
| rms | 17 (depth 4) | 43 (depth 5) | `((t0[...] * ?#1) * t2[...])` |
| layer | 27 (depth 6) | 70 (depth 6) | `((((t0[...] - ?#1) * ?#3) * t2[...]) + t3[...])` |
| softmax | 12 (depth 5) | 35 (depth 5) | `(exp((t0[...] - ?#0)) / ?#1)` |

The exported `body` attribute ships dangling `?#N` placeholders and `size`
/`depth` under-report by 2-3x.  Incoming-edge derivation
(`me_kernel.ml:84`, `Expr.Fold.sources body`) is latently wrong for the same
reason; it is correct today only because each of the three emitters already
loads `x`.

`lib/model_explorer_export/me_detail.ml:80-101` already handles this correctly,
emitting one root per local plus the emitter with a `names` function.  The two
exports disagree about the same data.

### Work

Preferred: widen `me_kernel.ml`'s `build ~values` to carry `Region_program.t`
rather than one `Expr.Value.t`, then derive

- edges from `Region_program.Fold.sources` (already used in `kernel_adapt.ml`),
- `depth` from `Region_program.Fold.max_depth`,
- `size` from the summed expression sizes,
- `body` text from `Region_program.pp`, which already names locals `l0`, `l1`, ...

Cheaper alternative: export `Region_program.specialize_pixel`.  Removes the
`?#N` and fixes the counts, but loses the local structure the Region form
exists to show.

### Acceptance

- A cram golden for a value graph containing `layer_norm`.  None exists today;
  that gap is why this shipped.
- Assert no `?#` appears in any exported `body` attribute.
- Existing Model Explorer crams unchanged except where the norm bodies appear.

---

## F3 — Memoize the specialized pixel body per stage

**Severity: medium.  Fixup target: `d549fb3`.**

### Defect

`lib/native/transform/ground_eval.ml:397` calls `Stage.pixel_body`, which runs
`specialize_pixel` — freshen, substitute locals, re-check size and depth — on
every grounded coordinate.  It is called per coordinate per side in
`map_verify_check.ml:382` and again inside `expand`'s recursion at
`ground_eval.ml:470`.  Previously this was a field read.

Measured per call: 6.04 us for a LayerNorm stage, and 0.31 us even for a `relu`
stage — the pixel path now pays a full `Expr.Check.value` traversal that used
to be free.

The result is stage-invariant.

### Work

Cache the specialized body per stage id in `Ground_eval.Env`, which already
holds per-stage state, or compute it once when `Env.of_program` builds.

### Acceptance

- `ground_eval_data_test.ml` and `region_compute_test.ml:225` goldens unchanged.
  This is a pure memoization; any golden movement means the cache is keyed
  wrongly.
- `make precommit` and `make pt2.runtest` green.

---

## F4 — Collapse the duplicated synthetic-operand plumbing

**Severity: medium (duplication, no live defect).  Fixup target: `16c5143`.**

### Defect

`region_result` and `fresh_synthetic_ids` are ~85 near-verbatim lines in both
`lib/native/eval_direct.ml:77-160` and `lib/native4d/eval_direct4.ml:81-165`,
differing only by the op match and `Shape4.to_vec6`.  The claim of "no second
numeric kernel" holds, but this is a second copy of the part that can drift.

Inside each copy the `fill` callback receives `role`, `value` and `shape` and
discards all three, after which `synthetic_shape` re-derives the identical
`(value, Norm_shared.normalized_shape ~x_shape ~dims)` by re-matching on the op.

### Work

Have the caller's `fill` record `(id, value, shape)` into a table it reads back
after `Region_computation.program` returns.  That deletes `synthetic_shape`
from both dialects and removes the drift risk between them.

What remains dialect-specific is `Shape4.to_vec6` and the op match — small
enough to leave duplicated.

### Acceptance

- Direct counter goldens for all three operations unchanged in both dialects.
- `test/native4d/region_mapping_test.ml` unchanged.

---

## F5 — Cleanups

**Severity: low.  Separate `fixup!` commits, grouped by defect.**

| Site | Change | Fixup target |
|---|---|---|
| `lib/native/region_trace.ml:65-95` | Replace the `failure` ref with `Err.Escape.throw esc`; the token is already in scope two lines up and this exits immediately instead of scanning the rest of the domain | `614dfe7` |
| `lib/native/kernel_eval.ml:164-180` | Test with `Region_program.pixel_expression` instead of lowering, discarding, rebuilding and lowering again; the `assert false` then disappears | `d549fb3` |
| `lib/native4d/region_computation4.ml:8-10` | `let is_region_authored op = native_op op <> None` — one line, and the two hand-maintained lists can no longer drift | `16c5143` |
| `lib/native/eval_symbolic.ml:77-86`, `lib/native4d/eval_symbolic4.ml:70-82` | Drop the `let program = ... in program` dead binding; restore the `: Stage_program.t` return annotations that were removed | `d549fb3` |
| `lib/native/ops/norm_shared.ml:38-47` | Use `Region_context.output_coord` instead of re-declaring the same six lines; no dependency reason blocks it | `36c7d78` |
| `test/native4d/region_mapping_test.ml:32` | Compare structurally — `Expr.Value.equal` on locals and output plus partition equality — rather than by `Format.asprintf "%a" Region_program.pp` string equality | `0523a0c` |

---

## D1 — Migrate the settled forward decisions into the tracked record

**Fixup target: `0a19476`.  Do this after F1-F3.**

Two of those findings change statements the tracked record currently makes: its
cost-model section describes per-key ownership that is true of the executor but
not of the per-node overhead around it (F1), and it states that the Model
Explorer consumes the structural program, which is half true today (F2).
Writing the record against fixed code means writing it once.

### What to promote from staging

Roughly one page.  Each earns its place because a future change would silently
violate it:

1. **Locals carry a shape in the IR even though only scalars exist today**
   (`Region_local.Shape`).  This is what makes the deferred shaped-local work
   not a language redesign.  Without this sentence someone simplifying
   `Region_local` deletes it as dead weight.
2. **Region semantics are separate from schedule tiles** — required sharing
   comes from the operation, locality blocking is schedule-selected.  This is
   the reason `Region_partition` has only `Whole` and `Singleton` and no
   `Block`.
3. **Why SDPA needs more than scalar sharing** — three sentences; it is what
   justifies the scope line the record already draws.

### What NOT to promote

The phase roadmap (Loop IR, schedule search, tiled conv as a Region-to-Region
rewrite, whole-graph single-node lowering, stencil acceptance cases): roughly
700 lines with no implementation and no measurement.  The staging design's
adopted-direction section already contains a paragraph that went stale the
moment `dfcb630` fixed `fold_outputs` — the failure mode in miniature.

Promoting them would repeat what the branch already did twice: two proposal
documents totalling 2,124 tracked lines were added by the branch's first commit
and never revisited by the 23 that followed.  Promote a phase design when that
phase has code.

---

## D2 — Residue cleanup in the tracked Region record

**Fixup target: `0a19476`.  Same commit as D1.**

By the stated migration standard — concise, no historical data, no completed
work — three things in the tracked Region computation design fail today:

- **It points at the gitignored staging area** (`.gitignore:1`), so a fresh
  clone cannot follow the reference.  The letter of the CLAUDE.md rule added by
  this branch covers commit messages and code comments, but the rationale is
  identical and this is the reference class the rule exists to remove.
- **The dated benchmark table** ("one 20-sample local run on 2026-09-02") is a
  measurement log.  Keep the reproduction command and the counter table above
  it, which are the durable parts; drop the ms/words rows.
- **"The closeout was validated with:"** — "closeout" and "Gate" are progress
  vocabulary.  Reframe as "Reproduce with:".

Also in the same pass: the kernel-DSL design record's status header still says
`Compute`, the scheduler and the transformation verifier are unchanged and that
nothing routes through the Kernel layer yet.  All three clauses are now false —
`Compute` became `Legacy_pixel` for three operations, grounding routes through
`Stage.pixel_body`, and Direct, Symbolic, Kernel adaptation and the Model
Explorer all consume Region programs.

---

## Sequencing

F1 -> F2 -> F3 -> F4 -> F5 -> D1 + D2.

F1 first: largest measured impact, smallest change.  D1/D2 last, so the record
is written once against fixed code.

## Validation record

| Date | Unit | Command | Result | Notes |
|---|---|---|---|---|
| 2026-09-02 | F1 | `dune runtest test/native` + `make precommit` | green | Hoisted `fresh_synthetic_ids` into `run`/`run_graph` in both dialects; new scaling-regression test in `region_compute_test.ml` confirmed quadratic before the fix (reverted `eval_direct.ml` and reran: reported `quadratic`), linear after. Commit `1ef39e4`. |
| 2026-09-02 | F2 | `dune runtest test/model_explorer` + `make precommit` | green | Widened `Me_kernel.build`'s `~values` to carry `Region_program.t`; added `Region_program.Fold.size`; new `me_kernel_test.ml` golden on a LayerNorm kernel value. Reverted `me_kernel.ml` and reran: reported `no_dangling_locals=false size=27` (matches the finding's numbers), `true`/`70` after the fix. Commit `3ad4cb2`. |
| 2026-09-02 | F3 | `dune runtest` + `make precommit` | green | Cached `Stage.pixel_body` per stage id in `Ground_eval.Env`; `ground_eval_data_test.ml` and `region_compute_test.ml` goldens unchanged, as expected for pure memoization. Commit `8723988`. |
| 2026-09-02 | F4 | `dune runtest` + `make precommit` | green | `fill` now records `(id, value, shape)` into a table both dialects read back, replacing `synthetic_shape`'s re-match on `op`. Direct/region_mapping goldens unchanged. Commit `ba1a888`. |
| 2026-09-02 | F5 | `dune runtest` + `make precommit` | green | Six independent cleanups, one fixup commit each: `region_trace.ml` throws immediately (`7a00be6`); `kernel_eval.ml`/`region_execution.ml{,i}` test `pixel_expression` directly via new `lower_region`, removing the `assert false` (`07eabf0`); `region_computation4.ml`'s `is_region_authored` derives from `native_op` (`71d282f`); dead `let program = ... in program` dropped and `: Stage_program.t` restored in both `eval_symbolic{,4}.ml` (`a8eab9a`); `norm_shared.ml` reuses `Region_context.output_coord` (`89278c1`); `region_mapping_test.ml` compares structurally instead of via `pp` string equality (`27b12f9`). |
| 2026-09-02 | D1 + D2 | `make precommit` | green | Added "Design headroom for deferred work" to `region_compute_design.md` (locals-carry-shape, Region-vs-schedule-tile separation, why SDPA needs shaped locals); dropped the `_ai_/` reference and the dated ms/words table; reframed "closeout" as "Reproduce with"; fixed `native_kernel_dsl_design.md`'s stale status paragraph. Commit `e37ff39`. |

## Findings checked and dismissed

Recorded so they are not re-raised:

- **`on_reduction` in the core evaluator** (`lib/expr_internal/eval.ml:259`)
  looked like benchmark instrumentation on the hottest inner loop.  A/B'd by
  removing the call: 24.3 vs 24.4 ns per reduction step.  No measurable cost.
  It remains public API surface added for a counter, which is a taste question,
  not a defect.
- **`normalized_count_unchecked` on the new Region path.**  Its documented
  precondition holds at these call sites, and `region_computation.ml:23`
  verifies the output contract independently.
- **`fold_outputs` losing its key bounds check.**  Both production callers pair
  it with `fold_keys`, and the precondition is documented at the definition.
- **The exception in `Eval_symbolic.run`** (`Err.raise_error` where
  `Eval_direct` returns a typed error) is asymmetric but not a live bug: Region
  program size is extent-independent (~70 nodes for LayerNorm), so default
  limits cannot be exceeded in practice, and the outward boundaries do match
  `Err.Exn.E` (`js/webapp/webapp_bridge.ml:239`,
  `lib/model_explorer_export/me_limits_diagnostic.ml:202`).
- **`Legacy_pixel` really is test-only.**  Repository-wide references are two
  definitions, two explanatory comments, and test invocations.
