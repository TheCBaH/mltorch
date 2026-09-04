# Native4D SDPA/Batched_matmul compatibility — implementation tracker

Live execution record for
[`native4d-sdpa-compatibility-plan.md`](native4d-sdpa-compatibility-plan.md).
Both changes it names (`Batched_matmul`, `Sdpa`) are done; §3.4's Bmm
consequence is now done too (own session, below). §5's `batch > 1` question
is still explicitly out of scope per the plan and is not started here.

## §3.4: Bmm now legalizes directly to Batched_matmul — done

`Bmm` converts to `Batched_matmul` unconditionally (any batch `H`, one node,
no relayout), retiring the old 1x1-convolution route (`Permute4` + `Conv2d`,
sound only at `batch = 1`). Changes:

- `lib/native4d/lower_engine.ml`: the `Bmm` arm merged with `Batched_matmul`'s
  (`simple (Op.Batched_matmul {...})`, one arm, two patterns) — same two-field
  payload, same node.
- `lib/native4d/domain.ml`: `check_bmm` removed entirely; `Bmm` moved into the
  unconditional "direct counterpart" arm. Its two preconditions
  (`` `Unsupported_bmm_batch ``, the batch check; `` `Lossy_bmm_operand ``, the
  mat2-format check guarding the old materialization) have no producer any
  more — both variants removed from `error.ml`/`.mli` and `me_classify.ml`
  ("nine domain rejections" -> "seven").
- Tests updated across `domain_test.ml`, `lower_test.ml`, `verify_test.ml`,
  `me_classify_test.ml`, `fixtures.ml` (`bmm_lossy_operand` deleted, now
  unused). `verify_test.ml`'s bmm cluster verdicts strengthen from `2 proved +
  1 tested + 1 vacuous` to `3 proved (structural)` — the old materialization
  was the only thing keeping one cluster off `Identical`'s strongest form.
  The historical "why the bmm legalization needed a verifier fix" narrative
  (a real `Ground_eval`/`value_tiers` defect the old materialized-weight shape
  once tripped) is now dead ground — nothing in the current lowering set
  produces a `Permute4`-into-`Conv2d`-weight shape any more — so its
  `"verify: bmm, the two terms"` test (which had nothing left to show once
  both sides ground to the same term) was removed; the fix itself is
  unaffected and undocumented nowhere else changes.
- `.ai/native4d_design.md` §7.4 rewritten (new route, why it's `Identical`,
  the retired route and why it needed two preconditions); §7.9's stale
  `check_bmm` cross-reference fixed; `.ai/matmul_softmax_design.md`'s
  `Batched_matmul`-landing paragraph's stale `check_bmm`/`Unsupported_bmm_batch`
  parenthetical fixed.
- `make pt2.json-model-support` re-run: byte-identical output, zero model
  movement. Expected and stated as such — real corpus `Bmm` usage is already
  always batch=1/f32 by construction (the importer routes any genuinely
  batched matmul to `Batched_matmul` instead, per
  `.ai/matmul_softmax_design.md`'s importer-split note), so both of `Bmm`'s
  now-removed preconditions were already vacuous on every real graph. This is
  a soundness/performance change, not a coverage one — matches the plan's own
  framing (§3.4: "a separate, measured change").
- Before/after performance, the plan's own ask ("land that as its own change
  with its own before/after numbers"): at `batch = 1`, the new one-node route
  is 2.9x-4.3x faster than the retired two-node one across `extent` 8/32/128
  (measured with a temporary, uncommitted bench addition since the old route
  no longer exists to compare against; numbers folded into
  `.ai/native4d_design.md` §7.4). Expected, not surprising: one node instead
  of two, no materialized intermediate tensor.
- `make precommit` green throughout.

## Scope lock

- Included: `Batched_matmul` (§3) — registry entry reusing
  `Matmul.Batched_matmul` unchanged, `Graph_shape4`/`Eval_op4` arms,
  `Output_transfer4.Continuous`, `Builder.batched_matmul`, a `lower_engine.ml`
  `simple` legalization arm, `Domain.check_node`'s arm made conditional on
  `D > 1` (was unconditional), `` `Batched_matmul_batch_axis ``'s message
  reworded to match.
- Included: `Sdpa` (§4) — same site list, plus `Region_computation4.native_op`
  gaining `Op.Sdpa -> Graph_ir.Sdpa` (payload reused verbatim, so Direct and
  Symbolic route through the identical `Region_program` Native uses — no
  second numeric kernel) and `Eval_direct4`'s synthetic-role list gaining
  `Sdpa_mask`, mirroring Native's own `eval_direct.ml`.
- Excluded (plan's own scope cuts): §3.4's `Bmm`-lowers-to-`Batched_matmul`
  consequence (a separate, measured change); §5's `batch > 1` question for
  either op (needs a relayout or a fifth dialect axis, neither in scope).
- §3.2's "check before claiming" is now done (`6899974`, follow-up session):
  `make pt2.json-model-support` is payload-free (no download needed) and
  moves exactly one model, `efficientvit_b0` (`native4d_converts` false ->
  true, its blocker having named `Batched_matmul`'s rejected batch axis
  verbatim). `mvitv2_tiny` and `fastvit_sa12`, the plan's two named examples,
  do NOT move — neither ever reached the node that changed (see
  `.ai/pt2_model_support.md`'s "Updated 2026-09-03" entry for the full
  finding). The real, downloaded-archive-gated sweep
  (`test/pt2_model_support_cram.t`) is still not re-run; nothing in the six
  models it covers uses either op.

## Progress

| Step | Scope | Status |
|---|---|---|
| 1 | `Op.Batched_matmul`/`Op.Sdpa` variants + registry (`op.ml`) | complete |
| 2 | `Graph_shape4` shape arms (delegate to Native's `output_shape`) | complete |
| 3 | `Eval_op4.pixel`: real `Batched_matmul` arm; `Sdpa` `invalid_arg` arm | complete |
| 4 | `Region_computation4.native_op` gains `Sdpa`; `Eval_direct4` gains `Sdpa_mask` | complete |
| 5 | `Domain.check_node`: both arms made conditional on `D > 1` | complete |
| 6 | `Error.pp`: both messages reworded for the conditional | complete |
| 7 | `Output_transfer4.classify`: both `Continuous` | complete |
| 8 | `Builder.batched_matmul`/`Builder.sdpa` | complete |
| 9 | `lower_engine.ml`: real conversion arms, removed from the `Unsupported_op` catch-all | complete |
| 10 | Tests: `op_json_test.ml` samples, `Fixtures4.per_op` fixtures, hand-computed Direct values, `region_mapping_test.ml` Sdpa structural check, `Fixtures.batched_matmul`/`Fixtures.sdpa` parametrized by batch + `domain_test.ml` batch=1/batch=2 vacuity proof, `me_group8_cram.t` re-promoted | complete |
| 11 | `.ai/` delta: `native4d_design.md` §7.4/§7.9, `matmul_softmax_design.md` §5, `region_compute_design.md`, `pt2_model_support.md` (staleness notes, not new claims) | complete |

## What was built

`Batched_matmul` and `Sdpa` both follow the "reuse the Native payload
unchanged, translate nothing" route `.ai/native4d_add_op.md` site 1
describes: neither op names an axis or carries a shape, so there is no
`Ops4` payload and no `4`-suffixed variant — `Op.Batched_matmul of
Matmul.Batched_matmul.t` and `Op.Sdpa of Attention.Sdpa.t` hold Native's own
record types directly.

The shared premise that used to reject both unconditionally (`D` has no
`Axis4` name, so nothing at any `D` extent converts) conflated a `D > 1`
tensor (genuinely outside the dialect) with a `D = 1` one, which
`Shape4.of_vec6` already admits like any other axis at unit extent. Fixing
this took two different shapes, because the two ops' real batch axes differ:

- `Batched_matmul`'s batch is `N`/`T`/`D`/`H` together (`output_shape`
  requires all four to agree between `input` and `mat2`), so with `T = D = 1`
  forced by `check_shapes`, `N` and `H` — both dialect axes — already carry
  the real batch. `D = 1` is the *only* restriction, and once satisfied there
  is nothing left to check; `lower_engine.ml`'s new arm is a `simple`
  pass-through, no different from any other direct counterpart.
- `Sdpa`'s batch is `D` alone (heads are on `H`), so `D = 1` is a genuine,
  narrow admission — not a slice of a wider representable set the way
  `Batched_matmul`'s is. It needed Stage A of
  `region-sdpa-computation-plan.md` first (already landed, `7ffba06`):
  without a Region-authored `Attention.Sdpa.Computation.program` to delegate
  to, there was no route that avoided a second numeric kernel.
  `Region_computation4.native_op`'s new arm (`Op.Sdpa t -> Some
  (Graph_ir.Sdpa t)`) is the whole of the wiring — the payload needs no
  translation, so Native4D adds no arithmetic of its own, and Native/Native4D
  are bit-identical by construction rather than by a numeric comparison.

`Domain.check_node`'s two arms went from unconditional `Err.fail` to a
`check_bmm`-shaped conditional (`check_batched_matmul`/`check_sdpa`, each
reading one operand's `D` extent), and the two error messages were reworded
to say `D > 1`, not "D, which the dialect has no name for" — the old wording
was true of the rejected case and false of the newly-admitted one.

## Verification

- **Vacuity proof** (CLAUDE.md's rule for a widened admission): both
  `test/native4d/fixtures.ml`'s `batched_matmul`/`sdpa` fixtures are now
  parametrized by the `D` extent, mirroring `bmm_batch`'s existing shape.
  `domain_test.ml`'s two new expect tests pin `batch=1` admitted and
  `batch=2` still rejected with the reworded message, in the same table —
  the D=1 admission and the D>1 rejection are proven side by side, not just
  asserted separately.
- **`test/me_group8_cram.t`** (the one hand-built SDPA graph Model Explorer
  has, since SDPA occurs in no reachable corpus graph — every exporter
  decomposes it) already builds its fixture at `D = 1`. Re-promoting it
  turned `stage:native4d unavailable outside_dialect_domain` into `stage:
  native4d available graph` with no diagnostic left to print — real,
  observable evidence the admission fires end to end through Model Explorer,
  not just through `Domain.check` in isolation. The surrounding prose was
  rewritten to match; the old prose's "every stage available EXCEPT
  Native4D" framing was the row's entire point and is no longer true.
- **`test/native4d/op_json_test.ml`**: both ops sampled (57/57 against the
  registry), printed and round-tripped through JSON.
- **`test/native4d/fixtures4.ml` / `compute_test.ml`**: both ops have a
  `Fixtures4.per_op` entry (57/57 against the registry) with Direct-vs-
  Symbolic bitwise agreement; `batched_matmul` exercises `H=2` with two
  distinct per-head answers (`[17 53]`, catching a `Bmm`-style hard-coded
  batch index); `sdpa`'s fixture has a broadcasting mask (`H`/`W` both 1)
  to exercise `Region_context.broadcast_coord`. Both also have a
  **hand-computed** Direct test (not Native-as-oracle, since both dialects
  share `Compute`/`Region_program` — hand values are the only thing that
  tests the arithmetic rather than self-consistency): `batched_matmul`'s
  is a from-scratch dot product; `sdpa`'s reuses `softmax4`'s own `[0, ln 3]`
  setup (`E=1`, default scale, so the score reduces to `q*k` exactly) with
  no mask, deliberately exercising `Sdpa_mask`'s synthetic-fill path.
- **`test/native4d/region_mapping_test.ml`**: a new `Sdpa` case proves
  `Region_computation4.program` and `Region_computation.program` build
  structurally identical Region programs (same locals, same output
  expression, same partition) for the same op value — the strong form of
  "Native4D adds no numeric kernel", checked directly rather than inferred
  from the `native_op` arm reading trivially.
- `make precommit` (build, format, full `dune runtest` with the JSON/expect
  promotions above applied, file-size, whitespace) green throughout.

### The file-size cap

`test/native4d/compute_test.ml` crossed the 1000-line tree cap while adding
the two hand-computed tests. Rather than add an exception-list entry, the two
new tests were tightened (shared `axis_int`/`chan`/local `step`/`mk`
per-coordinate helpers instead of one-off `Dim.to_int (Vec6.get c Axis.X)`
inline in each closure) until the file landed at exactly 1000 lines. No
existing test's content changed.

## Status: both changes complete

Nothing further is planned under this plan document. §3.4 (retiring the 1x1
conv legalization once `Batched_matmul` exists) and §5 (`batch > 1` for
either op) are the plan's own named follow-ons, each requiring its own
measurement or design decision — not started here.
