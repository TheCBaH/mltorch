# Making Native4D accept SDPA and Batched_matmul — analysis and plan

Working note.  Grounded against `unbind-int` at `e37ff39`.

**Which "Native".**  `Attention.Sdpa` and `Matmul.Batched_matmul` are both
fully wired in `lib/native`: `graph_ir`, `graph_builder`, `graph_shape`,
`eval_op`, `shape_error`, `transform/output_transfer`, `lib/native_interp`'s
lowering and `lib/native_op_walk`'s fuzz walks all carry them.  The only
dialect that refuses either is **Native4D**, so that is what this note is
about.

**Why both ops are in one note.**  They are rejected by the same argument.
`Domain.check_node`'s `Batched_matmul` arm cites `Sdpa`'s arm as its
precedent, and `Error`'s `` `Batched_matmul_batch_axis `` message cites
`attention_design.md` §9 by name.  That precedent does not hold up, and it has
already propagated once — which is the strongest reason to fix it at both
sites rather than one.  `Batched_matmul` turns out to be the **cheaper and
higher-value** of the two; §5 says why.

SDPA's route depends on Stage A of `region-sdpa-computation-plan.md`.
`Batched_matmul`'s route depends on nothing.

## 1. What actually rejects them today

| site | Sdpa | Batched_matmul |
| --- | --- | --- |
| `lib/native4d/domain.ml` | `:264`, unconditional | `:237`, unconditional |
| `lib/native4d/lower_engine.ml:908-910` | in the `Unsupported_op` group | in the same group |
| `lib/native4d/error.ml` | `` `Sdpa_batch_axis `` `:18,:68` | `` `Batched_matmul_batch_axis `` `:10,:37` |
| `lib/model_explorer_export/me_classify.ml` | `:33` `Outside_dialect_domain` | `:31`, same |

Tracked rationale: `.ai/native4d_design.md` §7.9 for SDPA;
`.ai/matmul_softmax_design.md` §5's last bullet for `Batched_matmul`, which
says "**no work at all beyond a typed rejection arm** — same argument
`attention_design.md` §9 already proved for Sdpa (`D` has no `Axis4` name)".

## 2. The shared premise, and why it does not hold

### 2.1 `D` is not the operative constraint — `Shape4` already handles it

`Axis4.of_axis Axis.D = None` (`lib/native4d/axis4.ml:25-30`) is true, and it
is why the dialect cannot let an op *name* `D` in a parameter.  But neither op
takes an axis parameter:

- `Attention.Sdpa.params` is `{ scale }` (`lib/native/ops/attention.ml:95`) —
  no axis field at all.
- `Matmul.Batched_matmul.t` is `{ input; mat2 }`
  (`lib/native/ops/matmul.ml:74`) — no params at all.

Neither `Compute` reads a `D` coordinate as a decision; both carry `out`
through and override only the axes they contract or index
(`attention.ml:426-448`, `matmul.ml:128-135`).  `D` is named in each op's
*frame documentation*, exactly as `Bmm`'s batch axis `H` is — and `Bmm` is
admitted at batch 1 and rejected above it (`check_bmm`, `domain.ml:119-127`).

What the dialect actually requires of `D` is that it be **1**, and
`Domain.check_shapes` (`domain.ml:49-59`) already requires exactly that of
every live tensor:

```ocaml
let unit_axis axis = Dim.to_int (Vec6.get shape axis) = 1 in
if unit_axis Axis.T && unit_axis Axis.D then Err.return ()
else Err.fail (`Non_four_dimensional_tensor (id, shape))
```

**Ordering note, stated correctly:** node predicates run *first*, then
`check_shapes` (`domain.ml:350-354`).  So it is not that a `D > 1` node "never
reaches" the arm — it reaches it first.  The load-bearing fact is the other
direction: **if these two arms returned `Err.return ()`, a `D > 1` graph would
still be rejected**, by `check_shapes`, because `D > 1` makes every operand
non-four-axis.  The arms are not what keeps `D > 1` out.  What they currently
do is reject the `D = 1` case as well — the representable one, which §4.3
calls a harmless use the converter should normalize away rather than refuse.

`domain.ml:350-357`'s own comment already frames this: the two checks overlap,
and the op-level diagnostic is preferred because it names the node to fix
rather than its consequence.  That argues for making these arms *conditional*
on `D`, exactly like `check_bmm`'s batch test — not for deleting them.  See
§3 step 5 and §4 step 4.

### 2.2 "No legalization is available" assumed legalization means decomposition

Both rationales say the target ops do not exist in `Ops4` — no `Bmm`, no
softmax (`Softmax4` has since landed: `ops4.ml:221`, `lower_engine.ml:895-899`,
and `domain.ml:230-235`'s own comment already acknowledges it).

But `.ai/native4d_add_op.md` site 4 records a route neither rationale
considered: a **retained** dialect operation whose computation is Native's
own — "Translate parameters and call the Native operation's natural
computation form… Native4D adds no numeric kernels."  Eleven of the nineteen
registry entries already reuse a Native payload unchanged.  No decomposition,
no `Ops4` payload, no second numeric definition.

So the correct reading is narrower than either tracked rationale:

> `D > 1` is outside the dialect and `check_shapes` says so.  `D = 1` is a
> **missing dialect registry entry**, not an impossibility.

## 3. `Batched_matmul` — do this one first

### 3.1 Its batch axes are dialect axes

```text
input [N, T, D, H, W=rows,     C=contract]
mat2  [N, T, D, H, W=contract, C=cols]
out   [N, T, D, H, W=rows,     C=cols]
```

`output_shape` (`matmul.ml:108-121`) requires `N`/`T`/`D`/`H` to agree on both
operands — so the batch is spread across **all four** leading axes, not "D and
H" as `domain.ml:230-236`'s comment and `error.ml:37-41`'s message both say.
With `T = D = 1` forced by `check_shapes`, the surviving batch axes are `N` and
`H`, **both of which `Axis4` names**.  There is no batch restriction to impose
at all: any `N`, any `H`.

This makes `error.ml:37-41`'s message actively misleading — it names `D` as
the fault, and `D` is necessarily 1 at the point the message is produced.

### 3.2 It has real corpus coverage

`.ai/matmul_softmax_design.md` §2c: `mvitv2_tiny` is real batched multi-head
attention, 20 `matmul.default` and 10 `softmax.int`, and **every sample is
`D = 1` with only `H` varying (1/2/4/8)**:

```text
[1,1,3136,196] x [1,1,196,96]  -> [1,1,3136,96]
[1,2,784,196]  x [1,2,196,96]  -> [1,2,784,96]
[1,4,196,196]  x [1,4,196,96]  -> [1,4,196,96]
[1,8,49,196]   x [1,8,196,96]  -> [1,8,49,96]
```

Every real occurrence in the corpus is in the four-axis representable set.
`Softmax4` already exists, so `Batched_matmul` is the remaining blocker for
this model's attention blocks.

**Check before claiming model movement.**  `.ai/pt2_model_support.md:159-162`
records `mvitv2_tiny` as `"unsupported_operator": matmul.default` — but that
note predates §5's landing (2026-08-29), which added both importer arms.
Re-run `make pt2.json-model-support` and read `native_builds` /
`native4d_converts` for `mvitv2_tiny` before writing any coverage claim into a
tracked doc.  If it now builds natively, this rejection is what keeps it out
of the Native4D domain, and that is a real, first, model-level payoff — unlike
SDPA, which has none (§6).

### 3.3 Plan

Following `.ai/native4d_add_op.md`'s site list:

1. **Payload — none.**  Site 1's rule: write an `Ops4` payload only if the op
   names axes, carries a shape, or has a grouping/mode.  `Batched_matmul` has
   no params at all.  Reuse `Matmul.Batched_matmul` unchanged, as
   `Pointwise.Add` and `Pool.MaxPool2d` are reused.
2. **Variant + registry** — `lib/native4d/op.ml`, one constructor
   `Batched_matmul of Matmul.Batched_matmul.t`, alphabetically (after
   `Batch_norm_no_stats`, before `Clamp` — the file orders by ASCII, which is
   why `Add_scalar` precedes `Adaptive_avg_pool2d`), plus the registry entry.
   That single entry wires
   JSON, `operands`, `map_operands` and `pp`.  No `4` suffix: the suffix marks
   an `Ops4` payload, and there is none.
3. **Shape** — `graph_shape4.ml`, delegate to
   `Matmul.Batched_matmul.output_shape` and wrap through `four`.  The
   `N`/`T`/`D`/`H` agreement and contraction check come along; do not restate
   them.
4. **Computation** — Pixel-authored, so site 4's first case applies: the
   existing `Matmul.Batched_matmul.Compute (S)` functor, one arm in
   `Eval_op4.pixel`.  **No Region work, no dependency on
   `region-sdpa-computation-plan.md`.**
5. **Domain** — `domain.ml:237`.  Replace the unconditional failure with a
   `check_bmm`-shaped conditional that fires only when `D > 1`, so the
   diagnostic still names the node rather than leaving `check_shapes` to name
   the tensor (`domain.ml:350-357`'s stated preference).  Reword the comment
   at `:230-236`: the batch axes are `N`/`T`/`D`/`H`, not "D and H", and the
   trailing "Native4D has no Bmm and no softmax" is stale on both counts.
6. **Claim transfer** — `output_transfer4.ml`, `Continuous`, mirroring
   Native's answer at `output_transfer.ml:112-120`.
7. **Builder** — `builder.ml`, alphabetical, `Shape4.t` in the signature.
8. **Legalization** — `lower_engine.ml`, a `simple` arm with no parameter
   translation at all (simpler than `Softmax4`'s, which converts an axis).
   Remove `Batched_matmul` from the `Unsupported_op` group at `:908-910`.
   Claim `Identical`: same op, same `Compute`, same coordinates.
9. **Error rows** — delete `` `Batched_matmul_batch_axis `` if step 5 uses a
   differently-named conditional row, or reword it to say what it now means
   (`D > 1`, not "the batch axis is D").  Do not leave a message that names an
   axis which is necessarily 1.

### 3.4 The `Bmm` consequence, worth stating and not folding in

`Bmm`'s frame is `[H=batch, W=rows, C=contract]` with `N`/`T`/`D` at extent 1
(`matmul.ml:1-6`) — **exactly** `Batched_matmul`'s frame at `N=T=D=1`.
`matmul.ml:63-72` states the only difference: `Bmm.Compute` hard-codes
`mat2`'s `N`/`T`/`D` at index 0, "correct only because the batch-less importer
restriction keeps them at extent 1", where `Batched_matmul.Compute` reads them
off the output coordinate.  At extent 1 both resolve to the same element, and
the reduction, its bounds, and the operand order of the `mul` are identical.

So once `Batched_matmul` is in the dialect, `Bmm` can lower to it **at any
batch extent**, claiming `Identical` — `Identical` is a *value* claim
(§9.2: "bit-for-bit equal at every corresponding coordinate"), not a
structural one, so `Index.zero` versus `Index.output N` at extent 1 does not
weaken it.  Two consequences:

- `` `Unsupported_bmm_batch `` (`domain.ml:119-127`) would have no producer:
  the `batch > 1` case §7.4 rejects becomes representable.
- `` `Lossy_bmm_operand `` would too.  It exists *only* because the conv
  legalization inserts a `Permute4` that materializes `mat2` in f32
  (`domain.ml:106-118`); a `Batched_matmul` route reads `mat2` directly, as
  Native's `Bmm` does, so the extra rounding it guards against does not occur.

**Do not fold this into the same change.**  The conv legalization may well be
faster at batch 1, and retiring or demoting it is a measurement question, not
a soundness one.  The honest statement is: after step 8, the 1x1-conv
legalization becomes a *performance* choice rather than the only sound route,
and its two rejection rows become removable.  Land that as its own change with
its own before/after numbers.

## 4. `Sdpa`

### 4.1 The mapping at `D = 1`

A rank-4 ATen tensor right-aligns onto the innermost four frame axes
(`lib/native/aten_shape.mli:1-4`), so Q/K/V land at `D,H,W,C`:

```text
query [D=batch=1, H=heads, W=Wq, C=E]
key   [D=1,       H=heads, W=Wk, C=E]
value [D=1,       H=heads, W=Wk, C=E]
out   [D=1,       H=heads, W=Wq, C=E]
```

With `D = T = 1` this is a legal `Shape4.t`.  `H` carrying heads rather than a
spatial extent is fine — `Axis4` names frame positions, not semantics, and
`Softmax4` already takes an arbitrary `Axis4.t`.

**No relayout, and none should be introduced.**
`.ai/attention_design.md` §2's "the axes the op reads are exactly the axes the
data lands on under right-alignment" is what keeps the ATen flash oracle
valid; a `D -> N` permute to put batch on `N` would move the op off that
argument and require re-running the backend probe and re-arguing §10's single
`atol_for_target` entry.  See §5 for what `batch > 1` would actually cost.

Unlike `Batched_matmul`, SDPA's batch really is `D` alone (heads are on `H`),
so `D = 1` is a genuine restriction here, not an artifact of the dialect.

### 4.2 Plan

Prerequisite: **Stage A of `region-sdpa-computation-plan.md`.**  Site 4's
second case requires a Region-authored op to route Direct and Symbolic through
the shared `Region_program` and forbids scalarizing it through
`Eval_op4.pixel` for production.  Without Stage A there is no program to
delegate to.

1. **Payload — none.**  `{ scale }` names no axis and carries no shape; reuse
   `Attention.Sdpa` unchanged.
2. **Variant + registry** — `Op.Sdpa of Attention.Sdpa.t`, alphabetically
   (between `Rsub_scalar` and `Select4`).
3. **Shape** — delegate to `Attention.Sdpa.output_shape` through `four`.  The
   eight-factor work bound and the mask broadcast check come along for free.
4. **Computation** — `region_computation4.ml`'s `native_op` gains
   `| Op.Sdpa t -> Some (Graph_ir.Sdpa t)`.  Because the payload is reused
   verbatim there is *nothing to translate* — simpler than `Softmax4`,
   `Rms_norm` and `Layer_norm`, whose `Axis4.t` params `graph_shape4.ml:177-195`
   must convert.  `is_region_authored` is derived from `native_op` ("one list
   to maintain, not two that can drift"), so nothing else changes.
5. **Domain** — `domain.ml:264`, conditional on `D > 1`, same shape as §3
   step 5.
6. **Claim transfer** — `Continuous`, mirroring `output_transfer.ml:118`.
7. **Builder**, **lowering arm**, **error row** — as §3 steps 7-9.

## 5. `batch > 1`, and why it is a separate decision

For SDPA at `D > 1` the tensor is not four-axis.  Making it work needs either

- **a `D -> N` relayout** at the importer — contradicts
  `.ai/attention_design.md` §2, changes which axes the op reads, requires
  re-running the flash-oracle probe, and breaks the "one numeric definition"
  property §4.2 depends on, since Native and Native4D would no longer share an
  axis contract; or
- **a fifth dialect axis**, which is not Native4D.

Neither belongs here.

For `Batched_matmul` there is no such question: `D > 1` is genuinely outside
the four-axis frame, but `D` is not where its batch has to live — `N` and `H`
already carry it, and the corpus only ever uses `H` (§3.2).

Note the three-way asymmetry, which the current diagnostics blur:

| op | why `batch > 1` is refused |
| --- | --- |
| `Bmm` | the **destination op** cannot express it (conv weights are shared across spatial positions) — and §3.4 removes this |
| `Batched_matmul` | nothing; `N`/`H` are dialect axes and carry the batch |
| `Sdpa` | the **shape** is not four-axis, because its batch axis is `D` |

## 6. What this does not deliver

- **SDPA: no model coverage.**  `.ai/attention_design.md` §1 — the target
  occurs in no reachable graph across all 23 core-ATen graphs and all 13
  downloaded archives; every exporter decomposes it.  Every fixture is
  hand-built, and `make pt2.runtest` / `make inference` /
  `make jsoo.pt2.runtest` will show no diff.  That must not be reported as
  model progress.
- **`Batched_matmul`: possible model coverage, unverified.**  See §3.2's
  check-before-claiming note.
- **No performance claim for SDPA beyond Stage A's.**  Native4D inherits
  exactly the Region program Native has.

## 7. Verification

The strong property for SDPA: delegating to the same Region program makes
Native and Native4D bit-identical by construction — the `Identical` claim
§9.2 wants, with no second numeric definition to keep in sync.  For
`Batched_matmul` the same holds through the shared `Compute` functor.

Per `.ai/native4d_add_op.md`'s test list:

- `test/native4d/op_json_test.ml` — a sample each; the count check against the
  registry fails until both are sampled.
- `test/native4d/compute_test.ml` — Direct values **hand-computed**, not
  Native-as-oracle (both sides instantiate the same functor, so agreement
  would prove staging, not arithmetic).  Include a `H = 2` `Batched_matmul`
  fixture — `test/native/linear_test.ml`'s `D=2` fixture exists specifically
  to catch a regression to `Bmm`'s `N=T=D=0` hard-code, and the Native4D side
  needs the `H`-varying analogue.
- Direct vs grounded Symbolic, bitwise, in `compute_test.ml`;
  `Fixtures4.per_op` is count-checked too.
- `Map_verify` cross-dialect runs.
- **Prove the new admission is not vacuous** (CLAUDE.md's rule): build a
  `D > 1` graph of each and watch it still be rejected; build a `D = 1` graph
  and watch it convert.  Revert the conditional and watch the `D > 1` test go
  red.
- Model Explorer: `stage:native4d` becomes *available* for these graphs.  Any
  pinned fixture asserting `unavailable outside_dialect_domain` must change,
  and that change is the observable evidence.
- ATen oracles (`Recipe_sdpa`, `Recipe_matmul`) are untouched — Native4D has
  no ATen oracle of its own; its claim is against Native.

## 8. Tracked-record delta

Land these in the same change as the code, per CLAUDE.md:

- **`.ai/native4d_design.md` §7.9** — both premises change.  Premise 1 narrows
  to "`batch > 1` is not four-axis, and `check_shapes` rejects it", dropping
  the "no name for `D` at extent 1" argument, which conflates `Axis4` having
  no constructor for `D` with a `D = 1` tensor being inadmissible, when
  `Shape4.of_vec6` exists precisely to admit it.  Premise 2 is superseded by
  registry reuse / Region delegation, which §6, §7.7 and §7.7a already
  describe for the landed ops.
- **`.ai/native4d_design.md` §7.4** — its `Bmm`/`Sdpa` contrast reads as though
  the difference were `H` versus `D`, rather than "has a destination op"
  versus "does not".  If §3.4 lands, §7.4's `batch > 1` rejection goes too.
- **`.ai/matmul_softmax_design.md` §5**, last bullet — "no work at all beyond a
  typed rejection arm" and its citation of `attention_design.md` §9 are the
  propagated precedent.  Replace both.
- **`.ai/native4d_add_op.md`** — if the domain contract table gains rows, per
  its own site 7.

## 9. Noted, not in scope

`Sdpa` is classified `Continuous` in `output_transfer.ml:118`, but its
emitter is `select (-inf < m) numer 0` — a `Select` on a comparison, which is
the shape the table calls `Discontinuous` elsewhere.  Whether the
`_safe_softmax` guard's branches agree at the boundary is a question about
**Native's** table, not about this port; Native4D should mirror whatever
Native answers.  Flagged because both ops route through the same classifier
and someone will notice it while editing `output_transfer4.ml`.
