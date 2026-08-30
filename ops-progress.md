# Ops coverage: progress ledger

Newly initialized at this landing — no prior copy of this file was found in
the working tree (`_ai_/` is gitignored, so this ledger does not survive a
fresh clone; see [`todo-ops.md`](todo-ops.md)'s "Progress tracking" section
for what it is supposed to hold going forward). Landing history before this
entry lives in `git log`, not here.

## Stage-qualified corpus scoreboard

Recomputed from `test/data/pt2_json_model_support.jsonl` after
`make pt2.json-model-support`, 2026-08-30, 100 tracked graphs:

| Measure | Result |
| --- | ---: |
| `native_builds:true` | 87 / 100 |
| `native4d_converts:true` | 56 / 100 |
| `kernel_converts:true` | 63 / 100 |
| All three stages succeed | 43 / 100 |

Unchanged by any landing below: `Select4`, `Stack4`, `repeat.default`,
`repeat_interleave.self_int`, and `copy.default` each moved a model's
*frontier*, not any of these four counts.

## Per-operation breadth matrix

| Op | ATen boundary | `Op_bridge` | `Native_interp` | Native Direct/Symbolic | Native4D | Kernel | Representative graph |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `select.int` / `Select` | landed pre-existing (`select.int` decomposed into `Select`) | landed | landed | landed (Native `Split.Select`, pre-existing) | **landed 2026-08-30**: `Select4`, `test/native4d/{op_json,fixtures,fixtures4,domain,lower,compute,verify}_test.ml` | reaches `Snapshot4`/lowering; no corpus graph exercises `Select4` end to end into the kernel path yet (csatv2, the only corpus model whose frontier `Select4` clears, is a graph-only CI target — see below) | `csatv2` (native4d frontier only; not a kernel-path graph) |
| `stack.default` / `Stack` | landed pre-existing (own `Stack` node, not a `Concat` decomposition) | landed | landed | landed (Native `Concat.Stack`, pre-existing) | **landed 2026-08-30**: `Stack4`, `test/native4d/{op_json,domain}_test.ml`, `test/native4d/fixtures{,4}.ml` | reaches `Snapshot4`/lowering; no corpus graph exercises `Stack4` end to end into the kernel path yet (csatv2 and skresnet18, the two corpus models whose frontier `Stack4` clears, stay blocked at a later stage — see below) | `csatv2` (native4d frontier only; not a kernel-path graph) |
| `repeat.default` / `Repeat` | landed 2026-08-30: read via `Aten_tensor.shape` only (no ATen kernel call needed), `test/native_bridge/repeat_test.ml` | landed 2026-08-30: `lib/native_aten_bridge/op_bridge_shape.ml` | landed 2026-08-30: `lib/native_interp/native_interp_lower_shape.ml` | landed 2026-08-30: `lib/native/ops/repeat.ml`, `test/native/{repeat_test,graph_direct_shape_test,graph_symbolic_shape_test}.ml` | none yet — no `Repeat4` counterpart; tracked as backlog, not an intrinsic-axis boundary (see landing note) | not reachable without a Native4D counterpart | `convit_tiny` (native-import frontier only; graph does not reach kernel path from here) |
| `repeat_interleave.self_int` / `RepeatInterleave` | landed 2026-08-30, `dim` required (`dim=None` rejected): same shape-only read, `test/native_bridge/repeat_interleave_test.ml` | landed 2026-08-30: `lib/native_aten_bridge/op_bridge_shape.ml` | landed 2026-08-30: `lib/native_interp/native_interp_lower_shape.ml` | landed 2026-08-30: `lib/native/ops/repeat.ml` (shares the file with `Repeat`), `test/native/{repeat_test,graph_direct_shape_test,graph_symbolic_shape_test}.ml` | none yet — no `RepeatInterleave4` counterpart; same backlog status as `Repeat4` | not reachable without a Native4D counterpart | `convit_tiny` (native-import frontier only; graph does not reach kernel path from here) |
| `copy.default` / `Expand` (direct bind, no new op) | **landed this entry**: `self`'s live shape read only, no ATen kernel call, `test/native_bridge/copy_test.ml` | **landed this entry**: `lib/native_aten_bridge/op_bridge_shape.ml` | **landed this entry**: `lib/native_interp/native_interp_lower_shape.ml` | reuses `Pointwise.Expand`'s own pre-existing Direct/Symbolic — no new Compute, `test/native_interp/copy_test.ml` pins the built graph structure | **full-stack already**: `Expand4` pre-exists, admitted unconditionally by `Domain.check_node` | reaches wherever `Expand4`'s own kernel path already reaches | `convit_tiny` (native-import frontier only; graph does not reach kernel path from here) |

## Landing records

### 2026-08-30 — `copy.default`, bound directly to the existing `Expand` node

Closes `copy.default`'s row in [`todo-ops.md`](todo-ops.md)'s deferred
target-name depth backlog (exposed as ConViT's new first frontier by this
session's `repeat_interleave.self_int` landing).

**What landed**

No new Graph_ir operation. `copy(Tensor self, Tensor src, bool
non_blocking=False) -> Tensor` binds directly to the pre-existing
`Pointwise.Expand` node: real ATen's `copy_`/`copy` fully OVERWRITES `self`
with `src` broadcast to `self`'s shape, so the result depends only on
`src`'s values and `self`'s declared shape/dtype, never on what `self` held
before — exactly `Expand{size=self's shape}` applied to `src`. This is the
same "map onto an existing op after translating parameters" route
`sub.Tensor`'s scalar form takes to `Add_scalar`
(`.ai/native_add_op.md`'s sanctioned case, not a decomposition: one node,
`copy.default`'s own identity not preserved in the graph, same as that
precedent). `non_blocking` is a device-transfer hint, read-and-discarded the
same way `implicit` is in `expand.default`'s own arm.

Verified against the whole 100-model corpus (not assumed): every
`copy.default` occurrence (40 total, ConViT and MViTv2) has IDENTICAL
`self`/`src` shape and dtype — zero broadcasts, zero casts observed. They
are all the functionalized form of `tensor[idx] = value`
(`__setitem__` → `select.int` view + `copy_`, functionalized to `select` +
`copy.default`, with a `select_scatter.default` immediately downstream
writing the computed value back into the base tensor). The landed
translation is nonetheless the fully general broadcast, not an overfit to
the observed equal-shape case — proved in both test suites with a genuine
`self` != `src` shape.

- `lib/native_aten_bridge/op_bridge_shape.ml`: `copy.default` dispatch arm,
  reading `self`'s shape via `Aten_tensor.shape` + `Aten_shape.of_aten`
  (no `-1` convention to resolve, unlike `expand.default`'s own `size`), and
  building `src`'s Native operand via `native_tensor_arg`. `self`'s own
  Native operand is never built — its VALUE is provably irrelevant.
- `lib/native_interp/native_interp_lower_shape.ml`: the matching
  payload-free arm — `self`'s declared shape is already a right-aligned
  `Vec6.shape` via `tensor_shape`, needing no rank/`-1` resolution at all.
- No changes to `lib/native/graph_ir.ml`, `graph_shape.ml`, `eval_op.ml`,
  `output_transfer.ml`, `graph_builder.ml`, or `lib/native4d/{domain,
  lower_engine}.ml` — binding to an EXISTING op touches none of the
  exhaustive-match sites a new `Graph_ir` constructor would force. Native4D
  is consequently full-stack already: `Expand4` (landed earlier this
  session's predecessors) is admitted unconditionally by
  `Domain.check_node`.
- Tests: `test/native_bridge/copy_test.ml` (**new file**, ATen-linked
  `Op_bridge.dispatch`, hand-derived values: equal-shape identity,
  `non_blocking` ignored, a genuine broadcast, and the non-broadcastable
  rejection reusing `Pointwise.Expand.output_shape`'s own check),
  `test/native_interp/copy_test.ml` (**new file**, payload-free, asserts the
  built graph structure — an `Expand` node, not a `Copy` node — mirroring
  `expand_test.ml`'s own `dump` pattern).

**Classification**: Full-stack across every surface in
[`todo-ops.md`](todo-ops.md)'s completion-rule table, Native4D included,
since the landing reuses an already-full-stack op rather than adding a new
one.

**Corpus effect**: `convit_tiny` — the one corpus model actually gated on
`copy.default` as its first frontier (MViTv2 also has 10 occurrences, but
they sit behind its already-earlier `_to_copy.default` blocker, so its
frontier is unchanged) — moves to `select_scatter.default`: the predicted
write-back node immediately downstream of each `copy.default`, 30
occurrences, same "Factories, indexing, and copies" deferred-backlog
family. `native_builds`/`native4d_converts`/`kernel_converts`/
all-three-stages are unchanged (87/56/63/43) — depth moved, stage-completion
did not.

**Next**: `select_scatter.default` (ConViT's new frontier) is the natural
next target, being the write-back counterpart of `select`/`copy` just
landed. `_to_copy.default` remains the 2-model target (EdgeNeXt, MViTv2)
needing the value-domain trunc/round `SEMANTICS` primitive flagged earlier
this session. Otherwise: `lstm.input` from the deferred backlog, or P1's
remaining one-model slices (`adaptive_max_pool2d`, `conv1d`/`unfold`,
`im2col`/`col2im`, `upsample_bicubic2d`).

### 2026-08-30 — `repeat_interleave.self_int`, the Native counterpart to ATen's `Tensor.repeat_interleave` (explicit `dim`)

Closes `repeat_interleave.self_int`'s row in
[`todo-ops.md`](todo-ops.md)'s deferred target-name depth backlog (exposed
as ConViT's new first frontier by this session's own `repeat.default`
landing).

**What landed**

- `lib/native/ops/repeat.ml`: a new `RepeatInterleave` module (the file's
  header comment retitled to cover ATen's whole `repeat` family), sharing
  the file's `bounded_axis_extent` helper (factored out of `Repeat`'s own
  `axis_extent` in this entry) for the same int64-bounded-product-before-
  narrowing guard. `RepeatInterleave.params = { axis : Axis.t; repeats :
  Op_config.Pos.t }` — ONE named axis and a scalar multiplier, unlike
  `Repeat`'s six-axis vector, matching the real ATen argument shape
  (`repeat_interleave(self, repeats: SymInt, dim: int?)`).
  `Compute.pixel` reads `out / repeats` (floor division, via the existing
  `S.index_floor_div_pos` primitive — no new `SEMANTICS` primitive needed,
  same as `Repeat`) on the named axis only; every other axis passes through
  unchanged. This is the exact *opposite* composition from `Repeat`'s
  wraparound modulo: contiguous duplication (`[x,y]` repeat-interleaved by 2
  is `[x,x,y,y]`) vs. tiling (`[x,y]` repeated by 2 is `[x,y,x,y]`).
- `lib/native/op_config.ml`/`.mli`: `` `Repeats `` added to
  `Op_config.Bad.param` — the shared closed vocabulary both importers use
  to reject a non-positive op-configuration value with one detection
  origin, the same mechanism `` `Stride ``/`` `Kernel_size `` already use.
- `lib/native/graph_ir.ml`/`.mli`, `graph_shape.ml`, `eval_op.ml`,
  `transform/output_transfer.ml` (classified `Reindexing`, the same
  "wraparound-gather" argument `Repeat`'s own entry makes, just with floor
  division standing in for modulo), `graph_builder.ml`/`.mli`: the usual
  registry/dispatch/builder wiring.
- `lib/native_aten_bridge/op_bridge_shape.ml`: `repeat_interleave.self_int`
  dispatch arm, restricted to an explicit `dim` (the `dim=None`
  flatten-first form is a genuinely different reshape-then-repeat
  composition with no corpus occurrence, rejected via
  `` `Validation_failure ``). `output_size` is read-and-discarded, the same
  treatment `implicit` gets in `expand.default`'s arm.
- `lib/native_interp/{native_interp_error,native_interp,
  native_interp_lower_shape}.{ml,mli}`: the payload-free dispatch arm (the
  one `test/data/pt2_json_model_support.jsonl` actually measures), a new
  `` `Repeat_interleave_input `` metadata role, and a new
  `` `Repeat_interleave_dim `` row on `unsupported_option` for the same
  `dim=None` rejection, reported through the existing `Unsupported_option`
  mechanism rather than a bespoke error type.
- Tests: `test/native/repeat_test.ml` (Direct compute, three cases: two
  axes' duplication + the by-1 identity), `test/native/
  graph_direct_shape_test.ml` + `graph_symbolic_shape_test.ml` (one-node
  graph, Direct-vs-Symbolic agreement), `test/native_bridge/
  repeat_interleave_test.ml` (**new file**, ATen-linked `Op_bridge.dispatch`,
  hand-derived values, covering both axes of a rank-2 tensor, the identity
  case, the absent-`dim` rejection, and the non-positive-`repeats`
  rejection).

**Classification**: Full-stack for the Native surfaces (ATen boundary, both
importers, Native Direct/Symbolic) per `todo-ops.md`'s completion rule, for
the explicit-`dim` domain only; **incomplete** for Native4D/Kernel (no
counterpart) and for the `dim=None` configuration (typed rejection, not
implemented). May count toward Native target/configuration depth and
`native_builds`, not toward `native4d_converts`/`kernel_converts`/
all-three-stages.

**Corpus effect**: `convit_tiny` is the one corpus model gated on
`repeat_interleave.self_int` (20 occurrences, all `dim` explicit — 0 or 1 —
on ConViT's GPSA relative-position grid, immediately downstream of the
`repeat.default` work this session already landed). Its frontier moves to
`copy.default` — the next row of the same "Factories, indexing, and copies"
deferred-backlog family (40 occurrences, also MViTv2's current frontier) —
rather than to `native_builds:true`. `native_builds`/`native4d_converts`/
`kernel_converts`/all-three-stages are unchanged (87/56/63/43) — depth
moved, stage-completion did not.

**Next**: `copy.default` is now the highest-leverage target (ConViT and
MViTv2 both gated on it). `_to_copy.default` remains the other 2-model
target but still needs the value-domain trunc/round `SEMANTICS` primitive
flagged earlier this session. Otherwise, P1's remaining one-model slices
(`adaptive_max_pool2d`, `conv1d`/`unfold`, `im2col`/`col2im`,
`upsample_bicubic2d`).

### 2026-08-30 — `repeat.default`, the Native counterpart to ATen's `Tensor.repeat`

Closes `repeat.default`'s row in [`todo-ops.md`](todo-ops.md)'s deferred
target-name depth backlog (previously exposed as ConViT's first frontier by
the P0A `zeros.default` landing).

**What landed**

- `lib/native/ops/repeat.ml` (**new file**): `Repeat.params = { repeats :
  Vec6.shape }` (right-aligned multiplier vector, reusing `Vec6.shape`'s
  positive-`Dim.extent` representation since a repeat count, like an extent,
  is always >= 1). `output_shape` computes `x_extent * repeats` per axis,
  bounded in `int64` against `Kernel.Limits.Hard.extent` before narrowing —
  the same "bound the product before multiplying" rule
  `Resize.Bilinear_axis.check` follows, reusing `Window_over_limit`'s
  `` `Output_extent `` row rather than a new fault (the same reuse
  `Pad`/`Concat` make). `Compute.pixel` reads each axis at `out mod
  x_shape`, the exact "mod x d = x - d*(x/d)" idiom `Reshape.delinearize`
  already uses per axis — reused directly, not re-derived — but WITHOUT
  `Reshape`'s flatten-then-redistribute: repeat keeps every axis's own
  identity, so no cross-axis linearization is needed, and `Compute.pixel`
  needs only `x_shape`, not `repeats` itself.
- `lib/native/graph_ir.ml`/`.mli`: `Repeat` constructor + registry entry
  (alphabetical, between `Relu` and `Reshape`).
- `lib/native/graph_shape.ml`, `lib/native/eval_op.ml`: shape and compute
  dispatch arms.
- `lib/native/transform/output_transfer.ml`: classified `Reindexing` — "the
  wraparound gather `Expand`'s broadcast is the degenerate case of": every
  output reads exactly one input element with no arithmetic, taken modulo
  rather than clamped to 0.
- `lib/native/aten_shape.ml`/`.mli`: new `Repeat_size` module +
  `resolve_repeat_size ~self_dims ~repeats`, mirroring `resolve_expand_size`'s
  self-rank check without its `-1` substitution (repeat has only one fault:
  `repeats` shorter than `self`'s own rank). Shared by both importers, so
  they cannot drift on the rank check. `pp_ints`/`pp_dims` hoisted out of
  `Expand_size` to top-level helpers so `Repeat_size` doesn't restate them.
- `lib/native_aten_bridge/op_bridge_shape.ml`: `repeat.default` dispatch arm
  — reads `self`'s shape only (no real ATen kernel call, the same as
  `Reshape`/`Expand`/`Select`'s own arms), so no `lib/aten` C++ binding for
  `repeat` was needed to satisfy the "ATen boundary" surface.
- `lib/native_interp/{native_interp_decode,native_interp_error,
  native_interp_lower_shape,native_interp}.{ml,mli}`: `resolve_repeat`
  (mirroring `resolve_expand`), `Bad_repeat`/`` `Repeat_input `` rows, and
  the payload-free dispatch arm — this is the importer
  `test/data/pt2_json_model_support.jsonl` (and therefore `native_builds`)
  actually measures.
- Tests: `test/native/repeat_test.ml` (Direct compute, hand-derived, mirrors
  `reshape_test.ml`), `test/native/graph_direct_shape_test.ml` +
  `graph_symbolic_shape_test.ml` (one-node graph, Direct-vs-Symbolic
  agreement through `Eval_symbolic`/`Stage_program.ground`),
  `test/native_bridge/repeat_test.ml` (ATen-linked `Op_bridge.dispatch`,
  hand-derived expected values against real `Aten_tensor.t` fixtures,
  covering the rank-extension and rank-too-small cases).

**Deliberately not done**: a `Repeat4` Native4D counterpart. This session
scoped `repeat.default` as the smaller, well-understood slice specifically
*because* it needed no new `SEMANTICS` primitive (unlike the corpus's other
high-leverage frontier, `_to_copy.default`'s float-to-int cast, which does);
extending into Native4D was out of scope for this landing and is tracked as
ordinary backlog (`Repeat` joins `Index_tensor`/`Softmax` in
`Domain.check_node`'s "dialect does not have it at all" bucket — an
**incomplete** state per `todo-ops.md`'s classification table, not a
declared intrinsic-axis exception). No corpus model's frontier reaches this
boundary yet, so there is nothing to verify it against today.

Also **not done**: the optional `lib/native_op_walk` random-walk fuzz
harness (native_add_op.md step 8) — skipped as the doc marks it, in favor of
the required ATen-linked `test/native_bridge/repeat_test.ml` dispatch test,
which already exercises `Op_bridge.dispatch` against real `Aten_tensor.t`
fixtures with hand-derived values.

**Classification**: Full-stack for the Native surfaces (ATen boundary, both
importers, Native Direct/Symbolic) per `todo-ops.md`'s completion rule;
**incomplete** for Native4D/Kernel, since no counterpart exists. May count
toward Native target/configuration depth and `native_builds`, not toward
`native4d_converts`/`kernel_converts`/all-three-stages.

**Corpus effect**: `convit_tiny` is the one corpus model gated on
`repeat.default` (10 occurrences, ConViT's GPSA relative-position grid).
Its frontier moves to `repeat_interleave.self_int` — the next row of the
same "Shape / sequence" deferred-backlog family in `todo-ops.md` — rather
than to `native_builds:true`, since `repeat_interleave.self_int` (20
occurrences) is a later blocker in the same graph. `native_builds`/
`native4d_converts`/`kernel_converts`/all-three-stages are unchanged
(87/56/63/43) — depth moved, stage-completion did not.

**Next**: `repeat_interleave.self_int` (ConViT's new frontier) or one of
P1's remaining one-model slices (`adaptive_max_pool2d`, `conv1d`/`unfold`,
`im2col`/`col2im`, `upsample_bicubic2d`); `_to_copy.default` remains the
higher-leverage (2-model) target but needs a new value-domain trunc/round
`SEMANTICS` primitive first — see `todo-ops.md`'s "Next priority" note.

### 2026-08-30 — `Stack4`, the Native4D counterpart to Native's `Stack`

### 2026-08-30 — `Stack4`, the Native4D counterpart to Native's `Stack`

Closes the `Stack4` row of [`todo-ops.md`](todo-ops.md)'s "Remaining Native4D
counterpart backlog" table.

**What landed**

- `lib/native4d/ops4_split.ml`: `Stack4.params = { axis : Axis4.t }`, `xs :
  Tensor_ref.t list`, delegating its shape rule and pixel map to Native's
  `Concat.Stack` (itself built on `Concat.Concat` over each operand's
  unsqueezed shape) rather than restating either.
- `lib/native4d/ops4.ml`: `Stack4` re-export; `lib/native4d/op.ml`: `Stack4`
  constructor + registry entry (alphabetical); `lib/native4d/builder.ml`:
  `stack4` builder function.
- `lib/native4d/domain.ml`: `Stack` now gets the same `check_dims`-style
  axis-domain rejection `Concat`/`Select`/`Slice`/`Split_with_sizes`/`Unbind`
  already had, replacing its prior blanket `unsupported`.
- `lib/native4d/graph_shape4.ml`: `stack_params` adapter (shared by
  `Eval_op4` and the output-shape rule) + `Stack4` output-shape arm, gathering
  every operand's shape and delegating whole to `Concat.Stack.output_shape`.
- `lib/native4d/eval_op4.ml`: `Stack4` compute arm, through
  `Concat.Stack.Compute` — selects an operand by index, not by a
  within-segment offset, unlike `Concat4`'s arm.
- `lib/native4d/output_transfer4.ml`: `Stack4` classified `Reindexing` (pure
  data movement, no arithmetic — the same argument `Concat4`/`Select4` make).
- `lib/native4d/lower_engine.ml`: the actual `Stack -> Stack4` conversion arm,
  mirroring `Select4`'s: converts the axis, then re-validates the single
  output via `Shape4.of_vec6` — `Stack` *inserts* an axis rather than
  dropping one, so the packed result re-enters the dialect only when `H` is
  unit (the precondition always lands on `H`, not on whichever of `H`/`W`/`C`
  the new axis itself names — see `Ops4_split.Stack4`'s own comment).
- Tests: `test/native4d/op_json_test.ml` (JSON round-trip sample),
  `test/native4d/fixtures.ml` (`stack_n`/`stack_h_batch1`/`stack_h_nonunit`/
  `stack_rank5_t`, the `Stack` analogue of the existing `Unbind` fixture
  family), `test/native4d/domain_test.ml` (the axis-rule/shape-consequence
  pair those fixtures pin), `test/native4d/fixtures4.ml` (a `stack4` per-op
  Direct-vs-Symbolic fixture, H=1 so the result stays four-axis).
- `test/me_visualize_frontier_cram.t` and
  `test/data/pt2_json_model_support.jsonl`: regenerated after
  `make pt2.json-model-support`.

**Classification**: Full-stack, per
[`todo-ops.md`](todo-ops.md)'s completion rule — ATen boundary, both
importers, Native Direct/Symbolic, Native4D counterpart with shape/eval
coverage, and the Native-to-Native4D lowering arm are all present, each with
its own test evidence enumerated above.

**Corpus effect**: two corpus models had `Stack` as their `native4d` frontier.
`csatv2` (`axis=H`, node n17) moves from
`no legalization for stack xs=[t413, t416, t419] params={axis=H}` to a new
frontier one op later: `n19`, a `permute.default` that swaps a non-unit `T`
extent into `H` — an intrinsic multi-axis transpose the N/H/W/C dialect
cannot represent, not a missing counterpart, so this is a genuine Native-only
boundary rather than further Native4D work. `skresnet18` (`axis=D`, node n28)
moves from the same generic `no legalization for stack ...` message to the
dialect's own axis-domain diagnostic, `node n28: axis D is outside the
N/H/W/C dialect` — the same node and the same (already-intentional) rejection,
now reported through `check_dims` instead of falling through to the lowerer's
catch-all. Both models keep `native4d_converts:false`; `csatv2` remains a
graph-only CI target (`kernel_reason: over_limit`) and `skresnet18` remains
`kernel_converts:true` independent of this change. None of the four scoreboard
counts move — "depth moved; stage-completion did not", per
[`todo-ops.md`](todo-ops.md)'s recording rule.

**Next**: `Softmax4` is the one remaining row of the Native4D counterpart
backlog table, with no corpus model currently gated on it (Softmax has no
Native4D counterpart at any axis today, per `.ai/matmul_softmax_design.md`
§3). Otherwise, P1's remaining one-model slices
(`adaptive_max_pool2d`, `conv1d`/`unfold`, `im2col`/`col2im`,
`upsample_bicubic2d`) and promoting
`repeat.default`/`_to_copy.default`/`lstm.input` from the deferred backlog
are the next-ordered breadth-first work per `todo-ops.md`.

### 2026-08-30 — `Select4`, the Native4D counterpart to Native's `Select`

### 2026-08-30 — `Select4`, the Native4D counterpart to Native's `Select`

Closes the `Select4` row of [`todo-ops.md`](todo-ops.md)'s "Remaining Native4D
counterpart backlog" table.

**What landed**

- `lib/native4d/ops4.ml` (payload moved to new `lib/native4d/ops4_split.ml`,
  see below): `Select4.params = { axis : Axis4.t; index : int }`, delegating
  its shape rule and pixel map to Native's `Split.Select` (which itself
  delegates to `Split.Slice`) rather than restating either.
- `lib/native4d/op.ml`: `Select4` constructor + registry entry (alphabetical).
- `lib/native4d/builder.ml`: `select4` builder function.
- `lib/native4d/domain.ml`: `Select` now gets the same `check_dims`-style
  axis-domain rejection `Slice`/`Concat`/`Split_with_sizes`/`Unbind` already
  had, replacing its prior blanket `unsupported`.
- `lib/native4d/graph_shape4.ml`: `select_params` adapter (shared by
  `Eval_op4` and the output-shape rule, so they cannot disagree about which
  axis is dropped) + `Select4` output-shape arm.
- `lib/native4d/eval_op4.ml`: `Select4` compute arm, through
  `Split.Select.Compute`.
- `lib/native4d/output_transfer4.ml`: `Select4` classified `Reindexing` (pure
  data movement, no arithmetic — the same argument `Slice4`/`Unbind` make).
- `lib/native4d/lower_engine.ml`: the actual `Select -> Select4` conversion
  arm (this was the missing piece found mid-session — `domain.ml` already
  admitted `Select` through the axis-domain check, but no lowerer arm existed
  yet, so a legal graph would have hit the lowerer's own
  domain-check/match-disagreement invariant). Converts the axis, then
  re-validates the single output via `Shape4.of_vec6` for the same reason
  `Unbind`'s multi-output arm does: `Select` drops its axis, so the result
  re-enters the dialect only when the packed shape stays four-axis.
- `lib/native4d/ops4_split.ml` (**new file**): `Slice4`/`Select4`/`Concat4`/
  `Unbind`/`Split_with_sizes4` moved out of `ops4.ml` verbatim, re-exported by
  the same names, to keep `ops4.ml` under the tracked 1000-line file-size
  ceiling (`scripts/check-file-size.sh`) after `Select4`'s addition pushed it
  to 1026 lines. Mirrors the precedent `ops4_conv.ml` set for the convolution
  family.
- Tests: `test/native4d/{op_json_test,fixtures,fixtures4,domain_test,
  lower_test,compute_test,verify_test}.ml` — JSON round-trip sample,
  axis-rule/shape-consequence fixtures (`select_c_batch1`/`select_n`/
  `select_c_batch2`/`select_rank5_t`, the `Select` analogue of the existing
  `Unbind` fixture family), a per-op Direct-vs-Symbolic fixture, a hand-value
  Direct oracle test (checked against `Split.Select.Compute`, not against
  Native — proving the arithmetic, not just self-consistency), a lowering
  golden, and an end-to-end Native-vs-Native4D numeric verify test.
- `test/me_visualize_frontier_cram.t` and
  `test/data/pt2_json_model_support.jsonl`: regenerated after
  `make pt2.json-model-support`.

**Classification**: Full-stack, per
[`todo-ops.md`](todo-ops.md)'s completion rule — ATen boundary, both
importers, Native Direct/Symbolic, Native4D counterpart with shape/eval/oracle
coverage, and the Native-to-Native4D lowering arm are all present, each with
its own test evidence enumerated above.

**Corpus effect**: `csatv2` is the only corpus model whose frontier `Select4`
was gating (`.ai/`-tracked design intent aside, `Select` occurs at `axis=H`
in that graph's `n3`). Its frontier moves from
`no legalization for select x=t405 params={axis=H index=0}` to
`no legalization for stack xs=[t413, t416, t419] params={axis=H}` — i.e. to
the next item in the same "Remaining Native4D counterpart backlog" table
(`Stack4`). `csatv2` stays `native4d_converts:false` either way and remains a
graph-only CI target (`kernel_reason: over_limit`, independent of this
change), so none of the four scoreboard counts above move. This is exactly
the "depth moved; stage-completion did not" outcome
[`todo-ops.md`](todo-ops.md) asks landings to record rather than treat as no
progress.

**Next**: `Stack4` (csatv2's new frontier) and `Softmax4` are the two
remaining rows of the Native4D counterpart backlog table; `Stack4` now has a
concrete corpus model to verify against, the same way `Select4` did coming
into this session.
