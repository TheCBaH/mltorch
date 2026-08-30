# `index.Tensor`: a genuine runtime gather

**Status: landed, 2026-08-29** (superseding this doc's original "investigated,
not implemented" verdict, recorded the same day). This is now a design record
for landed code, the same role `.ai/matmul_softmax_design.md` plays for the
batched-matmul family — not a scoping record for a deferred item.

## 1. The one occurrence

CSATv2 (945 nodes) has exactly one `torch.ops.aten.index.Tensor` node, and it
is the only occurrence anywhere in the 100-model sweep. Its `indices` operand
is `[None, None, <tensor>]` against a rank-3 `self` (`[784, 3, 64]`) — matches
`ops.md`'s framing, "an optional-tensor index list with only the final index
live": a single-axis gather over the last axis, broadcast over the two
leading axes. ATen's own promotion rule for this shape
(`self.shape[:p] + index.shape + self.shape[p+1:]`) gives `[784, 3, 64]`,
matching the archive's declared output shape exactly.

Tracing the index tensor back through the graph (`clone_1 <-
clone.default <- c_stem_dct_lifted_tensor_0`) shows it is a **lifted tensor
constant** — `model_constants_config.json` names it `stem_dct.lifted_tensor_0`,
dtype `LONG`, shape `[64]`, backed by real bytes in `data/constants/tensor_0`.
It is not derived from the model's runtime input; every run of this model
uses the same fixed reindexing (almost certainly the model's DCT
coefficient/zig-zag reorder, given the module name `LearnableDct2d`).

## 2. Why "bake the constant permutation into the graph" doesn't work for both importers

The obvious narrow scope — decode the index tensor's concrete `int64` values
at import time and store them as a static lookup table in a new op's params,
the same way `Split_with_sizes`' sizes or `Slice`'s offsets are compile-time
data — runs into an asymmetry between the two importers this repo requires to
stay in sync (`.ai/native_add_op.md`):

- **`Op_bridge`** is ATen-linked and reached with live `Aten_tensor.t` values
  already materialized, so it genuinely could read the constant's concrete
  bytes at dispatch time.
- **`Native_interp`** is metadata-only during lowering. A constant input is
  imported with only its shape/dtype (`Input.Constant` kind); its concrete
  bytes are not available until `Eval_direct.run ~constants` runs, well after
  the graph structure is fixed (`native_interp_exec.ml`'s `transform_lowered`/
  `constants_for`). There is no channel back from "the eval-time constant
  store" into "the graph-build-time op params" — building it would mean
  `Native_interp`'s import step stops being metadata-only for this one op,
  which is a boundary the rest of the importer (and the jsoo backend it
  feeds) currently relies on.

So a build-time-baked lookup table is only implementable for one of the two
importers required to land any op (`ops.md`'s five-layer "supported"
definition demands both). This investigation originally stopped here and
deferred the op; the user then chose to fully implement the genuine runtime
gather instead (§3 below is what that took).

## 3. The runtime gather that landed

The alternative to a build-time-baked table is a real two-tensor-input
`Graph_ir` node evaluated at `Eval_direct`/`Eval_symbolic` time like any other
op, with the gathered axis's position read out of the `index` tensor's
*value* rather than known at compile time. This is genuinely a bigger unit of
work than most single-op additions — comparable to `Sdpa` — and went through
five gated commits, each independently built and tested:

- **Gate 1 — raw `Long` read infrastructure.** `Tensor.read_i64_at6`, an
  exact single-cell `I64` accessor that typed-rejects every other format
  (never a lossy float round trip); `Native_interp_tensor.tensor_of_pt2`'s
  new `Int64`/`Long` arm, materializing a genuine `I64`-format tensor
  (`Tensor.materialize_i64`) rather than going through the existing
  F32-hardcoded `materialize`; and `Expr.Eval.resolve_gather_index`, which
  validates a raw stored `int64` against ATen's own `[-extent, extent-1]`
  range **before** narrowing to `int` — narrowing first would let a value
  near `Int64.min_int`/`max_int` wrap into a spuriously in-range position.
- **Gate 2 — `Index.Data`, propagation, `SEMANTICS.load_index`.** `Expr.Index`
  gained a `Data` constructor (`Source.t * position-coord * extent ->
  position`), self-recursive within `Index.t` rather than mutually recursive
  with `Value.t`. Every traversal that must handle a new `Index.t` case
  learned about it: `Fold` (including a real, index-aware `idx_fn` for
  `sources`, so a `Data`-embedded source is visible there — the previous
  `no_index` traversal would have missed it), `Pp`, `Rewrite` (including a
  new `map_index_sources`, since `map_sources`'s old `keep_indices` silently
  left a `Data` node's own source unrewritten), and `Value`'s structural
  compare/hash. The private evaluator behind the public `Eval.index`,
  `eval_index`, is now generalized over the caller's own error row via an
  explicit `resolve_data`/`widen` pair, so both `Eval.value` (at the wide
  `error` row, via a new `Env.load_index` field) and native's `Ground_eval`
  (at its own error row, via a new `resolve_data_source`) can resolve a
  `Data` source with no second copy of the bounds-checking/normalization
  logic. Native wiring: `SEMANTICS.load_index`, `Direct.load_index` (raises
  `Err.Exn.E` on failure, matching `Direct`'s existing exception-based
  contract — see `.ai/error_handling_design.md`), `Symbolic.load_index`
  (construction only, via `Expr.Index.data` — no check at build time),
  `Expr_bridge.env`'s new `load_index` field, and `Kernel_eval.machine`'s
  `env_for` (a virtual arm that is structurally dead — `kernel_elab.ml`'s
  pointwise-admission predicate already rejects `Data` via its catch-all —
  but must still type-check, so it reuses the shared
  `` `Data_index_unexpected_here `` tag rather than inventing a second one).
  `Ground_eval.resolve_data_source` resolves a `Data` source **only** for a
  directly-bound constant, with **no stage-walking fallback**:
  `Stage_program.ground` evaluates every stage through `Tensor.materialize`,
  which unconditionally allocates `F32`, so a stage's *expression* being
  value-preserving does not mean its *execution* is storage-preserving —
  tracking that distinction end-to-end was ruled out as its own, larger unit
  of work, the same way §4 (below) treats `Clone`. Anything not directly
  bound falls through to a new `` `Data_index_unresolved `` tag, feeding the
  existing generic `Ground_eval.error -> Unproved` conversion unchanged.
- **Gate 3 — the Native op, `Graph_ir.Index_tensor`.**
  `Index_tensor.Index_tensor` (`lib/native/ops/index_tensor.ml`): `self`/
  `index` `Tensor_ref.t` operands plus an `axis` param, scoped to `index`
  rank 1 so the shape rule needs no rank fields at all — a rank-1 index
  insertion is identity-preserving on `self`'s own rank
  (`output_rank = self_rank`), which sidesteps the six-axis frame's
  rank-erasure problem entirely rather than solving it (the same erasure
  `expand.default`'s importers already had to solve, per
  `Aten_shape.resolve_expand_size`'s own header comment). `output_shape`
  enforces the rank-1 restriction at the `Graph_ir` level too (every index
  frame axis other than `C` must have extent 1), reachable from a
  hand-built or JSON-decoded graph and not just from importer validation.
  `Compute.pixel` builds `index`'s own read coordinate **explicitly** — an
  all-zero `Vec6.t` with only `Axis.C` set to `out`'s value on the gathered
  axis — never inherited from `out` directly, which reads the wrong axis
  whenever the gathered axis isn't `C` and is out-of-bounds on `index`'s own
  shape otherwise. `self`'s read coordinate is `out` with the gathered axis
  replaced by the resolved `S.load_index` position. Wired through the usual
  sites (graph_ir registry, graph_shape, eval_op, output_transfer —
  `Discontinuous`, since which input element is read is data-dependent, the
  same argmax-shaped reasoning `Max_pool2d_with_indices`'s index output
  gets — graph_builder, Native4D's `Unsupported_op` bucket). No permute
  allowlist entry (default barrier is correct) and no ATen C binding
  (`Tensor?[]` has no `lib/aten_gen` C-shim support, the same gap
  `addcmul.default`/`group_norm.default` hit).
- **Gate 4 — import, both bridges, the trace-past-`Clone` rule.** Both
  importers decode `indices` and enforce the locked list-acceptance rule:
  accepted iff its length equals `self`'s ATen rank, exactly one entry is a
  Long-dtype tensor of ATen rank exactly 1, and every other entry is an
  explicit `None` — any other shape is a typed rejection naming what was
  found (`Op_bridge_error.Index_list`/`Native_interp_error.Index_list`,
  kept structurally identical between the two). `axis` comes from ATen's
  `indices` list position via each importer's existing dim-to-axis machinery
  (`Op_bridge`'s `dim_axis`, `Native_interp`'s `axes_for_rank`). Resolving
  the live `index` operand is where the two importers genuinely diverge:
  - **`Op_bridge`** needs **no** trace-back. It already resolves the live
    entry's real ATen *value* directly (`aten_env`'s SSA-name → tensor
    binding), whatever node produced it — `clone.default` included, since
    ATen's own `clone` preserves value and dtype exactly. Its `dispatch`
    signature is `~aten_env -> node -> ...`, with no surrounding node list to
    trace through in the first place, and it has nothing to gain from one:
    the round-6 problem below is specific to metadata-only import.
  - **`Native_interp`** genuinely needs the trace-back, since it *is*
    metadata-only: importing `indices`' live entry naively would bind
    `Index_tensor`'s `index` operand to `clone_1`'s own native output edge,
    whose signature `Graph_builder.op1` stamps `F32` by default regardless of
    the source's real dtype — a pre-existing, unrelated characteristic of
    every `clone.default` occurrence in this codebase, not a defect this op
    introduces. Worse, `clone_1`'s edge would then actually get *evaluated*
    through the ordinary pointwise `Clone` compute path (there is no
    dtype-preserving `Eval_direct` bypass for `Clone`, unlike `Unbind`/
    `Split_with_sizes`), materializing a lossy `F32` copy of the original
    `Long` values — so `Tensor.read_i64_at6` would then correctly *reject*
    that `F32` edge as the wrong format, and CSATv2's own occurrence would
    fail at evaluation time on every run. `Native_interp_lower_context` now
    carries `constant_names` (the SSA names of `Parameter`/`Buffer`/
    `Tensor_constant`-kind graph inputs — model data fixed across every run,
    as opposed to a genuine `User_input`), and
    `Native_interp_decode.resolve_index_source` peels a wrapping
    `clone.default` only when it requests no real format change **and** its
    own input is one of those captured constants, binding the *original*
    pre-clone edge instead of Clone's. Tracing past anything else (a
    computed, non-constant input) would have nothing to gain: only a
    directly-bound captured constant can hold a non-`F32` dtype in this
    engine at all, since every computed intermediate is `F32` by
    construction.
  - `native_interp_lower.ml`'s `reads` computation (used to detect a
    genuinely dead node output, e.g. `native_layer_norm`'s `mean`/`rstd`)
    tracked every other list-valued argument kind but not
    `Argument.Optional_tensors` — fixed alongside, since an
    `Optional_tensors`-referenced edge could otherwise have been wrongly
    treated as dead.

## 4. Verification

- A hand-derived `Direct` `Compute` fixture with the gathered axis at a
  **non-`C`** position and **nonzero** values on the frame axes that pass
  straight through — the only kind of test that can catch a
  wrong-but-shared coordinate mapping in `Compute.pixel`, since Direct and
  Symbolic run the identical code and so agree with each other even when
  both are wrong about which axis to read.
- A new Native Direct-vs-Symbolic fuzz walk
  (`lib/native_op_walk/index_tensor_nwalk.ml`), registered in
  `native_op_walk.ml`.
- Hand-derived `dispatch:`/importer fixtures on both bridges: a live entry
  at a non-last position, a wrong-length list, a boolean-mask entry, two
  live entries, a live entry of ATen rank 2, and the full `Long constant ->
  clone.default -> index.Tensor` end-to-end fixture (asserting
  `Index_tensor`'s `index` operand resolves to the **original** constant's
  SSA name, not Clone's output).
- An out-of-range gather value proving `Eval_direct.run` **raises**
  `Err.Exn.E`, not a silent out-of-bounds read and not a returned `Error`.
- **Confirmed against the real corpus**: `native_graph print --pt2
  csatv2.pt2` (via its committed payload-free `model.json` — see
  `test/me_visualize_frontier_cram.t`) now imports past every
  `index.Tensor` occurrence, CSATv2's first fully import-clean run, and
  stops instead on `Pow` not being a Const-SSA operation — an unrelated,
  pre-existing limitation. CSATv2 stays a graph-only CI fixture regardless:
  the deliberately-deferred batched `matmul.default` family still keeps that
  classification until it is also substantially complete, so this landing
  alone does not move it to a native-verified target, and it blocks no other
  model in the 100-model sweep.
- **The gap `print` alone couldn't catch**: `print` is metadata-only and
  never preloads a real payload, so it never exercises `Rewrite.origin`'s
  constant-payload check. `to4d --fold` does — and initially failed on
  CSATv2 with `payload for t400 does not match its signature`, because
  `native_interp_lower.ml`'s own `constant ~shape ()` allocation (the *pre*-clone
  edge the trace-back rule above binds to) stamped `F32` by default exactly
  like `Clone`'s own edge does, for the same `Graph_builder` reason — the
  round-6 trace-back moved `index`'s binding to an edge whose *declared*
  signature was just as wrong. Fixed by
  `Native_interp_decode.tensor_fmt`, which reads a captured
  `Parameter`/`Buffer`/`Tensor_constant`'s own `graph.tensor_values` dtype
  and stamps `I64` when it says `LONG`, F32 otherwise — a signature fix for
  the one non-`F32` dtype this engine models, not a general dtype validator
  (an unrelated bad dtype, e.g. a `BOOL` mask fed as `indices`, is left for
  the op-specific `Index_list.Wrong_dtype` check to reject with its own
  message). `Pow is not a Const-SSA operation` is `to4d --fold`'s blocker
  post-fix, confirmed by `test/pt2_model_support_cram.t`.
