# Scaled dot-product attention (Group 8) — design record

Approved design record for `torch.ops.aten.scaled_dot_product_attention.default`
(`op8.md`'s target), per `op8.md:41-53`'s requirement that this be fixed before
implementation. Grounded against the checkout at `unbind-int`
(`c62a16f`, Group 7 landed); see `op8-impl.md` for the finding-by-finding
derivation (F1-F13) and the six-commit sequence that implemented it.

## 1. Scope: `op8.md` as written, decomposition out of scope

Two user decisions fix the scope, made explicit here because they are the
reason several of `op8.md`'s premises turn out false in this checkout:

1. **Deliver `op8.md` as written** — the literal target, bound and
   interpreted, behind one retained Native `Sdpa` operation, meeting every
   completion criterion on its own terms. The decomposition set that would
   actually unblock the four ViT models (twelve primitives: `view`, `expand`,
   `permute`, `mul.Scalar`, `bmm` (importer), `logical_not`, `_softmax`,
   `eq.Scalar`, `any.dim`, `full_like`, `where.self`, `clone`, `squeeze.dims`,
   `unsqueeze.default`, `gelu.default`, `cat.default`) is explicitly out of
   scope, left for a separately ranked plan.
2. **Spike the ATen archive link before any Native code lands.** A link
   failure means stop and re-scope, not proceed on an oracle that does not
   exist.

**This row delivers zero model coverage.** The target occurs in NO reachable
graph across all 23 core-ATen graphs and all 13 downloaded archives — its
NAME appears 600-1200 times inside the four `vit_*` graphs'
`metadata.from_node` provenance (in `vit_b_32`, 300 of 839 nodes, 36% of the
graph), but every exporter decomposes it before writing `model.json`. Every
fixture in this row's test suite is hand-built; `make pt2.runtest`, `make
inference` and `make jsoo.pt2.runtest` show no diff, and must not be reported
as evidence of model progress. See `.ai/testing_strategy.md`'s Group 8
section for the full accounting.

## 2. Supported rank/layout and axis mapping

Q/K/V are rank-4, right-aligned by `Aten_shape.of_aten` (purely positional)
onto the six-axis frame's innermost four axes:

```text
query [D=batch, H=heads, W=Wq (query sequence), C=E (head dim)]
key   [D=batch, H=heads, W=Wk (key sequence),   C=E]
value [D=batch, H=heads, W=Wk,                  C=Ev]
out   [D=batch, H=heads, W=Wq,                  C=Ev]
```

**No relayout is needed.** This is `Bmm`'s "no relayout needed" case
(`.ai/native_aten_bridge_layout.md`), generalized to two batch-like axes
(`D`=batch, `H`=heads) instead of one: the axes the op reads are exactly the
axes the data lands on under right-alignment, so neither importer needs a
`Permute` around the op, and `Verify.compare_tensors` works directly.

**And this settles Native4D by construction.** The op names `D` as its batch
axis, and `Axis4.of_axis Axis.D = None` — the four-axis dialect has no name
for `D` at any extent, including 1. See §9.

## 3. `Ev = E`: the supported scope is exactly what keeps the flash kernel

`op8.md:227` lists "differing key/value feature dimensions" as a required
numerical case. **This plan deviates from that deliberately and says so
here, not quietly:** the supported scope requires `Ev = E`, so
`output_shape` is exactly `query_shape`.

The reason is the CPU dispatch gate, not a limitation of this
implementation's arithmetic. `sdp_utils_cpp.cpp`'s `use_flash_attention_cpp`
admits f32, dense, non-nested, `dropout_p = 0`, rank-4, last-dim stride 1,
non-zero sequence lengths — and `check_head_dim_size_cpp` additionally
REQUIRES `query.size(-1) == key.size(-1) == value.size(-1)`; the flash
kernel itself re-checks it (`attention.cpp:938-939`). A distinct `Ev` does
not error — ATen has no problem computing it — it silently moves the
computation to the `math` backend, a different kernel with a different
reassociation of the same arithmetic. Supporting `Ev <> E` here would mean
adopting a second oracle and a second tolerance policy, which is a separate
decision, not a widened `atol`. See §7 for the full flash-gate table.

## 4. The mask: additive f32 only, in the flash-admissible broadcast set

**No boolean mask.** `op8.md:144-146` requires an explicit choice here: a
boolean `attn_mask` is REJECTED with a typed row
(`Attention.Sdpa.Reject.Boolean_mask`), never reinterpreted as f32. ATen's
own `convert_boolean_attn_mask` turns a bool mask into an additive `0`/`-inf`
float mask internally, so the semantics ARE representable — but the operand
type is not, through the conversion this row performs (a Native tensor is
f32 by construction, and CLAUDE.md's payload rule forbids silently
reinterpreting a closed value's meaning). A bounded boolean-tensor capability
is a separate row, not folded in here.

**The admissible mask shapes are exactly the flash kernel's.**
`check_attn_mask_shape` (`sdp_utils_cpp.h:295-320`) admits, where each axis
is either the real extent or 1:

```text
2d : [ Wq|1 ,  Wk|1 ]
4d : [ D|1 ,  H|1 ,  Wq|1 ,  Wk|1 ]
```

A mask outside this set is not an ATen error either — it is a second silent
fallback to the `math` backend, alongside `Ev <> E`. Both are rejected here
as flash-oracle boundaries and the diagnostics say so
(`Attention.Sdpa.Reject`'s comments), because a reader who does not already
know this would otherwise assume they are mathematical restrictions.

**The mask is indexed at the SCORE coordinate, not the output coordinate.**
`score_shape ~query_shape ~key_shape` names it: `[D, H, Wq, Wk]`, which is
`query_shape` with `C` replaced by `key_shape.W` — not the query shape
whenever `Wk <> E`. Reading it requires `Pointwise.broadcast_coord`
(`load` is strict, so a broadcast axis must be reduced to index 0 first),
against a coordinate neither the op's own output shape nor another
operand's — see `.ai/native_tensor_design.md` §1c's Sdpa note.

## 5. Rank is an importer-only rule (F13)

`output_shape` checks everything the six-axis frame CAN express — per-axis
extent agreement, `Ev = E`, key/value's shared `Wk`, mask broadcast-or-equal
against `score_shape` — and NOTHING about rank, because it cannot: a rank-2
`[Wq,Wk]`, a rank-3 `[1,Wq,Wk]` and a rank-4 `[1,1,Wq,Wk]` mask all fold to
the identical Native shape once `Tensor_bridge.of_aten` erases the ATen
rank. The first and third are flash-admissible; the middle is not, and only
a check on the RAW ATen rank, before conversion, can tell them apart.

Both importers therefore check Q/K/V at exactly rank 4 and the mask at rank
2 or 4, on the raw tensor/metadata, before calling `Tensor_bridge.of_aten` /
right-aligning the serialized `sizes` list. **A standalone Native or
JSON-decoded graph carries no ATen rank at all** — this is not a gap to
close later, it is a property of the boundary: nothing downstream of
`of_aten` has anywhere to keep a rank, and a hand-built Native graph holding
a mask no admissible ATen node could have produced still evaluates soundly,
because the frame-level broadcast check is what `Compute` actually depends
on. See `.ai/native_aten_bridge_layout.md`'s Group 8 section for the
promoted general statement of this rule.

## 6. The arithmetic: ATen's `math` backend structure, not `op8.md`'s literal formula

`op8.md:118-123` specifies `score(q,k) = dot(query[q], key[k]) * scale`.
**Neither ATen backend computes that.** `_scaled_dot_product_attention_math`
(`attention.cpp:864-893`) splits the scale as its own square root, applied
to BOTH operands before the matmul:

```cpp
const auto scaling_factor = sdp::calculate_scale(query_acc, ...).sqrt();
const auto query = query_acc * scaling_factor;
auto attn = at::matmul(query, key_expanded.transpose(-2, -1) * scaling_factor);
```

for overflow stability (`attention.cpp:863-864` cites the rationale). The
corpus confirms it independently: `vit_b_32` node 30 is
`mul.Scalar(view_7, 0.3535533905932738)` and node 32 is
`mul.Scalar(permute_7, 0.3535533905932738)` — the SAME factor on both
operands, and `0.35355... = 64 ** -0.25` for `head_dim = 64` (i.e.
`sqrt(1/sqrt(64))`). `scale * dot(q,k)` and `dot(q*sqrt(scale), k*sqrt(scale))`
are mathematically equal in exact arithmetic but NOT f32-identical — proven
bitwise in `test/native/sdpa_test.ml`'s dedicated split-vs-single-multiply
case — so implementing `op8.md`'s literal formula is a required MUTATION
target, not the contract, and is exercised as one.

The reference computation this row implements, in full:

```text
sf        = sqrt(scale)                       (scale = Explicit s | 1/sqrt(E))
score(q,k)= sum_{e<E} (query[q,e]*sf) * (key[k,e]*sf) + mask(q,k)
m(q)      = max_reduce_{k<Wk} score(q,k)
z(q)      = sum_{k<Wk} exp(score(q,k) - m(q))
out(q,v)  = select (m(q) > -inf)
              (sum_{k<Wk} (exp(score(q,k) - m(q)) / z(q)) * value[k,v])
              0.                                (* _safe_softmax, unconditional *)
```

**`_safe_softmax`'s zero-row is unconditional, not only under a mask.** ATen
computes `attn = at::_safe_softmax(attn, -1)` regardless of whether a mask
was given: a row whose scores are all `-inf` yields `0`, not `NaN`. The
corpus proves this runs unconditionally too — `vit_b_32` nodes 40-45 are its
decomposition (`eq.Scalar(-inf)` -> `logical_not` -> `any.dim` ->
`logical_not` -> `where(all_neg_inf, 0, softmax)`), present even where no
mask feeds the block. `op8.md:118-123`'s stable-softmax sketch would produce
`exp(NaN)` there; the guard removed is a required mutation, observed
producing `nan` on this row's own all-`-inf` fixture.

**A negative explicit scale is legal in ATen and specially handled**
(`is_negative_scaling`: `|scale|` under the sqrt, then the query factor
negated) — rejected here rather than implemented, and the diagnostic says
so (`Attention.Sdpa.Reject.Negative_scale`). A non-finite explicit scale is
rejected too (`Non_finite_scale`).

**`m`/`z` are recomputed per output feature, not cached** — `op8.md:127-128`
requires this choice be explicit. It is what costs F12's eight-factor total
work bound (§8), not a smaller one, and it is what makes the `Symbolic`
AST ~3x one `score_at` evaluation's size (op7-impl.md F4's builder-is-not-a-
node point, applied here): measured at depth 12 / size 121 nodes, flat
regardless of the reduction extents, before any code landed.

**No new semantics primitive was needed.** `exp`, `max_reduce`, `sum`, `lt`
and `select` already existed on both `Direct` and `Symbolic`
(`ops/window_axis.ml`'s `max_reduce` was the existing consumer) —
`op8.md:163-168`'s worry about a new primitive's cross-cutting cost did not
materialize.

## 7. The flash-gate table, as a scope decision, not a mathematical one

| gate | requirement | consequence if violated |
|---|---|---|
| dtype/layout | f32, dense, non-nested, last-dim stride 1, non-zero sequence lengths | silent fallback to `math` |
| `dropout_p` | `= 0` | silent fallback to `math` |
| rank | Q/K/V rank 4 | silent fallback to `math` |
| head dim | `Ev = E` (`check_head_dim_size_cpp`) | silent fallback to `math` |
| mask rank | 2 or 4 | silent fallback to `math` |
| mask shape | every axis is the real extent or 1 (`check_attn_mask_shape`) | silent fallback to `math` |
| mask dtype | f32 or query's dtype | silent fallback to `math` |

Every row is a flash-ORACLE boundary. None of these is a mathematical
restriction on what attention computes — ATen's `math` backend would answer
every one of them, just from a different kernel with a different
reassociation, which is why "rank alone pins the backend" is false (an
earlier draft of the implementation plan claimed it did, and simultaneously
tried to walk `Ev <> E` — the two claims contradict each other) and why
`dropout_p`/`is_causal`/`enable_gqa`/a boolean mask/an out-of-set mask
shape/`Ev <> E` are all typed rejections here rather than supported cases
under a widened tolerance: a single `atol_for_target` cannot represent two
different kernels (§10).

## 8. Shape and resource ceilings

Three 32-bit aggregates, per CLAUDE.md's rule (`lib/native` is
js_of_ocaml-reachable, where `int` is 32 bits):

1. **Score count** `D·H·Wq·Wk`, bounded via `Vec6.numel_bounded` on
   `score_shape`.
2. **Output numel** `D·H·Wq·Ev` (`= D·H·Wq·E`), bounded via
   `Vec6.numel_bounded` on `query_shape`.
3. **Total work** `N·T·D·H·Wq·Wk·E·E` (F12) — the one the first two do not
   imply. The op recomputes score/max/denominator per output feature (§6),
   so direct work is proportional to all eight factors, and the
   counterexample is not subtle: `D=H=1, Wq=Wk=E=Ev=1024` passes both of the
   first two bounds by six orders of magnitude while real work is
   `~3·10^12`. `N` and `T` were left out of an earlier version of this bound
   (and of `output_shape`'s cross-operand checks entirely) — op8-impl-review.md
   P1 found that a standalone or JSON-decoded graph could give query/key/value
   disagreeing `N` or `T`, which `Compute` reads at the output coordinate
   unchanged (never through `broadcast_coord`, exactly like `D`/`H`), so an
   unequal one either reads an operand out of bounds or silently drops part
   of it — and the resource bound did not count their contribution to work
   either. Both are now checked the same as `D`/`H` and folded into this
   bound the same as the other six. No single `Vec6.shape` holds all eight
   factors (`E` appears twice; no
   operand's own shape has both `Wq` and `Wk`), so this is its own
   divide-before-multiply fold over an explicit factor list, following
   `Kernel.Bounds.signature` — see `.ai/native_add_op.md`'s generalization
   of this rule.

All three live in `output_shape`, the one site both `Graph_builder` and
JSON decode traverse, never in `Compute` (which has no error channel).

**Empty-domain policy:** the engine has no representation for a zero
extent (`Dim.extent` is `>= 1` by construction), so an empty query, key or
feature extent is unrepresentable rather than a case needing a rejection —
consistent with every other structural op in this engine.

## 9. Native4D: typed rejection at `D > 1`, a Region-authored counterpart at `D = 1`

This section originally argued for UNCONDITIONAL rejection (every reasoning
bullet below, struck through in `.ai/native4d_design.md` §7.9, conflated
`D > 1` with `D = 1`). That conclusion was superseded 2026-09-02: once
Region computation (`.ai/region_compute_design.md`) gave this op an
operation-authored `Region_program.t` shared with Native, a genuine
`D = 1` counterpart became representable without a second numeric kernel —
`Shape4.of_vec6` already admits any axis at unit extent, `D` included, the
same way it admits `Batched_matmul`'s own `D = 1` case (§7.4). Its own
`Domain.check_node` arm rejects only `D > 1`, with `` `Sdpa_batch_axis ``,
an operation-specific reason (`Live_max_pool_indices`'s precedent) naming
that the batch axis is `D`. Model Explorer reports
`stage:native4d unavailable outside_dialect_domain` for `D > 1`; the source
and Native projections stay available. See `.ai/native4d_design.md` §7.9 for
the full record, including why direct `Ops4.Sdpa` and a sound legalization
both remain unavailable at `D > 1` — only the Region-authored route changed.

`mobilenetv5_base` (§12 below) is the first corpus model to reach this
counterpart: every one of its SDPA occurrences has `D = 1` (its batch
broadcasting is entirely on `H`), so `check_sdpa` admits them all.

## 10. Dtype and tolerance policy

The engine computes in f32; verification against real ATen uses
`Verify.atol_for_target`, which carries one entry for this target,
`1e-5`, pinned independently of `default_atol` (also currently `1e-5`, but
the entry does not inherit that by coincidence — a future unrelated change
to the default must not silently change what this target was verified
against). The value comes from measurement, not assumption: a scratch probe
(never committed) measured `max |flash - math|` at `4.1e-7` across
`Recipe_sdpa`'s full shape space (3240 samples) and `1.5e-8` across
large-magnitude logits (the max-subtraction-stability case, x20 scale).

This is defensible as ONE entry only because §7's gates, enforced by the
recipe's types (F9) and independently checked by a second scratch probe over
the recipe's full axis product (5832 points, all `flash_attention`), make an
off-oracle (`math`-backend) configuration unrepresentable THROUGH THE
RECIPE-DRIVEN WALK specifically. `Recipe_sdpa` itself was not widened by
§12's head/batch broadcasting landing (2026-09-04) — it still shares one
`batch`/`heads` pair between query/key/value, so it still cannot generate a
broadcast configuration, the same scoping precedent
`.ai/matmul_softmax_design.md` §6 records for `Batched_matmul`'s own
broadcast landing. What DID change: `Attention.Sdpa.output_shape`/`Compute`
now ADMIT broadcast configurations Native rejected at the shape level
before, and those are provably off the flash gate (§7's own
`check_batch_size_and_num_heads_dense`, `sdp_utils_cpp.h`, requires equal
heads AND equal batch without GQA) — so this single
atol entry now ALSO covers hand-verified `math`-backend points, not only
`flash_attention` ones. Two hand-derived `verify_print` fixtures
(`test/native_bridge/sdpa_test.ml`, one per broadcast axis family: `H`
mismatch and `D` mismatch, both `enable_gqa=false`) confirm real ATen agrees
with this row's computation at the SAME `1e-5` — evidence, not a new
measured bound, since neither fixture is large enough to be a tight probe
the way the two scratch measurements above are. If broadcasting turns out
to need a materially different tolerance once more of the space is
exercised (a widened recipe, or corpus evidence beyond `mobilenetv5_base`),
it gets its own policy at that point, per this section's own original rule.

`lib/native_op_walk/native_verify.ml`'s Direct-vs-Symbolic comparison stays
bitwise `Float.equal` — untouched, per `op8.md`'s own requirement and
`op7.md:59-62`'s precedent.

## 11. What this row does not deliver

- No model coverage AT THIS ROW'S ORIGINAL LANDING; every fixture was
  hand-built (§1, `.ai/testing_strategy.md`). Superseded 2026-09-04: §12's
  head/batch broadcasting landing gives `mobilenetv5_base` real coverage
  through `native_builds`/`native4d_converts` (stopping at an unrelated
  kernel-stage depth ceiling) — still the only corpus model that reaches
  this op undecomposed.
- No causal masking, dropout, GQA, boolean-mask, negative-scale, or
  non-rank-4 support — each a typed rejection, not a partial implementation.
- No differing key/value feature dimension (`op8.md:227`'s case) — a
  deliberate deviation (§3), not a silently dropped requirement.
- No mask form outside flash's admissible set (§4) — narrower than ATen's
  general broadcast contract.
- No rank enforcement on a standalone Native or JSON-decoded graph (§5) —
  rank is an import-boundary property by construction, not an omission.
- No landed backend probe (§10) — the `_fused_sdp_choice` wrapper is
  scratch; CI does not re-verify the oracle identity. The durable guarantee
  is the recipe's type; widening the recipe requires rebuilding and
  re-running the probe (`.ai/native_walk_design.md`). §12's broadcast
  configurations are hand-verified (two `verify_print` fixtures), not
  recipe-driven — see §10's own note on what changed and what did not.
- No Native4D execution path for `D > 1` (unaffected by §12) — the `D = 1`
  counterpart itself landed 2026-09-02 and is no longer "not delivered";
  see §9.

## 12. Head and batch broadcasting (2026-09-04)

`todo-ops.md`'s "Configuration and transform gaps" table tracked this since
`Batched_matmul`'s own broadcasting landing (same day): real ATen
broadcasting (equal, or one side 1) in `N`/`T`/`D`/`H`, evidenced by
`mobilenetv5_base` rejecting query `H=8` against key `H=1` at Native
construction. Scoped harder than the matmul row: `output_shape` returned
`query_shape` directly rather than deriving a genuinely broadcast shape, and
BOTH `Compute` paths (`Legacy_pixel`, kept for differential testing, and the
authoritative Region-authored `Computation`) read key/value at the output
coordinate's `H` unchanged, with no `broadcast_coord`-style clamp.

**Why broadcasting here is real ATen behavior, not a relaxed check.**
`_scaled_dot_product_attention_math` (`attention.cpp:891`) computes
`at::matmul(query, key_expanded.transpose(-2,-1) * scaling_factor)` — real
ATen's own `matmul` broadcasting, not GQA's `repeat_interleave` (which only
runs when `enable_gqa=true`; `pre_process_group_query_attention_input`
returns `key`/`value` UNCHANGED otherwise, `attention.cpp:646-648`). A
mismatched-but-broadcastable head or batch count without GQA is therefore
real, standing ATen behavior — it simply cannot use the fused/flash kernel
(`sdp_utils_cpp.h`'s `check_batch_size_and_num_heads_dense` requires equal
batch unconditionally, and equal heads unless `enable_gqa=true`), so it
silently falls to the `math` backend, whose STRUCTURE this row's arithmetic
already mirrors (§6). `enable_gqa=true` itself stays a typed rejection
(`Reject.Gqa`, unaffected) — broadcasting via ordinary `matmul` and GQA's
`repeat_interleave` are two different ATen mechanisms that happen to admit
overlapping shapes.

**What landed**

- `lib/native/ops/attention.ml`: `batch_axes = [N;T;D;H]` (the four
  batch-like axes two chained matmuls broadcast, `W`/`C` excluded — mirrors
  `Matmul.Batched_matmul.batch_axes` exactly), `broadcast_extent`/
  `broadcast_batch` (the same per-axis "equal, or one side 1" rule as
  `Matmul.Batched_matmul.broadcast_extent`), and `batch_shape ~query_shape
  ~key_shape ~value_shape` returning `(qk_batch, full_batch)`: `qk_batch` is
  `query @ key^T`'s own broadcast batch (what the mask — added BEFORE the
  second matmul — broadcasts against); `full_batch` is `qk_batch` broadcast
  again with `value` (the second matmul), the real output batch. `Ev = E`
  (`C`) and key/value's shared sequence extent (`W`) stay STRICT equality —
  neither is a batch axis, so §3's flash-oracle scope there is unchanged.
  `output_shape` now returns `full_batch` (with `C` from query), not
  `query_shape` — a real behavior change when query's own extent on a batch
  axis is 1 while the broadcast output is not. `total_work_bounded` and the
  output-numel bound both now read `full_batch`, not `query_shape`'s own —
  reading `query_shape` directly (as both did before) would under-count
  exactly the way `Batched_matmul`'s Native4D domain check under-counted
  `D` before that was fixed (`todo-ops.md`'s own note on that landing).
- `Legacy_pixel.pixel` and `Computation.program`: every operand read
  (query/key/value/mask) now reduces the output coordinate against ITS OWN
  shape via `broadcast_coord` (`Pointwise_binary.broadcast_coord` /
  `Region_context.broadcast_coord`) before `load`, the same treatment
  `Matmul.Batched_matmul.Compute` already gives its two operands — a no-op
  when every operand agrees (every existing fixture), so this is backward
  compatible by construction. `Legacy_pixel.pixel` gained a genuinely new
  `~value_shape` parameter: it used to always equal `~key_shape` (implied by
  the old strict equality), and no longer does once value can broadcast
  independently of key.
- `lib/native/shape_error.ml`/`.mli`: a new `Batch_mismatch` variant
  (mirroring `Matmul.Batched_matmul`'s `Batch_mismatch`/`Contract_mismatch`
  split) for the four broadcastable axes' genuine-incompatibility case
  ("or one must be 1" in the message), keeping the STRICT `Extent_mismatch`
  wording for `C`/`W` unchanged — the same error constructor previously
  covered both meanings, which would have been misleading once broadcasting
  existed for only some axes.
- `lib/native/region_computation.ml`: `check_output`'s signature widened
  from "compare against one fixed operand's shape" to "compare against an
  explicit expected shape" (`~expected`, not an operand record) — the SDPA
  arm can no longer assume `output_shape = query.shape` (true before this
  landing, no longer true once query/key/value can broadcast against each
  other), so it recomputes the real expected shape via
  `Attention.Sdpa.batch_shape` instead. `Rms_norm`/`Layer_norm`/`Softmax`'s
  own call sites are mechanically updated to pass `~expected:x.Tensor_sig.shape`
  (identical behavior, since output genuinely does equal `x`'s shape for
  those three).
- **Real-ATen grounding**: two hand-derived `verify_print` fixtures in
  `test/native_bridge/sdpa_test.ml` (`enable_gqa=false`, one per broadcast
  axis family — query `H=2` vs key/value `H=1`; query batch=2 vs key/value
  batch=1), both agreeing with real ATen's `math`-backend answer at the
  EXISTING `1e-5` tolerance (§10's own note on what this does and does not
  establish). `Recipe_sdpa` itself was deliberately NOT widened — same
  scoping precedent as `Batched_matmul`'s own broadcast landing
  (`.ai/matmul_softmax_design.md` §6's "not done" note).
- Hand-computed `Direct` fixtures in `test/native/sdpa_test.ml`: shape-level
  broadcast acceptance/rejection across `N`/`T`/`D`/`H` (including a
  genuine, still-rejected mismatch, `N=2` vs `N=3`), and a bit-for-bit
  correctness proof that broadcasting COMPUTES the right thing (query with
  two DIFFERENT per-head queries against one shared key/value pair,
  H-broadcast) — not merely that the shape check now accepts the input.

**Corpus effect**: `mobilenetv5_base` moves from `native_builds:false`
(blocked at Native construction on this exact `H` mismatch) all the way to
`native_builds:true` AND `native4d_converts:true` (§9's `D = 1` Sdpa
counterpart — every one of its SDPA occurrences has `D = 1`, broadcasting
only on `H`). It stops at the kernel stage on the pre-existing, unrelated
evaluation-depth ceiling (`kernel_reason: over_limit`, the same ceiling
several other large corpus models hit). `native_builds` 89→90,
`native4d_converts` 57→58; `kernel_converts`/all-three-stages unchanged
(64/44).
