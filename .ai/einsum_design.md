# `einsum.default`: a legalization onto `Permute`/`Batched_matmul`

**Status: landed, 2026-09-10.** `torch.ops.aten.einsum.default` occurs 20
times in the 100-model corpus, all in `mvitv2_tiny`
(`timm.models.mvitv2.MultiScaleAttention`'s decomposed relative-position
attention, `add_decomposed_rel_pos`), split evenly across two equation
strings. No dedicated `Graph_ir` node exists for this op — both equations
legalize onto a `Permute`/`Batched_matmul` chain, reusing ops the
`Batched_matmul` broadcast landing (`.ai/matmul_softmax_design.md`) already
verified against real ATen.

## 1. The two occurrences

Inspecting the corpus JSON directly
(`modules/devcontainer.pytorch-image-models/models/mvitv2_tiny/models/model.json`)
shows exactly two equation strings, ten occurrences each:

- `"byhwc,hkc->byhwk"` — `self` rank 5 `[b,y,h,w,c]`, `other` rank 3
  `[h,k,c]`, the shared index `h` appearing in both operands and the output.
- `"byhwc,wkc->byhwk"` — same `self`, `other` rank 3 `[w,k,c]`, the shared
  index `w` instead.

In both, `c` is summed (present in both operands, absent from the output)
and the operand NOT holding the shared index contributes nothing else to
the output beyond its own free axis (`k`). This is exactly MViT's own
`einsum('bhwc,hkc->bhwk', ...)`/`einsum('bhwc,wkc->bhwk', ...)` pattern
(with a leading `y`/window axis folded in) for adding a decomposed relative
position bias to attention scores — a small, closed set of equations, not
an arbitrary contraction language.

## 2. Frame-position derivation

`Aten_shape.of_aten`'s positional right-alignment puts `self`'s rank-5
`[b,y,h,w,c]` on `[T,D,H,W,C]` and `other`'s rank-3 `[h,k,c]` (or
`[w,k,c]`) on `[H,W,C]`. Matmul.Batched_matmul's own convention
(`.ai/matmul_softmax_design.md`) is: `input`'s free axis at `W`, contract
axis at `C`; `mat2`'s contract axis at `W`, free axis at `C`; batch axes
`N`/`T`/`D`/`H` broadcast per-axis (equal, or one side 1) and pass through
to the output.

**`byhwc,hkc->byhwk` (`Shared_h`):** `self` frame is `T=b,D=y,H=h,W=w,C=c`
— already exactly `Batched_matmul`'s own `input` shape (batch `T,D,H`, free
`W`, contract `C`). `other` frame is `H=h,W=k,C=c` — its own free (`k`) and
contract (`c`) axes are SWAPPED relative to what `Batched_matmul` expects
for `mat2` (contract at `W`, free at `C`). Swapping `other`'s own `W`/`C`
axes (`Permute`, a transpose of `other`'s last two ATen dims) fixes this:
after the swap, `other` is `H=h,W=c,C=k`, matching `mat2`'s convention
exactly, `H=h` a genuine shared batch axis with `self`. No further permute
needed — `Batched_matmul`'s own output convention (self's `W` and `C`
untouched except `C` overwritten by `mat2`'s `C`) already produces
`T=b,D=y,H=h,W=w,C=k`, the required output frame.

**`byhwc,wkc->byhwk` (`Shared_w`):** the shared index (`w`) sits at
`self`'s own `W`, not a `Batched_matmul` batch axis. Swapping `self`'s own
`H`/`W` first moves it there (`T=b,D=y,H=w,W=h,C=c`); `other`'s own
`[w,k,c]` needs the identical `W`/`C` swap as `Shared_h`'s `other`
(`H=w,W=c,C=k`). Feeding these into `Batched_matmul` gives a raw output of
`T=b,D=y,H=w,W=h,C=k` — `self`'s free axis (`h`) rode along at frame `W`
instead of `H` the whole time, so the RESULT needs the same `H`/`W` swap
applied once more to land at the required `T=b,D=y,H=h,W=w,C=k`.

Both derivations are verified independently against hand-computed matrix
arithmetic in `test/native/graph_direct_einsum_test.ml` (two small 2x2
matmuls per plan, not the digit-encoded self-describing values most other
Direct fixtures use, so a wrong axis pairing produces a plainly wrong
number) — both matched the derivation above exactly on the first attempt.
`test/native_bridge/einsum_dispatch_test.ml` additionally runs both plans
over the SAME pair of tensors and confirms they disagree (proving the plan
selection, not the data, drives the contraction), and confirms one entry
coincides where the derivation predicts it should.

## 3. What was built

- `Aten_shape.Einsum` (`lib/native/aten_shape.ml`/`.mli`): `type plan =
  Shared_h | Shared_w` and `of_equation : string -> plan option`,
  recognizing exactly the two literal strings above. Shared by both
  importers so they can't drift on which equations they accept — the same
  reasoning `Index_list`'s own "mirrors exactly" comment gives.
- `Graph_builder.einsum` (`lib/native/graph_builder.ml`/`.mli`): builds the
  actual `Permute`/`Batched_matmul` chain for a given `plan`, `self`,
  `other`. Not a new `Graph_ir` node — `Permute`/`Batched_matmul` already
  exist and are already fuzzed/verified on their own, so this is pure graph
  construction, the same "legalize onto an existing op, no new dispatch
  surface" pattern `tile.default`'s own landing onto `Repeat` used.
- Both importers (`lib/native_aten_bridge/op_bridge_linalg.ml`,
  `lib/native_interp/native_interp_lower_compute.ml`): decode `equation`
  and the two-entry `tensors` list, check operand ranks are 5 then 3, call
  `Aten_shape.Einsum.of_equation`, and either build the graph or reject with
  a new typed `Einsum_unsupported { equation; ranks }` (mirrored exactly
  between `Op_bridge_error` and `Native_interp_error`, per the `Index_list`
  precedent) naming the equation string and the ranks actually found.
- No ATen C binding exists for this op (`Tensor[]` plus a string argument
  has no `lib/aten_gen` support, the same gap `addcmul.default`/
  `group_norm.default`/`index.Tensor` hit), so verification is hand-derived
  rather than `Interp_verify`-compared — the same choice already made for
  those ops.

## 4. What was deliberately not built

- **No dedicated `Graph_ir.Einsum` node.** Both evidenced equations reduce
  fully to existing, already-verified ops; inventing a new node would
  duplicate `Batched_matmul`'s own shape/compute logic for no corpus
  benefit.
- **No general einsum equation parser.** `of_equation` is a closed match
  over two literal strings, not a string-parsing engine — any other
  equation is a typed rejection naming what was found, not a best-effort
  guess.
- **No shared-axis extent validation beyond what `Batched_matmul` already
  does.** Real ATen's einsum requires an exact match on a repeated
  (contracted-as-batch) index; `Batched_matmul`'s own per-axis broadcast
  (`equal, or one side 1`) is slightly more permissive. Every real corpus
  occurrence is exported from a working model, so its shared axis is
  already exactly matched or one side is a genuine broadcastable 1 —
  `Batched_matmul`'s own shape check already rejects a genuine mismatch
  with its own typed error; adding a second, stricter check here would
  duplicate that with no evidenced need.
