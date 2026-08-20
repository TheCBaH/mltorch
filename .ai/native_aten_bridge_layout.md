# Native-ATen Bridge — Layout & the deferred structured ops

## Why this doc exists

`op_bridge.ml` dispatches a graph node to the native engine by converting each
ATen operand with `Tensor_bridge.of_aten` and running the matching native op.
That works directly for some ops and **cannot** work directly for others,
purely because of tensor *layout*. This doc records which ops fall on each side
and the plan for the ones that are deferred.

The split matters because `of_aten` (via `Aten_shape.of_aten`) is **purely
positional and right-aligned**: a rank-`r` ATen shape occupies the innermost `r`
axes of the 6-axis frame `N T D H W C`, padding the outer axes with 1. It does
*no* semantic remap — a rank-4 NCHW tensor `[N,C,H,W]` lands with `C` on frame
axis `H` and `W` on frame axis `C` (see `aten_shape.ml`'s own header).

`Verify.compare_tensors` (`lib/aten_native_verify/verify.ml`) is doubly
layout-sensitive: it requires `aten_shape = Aten_shape.to_aten ~rank native_shape`
*and* compares flat element `[i]` against `[i]`. So a native result that is
numerically correct but sitting in a different frame fails both the shape check
and the element-wise check.

## Two classes of native op

### No relayout needed — implemented

These reference frame axes through `Aten_shape.axis_of_dim` (so the reduced/
normalised axes track the *same* positional placement the data landed on), or
use rank-3/innermost layouts that already match what the native op reads:

| ATen target | Native op | Why it just works |
|---|---|---|
| `bmm.default` | `Matmul.Bmm` | rank-3 lands `H=batch, W=rows, C=cols` — exactly what `Bmm` reads |
| `mean.dim` | `Reduce.Mean` | dims via `axis_of_dim ~rank`; `kept_map` re-packs survivors so the output matches the ATen rank for both `keepdim` values |
| `rms_norm.default` | `Norm.RmsNorm` | normalises the trailing dims = the innermost frame axes; weight (rank-`k`) lands on those same axes |
| `layer_norm.default` | `Norm.LayerNorm` | same rule as `rms_norm`: the trailing `normalized_shape` axes are the innermost frame axes, and both rank-`k` affine operands land on exactly those |
| `native_layer_norm.default` | `Norm.LayerNorm` | the same node from the decomposed spelling; the extra `mean`/`rstd` tuple elements are dropped, not laid out |
| `view.default` | `Reshape.Reshape` | a contiguous reshape preserves flat row-major order, and `of_aten` inputs are already ATen-row-major, so the target is just `size` right-aligned — no surrounding permutes. (A whole-graph flow feeding a permuted native frame would need them, per the user's "Reshape between Transpose nodes".) |
| `unbind.int` | `Split.Unbind` | `dim` via `axis_of_dim ~rank`, so the selected axis is the one the data actually landed on; dropping it re-packs the survivors right-aligned by the *same* rule `mean(keepdim=false)` uses, which is the ATen output rank. The rank must be read from the ATen tensor before `of_aten`, since the frame cannot recover it. |
| `scaled_dot_product_attention.default` | `Attention.Sdpa` | rank-4 lands `D=batch, H=heads, W=sequence, C=head_dim` — exactly what `Sdpa` reads, no different from `Bmm`'s rank-3 case above (op8-impl.md F7). |

(`rms_norm` also needed one unrelated fix on the ATen side — the interp decode
generator had no `float?` case, so `aten.rms_norm`'s optional `eps` left it
without a dispatch arm. Adding `float_opt_ptr` to `interp_decode.ml` + the
`Optional (Base Float)` arm to `aten_decode_gen.ml` regenerates the dispatch.
`map_type` and the C shim already lowered `float?` to a `double*`/`std::nullopt`,
so only the interp-side decode was missing.)

### Relayout needed — DEFERRED

Native conv/pool/linear hard-bind semantic roles to fixed frame axes:
`Conv2d`/`MaxPool2d`/`AvgPool2d` window over `H`/`W` and (conv) reduce channels
over `C`; the conv/linear *weight* is read at `N = out-channel`. But:

* An ATen NCHW input lands channels on `H` (not `C`), so native conv/pool would
  window and reduce over the wrong axes — wrong values, not just a wrong label.
* The conv weight `[Cout,Cin,Kh,Kw]` must become `[Cout,1,1,Kh,Kw,Cin]` with
  `Cout` on the **`N`** axis — which `of_aten`'s right-alignment can **never**
  produce, because a rank-≤5 tensor never reaches `N`.
* Even with correct inputs, the native output is NHWC while the ATen reference
  is NCHW, so the comparison fails on both shape and flat order.

| ATen target | Native op | Transposes required |
|---|---|---|
| `conv2d` | `Conv2d` | input NCHW→NHWC, weight OIHW→`[Cout,1,1,Kh,Kw,Cin/groups]`, output NHWC→NCHW |
| `conv2d.padding` | `Conv2d_padding` | same as `Conv2d`; the op keeps the ATen string-padding overload distinct and lowers internally to concrete `Conv2d` params |
| `convolution` | `Convolution` | same as `Conv2d`; the op keeps the ATen lower-level overload distinct, including `transposed` and `output_padding`, and lowers supported non-transposed 2D calls internally |
| `max_pool2d*`, `avg_pool2d` | `Max/AvgPool2d` | input NCHW→NHWC, output NHWC→NCHW (no weight) |
| `max_pool2d_with_indices` | `Max_pool2d_with_indices` | same relayout as `max_pool2d`, but two outputs: the pooled values (relayout'd out) and the argmax indices. The indices output is materialised (via the `value_of_index` bridge + an argmax reduction) and routed into a `Discard` sink — dead in inference (see native_multi_output_design.md) |
| `addmm`/`linear` | `Linear` | weight only (`[N,In]` input already lands `In` on `C`); `addmm` carries `[In,Out]`, `linear` carries `[Out,In]` — different permutations |
| `_native_batch_norm_legit_no_training` | `Norm.BatchNorm` | input NCHW→NHWC, output NHWC→NCHW; the per-channel `[C]` weight/bias/running_mean/running_var vectors already land on `C` (no weight permute). Only out0 is emitted — the size-`[0]` save_mean/save_invstd are dropped (see native_multi_output_design.md) |

## Implemented: relayout as explicit Permute nodes in a graph

`op_bridge` now returns a `Graph_ir.graph` (not eager tensors).  The graph for
a relayout op is named `"<op>_relayout"` and contains explicit `Permute` nodes
wrapping the core op — exactly the plan above, now landed.

```
conv2d_relayout graph:
  x_in  = input (right-aligned NCHW)
  w_in  = input (right-aligned OIHW)
  [b_in = input (bias, optional)]
  x'    = Permute(x_in, perm_nchw_to_nhwc)
  w'    = Permute(w_in, perm_oihw_to_conv_weight)
  y'    = Conv2d(x', w', [b_in], params)
  y     = Permute(y', perm_nhwc_to_nchw)
  outputs: [y]
```

`conv2d_padding_relayout` has the same surrounding permutes, but its core node is
`Conv2d_padding`. That keeps the graph/API mapping one-to-one with
`torch.ops.aten.conv2d.padding`; internally the op derives kernel and
input-channel counts from the relaid weight shape, maps `"valid"` to zero pads,
and maps `"same"` to PyTorch's left/right padding convention before reusing the
`Conv2d` computation.

`convolution_relayout` also uses the same surrounding permutes, with a core
`Convolution` node. Its graph/API surface keeps
`torch.ops.aten.convolution.default` one-to-one, including `transposed` and
`output_padding`; internally it reuses `Conv2d` for the forward 2D path and
evaluates transposed 2D convolution with the same native weight relayout.

The six permutations are defined as module-level constants in `op_bridge.ml`.
All are full 6-axis bijections so `Permute.Compute` never hits `Not_found`.

| Permutation | From | To |
|-------------|------|----|
| `perm_nchw_to_nhwc` | NCHW right-aligned (D=N,H=C,W=H,C=W) | channel-last in the same outer frame (D=N,H=H,W=W,C=C) |
| `perm_nhwc_to_nchw` | channel-last in the same outer frame | NCHW right-aligned (inverse) |
| `perm_oihw_to_conv_weight` | OIHW right-aligned (D=Cout,H=Cin,W=Kh,C=Kw) | native weight (N=Cout,H=Kh,W=Kw,C=Cin) |
| `perm_addmm_weight` | [In,Out] rank-2 (W=In,C=Out) | native weight (N=Out,C=In) |
| `perm_linear_weight` | [Out,In] rank-2 (W=Out,C=In) | native weight (N=Out,C=In) |

Activation/pool relayouts leave the outer frame axes (`N/T/D`) intact and only
move `CHW` to `HWC` (and back). `perm_oihw_to_conv_weight` is separate: it must
move `Cout` from right-aligned axis `D` to native weight axis `N`, because
`Conv2d.output_shape` reads output channels from weight `N`.

Graph pretty-printing elides identity axis mappings, so a stored full permutation
like `[(N,N); (T,T); (D,D); (H,W); (W,C); (C,H)]` prints as
`perm=[H<-W, W<-C, C<-H]`.

Callers evaluate the returned graph via `Eval_direct.run graph ~inputs:bindings`
and extract outputs from `graph.outputs`.

Op-config params (`kernel`/`stride`/`pad` → `Op_config.Hw`/`Pos`/`Nonneg`) are
validated at the boundary with `try … with Invalid_argument` so bad args become
`Some (Error msg)` instead of an uncaught exception.

## Whole-graph import: `bin/native_graph print` / `eval`

`native_graph` now uses the pure `Native_interp` importer rather than the
ATen `Op_bridge` report. `print --pt2 MODEL.pt2` lowers the root exported graph
to one `Graph_ir.graph`, then prints its complete structure with each tensor
definition/reference and node id annotated inline by PT2 graph path, SSA name,
target, and captured payload target where applicable. Nested calls print their
positional wiring explicitly as `parent_arg -> subgraph_input` so a source such
as a captured weight can be traced to its inner use. `Graph_ir.pp_with
~printer` accepts generic `Printer.t` callbacks, so the native IR remains
name-free while PT2 diagnostics remain human-identifiable.

`eval --pt2 MODEL.pt2 --input INPUT.pt` uses that same import and sidecar to
load the single user input and every captured parameter/buffer, then runs
`Eval_direct`. The current static float32 implementation targets the ResNet-18
operator set and does not yet enforce dynamic-shape guards. The ResNet cram is
gated on `PT2_DATA` like the other real-model crams (`make pt2.runtest`).

## Two importers, one set of layouts (Group 2)

`Op_bridge` reads **live ATen tensors**; `Native_interp` reads **serialized
metadata**. They are separate code, they build the same `Graph_ir`, and the rule
that matters is that they must accept and reject the same node. The five Group-2
functional overloads — `conv2d.default`, `conv2d.padding`, `linear.default`,
`max_pool2d.default`, `rms_norm.default` — now have an arm in both.

**No downloadable model reaches any of them.** resnet18, mobilenet_v2/v3_small,
efficientnet_b0 and vit_b_32 are all exported post-decomposition and carry
`convolution.default`, `addmm.default` and `max_pool2d_with_indices.default`
instead. So no `interp_*_cram.t` covers these arms and `make pt2.runtest` is not
a meaningful gate for them; hand-built fixtures in `test/native_interp/`, the
generated walks, and `test/me_group2_cram.t` are the whole coverage.

### The permutations are shared, and two of them are near-identical on purpose

`Native_interp` defines the same constants `op_bridge.ml` does, with the same
names and the same meanings — including **both** rank-2 weight permutations.
`perm_addmm_weight` and `perm_linear_weight` are *not* a rename of each other:
`addmm`'s `mat2` is `[In, Out]` and `linear`'s `weight` is `[Out, In]`, the
transpose. Applying one where the other belongs produces a graph that still
builds for a square weight and computes the wrong thing — which is why every
`linear` fixture uses a non-square weight.

`in_channels = weight.C * groups` has three definitions
(`Op_bridge.make_conv2d_params`, `Conv2d_padding.to_conv2d_params`,
`Native_interp.conv_in_channels`). The third is **computed in `int64` and
bounded against `Kernel.Limits.Hard.extent` before narrowing**: both factors are
model-supplied, `lib/native_interp` is js_of_ocaml-reachable, and 0x8000_0000 is
exactly the 32-bit boundary. Bounding the factors would not catch it — the
product is its own quantity. See `.ai/js_backends_design.md`.

### `max_pool2d`: dilation and ceil_mode are REFUSED, not carried

`Pool.MaxPool2d.params` has a field for neither. Both were previously decoded by
neither importer — not approximated, *dropped*, so a serialized
`dilation=[2,2]` computed an undilated pool under the dilated name. Both now
reject, with one check (`Native_interp.pool_params`,
`Op_bridge.reject_pool_extras`) shared by the functional overload and by
`max_pool2d_with_indices.default`.

Rejection rather than an IR extension is the deliberate choice: an extension is
warranted only on a measured need, and since no reachable model serialises this
target at all there is nothing to measure. Revisit when `avg_pool2d.default`
forces the same question for `ceil_mode` and `count_include_pad`.

### `conv2d.padding`: the mode is carried unresolved

The importer matches `"valid"`/`"same"` itself and reports anything else as
`` `Unsupported_padding_mode ``. It must **not** call
`Conv.Conv2d_padding.padding_of_string`, which `invalid_arg`s on an unknown mode
— that string is model data. Nor does it resolve the mode:
`to_conv2d_params` is the single definition shared by shape inference, `Compute`
and Native4D, and resolving in the importer would make a fourth. `"same"` with
`stride > 1` therefore fails in **shape inference**, not in the importer, and
the fixtures pin where.

`same_padding` splits an odd total unevenly (`total/2` before,
`total - total/2` after). Nothing else pinned which side got the extra cell:
Direct-vs-Symbolic agreement resolves through the same function, and an
output-shape check is invariant under a reversal. It is pinned against
hand-computed values in `test/native/compute_test.ml`.

> **Stale constraint, now lifted.** This section used to add "and ATen reaches
> `constant_pad_nd`, which this repository's minimal static-dispatch build does
> not carry", and that is why `walk_meta`'s `conv2d_padding` kernels are all
> odd. Binding `aten.pad.default` put `native/PadNd.cpp` into the archive, so
> `constant_pad_nd` is real now and an even `conv2d_padding` kernel would
> execute. The walk was not widened in that change — the hand-computed fixture
> is still the evidence — but the reason for the restriction no longer exists,
> so do not re-derive it from the old sentence.

### `rms_norm`: an optional weight, and `normalized_shape` is validated

`Graph_ir`'s `Rms_norm` carries `weight : Tensor_ref.t option` and Native4D
reads the option. `Op_bridge` used to synthesize a ones tensor for an absent
weight; it no longer does. The numeric result is unchanged — multiplying by ones
is what the option's absence means — but the graph structure was not, and the
old shape made the two importers build different graphs for the same node while
leaving Native4D's optional-weight arm unreachable from the bridge.

Both importers now validate `normalized_shape`'s **extents** against the input's
trailing extents, and check `k <= rank` before computing the axes. The bridge
previously used only the length: `[2]` on a `[2,3]` input produced the same
answer as `[3]`, `[3,2]` the same as `[2,3]`, and an over-long shape reached
`trailing_axes`, whose `List.filteri` lower bound goes negative and keeps every
axis — so it normalized over the whole tensor. Four silently wrong answers,
now four typed rejections.

The float32-epsilon default lives in `Norm.RmsNorm.default_eps`, read by both.

### `layer_norm`: two optional operands, `cudnn_enable`, and a required-but-defaulted `eps`

`layer_norm.default` reuses `rms_norm`'s axis derivation wholesale — the shared
`normalized_dims` checks `1 <= k <= rank` *and* compares the trailing extents,
then hands back `trailing_axes` — and differs from it in three ways worth
recording, because each is a place the two rows must not be copy-edited into
each other.

**Both affine operands are optional, and neither importer materialises one.**
`Graph_ir`'s `Layer_norm` carries `weight` and `bias` as `Tensor_ref.t option`
and `Eval_op` fills the absent ones (`fill 1.` / `fill 0.`), for the reason the
`rms_norm` section above states: synthesizing a constant in one importer builds
a structurally different graph from the other's for the same node. There are
**four** states, not two, and "bias but no weight" is the one no exported model
produces — it is dispatched explicitly in both arms and in
`test/native_bridge_test.ml` rather than folded into a paired encoding, which is
where a two-state assumption would get it wrong.

An *entirely absent* `Tensor?` key is read as `None` by both native arms
(`optional_tensor_present` in the bridge, the same test in `Native_interp`).
`Interp_decode.tensor_arg` on the generated ATen path does **not**: it reports
an absent key as `` `Missing_argument ``, unlike every other optional decoder in
that file. No corpus node hits it, but it means the ATen oracle cannot express
the omitted spelling, so the `omitted:` line in the bridge fixture reads
`aten: missing required argument "weight"` and is a decoder limitation, not a
finding about the arm. Recorded rather than fixed: the decoder is shared with
every generated arm.

**`eps` is a required float with a schema default of `1e-05`**, decoded as
`float_arg ~default:1e-05` — never `eps_arg`. `rms_norm`'s `eps` is a `float?`
whose *absence* means "ATen picks", which is why `Norm.RmsNorm.default_eps`
exists; reading one with the other's decoder is how a null epsilon comes to be
read as zero. `native_layer_norm.default`'s `eps` is required with no default at
all, a third spelling of the same argument.

**`cudnn_enable` is decoded and then deliberately discarded**, and accepted at
both values. ATen's own composite is
`layer_norm_symint(..., bool /* cudnn_enable, deprecated */)`: it names the
parameter in a comment and drops it, computing `native_layer_norm` either way.
So the native contract is backend-independent by construction, and rejecting
`true` would reject the schema's own default and with it almost every real node.
It is still decoded rather than ignored — a non-boolean there is a malformed
node, and the decode is what says so. This is the one argument in this tree
where discarding is the faithful reading rather than the `alpha`-shaped bug.

`normalized_shape`'s diagnostics now name the op (`Normalized_rank` /
`Normalized_shape` carry an `op : Norm.Target.t`), since the rows are shared by
three targets and a message reading `rms_norm:` on a `layer_norm` node is a
wrong answer about which node failed.

### `native_layer_norm`: the same body, three differences, and 148 real nodes

`layer_norm.default` and `native_layer_norm.default` are **one arm** in each
importer, not two. They differ in exactly three things — `eps` is required with
no schema default, there is no `cudnn_enable`, and the return is a 3-tuple — and
in nothing else: same axis derivation, same `normalized_shape` validation, same
affine handling, same node built. Two copies of that body would be two copies of
a shape rule free to drift, so the arm names both targets explicitly (no
fallthrough) and every diagnostic under it carries the resolved target, exactly
as `view`/`_unsafe_view` already does.

**This is the row that carries model coverage.** `layer_norm.default` occurs in
no reachable graph — every ExportedProgram in the corpus lowers it to the
decomposed target before writing `model.json`, recording the functional one only
in `metadata.from_node` — while `native_layer_norm.default` occurs 148 times
across `vit_b_16`, `vit_b_32`, `vit_l_16` and `vit_l_32`, all in one
configuration: `normalized_shape` of arity 1 (`[768]` / `[1024]`), `eps` spelled
`1e-06`, both affine operands present, and both statistics dead.

For `vit_b_32` that is 25 of 839 nodes and takes the importer from 559 to 584.
It does **not** move any cram gate yet: the first `expand.default` is node 3 and
the first `native_layer_norm.default` is node 7, so the import still stops in the
same place. Thirteen targets remain (`expand.default`, `select.int`,
`mul.Scalar`, `bmm.default`, `logical_not.default`, `unsqueeze.default`,
`squeeze.dims`, `_softmax.default`, `eq.Scalar`, `any.dim`, `full_like.default`,
`where.self`, `gelu.default`, plus one `cat.default`). "Removes one of fourteen
blockers" is the claim; "vit is importable" is not.

The dead `mean`/`rstd` are a dead-output shape neither existing rule covered —
see `native_multi_output_design.md` §3, which also states why the liveness check
lives in the importer and not in the single-node bridge.

### `view.default` / `_unsafe_view.default`: one shared body, `Identical` after materialization

Both overloads decode identically (`self`, `size`) and legalize to the same
`Reshape` node — `_unsafe_view` differs from `view` only in ATen's own aliasing
contract (it skips the alias-safety check `view` performs, since the exporter
has already proven the reshape is the only consumer), which has no native
counterpart to differ over. Both importers dispatch the two targets through one
match arm (`op_bridge.ml`, `native_interp.ml`) rather than two copies of an
identical body — CLAUDE.md's rule against restating a shape/build rule a
second time free to drift, applied to dispatch rather than to a shape formula.
This is still exact-target dispatch in `op3.md`'s sense: both targets are
named explicitly, there is no fallthrough case, and every diagnostic below the
match reads `node.target` (`Op_bridge`) or the resolved tensor name
(`Native_interp`), so a failure still says which overload produced it.

**The correspondence is `Identical`, but only AFTER materialization, never
alias-identical.** ATen's `view`/`_unsafe_view` may return a tensor that
shares storage with `self` — mutating one would mutate the other. Native graph
edges are immutable values with no aliasing relation at all; two edges either
hold the same values or they don't. `Reshape`'s own header states the
reinterpretation rule (row-major flat offset preserved under the new shape);
this note is only that "identical" in this codebase's verification sense
(`Verify.verify_node`'s element-wise comparison) never means alias-identical,
and no test here asserts storage sharing.

`Reshape` has one output and `Argument.Tensor`, so `Native_interp`'s generic
`output_names`/provenance path is correct unmodified for both overloads — the
per-op override in `materialized_output_names` exists only for the two
multi-output arms (`_native_batch_norm_legit_no_training`,
`max_pool2d_with_indices`), neither of which this is.

## `aten.pad.default` (Group 6): the serialized list is REVERSED

`pad(Tensor self, SymInt[] pad, str mode="constant", float? value=None)`
serializes its amounts as a flat list of pairs, **innermost dimension first**:
`[left, right, top, bottom, …]`. Pair *k* is therefore ATen dim `rank - 1 - k`,
and the frame axis is that dim through `Aten_shape`'s right-aligned embedding.

Three things follow, and none of them is optional.

**The rank is load-bearing twice.** It converts each dim to a frame axis, *and*
it decides which dims the list addresses at all. With a wrong rank the same list
pads different axes and still produces a perfectly plausible graph — no shape
check catches it, because the shape it produces is the shape that configuration
would legitimately have. Both arms therefore read the rank before conversion:
`Op_bridge` from the live ATen tensor (`aten_rank`, before `of_aten` right-aligns
it away), `Native_interp` from the serialized `TensorMeta`.

**The un-reversal happens once**, in `Pad.Pad.params_of_aten`, which both
importers call. Its output is a list of `(Axis.t, {before; after})` in `Axis.all`
order, so nothing downstream — shape rule, `Compute`, JSON, Model Explorer — ever
sees the serialized ordering. That is the payload type doing the work: a raw
`int list` in the IR would leave every consumer free to re-derive the convention,
and one of them would get it wrong.

**The accepted contract is ATen's own, checked in ATen's order** (`PadNd.cpp`'s
`_pad_enum_symint`):

| rule | outcome |
|---|---|
| even length, at most `rank` pairs | else `` `Odd_length `` / `` `Too_many_pairs `` |
| `constant`: `value` absent means `0.0` | the fill, narrowed to f32 by the builder |
| non-constant with a non-zero `value` | `` `Value_with_non_constant_mode `` — ATen's test is `!value.has_value() \|\| *value == 0`, so an explicit `0.0` is fine |
| `reflect` at (1 pair, rank 2–3), (2, rank 3–4), (3, rank 4–5) | accepted |
| `reflect` at any other (pairs, rank) | `` `Unsupported_rank `` |
| `replicate`, `circular`, anything else | `` `Unsupported_mode `` |

That fourth row is the one worth stating a reason for: ATen dispatches reflect to
per-rank kernels and *rejects* the combinations without one. Accepting a pair
ATen refuses would turn a boundary into a **walk mismatch**, which reads as a
wrong answer rather than as an unsupported configuration.

**Negative amounts crop, and are supported.** The only Native-specific refusal is
the resulting extent, and it lives in `Pad.output_shape`
(`Shape_error.Pad`, fault `` `Empty ``) rather than in either importer — so
`Graph_builder` and a JSON-decoded graph meet it too. `reflect`'s width rule
(each *positive* side below the extent it mirrors) is checked there for the same
reason.

**Binding this target changed the C++ archive**, which no previous op row did:
`native/PadNd.cpp`, `ReflectionPad.cpp`, `ReplicationPadding.cpp` and
`cpu/PaddingKernel.cpp` joined `lib/aten/build_archive.sh`, the
`constant_pad_nd` throwing stub was removed as now-duplicate, and a quantized
tensor-factory leaf was stubbed in its place. `_pad_enum_symint`'s mode switch is
straight-line, so replicate and circular link even though the native engine
refuses them — the interpreter runs real ATen, where the mode is just an
argument. See `.ai/aten_core_build.md`.

## `aten.scaled_dot_product_attention.default` (Group 8): rank is erased, so every rank rule is an importer rule

`unbind.int`'s row above and `pad`'s section both note, per op, that the rank
must be read before conversion. `Attention.Sdpa` is the row that promotes this
from a per-op remark to a stated rule, because it is the first op where the
same erasure applies to an **optional** operand and the trap is concrete
rather than hypothetical: `Aten_shape.of_aten` right-aligns and is lossy in
exactly one direction (`aten_shape.ml:4-6` — recovering the rank back out
needs it, "since the frame alone can't recover it"), and `Tensor_sig.t`
retains only the folded `Vec6.shape`, no rank field. Concretely, these three
attention-mask shapes fold to the **same** Native shape:

```text
rank 2  [Wq, Wk]        ⟶  D=1 H=1 W=Wq C=Wk
rank 3  [1, Wq, Wk]     ⟶  D=1 H=1 W=Wq C=Wk
rank 4  [1, 1, Wq, Wk]  ⟶  D=1 H=1 W=Wq C=Wk
```

The first and third are exactly what ATen's flash-attention CPU kernel admits
(a rank-2 or rank-4 mask, each axis either the real extent or 1); the middle
one is not, and `Attention.Sdpa.output_shape` cannot tell them apart — nor
should it try to. Its checks stay to what the six-axis frame can express
(per-axis extent agreement, broadcast-or-equal against `score_shape`); rank —
Q/K/V at exactly 4, the mask at 2 or 4 — is enforced by **both** importers,
before conversion: `Op_bridge` from `Aten_tensor.shape` (the same array
`Tensor_bridge.of_aten` reads), `Native_interp` from the serialized
`TensorMeta.sizes` length. The rank-3 fixture above is what proves the
ordering — it passes every frame-level check and is caught only if raw rank
is read first (`test/native_bridge_test.ml`'s and `sdpa_test.ml`'s "rank is
checked on the declared metadata, not the normalized frame" cases both carry
it deliberately).

**A standalone Native or JSON-decoded graph carries no ATen rank at all.** A
hand-built graph via `Graph_builder` (or one read back from JSON) can hold a
mask no admissible ATen node could have produced, and it still evaluates
soundly — the frame-level broadcast check in `Compute` is what the
computation actually depends on, not a rank ATen never gave it. This is a
property of the boundary, not a gap: rank is an *import*-time fact, and
nothing downstream of `of_aten` has anywhere to keep it.

**The typed-rejection boundary belongs to the op, not to either importer.**
`Attention.Sdpa.Reject` (`lib/native/ops/attention.ml`) is shared by both
arms for `Op_config.Bad`'s reason — dropout/causal/GQA/boolean-mask/scale
rejections must read identically from both paths, or the two importers
silently diverge on what "supported" means for the same node. Two of the
frame-level checks (`Attention.Sdpa.output_shape`'s `` `Extent_mismatch ``
on `C`, and its `` `Mask_shape ``) are flash-oracle boundaries rather than
mathematical ones (op8-impl.md F4): ATen does not error on `Ev <> E` or an
inadmissible mask shape, it silently falls back to the `math` backend, which
is why they are rejections here rather than supported cases with a wider
tolerance.

**No relayout, same as `Bmm`** (added to the table above): a rank-4 SDPA
operand `[B, heads, S, E]` lands `D=batch, H=heads, W=sequence, C=head_dim`,
exactly the axes `Compute` reads.

## `aten.slice.Tensor` (Group 6): one resolver, two extents

`slice.Tensor(Tensor self, int dim=0, SymInt? start=None, SymInt? end=None,
SymInt step=1)`. Unlike `pad`, the serialized spelling has no ordering trap — but
it has four *conventions*, and every one of them is ATen's rather than the
engine's:

| spelling | meaning |
|---|---|
| absent or explicit null `start` / `end` | `0` and `extent` |
| a negative bound | `+ extent`, then clamped |
| `start` past the extent, or `end` past it | **clamped**, not rejected |
| `end < start` | `end` becomes `start` (an empty result) |
| `step < 1` | refused, as ATen refuses it |

All five live in **one** helper, `Aten_shape.resolve_slice`, which both importers
call. What differs between the two arms is nothing but where the extent comes
from: the live ATen tensor on the bridge, the serialized `TensorMeta` in
`Native_interp`. That is the entire reason the helper is shared rather than
written twice — two normalizations of a five-rule convention is exactly the drift
this layout doc exists to prevent.

**Clamping is accept-and-narrow; emptiness is reject.** Both are ATen-faithful
and only the second is a Native limitation, so only the second appears in a
coverage report. And the emptiness check is not in the resolver: it is a fact
about the extent, so it belongs to `Split.Slice.output_shape`
(`Shape_error.Slice`, fault `` `Empty ``) where `Graph_builder` and a
JSON-decoded graph meet it too — the same layering `pad`'s crop follows.

**`SymInt` arguments arrive here first.** `start`/`end`/`step` are the engine's
first bound arguments that can be spelled `as_sym_int`, and the rule is: a
**resolved** `Sym_int (Int n)` is a value spelled differently and is accepted; a
**named** `Sym_int (Name s)` is an unresolved symbol and is refused *with the
symbol*. That mirrors the `` `Bad_dimension { fault = `Symbolic } `` rule tensor
metadata already applied, and it is implemented once —
`Interp_decode.sym_int_value` for the generated ATen path and for `Op_bridge`,
`Native_interp.sym_int_value` for the importer, which cannot share a module
because it reports through the escape token. Both are asserted, on both
spellings, in `test/native_bridge_test.ml` and
`test/native_interp/slice_test.ml`. Before this target the three decoders agreed
only by refusing every `Sym_int` alike, which satisfied the requirement by
accident.

**Binding it needed no C++ change** and no new codegen capability — but only
because commit `4f37a9c` had already closed the `int?`/`SymInt?` decode gap and
lifted the OCaml-keyword escape into `Aten_gen.Aten_ident`. Without the latter
the generated arm is `let* end = …`, which does not parse; the emitted source
now binds `end_`, and that is checked by reading
`_build/default/lib/interp/interp_dispatch.ml`, not inferred from a green build.
