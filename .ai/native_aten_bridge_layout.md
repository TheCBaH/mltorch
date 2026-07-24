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
| `view.default` | `Reshape.Reshape` | a contiguous reshape preserves flat row-major order, and `of_aten` inputs are already ATen-row-major, so the target is just `size` right-aligned — no surrounding permutes. (A whole-graph flow feeding a permuted native frame would need them, per the user's "Reshape between Transpose nodes".) |

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

## Whole-graph print: `bin/native_graph print` (`lib/native_graph`)

`bin/native_graph` is a general-purpose, cmdliner-based CLI for the native
engine's graph representation, meant to grow more subcommands over time. Its
first subcommand, `print`, walks every node of a real `.pt2` model's exported
graph (in program order) and, for each one, prints `Op_bridge.dispatch`'s
native translation via `Graph_ir.pp` — i.e. this table's "implemented" ops
rendered node-by-node across a whole real model, not just the hand-built
single-node graphs in `test/native_bridge_test.ml`. The model is sourced via
`--pt2 MODEL.pt2` (an option, not a bare positional, so other graph sources
can land as sibling options later); `-v`/`--verbose` additionally prints each
node's source ATen target/name before its translated graph — without it, only
the translated graphs themselves print, with no conversion narration.

It does not stitch the per-node graphs into one combined `Graph_ir.graph`:
each node gets its own independent relayout graph (as `Op_bridge` already
returns), printed in sequence, labeled (under `--verbose`) by the node's ATen
target. Graph inputs are synthesized (uninitialised, correctly shaped/typed,
from the exported graph's own `tensor_values` metadata) rather than loaded
from a real sample image — only shapes need to flow through the conversion.
Parameters/buffers are still loaded for real via
`Interp.load_referenced_params` (extracted from `Interp.run` for this reuse),
since relayout reads their actual shapes.

The walk stops at the first node whose target has no `Op_bridge` mapping or
whose decode/shape-validation fails — a coverage report, not a best-effort
partial dump. `test/native_graph_cram.t` pins `native_graph print --verbose`'s
full output over resnet18 — every converted node's ATen target/name and its
translated native graph, in program order. resnet18 now converts **every node
end to end** — the whole graph, ending at the fc `addmm`, with no unsupported
op or conversion failure. (Reaching `addmm` required `Tensor_bridge.of_aten`
to materialize the non-contiguous transposed fc weight; it now does, via
ATen's `contiguous`.) The cram is gated on `PT2_DATA` like the other
real-model crams (`make pt2.runtest`).
