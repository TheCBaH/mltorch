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
| `convolution`/`conv2d` | `Conv2d` | input NCHW→NHWC, weight OIHW→`[Cout,1,1,Kh,Kw,Cin/groups]`, output NHWC→NCHW |
| `max_pool2d*`, `avg_pool2d` | `Max/AvgPool2d` | input NCHW→NHWC, output NHWC→NCHW (no weight) |
| `addmm`/`linear` | `Linear` | weight only (`[N,In]` input already lands `In` on `C`); `addmm` carries `[In,Out]`, `linear` carries `[Out,In]` — different permutations |

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
