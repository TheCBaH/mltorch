# `im2col.default` and `col2im.default`

## Scope established from the corpus

`volo_d1_224` is the only current Native-import blocker.  It contains four
identical pairs:

- `im2col.default(self, kernel_size=[3,3], dilation=[1,1],
  padding=[1,1], stride=[2,2])`;
- `col2im.default(self, output_size=[28,28], kernel_size=[3,3],
  dilation=[1,1], padding=[1,1], stride=[2,2])`.

The implementation should parse the complete two-dimensional ATen contract,
then have its typed shape rule reject an unsupported rank or inconsistent
column geometry.  The first landing is bounded to rank-4 `im2col` inputs and
rank-3 `col2im` inputs, which are the two ATen shapes the paired operations
produce and consume in this corpus.  This is a Native frame limit, not an
assumption made only by an importer.

## Frame mapping

`Aten_shape.of_aten` right-aligns each tensor in `[N;T;D;H;W;C]`.  Therefore:

| ATen tensor | Native frame axes |
| --- | --- |
| input `[batch, channels, input_h, input_w]` | `D,H,W,C` |
| im2col `[batch, channels * kh * kw, out_h * out_w]` | `H,W,C` |
| col2im output `[batch, channels, out_h, out_w]` | `D,H,W,C` |

The native operators must retain this mapping directly.  The bridge may place
the rank-4 NCHW input into the engine's channel-last working layout with the
existing `perm_nchw_to_nhwc` relayout, and restore it after `col2im`; the
operation itself still remains one native node.  A lowering to a chain of
`Pad`, `Unfold`, `Permute`, and `Reshape` would lose the original ATen
operation from the IR, violates `native_add_op.md`'s one-node rule, and
cannot give Native4D a corresponding operation.

## Semantics

For im2col output `(n, q, l)`, decode
`q = ((channel * kh) + kernel_h) * kw + kernel_w` and
`l = out_h * out_w_extent + out_w`.  The source is

```
input[n, channel,
      out_h * stride_h + kernel_h * dilation_h - pad_h,
      out_w * stride_w + kernel_w * dilation_w - pad_w]
```

and reads as zero when either spatial coordinate is outside the input.
The implementation uses the existing clamp-and-compare idiom: clamp each
coordinate before the strict `load6`, then select the load only if it equals
the original coordinate.  This preserves the in-bounds requirement under
both Direct and Symbolic evaluation.

`col2im` is overlap-add, not a reshape or inverse gather.  For each output
pixel it enumerates the finite kernel offsets, derives the candidate output
window coordinates by floor division, verifies exact stride divisibility and
the output-window bounds by clamp-and-compare, then sums the matching column
entries.  This naturally handles padding, dilation, and every overlap count
without a scatter primitive.

## Native4D

Both operations need `Im2col4`/`Col2im4` payloads.  Their parameters carry no
named frame axis, but their rank-changing layout is dialect-specific: Native4D
must only admit the four-axis forms above and must re-use the Native shape and
compute definitions.  The ordinary single-item corpus batch is extent one on
Native's `D` axis before the bridge's channel-last relayout, so it is not a
new intrinsic axis boundary.  The corresponding lowering arm still proves
that every non-unit input/output coordinate lies in `N/H/W/C`; a non-unit
Native `T` or `D` remains an ordinary domain rejection.

## Verification

Add direct hand-computed cases that expose padding zeros and overlapping
scatter-add, symbolic-grounding agreement, one real-ATen bridge verification
per operation, serialized-importer coverage, Native4D direct and lowering
coverage, then recensus `volo_d1_224`.  Once the pair lands, its next exposed
frontier is `max.dim`, which is a separate two-output operation.
