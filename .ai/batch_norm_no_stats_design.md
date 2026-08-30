# BatchNorm without running statistics

## Scope

`aten._native_batch_norm_legit.no_stats` is a training-mode batch
normalisation with no running mean or variance operands:

```
(output, save_mean, save_invstd) =
    _native_batch_norm_legit.no_stats(input, weight?, bias?, training,
                                      momentum, eps)
```

The PT2 corpus uses `training=true`, `momentum=0.0`, f32 inputs, optional
per-channel affine values, and `eps` of `1e-5` or `1e-8`.  It is not the
inference `_native_batch_norm_legit_no_training` target: replacing it with
`BatchNorm` would invent running statistics and produce different values.

## Native contract

Introduce `Norm.BatchNormNoStats`, with `x`, optional `weight` and `bias`, and
`{ channel; eps }`.  The target importers must decode `training` and
`momentum`; this initial slice accepts only `training=true` and preserves the
observed `momentum=0.0` contract (it has no state to update).  Weight and bias
are each independent options; absent means a scale of one or a shift of zero.

For every channel `c`, reduce every axis except `channel`:

```
mean[c] = sum(x[..., c]) / count
var[c]  = sum((x[..., c] - mean[c])^2) / count
inv[c]  = 1 / sqrt(var[c] + eps)
output  = (x - mean[c]) * inv[c] * weight[c] + bias[c]
```

`var` is deliberately biased, as in ATen's forward result.  The three Native
outputs are, in order, the input shape, then two C-vector shapes
`[1,1,1,1,1,C]`.  The reduction count is bounded at shape inference, before
the Compute functor converts it to a float.  Affine operands must have that
C-vector shape; malformed operands are a shape error, never an out-of-bounds
tensor read.

`Compute.pixel` is selected by output ordinal.  The activation branch uses the
output coordinate; the statistic branches use only its channel coordinate.
Each branch independently expresses the required reductions with
`Semantics.sum`, so Direct and Symbolic share exactly the same operation.

## Import, liveness, and Native4D

Both PT2 import paths relayout ATen NCHW to Native NHWC around one
`BatchNormNoStats` node and expose all three outputs.  The activation is
relayouted back to NCHW; mean and inverse-standard-deviation remain C-vectors.
Unlike inference batch norm's CPU-exported zero-size statistic placeholders,
these are real tensors and may be read.  They must not be silently dropped or
replaced by `Discard`.

`BatchNormNoStats4` retains the operation rather than lowering to the
inference depthwise-convolution rewrite: its batch reductions are data
dependent.  Its channel is fixed to `Axis4.C`; its two statistic outputs use
the valid Shape4 C-vector representation `[N=1,H=1,W=1,C]`.  It accepts
dynamic input and affine operands, validates the C-vector extents, and lowers
through the shared symbolic reduction expressions to the kernel path.  This is
an Equivalent conversion, not an approximation or a running-stat fold.

## Proof surface

The landing change needs:

- Native Direct/Symbolic agreement, including each affine-option combination
  and all three output ordinals;
- an explicit small numerical fixture checking `mean`, biased `var`, and
  `invstd` against ATen's formula;
- both importer fixtures for `training=true`, `momentum=0`, optional affine
  operands, and a live statistic consumer;
- a Native4D conversion/direct-symbolic/kernel fixture showing the three
  outputs survive and a reduction reaches the generated kernel; and
- a real-ATen comparison when the curated binding closure includes this CPU
  dispatch target.
