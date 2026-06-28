# Native TensorFlow Conv/Linear Semantics Plan

This note records the semantic gaps between the current native kernels and the
TensorFlow-style ops they should cover, plus a concrete remediation plan.

The important split:

- `Linear` already matches Dense/FullyConnected when the input is already shaped
  as `[..., In]`.
- `Conv2d` matches only dense NHWC Conv2D with explicit symmetric padding,
  stride, no dilation, and no groups.
- TensorFlow depthwise convolution is not a separate magic primitive; it is the
  depthwise case of grouped convolution. The current native `Conv2d` does not
  model groups, so it cannot express depthwise semantics yet.

## Current Native Semantics

### Conv2d

Native `Conv2d` expects:

```text
x      : [N,T,D,H,W,Cin]
weight : [Cout,1,1,Kh,Kw,Cin]  (Cout on axis N)
bias   : [1,1,1,1,1,Cout]
out    : [N,T,D,Hout,Wout,Cout]
```

The pixel formula is dense cross-correlation:

```text
out[..., oh, ow, oc] =
  bias[oc] +
  sum ic=0..Cin-1
  sum kh,kw
    x[..., oh*stride_h + kh - pad_h,
          ow*stride_w + kw - pad_w,
          ic]
    * weight[oc, kh, kw, ic]
```

This matches TensorFlow `Conv2D` for NHWC inputs and filter layout
`[Kh,Kw,Cin,Cout]` after bridge relayout, subject to the limitations below.

### Linear

Native `Linear` expects:

```text
x      : [..., In]        (feature axis is native C)
weight : [Out,1,1,1,1,In] (Out on native N)
bias   : [..., Out]       (broadcast on outer axes)
out    : [..., Out]
```

The pixel formula is:

```text
out[..., oc] = bias[oc] + sum ic=0..In-1 x[..., ic] * weight[oc, ic]
```

This matches TensorFlow Dense/FullyConnected when the input is already
`[..., In]`. It does not perform implicit flattening.

## Gaps

### 1. Grouped and Depthwise Conv2D

TensorFlow/PyTorch grouped convolution partitions input and output channels into
groups. Depthwise convolution is the special case:

```text
groups = input_channels
output_channels = input_channels * channel_multiplier
```

TensorFlow depthwise formula:

```text
out[..., ih, iw, ic * channel_multiplier + m] =
  sum kh,kw
    x[..., ih*stride_h + kh - pad_h,
          iw*stride_w + kw - pad_w,
          ic]
    * depthwise_filter[kh, kw, ic, m]
```

Current native `Conv2d` always reduces over every input channel for every output
channel:

```text
sum ic=0..Cin-1
```

That is dense convolution. It cannot express depthwise unless the caller expands
the depthwise filter into a sparse dense Conv2D weight tensor, which is wasteful
and loses the semantic structure.

### 2. Padding Semantics

Current native padding is symmetric per axis:

```ocaml
pad : { h : nonnegative; w : nonnegative }
```

This covers:

- `VALID` as `pad = 0`
- some `SAME` cases where before/after padding are equal

It does not cover general TensorFlow padding:

- `SAME` when total padding is odd and split asymmetrically
- explicit paddings like `[[0,0],[top,bottom],[left,right],[0,0]]`

### 3. Dilation

Current native Conv2D has no dilation parameter. TensorFlow Conv2D and
DepthwiseConv2D support spatial dilation:

```text
src_h = oh * stride_h + kh * dilation_h - pad_top
src_w = ow * stride_w + kw * dilation_w - pad_left
```

### 4. FullyConnected Implicit Flattening

Native `Linear` only reduces over the native `C` axis and preserves all outer
axes. Some TensorFlow/TFLite FullyConnected forms flatten all non-batch
dimensions before applying the weight matrix:

```text
[B, D1, D2, ..., In_last] -> [B, D1*D2*...*In_last]
```

That flatten is not part of native `Linear`; it must be represented explicitly.

## Remediation Plan

### Step 1: Generalize Conv2d Parameters

Replace symmetric `pad` with explicit low/high padding and add dilation/groups:

```ocaml
type axis_window = {
  kernel : Dim.extent Dim.t;
  stride : Op_config.Pos.t;
  pad_before : Op_config.Nonneg.t;
  pad_after : Op_config.Nonneg.t;
  dilation : Op_config.Pos.t;
}

type params = {
  h : axis_window;
  w : axis_window;
  in_channels : Dim.extent Dim.t;
  groups : Op_config.Pos.t;
}
```

Validation rules:

- `in_channels % groups = 0`
- `out_channels % groups = 0`
- `weight Cin_per_group = in_channels / groups`
- For depthwise: `groups = in_channels` and
  `out_channels = in_channels * channel_multiplier`

### Step 2: Extend Window Arithmetic

Update `Window_axis` to support asymmetric padding and dilation.

Output extent:

```text
effective_kernel = (kernel - 1) * dilation + 1
out = floor((in + pad_before + pad_after - effective_kernel) / stride) + 1
```

Source index:

```text
src = out * stride + k * dilation - pad_before
```

The clipped reduction bounds must still guarantee every generated `src` is
in-bounds before calling strict `load`.

### Step 3: Implement Grouped Conv Indexing

For grouped convolution, compute:

```text
out_per_group = Cout / groups
in_per_group = Cin / groups
group = oc / out_per_group
local_oc = oc % out_per_group
```

Then reduce only over `local_ic in [0, in_per_group)`:

```text
ic = group * in_per_group + local_ic
```

Weight layout should be:

```text
weight[oc, kh, kw, local_ic]
```

This matches PyTorch grouped Conv2D weight layout
`[Cout, Cin/groups, Kh, Kw]` after bridge relayout.

### Step 4: Add Depthwise-Friendly Construction

Depthwise can be implemented as grouped Conv2D with:

```text
groups = input_channels
out_per_group = channel_multiplier
in_per_group = 1
```

For TensorFlow depthwise filter layout `[Kh,Kw,Cin,channel_multiplier]`, bridge
or importer should relayout to grouped-conv weight layout:

```text
oc = ic * channel_multiplier + m
weight[oc, kh, kw, 0] = filter[kh, kw, ic, m]
```

Prefer not to expand to dense `[Cout,Kh,Kw,Cin]`; use grouped layout directly.

### Step 5: Bridge/Importer Behavior

ATen bridge:

- Stop rejecting `groups <> 1` once grouped Conv2D is implemented.
- Preserve the current relayout principle:
  - activations: only channel-first/channel-last movement
  - weights: move output channels to native `N`
- Validate unsupported cases at the boundary and return `Some (Error msg)` or
  `None` consistently.

TensorFlow importer:

- `Conv2D`:
  - parse `data_format`
  - parse `strides`
  - parse `dilations`
  - parse `padding`
  - convert `SAME` to explicit `pad_before/pad_after`
  - set `groups = 1`
- `DepthwiseConv2dNative`:
  - parse `channel_multiplier` from filter shape
  - set `groups = input_channels`
  - relayout filter to grouped weight layout
- `FullyConnected`:
  - insert explicit flatten/reshape when the source op semantics require it
  - otherwise map directly to native `Linear`

### Step 6: Tests

Add expect tests that print both graph decompositions and numeric outputs.

Conv2D:

- `VALID`, stride 1
- `SAME`, odd total padding requiring asymmetric split
- dilation > 1
- multi-channel dense conv

Grouped Conv2D:

- `groups = 2`, each group with distinct values to catch cross-group leakage
- output channels per group > 1

Depthwise:

- `channel_multiplier = 1`
- `channel_multiplier > 1`
- values chosen so `oc -> (ic,m)` mistakes are visible

FullyConnected:

- direct `[..., In] -> [..., Out]`
- implicit-flatten source op represented as explicit flatten + `Linear`
- no-bias case equals zero bias

### Step 7: Keep Linear Small

Do not add flattening behavior to `Linear`. Keep it as:

```text
Dense over the last/native C axis, preserving outer axes.
```

Flattening is a shape transformation and should remain explicit in the graph.
