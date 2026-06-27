# Native-ATen Bridge — Design

## Purpose

The bridge (`lib/native_aten_bridge/`) converts between:
- **ATen tensors**: opaque C++ handles (`[ `atc_tensor_opaque ] structure ptr`),
  managed by the GC via ctypes finalizers, with arbitrary strides and rank
- **Native tensors**: `Tensor.packed` (6D NHWC Bigarray, always dense, always
  float32 on output, GC-managed)

It also provides per-op dispatch: given a graph node and the ATen environment,
run the equivalent Native op and return the result(s).

## Tensor Bridge (`tensor_bridge.ml`)

### `of_aten` — ATen → Native

1. **Contiguity check**: `Aten_tensor.is_contiguous t`.  Non-contiguous tensors
   return `Error "non-contiguous ..."`.  ATen op outputs are typically contiguous;
   views (permute, select) are not and should be made contiguous before comparison.
2. **Shape**: `Aten_tensor.shape t` → `int array` → `Aten_shape.of_aten` →
   `Vec6.shape` (right-aligned, outer axes set to 1).
3. **Data copy**: `Aten_tensor.data Aten_dtype.float32 t` returns a Bigarray
   **view** over the C++ storage; `Array1.blit` copies it into a new Bigarray
   so the native tensor's lifetime is independent of the ATen handle.
4. **Payload**: assemble `{ Payload.fmt = F32; quant = No_quant; data }`.

Supported dtypes: `Float` → `F32`, `Long` → `I64`.  Others return `Error`.

### Layout: why flat copy works

For a contiguous ATen tensor of shape `[d₀, d₁, …, dₙ₋₁]` (row-major):
```
linear index = i₀ * (d₁*…*dₙ₋₁) + i₁ * (d₂*…*dₙ₋₁) + … + iₙ₋₁
```

`Aten_shape.of_aten` right-aligns into the 6D frame, placing `d₀` on axis
`6-n`, `dₙ₋₁` on axis `C` (innermost).  The native 6D offset is:
```
offset = i_N * (T*D*H*W*C) + … + i_C   (Horner fold, C innermost)
```

With the padding axes fixed at 1 (contributing 0 to offset), this reduces to
exactly the same linear index as ATen's row-major order.  **So for contiguous
tensors, the flat element order is identical.**  No transposition is needed.

> **Important semantic note**: this right-alignment is *positional*, not
> *semantic*.  A rank-4 ATen NCHW tensor `[N, C, H, W]` lands with C on axis `H`
> and W on axis `C`.  The native ops are written to work on any axis layout; they
> never assume "axis C = channels" at the tensor level.  Semantic layout concerns
> (NCHW vs NHWC transposition for conv weights) are an op-level responsibility.

### `to_aten_flat` — Native → ATen (for comparison)

Creates a 1D ATen tensor `[numel]` from the native Bigarray.  Shape ambiguity is
intentional: callers compare element-wise (not shape-wise), so a flat 1D tensor
is unambiguous.  Uses `Aten_tensor.of_bigarray` which copies the data.

## Op Bridge (`op_bridge.ml`)

### Dispatch signature

```ocaml
val dispatch :
  aten_env : Interp_decode.env ->
  node     : Pytorch_types.Node.t ->
  (Tensor.packed list, string) result option
```

- `None` — no native implementation; verification skips this node
- `Some (Error msg)` — tensor conversion or execution failed
- `Some (Ok outputs)` — native outputs, matched by position to ATen outputs

### Argument decoding

Tensor args are resolved from `aten_env` (ATen handles) and converted via
`Tensor_bridge.of_aten`.  Non-tensor args (`int_arg`, `ints_arg`, `bool_arg`,
`float_arg`, `scalar_arg`) reuse `Interp_decode`'s helpers directly.

### Direct scheduler

Each op instantiates its `Compute(Direct)` functor and runs via `Schedule.evaluate`:

```ocaml
let run_relu x =
  let module C = Pointwise.Relu.Compute(Direct) in
  let (Tensor.Tensor r) = x in
  Schedule.evaluate (Pointwise.Relu.output_shape r.shape) (C.pixel x)
```

The `Direct` module (`lib/native/direct.ml`) has `type t = float` and `type
'role index = int`, so `pixel` reduces to a plain `(Axis.t -> int) -> float`
function that `Schedule.evaluate` applies at every output coordinate.

### Op coverage

| ATen target | Native op |
|-------------|-----------|
| `torch.ops.aten.relu.default` / `relu_.default` | `Pointwise.Relu.Compute(Direct)` |
| `torch.ops.aten.add.Tensor` / `add_.Tensor` | `Pointwise.Add.Compute(Direct)` |
| `torch.ops.aten.bmm.default` | `Matmul.Bmm.Compute(Direct)` |
| `torch.ops.aten.mean.dim` | `Reduce.Mean.Compute(Direct)` |
| `torch.ops.aten.rms_norm.default` | `Norm.RmsNorm.Compute(Direct)` |

These are exactly the native ops whose operand/output layouts already line up
under `of_aten`'s positional right-alignment, so no relayout is needed (`bmm`,
`mean.dim`, `rms_norm` resolve their reduced/normalised axes via
`Aten_shape.axis_of_dim`, keeping them consistent with where the data landed).

The remaining native ops — `Conv2d`, `Max/AvgPool2d`, `Linear` — need NCHW↔NHWC
input/weight/output relayout and are **deferred**; see
`native_aten_bridge_layout.md` for the two-class split and the planned
transpose-op approach.  All other ATen ops return `None` (Skipped).

### Extending coverage

To add an op, add a match arm in `Op_bridge.dispatch`.  The pattern is:
1. Decode tensor args via `native_tensor_arg`
2. Decode non-tensor args via `D.int_arg`, `D.ints_arg`, etc.
3. Instantiate and run `Compute(Direct)` via `Schedule.evaluate`
4. Return `Some (Ok [result])`

For ops requiring `op_config` param types (`Op_config.Pos.t`, `Hw.t`), validate
ATen args at the boundary and fail explicitly if out-of-range.

## Dependency graph

```
native_aten_bridge
  ├── aten          (Aten_tensor, Aten_dtype, Aten_scalar_type, Aten_shape)
  ├── native        (Tensor, Payload, Vec6, Direct, Schedule, ops/*)
  ├── interp        (Interp_decode — arg decoders, env type)
  └── core          (Core.Error — for Aten_shape.of_aten error type)
```

The `interp` library is `(wrapped false)` so `Interp_decode` and
`Interp_dispatch` are accessible as top-level modules.
