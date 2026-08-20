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

1. **Materialize**: `Aten_tensor.materialize_for_raw_read t`.  Step 3 reads the
   tensor as a flat Bigarray off the storage base, so any view must be copied
   dense at offset zero first.  **Not `is_contiguous` alone** — a `select` /
   `slice` / `unbind` view *is* contiguous at a non-zero offset, and that guard
   passed it through, converting another slice's values under this one's shape.
   See "The flat-`Bigarray` boundary" in `aten_core_build.md`.
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
  ( Graph_ir.graph * (Graph_ir.Tensor_id.t * Tensor.packed) list,
    Op_bridge.error )
  Err.t option
```

- `None` — no native implementation; verification skips this node
- `Some (Error e)` — decode / tensor-conversion / shape-validation failed, with
  a typed `Op_bridge.error`
- `Some (Ok (graph, bindings))` — a native graph encoding the op's computation
  (including relayout permutes where needed), plus input bindings mapping each
  graph input id to its pre-converted native tensor.  Callers evaluate the graph
  via `Eval_direct.run graph ~inputs:bindings` and extract outputs from
  `graph.outputs`.

`Op_bridge.error` is intentionally small and boundary-oriented:
- `Decode of Interp_decode.error` — argument lookup/kind/default failures lifted
  from the interpreter decoder
- `Tensor_bridge of { arg_name; message }` — ATen→native conversion failure for a
  specific tensor argument
- `Build of Graph_builder.error` — native graph construction failure
- `Invalid_hw_arg { name; values }` — malformed `[h; w]` / `[v]` style config
  lists
- `Validation_failure of string` — low-level param conversion failure (for
  example `Dim.extent` / `Op_config` checks)
- op-specific invalid-rank variants such as
  `Conv2d_padding_invalid_weight_rank of int array`

### Graph building

Each dispatch arm builds a `Graph_ir.graph` using `Graph_builder`, rather than
eagerly evaluating.  The helper `build_g ~name tensors body` allocates one input
edge per native tensor, runs `body` with those ids to produce output ids, and
returns the finished graph plus `[(input_id, packed)]` bindings.

Ops requiring relayout produce a graph named `"<op>_relayout"` containing
explicit `Permute` nodes before/after the core op (see `Op coverage` below and
`.ai/native_aten_bridge_layout.md`).

### Argument decoding

Tensor args are resolved from `aten_env` (ATen handles) and converted via
`Tensor_bridge.of_aten`.  Non-tensor args (`int_arg`, `ints_arg`, `bool_arg`,
`float_arg`, `scalar_arg`) reuse `Interp_decode`'s result-returning helpers
directly, and the bridge lifts those typed decode errors into `Op_bridge.error`.
```

The `Direct` module (`lib/native/direct.ml`) has `type t = float` and `type
'role index = int`, so `pixel` reduces to a plain `(Axis.t -> int) -> float`
function that `Schedule.evaluate` applies at every output coordinate.

### Op coverage

**No relayout needed** (op reads axes via `Aten_shape.axis_of_dim`, so they
track where `of_aten` placed the data):

| ATen target | Native op | Graph name |
|-------------|-----------|------------|
| `torch.ops.aten.relu.default` / `relu_.default` | `Relu` | `"relu"` |
| `torch.ops.aten.add.Tensor` / `add_.Tensor` | `Add` | `"add"` |
| `torch.ops.aten.mul.Tensor` / `mul_.Tensor` | `Mul` | `"mul"` |
| `torch.ops.aten.bmm.default` | `Bmm` | `"bmm"` |
| `torch.ops.aten.mean.dim` | `Mean` | `"mean"` |
| `torch.ops.aten.permute.default` | `Permute` | `"permute"` |
| `torch.ops.aten.rms_norm.default` | `RmsNorm` | `"rms_norm"` |
| `torch.ops.aten.unbind.int` | `Unbind` | `"unbind"` |

`unbind.int` is the only arm returning a VARIABLE number of outputs, and the
only one whose count is fixed by the operand rather than the op. Three things
follow:

- **All of them are exposed.** There is no dead output to drop, and
  `Verify.verify_node` requires exact cardinality for a dynamic list for exactly
  that reason (see `aten_native_verify_design.md`).
- **The rank comes from the ATen tensor, before conversion.** `of_aten`
  right-aligns into the six-axis frame, after which the rank is not recoverable,
  so `dim` must be normalised against `aten_rank` and not against the frame.
- **A non-f32 operand is refused, not reinterpreted.** The engine computes in
  f32 and `Graph_builder` gives every op output `Payload.F32`, while
  `Tensor_bridge.of_aten` happily accepts i64 — so an i64 unbind would produce
  f32 results that then fail the verifier's dtype pairing. The arm reports
  `Unsupported_input_dtype` carrying ATen's own dtype. Native unbind results are
  materialised values, not alias-preserving views; numerical equivalence is the
  contract.

**Relayout needed** (graph wraps the core op in `Permute` nodes; named
`"<op>_relayout"`; see `.ai/native_aten_bridge_layout.md` for derivation of
the six permutations):

| ATen target | Native op | Relayout |
|-------------|-----------|---------|
| `conv2d.default` | `Conv2d` | input NCHW→NHWC, weight OIHW→native, output NHWC→NCHW |
| `conv2d.padding` | `Conv2d_padding` | same conv relayout; native op translates `"valid"`/`"same"` to concrete windows before delegating to `Conv2d` compute |
| `convolution.default` | `Convolution` | same conv relayout; native op preserves `transposed`/`output_padding` params, reuses `Conv2d` for forward 2D, and evaluates transposed 2D directly |
| `adaptive_avg_pool2d.default` | `AdaptiveAvgPool2d` | input NCHW/CHW→NHWC, output NHWC→NCHW/CHW |
| `max_pool2d.default` | `MaxPool2d` | input NCHW→NHWC, output NHWC→NCHW |
| `linear.default` | `Linear` | weight [Out,In]→[N=Out,C=In] |
| `addmm.default` | `Linear` | weight (mat2) [In,Out]→[N=Out,C=In] |

Skipped (return `None`): `max_pool2d_with_indices` (2-output mismatch). Dense,
grouped/depthwise, dilated forward Conv2d, transposed 2D convolution, and
`conv2d.padding` string modes `"valid"`/`"same"` are covered. Native
`convolution.default` keeps `output_padding` for the transposed output-shape
formula; native `"same"` matches PyTorch's stride restriction: stride must be 1.

### Extending coverage

To add an op without relayout, add a match arm in `Op_bridge.dispatch`:
1. Decode tensor args via `native_tensor_arg` / `native_of_aten`
2. Decode non-tensor args via `D.int_arg`, `D.ints_arg`, etc.
3. Call `build_g ~name tensors body` where `body` uses `Graph_builder` ops
4. Return `Some (Ok [result])`

For ops requiring `op_config` param types (`Op_config.Pos.t`, `Hw.t`), validate
ATen args at the boundary and fail explicitly if out-of-range.  Rank-sensitive
checks should stay typed as bridge errors rather than collapsing to ad hoc
strings; for example `conv2d.padding` now reports
`Conv2d_padding_invalid_weight_rank` carrying the offending weight shape.

**Every decoded dim goes through `dim_axis ~op ~rank dim`.** `Aten_shape.axis_of_dim`
asserts its precondition and raises `Invalid_argument` — a fine contract for a
trusted caller, but `dim` here is untrusted model data. `dim_axis` is the one
approved route from a decoded `dim` to `axis_of_dim`: it normalises, range-checks,
and reports the fault as `` `Invalid_dim { op; dim; rank } `` (the *original*,
unnormalised `dim`), where `op` names the calling arm so the same row serves
`unbind.int`, `mean.dim` and `permute.default` (`dims_arg`, `native_perm_of_aten`)
without a per-arm vocabulary. Before this helper existed, `mean.dim` and
`permute.default` called `axis_of_dim` directly on decoded data with no guard,
so an out-of-range `dim` escaped `Op_bridge.dispatch` as an uncaught
`Invalid_argument` instead of a typed row (`test/native_bridge_test.ml`, the
"rejects an out-of-range dim" cases). `native_perm_of_aten` also now checks that
a decoded `dims` list has exactly `rank` entries, reporting
`` `Dims_count { op; rank; got } `` — previously a short/long list produced a
non-bijective permutation caught later, with a worse message, by
`Permute.output_shape`.

## Dependency graph

```
native_aten_bridge
  ├── aten          (Aten_tensor, Aten_dtype, Aten_scalar_type, Aten_shape)
  ├── native        (Tensor, Payload, Vec6, Direct, Schedule, ops/*)
  ├── interp        (Interp_decode — arg decoders, env type)
  └── core          (Err.Error — for Aten_shape.of_aten error type)
```

The `interp` library is `(wrapped false)` so `Interp_decode` and
`Interp_dispatch` are accessible as top-level modules.
