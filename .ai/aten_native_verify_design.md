# ATen vs Native Verification — Design

## Goal

Validate the native inference engine (`lib/native/`) by running both the ATen
path (C++ kernels via ctypes) and the native path (pure-OCaml SEMANTICS functors)
on every graph node, and asserting element-wise agreement within a tolerance.

## Architecture

```
Interp_verify.dispatch  (lib/aten_native_verify/)
    │
    ├── Interp_dispatch.dispatch   (ATen path — unchanged)
    │       → aten_env' : env
    │
    └── Op_bridge.dispatch         (native path)
            → Some (Ok native_outputs) | Some (Error msg) | None
                        │
                        └── Verify.verify_node
                                → node_result
                                → Verify.report ppf
```

## Components

### `Verify` (`verify.ml`)

**Types:**
```ocaml
type mismatch = { output : string; max_diff : float; atol : float }
type node_result =
  | Skipped              (* no native impl, or non-f32 output *)
  | Conv_error of string (* tensor conversion / execution error *)
  | Matched              (* all outputs agree within atol *)
  | Mismatched of mismatch list
```

**Comparison strategy:**

For float32 outputs, compare Bigarray element-wise:
- Materialize the ATen tensor first: `Verify.logical_tensor` is
  `Aten_tensor.materialize_for_raw_read`, because `as_float32` reads flat off the
  storage base. Guarding on `is_contiguous` alone — which it used to — compares a
  `select`/`unbind` view against the wrong region of its storage, since such a
  view *is* contiguous at a non-zero offset. See "The flat-`Bigarray` boundary"
  in `aten_core_build.md`.
- Extract ATen tensor as `float32_array` via `Aten_tensor.as_float32`
- Extract native Payload.F32 `.data` field directly (no copy needed)
- Since both are in the same flat order for contiguous right-aligned tensors
  (see `native_aten_bridge_design.md`), element [i] corresponds to element [i]

Non-float32 outputs (int64, bool) and outputs with numel mismatch → `Skipped`.

**Tolerance:** `atol = 1e-5` by default, with selective widening for
reduction-heavy ops whose accumulation order legitimately differs. Pure
float32 pointwise arithmetic is deterministic on the same hardware, so those
ops should still agree exactly or near-exactly. The current per-target
override is `torch.ops.aten.addmm.default -> 1e-4`, which is enough for the
resnet18 fixture's matmul noise without weakening pointwise verification.

**Output pairing:** `node.outputs : Argument.t list` provides the SSA names.
`Tensor` arguments in that list are matched positionally to `native_outputs`.
Non-tensor outputs (rare in inference graphs) are silently skipped.

### `Interp_verify` (`interp_verify.ml`)

A drop-in replacement for `Interp_dispatch.dispatch` with an optional `~verify`
flag:

```ocaml
val dispatch :
  verify : bool ->
  ?ppf   : Format.formatter ->
  Interp_decode.env ->
  Pytorch_types.Node.t ->
  Interp_decode.env
```

When `~verify:false` (default), it's a pure pass-through with zero overhead.
When `~verify:true`, it calls `Op_bridge.dispatch` after the ATen dispatch and
pipes the result to `Verify.report`.

Mismatches are printed to `ppf` (default: `Format.err_formatter`) so they appear
on stderr without interrupting the computation — the ATen path continues as the
source of truth.

## Integration with `interp_run.ml`

To enable verification in the test runner:
```ocaml
(* In test/interp_run.ml, replace: *)
let env = List.fold_left Interp_dispatch.dispatch env g.nodes in
(* with: *)
let env = List.fold_left (Interp_verify.dispatch ~verify:true) env g.nodes in
```

Or thread a `--verify` flag through `interp_run.ml`'s command line.

## Semantic differences handled

| Category | ATen | Native | Verification approach |
|----------|------|--------|-----------------------|
| Float32 ops (relu, add) | exact C++ float | exact OCaml float | direct compare; should match |
| ATen views (strided **or** offset) | possible | not supported (always dense) | `materialize_for_raw_read` copies dense at offset 0, then compare |
| Non-float32 dtypes (int64, bool) | yes | partial (I64 supported) | `Aten_tensor.as_float32 = None` → Skipped |
| Ops without native impl | yes | no | `Op_bridge = None` → Skipped |

## Expected tolerances

`atol = 1e-5` is conservative.  For pure float32 pointwise ops (relu, add), the
native and ATen implementations perform the same FP operations in the same order,
so differences should be 0.0 or at most ULP-level (~1e-7 for f32).

If running under BLAS/vectorized ATen kernels, reduction ops (sum, matmul) may
accumulate in different orders → expect larger differences (up to ~1e-4 for
large tensors).  Use a wider **per-op** tolerance for those targets when
adding coverage, rather than raising the global default.

## Files

| Path | Role |
|------|------|
| `lib/aten_native_verify/verify.ml` | comparison + reporting |
| `lib/aten_native_verify/interp_verify.ml` | dual-dispatch wrapper |
| `lib/aten_native_verify/dune` | library declaration |
