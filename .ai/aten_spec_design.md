# ATen Op JSON Spec — Design

## Purpose

A compact, reproducible JSON format describing **one ATen operation invocation**,
replayed through the existing ATen-vs-native verifier ([[aten_native_verify_design]]).
A spec names a `target` and the op's `args`; tensor inputs are normally *synthesized*
(not stored), so a verification case is a few lines of JSON and needs no model weights.
The format is bidirectional — read JSON → `Op_spec.t`, and write `Op_spec.t` → JSON.

## The JSON format

```json
{
  "target": "torch.ops.aten.add.Tensor",
  "args": {
    "self":  { "dtype": "f32", "shape": [2, 3], "random": { "uniform": { "low": -1.0, "high": 1.0 } } },
    "other": { "dtype": "f32", "shape": [2, 3], "random": { "normal":  { "mean": 0.0, "variance": 1.0 } } }
  }
}
```

- `target` selects the op. Synthesis uses one RNG instance (`Pcg.default`, a fixed default
  seed) threaded through every input in declared parameter order, so a case is reproducible
  from the spec alone. A per-tensor `seed` may later be added inside the `Random` payload to
  make individual draws independently reproducible.
- `args` is keyed by the op's parameter names. A non-tensor arg is its literal value;
  an omitted arg with a default is filled in (e.g. `add`'s `alpha` defaults to `1`).
- A **tensor** arg is `{ dtype, shape, <payload> }` where `<payload>` is exactly one of:
  - `"values"`: explicit, bit-exact contents (a list of scalars, see below);
  - `"random"`: a distribution — `{ "uniform": { low, high } }` or
    `{ "normal": { mean, variance } }`;
  - `"sequence"`: `{ start, step }` (the arithmetic sequence `start + i*step`).
- A **scalar** (an op's `Scalar` arg, or one `values` entry) is a single-key object so
  its tag is explicit and floats stay exact: `{ "int": 1 }`, `{ "float": "0x3f800000" }`,
  `{ "bool": true }`. A bare JSON number/bool also decodes (number → float).

`dtype` is one of `"f32" | "f64" | "i64" | "i32" | "bool"`.

## Float reproducibility

JSON decimal text loses precision, so a float32 value is stored as its **IEEE-754 single
bit pattern in hex** (`"0x3f800000"`), via `Int32.bits_of_float`/`float_of_bits`. Random
synthesis is given matching care so a distribution reproduces bit-for-bit and a `Random`
payload is losslessly equivalent to materializing it as `Values`:

- generated/derived floats are rounded to their **f32-canonical** value (`Float32.to_f32`)
  — exactly what an f32 tensor stores;
- uniforms are drawn from the PCG output's **high 24 bits** (f32 mantissa width), avoiding
  low-bit bias and landing on representable fractions;
- the Box-Muller normal draws `u1 ∈ (0,1]` so `log` never sees 0 — **no NaN/∞** is ever
  synthesized (which would break the verifier's element-wise tolerance check).

The RNG is **PCG32** (`pcg_setseq_64_xsh_rr_32`), kept purely functional:
`next : t -> int * t` returns the value and the advanced state.

## Layering

```
lib/aten_spec/        (pure; no C++/ATen dep)
  Pcg, Float32, Dtype, Scalar_value, Distribution, Tensor_spec,
  Arg_value, Op_spec, Spec_util          ← types + jsont codecs + RNG
        ▲
        │ references Aten_spec.* fully qualified
lib/aten_op_spec/     (generated; pure)
  aten_op_spec.ml = per-op codecs + target registry + { target, args } envelope
        ▲
        │ decode → Op_spec.t ;  encode Op_spec.t → JSON
lib/aten_spec_run/    (C++ ATen dep)
  synthesize inputs (Pcg) → lower to Pytorch_types.Node + env → Interp_verify path
```

The pure/generated/runner split mirrors `pt2` vs `pt2_aten` ([[native_aten_bridge_design]]):
the parsing libraries carry no libtorch dependency, so the codecs and their round-trip
test build and run without the C++ build; only `aten_spec_run` (which actually executes
ATen) needs it.

### How it maps to the generated op config ([[aten_op_config_design]])

`bin/aten_ops_gen.ml` feeds the **same** curated `Aten_func_ast.t` op list to every
generator, so the new codec can't drift from the runtime config or the bindings:

```
native_functions.yaml + curated selection (bin/aten_ops_gen.ml)
   ├─ aten_emit.ml        → aten_operation_description.ml   (ctypes binding)
   ├─ aten_decode_gen.ml  → interp_dispatch.ml              (decode Node → call binding)
   ├─ aten_config_gen.ml  → aten_op_config.ml               (runtime schema: param_type + arity)
   └─ aten_spec_gen.ml    → aten_op_spec.ml   ← THIS         (per-op JSON codec)
```

`aten_config_gen`'s `param_type` enum is the pivot. `aten_spec_gen` mirrors it
field-for-field into each per-op record, and renders the **same YAML default** into the
codec's `~dec_absent` that `aten_config_gen` renders into `default_`:

| `param_type` | spec field type | `Arg_value` ctor | lowered `Argument` | `interp_decode` reader |
|---|---|---|---|---|
| `Tensor` | `Tensor_spec.t` | `Tensor` | `Tensor {name}` | `tensor_arg` |
| `Tensor_opt` | `Tensor_spec.t option` | `Tensor_opt` | `None`/`Tensor` | `tensor_arg` |
| `Tensor_list` | `Tensor_spec.t list` | `Tensor_list` | `Tensors` | `tensors_arg` |
| `Int` / `Int_list` | `int` / `int list` | `Int` / `Int_list` | `Int` / `Ints` | `int_arg` / `ints_arg` |
| `Float` | `Float32.t` | `Float` | `Float` | `float_arg` |
| `Bool` | `bool` | `Bool` | `Bool` | `bool_arg` |
| `Scalar` | `Scalar_value.t` | `Scalar` | `Int`/`Float`/`Bool` | `scalar_arg` |
| `Str` | `string` | `Str` | `String` | `string_arg` |
| `ScalarType?`/`Layout?`/`MemoryFormat?`/`Device?` | *(dropped)* | — | *absent* | `*_opt_arg` → `None` |

At runtime the two meet: `aten_op_spec` decodes the spec to `Op_spec.t`; the runner asks
`Aten_op_config.find target` for the **return arity** (how many verify outputs to
allocate), lowers `args` to a `Node.t`, and runs it through `interp_dispatch.ml` (ATen)
and `Op_bridge` (native), comparing with `Verify`. So config supplies the schema/arity,
the spec supplies the concrete values, both from the identical op list.

## How to run

```sh
# from file(s), or "-"/no args to read one spec from stdin:
opam exec -- dune exec bin/aten_spec_verify.exe -- spec1.json spec2.json
echo '{ "target": ... }' | opam exec -- dune exec bin/aten_spec_verify.exe
# [spec] torch.ops.aten.add.Tensor: matched   (exit 0; non-zero on any mismatch/error)

# --print evaluates and shows the output tensor from BOTH paths instead of comparing:
echo '{ "target": ... }' | opam exec -- dune exec bin/aten_spec_verify.exe -- --print
# [eval] torch.ops.aten.add.Tensor
#   aten   out0 = [1.5; 2.5]
#   native out0 = [1.5; 2.5]
```

Each spec is decoded, its inputs synthesized (from `Pcg.default`), and the op run on both paths.
In the default (verify) mode, ops with no native impl print `skipped` and pass. The CLI
needs the C++ ATen build but no downloaded weights.

## Tests

- `test/aten_spec_test.ml` — hermetic (no C++): Float32 hex round-trip, PCG reproducibility
  / f32-canonical / finite draws, and `Op_spec` decode↔encode round-trip via the generated
  codec.
- `test/aten_spec_run_test.ml` — in-process expect tests over `Aten_spec_run.run`/
  `eval_print` (links the ATen build; libtorch loaded once, so it's fast). Covers
  `add.Tensor` across every payload generator (values, uniform, normal, sequence) including
  **different generators for its two inputs** (uniform+normal, sequence+values), plus relu;
  random cases assert `matched`, exact cases assert the printed output tensor from both
  paths.
- `test/aten_spec_verify_cram.t` — end-to-end through the actual CLI (needs the ATen build,
  no weights): two cases that exercise the **stdin** and `--print` plumbing of
  `bin/aten_spec_verify`. Kept small on purpose — the generator matrix is in the expect
  test above.

## Files

| Path | Role |
|------|------|
| `lib/aten_spec/*.ml` | pure types, PCG, jsont codecs (decode + encode) |
| `lib/aten_gen/aten_spec_gen.ml` | per-op codec generator (new artifact target) |
| `lib/aten_op_spec/` | generated `aten_op_spec.ml` + dune rule |
| `lib/aten_spec_run/aten_spec_run.ml` | synthesis + Node lowering + verify |
| `bin/aten_spec_verify.ml` | CLI |
| `test/aten_spec_test.ml`, `test/aten_spec_verify_cram.t` | tests |

Current op coverage of the *verifying* path is whatever `Op_bridge` implements (add, relu);
specs for other curated ops decode and run on ATen but report `skipped` until the native op
lands. See [[native_aten_bridge_design]] for extending native coverage.
