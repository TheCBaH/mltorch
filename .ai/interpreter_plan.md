# Interpreter design (graph executor for `.pt2` models)

Status: **design only** — the pt2 reader (`lib/pt2`), the ATen bridge
(`lib/pt2_aten`), and the op layer (`lib/aten`, ~1300 generated ops + Tensor API)
all exist and are tested. This document is the plan for the executor that ties
them together to actually run inference. Start with **resnet18** (9 distinct
ops, all already bound).

## What's already in place

- `Pt2_archive.open_pt2 path` → `{ program : ExportedProgram.t; weights }`.
  - `Pt2_archive.load_weight t name` → `Pt2_tensor.t` (raw, strided).
  - `Pt2_archive.load_pt path` → `Pt2_tensor.t` (a sample input image).
- `Pt2_aten.to_tensor : Pt2_tensor.t -> Aten_tensor.t` (materialises contiguous
  float32 and hands it to ATen via `of_bigarray`). Verified byte/stride-correct
  against numpy.
- Op layer: `module O = Aten_c.Aten_operations`, called on raw tensor handles —
  the same type as `Aten_tensor.t` — e.g. `O.add_Tensor a b (Aten_scalar.Int 1L)`,
  `O.relu g`, `O.convolution ...`. Multi-output ops fill a `tensorsN` struct.

## Graph shapes we must read (generated `Pytorch_types`)

- `ExportedProgram.graph_module.graph : Graph.t` with
  `inputs / outputs : Argument.t list` and `nodes : Node.t list`.
- `Node.{ target : string; inputs : NamedArgument.t list; outputs : Argument.t list }`.
  `target` = `"torch.ops.aten.<op>.<overload>"`.
- `NamedArgument.{ name; arg : Argument.t; kind }`; `Argument` is the union
  documented in `.ai/model_json.md` (`as_tensor {name}`, `as_int`, `as_ints`,
  `as_float`, `as_bool`, `as_none`, `as_scalar_type`, `as_tensors`, …).
- `graph_module.signature.input_specs : InputSpec.t list`, where the relevant
  variants are `Parameter (InputToParameterSpec {arg = {name}; parameter_name})`
  and `Buffer (InputToBufferSpec {arg = {name}; buffer_name; persistent})` and
  `User_input (UserInputSpec {arg})`. This maps a graph SSA input name (e.g.
  `p_conv1_weight`) to the config/weight name (`conv1.weight`).

## Executor (proposed `lib/interp/`)

### 1. Environment
`type env = (string, Aten_tensor.t) Hashtbl.t` — SSA value name → tensor.

Seeding (walk `signature.input_specs`, in `graph.inputs` order):
- `Parameter {arg={name}; parameter_name}` / `Buffer {arg={name}; buffer_name}`
  → `Hashtbl.add env name (Pt2_aten.to_tensor (Pt2_archive.load_weight t pname))`.
- `User_input {arg = as_tensor {name}}` → bind `name` to the input image tensor
  (`Pt2_aten.to_tensor (Pt2_archive.load_pt image)`).

Decision deferred to implementation: load all weights up front (simplest;
resnet18 ≈ 45 MB of float32, fine) vs lazily per node. Start eager.

### 2. Node loop
For each `node` in `graph.nodes`:
1. Strip the `torch.ops.aten.` prefix and split `target` into `(op, overload)`.
2. Decode each `node.inputs[].arg` to a runtime value:
   - `as_tensor {name}` → `Hashtbl.find env name`
   - `as_tensors [..]` → list of env tensors (for `cat`-like ops)
   - `as_int` / `as_ints` → `int64` / `int64 array` (kernel sizes, dims, strides)
   - `as_float` → `float`; `as_bool` → `bool`; `as_none` → the op's optional None
   - `as_scalar_type` → `Aten_scalar_type.t`
   - Scalars feeding a `Scalar` arg → `Aten_scalar.{Int|Float}`.
3. Dispatch to the bound op (see §3), then bind results: zip `node.outputs`
   (each `as_tensor {name}`) to the returned tensor(s). Multi-output ops
   (`max_pool2d_with_indices`, `_native_batch_norm_legit_no_training`) return a
   `tensorsN` struct — bind each `vK` to the K-th output name; outputs the graph
   never reads (e.g. the indices) can be ignored/bound anyway.

### 3. Dispatch — the crux
Generated ops have fixed monomorphic OCaml signatures, so a `string → thunk`
table needs per-op marshalling glue. **Recommended: hand-write dispatch for
resnet18's 9 ops first**, proving the loop end-to-end, then evaluate generating
the table from the same op metadata `bin/aten_ops_gen.ml` already consumes.

resnet18 op set (from `test/model_cram.t`) and their `O.*` calls:

| target | handler |
|---|---|
| `convolution.default` | `O.convolution input weight bias[?] stride pad dilation transposed out_pad groups` |
| `_native_batch_norm_legit_no_training.default` | 3-tuple: `O._native_batch_norm_legit_no_training` into a `tensors3` struct (bind v0) |
| `relu.default` | `O.relu x` |
| `add.Tensor` | `O.add_Tensor a b (Aten_scalar.Int 1L)` |
| `max_pool2d_with_indices.default` | 2-tuple struct (bind v0 = values) |
| `mean.dim` | `O.mean_dim x dims keepdim dtype?` |
| `permute.default` | `O.permute x dims` |
| `view.default` | `O.view x sizes` |
| `addmm.default` | `O.addmm bias mat1 mat2 beta alpha` |

Each handler pattern-matches the decoded arg list (positional, by the op's
schema order) and calls the binding. A mismatch (arg count/type) should raise a
clear "unhandled <target>" so missing ops surface immediately when extending to
other models.

### 4. Output / result
- `graph.outputs : Argument.t list` — read the final `as_tensor` name(s) from
  `env`. resnet18 returns one `[1;1000]` logits tensor.
- argmax over the logits (via `Aten_tensor.data` / a small reduction, or bind
  `O.argmax`), then map the class index through the ImageNet label files shipped
  in the release zip (`imagenet_lsvrc_2015_synsets.txt` /
  `imagenet_metadata.txt`) to a human label.

### 5. Test
- A gated cram (same `PT2_DATA` mechanism as `pt2_load_cram.t`): run resnet18 on
  a known sample image, assert top-1 class. Golden value from a one-off Python
  `torch` run on the same image (record it in the cram).
- Memory note: ATen tensors are GC-managed (`Aten_tensor.manage`); the env holds
  references so intermediates stay alive across the loop. `Aten_tensor.live_count`
  can sanity-check there's no runaway leak.

## Risks / open questions
- **Strided inputs**: the sample image is channels-last; the bridge currently
  materialises NCHW-contiguous, which is correct but not what PyTorch feeds conv
  (it keeps channels-last for speed). Numerically identical, so fine for now.
- **Arg order / optional bias**: `convolution`'s `bias` is `Tensor?`; when the
  graph passes `as_none`, hand the binding its null-handle/None form.
- **Dispatch breadth**: extending past resnet18 (efficientnet/vit add gelu, cat,
  layer_norm, softmax, etc. — all already bound) is "add a handler row", until
  the auto-generated table is worth building.
