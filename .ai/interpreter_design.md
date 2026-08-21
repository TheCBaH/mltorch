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
- `Pt2_aten.to_tensor : Pt2_tensor.t -> Aten_tensor.t` (preserves float32 and
  int64 storage/layout and hands it to ATen via `of_storage`). Verified
  byte/stride-correct against numpy.
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
3. Dispatch to the bound op (see §3), then bind results. **There are two output
   shapes, and the serializer distinguishes them:**
   - **One tensor or a fixed tuple** — `node.outputs` is a list of `as_tensor
     {name}`, one per result, zipped positionally (`bind1` / `bind_many`).
     Multi-output ops (`max_pool2d_with_indices`,
     `_native_batch_norm_legit_no_training`) return a `tensorsN` struct — bind
     each `vK` to the K-th output name; outputs the graph never reads (e.g. the
     indices) can be ignored/bound anyway.
   - **A `Tensor[]` return** (`unbind.int`) — `node.outputs` is a **single**
     `as_tensors [name0; …; nameN]` carrying every result name. Positional
     zipping does not apply, so `bind_tensor_list` handles it: it requires
     exactly that shape and pairs names to tensors through
     `Err.List.map2 ~unequal_lengths`, so the arity check cannot drift from the
     pairing it guards. Silently zipping would leave SSA names unbound or drop
     tensors.

   Anything reading a node's result names uses `Interp_decode.output_names`,
   which flattens the `as_tensors` case in place. Filtering for `Argument.Tensor`
   alone yields *zero* names for a list-returning node — a report with no output
   lines, which reads as a pass.

   **`Native_interp` had the harsher form of the same bug**: its `output_names`
   *rejected* the shape outright, so every unbind graph failed as
   `` `Non_tensor_node_output `` — earlier than operator dispatch, and therefore
   never reaching `` `Unsupported_operator ``. It now flattens the same way, at
   both sites that read names (a node's outputs and the graph's own).

   Two things follow that the ATen path does not need:

   - **The name count is checkable, not just zippable.** The Native arity comes
     from the operand's extent at the selected axis, so `Native_interp` compares
     the two and reports `` `Output_arity `` on disagreement — *before* the
     builder allocates. `add_env`'s `Invalid_argument` stays an invariant about
     that module rather than becoming a report about the graph.
   - **The list is untrusted, so flattening is bounded.** The ceiling lives in
     the flattening helper, not in the operator arm, because `lower_node` reads
     output names *before* it calls `lower_op` — an arm-local check would run
     after the first `List.map` had already built a list sized by model data.
     The traversal stops at the limit and never learns the real length, which is
     why `Shape_error.Output_count` distinguishes `At_least` from `Exact`. The
     budget is threaded across the graph's outputs, so several individually
     legal lists cannot exceed the ceiling together.

   Model Explorer's own `max_outputs_metadata_per_node` (1024) is *tighter* and
   fires first on the `visualize` path; the lowering bound is what protects every
   caller without Model Explorer limits. `test/native_interp/output_limit_test.ml`
   drives it directly for that reason.

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

### 3a. The decoder set is the real op selection, and a gap in it is silent

`bin/aten_ops_gen.ml`'s curated list says which ops get a *binding*.
`Interp_decode`'s helper set says which of those get a *dispatch arm*, and the
two are not the same: `Aten_decode_gen.decode_arg` returns `None` for an argument
type it cannot decode, and `dispatch_arm` then drops the whole op — **silently**,
unlike `bin/aten_ops_gen.ml:105` where a generator skip is fatal.

The observable consequence is a target that is selected, has a C binding, and
answers `` `Unhandled_op `` at runtime. `avg_pool2d.default` and `argmax.default`
were both in that state until `int_opt_ptr` landed, purely because their `int?`
argument had no decoder. Nothing in the tree said so; the generated
`interp_dispatch.ml` is a build artifact, and the walk suite reported
`skipped (aten interp: unhandled op)`, which reads like an unselected op.

Two rules follow:

- **A binding change is not done until the emitted `interp_dispatch.ml` has been
  read.** A one-line addition to `selection` regenerates six artifacts and the
  build is green whether or not the op became dispatchable.
- **`skipped (aten interp: unhandled op)` in `test/native_walk_test.ml` is a
  decoder gap, not a coverage decision.** `skipped (no native impl)` is the
  honest state for an op ATen can run and native cannot.

The decoder set also carries the argument-name escape: identifiers the generator
emits go through `Aten_ident.ml_id`, shared with `Aten_spec_gen`, because
`slice.Tensor`'s bound argument is named `end`.

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

  **An entirely absent `Tensor?` key is not the same as `as_none`, and
  `Interp_decode` treats it differently from every other optional decoder in the
  file.** `tensor_arg` reports a missing key as `` `Missing_argument ``, while
  `float_opt_ptr`, `int_opt_ptr`, `scalar_opt_arg` and `bool_opt_arg` all decode
  absence to `None`. No corpus node hits it — every exported node spells its
  optionals — but `layer_norm`'s `weight`/`bias` both default to `None` in the
  schema, so a hand-written or re-exported node legitimately omits them, and the
  generated ATen path then cannot decode a node both native importers accept.
  It shows up in `test/native_bridge_test.ml` as the `omitted:` line reading
  `aten: missing required argument "weight"`: a decoder limitation, not a finding
  about the arm under test.

  Left as-is on purpose. `Interp_decode` is shared by every generated arm, and
  the two native importers read the argument themselves (`optional_tensor_present`
  in `Op_bridge`, the same test in `Native_interp`), so absence is already `None`
  where it matters. Recorded here so the next reader does not rediscover it as a
  bug in a specific op.
- **Dispatch breadth**: extending past resnet18 (efficientnet/vit add gelu, cat,
  layer_norm, softmax, etc. — all already bound) is "add a handler row", until
  the auto-generated table is worth building.
