# Native graph IR — representation, evaluation, transformation

The op layer in `native_compute_design.md` gives *one op* two interpretations
(Direct / Symbolic) through the `SEMANTICS` functor. What was missing was a
first-class **graph**: a holdable value that wires ops together, that you can
evaluate as a whole, print intermediates of, and — the long-term purpose —
**transform** (decompose, fuse, relayout). This doc is that layer: `lib/native/`
modules `Graph_ir`, `Graph_builder`, `Graph_shape`, `Eval_op`, `Eval_direct`,
`Stage_program`, `Eval_symbolic`.

The immediate goal is **evaluation**; the transformation framework has moved to its
own doc, `native_transform_design.md`, and is **not yet implemented** (the conv
decomposition is demonstrated by *constructing* the decomposed graph, §6). Where a
choice here was made with transformation in mind, it is called out.

## 1. The IR (`graph_ir.ml`)

```
Tensor_id.t / Node_id.t = private int      (* builder-allocated, unique tree-wide *)
tensor_ref = Tensor_id.t                    (* edges are single-assignment *)

type op =
  | Add of Pointwise.Add.t | Mul of Pointwise.Mul.t | Relu of Pointwise.Relu.t
  | Bmm of Matmul.Bmm.t | Conv2d of Conv.Conv2d.t | Linear of Linear.Linear.t
  | Permute of Permute.Permute.t | Mean of Reduce.Mean.t | Rms_norm of Norm.RmsNorm.t
  | Max_pool2d of Pool.MaxPool2d.t | Avg_pool2d of Pool.AvgPool2d.t
  | Clamp of Pointwise.Clamp.t | Clone of Pointwise.Clone.t
  | Hardtanh of Pointwise.Hardtanh.t
  | Add_scalar of Pointwise.Add_scalar.t | Div_scalar of Pointwise.Div_scalar.t
  (* each payload (params + typed operand refs) is a record `t` owned by that
     op's module, e.g. Conv.Conv2d.t = { params; x; weight; bias : _ option } *)

module Node  : sig type t = { id : Node_id.t; op; outputs : Tensor_id.t list } end
module Group : sig type t = { id; label; items : Node of Node_id.t | Group of t } end
module Graph : sig type t = { nodes; root : Group.t; tensors; inputs; input_kinds; outputs } end
```

Design decisions:

- **Typed operand fields, not a positional/string-keyed list.** An op names each
  input by a typed field (`Conv2d {weight; …}`). There is no position to track and
  no string to mistype; a node is fully described by `op` + `outputs`. The generic
  dataflow accessors `operands : op -> tensor_ref list` and `map_operands :
  (tensor_ref -> tensor_ref) -> op -> op` are not per-op matches — each op module
  supplies its own `operands`/`map_operands` on its payload `t`, and `Graph_ir`
  folds them over `op_registry` (see below).
- **Per-op metadata lives in the op module; common code folds a registry.** Each
  op module exposes its payload `type t`, a JSON `name`/`jsont`, `operands`,
  `map_operands`, and `pp`. `Graph_ir` wraps each in an `OP` first-class module
  (adding `inject`/`project`, the only parts that mention the variant) and lists
  them in `op_registry`; JSON encode/decode, `operands`, `map_operands`, and
  `pp_op` are then single registry folds rather than per-constructor matches. The
  payloads now depend on `Tensor_ref` (= `Tensor_id.t`) — the accepted cost of
  letting an op own its operand refs without depending on `Graph_ir`. `Discard`,
  which has no payload module, is handled inline in each fold.
- **Optional tensors are `tensor_ref option`.** "Conv without bias" is just
  `bias = None`; there is no inputs-list length to reason about. The evaluator
  resolves a `None` to a constant-filled handle (`fill 0.` bias, `fill 1.`
  rms-norm weight) — so the IR stays total and no op is modified. See §4.
- **Closed variant.** Transformation wants exhaustiveness, structural matching,
  and cheap structural equality — all defeated by an open/first-class-module op.
  Only `Eval_op` and `Graph_shape` still match per op (they need shape/semantics
  context the payload can't carry); serialise/dataflow/pp are registry folds.
  Adding an op is its module + one `op_registry` line + the two shape/eval arms.
- **Each payload record lives in its own module, type `t`** (`Conv.Conv2d.t`,
  `Node.t`, `Graph.t`), per the project rule (CLAUDE.md). Field labels are reused
  across op payloads (`x`, `params`, …) but stay unambiguous because each `t` is
  in its own module; the few cross-module match/build sites (`Eval_op`,
  `Graph_shape`, `Graph_builder`) qualify the first label (`{ Conv.Conv2d.params;
  x; … }`), the same convention as `node.Node.outputs`.
- **Ids are global and deterministic.** One builder owns a single tensor-id and
  node-id counter for the whole graph, so every ID denotes one concrete SSA value
  or operation and starts at 0 per top-level build. The id is the carrier the
  symbolic `Expr.Load` already references via `Tensor_sig`.
- **No native display names.** `Graph.name` and `Tensor_sig.name` are omitted:
  graph structure is identified only by deterministic node/tensor ids.
  Importer-facing names and paths belong in a provenance sidecar, not the
  execution IR.
- **One node per source ATen op — the dialect stays close to ATen by
  construction, not by convention.** An importer arm that needs another op's
  shape/compute logic reuses that *implementation* (a shared `Compute` functor
  or shape rule, called from the new op's own module) rather than emitting
  that op's node directly in place of its own; see `native_add_op.md`'s
  "Design goal" section for the sanctioned pattern and the current exceptions
  still owed a fix (`select.int`, `stack.default`). This is what lets
  `Native_transform`, Native4D conversion, and provenance treat `op_registry`
  as a faithful, closed inventory of "what the source program did" rather than
  an inventory of "what the importer happened to build it from."
- **PT2 provenance is a wrapper, not an IR field.** `Pt2_native_graph.t` pairs
  a native graph with qualified PT2 origins: tensor origins are
  `(graph_path, SSA name, TensorMeta)`, node origins are `(graph_path, node
  index, target, optional FX name, metadata)`, and constant tensor ids map to
  archive payload targets. One native node may carry several PT2 origins and a
  tensor may be `Derived`, so relayout/decomposition/fusion does not require
  fabricating source identities.
- **Inference inputs are kinded.** Every id in `Graph.inputs` has an
  `Input.kind`: `Input` is supplied by the caller for each run, while
  `Constant` is captured model state supplied by the model/archive loader.
  Parameters, buffers, and PT2 tensor constants deliberately collapse to the
  latter: inference does not need training ownership or mutation semantics.
  Importer-specific payload lookup lives in a separate provenance sidecar. The
  graph keeps an id→kind map instead of changing its ordered id list, so
  edge order and existing operand machinery remain unchanged. JSON records only
  non-default constant entries (`input_constants`); absent metadata decodes as
  `Input` for backward compatibility.
- **Groups are structural, not callable.** `Graph.root` is an authoritative,
  ordered tree of groups over globally stored nodes. Every node belongs to exactly
  one group; group items are node IDs or nested groups. Groups neither create
  inputs/outputs nor affect evaluation. Their boundary is derived from global
  dataflow. The PT2 importer uses a group for a non-trivial one-to-many lowering
  (such as NCHW/NHWC relayout around convolution), keeping that implementation
  inspectable as one unit without inventing a call boundary.

## 2. The builder (`graph_builder.ml`) — a state monad

Functional, not imperative: `type 'a t = state -> 'a * state` with `let*`/`let+`,
threading the id counters, default element type, and accumulators. Output shapes
are **computed** (`Graph_shape`), never supplied.

- `input ~shape ?name ?fmt ?quant ()` — `?fmt` defaults to the builder's default
  element type (F32 unless `build ?dtype` overrides). `?name` is an ignored
  compatibility argument.
- Op constructors (`relu`, `add`, `conv2d`, …) — `?name` is likewise ignored;
  omitting `?bias`/`?weight` records `None`.
- `group ?label body` runs `body` in a child structural accumulation that shares
  the global SSA counters and tables, then records its emitted node IDs as a
  nested group. It returns the body's result directly.
- `build ?dtype ~name ~outputs m` runs from the empty state and finalises;
  `~name` is ignored for compatibility.

A DSL with sugar beyond this (e.g. operator notation) can layer on top; the monad
already composes (groups and future transform passes are builder computations).

## 3. The two per-op dispatch points

- `Eval_op.Make (S)` (`eval_op.ml`) — the single value-dispatch, functorised over
  `S`, applied once at `Direct` and once at `Symbolic`. Each arm
  reads the op's typed fields, resolving a `tensor_ref` to a data handle via
  `operand : tensor_ref -> S.input`, shapes via `shape_of`, and an absent optional
  via `fill : float -> Vec6.shape -> S.input`.
- `Graph_shape.output_shape` (`graph_shape.ml`) — the S-independent twin, calling
  each native op's own `output_shape` on the operand sigs.

### 3a. `Unbind` — a rank-removing, variable-output op

`Unbind` (ATen `aten.unbind.int`) selects along one axis and produces **one output
per coordinate of that axis**. It is the only op whose output count is not a
constant of the op, so it is worth stating what its two dispatch arms do.

`Graph_shape` returns the whole list from `Split.Unbind.output_shapes` instead of
building a fixed-length one. `Eval_op` passes `~output` straight through as the
coordinate to read — no per-ordinal branch, unlike `Max_pool2d_with_indices`.

The shape rule is exactly `Mean keepdim=false` with a single dim, and shares its
implementation (`Aten_shape.repack_dropped`): the selected axis is removed and the
survivors re-pack right-aligned, so an input axis *outside* the selected one
shifts inward by one. Right-aligned `[2,3,4]` is `[H=2 W=3 C=4]`; unbinding `W`
gives three `[W=2 C=4]` outputs, and output `k` reads `[H=out.W, W=k, C=out.C]`.
Like `Mean`, this needs no input rank: under the right-aligned embedding the axes
outside the data are already extent 1.

Two boundaries the op inherits from the engine rather than from ATen:

- **No empty outputs.** `Dim.extent` is ≥ 1, so the output list is never empty and
  ATen's zero-length-`dim` case has no Native form at all — it is refused before
  reaching the op.
- **Arithmetic outputs are materialised F32 values, not views.** ATen's unbind
  returns dtype-preserving *views* onto the operand's storage. Native models it
  as a fresh dense value, but copies storage cells directly and keeps the input
  format and quantization metadata. Therefore `unbind` supports `I64` exactly
  (without routing values through float); it offers value equivalence, not
  aliasing.

Its output-arity, ceiling and claim-transfer rules are in
`native_multi_output_design.md` §1a and `native_transform_design.md` §8.

## 4. Optional operands

`fill` is the only S-specific capability beyond `SEMANTICS`: Direct fills a real
zero/one tensor; the Symbolic runner mints a fresh `Tensor_sig.t` and records it
in `Stage_program.consts` to bind to that constant at ground time. This keeps
`Eval_op` semantics-agnostic — it never fabricates a default `S.input` (impossible
generically, since `S.input` is a packed tensor for Direct but a `Tensor_sig.t`
for Symbolic). Conv/Linear bias-`None` fills `[1,1,1,1,1,Cout]` zeros (Cout =
weight's `N`); rms-norm weight-`None` fills ones.

## 5. Evaluation

- **Direct** (`eval_direct.ml`): `run : graph -> inputs:(id*packed) list -> packed
  Tensor_id.Map.t`. A single topo-walk threads an immutable global env; per node,
  `Schedule.evaluate out_shape (Eval_op.Make(Direct).pixel op ~operand ~shape_of
  ~fill)`. Returns **every** edge's tensor, so any intermediate is printable.
- **Symbolic** (`eval_symbolic.ml` + `stage_program.ml`): `run : graph ->
  Stage_program.t`. `Symbolic` is stateless, so there is no per-graph instance:
  each stage body is an `Expr.Builder` computation run on its own, and stage
  expressions therefore reuse reducer ordinals — correct, because an identity
  means nothing outside the expression that binds it, and a consumer composing
  two stages freshens the inserted one. Env maps each edge
  id to its `Tensor_sig.t` (the builder already created one per edge), so a node's
  per-pixel `Expr.t` loads its producers' signatures — i.e. the **stage DAG falls
  out automatically**: a downstream stage references an upstream one purely through
  the signature it `Load`s (the "source = Input | Stage" distinction is recovered
  by membership, *not* an `Expr` variant — `Expr` is unchanged). Groups are
  ignored by the flat stage DAG.
  `Stage_program.ground ~bind` chain-grounds the stages in topo order, feeding each
  result into the next stage's binding — making symbolic execution extend through
  the whole graph, and verifiable against Direct.

## 6. Conv decomposition (the expressiveness check)

A native conv runs in NHWC; an NCHW input is bracketed by two `Permute`s built the
way `op_bridge.native_perm_of_aten` builds a `Permute.perm`:

```
x_nchw --Permute(NCHW->NHWC)--> x_nhwc --Conv2d--> y_nhwc --Permute(NHWC->NCHW)--> y_nchw
```

Constructed directly with the builder (`test/native/graph_test.ml`). Direct eval
returns `x_nhwc`/`y_nhwc` as intermediates (printed), and `y_nhwc` is asserted
equal to a single native conv on the equivalently-laid input. Symbolically
(`graph_symbolic_test.ml`) the conv stage loads the permute stage's signature and
the final permute loads the conv stage's — the stage DAG made visible.

## 7. Transformation framework

Designed in **`native_transform_design.md`**, which supersedes the
`template`/`edit`/`recipe`/`remap` sketch this section used to carry. The shape of
the answer changed in three ways worth knowing before reading that doc: a recipe is
a list of independently placed *replacements* rather than a flat edit list; the
old→new `remap` became three relations (tensor value clusters, node clusters, and
directed provenance) because merging them loses claims; and the mapping carries an
explicit **equivalence claim** per cluster, so a future harness can verify that
both graphs compute the same values.

Why *this* IR is shaped for it, which has not changed: stable ids let a recipe
reference nodes and tensors by id and let `apply` preserve every untouched id, so
the mapping stays sparse and external references (the ATen-vs-native verify
harness) stay valid; immutability lets the input graph's analysis (the symbolic
stage DAG, future footprint) stay valid while the output is built;
`operands`/`map_operands` are the generic hooks an id remap needs; and the closed
op variant gives the matcher exhaustiveness and cheap structural comparison. The
conv decomposition of §6 is the canonical 1:N case.

## 8. Pretty-printing

`Graph_ir.pp` is the canonical graph dump for expect tests and debugging. It uses
`Fmt` as the pretty-printing composition layer: tensor lists are `Fmt.list`
inside brackets, optional operands use `Fmt.option`, and graph/node/op structure is
expressed with `Format` boxes through `Fmt.pf`. Avoid manual indentation counters
or separator refs in graph printers; prefer adding one small `pp_*` function per
record/variant payload and composing them.

Permutation printing intentionally omits identity axis mappings. A full 6-axis
`Permute.perm` is still stored in the IR so execution has a total bijection, but
the dump shows only the semantic moves, e.g. `perm=[H<-W, W<-C, C<-H]`.

## 9. Tests

`test/native/graph_test.ml` (Direct): sequence (add→relu, intermediate shown),
add, conv decomposition (NHWC intermediate + NCHW result + match vs reference
conv), optional bias omitted (None→zeros vs explicit zero), nested group
(matches the flat build). `test/native/graph_symbolic_test.ml`: the add→relu and
conv-decomposition stage DAGs printed, then chain-grounded and checked equal to
Direct.
