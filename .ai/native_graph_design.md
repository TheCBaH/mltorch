# Native graph IR — representation, evaluation, transformation

The op layer in `native_compute_design.md` gives *one op* two interpretations
(Direct / Symbolic) through the `SEMANTICS` functor. What was missing was a
first-class **graph**: a holdable value that wires ops together, that you can
evaluate as a whole, print intermediates of, and — the long-term purpose —
**transform** (decompose, fuse, relayout). This doc is that layer: `lib/native/`
modules `Graph_ir`, `Graph_builder`, `Graph_shape`, `Eval_op`, `Eval_direct`,
`Stage_program`, `Eval_symbolic`.

The immediate goal is **evaluation**; the transformation framework is **designed
here but not yet implemented** (the conv decomposition is demonstrated by
*constructing* the decomposed graph, §6). Where a choice was made with future
transformation in mind, it is called out.

## 1. The IR (`graph_ir.ml`)

```
Tensor_id.t / Node_id.t = private int      (* builder-allocated, unique tree-wide *)
tensor_ref = Tensor_id.t                    (* edges are single-assignment *)

type op =
  | Add of Pointwise.Add.t | Mul of Pointwise.Mul.t | Relu of Pointwise.Relu.t
  | Bmm of Matmul.Bmm.t | Conv2d of Conv.Conv2d.t | Linear of Linear.Linear.t
  | Permute of Permute.Permute.t | Mean of Reduce.Mean.t | Rms_norm of Norm.RmsNorm.t
  | Max_pool2d of Pool.MaxPool2d.t | Avg_pool2d of Pool.AvgPool2d.t
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
  `S`, applied once at `Direct` and once at a fresh `Symbolic.Make ()`. Each arm
  reads the op's typed fields, resolving a `tensor_ref` to a data handle via
  `operand : tensor_ref -> S.input`, shapes via `shape_of`, and an absent optional
  via `fill : float -> Vec6.shape -> S.input`.
- `Graph_shape.output_shape` (`graph_shape.ml`) — the S-independent twin, calling
  each native op's own `output_shape` on the operand sigs.

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
  Stage_program.t`. One `Symbolic.Make ()` for the whole graph; env maps each edge
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

## 7. Transformation framework (designed, not implemented)

Applicative: a transformer returns a **recipe** of edits; `apply` produces a *new*
graph plus an old→new id mapping. Inputs are never mutated.

```
type template = Graph_builder.t -> ins:Tensor_id.t list -> Tensor_id.t list
type edit  = Replace_node of Node_id.t * template
           | Remove_node  of Node_id.t
           | Insert_before of Node_id.t * template
type recipe = edit list
type remap = { nodes : Node_id.t -> Node_id.t option;
               tensors : Tensor_id.t -> Tensor_id.t option }
val apply : graph -> recipe -> graph * remap
```

Why the IR is shaped for this: stable ids let a recipe reference nodes/tensors by
id and let `apply` preserve every untouched id (keeping the rest of a recipe — and
external references like the ATen-vs-native verify harness — valid); `remap`
chains transform passes; immutability lets the input graph's analysis (the
symbolic stage DAG, future footprint) stay valid while the output is built. The
conv decomposition is the canonical `Replace_node (conv, decomposition_template)`.
`operands`/`map_operands` are the generic hooks an id-remap needs.

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
