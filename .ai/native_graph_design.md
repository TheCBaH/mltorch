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

type 'g gop =                               (* parametrised over the subgraph type *)
  | Relu of { x } | Add of { a; b } | Bmm of { input; mat2 }
  | Conv2d of { params; x; weight; bias : tensor_ref option }
  | Permute of { perm; x } | Mean of { params; x }
  | Rms_norm of { params; x; weight : tensor_ref option }
  | Linear of { params; x; weight; bias : tensor_ref option }
  | Max_pool2d of { params; x } | Avg_pool2d of { params; x }
  | Subgraph of { graph : 'g; args : tensor_ref list }

module Node  : sig type t = { id : Node_id.t; op : Graph.t gop; outputs : Tensor_id.t list } end
module Graph : sig type t = { name; nodes; tensors; inputs; outputs } end
type op = Graph.t gop
```

Design decisions:

- **Typed operand fields, not a positional/string-keyed list.** An op names each
  input by a typed field (`Conv2d {weight; …}`). There is no position to track and
  no string to mistype; a node is fully described by `op` + `outputs`. The only
  generic dataflow accessors are `operands : op -> tensor_ref list` and
  `map_operands : (tensor_ref -> tensor_ref) -> op -> op` (one match each), needed
  by topo/transform passes.
- **Optional tensors are `tensor_ref option`.** "Conv without bias" is just
  `bias = None`; there is no inputs-list length to reason about. The evaluator
  resolves a `None` to a constant-filled handle (`fill 0.` bias, `fill 1.`
  rms-norm weight) — so the IR stays total and no op is modified. See §4.
- **Closed variant.** Transformation wants exhaustiveness, structural matching,
  and cheap structural equality — all defeated by an open/first-class-module op.
  The cost is the four per-op match functions (`Eval_op`, `Graph_shape`,
  `operands`, `map_operands`); adding an op is a compiler-guided edit of exactly
  those.
- **Each record type lives in its own module, type `t`** (`Node.t`, `Graph.t`),
  per the project rule (CLAUDE.md). This guarantees field-label uniqueness by
  construction (`Node.outputs` vs `Graph.outputs`) — no warning-30 suppression.
  `op` is parametrised over the subgraph type (`'g gop`) so it can be defined once
  rather than copied into the `module rec Node/Graph` group; `type op = Graph.t
  gop` ties it down.
- **Ids are graph-local and deterministic.** One builder owns a single tensor-id
  and a single node-id counter for the whole graph tree (subgraphs share them), so
  ids are unique across the tree (no collision when recursing/merging) and start
  at 0 per top-level build (stable expect-test output). The id is the carrier the
  symbolic `Expr.Load` already references via `Tensor_sig`.
- **Nested graphs are embedded, not registered.** `Subgraph` holds its nested
  `Graph.t` by value — no `subgraphs` map, no `Subgraph_id`. The nested graph
  carries its own `name`; `args` map positionally to its ordered `inputs`
  (function-application convention). A registry (sharing one definition across
  call sites) is a possible transform-era addition, not needed now.

## 2. The builder (`graph_builder.ml`) — a state monad

Functional, not imperative: `type 'a t = state -> 'a * state` with `let*`/`let+`,
threading the id counters, default element type, and accumulators. Output shapes
are **computed** (`Graph_shape`), never supplied.

- `input ~shape ?name ?fmt ?quant ()` — `?fmt` defaults to the builder's default
  element type (F32 unless `build ?dtype` overrides); `?name` defaults to
  `"input_<id>"`.
- Op constructors (`relu`, `add`, `conv2d`, …) — `?name` defaults to
  `"<op>_<output-id>"`; omitting `?bias`/`?weight` records `None`.
- `subgraph ~name body` runs `body` in a child accumulation that shares the global
  counters (ids stay unique) and returns a `Graph.t`; `invoke ?names g args`
  embeds it in a `Subgraph` node and allocates the parent-side output edges.
- `build ?dtype ~name ~outputs m` runs from the empty state and finalises.

A DSL with sugar beyond this (e.g. operator notation) can layer on top; the monad
already composes (subgraphs and future transform passes are builder computations).

## 3. The two per-op dispatch points

- `Eval_op.Make (S)` (`eval_op.ml`) — the single value-dispatch, functorised over
  `S`, applied once at `Direct` and once at a fresh `Symbolic.Make ()`. Each arm
  reads the op's typed fields, resolving a `tensor_ref` to a data handle via
  `operand : tensor_ref -> S.input`, shapes via `shape_of`, and an absent optional
  via `fill : float -> Vec6.shape -> S.input`. `Subgraph` is *not* a pixel — the
  graph traversal handles it.
- `Graph_shape.output_shape` (`graph_shape.ml`) — the S-independent twin, calling
  each native op's own `output_shape` on the operand sigs; `Subgraph` returns its
  embedded graph's output sigs' shapes.

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
  Tensor_id.Map.t`. Topo-walk threading an immutable env; per node,
  `Schedule.evaluate out_shape (Eval_op.Make(Direct).pixel op ~operand ~shape_of
  ~fill)`; `Subgraph` recurses (`args`→sub `inputs`, sub `outputs`→node
  `outputs`). Returns **every** edge's tensor, so any intermediate is printable.
- **Symbolic** (`eval_symbolic.ml` + `stage_program.ml`): `run : graph ->
  Stage_program.t`. One `Symbolic.Make ()` for the whole graph; env maps each edge
  id to its `Tensor_sig.t` (the builder already created one per edge), so a node's
  per-pixel `Expr.t` loads its producers' signatures — i.e. the **stage DAG falls
  out automatically**: a downstream stage references an upstream one purely through
  the signature it `Load`s (the "source = Input | Stage" distinction is recovered
  by membership, *not* an `Expr` variant — `Expr` is unchanged). Subgraphs are
  inline-expanded (flat DAG; nested-program form is future work).
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

## 8. Tests

`test/native/graph_test.ml` (Direct): sequence (add→relu, intermediate shown),
add, conv decomposition (NHWC intermediate + NCHW result + match vs reference
conv), optional bias omitted (None→zeros vs explicit zero), nested subgraph
(matches the flat build). `test/native/graph_symbolic_test.ml`: the add→relu and
conv-decomposition stage DAGs printed, then chain-grounded and checked equal to
Direct.
