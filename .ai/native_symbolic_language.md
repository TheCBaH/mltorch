# Native symbolic language — dependencies & tiling validation

Deepens the `Symbolic` mode of `native_compute_design.md`. The earlier doc gave a
value IR (`expr`) with affine `Load` indices and `Reduce`. That is enough to *run*
symbolically; it is **not** enough to answer the question this engine actually
needs: **for a chosen output tile, which input pixels does it read?** That
input→output dependency relation is what validates a tiling/fusion decision
(halo size, whether an axis is reduction-coupled, whether a dependency is
data-dependent and therefore untileable).

Two operations drive every design choice here:

- **softmax** — a *reduction whose result is reused across the reduced axis*:
  `y[p,c] = exp(x[p,c] − m[p]) / s[p]`, with `m[p]=max_c x[p,c]`,
  `s[p]=Σ_c exp(x[p,c]−m[p])`. Each output pixel depends on the **whole** reduced
  axis of `x`, and `m`,`s` are shared by every `c`. Tiling must learn "free on
  `N..W`, not tileable on `C`."
- **gather / index** — *input coordinates depend on the values of another tensor*:
  `y[..] = table[ idx[..] , c ]`. The footprint into `table` is **not** an affine
  function of the output tile; it is data-dependent. Tiling must learn "the
  gathered axis of `table` must be fully resident."

## 1. What "capture dependencies" means

For an op we want the relation `R ⊆ OutCoord × (Source × InCoord)`: which input
(or producer) pixels each output pixel reads. We don't materialize `R`; we
compute its **footprint function** per source:

```
F_src : OutputBox → SourceRegion        (region of `src` read by an output tile)
```

A tiling/fusion choice is then validated by the *shape* of `F_src`:

| `F_src(tile)` shape | dependency class | tiling consequence |
|---|---|---|
| endpoints = tile endpoints (halo 0) | **local / elementwise** | tile any axis freely; no inter-tile sharing |
| endpoints = tile ± const | **stencil** (conv, pool) | tileable with a constant halo / recompute; neighbors share a boundary |
| `[0, extent)` independent of the tile, via a `Reduce` | **reduction-coupled** | that axis can't be output-tiled without a split/two-pass reduction; free axes still tile |
| `⊤` (full extent) from a data-dependent index | **data-dependent** (gather) | gathered axis can't be tiled; source must stay resident |
| data-dependent on the **output** side | **scatter** (dual) | output tiles overlap unpredictably → write conflicts |

So the language must make indices, reductions, and reuse all *first-class and
analyzable*, and the analysis is simply a third interpretation of the same
program.

## 2. Language additions

### 2.1 First-class index expressions

This **revises** `native_compute_design.md` (where indices were concrete `int`).
Indices become a small typed sub-language so they can be interpreted concretely
(Direct), structurally (Symbolic), or as intervals (Footprint). The split that
matters is **affine vs data-dependent**:

```ocaml
module type INDEX = sig
  type t                                      (* a value, from SEMANTICS *)
  type 'role index

  val avar  : axis -> position index          (* an output-coordinate component (n..c) *)
  val rvar  : rdom -> position index          (* a reduction-domain variable *)
  val ic    : int -> delta index
  val ( +@ ) : delta index -> delta index -> delta index
  val scale : int -> delta index -> delta index   (* affine: k * index *)

  (* the gather primitive: turn a *value* into an index. This is the ONE
     constructor that makes a footprint data-dependent. *)
  val data  : t -> delta index
  val clamp : lo:int -> hi:int -> delta index -> position index
end
```

`avar`/`rvar`/`ic`/`+@`/`scale` build **affine** index forms — exactly
analyzable. `data e` injects a loaded value as an index (gather); its footprint is
the declared extent of the source (or `[lo,hi)` if `clamp`ed). A `Load` carries
six index expressions (one per axis `N T D H W C`); **which `avar`s appear** in each
encodes reuse — softmax's `m[p]` is read with an index that *omits* `avar C`, so
the same value serves every `c`.

### 2.2 Stages — a producer DAG, not one inlined expression

Softmax is multi-stage: two reductions feed a pointwise stage, and the reductions
are *reused*. Inlining them into one per-pixel `expr` (the Phase-1 model) loses
the reuse and recomputes the reduction per `c`. So the symbolic program is a small
**DAG of stages**, each a named producer with its own iteration **domain** (the
subset of the 6 axes it is defined over):

```ocaml
type source = Input of input_id | Stage of stage_id     (* what a Load reads *)

type 'e stage = {
  id     : stage_id;
  domain : axis_set;                 (* axes this stage ranges over *)
  body   : 'e;                       (* pixel expr in terms of its domain vars *)
}

type 'e program = { stages : 'e stage list; output : stage_id }   (* topo-ordered DAG *)
```

**Status (implemented).** This stage DAG is now *produced* by `Eval_symbolic.run`
over a `Graph_ir.graph` (see `native_graph_design.md`), as `Stage_program.t`:
each graph node becomes one stage whose `body` is its per-pixel `Expr.t`. The
`source = Input | Stage` distinction is **not** an `Expr` variant — every `Load`
already carries the producer's `Tensor_sig.t`, so a stage references an upstream
stage purely through the signature it loads; Input-vs-Stage is recovered by
membership in `graph.inputs`. The per-axis `domain`/`axis_set` and the
reduction-collapses-an-axis bookkeeping below remain future work (footprint
analysis); the current `stage` carries `{ id; sg; body }` and `output_shape` is
supplied per-op via `Graph_shape` rather than inferred.

A **reduction collapses an axis**: a `Reduce` over `rdom` produces a stage whose
`domain` excludes the reduced axis. So `m` and `s` have domain `{N,T,D,H,W}` (no
`C`); `y` has domain `{N,T,D,H,W,C}` and `Load`s `m`/`s` with `C` absent from
their index — that absence *is* the broadcast/reuse, and the footprint analysis
reads it directly.

For a windowed stage (conv/pool) the domain isn't just "minus the reduced
axis" — `H`/`W` shrink by the kernel/stride/pad relationship, not just drop.
`native_compute_design.md` §2b names the missing piece: each op needs its own
`output_shape`, computed from its params and its inputs' shapes, and *that* is
what a stage's `domain`'s extents actually are. Not yet implemented.

### 2.3 The value IR (extended)

```ocaml
type expr =
  | Const  of float
  | Binary of binary_op * expr * expr           (* add sub mul div max min *)
  | Unary  of unary_op  * expr                  (* exp relu … *)
  | Load   of source * index_expr array         (* read Input/Stage at a 6-axis index *)
  | Reduce of { kind : [`Sum|`Max_reduce]; rdom : rdom; (* lo,hi *) body : expr }
```

`Reduce.body` may itself `Load` other stages, so `s = Σ_c exp(x[..c] − m[p])`
expresses "reduction over an operation with the result of a prior reduction"
directly — the user's softmax shape — as a stage `s` whose body loads stage `m`.

### 2.4 What a symbolic input *is* — a tensor signature, not data

`SEMANTICS.input` is abstract on purpose, and this is where the two
instantiations diverge most:

- in **`Direct`**, `input = Tensor.packed` — a concrete tensor; its *elements*
  are what the computation consumes.
- in **`Symbolic`**, there are no elements to consume. An input is a
  **`Tensor_sig.t`** — an *abstract tensor*: everything an op or analysis needs to
  build and reason about the expression **except the bulk data**.

```ocaml
module Tensor_sig : sig
  type t = {
    id    : int;                     (* binding key; the `source` a Load refers to *)
    name  : string;                  (* the graph SSA value name, for debug *)
    shape : Shape.t;                 (* a `size` vec6 — may be `Sym` (native_tensor §4) *)
    fmt   : Format.packed;           (* which of F32/…/I8 — existential over the GADT *)
    quant : quant option;            (* Some iff fmt is quantized; compile-time metadata *)
  }
end
```

So **a tensor is "specified" symbolically by declaring its interface**, exactly
like a function parameter's type. Each field earns its place by being needed to
*build* the expression without any data:

- **`shape`** drives index bounds, broadcast (`broadcast_coord` reads which axes
  have extent 1), reduction extents, and the whole footprint analysis — and it may be a
  `Sym` shape, so the symbolic program is generic over dynamic sizes.
- **`fmt`** decides what a `Load` *emits*: a plain read, a `Dequant`, or an
  `F16` decode — and tells codegen which load instruction to lower to.
- **`quant`** is compile-time metadata baked from the model (small: a scale/zero_point
  or a per-channel array), so it lives in the signature, *not* in the data. Per-channel
  scale even enters the expression — it depends on the `C` index — so it must be
  known when building, not at run time.

`Load`'s `source` therefore carries a signature on both sides —
`source = Input of Tensor_sig.t | Stage of Tensor_sig.t` — because an intermediate
**stage** is also "an input" to its consumers and needs the same interface. A
stage's signature is *derived*: its `shape` is its domain's extents, its `fmt` is
the op's output format.

**Binding — how the symbolic world meets data.** A symbolic program is a function
of its input signatures; to turn it into numbers you supply a
`binding : id -> Tensor.packed`:

- a **full** binding (every signature → a concrete tensor) is what evaluation
  uses, and is exactly the Phase-3 verification check `eval (Symbolic.pixel) under
  binding == Direct.pixel`.
- a **partial** binding — only the *constant* inputs (the weights/buffers), leaving
  activations unbound — is **constant-folding / partial evaluation**: weights
  collapse to `Const` leaves while data-dependent inputs stay symbolic, which is
  how codegen specializes a kernel to known weights without baking in the activation.

**Where the signatures come from.** For a `.pt2` subgraph they are the model's
`signature.input_specs` (the same the ATen `Interp.run` walks): a `Parameter`/
`Buffer` becomes a signature with a known static shape + format + quant, supplied
in the constant binding (foldable); a `User_input` becomes a signature whose shape
may be `Sym`, left unbound. No weights are read to build the symbolic program — the
specification is purely the interfaces.

**Implementation gap (found 2026-06-24):** the `Tensor_sig.t` sketched above is
the agreed design, but `lib/native/tensor_sig.ml` as built only has
`{ id; name; shape }` — `fmt` and `quant` were dropped somewhere between this
note and the code, and `Tensor_sig.create` has no way to supply either. So
today a symbolic input can't actually carry its dtype or quant params — the
two reasons this doc gives for needing them (deciding what `Load` emits;
per-channel scale entering the expression via the `C` index) aren't available
to build with yet. The concrete types already exist (`Payload.packed_fmt`,
`Quant.t`) and just need wiring in: `fmt : Payload.packed_fmt` and
`quant : Quant.t option`, threaded through `Tensor_sig.create`. Not yet
implemented; the two current call sites are `test/native/symbolic_test.ml`
(`Tensor_sig.create ~name ~shape`, no fmt/quant today).

## 3. Three interpretations of one program

The `INDEX`/`SEMANTICS` pair is instantiated three ways; the analysis we care
about here is the third:

| Instance | `t` (value) | `index` (index) | yields |
|---|---|---|---|
| `Direct` | `float` | `int` (`data e` = clamp∘round) | the computed pixel |
| `Symbolic` | `expr` | affine-or-`data` form | the stage DAG (codegen/fusion) |
| **`Footprint`** | abstract value (range/`⊤`) | **interval** (affine) or `⊤` | the dependency footprint of a tile |

### `Footprint` — the dependency analysis

Run the program with the output coordinate bound to a **symbolic tile**: each
output axis `a` is an interval `[Lo_a, Hi_a)` (symbolic). Index interpretation:

- `avar a` → `[Lo_a, Hi_a)`; `rvar r` → `[0, extent_r)` (the whole reduction range);
  `ic k` → `[k,k+1)`; `+@`/`scale` → interval arithmetic (affine ⇒ exact, with
  endpoints affine in the tile bounds, so a constant **halo** stays visible).
- `data e` → `⊤` (full source extent), narrowed to `[lo,hi)` by `clamp`.

A `Load(src, index_expr array)` contributes a 6-axis **region** of `src` (the box of
the six interval indices). Values combine by **union** of their input regions
(`Binary`/`Unary` union operands; `Reduce` unions `body` over the reduction range —
which is what makes the reduced axis full). Per-stage footprints compose along the DAG: the
output tile's footprint into an `Input` is obtained by pushing the tile through
each consuming stage and unioning.

```
region        = axis_range array  (* length 6, axis order N T D H W C *)
axis_range    = Empty | Range of bound * bound | Full     (* bound affine in tile Lo/Hi *)
```

## 4. Worked examples — and the tiling verdict each yields

**Elementwise** `y[o]=relu(x[o])`: `Load(x,[avar n..avar c])` →
`F_x(tile)=tile`. Verdict: **local**, tile every axis, halo 0.

**Conv2d** (stride `s`, kernel `K`, pad `P` on H): the H index is
`scale s (avar H) +@ rvar kh +@ ic(−P)`, `kh∈[0,K)`. For tile `H∈[Lo,Hi)`:
`F_x.H = [s·Lo−P, s·(Hi−1)+K−1−P+1)`. Verdict: **stencil**, halo `K−1` (read-only
overlap → safe to parallel; for fused producer, recompute the halo). `C_in` is a
`rvar` → full `[0,Cin)` (reduction-coupled, as expected for conv's channel sum).

**Softmax** (stages `m`,`s` over `{N..W}`, output `y` over `{N..W,C}`):
- `y` loads `x[..,avar C]`, `m[p]` and `s[p]` with **`C` absent** from their index.
- `m`/`s` bodies `Reduce` over `rvar C` → their `F_x.C = [0, Cn)`.
- Compose: `F_x(tile of y) = (P-box) × [0, Cn)`.
  Verdict: **free on N,T,D,H,W; reduction-coupled on C** — output-tiling `C` is
  illegal without a two-pass/split reduction; `m`,`s` are computed once per `p`
  and reused across `C` (read straight off the index that omits `C`). Exactly the
  reuse the spec called out, *derived* rather than asserted.

**Gather** `y[..,c] = table[ data(load idx) , c ]`: `table`'s row index is
`data(...)` → `F_table.row = ⊤` (or `clamp`ed range). Verdict: **data-dependent**
— the gathered axis can't be tiled; `table` must be resident for any output tile;
`idx` itself is read locally (`F_idx = tile`). The dual, **scatter**
(`out[data(...)] = …`), makes the *output* footprint `⊤` → overlapping write
regions → flagged as not independently tileable.

## 5. Validating a tiling/fusion decision

A proposed schedule names a tile shape and which axes run in parallel / are fused.
Validate against the footprints:

1. **Parallel axis legal** iff, for read sources, distinct tiles' footprints may
   overlap only on read-only data (always true for feed-forward inputs) and, for
   any *written* producer, footprints are disjoint (fails under scatter `⊤`).
2. **Fuse producer into consumer tile** iff the producer's footprint for the
   consumer tile is a **bounded box** (local or stencil-with-finite-halo). A
   reduction-coupled or `⊤` footprint means the producer can't be fused per-tile —
   it needs its own pass (two-pass softmax; resident gather table).
3. **Halo / recompute cost** = the stencil overlap = the non-tile part of the
   footprint endpoints; the analysis reports it so a schedule can choose
   recompute-vs-store.

The point: every one of these is a predicate over `F_src`, computed by the
`Footprint` interpretation of the *same* op functor used to run — so a tiling
decision can never disagree with the actual computation.

## 6. Tests / validation

- **Footprint vs brute force**: for small shapes, compare `F_src(tile)` against the
  set of input coords `Direct` actually touches (instrument `load`) for a sweep of
  tiles. Must be a sound over-approximation, and *exact* for affine cases.
- **The three stressors**: assert the verdict for elementwise (local), conv
  (halo `K−1`), softmax (full `C`, local `N..W`), gather (`⊤` on the gathered
  axis, local on `idx`).
- **Reuse check**: softmax `m`/`s` footprint endpoints are independent of `avar C`
  (proves cross-`C` reuse is visible to the analysis).

## 7. Open questions

- **Precision**: interval-affine endpoints capture halos and full axes but lose
  correlations (e.g. diagonal access). A small polyhedral domain would be exact;
  intervals are the pragmatic start, sound (over-approximating) by construction.
- **Clamped gather**: if `idx` has a statically known range, `data` + `clamp`
  yields a bounded box → a gather can become tileable. Worth modeling a declared
  `idx` range so common embedding-lookup patterns aren't pessimized to `⊤`.
- **Dynamic extents / `SymInt`** (`N` batch, `T` sequence): tile bounds and
  reduction extents may be a `Sym` size (`native_tensor_design.md` §4). The affine
  interval domain already takes symbolic endpoints, so a `Sym` extent is the
  interval `[0, s_i)` and a footprint over it validates a tiling for the *whole
  family* of admissible shapes at once. The legality predicates discharge their
  side-conditions against the `ShapeEnv` (`s_i mod tile = 0`, halo `< s_i`,
  unknown-sign coefficients treated conservatively); where a guard is missing, the
  verdict is "tileable with a remainder/tail loop" rather than "illegal."
- **Unbacked `SymInt`** (a size read from tensor data, e.g. `nonzero`): the
  size-level analogue of `data`-indexed gather — it produces a `⊤` extent, so the
  axis is flagged untileable unless the `ShapeEnv` declares an upper bound. Gather's
  data-dependent footprint and unbacked sizes are the same phenomenon at index vs
  size level.

## Compact pool stencils

The value IR has an opaque `Max_pool` node. Its namespaced descriptor record
holds an input `Tensor_sig`, fixed kernel/stride/padding, symbolic output
coordinate, and `Value` or `Index` result kind. The configuration retains the
validated domain types (`Dim.extent`, `Op_config.Pos`, and `Op_config.Nonneg`),
rather than untyped integers. The geometry is retained rather than expanded
into nested generic reductions, letting a footprint pass recognize the clipped
2D stencil directly. `AvgPool2d` remains a generic clipped sum and
divide: unlike max pooling, it has neither a padding-as-zero hazard nor an
index-output contract.

### The selection predicate lives in `Max_op` (added 2026-07-28)

Pool selection is ATen's, transcribed from
`aten/src/ATen/native/cpu/MaxPoolKernel.cpp:215` (and matching the vectorised
path at `:116`, which blends on the same mask, so ATen has one well-defined
answer here):

```cpp
bool mask   = (val > maxval) || is_nan(val);
out_data[d2] = mask ? val   : maxval;
ind[d2]      = mask ? index : maxindex;
```

Two consequences. A NaN anywhere in the window propagates, and since every NaN
re-satisfies the predicate the **last** NaN wins together with its index. An
ordinary tie leaves the incumbent in place, which is what preserves the
first/smallest-flattened-index contract stated above. Value and index must be
updated **together** under this one predicate.

It lives in `lib/native/max_op.ml` rather than open-coded at each site because
three interpreters have to agree bit-for-bit — `Direct` runs the computation,
`Expr.eval` grounds the symbolic form against it, and the transformation
verifier (`.ai/native_transform_verify.md`) probes against both. Before the
module existed they did not: `Direct` pooled with `Float.max` while `Expr.eval`
pooled with a strict `>`, so the two disagreed on `-0.` vs `+0.` and on NaN, and
`Direct`'s own value and index paths disagreed with each other (the value
propagated a NaN, the index did not). `test/native/compute_test.ml` pins the
ATen answer in f32 bits; `test/native/symbolic_test.ml` pins Direct == Symbolic.

`Max_op.Float_max` — generic `Reduce Max_reduce` — is a **different** operator
and deliberately still `Float.max`. It does not match ATen's `amax`, but there
is nothing to match: `max_values_kernel_impl`
(`ReduceOpsKernel.cpp:370`) dispatches a scalar reducer (`zmath.h:210`: first
operand on a signed-zero tie, a canonical `quiet_NaN`) *and* a vector one
(`vec_base.h:862`: second operand on a tie, the operand's own NaN bits), and
which runs depends on shape and vector width. Fixing it needs an empirical study
against real ATen across both zero orders and several vector widths. Until then
the tests assert only that the two interpreters agree, not what the answer is.
