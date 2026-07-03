# Native compute — semantics functor, direct & symbolic, schedule

Pillars 3 & 4 of `native_inference_plan.md`. An operation is the *algorithm* for
one output pixel; the *schedule* is how the output coordinate space is iterated.
The algorithm is written once against an abstract numeric domain (`SEMANTICS`) and
instantiated as a concrete evaluator (`Direct`) or a symbolic-expression builder
(`Symbolic`) for codegen/fusion.

A **pixel** is the **scalar** at one 6D `coord` (`N T D H W C`) — the unit the
functor produces per call. C-vectorization is a *schedule* concern, not part of
the algorithm.

Ops are wired into a holdable, evaluable, transformable **graph** one layer up;
see `native_graph_design.md` (`Graph_ir`, `Eval_op` = the single per-op dispatch
applied at both `Direct` and `Symbolic`, `Eval_direct`, `Eval_symbolic`).

## 1. The `SEMANTICS` signature — the abstract numeric domain

An op never names `float` or `expr` directly; it speaks `SEMANTICS`. `t` is "a
real-valued scalar in this domain." Inputs are read through `load`, which hides
dequantization. Reductions over input windows (conv/matmul/pool) are a primitive
— a **bounded reduction domain** — so the algorithm states *what* it reduces while
each semantics decides *how* (run the loop now, or emit a `Reduce` node).

```ocaml
module type SEMANTICS = sig
  type t                          (* a real scalar in this domain *)
  type 'role index                (* an index expression; role = position | delta *)

  (* value domain — minimal primitive basis. [max]/[min]/[relu] are NOT here:
     they derive from [select]+[lt] at each call site, so adding any activation
     costs no new signature entry, Expr constructor, or eval/pp arms.
     relu x = select (lt x (const 0.)) (const 0.) x *)
  val const : float -> t
  val add : t -> t -> t
  val sub : t -> t -> t
  val mul : t -> t -> t
  val div : t -> t -> t
  val exp : t -> t                (* transcendental — not select-expressible *)
  val sqrt : t -> t

  (* boolean + selection — the scalable basis for activations and clamps *)
  type b
  val lt : t -> t -> b
  val select : b -> t -> t -> t

  (* index domain. The phantom role splits: [position] is a known ≥ 0
     (the only thing [load] accepts); [delta] is a signed affine. The whole
     window arithmetic is built in [delta]; a delta becomes a [position] only via
     [clamp_low] (= max 0, provably ≥ 0) or [assume_index] (one unchecked
     claim, encapsulated in Window_axis.window where the clip invariant holds). *)
  val index_zero   : position index
  val index_extent : Dim.extent Dim.t -> delta index
  val index_const  : int -> delta index
  val of_index     : position index -> delta index
  val index_add    : delta index -> delta index -> delta index
  val index_scale  : int -> delta index -> delta index
  val index_min    : delta index -> delta index -> delta index
  val clamp_low    : delta index -> position index
  val assume_index : delta index -> position index

  type input
  val load : input -> position index Vec6.t -> t

  (* bounds are [index] (not [int]) so windowed reductions can express
     position-dependent clipped bounds: lo = clamp_low(...), hi = index_min kernel (...) *)
  val sum        : lo:position index -> hi:delta index -> (position index -> t) -> t
  val max_reduce : lo:position index -> hi:delta index -> (position index -> t) -> t
end
```

Only *values* flow through `SEMANTICS.t`. The index sub-language (`index`) is separate:
affine expressions built in the signed `delta` role, with `load` accepting only
`position` — so a raw windowed offset (which may be negative in the pad region) can't
reach a read without going through `clamp_low` or the one `assume_index` inside
`Window_axis.window`. See `semantics.ml` for the phantom-role types.

**`load` is strict.** It reads exactly the index it is given and raises if that
index is outside the source extent — it does *not* broadcast an extent-1 axis nor
pad an out-of-range one. Both are explicit, opt-in steps the op takes *before*
`load`, so every read is in-bounds by construction at the call site:

- **broadcast** — reduce the output coordinate against the operand's *own* shape
  (`Pointwise.broadcast_coord ~index_zero`, mapping each extent-1 axis to index 0). A
  per-channel term's value fans out across the broadcast axis without `load` ever
  seeing an out-of-bounds index.
- **padding** — take a guarded tap (`Tensor.shift_in_bounds`, returning the
  in-bounds coord or `None` in the pad region) and supply the pad value yourself.
- **fixed-layout size-1 axes** (conv/linear weight & bias, reduce/norm reads) pass
  an explicit `index_zero` on the structurally-size-1 axes — a degenerate
  broadcast the op knows statically, again always in-bounds.
- **windowed reductions** clip their bounds (§4) so the source position is in
  range by construction.

This keeps the one capability that *was* baked into the read primitive
(silently returning 0 out of bounds) out of it, where it had two failure modes:
masking a real indexing bug, and being wrong for `max_reduce` (§2c).

## 2. An op is a functor `(S : SEMANTICS) -> { pixel }`

Every op has the same shape: given its input handles and an output `coord`,
return one `S.t`.

```ocaml
module type OP = functor (S : SEMANTICS) -> sig
  type params
  val pixel : params -> S.input list -> coord -> S.t
end
```

### Example — pointwise (relu, add)

```ocaml
module Relu (S : SEMANTICS) = struct
  let pixel x (o : coord) = S.relu (S.load x o)
end

module Add (S : SEMANTICS) = struct
  (* [load] is strict, so each operand is read at the output coord reduced against
     its OWN shape ([Pointwise.broadcast_coord]): an extent-1 axis (a per-channel term) fans
     out by reading index 0 there, never by handing [load] an out-of-bounds index.
     Broadcast needs no strides, see native_tensor_design §1b. *)
  let pixel ~a_shape ~b_shape a b (o : coord) =
    let read sh t = S.load t (Pointwise.broadcast_coord ~index_zero:S.index_zero sh o) in
    S.add (read a_shape a) (read b_shape b)
end
```

### Example — conv2d (NHWC)

Output `[n,t,d,h,w,oc]` reduces over the kernel window and the input channels
for `oc`'s group. Weight is `[Cout,1,1,Kh,Kw,Cin_per_group]`; bias is
`[1,1,1,1,1,Cout]`, read at an explicit `index_zero` on its structurally-size-1
axes (a degenerate broadcast, in-bounds). The kh/kw reduction bounds are
clipped via `Window_axis.Compute(S).window` so the source position is always a
valid index, never a generate-then-guard read of the padding region. See
`ops/conv.ml` for the implementation.


### 2b. Output shape inference — an op must compute its own output shape

`Schedule.evaluate shape pixel` (§5) iterates the *output* coordinate space —
so that shape has to exist before iteration can start. Today nothing computes
it: every call site just writes the answer by hand. `test/native/compute_test.ml`
has a 3×3 input through a 2×2 kernel (stride 1, no pad) and passes
`Vec6.shape ~h:2 ~w:2 ...` to `Schedule.evaluate` as a *separate*, unconnected
literal — get the arithmetic wrong (or change the kernel/stride/pad and forget
to update it) and nothing catches it; you just iterate the wrong region of the
output, silently. The op that owns `kernel`/`stride`/`pad` is the only thing
that can compute this correctly, so it has to be the one that exposes it.

This is plain shape arithmetic — never touches `S.t`/`S.index` — so, like
`params`, it's hoisted **outside** the `(S : SEMANTICS)` functor: one
`output_shape` serves `Direct` and `Symbolic` identically (and, per
`native_symbolic_language.md` §2.2, is exactly the missing mechanism behind
"a stage's shape is its domain's extents" — that note already assumed this
existed; it doesn't until now). Concretely, each op is a plain module holding
everything that doesn't depend on `S` — `params` (if it has one),
`output_shape` — plus a nested functor, `Compute`, for the one thing that
does:

```ocaml
module Relu = struct
  let output_shape (x_shape : Vec6.shape) = x_shape   (* identity *)
  module Compute (S : Semantics.SEMANTICS) = struct
    let pixel x (out : Semantics.position S.index Vec6.t) = S.relu (S.load x out)
  end
end

module Add = struct
  let output_shape (a_shape : Vec6.shape) (b_shape : Vec6.shape) =
    (* per axis: extents must be equal or one must be 1 (its broadcast axis,
       taking the other's extent); any other pairing is incompatible and raises —
       NOT silently the larger of the two *)
    ...
  module Compute (S : Semantics.SEMANTICS) = struct
    let pixel a b (out : Semantics.position S.index Vec6.t) =
      S.add (S.load a out) (S.load b out)
  end
end

module Conv2d = struct
  type axis_window = {
    kernel : ...;
    stride : ...;
    pad_before : ...;
    pad_after : ...;
    dilation : ...;
  }

  type params = { h : axis_window; w : axis_window; in_channels : ...; groups : ... }
  let output_shape ~(x_shape : Vec6.shape) ~(weight_shape : Vec6.shape)
      (p : params) =
    (* N/T/D pass through from x_shape; C comes from weight_shape's Cout
       (weight is [Cout,1,1,Kh,Kw,Cin_per_group] — params.in_channels is Cin,
       so this needs weight_shape, not just x_shape + params); H/W: *)
    let out_hw axis window =
      let in_extent = (Vec6.get x_shape axis :> int) in
      let effective_kernel = ((kernel - 1) * dilation) + 1 in
      ((in_extent + pad_before + pad_after - effective_kernel) / stride) + 1
    in
    ...
  module Compute (S : Semantics.SEMANTICS) = struct
    let pixel (p : params) ~x_shape ~weight_shape ~x ~weight ~bias
        (out : Semantics.position S.index Vec6.t) = ...
  end
end
```

Without the nesting, `output_shape` would have had to either live inside
`Conv2d(S)` (forcing a meaningless choice of `S` just to call something that
can't depend on it, and duplicating the same body once per instantiation) or
sit at the top level under an op-prefixed name (`conv2d_output_shape`) since
two ops' `output_shape`s can't both be named `output_shape` at the same
level. The nested-module shape gives every op one name (`Relu`, `Add`,
`Conv2d`) holding its params/shape/compute together, with `Compute(S).pixel`
as the only thing that actually varies by `S`. Implemented — see
`ops/conv.ml`, `ops/pointwise.ml`.

Two things fall out of writing the arithmetic down:

- **`Conv2d.output_shape` needs `weight_shape`, not just `x_shape` and
  `params`.** `Cout` lives in the weight tensor's shape, and `params` today
  carries `in_channels` (`Cin`) and `groups`, not `Cout`. There is no single uniform
  `output_shape : params -> Vec6.shape -> Vec6.shape` across ops — `pixel`
  already takes different inputs per op (`~x ~weight ~bias` vs. `a b`), and
  `output_shape` has to take whatever subset of those shapes it actually
  needs, op by op.
- **The H/W formula here is the *forward* direction of the exact same
  windowed-axis relationship as the clipped reduction bounds used to read
  the window** —
  given `in_extent`/`kernel`/`stride`/`pad_before`/`pad_after`/`dilation`, one direction asks "how many
  output positions are there," the other asks "for this output position,
  which kernel offsets are valid." **Partially unified, now shared across
  ops**: both directions live in `lib/native/ops/window_axis.ml`
  (`output_extent` and `Compute.bounds`) — `Conv2d.output_shape` and
  `Pool2d.output_shape` (§2c) both call `output_extent`; `Conv2d.Compute`
  and `Pool2d.Compute` both call `bounds` via `Window_axis.Compute (S)`. Still
  two distinct closed forms, not one — there's no way to derive one from the
  other in O(1) without restating the relationship (tried; the
  natural-seeming "output_extent is where the window first goes empty" isn't
  even true in general, see the test below), so full algebraic unification
  isn't achievable, only co-location — which is now real co-location in one
  shared module, not just "side by side in the same file." What's load-bearing
  for staying correct: `test/native/compute_test.ml`'s "windowed axis:
  output_extent and bounds agree" cross-checks them against each other for
  several `(kernel,stride,pad,in_extent)` configs — if the two formulas'
  relationship broke, this is what would catch it, the practical mitigation
  for the drift risk (see the `tap`/`Dim.delta` and `Reduce.lo/hi : int`
  drifts already found in this doc) when true code sharing isn't available.

`Op_config.Pos` (stride ≥ 1, never 0) already pays for itself here too: the
division in `out_hw` would be undefined at `stride = 0`, which is exactly the
value `Pos.of_int` was added to reject — the earlier op-config typing
decision holds up under this new requirement without needing revision.

Instantiating chooses the world, with **no change to the op**:

```ocaml
module Conv_direct = Conv2d.Compute (Direct)            (* pixel : ... -> coord -> float *)
let module S = Symbolic.Make () in
module Conv_sym    = Conv2d.Compute (S)                 (* pixel : ... -> coord -> Expr.t *)
```

### 2c. `MaxPool2d` — the second windowed op, extracting `Window_axis`

`lib/native/ops/pool.ml`, the engine's `max_pool2d` (the ATen op resnet18's
graph actually calls, `max_pool2d_with_indices`, also returns argmax
indices for backprop — irrelevant for an inference-only engine, not
modeled). Structurally simpler than conv: no `in_channels`/weight, no `cin`
reduction — each output channel reduces only its own channel's kernel
window via `max_reduce`, never mixing channels. `params` is `Conv2d.params` minus
`in_channels`, exactly as `.ai/native_op_config.md` anticipated when it was
written ("the same kernel/stride/pad shape is expected to carry over
unchanged to maxpool/avgpool"). (Named `MaxPool2d`, not `Pool2d` — once
`AvgPool2d` landed alongside it below, the originally unprefixed name would
have been ambiguous between "pooling in general" and "this one pooling op.")

This is the second windowed op the §4 "Resolved" section and the old §2a
note were both waiting for before extracting a shared module — guessing the
cut from one call site (`Conv2d` alone) risked baking in something
conv-specific. With `MaxPool2d` actually needing the identical bounds
computation, `lib/native/ops/window_axis.ml` now holds it once:
`output_extent` (the `output_shape` direction, used by `Conv2d.output_shape`
and both pooling ops' `output_shape`) and `Compute.bounds` (the
clipped-reduction direction, used by `Conv2d.Compute` and both pooling ops'
`Compute`, each via `module Wa = Window_axis.Compute (S)`).

For max-pooling, the clipped bounds aren't just architecturally cleaner than a
guarded read — they're *required* for correctness. `max_reduce` over real,
possibly-negative activations cannot tolerate the padding region silently
contributing `0`: `0` would beat every real value in a window of all-negative
inputs, which is exactly what `test/native/compute_test.ml`'s "negative
inputs catch a padding=0 bug" test checks (and what `Direct`'s `load`-based
fallback used to be vulnerable to, before §4's fix — conv tolerated it
because `0` is sum's identity; pooling would not have).

### 2d. `AvgPool2d` and `Linear` — a constant divisor, and a windowless reduction

`AvgPool2d` (`lib/native/ops/pool.ml`, alongside `MaxPool2d`) is `params`,
`output_shape`, and the clipped `Window_axis.bounds` reduction range all
unchanged from `MaxPool2d` — only the reduction itself differs: `S.sum`
instead of `S.max_reduce`, then `S.div` by the kernel area (`p.kernel.h *
p.kernel.w`, a position-independent constant). That constant divisor is the
ATen default (`count_include_pad=true`, `divisor_override=None`): every
window divides by the FULL kernel area even where part of it falls in
padding, so a 2x2 kernel touching a padded corner with only one real tap
still divides by 4, not 1 (`test/native/compute_test.ml`'s
"count_include_pad=true divides by the full kernel area" test exercises this
directly with a constant-1 input, where the result is otherwise just the
real-tap count). This is also why `AvgPool2d` tolerates `sum`'s ordinary
zero-padding behavior the way `MaxPool2d` couldn't: the clipped bounds make
the padding contribute nothing to the *numerator*, and the constant divisor
handles the *denominator* — no guarded read needed for either.

`Linear` (`lib/native/ops/linear.ml`), the engine's `addmm`/fully-connected
layer, is `Conv2d`'s `cin` reduction with the windowed kh/kw loop and
`in_channels`/weight's `[Cout,1,1,Kh,Kw,Cin]` layout *removed down to*
`[Out,1,1,1,1,In]` — there is no spatial kernel at all (resnet18's FC layer
runs on an already globally-pooled, 1x1-spatial tensor, so a kernel would be
vacuous). `params` is just `{ in_features : Dim.extent Dim.t }`; `pixel` is a
single `S.sum` over `in_features`, no `Window_axis` involved — the windowed
machinery genuinely doesn't generalize to it, rather than being a missed
reuse opportunity.

## 3. `Direct` — concrete evaluation

```ocaml
module Direct : SEMANTICS with type t = float = struct
  type t = float
  type 'role index = int
  let const x = x
  let add = ( +. )  and sub = ( -. )  and mul = ( *. )  and div = ( /. )
  let exp = Stdlib.exp  and sqrt = Stdlib.sqrt
  type b = bool
  let lt a b = a < b
  let select c a b = if c then a else b
  let index_zero = 0
  let index_extent e = (e :> int)  and index_const n = n
  let of_index i = i
  let index_add = ( + )  and index_scale k i = k * i  and index_min = Stdlib.min
  let clamp_low x = Stdlib.max 0 x
  let assume_index x = x
  type input = Tensor.packed
  let load inp idx = Tensor.read_at inp idx   (* strict: an OOB index raises *)
  let sum ~lo ~hi f =
    let rec loop i acc = if i >= hi then acc else loop (i+1) (acc +. f i) in
    loop lo 0.
  let max_reduce ~lo ~hi f =
    let rec loop i acc = if i >= hi then acc else loop (i+1) (Float.max acc (f i)) in
    loop lo neg_infinity
end
```

## 4. `Symbolic` — an expression IR for codegen/fusion

`t = Expr.t`, a pure formula over `Load`s. `Reduce` carries explicit bounds so the
codegen can emit a loop; `Load` carries a symbolic index expression (affine in
the loop and reduction variables) so reads survive lowering and fusion.

`sum` and `max_reduce` mint fresh reduction-variable IDs — they need mutable state. Rather
than a global counter with a `reset()` that callers must call before each use (fragile,
non-reentrant, impossible to use correctly in two threads), `Symbolic` is a
**generative functor**: each `Make()` application allocates its own independent counter.
No global state, no reset, no hidden ordering dependency between tests.

```ocaml
module Make () = struct
  type t = Expr.t
  type 'role index = Expr.index_expr
  type input = Tensor_sig.t

  (* … all pure SEMANTICS ops unchanged … *)

  let c = ref 0                      (* local counter, one per Make() application *)

  let sum ~lo ~hi f =
    incr c; let v = !c in
    Expr.Reduce { kind = Sum; var = v; lo; hi; body = f (Expr.Reduce_var v) }

  let max_reduce ~lo ~hi f =
    incr c; let v = !c in
    Expr.Reduce { kind = Max_reduce; var = v; lo; hi; body = f (Expr.Reduce_var v) }
end

(* pure: not inside Make; the same constant regardless of which node is being
   built — every axis is just a placeholder [Index_var], not a real per-pixel
   position (unlike [Direct]'s [out], which really is one), so this is built
   once, not per pixel or even per [pixel] call. *)
let out_vec : Expr.index_expr Vec6.t =
  Vec6.make ~n:(Expr.Index_var N) ~t:(Expr.Index_var T) ~d:(Expr.Index_var D)
    ~h:(Expr.Index_var H) ~w:(Expr.Index_var W) ~c:(Expr.Index_var C)
```

Canonical usage at a call site (e.g. in a test):

```ocaml
let module S = Symbolic.Make () in
let module Cs = Conv.Conv2d.Compute (S) in
let e = Cs.pixel p ~x_shape ~x:xs ~weight:ws ~bias:bs Symbolic.out_vec in
```

The state monad alternative (threading a counter through every `SEMANTICS` value
operation) was considered and rejected: it would force `Direct` into an identity
monad wrapper with no benefit, adding verbosity to every call site while giving
`Symbolic` nothing that `Make()` doesn't already provide cleanly.

A singleton `module Symbolic = Make()` in the library would work for the common
single-threaded case but breaks as soon as two independent subgraphs need separate
expression numbering (e.g. parallel fusion passes) — the generative functor handles
both cases identically.

### Bounds clipping — windowed reductions never produce out-of-range positions

Windowed ops clip the reduction bounds so the source position `out*stride + k − pad`
is always in `[0, in_extent)` by construction, never a generate-then-guard read:

```
lo = clamp_low (pad − out*stride)              (* = max 0 (pad − out*stride) *)
hi = index_min kernel (in_extent + pad − out*stride)
```

For `MaxPool2d` this is required for correctness: a guarded-read-returns-0 fallback
would be wrong for `max_reduce` over possibly-negative data. The bounds are `index` (not `int`)
so `Direct` (`index = int`) and `Symbolic` (`index = index_expr`) share the same formula.
`Window_axis.Compute(S).window` computes them once per pixel per axis, and returns
`{lo; hi; src}` — `src k` being the source position for offset `k`, typed as `index`
(using the one `assume_index` in the engine, justified by the clip above). See
`window_axis.ml` and §2c.

### What the symbolic IR buys (the codegen/fusion goal)

- **Fusion = substitution.** A consumer's `Load (producer, idx)` is replaced by
  the producer's own `pixel` expression with its loop vars bound to `idx`. So
  `relu(add(conv(x), y))` becomes a single expression → a single fused loop nest,
  no intermediate tensors. Pointwise chains fuse for free; producers with
  reductions fuse into the consumer's body.
- **Requantization elision.** When fused ops cross a quant boundary, the IR holds
  `quantize(... dequantize(load) ...)`; the codegen folds `dequant∘quant` pairs
  (or lowers to the integer-accumulator path) instead of round-tripping memory.
- **CSE.** Per-pixel scalar granularity recomputes shared window loads; common
  subexpression elimination over the `expr` recovers them before emitting code.

A lowering pass (`codegen.ml`, Phase 4) walks `(expr, schedule)` → a loop-nest IR
→ emitted OCaml (or a future backend). It is *not* needed for correctness: a
plain recursive `eval : env -> expr -> float` interpreter over `Symbolic.pixel`
must equal `Direct.pixel`, which is the Phase-3 verification test.

## 5. The schedule — the external iterator

The driver owns the loop nest over the output coordinate space and is independent
of any op. It binds loop variables, calls `pixel`, and stores the result
(requantizing for a `` `Quant `` output).

```ocaml
val for_each : shape -> (coord -> unit) -> unit
(* default: nested loops with C innermost (channels-last write locality) *)

val evaluate : out:('e,'b,'q) Tensor.t -> (coord -> float) -> unit
(* for each output coord: store_at out coord (pixel coord), requantizing if `Quant` *)
```

`evaluate` drives a `Direct` pixel (a partially-applied function that already
reads real data, taking just the output coord); the `Symbolic` side needs the
same loop but over an `Expr.t` that has no data of its own — that's `ground`:

```ocaml
val ground : Vec6.shape -> binding:(Tensor_sig.t -> Tensor.packed) -> Expr.t -> Tensor.packed
(* for each output coord: store (Expr.eval ~binding ~coord e) *)
```

A `Compute (Symbolic).pixel` is built once (it's pure data — an expression
tree); `ground` is what turns it back into a real tensor, and it can be called
repeatedly with different `binding`s without rebuilding the expression. This
is the practical point of the symbolic IR being separate from evaluation: the
formula for "2x2 box-filter conv" is the same `Expr.t` whether it's applied to
one input image or fifty — only `binding` (and the coords iterated, if the
output shape changes) varies per call. `test/native/symbolic_test.ml` exercises
this directly: one `pixel` call per op, several `ground` calls against
different concrete inputs, each checked against `Direct` on the same input.

Schedule variants are swapped without touching the op (the "external iterator may
execute for each output pixel"):

| Schedule | What changes |
|---|---|
| naive | C-innermost nested loops |
| tiled | block H/W (and C) for cache reuse of input windows |
| parallel | split N (or H) across `Domain`s; pixels are independent |
| vectorized | batch the C axis (channels-last → contiguous lanes) |

For the symbolic path, the schedule is *encoded into* the lowered loop nest
rather than executed as higher-order calls, but it is the same set of choices.

## 6. Worked dispatch sketch (Phase 5 tie-in)

`Native_interp` mirrors `lib/interp/interp.ml` but dispatches a graph `target` to
an op functor instantiated at `Direct`, then `evaluate`s into a fresh output
tensor:

```
"…aten.relu.default"        -> let module M = Relu.Compute (Direct) in evaluate ~out (M.pixel x)
"…aten.add.Tensor"          -> let module M = Add.Compute  (Direct) in evaluate ~out (M.pixel a b)
"…aten.convolution.default" -> let module M = Convolution.Compute (Direct) in evaluate ~out (M.pixel p ins)
```

`~out`'s shape itself comes from `Relu.output_shape`/`Add.output_shape`/
`Convolution.output_shape` (§2b) applied to the graph node's already-known input
shapes — not from the `.pt2` node's declared output shape (which exists, but
checking the two agree is exactly the kind of guard the ATen `Interp.run`
path already does and this one should too; out of scope here).

Same env-threading and arg-decoding-by-name as the ATen interp; the only
difference is the dispatch target and that tensors are native `packed` values.

## 7. Tests

- **Phase 1/2 (Direct):** each op element-wise `allclose` against the ATen path on
  random tensors and on real `.pt2` weights (reuse the `PT2_DATA` gating).
- **Phase 3 (Symbolic):** for each op, `eval (Symbolic.pixel c) = Direct.pixel c`
  over sampled coords — equivalence of the two instantiations.
- **Phase 4 (codegen):** fused-vs-unfused agreement; dequant/quant-elision keeps
  results within one quantum of the dequant-to-float path.
