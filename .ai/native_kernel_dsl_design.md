# Native executable kernel DSL and fusion

Status: **Kernel IR and Region Foundation implemented**, August 2026; later
Region-native lowering, locality scheduling, and Loop IR phases remain
proposed.

Landed: the semantic Kernel IR (`lib/native/kernel.ml`), its `Stage_program`
adapter (`kernel_adapt.ml`), the reference interpreter (`kernel_eval.ml`),
capture-safe one-edge elaboration (`kernel_elab.ml`), the first fusion planner
(`fusion_plan.ml`), and the Region Foundation.  Every `Kernel.Value.t` now
stores `computation : Region_program.t`; the O(1) `Region_program.pixel`
embedding preserves the existing `Expr.Value.t` object, while non-degenerate
scalar-local programs have validated scope, partition, specialization, and a
reference executor.  Conv2D followed by Add executes with the conv's buffer
eliminated, bit-for-bit identical to materializing both stages.

`Compute` became `Legacy_pixel` for the three Region-authored operations
(RMSNorm, LayerNorm, Softmax); it is unchanged for every other operation.
Grounding (transform verification) routes through
`Stage_program.Stage.pixel_body`.  The Kernel layer is no longer additive:
Native and Native4D's Direct and Symbolic drivers, Kernel adaptation, and the
Model Explorer export all consume `Region_program.t`.  See
`region_compute_design.md` for the landed design.

Still proposed: Loop IR and its interpreter, the JavaScript and C emitters,
analysis-driven fusion, and solver-backed symbolic-coordinate comparison. The
sections below marked as sequence phases describe those.

## Decision summary

The proposed DSL should not start as a second, independent description of each
operator. The existing `Compute(S)` functors are already an embedded DSL:
`Direct` interprets them as numbers and `Symbolic` interprets the same program as
an `Expr.Value.t` construction computation. The scalar language is now a
separate, implemented `Expr` library; the safest next step is therefore to wrap
its completed expressions and `Stage_program.t` in an executable **kernel IR**,
rather than introduce another handwritten operator language that can drift from
`Compute`.

Use two levels of IR:

1. **Kernel IR** describes what one or more output elements mean as pure
   expressions over tensor loads, output coordinates, and ordered reductions.
   Fusion, dependency analysis, and semantic comparison happen here.
2. **Loop IR** describes how a kernel runs: loops, local variables, reduction
   accumulators, loads, explicit rounding, and stores. JavaScript and C printers
   consume this IR.

Keep graph optimization and code-generation fusion distinct:

- A graph rewrite changes the semantic graph and continues to produce a
  `Graph_map` with an `Identical` or `Equivalent` claim.
- A fusion plan groups existing stages and removes intermediate storage. It does
  not need to rewrite the graph. When it retains an explicit float32 round at
  every eliminated materialization boundary, it implements the same graph
  bit-for-bit.

For example, exact Conv2D + Add fusion is conceptually:

```text
conv_value = round_f32(conv_expression(...))
result     = round_f32(conv_value + residual[...])
```

The first round stays even though `conv_value` is no longer stored in a tensor.
Removing it is a different, numerical optimization and must not be called
`Identical`.

## Goals

The DSL should provide one concrete, inspectable AST that can:

- be interpreted directly;
- represent the current `Compute` semantics;
- be composed into fused kernels without manually reimplementing operators;
- be analyzed for dependencies, reduction axes, backend support, and cost;
- be lowered deterministically to JavaScript and later C;
- support mechanical comparison of two matched graph clusters at a symbolic
  output coordinate;
- preserve the repository's exact stage-rounding, max/NaN, layout, and index
  semantics;
- retain `Map_verify` as an independent integration oracle while a stronger
  symbolic kernel checker is introduced.

It is not initially intended to be:

- a user-facing textual language or stable serialization format;
- a general-purpose programming language;
- a replacement for graph shape inference;
- a promise that every legal graph cluster is profitable to fuse;
- a proof of JavaScript or C compiler correctness;
- a solver for arbitrary floating-point algebra.

## What already exists

The repository has most of the semantic front end already:

- Each operation's `Compute(S).pixel` is a pure per-output-element program over
  `Semantics.SEMANTICS`.
- `Direct` interprets that program with concrete floats and indices.
- `Symbolic` is stateless. Its value and boolean domains are `Expr.Builder`
  computations producing `Expr.Value.t` and `Expr.Bool.t`; its index domain is
  the pure, role-typed `'role Expr.Index.t`. The caller runs each completed
  computation once at the expression boundary.
- `Expr.Eval.value` is the scalar reference interpreter, with storage supplied
  through a host callback.
- `Eval_symbolic.run` builds a topologically ordered `Stage_program.t`, one
  independently closed `Expr.Value.t` stage per graph output edge.
- `Stage_program.ground` walks that DAG; `Schedule.ground` supplies each stage's
  output-coordinate loop and materializes a float32 tensor.
- `Ground_eval` expands a stage at a concrete coordinate and records
  materialization with `Ground_expr.Round`.

This means the current relationship is already:

```text
Compute(S) ── Direct ──────> concrete element
     │
     └────── Symbolic ─────> Expr.Builder computation
                                      │ run once
                                      ▼
                                Expr.Value.t
                                      ├──── Expr.Eval.value ──> concrete element
                                      └───────────────────────> analysis/codegen
```

The scalar composition primitives now exist, and the kernel wrapper, source
resolution, capture-safe elaboration and explicit storage semantics are built on
top of them (`Kernel`, `Kernel_adapt`, `Kernel_eval`, `Kernel_elab`,
`Fusion_plan`). What remains missing is a scheduled loop IR and backend
printers.

## Why there should be two IRs

A scalar expression tree alone is excellent for saying what an element means,
but poor at saying where values are stored, which reductions are shared, or how
loops should be ordered. Conversely, an imperative loop nest is awkward to
compare semantically and difficult to fuse by substitution.

The two levels keep these concerns separate:

| Concern | Kernel IR | Loop IR |
|---|---:|---:|
| Tensor element meaning | primary | inherited |
| Fusion by composition | primary | no |
| Symbolic equivalence | primary | translation validation only |
| Dependency footprint | primary | checked against schedule |
| Loop order and tiling | no | primary |
| Common subexpressions | represented as sharing | locals |
| Memory allocation | logical boundaries | physical buffers |
| JavaScript/C emission | no | primary |

A backend should never have to rediscover tensor semantics from an imperative
loop, and the semantic verifier should never need to reason about JavaScript
syntax.

## Kernel IR

### Program shape

Record types follow the repository convention: each record is a `t` owned by a
distinct module. A first wrapper can be small and namespaced under `Kernel`:

```ocaml
module Kernel : sig
  module Binding : sig
    type t = Caller | Captured_constant | Filled of float
  end

  module Input : sig
    type t = {
      id : Tensor_id.t;
      sg : Tensor_sig.t;
      binding : Binding.t;
    }
  end

  module Result_conversion : sig
    type t = Round_f32
  end

  module Value : sig
    type t = {
      id : Tensor_id.t;
      sg : Tensor_sig.t;
      computation : Region_program.t;
      result : Result_conversion.t;
    }
  end

  module Output : sig
    type t = { value : Tensor_id.t; sg : Tensor_sig.t }
  end

  type t = {
    inputs : Input.t list;
    values : Value.t list;
    outputs : Output.t list;
  }
end
```

This is deliberately close to `Stage_program.t`. A fusion group consists of a
set of stages; its inputs are edges entering the group, and its outputs are
edges leaving the group or requested as graph outputs. Internal edges are
logical values, not necessarily buffers. `result` records the conversion at a
stage boundary independently of whether a schedule stores the value. For the
current engine it is `Round_f32`: evaluate the body in working precision, round
to f32, then make that rounded value available to consumers. A fusion plan may
change a value's placement from materialized to virtual, but it does not change
this result conversion.  For a Pixel-form computation the emitter is the
original expression; a non-degenerate Region computation applies the same
conversion once to its emitter, never to scalar locals.

`Tensor_id.t` can initially be the authoritative stored identity because
`Expr_bridge` already gives a stateless bijection to `Expr.Source.t`. Kernel
validation resolves every `Expr.Value.Load` source by that mapping and by
membership in `inputs` or `values`; `Expr` itself remains independent of native
tensor and graph types. Synthetic entries from `Stage_program.consts` become
ordinary kernel inputs with `Filled`, rather than evaluator-only special cases.

Multiple outputs matter. Pooling with indices already has paired results, and a
stage used both inside and outside a fusion group may need both a fused use and a
store. A one-expression/one-output kernel model would quickly become a special
case factory.

## `Expr` foundation after the refactoring

The scalar-language refactoring is complete. Kernel work consumes the current
API; it does not redesign `Expr` again.

### Implemented language boundary

`Expr` is a separate library depending only on `core` and `fmt`. Native depends
on it, never the reverse:

```text
core, fmt
    │
    ▼
  expr
    │
    ▼
  native: Stage_program, Kernel, scheduling, storage
```

That boundary determines ownership:

- `Expr.Value.t`, `Expr.Bool.t`, and role-typed `Expr.Index.t` own pure scalar,
  predicate, and index syntax;
- `Expr.Source.t` is an opaque symbolic load key, not a tensor descriptor;
- `Tensor_sig.t`, tensor identity, payload decoding, graph membership, and
  materialization belong to the native host and the future Kernel wrapper;
- output loops, tiling, buffers, stores, and backend syntax belong to Loop IR.

The public variants are private: consumers can inspect them exhaustively, while
construction goes through smart constructors. `Expr.Index.t` retains
position-versus-delta roles in the AST; `Expr.Value.Load` accepts only position
indices; `Expr.Index.assume_position` remains a visible proof obligation instead
of erasing the claim.

Constants remain OCaml `float`, but `Expr.Value.compare`, `equal`, and `hash`
use their binary64 bits. `Round_f32` is already a first-class value form. The
compact max-pool family is already an `Expr.Intrinsic` whose checked geometry and
selection semantics are shared by the interpreters.

### Construction and stage boundaries

Reducer allocation is purely state-passing:

```ocaml
Symbolic.t = Expr.Value.t Expr.Builder.t
Symbolic.b = Expr.Bool.t Expr.Builder.t
```

`Symbolic` has no `Make` functor and no mutable reducer supply. Every combinator
threads the immutable supply, and `Eval_symbolic` runs each completed stage
computation exactly once. Each stage starts from `Expr.Builder.initial`, so two
closed stage bodies deliberately reuse reducer ordinals. An ordinal has meaning
only inside the expression that binds it; cross-stage numerical uniqueness is
not a Kernel invariant.

Using one `Symbolic.t` twice runs its construction twice. If it contains a
reduction, the resulting scalar tree contains two alpha-equivalent reductions
with different identities. That is semantically pure but can duplicate work.
Kernel-level tensor sharing must therefore come from the logical value DAG, not
from OCaml physical sharing or from assuming a symbolic computation is memoized.

### Existing analysis, validation, and composition API

Kernel code must use the existing trusted operations:

- `Expr.Fold` supplies sources, output axes, free reducers, binder inspection,
  intrinsic count, and size/depth measurements;
- `Expr.Check.value` rejects free reducers, a binder repeated on one lexical
  path, and configured size/depth excesses. Its two budgets share one metered,
  early-aborting traversal; Kernel must run it before any unmetered full-tree
  analysis of an elaborated body;
- `Expr.Rewrite.freshen` renames bound reducers without capturing free ones;
- `Expr.Rewrite.substitute_output` replaces output coordinates while leaving
  reducer variables alone;
- `Expr.Rewrite.map_sources` changes load keys without rebuilding syntax by
  hand;
- `Expr.Rewrite.alpha_normalize` and alpha-aware `Expr.Value.compare` provide
  deterministic structural comparison;
- `Expr.Eval.index` uses checked host-`int` arithmetic and `Expr.Eval.value`
  provides the independent scalar interpreter;
- `Expr.Pp` assigns reducer display names lexically, independent of allocation
  history.

These construction and traversal paths hold no mutable supply, measurement
budget, rewrite state, or printer counter. Kernel elaboration should continue
the same explicit state-passing model rather than recreate a private reference
around `Expr.Builder.state`.

`Expr.Check` intentionally does not prove that a position index is below a
tensor's upper extent: `Expr` owns neither shapes nor tensors. Kernel validation
must resolve each source to a `Tensor_sig.t`; static upper-bound proof belongs to
Kernel's later index/footprint analysis, while concrete interpretation continues
to reject an out-of-range load.

### Remaining Kernel-specific limitations

The refactor resolves typed syntax, constant identity, explicit f32 rounding,
pure reducer construction, alpha-aware comparison, and capture-safe rewrite
primitives. It does not resolve these higher-level concerns:

- stage bodies are separate trees connected only by `Expr.Source.t` membership;
- result conversion is not yet attached to each logical stage in the symbolic
  Kernel representation, even though grounding records it;
- repeated physical expansion can duplicate expensive producers;
- `Expr` deliberately has no scalar `Let`; whether kernels need one is still an
  evidence-driven question;
- max-pool remains a compact intrinsic requiring dedicated analysis and lowering;
- `Expr.Index` uses checked host `int`, while dense storage offset products and
  backend address domains require separate aggregate checks;
- `Expr.Fold` is a small set of reviewed queries, not yet a general kernel
  footprint or use-count analysis.

These are reasons to add Kernel and Loop IR, not reasons to reopen the Expr
representation.

### Working values versus stored formats

There are three different notions that should not be collapsed into one type:

- **working value**: currently an OCaml binary64 `float` used by arithmetic;
- **result conversion**: currently an f32 round followed by widening back to the
  working value when a consumer reads it;
- **payload format**: F32, F16, BF16, integer, or quantized storage described by
  `Tensor_sig.t` and decoded by a load.

`Round_f32` therefore has semantic type `work -> work`: its result is still used
by ordinary arithmetic, but it is known to have passed through binary32. It
does not require a distinct arithmetic sort called `F32_stored`. Actual integer
tensor values may justify additional value sorts later; they should be added
from concrete operation requirements rather than speculatively.

Working binary64 is intentional. Current OCaml arithmetic is `float` arithmetic
and is rounded to float32 by `Tensor.materialize` at a stage output. Emitting
every operation as float32 arithmetic would silently change existing semantics.

### Where rounding belongs

Store stage-result conversion in `Kernel.Value.result`, and make composition
elaborate it into the scalar language:

```text
load Value(v)[i]
    ==> round_f32(substitute(kernel_value(v).body, output_coord := i))
```

`Expr.Value.Round_f32` is the scalar representation used by proof normalization
and Loop IR lowering. The authoritative placement is nevertheless the value's
result contract, not a printer guessing where a store used to be.

A fused kernel may eliminate the buffer but must retain the conversion. Loads
must also retain the source format decode semantics so F16, BF16, quantized, and
integer payloads mean the same thing across interpreters and backends.

### Sharing without tree explosion

Do not make physical tree substitution the stored representation of fusion.
Keep `Kernel.values` as an SSA-like DAG. A load whose `Expr.Source.t` resolves by
membership to a kernel value references that logical definition. Expansion for
proof or lowering should be lazy, memoized, and budgeted. This directly
represents “Conv2D then Add, with Conv2D virtual” without copying the Conv2D body
into every use.

Within one scalar body, a hash-consed arena or an A-normal lowering can recover
common subexpressions. A semantic `Let` constructor is not required for Phase 1
and adds another binder class to substitution. Add it only if real operators
need scalar sharing that cannot be represented as a kernel value. Loop IR still
uses locals freely.

This decision makes the kernel AST a DAG even while each `Expr.Value.t` body
remains a tree, and it gives cost analysis an explicit place to count virtual
uses and detect recomputation.

### Binder and builder discipline

Reduction variables need capture-safe identities. `Symbolic` allocates them from
a threaded `Expr.Builder` supply, which makes them fresh within one expression
but deliberately not across expressions. The stored Kernel DAG does not require
cross-body uniqueness; elaborating one body beneath another does.

The composition protocol is:

1. establish one `Expr.Builder` namespace for the elaborated result;
2. freshen the destination body and every inserted producer fragment by
   threading that same builder state—running `freshen` independently from
   `Builder.initial` is not sufficient;
3. substitute the fresh producer's output variables with the consumer load's
   six indices through `Expr.Rewrite.substitute_output`;
4. wrap the substituted producer in its `Kernel.Value.result` conversion;
5. validate the completed expression with `Expr.Check.value` and the configured
   budgets.

Freshening must happen before insertion. Once two nominally identical reducer
IDs have captured each other, renaming the combined tree cannot recover which
binder a reference originally meant. Comparison can then use the alpha-aware
`Expr.Value.compare`; explicit `Expr.Rewrite.alpha_normalize` is useful when a
canonical rendered or solver input is required.

Kernel should add only the recursive operation it still lacks: resolution and
lazy elaboration of logical-value loads. It must delegate binder renaming,
output substitution, source mapping, structural comparison, size limits, and
printing to `Expr` rather than introduce a second traversal with subtly
different scope rules. Direct mutation tests must cover capture, shadowing,
sibling reducer reuse, nested insertion, and repeated use.

### Reductions and intrinsics

Reduction order is semantic. The current interpreter folds from `lo` to `hi-1`
with a specified initial value. Kernel comparison may rename the binder, but it
must not treat a floating-point reduction as a commutative multiset when proving
`Identical`.

The existing opaque `Max_pool` node may remain as a first implementation step,
provided the interpreter, footprint pass, and each backend call the same
specified `Max_op` selection behavior. Longer term it can lower to a structured
ordered reduction that carries paired value/index state. Lowering it prematurely
to an ordinary maximum would lose NaN and tie behavior.

Do not force every intrinsic into the generic AST before its semantics can be
represented faithfully. Each retained intrinsic must instead provide one
coherent bundle: interpreter, printer, footprint summary, well-formedness
checks, Loop IR lowering, and proof-comparison rule. An intrinsic missing any of
those is unsupported for generated kernels.

### Migration from stages to kernels

The Expr migration is finished, and steps 1–4 below have landed without a flag
day. Step 5 remains:

1. **Done.** `Kernel.t` over existing `Expr.Value.t` stage bodies, with an
   explicit result conversion per value.
2. **Done.** `Kernel_adapt` adapts a complete or selected `Stage_program.t`,
   resolving sources by membership and keeping bodies a compact DAG.
3. **Done.** `Kernel_eval`, with a buffer-based `run` and a recursive `value_at`
   that expands logical-value loads without syntactic substitution.
4. **Done.** `Kernel_elab` elaborates one edge capture-safely through
   `Expr.Rewrite` and one threaded `Expr.Builder` namespace.
5. Lower validated kernels to Loop IR without changing `Compute(S)` or Expr.

At every step, `Compute(Direct)` versus symbolic interpretation remains the
front-end oracle, and `Stage_program.ground` remains the whole-stage oracle.

### Alternatives considered

**Redesign Expr again as part of Kernel.** The completed library already has
separate value/bool modules, role-typed indices, private construction, explicit
rounding, checked evaluation, and scoped rewrites. Replacing it now would remove
the independent `Direct`/`Symbolic` comparison and delay the kernel wrapper for
no demonstrated expressive need.

**Encode boundary-versus-logical sources in `Expr.Source`.** That would make the
scalar library depend on native tensor/stage ownership. Membership is a Kernel
property and already suffices to resolve the opaque source key.

**Store fusion as a completely inlined scalar expression.** This makes a small
Conv2D + Add printout attractive, but duplicates producers under fan-out and can
explode for nested stencils/reductions. The logical value DAG is the concrete
kernel AST; its pretty-printer may show an inlined view, and Loop IR may inline a
profitable single use.

**Put loops directly in `Expr`.** This makes code generation immediate but
couples semantic comparison to one schedule. Ordered `Reduce` belongs in scalar
semantics; output loops, tiling, locals, and stores belong in Loop IR.

### A possible surface notation

The first Kernel surface should be a typed OCaml construction API consuming
completed `Expr.Value.t` bodies. `Expr.Builder` already owns scalar reducer
construction; the Kernel API should not replace or wrap it with another reducer
supply. A textual syntax is useful for printing and tests, but should not
initially become a second source of truth.

For readability, a fused Conv2D + Add kernel could print approximately as:

```text
kernel conv_add(
  x        : tensor<N,H,W,Cin>,
  weight   : tensor<Kh,Kw,Cin,Cout>,
  bias     : tensor<Cout>,
  residual : tensor<N,Oh,Ow,Cout>
) -> y : tensor<N,Oh,Ow,Cout> {
  y[n,h,w,c] = round_f32(
    let conv = round_f32(
      reduce_sum kh in [0,Kh), kw in [0,Kw), ci in [0,Cin) {
        x[n, h*Sh + kh-Ph, w*Sw + kw-Pw, ci]
        * weight[kh,kw,ci,c]
      } + bias[c]
    ) in
    conv + residual[n,h,w,c]
  )
}
```

This notation is explanatory, not a proposed parser grammar. Shapes and valid
window bounds still come from `Graph_shape` and the validated op configuration.

## Fusion is a plan over stages

### Do not rewrite the graph merely to fuse code

For ordinary buffer-elimination fusion, retain the original graph and build a
separate `Fusion_plan.t`. A group names its member stage IDs and its externally
visible outputs. This has several advantages:

- graph debugging still shows Conv2D followed by Add;
- shape inference and graph maps do not need a synthetic `Conv_add` op;
- an intermediate can be materialized when debugging or when it has an outside
  consumer;
- legality and profitability remain scheduling decisions;
- exactness is expressed by preserved virtual stage boundaries.

Only introduce a graph rewrite when the graph itself changes, such as algebraic
simplification or a backend-specific operator replacement.

### Composition rule

Suppose producer stage `P` defines:

```text
P[o] = p_body(o)
```

and consumer body `C` contains a load whose source resolves to `P` at `i(o,r)`,
fusion changes `P`'s placement from materialized to virtual; it does not
physically copy `p_body` into the stored AST. The semantic elaboration of that
load is:

1. freshen `C` and every inserted fragment in one threaded `Expr.Builder`
   namespace, so producer binders cannot collide with consumer binders;
2. replace each fresh producer output coordinate variable with the
   corresponding index expression from `i` through
   `Expr.Rewrite.substitute_output`;
3. apply `P`'s result conversion, currently `round_f32`;
4. memoize the elaborated use instead of repeatedly copying its tree;
5. leave any externally live producer value as a kernel output/store as well;
6. run `Expr.Check.value` with configured size/depth budgets on the completed
   body.

The transformation is mechanical and local. One small, heavily tested Kernel
elaborator should orchestrate `Expr.Rewrite.freshen` and
`substitute_output`; no second scope-aware Expr traversal is permitted.
`Kernel.values` remains the compact source representation.

### Legality is not profitability

Substitution may be semantically legal while producing terrible code. A fusion
planner must consider:

- cycles and topological order;
- matching concrete output shapes and valid index domains;
- backend support for every expression node and payload format;
- fan-out and whether an internal edge is used outside the group;
- duplicated producer evaluation;
- reduction reuse, especially softmax-like stages;
- stencil halos and recomputation cost;
- data-dependent indexing;
- estimated arithmetic, load, and temporary counts;
- code-size and compiler limits.

The planned footprint interpretation from `native_symbolic_language.md` supplies
the key dependency classification. Elementwise consumers are the easiest first
target. Stencils are possible with a bounded halo. Reduction-coupled and
data-dependent footprints usually require their own materialized pass.

### First supported fusion class

Start with a single producer and a pointwise consumer where:

- the consumer loads the producer once per output element;
- the load index is the output coordinate, possibly with ordinary broadcasting;
- the producer output has no required external materialization, or an extra
  output store is explicitly planned;
- every operation is supported by the target backend;
- the composed AST is under a fixed size budget.

Conv2D followed by Add fits this class: Conv2D remains the inner ordered
reduction and Add becomes the output epilogue. More aggressive producer-into-
stencil or producer-into-reduction fusion should wait for dependency and cost
analysis.

## Loop IR

Kernel IR plus a schedule lowers to a small structured imperative language. A
representative shape is:

```ocaml
module Loop_var : sig
  type t
end

module Loop_stmt : sig
  type t =
    | Let of Loop_var.t * Scalar.t * t list
    | For of {
        var : Loop_var.t;
        lo : Index.t;
        hi : Index.t;
        body : t list;
      }
    | If of Bool.t * t list * t list
    | Store of Buffer.t * Index.t list * Scalar.t
end

module Loop_program : sig
  type t = {
    inputs : Buffer.t list;
    outputs : Buffer.t list;
    body : Loop_stmt.t list;
  }
end
```

Exact constructor details can wait for implementation, but the invariants
cannot:

- loops are structured and have explicit half-open bounds;
- expressions are pure;
- loads and stores name typed buffers and explicit indices;
- float32 rounds are explicit scalar operations;
- reducer initialization, update order, and result are explicit;
- generated variable names are deterministic;
- an interpreter exists before either text backend is trusted.

The initial schedule is the current dense six-axis order with C innermost.
Tiling, vectorization, parallelism, and layout-specific variants can be added as
separate lowering policies after the naive path is validated.

## JavaScript and C semantics

### Index and offset boundary

`Expr.Eval.index` checks every operation in the host `int` domain, but that does
not by itself establish that a tensor allocation or dense storage address is
safe. The initial Kernel admission contract is deliberately stricter and
backend-independent: tensor extents, concrete coordinates, and reachable
storage offsets stay below `2^31`.

Products and sums used to prove that contract are evaluated with checked
`int64` arithmetic before any narrowing, and each aggregate is bounded
separately. Individually valid dimensions do not prove that their product or a
stride-weighted sum is valid. Code reachable through js_of_ocaml must still
respect its narrower host-`int` range; the `< 2^31` runtime contract is not
permission to construct a too-large OCaml `int` there.

A future runtime using a wider address domain must name that domain explicitly
and retain checked overflow for every aggregate. Native OCaml's signed host
`int` is not a substitute for a `2^64` address contract, and moving the limit
does not justify validating an already wrapped result.

### JavaScript

JavaScript `Number` is binary64, which matches the working precision of OCaml
`float` closely for the current primitive arithmetic. Emit `Math.fround` only at
explicit `round_f32` nodes and output stores, not after every arithmetic node.

Generated code should use ordinary named functions and loops, not `eval` or
closures per element. Runtime helpers should make format decoding, bounds, and
special max selection visible and testable.

Tensor indices represented as JavaScript numbers must also be proven within the
integer-exact range. The stricter initial Kernel limit implies this for runtime
addresses, but the check remains a named backend obligation rather than an
accidental consequence.

### C

Using C `float` for every temporary would round after every operation and would
not implement the current interpreter. The straightforward exact lowering uses
`double` working temporaries and an explicit conversion through `float` at each
`round_f32` node. Index products and offsets should use checked `int64_t` or a
carefully bounded unsigned size type.

`exp` and `sqrt` may differ in their last bits across OCaml, JavaScript engines,
and C libraries. Backend validation therefore needs an operation-specific
claim: bitwise identity where the runtime contract supports it, and a stated
numerical tolerance otherwise. This does not weaken structural fusion proofs;
it describes the separate code-generator/runtime boundary.

## Static analyses over Kernel IR

The AST makes useful compiler questions mechanical:

- free tensor inputs and internal stage dependencies;
- free output and reduction variables;
- capture and binder validity;
- tensor-load bounds under the output/reduction domains;
- local, stencil, reduction-coupled, or data-dependent footprints;
- broadcast and reuse axes;
- pure common subexpressions and loop invariants;
- reduction nesting and estimated operation count;
- virtual versus materialized float32 boundaries;
- supported payload formats and required decode helpers;
- backend capability and code-size checks.

All analysis results should be derived from the AST, not separately asserted by
an op. Small op-specific annotations may guide cost choices, but they must not
override semantic dependencies found in the expression.

## Mechanical cluster comparison

### Compare functions, not every output element

Given a source cluster and a candidate destination cluster:

1. identify the same ordered boundary inputs and corresponding outputs;
2. lower each cluster to a `Kernel.t`;
3. introduce one arbitrary symbolic coordinate for each output axis;
4. state its domain constraints, such as `0 <= h < H`;
5. compare the two output bodies under those constraints.

If this succeeds, the result covers every coordinate in that concrete output
shape without enumerating the output tensor. It does not require or justify
removing concrete differential tests; it replaces enumeration as the proof
mechanism for the supported AST fragment.

### Tiered checker

Use deliberately narrow proof tiers:

1. **Well-formedness.** Shapes match, binders are scoped, loads are in bounds,
   boundary inputs correspond, and outputs are paired.
2. **Alpha/canonical structural equality.** Fresh names, harmless index syntax,
   and DAG sharing are normalized. Exact scalar and ordered reduction structure
   must match.
3. **Integer/index solver.** Prove corresponding load indices, conditions, and
   reduction bounds equal under coordinate and shape constraints. This handles
   identities involving floor division, reshape linearization, min/max, and
   broadcasting that are not syntactically equal.
4. **Approved scalar rules.** Apply only explicitly registered and tested rules,
   classified by claim. Floating-point reassociation is not `Identical`.
5. **Unproved.** If no tier establishes equality, preserve the candidate only
   under a policy that permits an unproved claim, or reject it. Random probes may
   find a counterexample but never turn an unproved rule into a proof.

The integer solver should be hidden behind a small interface and consume an
SMT-LIB representation of index obligations. A practical initial policy is Z3
by default with the ability to run cvc5 for cross-checking or difficult cases.
The proof report must record the solver, version, timeout, and whether the result
was `unsat`, `sat` with a model, `unknown`, or an infrastructure error.

Do not initially ask the solver to prove arbitrary floating-point expression
equivalence. Most exact fusion comparisons should have the same scalar tree once
loads and index expressions are aligned. Keeping solver work in integer/index
logic makes the trusted encoding smaller and failures more understandable.

### Concrete SMT proof model

#### The logical obligation

For corresponding cluster outputs `L` and `R`, correctness means:

```text
forall boundary tensors B, valid shapes S, and valid output coordinates o:
  eval(L, B, S, o) = eval(R, B, S, o)
```

An SMT solver normally searches for the negation:

```text
preconditions(S)
and output_in_bounds(S, o)
and eval(L, B, S, o) != eval(R, B, S, o)
```

`unsat` proves that no counterexample exists in the encoded semantics. `sat`
provides a model for the free variables in a full encoding. `unknown`, timeout,
or a solver error is `Unproved`, never success.

Mltorch should initially implement this obligation compositionally rather than
emit one enormous value formula. Boundary tensor cells remain opaque leaves;
the solver proves that corresponding leaves address the same source cell. The
trusted comparator then uses congruence of identical scalar operators to build
the whole proof.

#### Boundary tensors and cluster correspondence

The source and destination clusters first receive the same ordered boundary
variables from the `Graph_map`. A symbolic load is identified by:

```text
(boundary variable, payload/decode contract, six index expressions)
```

Two loads are equal when:

1. they name the same boundary variable;
2. their decode contracts are identical;
3. the solver proves all six indices equal under the current context.

This quantifies over every possible input payload without placing the payload
values in SMT. If two expressions read the same arbitrary cell and apply the
same decode, they receive the same value. A future full-value encoding can model
each boundary tensor as an SMT array or uninterpreted function from coordinates
to values, but that is unnecessary for structural fusion proofs.

An index query has this shape:

```text
constraints(S, o, reduction_vars)
and (lhs_index != rhs_index)
```

An `unsat` result establishes index equality. A `sat` result here is an **index
witness**, not automatically a complete numerical counterexample: surrounding
arithmetic might mask the different load. The report must preserve that
distinction. Only a satisfying model of the full output-inequality formula, or a
replayed concrete payload, is a semantic counterexample.

Model constants require the same care as graph inputs. A bound constant may be
read into concrete leaves or compared by an exact payload identity; an unbound
constant must be a mapped boundary variable. Independently numbered constants
must never be assumed equal merely because both graphs classify them as
constants.

#### Recursive proof rules

After alpha-normalization and virtual-stage elaboration, compare expressions
with a small proof calculus:

| Left and right form | Obligation |
|---|---|
| constants | binary64 bits equal |
| same binary/unary operator | recursively prove operands |
| same `Round_f32` | recursively prove arguments |
| loads | same boundary/decode, SMT-equal indices |
| selects | prove guards and corresponding branches |
| index comparisons | SMT-equivalent predicates |
| reductions | same ordered fold, then use the reduction rule below |
| supported intrinsic | apply its dedicated proof rule |
| anything else | `Unproved` |

This is not algebraic simplification. In particular, operand reordering,
reassociation, erased rounds, or a changed select structure cannot pass the
`Identical` tier merely because they look mathematically reasonable.

#### Ordered reductions without unrolling

For two reductions:

```text
reduce kind r from lo to hi: body(r)
reduce kind r' from lo' to hi': body'(r')
```

prove:

1. the kind, seed, update operation, and iteration direction are identical;
2. SMT proves `lo = lo'` and `hi = hi'`;
3. introduce one arbitrary reducer coordinate `q` with
   `lo <= q && q < hi`;
4. alpha-substitute both binders with `q` and recursively prove their bodies.

The trusted reduction rule concludes that the two folds perform the same
updates in the same order. SMT does not unroll `Cin`, and it is not asked to
prove floating-point associativity. Nested reductions apply the rule
recursively. A rewrite that changes the reduction order or factorization needs
a separate claim-aware theorem and is not covered.

#### Example: reshape versus direct indexing

One cluster might load an input through flattened and delinearized coordinates,
while the other loads it through a direct permutation. For output coordinate
`o`, the checker asks whether any corresponding input axis can differ:

```text
0 <= o.N < Out.N
and ...
and 0 <= o.C < Out.C
and delinearize(linearize(o)).axis != permute(o).axis
```

If every per-axis query is `unsat`, both clusters read the same arbitrary input
cell at every output coordinate. The scalar tree above the load then determines
whether the output expressions are identical. This proves the whole concrete
shape without enumerating its pixels.

#### Conv2D plus Add

For exact buffer-elimination fusion, virtual-stage elaboration produces this on
both sides:

```text
round_f32(round_f32(conv_body(o)) + residual[broadcast(o)])
```

The comparator proves the two ordered convolution reductions recursively. SMT
proves the input, weight, bias, and residual index expressions and bounds equal.
It does not reason algebraically about the multiply/add sequence. If the fused
kernel omits the inner `Round_f32`, structural comparison fails immediately,
which is the correct `Identical` result.

#### Integer theory and bounds

Encode tensor coordinates and shape arithmetic as mathematical SMT integers,
not machine bit-vectors. This prevents an overflow from accidentally satisfying
an index identity. Encode operations to match `Expr.Index` exactly:

- positive floor division, including negative numerators;
- ceiling division as the specified dual of floor division;
- min/max as `ite` terms;
- half-open reduction and tensor bounds;
- broadcasting conditions and positive extents.

The encoder needs differential tests against `Expr.Eval.index`, especially for
negative numerators and division boundaries. Do not assume a host language's
`/` or an SMT operator has the required rounding convention without those
tests.

Machine safety is a second obligation. After proving mathematical index
correctness, prove or check that aggregate offsets fit the chosen runtime type
and, for JavaScript numbers, the exact-integer range. Narrow to OCaml `int` only
after both individual values and aggregate products/sums have been bounded.

Current Native shapes are concrete six-axis values, so the first solver context
contains concrete extents and six symbolic coordinates. Future symbolic shapes
add integer variables plus explicit preconditions such as positivity, equality,
products, and divisibility. A missing shape precondition must produce
`Unproved`, not be inferred from the rewrite's desired result.

#### Well-formedness uses the same solver

Before equivalence, check each symbolic load independently. For index `i` into
extent `E`, refute:

```text
context and (i < 0 or i >= E)
```

An `unsat` result proves the load in bounds. `assume_position` specifically
creates such an obligation; its distinct representation in the current
`Expr.Index` AST prevents an unchecked role cast from disappearing. The same
mechanism checks positive divisors, valid reduction intervals, broadcast
compatibility, and schedule tile bounds.

Legality proof and equivalence proof should remain separate in reports. Two
expressions can be equal while both perform an invalid load.

#### Limited full floating-point tier

An optional later tier may encode small local expressions with SMT IEEE
binary64 arithmetic and explicit binary64-to-binary32-to-binary64 conversion for
`Round_f32`. It can be useful for finding counterexamples to a proposed
algebraic rule. It should not be the initial proof foundation because:

- large reductions become impractical if unrolled;
- transcendental `exp` is outside the ordinary SMT floating-point fragment;
- runtime `exp`/`sqrt` behavior is not automatically the solver's behavior;
- NaN payloads, signed zero, and mltorch's bitwise comparison contract need more
  care than the ordinary floating-point equality predicate;
- quantized and integer load decoding enlarges the trusted encoding.

Any floating-point rule must name its exact domain and claim. A real-arithmetic
proof cannot establish mltorch `Identical`; at most it supports an explicitly
defined numerical `Equivalent` claim with separate error reasoning.

#### Solver interface and evidence

Use a narrow solver interface with results such as:

```ocaml
type result =
  | Unsat of Proof_metadata.t
  | Sat of Model.t
  | Unknown of string
  | Error of Err.Error.t
```

Record the normalized obligation, solver name and version, timeout, elapsed
time, and model where available. Deterministic SMT-LIB output makes failures
replayable and permits selected CI obligations to be cross-checked with both Z3
and cvc5. The trusted surface includes cluster lowering, the recursive proof
rules, index encoding, and solver-result parsing; mutation tests must target all
four. A second solver is useful defense, but agreement between solvers does not
validate a wrong encoding.

### Fusion by construction

For a fusion plan that only makes a materialized `Kernel.Value` virtual, the
logical kernel definition does not change. Its consumer still loads the same
value at the same index and the value retains its result conversion. The
capture-safe elaboration rule turns that virtual load into the producer body at
the exact index with the old `round_f32` retained. Running the cluster comparator
is still useful as defensive validation of elaboration and lowering, but it
should normally prove immediately.

This is stronger and cheaper than inventing a special `Conv_add` implementation
and trying to rediscover that it matches Conv2D followed by Add.

## Relationship to `Map_verify`

`Map_verify` remains valuable, but has a different role.

Today it grounds symbolic expressions at concrete output coordinates, unrolls
reductions, expands stage boundaries with `Round`, and compares binder-free
terms. It is sensitive to exact cell coordinates and the graph correspondence,
which makes it an excellent integration oracle for graph transformations.

The kernel checker adds symbolic-coordinate reasoning so a supported cluster can
be checked once for all coordinates instead of one coordinate at a time. The two
should coexist:

- **Kernel comparison** validates composition and future graph rewrite rules at
  an arbitrary coordinate.
- **`Map_verify`** validates the actual `Graph_map`, cluster boundary, and current
  transformation pipeline.
- **Concrete execution** validates AST interpreters and generated backends.

Do not create a circular test by generating the AST with `Symbolic` and only
comparing it to the same `Symbolic` result. The independent observations are:

1. `Compute(Direct)` versus
   `Expr.Eval.value(Expr.Builder.run (Compute(Symbolic)))` on concrete inputs;
2. semantic Kernel interpreter versus Loop IR interpreter;
3. Loop IR interpreter versus generated JavaScript/C;
4. graph before/after through `Map_verify` for real graph rewrites;
5. deliberate AST mutations that every comparison layer must reject.

## Validation strategy

### Front-end validation

For every supported operation and a spread of valid configurations:

- build the AST through the existing `Compute(Symbolic)` path;
- interpret it at all coordinates of small shapes;
- compare bitwise with `Compute(Direct)` after the same output rounding;
- include zero, signed zero, infinities, NaNs, subnormals, boundary indices,
  empty/one-element reductions where legal, and broadcast dimensions;
- retain existing operation-specific max-pool and quantization tests.

If operations are later authored directly in Kernel IR, keep the legacy
`Compute` implementation until this differential suite establishes parity.

### Fusion validation

For each fusion pattern:

- compare unfused and fused Kernel IR symbolically;
- interpret both on small exhaustive shapes and randomized values;
- run `Map_verify` when a graph rewrite is actually involved;
- assert that removing the internal `round_f32` is rejected as `Identical`;
- test fan-out, graph-output intermediates, broadcast operands, and reduction
  binders;
- mutation-test wrong axes, wrong reduction limits, operand swaps, and missing
  format conversions.

### Backend validation

- interpret Loop IR as the backend-independent reference;
- emit deterministic JavaScript and compare every output element;
- compare bits for operations with an exact cross-runtime contract;
- use explicit, recorded tolerances for transcendental runtime differences;
- repeat the same suite for C when introduced;
- include aggregate-size overflow and out-of-range index rejection tests;
- retain end-to-end model comparisons because a locally correct kernel can still
  be wired to the wrong graph edge or shape.

## Resource limits and the cross-backend stack ceiling

Kernel construction carries budgets in four independent dimensions. They are
independent because bounding one leaves the others open, and the third exists
only because the first two compose.

| Dimension | Fields | Bounds |
|---|---|---|
| per-expression | `max_size`, `max_depth` | one `Expr.Value.t` |
| value DAG | `max_values`, `max_dep_depth` | the logical-value graph |
| interface | `max_inputs`, `max_outputs` | the public arity |
| combined | derived `eval_depth` | the recursive `value_at` stack |

Bounding only the per-expression dimension leaves an arbitrarily long chain of
one-node bodies, on which the recursive evaluator exhausts the stack before it
can return the `Err.t` it promises. Bounding the first two *independently*
is still not a stack bound: `Expr.Eval.value` descends a body and invokes the
host load callback with its enclosing frames live, so a chain of `L` values whose
producer load sits `D` nodes deep retains on the order of `D x L` frames. Hence
the derived quantity, computed iteratively over the topo-ordered values:

```text
eval_depth(v) = Expr.Fold.depth(Result_conversion.apply v.result v.body)
              + max over v's value-sources p of eval_depth(p)
```

with inputs contributing zero. It measures the CONVERTED body, not the raw one:
every consumer — a store, a load, `value_at` — evaluates
`Result_conversion.apply`, so the conversion node is a level the evaluator
really walks, and measuring `v.body` undercounts each value by it. It
over-approximates when a body's deepest branch holds no load, which is the safe
direction.

### A static level count does not bound a recursive evaluator

`eval_depth` bounds a *flat* expression. It does not bound
`Kernel_eval.value_at`, whose recursion crosses a producer transition per value,
and a transition costs far more stack than an expression level. Calibrating the
ceiling against a flat 2048-deep expression was therefore wrong. The failing
configuration, measured under the ORIGINAL raw-body formula, was a 1024-value
chain whose computed `eval_depth` came to exactly 2048: accepted by
`Kernel.create`, then overflowing under node. Conversion-inclusive validation
now rejects that particular shape, but only incidentally — the frontier is still
not a level count, which is why the real bound moved to runtime.

The frontier is also **unstable and not modelable** — that same chain passed
once before failing three consecutive runs, and 384 transitions over depth-4
bodies overflows while 192 transitions over depth-16 bodies does not, so
transition count dominates but no simple weighted sum separates the observed
safe and unsafe points.

So the recursion is bounded at **runtime**, by `Hard.eval_recursion`, where the
recursion actually is. Rejecting deep DAGs at construction would have been the
wrong fix twice over: the buffer-based `run` never recurses, so a long chain
executes perfectly well, and a limit low enough to be safe (~128 transitions)
would sit uncomfortably close to a real model's layer count. The static
`dep_depth` and `eval_depth` limits remain as cheap early guards on the stored
DAG; the runtime bound is what makes the no-overflow promise true.

The general rule: a resource guard belongs at the recursion it guards, and a
count of *structure* is not a count of *frames*.

### The `Hard` ceilings are measured, not chosen

Custom limits may tighten any ceiling, never widen it — including the per-body
ones, because `Expr.Check.value` is itself recursive and bounded by the
*configured* depth, so an over-wide `max_depth` lets validation overflow before
it reaches the limit it was asked to enforce.

`Hard.depth` and `Hard.eval_depth` are empirical stack limits under **node**,
which has the tighter stack. Measured on this tree, natively every traversal
survives depth 16384; under node:

| Depth | Node result |
|---:|---|
| 1024 | all survive |
| 2048 | overflow in `Pp.value`, `Value.compare`, `Value.hash` |
| 4096 | additionally `Check.value`, every `Fold`, every `Rewrite`, `Eval.value` |

Two consequences fix the constants:

- **`Check.value` is not a proxy for the traversals that follow it.** It still
  survives at 2048, where the printer and the structural comparison already
  fail. Since Kernel's safety argument is "check first, then unmetered `Fold`",
  `Hard.depth` must sit below the *minimum* over every recursive traversal
  applied to a validated body — not below the checker's own threshold. Adding a
  traversal to `Expr` means re-running the probe.
- **`Eval.value` is the outlier upward**, surviving 4096 and failing at 8192, so
  the combined ceiling is legitimately higher than the per-body one. It has to
  be: a whole-program resnet18 kernel reaches roughly 70 layers x 11 levels of
  combined depth, and a bound below that would reject a model the buffer-based
  evaluator never recurses through.

`Hard.depth = 256` (8x margin under node's 2048, 18x the largest observed body
depth) and `Hard.eval_depth = 2048` (4x margin, 2.6x resnet18). The remaining
`Hard` values bound memory and time rather than stack. `Hard.extent` and
`Hard.numel` are `2^31`, the JS-reachable runtime domain.

`test/native/depth_probe.ml` pins both constants on both backends and is what
makes the claim falsifiable — a compiler, jsoo or node change that lowers the
real threshold turns `make jsoo.inline-runtest` red.

Defaults are census-derived with headroom over the largest observed values
(body size 246, body depth 14, 36 stages, dependency depth 6, 17 inputs), not
guessed; the per-graph census is regenerated as gitignored scaffolding rather
than tracked, so the selected values and their basis live here.

## What construction validates, and what it deliberately does not

`Kernel.create` is a validating public boundary, not an adapter-internal
helper, so nothing about a hand-built kernel is a trusted precondition. Two
rules are worth stating because they are not obvious from the type:

**A locally materialized signature must be honest.** `Tensor.materialize` always
produces an f32, unquantized payload and is the only materializer the evaluator
has, so a `Filled` input or a stored value declaring f16 or a quantized format
would be handed a tensor that does not match what every analysis says it is —
`Expr_bridge.env` would decode the real f32 payload regardless. Caller and
captured-constant inputs carry real data and stay free to be f16, bf16 or
quantized. `Graph_builder.new_edge` already gives every op output `~fmt:f32`, so
no graph built through the builder can violate this; the two fixtures that do
reach it patch a format afterwards specifically to construct a state the builder
cannot.

**Quantization is checked in both directions.** `quant` must be present exactly
for a quantized format — the `Payload` GADT enforces that for real tensors
through its `` `Real ``/`` `Quant `` tag, but `Tensor_sig.t` is a plain record
that cannot — and a per-channel value's array length must equal the C extent,
asked through `Quant.channel_count`. That the two arrays agree with *each other*
is `Quant.per_channel`'s invariant, established at construction.

There is deliberately **no** "output axis extent > 0" rule: `Vec6.shape`
components are `Dim.extent`, valid from 1 by construction, so such a check could
never be turned red. Likewise no output-signature mismatch, because `create`
derives each `Output.sg` from the value it names.

### Adapting a `Stage_program`

`Stage_program.t` is a public record, so the adapter treats it as untrusted and
validates the **whole definition table** before projecting any selection.
Skipping that would let a selection launder a structural defect: with stage `a`
reading a later stage `b`, selecting only `a` turns `b` into a synthetic
boundary input and `create`'s forward-reference rule never sees it.

Every boundary rule reads **every source**, never only the loads.
`Symbolic.max_pool` names its input solely inside the intrinsic descriptor, with
no `Load` in the body, so a loads-only rule would fail to create a synthetic
input for a selected max-pool's outside producer and — worse — fail to mark a
producer live when its max-pool consumer sits outside the selection, letting a
later fusion plan drop a store that consumer still needs.

Three further contracts:

- **Boundary inputs are derived** from the selected bodies' free sources, not
  copied wholesale, so a kernel cut from one branch does not demand bindings for
  another branch's inputs — the evaluator validates every declared input before
  computing anything, so an inherited one becomes a spurious failure.
- **Dead terminals are promoted, not sliced.** A `Stage_program` may legitimately
  contain a stage nothing consumes: `Eval_symbolic.run` emits one per node output
  while a `Discard` node emits none. `Stage_program.ground` materializes every
  stage, so dropping them would change behaviour under the guise of adapting it.
- **Pass-through graph outputs are rejected, not filtered.** A graph output that
  is a boundary input has no value to name; silently returning a kernel that
  computes fewer outputs than its source program is the one failure no
  downstream comparison would catch.

## Placement, and where the retained round is observable

Fusion never rewrites `Kernel.t`. A `Fusion_plan.t` records placement beside an
unchanged kernel and **carries the kernel it was built for**: `Tensor_id.t` is a
small integer local to a graph, so a plan built for kernel A is otherwise
silently applicable to kernel B — with overlapping ids the evaluator would skip
stores or virtualize unrelated dependencies, and without them it would return an
incomplete result map.

Placement is **two facts**, not one enum: which dependency *edges* are virtual,
and which *values* need stores. An externally live producer is both. A single
`Materialized | Virtual` per value cannot say that.

`stores` is derived, never accumulated: a value is stored unless it is the
producer of a virtual use. Placement closure — every load surviving elaboration
reads a buffer — follows, but from the **unique-use rule**, not from the
one-edge representation. Inlining `P` into `C` leaves `P`'s own loads of some
`Q`; were `Q` virtualized on another edge it would have two ordinary uses
(`Q→P` and `Q→D`) and be rejected as `Multiple_uses`.

### The round is only observable where nothing is stored

This is the subtlety the whole acceptance case turns on. `Tensor.materialize`
writes into a **float32** bigarray, so a stored value is rounded at the store
whether or not the kernel says so. Removing `Result_conversion.apply` therefore
changes nothing for a materialized value — `run` still matches
`Stage_program.ground`.

It becomes load-bearing exactly when a value is **not** stored, which is the
virtual case. With Conv2D virtual there is no store to round it, and the
explicit `Round_f32` is the only thing between the fused result and `2^24+2`.

Two consequences for testing, both learned by watching a check fail to fail:

- fixture data of small integers cannot test the rule at all, since every result
  is f32-exact. A discriminating value — 2^24+1 is the smallest positive integer
  binary32 cannot hold — is required;
- the elaborator's conversion and the evaluator's are *separate* sites. A test
  exercising one says nothing about the other, and each needs the
  non-representable value to be observable.

### Elaboration is site-local; the planner adds placement policy

`Kernel_elab` admits one edge with one occurrence of that producer in that
consumer and a shape-compatible pointwise coordinate — `Output` axis with equal
extents, or `zero` where the producer extent is 1. The extent equality is not
decoration: `create` does not prove symbolic load upper bounds, so a producer of
extent 1 under a consumer of extent 2 satisfies single-site syntax, and while
both execution paths bounds-check such a load, substituting the producer *body*
carries no such check and a constant producer would quietly succeed.

The class is decided **before** rewriting. `load_uses` is a set of pairs and
loses site multiplicity, so a legitimate pair occurring many times would take a
freshened producer copy per occurrence; checking the finished tree cannot
prevent building it.

The planner's rules — unique ordinary use, no intrinsic use, pointwise consumer
— are about *placement*, not about bounded rewriting, and are deliberately not
the elaborator's. An edge the planner refuses may still elaborate correctly.

Candidate enumeration reads `uses`, not `load_uses`: an edge reaching its
producer only through an intrinsic descriptor must be *reported* as
unfusable, and enumerating loads alone would leave that rejection unreachable
and produce no decision for the producer at all.

## Implementation sequence

### Phase 1: make the current AST a kernel — DONE

- [x] `Kernel.Input`, `Kernel.Value`, `Kernel.Output` and `Kernel` over selected
      `Stage_program` stages.
- [x] Sources classified as boundary or logical-value by membership, with an
      explicit result conversion on every kernel value.
- [x] Composition through `Expr.Rewrite` only. It gained
      `Rewrite.substitute_loads` and the `Fold.loads` / `intrinsic_sources`
      queries, so `lib/native` still holds no second scope-aware traversal.
- [x] Virtual loads elaborated with explicit `round_f32`; the stored
      representation stays a value DAG.
- [x] Every completed and elaborated body validated through `Expr.Check.value`
      under configured limits.
- [x] A Kernel interpreter plus a placement plan separate from the kernel.
- [x] Pointwise-consumer fusion only.

Exit criterion **met**: `conv_add` executes with the conv buffer eliminated,
bit-for-bit identical to the two materialized stages, and each of the two
mutations that would drop the intermediate float32 round turns a test red.

Two contracts were added during implementation that the phase plan did not
anticipate, both because a check turned out to be unfalsifiable as first
written: recursion through the value DAG is bounded at runtime rather than by a
static level count (see above), and locally materialized signatures must be
f32/unquantized because `Tensor.materialize` produces nothing else.

### Phase 2: naive Loop IR and JavaScript

- Lower Kernel IR with the current dense C-innermost schedule.
- Interpret Loop IR.
- Emit readable deterministic JavaScript.
- Validate format loads, reductions, rounding, max selection, and stores.

Exit criterion: generated JavaScript matches the Kernel and Loop interpreters on
the supported operation suite and an end-to-end small graph.

### Phase 3: analyses and fusion planning

- Implement dependency footprints.
- Add cost and backend-capability checks. Fan-out and external liveness are
  already in the Phase 1 planner — fan-out as `Multiple_uses`, liveness as the
  `also_stored` half of a placement — but only as conservative rejections, not
  as a cost model that could accept a profitable duplicate.
- Add common-subexpression elimination and loop-invariant motion.

`Fusion_plan` is already separate from `Graph_ir`, and falling back to full
materialization is `Fusion_plan.default`.

Exit criterion: the planner accepts the intended Conv2D epilogues and rejects or
materializes reduction-coupled, data-dependent, and excessive-recomputation
cases with an explanatory report.

### Phase 4: symbolic-coordinate comparison

- Normalize binders and index expressions.
- Emit well-formedness and integer/index equivalence obligations through a
  solver interface.
- Compare arbitrary-coordinate load leaves and ordered scalar/reduction trees.
- Record replayable queries, solver metadata, and the distinction between an
  index witness and a complete semantic counterexample.
- Integrate results into transformation verification reports without weakening
  current policy.

Exit criterion: supported rules are proved once per output expression rather
than by output-coordinate enumeration, and seeded mutations return a concrete
counterexample or `Unproved`.

### Phase 5: C and optimized schedules

- Add a C printer over the same Loop IR.
- Add tiling and other schedules only with footprint legality checks.
- Benchmark recomputation, code size, and memory traffic before widening fusion.

## Important failure modes

- **Missing internal round:** changes exact Conv2D + Add results even when every
  shape and index is correct.
- **Rounding every primitive:** also changes current semantics; JavaScript
  `Math.fround` and C `float` must not be inserted indiscriminately.
- **Tree substitution:** can exponentially duplicate reductions and code.
- **Captured reduction variable:** silently loads the wrong element after fusion.
- **Treating reductions as sets:** floating-point order and max tie behavior are
  observable.
- **Confusing proof and probing:** random agreement is evidence, not proof.
- **Circular validation:** the same Symbolic lowering cannot independently
  validate itself.
- **Fusing through fan-out without liveness:** loses an externally needed
  intermediate or duplicates expensive work.
- **Backend helper drift:** format decode and max/NaN behavior must have one
  specification shared by interpreters and emitters.
- **Overusing SMT:** solver-normalizing all floating-point arithmetic makes the
  trusted surface and diagnostics worse; solve index obligations first.

## Resolved choices and open questions

Resolved:

- Evolve the current symbolic language instead of duplicating `Compute`.
- Use semantic Kernel IR plus scheduled Loop IR.
- Model fusion as a plan over existing stages by default.
- Preserve removed materialization as explicit `round_f32`.
- Keep the stored Kernel representation as an SSA-like value DAG; elaborate
  virtual loads lazily instead of storing an inlined expression tree.
- Use the implemented `Expr.Index`, `Expr.Bool`, and `Expr.Value` namespaces,
  retaining typed position/delta indices and abstract reduction binders.
- Treat reducer identity as expression-local; compose bodies only through one
  threaded builder namespace and `Expr.Rewrite.freshen`.
- Treat a stage conversion, working value, and payload format as three distinct
  concepts.
- Start with a typed OCaml builder, not a textual parser.
- Compare arbitrary output coordinates with structural scalar equality and
  solver-backed integer indices.
- Use compositional ordered-reduction proof rules instead of SMT unrolling.
- Keep `Map_verify` and concrete differential execution as independent layers.

Open questions to answer during Phase 1:

- Exact representation of paired reducers such as max-with-index.
- Whether real operator bodies require a semantic `Let`; until demonstrated,
  sharing belongs in the kernel value DAG and derived A-normal Loop IR.
- Whether the solver interface uses subprocess SMT-LIB initially or an OCaml
  binding; the proof obligation and reporting interface should not depend on
  that choice.
- The cross-runtime exactness policy for `exp`, `sqrt`, NaN payloads, and signed
  zero in generated JavaScript and C.
