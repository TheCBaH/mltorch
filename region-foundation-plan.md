# Region language Foundation implementation plan

## Status and purpose

This plan implements only the **Foundation task** from
[region-compute-design.md](region-compute-design.md).  It creates a validated
Region computation language, mechanically embeds every existing Pixel
computation, executes synthetic scalar-local Region programs, and preserves the
current specialized Pixel path.

The task does **not** convert RMSNorm, LayerNorm, Softmax, attention, Conv, or
any other operation.  Those conversions consume this foundation in later,
separately bounded tasks.  Consequently, completing this plan establishes
language and execution capability but claims no operation speedup.

Every numbered gate is intended to be a reviewable commit.  A gate is complete
only after its focused tests and the listed compatibility checks pass.  Do not
combine operation conversion with a Foundation gate.

## 1. Completion contract

Foundation is complete when all of the following statements are true:

1. `Region_program.t` is the computation stored for every logical
   `Kernel.Value.t`.
2. `Region_program.pixel expr` embeds an existing `Expr.Value.t` in O(1): it
   reuses the expression tree, adds no local, and selects `Singleton` on all six
   axes.
3. `Kernel_adapt.of_stage_program` performs that embedding mechanically for
   every selected `Stage_program.Stage.t`; operation modules and
   `Stage_program.t` are unchanged.
4. A Region program can select `Singleton` or `Whole` independently on each of
   N, T, D, H, W, and C; define an ordered list of scalar locals; use nested
   existing `sum` and `max` reductions in local bodies; and emit one tensor
   value for each output coordinate in the region.
5. Local definitions may reference only earlier locals.  Each local is
   invariant over every `Whole` axis.  The emitter may reference all defined
   locals and all output axes.
6. Construction and `Kernel.create` reject malformed scope, duplicate locals,
   invalid partitions, oversized source programs, oversized specialization,
   free reducers, and unresolved tensor sources without unbounded work first.
7. The Region reference interpreter computes a local once per region key and
   reuses it for every output in that region.
8. A program classified as Pixel executes through a derived specialized Pixel
   loop with the current N/T/D/H/W/C scan, C innermost, no region-key object,
   no local environment, and no per-output classification branch.
9. Region-to-Pixel specialization substitutes locals capture-safely and can
   prove that a Region program reconstructs a supplied Pixel expression.
10. Existing Kernel, fusion, Model Explorer, Native, and JavaScript tests pass;
    a dedicated microbenchmark shows no measurable Pixel regression beyond the
    declared noise threshold.

The Foundation language has one emitter **per logical Kernel value**.  A
`Kernel.t` still contains multiple topologically ordered values and can still
have multiple public outputs.  This preserves the current multi-output model
without introducing shared locals across differently shaped values.  A later
whole-graph single-node task may admit several internal values under one outer
Region schedule, but it must reuse the local, partition, specialization, and
execution contracts defined here.

## 2. Fixed representation decisions

### 2.1 One scalar expression language

Do not create a second Region arithmetic AST and do not represent a local as a
synthetic tensor source.  Extend the shared `Expr.Value.t` with one leaf:

```ocaml
module Local_var : sig
  type t

  val compare : t -> t -> int
  val equal : t -> t -> bool
  val hash : t -> int

  module Map : Map.S with type key = t
  module Set : Set.S with type elt = t
end

module Value : sig
  type t = private
    | ...
    | Local of Local_var.t

  val local : Local_var.t -> t
end
```

`Local_var.t` is an opaque scalar identity allocated by the existing immutable
`Expr.Builder` supply.  Add `Builder.fresh_local`; do not use a process-global
counter.  Reducer and local identities may share the supply ordinal because
their types and namespaces are distinct.

`Expr` owns the reference because it owns scalar syntax.  `Region_program` owns
the binding, scope, lifetime, partition, and evaluation order.  Ordinary
`Expr.Check.value` continues to mean “closed expression” and rejects every
local reference.  Region validation uses a new explicitly open check with an
allowed-local set.

This keeps existing Pixel expressions structurally unchanged.  It also keeps
all arithmetic, Boolean, index, reduction, source, comparison, hashing,
printing, rewriting, and evaluation rules in one language.

### 2.2 Region program shape

Add these Native modules as separate compilation units:

- `region_partition.ml/.mli`
- `region_local.ml/.mli`
- `region_program.ml/.mli`
- `region_eval.ml/.mli`
- `region_execution.ml/.mli`

Use private records and result-valued construction.  Record types live in their
own modules to preserve the repository's field-label rule.

The initial semantic shape is:

```ocaml
module Region_partition.Axis_mode : sig
  type t = Singleton | Whole
end

module Region_local.Shape : sig
  type t
  val scalar : t
end

module Region_local : sig
  type t = private {
    id : Expr.Local_var.t;
    shape : Shape.t;
    value : Expr.Value.t;
  }

  val scalar : id:Expr.Local_var.t -> value:Expr.Value.t -> t
end

module Region_program : sig
  type t = private {
    partition : Region_partition.t;
    locals : Region_local.t list;
    output : Expr.Value.t;
  }

  val pixel : Expr.Value.t -> t

  val create :
    max_size:int ->
    max_depth:int ->
    partition:Region_partition.t ->
    locals:Region_local.t list ->
    output:Expr.Value.t ->
    (t, error) Err.t

  val check :
    max_size:int -> max_depth:int -> t -> (unit, error) Err.t
end
```

`Region_local.Shape.t` is abstract and initially has only `scalar`.  Recording
the shape prevents scalar-ness from becoming an undocumented convention, while
the abstract interface avoids exposing a closed one-constructor variant that
would make a later shaped-local extension unnecessarily disruptive.

The local list is the preparation phase order.  Do not add a general phase
variant, mutable statement, loop statement, custom fold, shaped local, block
partition, or store statement in Foundation.  Existing `Expr.Reduction` already
provides ordered, nestable `Sum` and `Max` reductions.

`pixel` is the O(1) structural embedding: its partition and local scope are
valid by construction, but it deliberately does not traverse the supplied
expression.  `create` builds and checks a general program.  `check` lets the
standalone `Kernel.create` boundary revalidate either form under that Kernel's
possibly tighter limits.

### 2.3 Partition meaning and ordered scan

`Region_partition.t` is a private six-axis value, one mode per axis.  Provide:

```ocaml
val singleton : t
val of_whole_axes : Expr.Axis.t list -> (t, error) Err.t
val mode : t -> Expr.Axis.t -> Axis_mode.t
val is_singleton : t -> bool

val key_shape : output_shape:Vec6.shape -> t -> Vec6.shape
val key_of_output : t -> Vec6.coord -> Vec6.coord
val fold_keys : ...
val fold_outputs : ...
```

`of_whole_axes` rejects a duplicate axis instead of silently normalizing it.
This supplies a real invalid-partition boundary and keeps an erroneous
normalization-axis list observable.  It fills unlisted axes with `Singleton`.

For an output shape `S`:

- a `Singleton` key axis has extent `S[axis]` and the key carries the output
  coordinate on that axis;
- a `Whole` key axis has extent 1 and canonical key coordinate 0;
- outputs of a key keep every `Singleton` coordinate fixed and enumerate the
  full output extent of every `Whole` axis.

Both key and output enumeration use the repository's N, T, D, H, W, C order
with C innermost.  This order is semantic in Foundation because reductions and
floating-point rounding are order-sensitive.  Later scheduling may change it
only under an explicit numerical contract.

### 2.4 Local invariance

A local body is evaluated with the region key as its output-coordinate
environment.  Therefore it must not contain `Expr.Index.Output axis` for an
axis whose partition mode is `Whole`.  Such an axis must be represented by a
reduction variable where the local traverses it.

Validate this syntactically with `Expr.Fold.output_axes`:

```text
output_axes(local.value) intersect whole_axes(partition) = empty
```

Earlier local references are safe because each earlier binding has already
passed the same invariant.  This rule is intentionally conservative.  Do not
attempt algebraic cancellation or value-range reasoning in Foundation.

The emitter is not invariant: it may use all output axes and every defined
local.

### 2.5 Kernel integration

Change the logical value record to:

```ocaml
module Kernel.Value : sig
  type t = {
    id : Tensor_id.t;
    sg : Tensor_sig.t;
    computation : Region_program.t;
    result : Result_conversion.t;
  }
end
```

Do not retain parallel `body` and `region` fields, and do not add a semantic
`Pixel | Region` sum.  `Region_program.pixel_expression` returns `Some expr`
only when the program has all-singleton partition and no locals; otherwise it
returns `None`.  Consumers that are valid only for Pixel-form computations must
use that query and reject the other case explicitly.

`Result_conversion.Round_f32` applies to the Region emitter exactly once.  It
does not apply to locals.  Scalar locals remain working OCaml floats, matching
the current expression reduction before the existing logical-value boundary
round.  Relaxed rounding and reassociation are outside Foundation.

### 2.6 Source and size analysis without expansion

Add structural Region queries that fold the local bodies and emitter directly:

```ocaml
Region_program.Fold.sources
Region_program.Fold.loads
Region_program.Fold.intrinsic_sources
Region_program.Fold.binders
Region_program.Fold.intrinsics
Region_program.Fold.max_depth
```

Kernel dependency, reachability, fusion-candidate, and evaluator-depth analysis
must use these queries.  Never specialize a Region program merely to discover
sources: local substitution may duplicate a subtree exponentially.

Retain the existing `Kernel.Limits.max_size` and `max_depth` fields rather than
adding speculative configuration.  Refine their documented meaning:

- `max_size` bounds the total source AST nodes across all local bodies plus the
  emitter, and also bounds the number of local-list cells before traversal;
- `max_depth` bounds each local body and the emitter;
- `max_size` separately bounds the expanded Pixel specialization.

Check the local list one cell past `max_size` before examining its members.
Then check each expression against the remaining source-size budget.  Only
after a fragment has passed that bound may validation call the existing
unmetered `Expr.Fold.size` to subtract its exact consumption.  Never add
unbounded sizes and compare after the addition.

For expansion, perform a saturating preflight before allocating the result.
Walk bindings in order and compute each binding's expanded size/depth from the
already bounded measures of referenced earlier locals.  Saturate at
`max_size + 1` and `max_depth + 1`; do not form an unchecked aggregate.  Build
the expanded expression only after the preflight passes.

`Region_program` takes primitive `max_size` and `max_depth` arguments rather
than depending on `Kernel.Limits`; this keeps the module below `Kernel` in the
dependency graph.  `Kernel.create` passes its configured fields.  A program
constructed under wider limits is always rechecked when inserted into a Kernel
with tighter limits.

## 3. Gate 0 — baseline, census, and red tests

### Code and investigation

1. Record every direct `Kernel.Value.body` consumer.  The current required set
   includes `kernel.ml`, `kernel_eval.ml`, `kernel_elab.ml`, `fusion_plan.ml`,
   `kernel_adapt.ml`, the Native tests, and Model Explorer's kernel/detail
   exporters.  Re-run the census immediately before the migration so no new
   consumer is missed.
2. Add a temporary or permanent Pixel benchmark driver before changing the
   representation.  Use a large one-input identity/clone expression and a
   two-input add expression so wrapper overhead is visible.  Include one
   convolution-shaped and one indexing-shaped fixture as scan/order smoke
   cases, but use the trivial kernels for the latency threshold.
3. Warm up each case, run at least 20 samples, and record median wall time,
   input-load count, output-cell count, and GC words per output cell.  Run the
   benchmark from the built executable rather than through repeated
   `dune exec` compilation.
4. Add tests that will turn red for the new behavior before implementing it:
   an expression local rejected by closed `Expr.Check.value`, a dependent
   Region local, and a whole-C synthetic reduction shared across C outputs.
   Run each test against the pre-change code and record the expected failure;
   do not commit a knowingly failing suite.  Commit the test together with the
   first gate that makes it pass.

### Gate checks

- `dune runtest test/expr test/native`
- `make benchmark.region_pixel` or the equivalent direct benchmark command
- Save the exact command, machine/compiler description, sample count, and
  medians in this working plan when implementation begins.

### Acceptance

The existing test suite is green, each new capability test has a recorded red
witness for the expected missing feature, and a pre-change Pixel measurement
is available for Gate 6 comparison.  The Gate 0 commit itself contains only
green baseline/benchmark work.

## 4. Gate 1 — scalar-local support in `Expr`

### Representation work

1. Add `lib/expr_internal/local_var.ml` with opaque identity, comparison,
   equality, hashing, `Map`, and `Set`.
2. Extend `Expr.Builder.state` and its API with `fresh_local`.  Preserve the
   immutable threaded supply and `Supply_exhausted` behavior.
3. Add `Local` to `expr_repr.value`, `Expr.Value.t`, and the public facade.
   Keep closed variant declarations and exhaustive arms alphabetized according
   to repository convention.

### Required exhaustive updates

Update every `Expr.Value.t` consumer, including:

- `Value.compare`, `Value.equal`, and `Value.hash`;
- `Fold.walk`, metered measurement, sources, loads, output axes, reducers,
  binders, and the new `Fold.locals` query;
- `Rewrite.rebuild`, freshening, source mapping, output substitution, load
  substitution, and a new capture-safe `substitute_locals` operation;
- `Check`, `Eval`, and `Pp`;
- Model Explorer expression labels and child traversal;
- `Ground_expr`/`Ground_eval` or any other exhaustive match reported by the
  compiler.

The generic rewrite rebuild function gets an `on_local` leaf callback so local
substitution remains inside the one scope-aware traversal.  Inserted
replacements are not recursively revisited, matching `substitute_loads`.

### Scope, evaluation, and printing APIs

Add:

```ocaml
Expr.Check.fragment :
  ?max_size:int ->
  ?max_depth:int ->
  locals:Expr.Local_var.Set.t ->
  Expr.Value.t ->
  (unit, error) Err.t

Expr.Eval.value :
  ?local:(Expr.Local_var.t -> float option) ->
  Env.t -> output:int Expr.Coord.t -> Expr.Value.t -> ...

Expr.Pp.value_open :
  names:(Expr.Local_var.t -> string option) ->
  Format.formatter -> Expr.Value.t -> unit
```

`Expr.Check.value` calls the common implementation with an empty allowed set.
It reports `Unbound_local` for the first local not in scope.  `Eval.value`
defaults to no locals and reports the same semantic condition rather than
raising.  `Pp.value` remains deterministic for closed Pixel expressions;
`value_open` lets `Region_program.pp` assign lexical names `l0`, `l1`, ...
independent of allocation history.

### Tests

Add or extend `test/expr` coverage for:

- opaque local allocation from one threaded builder supply;
- closed check rejecting a local;
- fragment check accepting exactly its allowed set;
- local occurrence discovery through arithmetic, Boolean expressions, and
  reduction bodies;
- compare/hash distinction for different free locals;
- evaluation with a binding and failure without one;
- substitution under a reduction without reducer capture;
- deterministic open printing independent of local allocation ordinal;
- size/depth metering counting a local as one value node;
- all existing closed-expression golden output unchanged.

### Gate checks and acceptance

- `dune runtest test/expr`
- `dune build lib/expr lib/native`
- `make jsoo.inline-runtest`

Gate 1 is complete when `Expr` can carry a scalar local reference but every
existing Pixel construction, check, evaluation, and printed form behaves as
before.

## 5. Gate 2 — Region structure, validation, analysis, and printer

### Partition module

Implement `Region_partition` first.  Its iterator helpers must:

- reject an out-of-shape output coordinate before deriving a key;
- avoid unchecked extent products;
- preserve the six-axis scan order;
- visit every output exactly once across all keys;
- make `singleton` a shared immutable value used by
  `Region_program.pixel`.

### Local and builder modules

`Region_local.t` has no public record constructor.  Add the smart
`Region_local.scalar` constructor shown above and a `Region_program.Builder`
that threads `Expr.Builder.state`, appends bindings in order, and hands the
continuation an `Expr.Value.local id` reference:

```ocaml
val scalar :
  Expr.Value.t ->
  (Expr.Value.t -> 'a t) ->
  'a t

val finish :
  max_size:int ->
  max_depth:int ->
  partition:Region_partition.t ->
  output:Expr.Value.t ->
  (Region_program.t, Region_program.error) Err.t t
```

Also retain the result-valued low-level `Region_program.create` for transforms
that allocate identities through `Expr.Builder.fresh_local` and assemble
binding records directly.  Repeating the same binding record must be
detectable as `Duplicate_local`; private identities alone do not make that
state impossible.

### Validation order

`Region_program.create` and the recheck performed by `Kernel.create` use this
order:

1. bound local-list cells before traversing them;
2. source AST size and depth budgets;
3. duplicate local definitions;
4. each local body's scalar shape;
5. local scope in list order, distinguishing a later-known id
   (`Forward_local`) from an id never defined (`Unknown_local`);
6. free/duplicate reducer checks through `Expr.Check.fragment`;
7. whole-axis invariance for each local;
8. emitter scope against the complete local set;
9. specialization expansion preflight when the caller requests
   specialization or reconstruction.

All error rows carry typed payloads and deterministic printers.  Every check
must have a test that produces its error.  Do not report an invalid actual
aggregate by computing the aggregate that the limit exists to prevent.

### Program analysis and printing

Implement `Region_program.Fold` over source syntax without substitution.
Deduplicate sources with `Expr.Source.Set`; retain lexical repeats for loads.
The deterministic printer renders:

```text
region [N=singleton T=singleton D=singleton H=singleton W=singleton C=whole]
  let l0 : scalar = ...
  let l1 : scalar = ... l0 ...
  emit ... l1 ...
```

For a Pixel-form program, provide a compact expression-only rendering used by
`Kernel.pp`, so current Kernel golden output need not change merely because of
the mechanical wrapper.  The full `Region_program.pp` remains available for
diagnostics and Region-specific tests.

### Synthetic tests

Create `test/native/region_program_test.ml` with:

- all-singleton Pixel classification and O(1) expression identity reuse;
- whole C, whole W+C, and all-whole partitions;
- exact key/output enumeration and no duplicate or missing output;
- dependent scalar locals;
- a local containing nested sum then max reductions;
- duplicate, forward, unknown, and emitter-only unknown local failures;
- a local varying on a Whole axis rejected as non-invariant;
- the same axis use in the emitter accepted;
- duplicate whole-axis specification rejected;
- source, load, binder, and intrinsic folds seeing every fragment;
- total source-size, local-list, depth, and expanded-size limits;
- deterministic printer output from independently allocated local identities.

### Gate checks and acceptance

- `dune runtest test/expr test/native`
- `make build`

Gate 2 is complete when the Region language is independently constructible and
validatable, but no Kernel representation has changed yet.

## 6. Gate 3 — Kernel representation and adapter migration

### Kernel changes

1. Rename `Kernel.Value.body` to `computation` and change its type to
   `Region_program.t`.
2. Change `Kernel.create` validation from `Expr.Check.value` to
   `Region_program.check` using the Kernel's configured limits.
3. Replace direct expression folds in source resolution, dependency depth,
   reachability, `uses`, `load_uses`, and printing with Region structural
   queries.
4. Compute evaluator depth as the maximum of each local body and the emitter
   with `Result_conversion` around the emitter only.  Ordered local evaluation
   is iterative, so local-list length is not expression recursion depth.
5. Add helpers that expose the Pixel expression only through derived
   classification.  Do not let downstream modules inspect private Region
   record fields to duplicate the classification rule.

### Stage adapter

Keep `Stage_program.Stage.body : Expr.Value.t`.  In
`Kernel_adapt.of_stage_program`, construct:

```ocaml
computation = Region_program.pixel st.body
```

The wrapper must reuse `st.body` physically and perform no traversal or AST
copy.  The adapter's existing early validation of every untrusted Stage body
remains, because it bounds selection/liveness analysis before Kernel
construction.  `Kernel.create` revalidates the resulting computation because
it is also a standalone public boundary.

### Existing fusion behavior

`Kernel_elab` and `Fusion_plan` remain Pixel-only in Foundation:

- obtain producer and consumer expressions through
  `Region_program.pixel_expression`;
- preserve the current pointwise legality and substitution implementation for
  two Pixel-form values;
- reject any edge whose producer or consumer has a non-degenerate Region
  computation with a typed `Regional_computation` decision;
- derive all dependency edges through Region source queries so the rejection
  cannot disappear merely because a source occurs in a local rather than the
  emitter.

This preserves current fusion rather than broadening it into Region graph
fusion.  Region-aware whole-graph admission is a later task.

### Model Explorer and diagnostics

Update Model Explorer at the same gate:

- Kernel graph nodes use `Region_program.Fold.sources` and a bounded Region
  rendering;
- Pixel detail views remain byte-for-byte unchanged;
- a Region detail view shows one root per ordered local plus the emitter, with
  local-reference edges and expression children, under the existing detail
  node ceiling;
- fusion diagnostics can render `Regional_computation` rejections.

Do not specialize a Region computation to feed Model Explorer; visualize its
sharing explicitly.

### Tests

- Update hand-built Kernel helpers to wrap expressions with
  `Region_program.pixel`.
- Keep existing Kernel printer expectations unchanged for Pixel-form values.
- Assert that every current graph fixture adapts to Pixel-form Region
  computations.
- Assert physical equality between each Stage body and the embedded emitter.
- Re-run all existing Kernel source, reachability, bounds, result-conversion,
  elaboration, and fusion rejection tests.
- Add a mixed Kernel with one synthetic Region value and verify that its
  dependencies remain visible while fusion rejects the regional edge.

### Gate checks and acceptance

- `dune runtest test/native test/model_explorer`
- `make runtest`
- `make jsoo.inline-runtest`

Gate 3 is complete when Kernel stores only Region computations, every existing
Stage adapts mechanically, and all existing Pixel fusion semantics remain
available.

## 7. Gate 4 — Region reference interpreter and derived execution form

### Reference evaluator

Implement `Region_eval` over a validated program, output shape, and existing
`Expr.Eval.Env.t`:

```ocaml
val value_at :
  Region_program.t ->
  output_shape:Vec6.shape ->
  env:Expr.Eval.Env.t ->
  output:Vec6.coord ->
  (float, error) Err.t

val materialize :
  Region_program.t ->
  output_shape:Vec6.shape ->
  env:Expr.Eval.Env.t ->
  (Tensor.packed, error) Err.t
```

For each region key:

1. allocate one local array sized to the validated local count;
2. evaluate local bodies in declaration order, passing an O(1) local resolver
   backed by that array;
3. enumerate outputs owned by the key;
4. evaluate the emitter at the actual output coordinate using the same array;
5. write each result once.

The array is allocated once per region, never per output.  Build one
`Local_var`-to-dense-slot table from the validated binding list when lowering
the program, not once per region.  A local read performs one expected-O(1)
identity lookup and one array access.  Unknown or forward references are
validation failures; they do not trigger a fallback lookup or dynamic scope
walk.  A later Loop lowering may replace this lookup with a register/slot
operand, but Foundation does not add a second lowered expression AST.

`value_at` derives the containing key, evaluates its locals once, and emits
only the requested coordinate.  It is the intentionally non-optimized Pixel
specialization oracle for a genuine Region program.

### Specialized execution classification

Implement `Region_execution.lower` as a derived execution form:

```ocaml
type t =
  | Pixel_loop of Expr.Value.t
  | Region_loop of Region_program.t
```

Classify once per logical Kernel value before materialization.  `Pixel_loop`
is available only for all-singleton/no-local programs.  Its evaluator must use
the existing direct `Tensor.materialize` callback and `Expr.Eval.value` path,
with the output coordinate passed directly.  It must not call
`Region_partition.key_of_output`, construct a local resolver, allocate an
array, or branch on program class in the callback.

`Region_loop` uses `Region_eval`.  This variant is execution IR only; Kernel
continues to store one semantic `Region_program.t`.

### Kernel evaluator integration

Refactor `Kernel_eval` without changing its public result contract:

- build a `Region_execution.t Tensor_id.Map.t` once per invocation;
- apply `Result_conversion` by mapping only the selected computation's emitter;
- `run` stores every value in topological order using its selected loop;
- `value_at` uses the direct Pixel expression for Pixel-form values and
  `Region_eval.value_at` otherwise;
- `run_plan` supports mixed kernels, but virtual edges remain only between
  Pixel-form computations because Gate 3's planner enforces that restriction;
- keep producer-recursion depth checks and per-coordinate memoization;
- keep binding validation before any output allocation.

For a stored Region value, producer loads come from already materialized input
or earlier-value tensors exactly as in current `run`.  For reference
`value_at`, the existing recursive source environment may evaluate a Pixel
producer on demand.  Do not add Region producer recomputation to fusion in this
gate.

### Numerical contract

Add bitwise tests for:

- sum and max seeds and left-fold order;
- nested reductions;
- `Select` short-circuiting inside a local and emitter;
- NaN and signed-zero behavior inherited from `Expr`;
- a Region local retaining full working precision until the emitter's single
  logical-value `Round_f32`;
- no extra round on a local and no missing/doubled round at the output.

### Synthetic execution tests

Create `test/native/region_eval_test.ml` covering:

- Pixel-loop output bitwise equal to direct `Expr.Eval` over every coordinate;
- whole-C sum shared across C emissions, with a counting input environment
  proving C reduction loads rather than C-squared loads;
- whole-W+C nested reduction with multiple independent region keys;
- dependent locals where a later reduction reads an earlier scalar;
- mixed max then sum locals matching a separately expanded Pixel expression;
- output ownership: every cell written exactly once;
- invalid requested coordinate rejected before evaluation;
- a multi-value Kernel where a Region consumer reads a stored Pixel producer;
- `run`, `value_at`, and default `run_plan` agreement.

### Gate checks and acceptance

- `dune runtest test/native`
- `make runtest`
- `make jsoo.runtest`
- `make jsoo.inline-runtest`

Gate 4 is complete when synthetic Region programs execute with actual
cross-output sharing and all Pixel fixtures still agree bitwise with
`Stage_program.ground` and `Eval_direct`.

## 8. Gate 5 — specialization, reconstruction, and proof helpers

### Capture-safe specialization

Implement:

```ocaml
Region_program.specialize_pixel :
  max_size:int ->
  max_depth:int ->
  t ->
  (Expr.Value.t, error) Err.t
```

The result is the emitter with locals substituted in declaration order.  Use
one `Expr.Builder` state for the complete rewrite.  Freshen every inserted
fragment before placing it beneath an emitter or another reduction.  A local
used twice is expanded twice with independently freshened reducer binders.

Run the saturating expansion preflight described in section 2.6 before
building.  After substitution, run closed `Expr.Check.value` under the same
limits.  A specialization with any remaining local reference is an error, not
an assertion failure.

### Reconstruction

Implement:

```ocaml
Region_program.reconstructs :
  max_size:int ->
  max_depth:int ->
  pixel:Expr.Value.t ->
  t ->
  (bool, error) Err.t
```

Specialize, then compare with `Expr.Value.equal`, whose reducer comparison is
already alpha-equivalent and preserves floating-point constant bits, operand
order, reduction nesting, and reduction bounds.  Do not simplify arithmetic,
reassociate reductions, or compare pretty-printed text.

Expose a separate typed failure for “valid Region program, but does not
reconstruct the supplied Pixel expression” at the future conversion boundary.
The low-level `reconstructs` query may return `false`; a converter can map it to
its own rejection reason while retaining the original Pixel program.

### Proof-oriented tests

- Pixel-form specialization is physically or structurally the original
  expression with no copy required.
- A whole-axis local specializes to the original repeated reduction.
- Dependent locals substitute transitively.
- Independently allocated reducer identities compare alpha-equivalently.
- A changed constant, operand order, reduction kind, bound, or rounding node
  fails reconstruction.
- A local that refers to a Whole output axis fails invariance before
  reconstruction.
- An exponentially duplicating local chain fails the expansion preflight
  without allocating the expanded tree.
- A boundary-sized specialization succeeds; one node beyond fails.

### Gate checks and acceptance

- `dune runtest test/expr test/native`
- `make precommit`

Gate 5 is complete when a later operation converter can construct a Region
program, prove scope/invariance, reconstruct the authoritative Pixel
expression, and receive a typed failure without changing operation code.

## 9. Gate 6 — Pixel no-regression enforcement and Foundation closeout

### Hot-path audit

Inspect the generated execution path for `Region_program.pixel` and verify:

- classification occurs before `Tensor.materialize`;
- the per-output callback contains no partition-mode match;
- it constructs no region key, local map, local array, or callback closure;
- output scan remains `Vec6` N/T/D/H/W/C with C innermost;
- input load coordinates and counts are unchanged;
- `Result_conversion` remains exactly one outer expression node per logical
  value;
- the Stage-to-Region wrapper retained the original expression object.

Add a deterministic counter test for scan order and load count.  Keep wall-time
and GC measurements in the benchmark rather than a flaky inline test.

### Benchmark comparison

Run the Gate 0 benchmark under the same compiler profile and machine
conditions.  Build the Gate 0 commit and the final Foundation commit in two
clean worktrees (or preserve two explicitly named benchmark executables), then
alternate baseline and Foundation processes so thermal/load drift does not
favor one version.
Acceptance thresholds:

- identical output-cell and input-load counts;
- zero additional allocation attributable to region machinery in the
  per-output callback;
- median trivial-kernel runtime no worse than 5% after alternating old/baseline
  and new measurements, or within the pre-recorded run-to-run noise if that is
  larger;
- no regression in the convolution-shaped and indexing-shaped scan smoke
  cases that indicates a changed traversal or coordinate construction.

If the runtime threshold fails, profile before relaxing it.  A Region wrapper
cost visible on Add/Clone but hidden by Conv is a Foundation defect.

### Full verification

Run:

```text
make format
make build
make runtest
make jsoo.runtest
make jsoo.inline-runtest
make precommit
```

Also run the existing Kernel/Native4D/model fixture paths that consume Kernel
or Model Explorer output, even though no operation module changed.

### Documentation closeout

Update the tracked Region and Kernel design records with only facts established
by the implementation:

- final module/API names;
- actual limit semantics and hard ceilings;
- the chosen local-reference representation;
- the measured Pixel benchmark result;
- any deliberately conservative rejection discovered during implementation.

Keep this plan as working scaffolding.  Do not claim RMSNorm, LayerNorm,
Softmax, attention, graph fusion, or Conv locality improvements at Foundation
completion.

## 10. File-level change map

Expected primary files:

| Area | Files | Foundation change |
|---|---|---|
| Expr representation | `lib/expr_internal/expr_repr.ml`, `local_var.ml`, `value.ml`, `builder.ml` | Add opaque scalar-local leaf and allocation |
| Expr operations | `fold.ml`, `rewrite.ml`, `check.ml`, `eval.ml`, `pp.ml` | Make locals analyzable, substitutable, checkable, evaluable, and printable |
| Expr facade | `lib/expr/expr_api.ml`, `expr.ml`, `expr.mli` | Publish the bounded local APIs |
| Region language | new `lib/native/region_partition.*`, `region_local.*`, `region_program.*` | Partition, ordered scalar bindings, validation, analysis, printer, specialization |
| Region execution | new `lib/native/region_eval.*`, `region_execution.*` | Reference Region loop and derived Pixel loop |
| Kernel | `lib/native/kernel.*`, `kernel_adapt.*`, `kernel_eval.*` | Store Region computations and execute derived form |
| Existing fusion | `kernel_elab.*`, `fusion_plan.*` | Preserve Pixel fusion; explicitly reject regional edges |
| Visualization | `lib/model_explorer_export/me_detail.*`, `me_kernel.*`, `me_fusion.*` | Render Region structure without expansion |
| Expr tests | `test/expr/*` | Local leaf, scope, rewrite, eval, printing, limits |
| Native tests | new Region tests plus `kernel_test.ml`, `kernel_eval_test.ml`, `fusion_test.ml`, `depth_probe.ml` | Synthetic Region capability and Pixel compatibility |
| Benchmark | a small dedicated executable and Make target | Reproducible Pixel latency/allocation baseline |

The compiler's exhaustive-match failures are part of the migration census.
Resolve each by deciding whether the consumer accepts a scalar local, rejects
an open expression, or must render it.  Do not add wildcard arms to silence the
new constructor.

## 11. Explicit non-goals and post-Foundation handoff

Foundation does not:

- modify `Norm.RmsNorm.Compute`, LayerNorm, Softmax, or any operation-specific
  `Compute(S).pixel` implementation;
- retain or infer operation identity/provenance;
- select normalization axes or hoist operation subtrees;
- change Graph IR, shape inference, importer behavior, or operation JSON;
- add Block partitions, shaped locals, vector caches, online folds, Loop IR,
  parallel reductions, or backend-specific placement;
- fuse a general multi-node graph;
- relax f32 rounding, reassociate reductions, or change NaN/signed-zero rules;
- optimize dot products, attention, or convolution locality.

The post-Foundation implementation guide is
[`region-native-implementation-guide.md`](region-native-implementation-guide.md).
It defines two separate consumers of this Foundation; neither retrospectively
changes the contracts above.

1. **Region-native shared-work execution.**  After Gate 6 closes out, a
   selected operation may use typed provenance, validated parameters and
   shapes, and its Pixel expression to attempt regionization.  The optional
   converter must run before provenance is discarded (or receive a bounded
   private candidate), build a non-degenerate `Region_program`, and prove
   `Region_program.reconstructs`.  Every rejection retains the original Pixel
   computation.  Accepted programs are lowered by a dedicated Region
   executor; `Region_eval` remains its reference oracle, not its hot path.

2. **Locality scheduling.**  Independently, footprint analysis may tile a
   bounded Pixel-form computation for input/weight/intermediate reuse.  This
   is normally schedule-only and keeps the stored semantic program unchanged.
   A lowering may derive a block-local Region program only when the selected
   cache, accumulator, or ordered phase has a Region-level lifetime that must
   be represented, checked, and inspected.

Neither track changes operation-specific `Compute(S).pixel` or extends
`Semantics.SEMANTICS`; the former remains the scalar semantic authority and
the latter remains a scalar value/index interface.
