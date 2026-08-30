# Const-SSA detailed implementation plan

This document turns [const-ssa-plan.md](const-ssa-plan.md) into implementation
work against the current Native transform stack.  It preserves the design
decisions in [const_ssa_design.md](.ai/const_ssa_design.md): Const-SSA is
Native-owned,
symbolic folding is the default, and direct evaluators receive only materialized
`Tensor.packed` values.

## Starting point in the codebase

The implementation must change the following existing seams rather than create
a parallel transform pipeline.

| Current seam | Current behavior | Required change |
| --- | --- | --- |
| `lib/native/transform/passes/fold_const.ml` | folds only if each graph constant has a payload in `Pass.env.constants`; evaluates a one-node region with `Eval_direct` | construct a Const-SSA `Apply` definition when each operand has a Const-SSA binding; retain materialized evaluation only for the explicit fallback path |
| `lib/native/transform/recipe.{ml,mli}` | `fold_to_constant` removes a node and binds `Tensor.packed` bytes | add a typed deferred-constant recipe action; preserve the existing materialized action during migration |
| `lib/native/transform/rewrite.{ml,mli}` | versioned state is graph + payload map + id supply; `apply`/`pack` keep that map aligned with graph ids | versioned state additionally carries validated Const-SSA definitions and graph-tensor-to-plan bindings; apply and pack preserve/remap bindings |
| `lib/native/transform/pass.ml` | a pass receives `{ constants; view }` | add the constant store to this environment; payload map remains a cache, not the source of staticness |
| `lib/native/transform/passes/fold_batch_norm.ml` | emits Native arithmetic plus literal `Tensor.packed` constants, then relies on `Fold_const` materialization | keep emitting the same Native arithmetic, but make generated literals Const-SSA leaves and let the second symbolic fold capture the arithmetic DAG |
| `lib/native_interp/native_interp.{ml,mli}` | `transform_lowered` is archive-free but folding declines; `transformed` exposes only computed payloads; `evaluate` uses state payloads then lens archive lookup | seed captured Const-SSA leaves from the lowered PT2 sidecar, return a symbolic constant store, and add explicit materialization before the existing evaluator is called |
| `lib/native4d/lower.{ml,mli}` | takes optional payload map and returns optional generated payloads | carry/remap Const-SSA bindings from canonical Native to Native4D; retain raw payload map only as a materialization cache |
| `lib/native/transform/map_verify.ml` and grounding | unbound constants are opaque/free; a folded `permute(w)` replaced by a new constant cannot be related structurally | expand Const-SSA exported values symbolically, with captured leaves represented by their stable source identities |

`Pipeline.canonical` is already the common Native/Native4D definition of
canonicalization.  Its two `Fold_const` rounds must become symbolic; its
structure must not depend on `~fold`/preload.

## Target data model and interfaces

### Module layering

Add these Native modules under `lib/native/transform`.

```text
const_ssa.{ml,mli}             immutable data, registry, signatures, validation
constant_store.{ml,mli}        graph bindings + materialized cache + remapping
const_ssa_symbolic.{ml,mli}    Const-SSA -> Ground_expr/Stage_program adapter
const_ssa_materialize.{ml,mli} explicit plan evaluator and capture resolver
```

`Const_ssa` must not depend on `Rewrite`, `Pass`, `Native_interp`, or Native4D;
otherwise the rewriter cannot store the plan without an import cycle.
`Const_ssa_symbolic` may depend on grounding/map-verification modules.
`Const_ssa_materialize` may depend on `Region`, `Eval_direct`, and archive
loading adapters, but direct evaluators must not depend back on it.

`Rewrite.Make` remains shared by Native and Native4D.  It should carry an
opaque `Constant_store.t` as additional versioned state.  Native recipes are
the only constructors that add Const-SSA operations; `Rewrite4` merely preserves
or remaps bindings that arrive from Native conversion.  This keeps the common
id/versioning machinery shared without making Native4D a second constant
language.

### Const-SSA identity

Use plan value ids backed by `Tensor_id.t`, but give them an abstract
`Const_ssa.Value_id` API.  A plan value normally starts with the identity of the
Native graph edge that first computed it.  That edge may later disappear from
the main graph, but its plan id remains valid because transform ids are monotone
and never reused.

This deliberately avoids a second allocator.  Native op payloads already name
`Tensor_id.t`, so a Const-SSA `Apply` can retain the existing `Graph_ir.op`
payload and reuse `Graph_ir.operands`, `map_operands`, Native shape inference,
and direct evaluation.  The abstraction prevents callers from treating a plan
id as a currently live graph edge.

Packing must only remap **exports**:

```text
current graph tensor id  -> immutable Const-SSA value id
```

Internal Const-SSA ids remain stable.  If packing changes graph `t42` to `t17`,
the binding changes from `t42 -> v42` to `t17 -> v42`; plan definitions do not
need to be rewritten.  This is essential when a removed captured edge is still
an input to a derived plan value.

### Proposed public interfaces

The exact record namespaces should follow project conventions, but the
interfaces should have this shape.

```ocaml
module Const_ssa : sig
  module Value_id : sig
    type t
    val of_tensor_id : Graph_ir.Tensor_id.t -> t
    val to_tensor_id : t -> Graph_ir.Tensor_id.t
  end

  module Capture : sig
    type t                    (* stable importer/archive source identity *)
    val of_string : string -> t
  end

  type leaf =
    | Captured of Capture.t
    | Literal of Tensor.packed
    | Opaque_materialized
        (** Test/programmatic source with bytes but no symbolic capture identity.
            It may materialize, but does not make an unrelated source equivalent. *)

  type definition =
    | Leaf of leaf
    | Apply of { op : Graph_ir.op; output : Tensor_sig.t }

  type t
  type error = [ ... ]

  val empty : t
  val add : t -> id:Value_id.t -> definition -> (t, error) Err.t
  val find : t -> Value_id.t -> definition option
  val sig_of : t -> Value_id.t -> Tensor_sig.t option
  val operands : t -> Value_id.t -> Value_id.t list
  val validate : t -> (unit, error) Err.t
  val pp : Format.formatter -> t -> unit
end

module Constant_store : sig
  type t
  type error = [ Const_ssa.error | ... ]

  val empty : t
  val materialized : t -> Tensor.packed Tensor_id.Map.t
  val plan : t -> Const_ssa.t
  val binding : t -> Tensor_id.t -> Const_ssa.Value_id.t option
  val is_effective_constant : t -> Tensor_id.t -> bool

  val bind_captured : t -> tensor:Tensor_id.t -> Const_ssa.Capture.t ->
    (t, error) Err.t
  val bind_literal : t -> tensor:Tensor_id.t -> Tensor.packed ->
    (t, error) Err.t
  val bind_apply : t -> tensor:Tensor_id.t -> Graph_ir.op -> Tensor_sig.t ->
    (t, error) Err.t
  val bind_materialized : t -> tensor:Tensor_id.t -> Tensor.packed ->
    (t, error) Err.t

  val restrict_and_rename_exports :
    t -> (Tensor_id.t -> Tensor_id.t option) -> (t, error) Err.t
end
```

`Const_ssa.add` validates locally on insertion and `validate` performs the full
topological/signature pass.  `Apply.op` is legal only if every operand maps to a
previous plan value and the closed operation registry admits its exact Native
constructor/configuration.  `Graph_shape.output_shape` must produce exactly one
signature equal to `output`; this rejects an operation with a plausible-looking
but forged result signature.

`Captured` contains the stable PT2 target string currently held in
`Pt2_native_graph.captured_targets`.  The core module knows it only as
`Const_ssa.Capture.t`; archive-specific loading remains in `Native_interp`.

### Rewrite and recipe interface changes

Change `Rewrite` state from:

```ocaml
{ constants : Tensor.packed Tensor_id.Map.t; ids; snapshot }
```

to:

```ocaml
{ constant_store : Constant_store.t; ids; snapshot }
```

Keep `Rewrite.constants` temporarily as the compatibility projection
`Constant_store.materialized`; add `Rewrite.constant_store`.  Remove the
compatibility accessor only after all Native and Native4D callers use the store.

Change origin and recipes in one coherent change:

```ocaml
Rewrite.origin :
  ?constant_store:Constant_store.t ->
  ?constants:(Tensor_id.t * Tensor.packed) list -> graph -> ...

Recipe.fold_to_materialized :
  node:Node_id.t -> output:'v source -> value:Tensor.packed -> ...

Recipe.fold_to_deferred :
  node:Node_id.t -> output:'v source -> op:Graph_ir.op ->
  sources:'v source list -> ...
```

The first call is the existing behavior under a clearer name.  The second
attaches an `Apply` definition for the preserved output id.  Both add the same
`Identical` value claim and diagnostic provenance.  A replacement also needs a
deferred-literal action for `Fold_batch_norm`'s epsilon and identity values.

`Rewrite.apply` must validate each new materialized value or plan definition
against the post-rewrite graph signature, update input kind to `Constant`, drop
unreferenced graph inputs, and retain plan values that those inputs used.  It
must reject an overwrite of either a materialized constant or a plan binding.
`Rewrite.pack` calls `restrict_and_rename_exports` with its tensor renaming
function alongside the existing materialized-map renaming.

Extend `Pass.env` to:

```ocaml
{ constant_store : Constant_store.t;
  constants : Tensor.packed Tensor_id.Map.t;  (* compatibility cache view *)
  view : View.t }
```

`Fold_const` must consult `constant_store`, never use `constants` as its
foldability condition.  Other existing passes continue to see their cache view
until explicitly migrated.

## Detailed implementation sequence

Each numbered gate below is a small, reviewable change.  Do not start the next
major milestone until its gate has passed and its changes have been committed.
The commit requirement is intentional: transform state, maps, and Native4D
conversion are highly coupled; an independently green checkpoint provides a
reliable bisect point.

### Gate 0 — baseline and contract tests

**Code work**

1. Run the current Native and Native4D unit suites and record the current
   payload-backed fold outputs for `const_permute`, the batch-norm fixture, and
   the existing Native4D cram models.
2. Add no behavior yet.  Add a short implementation note to the test fixture
   naming the baseline model/configuration list to be used by milestone 3.
3. Decide and document whether the public `Pipeline.canonical ~fold` argument
   is removed or retained as a deprecated compatibility argument.  Recommended:
   make canonicalization always symbolic and replace CLI `--fold` with an
   explicit `--materialize-constants` action.  There must be no mode in which
   preload changes canonical graph structure.

**Tests and verification**

- `opam exec -- dune runtest --root . test/native test/native4d`
- `make runtest` for the hermetic repository suite.
- When the local model data is available, run the existing payload-gated
  `make pt2.runtest` baseline as well.

**Commit gate**

Commit the recorded baseline and API decision as a standalone commit.  The tree
must be clean after the test golden updates before beginning Const-SSA code.

### Gate 1 — immutable Const-SSA and store, unused by transforms

**Code work**

1. Add `Const_ssa` and `Constant_store` with the interfaces above.
2. Implement the closed registry for the M1 language only:
   `Captured`, `Literal`, and `Permute`.
3. Implement validation: duplicate definition, missing operand, cycle,
   unsupported op/configuration, multi-output op, output-signature mismatch,
   invalid literal signature, and exporting an undefined plan value.
4. Implement deterministic printing and JSON only if transform state already
   crosses a serialized boundary; otherwise provide a stable printer first and
   defer wire encoding to the first consumer that needs it.
5. Add an origin helper in `Native_interp` that seeds every captured constant
   from `Pt2_native_graph.captured_targets` as a `Captured` leaf.  Preserve raw
   source target strings; do not consult `Pt2_native_graph.lens` here.

**Tests**

Add `test/native/const_ssa_test.ml` with expect tests for:

- one captured 6D input and a valid 4D `Permute` export;
- typed scalar literal;
- bad shape/dtype/quantization; duplicate id; unknown operand; cycle;
- rejected `Add` in M1 and rejected multi-output `Max_pool2d_with_indices`;
- deterministic printer order independent of map insertion order.

These tests are pure: no archive, no `Eval_direct`, and no Native4D graph.

**Verification and commit gate**

Run the Gate 0 commands plus `dune build` for `lib/native`, `lib/native4d`, and
`bin/native_graph.exe`.  Commit only the new data/store layer and its tests.
No existing transformation output may change at this gate.

### Gate 2 — versioned rewrite state and symbolic `Permute` folding

This is the Native half of milestone 1.

**Code work**

1. Add `constant_store` to `Rewrite.Make` state, `origin`, `apply`, and `pack`.
   Because `Rewrite4` is another application of the functor, add regression
   coverage showing an empty store survives a Native4D rewrite unchanged.
2. Add recipe support for deferred fold definitions and literals.  Keep the
   materialized path temporarily so old numeric tests remain useful.
3. Update `Pass.env` and `Fold_const`: a single-output node folds when every
   operand is both a graph `Input.Constant` and bound in `Constant_store`.
   It records `Apply { op = node.op; output = output_sig }`; it does not call
   `Region.extract` or `Eval_direct`.
4. Make initial PT2 constants symbolic captures in `transform_lowered`.
   Programmatic fixtures can explicitly seed captures or literals.
5. Make `Pipeline.canonical` run symbolic fold rounds regardless of archive
   preload.  Update pipeline test expectations to assert equal graph structure
   for preloaded and payload-free runs.
6. Update `Pt2_native_graph`/lens-facing reporting so a derived constant can be
   printed as a plan definition while archive lookup remains unavailable through
   `captured_target`.

**Tests**

Extend existing tests rather than duplicating their coverage:

- `test/native/fold_const_test.ml`: the `const_permute` fixture folds with no
  raw payload, prints `t2 = permute(t1)`, and has no `t1` main-graph input.
- `test/native/pack_test.ml` and `test/native/rewrite_test.ml`: a packed graph
  has renamed exported bindings, unrenamed internal plan ids, and no dangling
  references after source graph inputs are removed.
- `test/native/pipeline_test.ml`: payload-free and preloaded canonical runs have
  the same graph and plan; applying canonicalization twice yields an identity
  map and unchanged plan.
- `test/native/lens_test.ml`: `captured_target` remains `None` for the derived
  permuted constant even though the plan leaf retains the source target.

**Verification and commit gate**

Run focused Native tests first, then Gate 0 commands.  Inspect the updated
expect output for the single artificial permute fixture.  Commit this Native
symbolic-fold slice before adding a materializer, Native4D handoff, or
BatchNorm operations.

### Gate 3 — symbolic verification, Native4D handoff, and explicit materialization

This completes milestone 1.

**Code work**

1. Add `Const_ssa_symbolic`.  Extend `Ground_eval.Env` and `Map_verify` APIs to
   accept a `Constant_store.t` rather than just the materialized map.  Grounding
   a plan export recursively expands its allowed `Apply` nodes; a captured leaf
   becomes the source-side symbolic constant keyed by its stable capture id.
2. Thread the store through `Pass.verified`, final composition in
   `Native_interp.transform_lowered`, and
   `Native4d.Framework.Verify_from_native.run`.  Preserve existing behavior for
   callers with an empty store.
3. Extend `Native4d.Lower.t` with `constant_store : Constant_store.t` and
   change `Lower.convert` to accept it.  Restrict/remap only destination-visible
   export bindings.  Do not change `Eval_direct4`.
4. Add `Const_ssa_materialize`:

   ```ocaml
   type resolver = Const_ssa.Capture.t -> (Tensor.packed, error) Err.t
   val materialize :
     ?needed:Tensor_id.Set.t -> resolver -> Constant_store.t ->
     (Constant_store.t * report, error) Err.t
   ```

   It performs a topological, memoized evaluation.  For each `Apply`, construct
   a one-node/extracted Native graph and call the same `Eval_direct` operation
   path currently used by `Fold_const`; validate the result against its stored
   signature before caching it.
5. Change `Native_interp.evaluate` to call materialization explicitly with an
   archive resolver before invoking the unchanged `Eval_direct`.  Add the same
   explicit caller-side materialization for Native4D evaluation helpers.  Extend
   the `loaded` report with plan materialization/cache statistics rather than
   misreporting a derived value as archive-loaded.

**Tests**

Add an M1 fixture, preferably to `test/native/graph_fixtures.ml`, and exercise
it from `fold_const_test.ml`, `native4d/lower_test.ml`, and
`native4d/compute_test.ml`:

- captured 6D convolution weight -> plan `Permute` -> visible 4D weight ->
  `Conv2d`;
- two convolution consumers of the same visible constant;
- source resolver with a coordinate-varying ramp;
- no-payload symbolic fold and Native4D conversion;
- typed missing-constant error when either direct evaluator is called before
  materialization;
- one capture read and one permute evaluation for two consumers;
- numerical equality with the original materialized Native graph; and
- map verification that proves the folded weight is the original permutation,
  not an unrelated free constant.

Add negative tests that a 6D **visible** plan export is rejected by Native4D,
while the 6D captured leaf behind a 4D export is accepted.

**Verification and commit gate**

Run all Native and Native4D tests, `make runtest`, and `make js.runtest` if
Const-SSA printing/serialization reaches the JS build.  Run payload-gated tests
when available.  Commit the complete milestone-1 vertical slice only after the
artificial graph passes all symbolic, conversion, materialization, and direct
evaluation assertions.

### Gate 4 — BatchNorm constant language and symbolic canonicalization

This is the language half of milestone 2.

**Code work**

1. Extend the registry and all four operation hooks (validation, signature,
   symbolic grounding, materialization) for `Add`, `Sub`, `Mul`, `Div`, and
   `Sqrt`.  Reuse each existing Native op payload and broadcast rules.
2. Represent transform-created epsilon/identity tensors in
   `Fold_batch_norm.literal` as Const-SSA `Literal` definitions.  Do not place
   them into `Rewrite.constants` merely because their values are known.
3. Confirm `Fold_batch_norm.pattern` sees the first symbolic permute result as
   a regular graph constant.  If a predicate needs more than input kind, route
   it through `Constant_store.is_effective_constant`; do not add a second
   batch-norm-specific constant test.
4. Run the existing second fixed-point `Fold_const` round.  It must convert
   `shifted`, `denom`, `scale`, `scale_n`, `weight'`, `centred`, `scaled`, and
   `bias'` into plan values.  The rebuilt convolution consumes only exported
   `weight'` and `bias'` constants.
5. Update the canonical pipeline API/CLI migration selected at Gate 0.  Preload
   can warm materialization but cannot affect the canonical graph.

**Tests**

Extend `test/native/fold_batch_norm_test.ml`:

- no-payload test: after `Fold_batch_norm` plus symbolic fold, no BatchNorm or
  parameter-arithmetic node remains in the main graph; the plan prints the
  eight expected definitions;
- all eight optional-operand combinations from the existing numeric test run
  without payloads and have valid plan bindings;
- materialize the same plan with fixture tensors and compare the resulting
  weights/biases and model output with the existing payload-backed baseline;
- map verification preserves the existing `Equivalent` claim on the rebuilt
  convolution and does not turn it into `Identical`;
- each newly supported operation has scalar/broadcast, bad-signature, and
  direct-materialization tests in `const_ssa_test.ml`.

**Verification and commit gate**

Run focused fold, symbolic, and verification tests, followed by Gate 3's full
suite.  Commit the completed BatchNorm language slice before adding any real
`model.json` fixture or broader operation support.

### Gate 5 — payload-free MobileNetV1-class `model.json` to Native4D

This completes milestone 2.

**Code work**

1. Identify a committed MobileNetV1 `model.json` source.  If the repository
   does not contain one, add a deterministic MobileNetV1-shaped ATen JSON
   fixture with standard conv, depthwise conv, and inference BatchNorm patterns.
   It must travel through the existing importer, not a hand-built Native graph.
2. Add a payload-free conversion entry point or test helper that performs:

   ```text
   lower model.json -> transform_lowered -> Pipeline.canonical
   -> Native4d.Lower.convert with Constant_store
   ```

   It must not construct a `Pt2_archive` or seed zero-valued weight payloads.
3. Add a parallel fixture/archive resolver for the same graph so the explicit
   materializer can be tested numerically without changing the payload-free
   structural path.

**Tests**

- A hermetic cram or expect test proves the payload-free `model.json` path
  reaches Native4D and reports no `Requires_payloads`/missing-payload failure.
- Inspect graph output: no inference BatchNorm and no main-graph folded
  parameter arithmetic; only plan definitions carry that work.
- Native-to-Native4D symbolic verification succeeds with the expected
  equivalent cluster(s).
- Materializing with deterministic fixture weights produces an executable
  Native4D result agreeing with the materialized Native result.

**Verification and commit gate**

Run the full hermetic suite including the new `model.json` test.  When a real
fixture/archive is used, also run its gated `make pt2.runtest` target.  Commit
milestone 2 only once the same source supports both no-payload conversion and
explicitly materialized numeric execution.

### Gate 6 — establish the current payload-backed Native4D corpus

This begins milestone 3; it is a measurement commit, not a speculative language
extension.

**Code work**

1. Add a deterministic fold trace to the current materialized fold path.  It
   records `(Native op name, static configuration, output signature)` for each
   successful single-output constant fold.
2. Run the current payload-backed Native-to-Native4D cohort, including the
   existing `resnet18`/`mobilenet_v2` Native4D cram cases and every currently
   successful configuration in project model manifests.
3. Check in a corpus manifest plus sorted Const-SSA operation manifest.  The
   latter is the exact completion target; expected additions such as `Reshape`,
   `Clone`, or scalar forms remain hypotheses until the trace proves them.
4. Add a CI test that compares the trace manifest with the Const-SSA registry
   and names the first missing operation/configuration.

**Verification and commit gate**

The trace must be deterministic across repeated runs and must not change any
model output.  Run all available payload-gated tests.  Commit the baseline
corpus/manifest before implementing the first newly required operation.

### Gate 7 — Const-SSA parity operation slices

Repeat this gate once per missing operation family from Gate 6.  Do not batch
unrelated operations: each needs separate numerical and symbolic review.

**Per-operation work**

1. Add the exact operation/configuration to the closed registry.
2. Reuse the corresponding `Graph_ir` payload, `Graph_shape`, symbolic form,
   and direct evaluation path.  Add a new dedicated representation only when
   those cannot express the operation exactly.
3. Add validation, symbolic grounding, materialization, cache-sharing, and
   signature tests.
4. Add the relevant current model to the payload-free Native->Native4D
   regression matrix.
5. Regenerate the operation manifest; it must remain a subset of the registry.

**Verification and commit gate**

Run focused tests plus `make runtest`; run the affected real model in
payload-backed and payload-free/materialized modes where available.  Commit one
operation family per verified slice.  Only after the manifest is empty may the
milestone-3 completion gate run.

### Gate 8 — milestone 3 completion

**Required evidence**

1. Every model/configuration in the frozen payload-backed Native4D corpus
   converts from its payload-free `model.json` form.
2. For each model, payload-backed and payload-free canonical Native graphs are
   structurally identical; their plan exports are equivalent and preload did
   not choose a different rewrite.
3. Explicit materialization with the real archive recreates the executable
   Native/Native4D behavior under the existing exact/equivalent policy.
4. The fold trace manifest is fully covered by the registry.
5. No direct evaluator has acquired an archive parameter, a plan evaluator, or
   lazy fallback behavior.

**Final verification and commit gate**

Run `make runtest`, `make js.runtest` when applicable, and the complete
available `make pt2.runtest` cohort.  Review `git diff --check`, all updated
expect/cram outputs, and the manifests.  Commit the milestone-3 completion as
a dedicated commit; only then begin any extension beyond the current Native4D
corpus (multi-output plans, additional Native ops, or a richer Native4D domain).

## Error and compatibility policy

Add typed errors rather than strings for: malformed plan, unsupported
Const-SSA op, missing captured source, materialization signature mismatch,
plan binding overwrite, missing exported plan value, and a non-four-axis
Native4D-visible export.  Existing `Missing_constant` remains the direct
evaluator's error when callers bypass explicit materialization.

During migration, preserve the old materialized fold behind
`fold_to_materialized` and use it only in tests/fallback compilation.  Delete
it only after Gate 8 proves the current supported corpus no longer depends on
payloads to select a graph rewrite.  Do not retain a silent compatibility mode
where `--fold` changes graph structure; that would reintroduce the original
analysis problem.

## Completion criteria

The work is complete for this plan when Gate 8 has a committed green checkpoint.
At that point Const-SSA is a Native symbolic constant representation, all
currently payload-exportable Native4D models have a payload-free conversion
path, and numeric execution remains an explicit materialization followed by the
unchanged direct evaluators.
