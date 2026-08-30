# Const-SSA implementation plan

**Status update (2026-08-30): Gates 0-6 are implemented and committed**
(`ffbef44 feat(native): fold constants and batch norm through Const-SSA`, plus
the alphabetization pass `51c5ea8`). `lib/native/transform/const_ssa.ml`,
`constant_store.ml`, `const_ssa_symbolic.ml`, and `const_ssa_materialize.ml`
all exist; `Const_ssa.allows` (const_ssa.ml:79-81) admits the M2 language
`{Add, Sub, Mul, Div, Sqrt, Permute}`; the Gate 6 fold-trace baseline lives at
`data/const-ssa-native4d-fold-manifest.txt` (mobilenet_v2-shaped corpus, no
`Reshape`/`Clone` in that trace). See [const-ssa-impl.md](const-ssa-impl.md)
for the gate-by-gate mapping.

**Gate 7 update (2026-08-30, same day): done for the current 100-model
corpus.** Five operation slices landed in sequence, each discovered by
recensus after the previous one landed rather than by rerunning Gate 6's
fold-trace tool (which needs downloaded `.pt2` archives; the payload-free
`model.json` census was enough to find and verify each gap):

1. `Reshape` (`2f180d9`) — cleared `nf_regnet_b0`/`vit_tiny_r_s16_p8_224`'s
   frontier, which moved to `Expand`.
2. `Expand` (`9b97429`) — cleared that and `test_vit4`'s tracked gap; both
   reached `native_builds:true`. `nf_regnet_b0`/`vit_tiny_r_s16_p8_224`'s
   frontier moved to `Mul_scalar`.
3. `Mul_scalar` (`035772c`) — cleared it; `nf_regnet_b0` reached
   `native_builds:true`.
4. `Pow` (`5cf86d9`) — cleared `csatv2`'s frontier (moved to `Add_scalar`).
5. `Add_scalar` (`19f33d2`) — cleared it; `csatv2` reached
   `native_builds:true`.

Recensus after the fifth landing: **zero `"is not a Const-SSA operation"`
blockers remain anywhere in the 100-model corpus.** `Const_ssa.allows` now
admits `{Add, Sub, Mul, Div, Sqrt, Permute, Reshape, Expand, Mul_scalar, Pow,
Add_scalar}`. All four models this unblocked now stop at a real Native4D
boundary instead (three on an intrinsic `D`/`T` axis, `csatv2` on a missing
`Select4` legalization) — see [todo-ops.md](todo-ops.md).

This closes Gate 7 for the corpus as currently frozen. Gate 8's
completion criteria (the Gate 6 fold-trace manifest fully covered, run against
the frozen mobilenet_v2-shaped baseline in
`data/const-ssa-native4d-fold-manifest.txt`) is a separate, narrower claim —
that manifest already only names ops in `{Add, Sub, Mul, Div, Sqrt, Permute}`
and was not touched by this work. A new operation surfacing behind a future
landing (e.g. once `Select4` unblocks csatv2's next frontier) is expected and
should repeat this same recensus-driven Gate 7 slice, not be treated as a
regression.

## Goal

Make constant folding a Native, payload-independent transformation.  A
constant-only Native subgraph is replaced by a Const-SSA definition even when
captured tensor bytes are unavailable.  Native4D then converts the resulting
canonical Native graph; it does not own a separate constant language.

The resulting system has two explicit stages:

```text
PT2 model.json / Native graph
  -> Native symbolic folding -> Native graph + Const-SSA plan
  -> Native4D conversion      -> Native4D graph + mapped Const-SSA plan
  -> materialize_constants    -> executable graph + Tensor.packed constants
  -> Eval_direct / Eval_direct4
```

Direct evaluators must remain materialized-only.  Const-SSA is evaluated
symbolically for analysis, shape/signature checking, and transform verification;
numeric evaluation happens only at the explicit `materialize_constants`
boundary.

The current blocker is localized: `Fold_const` only folds when every constant
operand has a bound `Tensor.packed`, while canonicalization needs folding before
batch-norm folding and before Native4D sees the relaid weight.  The plan changes
that condition to "every operand is an effective constant with a Const-SSA
definition".

## Scope and invariants

### Native owns the plan

Const-SSA belongs to `lib/native/transform`, alongside `Rewrite`, `Recipe`, and
the canonical pipeline.  It is present in the result of
`Native_interp.transform_lowered`, which is already the archive-free
`model.json` path.  It must not be implemented as a Native4D-only sidecar.

Native4D conversion consumes an already-folded Native graph.  It carries forward
the Const-SSA values referenced by destination constants, remapping their
exported tensor ids as required by the Native-to-Native4D conversion map.

### Const-SSA is a separate shared DAG

The representation is a flat, validated SSA program, not a recursive payload in
`Tensor.packed` and not a nested expression per graph input.  Each operation has
one result id and named input result ids.  An output constant in the main graph
maps to one Const-SSA result id.

The plan has three leaf/definition forms:

```text
Captured(source)       original archive/model constant, with Native signature
Literal(tensor)        transform-created scalar or tensor value
Apply(op, operands)    a permitted pure, single-result constant operation
```

`Captured` must retain a direct captured-source identity usable by the archive
resolver.  Graph provenance is diagnostic data only: it cannot replace the
source identity because a permuted output does not have the captured tensor's
physical layout.

### Required properties

- Every Const-SSA result has a validated Native `Tensor_sig`: shape, format,
  quantization, and static operation parameters.
- The plan is topologically ordered and acyclic; all operands are defined
  earlier in the same plan or are declared leaves.
- The operation registry is explicit and closed.  A new `Graph_ir` operation is
  not automatically legal merely because `Fold_const` could evaluate it today.
- Direct execution consumes only `Tensor.packed` values produced by the explicit
  materializer.  It has no archive access and no fallback evaluation of a plan.
- A Native4D-visible plan output must have `T = D = 1`.  Captured leaves and
  intermediate Native plan values may be six-dimensional; for example a 6D
  captured convolution weight can feed `Permute` whose exported result is 4D.
- Symbolic evaluation and numeric materialization share the Native operation
  definitions as far as possible.  Const-SSA must not grow independent shape,
  broadcasting, dtype, quantization, or rounding rules.

## Operation registry

The registry is deliberately staged.  It describes *compile-time constant
expressions*, not runtime model operations: ReLU or convolution may be valid
Native/Native4D runtime operations without being admitted to Const-SSA.

| Family | Operation | M1 | M2 | M3 completion rule |
| --- | --- | :---: | :---: | --- |
| leaves | `Captured`, typed `Literal` | yes | yes | yes |
| layout | `Permute` | yes | yes | yes |
| layout | `Reshape`, `Clone` | no | no | admit if present in the weighted Native4D corpus's fold trace |
| pointwise | `Add`, `Sub`, `Mul`, `Div` | no | yes | yes |
| pointwise | `Add_scalar`, `Div_scalar`, `Mul_scalar` | no | no | admit if present in the corpus trace |
| unary | `Sqrt` | no | yes | yes |
| unary | `Clamp`, `Hardtanh`, `Relu`, `Silu`, `Hardsigmoid`, `Hardswish` | no | no | admit only if a current supported model folds one |
| reductions / spatial | `Mean`, pools, convolution | no | no | out of the initial completion set unless the recorded corpus requires one |
| multi-result | `Max_pool2d_with_indices`, `Unbind`, etc. | no | no | out of scope; requires atomic multi-result plan definitions |

M3's completion set is empirical and reproducible, rather than guessed: trace
the operation names of every successful payload-backed `Fold_const` application
over the frozen Native4D model corpus, then require Const-SSA support for their
union.  The expected base is the M2 set, with `Reshape`/`Clone` and scalar forms
as the likely extensions.  The trace, checked into the test fixture or generated
deterministically, is the authority for additions; no broad "all pure Native
ops" promise is implied.

For every admitted operation, the registry entry must provide:

1. operand count and static-parameter validation;
2. Native output-signature inference;
3. symbolic construction/evaluation used by map verification; and
4. materialization through Native's existing direct operation semantics.

An operation outside the registry stays in the main graph during symbolic
folding.  In a materialization-enabled compilation mode, it may instead be
evaluated to a literal only when all of its dependencies are available; absent
payloads must produce a typed "cannot materialize" result or a documented skip,
never an implicit direct-evaluation fallback.

## Architecture work

### 1. Const-SSA data and validation

Add a Native transform module for the immutable plan plus a typed validation
error row.  Its API should expose plan construction, lookup by graph tensor id,
signature lookup, printing/serialization, and a deterministic topological walk.

Extend rewrite state so that it carries:

```text
materialized constants: Tensor_id -> Tensor.packed
constant definitions:   Tensor_id -> Const_ssa.value_id
constant plan:          Const_ssa.t
```

`Recipe` needs a symbolic counterpart to `fold_to_constant`: it preserves the
main-graph output id, marks it `Input.Constant`, records correspondence and
provenance, and attaches a Const-SSA result instead of a payload.  Literal
constants created by `Fold_batch_norm` are plan `Literal` leaves in a symbolic
run; they need not be eagerly allocated as data in rewrite state.

The plan must survive `Rewrite.apply`, fixed points, packing, and Native-to-
Native4D id remapping.  Deleting a main-graph edge must not delete a captured
plan source still needed by a derived output.

### 2. Effective constants and symbolic verification

Define one `is_effective_constant` query for Native passes.  It is true for a
direct constant with a materialized payload, a captured plan leaf, a literal,
or an `Apply` whose dependencies are effective constants.  Pattern matching for
batch-norm and future constant consumers uses this query rather than only
`Graph_view.is_constant`.

Extend symbolic grounding/map verification to expand a plan output as its
Const-SSA expression.  Without this, replacing `permute(w)` by a new graph
constant would make the symbolic verifier treat the two as unrelated free
constants.  A symbolic expansion has no archive reads; captured leaves remain
symbolic constants with their original source identities.

### 3. Materialization

Implement an explicit Native `materialize_constants` operation over a symbolic
transformed graph.  It accepts a resolver for captured sources (archive-backed
in production and synthetic in tests), evaluates requested plan values in
topological order, and memoizes every result.  The output is the existing
materialized constant table used by `Eval_direct`.

Materialization must validate that resulting payloads match the plan's stored
signatures.  It must preserve current failure behavior for unsupported dtypes,
missing capture data, and invalid operations.  The direct evaluator itself is
not changed to perform this work.

### 4. Pipeline and conversion

Change `Pipeline.canonical` so both constant-fold rounds are symbolic.  The
payload/preload flag no longer chooses graph structure; it only determines
whether a later caller can materialize immediately.

Native4D domain checking and lowering operate after Native canonicalization.
They receive the graph plus its plan bindings.  The shape check applies to the
main graph and exported plan results, not to hidden captured source leaves.
`Native4d.Lower.t` must carry the mapped plan bindings in addition to its
current materialized-constant table, so a payload-free `model.json` conversion
is a real result that can later be materialized.

## Milestone 1 — end-to-end vertical slice

### Purpose

Exercise every architectural boundary with the smallest possible artificial
graph.  This milestone proves staging and identity handling; it does not claim
useful model coverage.

### Language

Only these operations are admitted:

```text
Captured, Literal, Permute
```

### Fixture

Build a Native graph containing an input activation, a captured six-axis
convolution weight, `Permute(weight)` whose output has `T = D = 1`, and a
`Conv2d` consuming that output.  The test source resolver supplies a small ramp
payload for the captured weight.  The output shape and permutation must be
non-trivial so a missing transform is observable.

### Required evidence

1. A payload-free Native canonical run removes the main-graph `Permute` and
   prints a plan definition for the new constant.
2. Native symbolic/map verification proves the folded edge represents the
   source permutation without reading its captured payload.
3. Native4D conversion succeeds even though the captured plan leaf is 6D; only
   the exported permuted value is checked as 4D.
4. Calling direct Native or Native4D evaluation before materialization returns a
   typed missing-constant result, not a hidden archive read.
5. `materialize_constants` loads the captured ramp once, materializes the
   permutation once, and direct Native4D evaluation agrees with the original
   materialized Native graph.
6. Reusing the transformed weight from two consumers proves cache sharing.

This fixture must cover plan validation, rewrite/packing, provenance versus
captured identity, symbolic evaluation, Native-to-Native4D mapping, explicit
materialization, and both direct evaluators.

## Milestone 2 — payload-free MobileNet-class export

### Purpose

Export a simple convolution-plus-inference-BatchNorm model, such as
MobileNetV1, from its ATen `model.json` representation to Native4D without a
weights archive.  The success criterion is graph conversion and symbolic
verification, not numeric inference.

The repository's existing payload-free `model.json` lowering path through
`Native_interp.transform_lowered` is the integration boundary.  If no committed
MobileNetV1 JSON fixture exists, add a small deterministic MobileNetV1-shaped
ATen fixture; do not weaken the milestone to a hand-built Native graph.

### Language extension

Add the complete expression emitted by the current Native batch-norm fold:

```text
Captured, Literal, Permute,
Add, Sub, Mul, Div, Sqrt
```

This includes Native broadcasting and the typed scalar literals for epsilon and
for absent `gamma`, `beta`, or convolution bias.  In graph form the folded
parameters are:

```text
shifted  = running_var + epsilon
denom    = sqrt(shifted)
scale    = gamma / denom
scale_n  = permute(C -> N, scale)
weight'  = conv_weight * scale_n
centred  = conv_bias - running_mean
scaled   = centred * scale
bias'    = scaled + beta
```

The second canonical constant-fold round must turn each constant-only result
above into Const-SSA.  It must not require payloads and must leave the rebuilt
convolution's `weight'` and `bias'` as exported effective constants.

### Required evidence

1. `model.json` import, Native canonicalization, and Native4D conversion
   complete with no archive and no preload flag.
2. The converted graph contains no inference batch-norm nodes and no
   main-graph constant arithmetic nodes for folded parameters.
3. Symbolic map verification reports the correct `Equivalent` status for the
   batch-norm-to-convolution reassociation and does not report payload absence
   as a refutation.
4. A weighted version of the same model materializes the plan and matches the
   existing payload-backed canonical/Native4D result within the current
   equivalence policy.
5. Tests cover all eight combinations of optional BatchNorm weight/bias and
   convolution bias, preserving the current fold's single arithmetic path.

## Milestone 3 — parity with currently payload-exportable Native4D models

### Purpose

Make payload-free symbolic conversion cover every model that the baseline code
can currently convert to Native4D when its payloads are preloaded.  This is
parity with the existing supported corpus, not a commitment to support models
outside Native4D's domain.

### Freeze and measure the corpus

At the start of this milestone, run the current payload-backed Native-to-
Native4D tests and record every successful model/configuration in a checked-in
manifest.  The existing Native4D cram coverage (including `resnet18` and
`mobilenet_v2`) and the project model manifests are inputs to that baseline;
the manifest is the acceptance authority rather than an informal model name
list that can drift.

Instrument the old payload-backed fold over that exact corpus to emit the set
of folded Native operation names and static configurations.  Compare it with
the Const-SSA registry.  Extend the registry, one operation family at a time,
until the sets agree.  Expected first additions beyond M2 are `Reshape`,
`Clone`, and scalar pointwise forms, but the recorded trace decides.

### Required evidence

1. Every baseline model converts from its payload-free `model.json` through
   Native canonicalization to Native4D.
2. Its payload-backed and payload-free canonical graphs have the same structure
   and equivalent mapped Const-SSA definitions; preload does not change the
   graph.
3. Materializing the payload-free result with its real archive produces the
   same executable constants and Native4D graph behavior as the baseline
   payload-backed run, under the existing exact/equivalent claim policy.
4. The checked-in fold-op manifest is a subset of the Const-SSA capability
   manifest.  Adding a model or a new fold-producing rewrite that violates this
   relation fails CI with the missing operation named.
5. Unsupported operations, multi-output folding, and models outside the
   Native4D domain continue to produce typed diagnostics rather than silently
   materializing or widening the dialect.

## Cross-cutting test matrix

Every newly admitted Const-SSA operation needs:

- validation failures for bad arity, signature, format/quantization, undefined
  operand, cycle, and illegal operation admission;
- a symbolic evaluation test with no payload source;
- a materialization test against the existing Native direct operation;
- a cache-sharing test when the result has multiple consumers;
- rewrite, pack, and serialization/printing coverage; and
- Native-to-Native4D conversion coverage whenever the result can be a visible
  four-axis constant.

Batch-norm tests additionally pin the `Equivalent` correspondence claim and
each optional-parameter combination.  The model milestones add full-pipeline
cram tests from `model.json`; real-weight numeric tests belong after explicit
materialization and must not be used to make symbolic conversion pass.

## Non-goals for this plan

- Making `Eval_direct` or `Eval_direct4` lazy or archive-aware.
- Treating graph provenance as payload identity.
- Adding a general second graph dialect that duplicates all `Graph_ir` ops.
- Supporting multi-output constant folding before the plan has an atomic
  multi-result definition and materialization contract.
- Expanding Native4D's operation/domain surface merely because a captured
  constant's private source is six-dimensional.

