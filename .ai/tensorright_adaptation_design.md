# TensorRight analysis and adaptation for mltorch

## Status

Analysis and design proposal. No implementation has landed.

This document is based on the TensorRight POPL 2025 paper at
`/tmp/TensorRight/3704865.pdf`, the Haskell implementation under
`/tmp/TensorRight`, and mltorch's current Native and Native4D implementation and
design records.

The algorithms below were checked against the source, not inferred from the
README alone. The TensorRight suite was not executed in this workspace because
`stack`, Z3, and cvc5 are not installed; performance and rule-count results are
therefore reported from the paper and repository documentation rather than
re-measured here.

## 1. Executive conclusion

TensorRight should influence mltorch, but it should not be imported wholesale
and it should not replace `Map_verify`.

The two systems prove different and complementary facts:

- TensorRight verifies a handwritten, parameterized **rewrite schema** for
  arbitrary ranks and sizes, under the mathematical semantics of its DSL.
- mltorch verifies an **actual rewrite application** between two concrete graph
  snapshots, using the same symbolic per-pixel implementation as the Native
  evaluator and preserving float32 materialization, formats, constants, and the
  transformation's correspondence map.

The most useful first adaptation is therefore a native-only, offline mltorch
rule-schema checker with symbolic positive extents and explicit preconditions.
It should use SMT to prove shape, access, and index-transform obligations for
the fixed `N/T/D/H/W/C` frame. The current `Map_verify` remains the second layer
that validates the executable pass on every concrete application.

Do **not** begin by adding aggregated axes or arbitrary-rank tensors to Native.
Native deliberately uses a fixed six-axis, channels-last frame, Native4D is a
checked subset of it, and mltorch's important rules include layout-sensitive
`Permute` and `Reshape` operations that TensorRight explicitly does not support.
Arbitrary rank would add a new semantic problem without improving the current
IR's guarantee.

## 2. TensorRight in plain language

### 2.1 The problem it solves

A rewrite such as

```text
slice(x, start=0, end=size, stride=1)  ==>  x
```

is intended to hold for every legal tensor size and often for every rank. A few
unit tests at rank 2 do not establish that. TensorRight provides:

1. a language for writing the two tensor expressions and their precondition;
2. mathematical semantics for the supported tensor operators;
3. an algorithm that reduces arbitrary-rank checking to finitely many ranks;
4. symbolic execution at each required rank; and
5. SMT queries over arbitrary axis sizes, indices, attributes, and input values.

The key distinction from ordinary randomized or small-shape testing is that
axis sizes remain symbolic. A bounded-rank obligation still covers every size.

### 2.2 The rewrite language

TensorRight is an embedded Haskell DSL. A rule declares:

- **rank classes (`RClass`)**: families of axes that must have the same rank;
- **aggregated axes**: a group containing any positive number of ordinary named
  axes that play the same role;
- **maps**: per-axis symbolic values such as sizes, padding, strides, or slice
  bounds;
- **input tensors**: uninterpreted functions from valid coordinates to scalar
  values;
- **LHS and RHS expressions**: compositions of tensor operators;
- **preconditions**: predicates folded elementwise over maps; and
- **a rewrite declaration** naming the LHS and RHS.

The README's dynamic-slice example has the characteristic shape:

```haskell
rcls <- newRClass "rcls"
[size, start, start', length, end, stride] <- newMaps [...] rcls
x <- newTensor "X" [rcls --> size]

lhs <- dynamicSlice x ...
rhs <- slice x ...

precondition [end, start', length] ...
precondition [stride] ...
precondition [start, start'] ...
rewrite "DynamicSlice(X) => Slice(X)" lhs rhs
```

An aggregated axis is not merely an unknown dimension name. It represents an
unknown-length set of named axes. Maps can be combined only pointwise, which is
one of the restrictions that makes the rank theorem possible.

An `RClass` relates several aggregated axes. They instantiate to the same rank
and have a canonical position-wise bijection. This is what lets a relabel rule
say that two different axis groups have corresponding first, second, and later
axes without choosing a rank in the rule.

### 2.3 Operator semantics

TensorRight treats a tensor as:

```text
valid coordinate -> scalar value
```

An operator defines both its output shape and how an arbitrary output
coordinate is mapped to input coordinates or scalar values. Examples:

- `expand` drops the introduced axes from the input access;
- `slice` maps output coordinate `a` to `start + a * stride`;
- `pad` uses a conditional to choose a padding value or an adjusted input load;
- `relabel` renames axes through a bijection;
- `concat` chooses one input and adjusts the concatenation coordinate; and
- `reduce` produces a special uninterpreted reduction element because the
  number of reduced values is itself unbounded.

The implementation supports Boolean, mathematical integer, and mathematical
real scalars. “Real” here is not IEEE float32.

The core source locations are:

| Concern | TensorRight source |
|---|---|
| DSL construction and operator checks | `src/TensorRight/Internal/DSL/DSL.hs` |
| expression and rule records | `src/TensorRight/Internal/DSL/Expr.hs` |
| abstract shapes and rank-class references | `src/TensorRight/Internal/DSL/Shape.hs` |
| concrete-rank symbolic evaluation | `src/TensorRight/Internal/DSL/Eval.hs` |
| tensor/operator semantics | `src/TensorRight/Internal/Core/Tensor/Typed.hs` |
| rank-bound inference | `src/TensorRight/Internal/DSL/BoundInference.hs` |
| rank enumeration | `src/TensorRight/Internal/DSL/Verify.hs` |
| SMT proof obligations and counterexamples | `src/TensorRight/Internal/Core/Verify.hs` |

## 3. The verification algorithm

### 3.1 The logical obligation

For a rule `LHS ==> RHS` guarded by `Pre`, the intended obligation is:

```text
for every tensor and attribute valuation,
  Pre and Valid(LHS)
  imply Valid(RHS), Shape(LHS) = Shape(RHS), and Value(LHS) = Value(RHS).
```

Tensor equality is reduced to equality at an arbitrary symbolic output access.
Input tensors are uninterpreted functions, so equality of loads is decided by
their tensor identity and coordinate arguments rather than by enumerating data.

### 3.2 Why checking rank 1 is insufficient

The paper gives a slice/update example that is valid at rank 1 and invalid at
rank 2. The mismatch requires two different axes to preserve different control
conditions. This demonstrates why “the rule worked for a vector” is not a proof
for matrices or higher-ranked tensors.

### 3.3 Counterexample projection

TensorRight's central argument runs backward from counterexamples.

Suppose a rule fails when an `RClass` has rank `k + 1`. The failing output
coordinate and all attributes contain `k + 1` per-axis entries. TensorRight
asks whether one axis can be projected away while retaining:

- the truth value of every relevant control condition; and
- the equality or inequality relationships among every distinct access to the
  same input tensor.

Each folded condition needs at most one “distinguishing” axis to preserve its
truth value. Each pair of distinct accesses to one input tensor needs at most
one axis to prevent two unequal accesses from collapsing onto the same point.
If `k` is at least the total number of potentially required axes, some
projection keeps all of them. The rank-`k + 1` counterexample then induces a
rank-`k` counterexample.

Taking the contrapositive gives the induction step:

```text
Valid at rank k  ==>  Valid at rank k + 1
```

for every `k` at or above the inferred cutoff.

### 3.4 The inferred bound

For rank class `c`, the paper's conservative bound is:

```text
K(c) = max(1,
           relevant_conditions(c)
           + sum over input tensors t of choose(relevant_accesses(t, c), 2))
```

Singleton rank classes have bound 1.

The implementation improves this conservative syntactic count before applying
the formula:

- solver-equivalent conditions are deduplicated under the rule precondition and
  validity assumptions; and
- accesses to the same input tensor that the solver proves equal are
  deduplicated.

`BoundInference.hs` attaches metadata to symbolic map entries and tensor
functions, walks the Grisette term, groups function applications by input
tensor, and computes the pair count. It obtains the term from a rank-1 symbolic
execution; the DSL restrictions ensure the scalar control/access structure is
rank-independent.

With several rank classes, each class is analyzed independently while the other
ranks are held arbitrary. TensorRight then checks the Cartesian product:

```text
rank(c1) in 1..K(c1), ..., rank(cp) in 1..K(cp)
```

This product can be exponential in the number of classes, although the paper
reports that 110 of its 115 verified XLA rules needed only one bounded task.

### 3.5 A bounded-rank SMT obligation

For each rank combination, TensorRight:

1. creates one symbolic integer for every map entry and axis extent;
2. creates one uninterpreted input function per tensor;
3. symbolically evaluates both expressions at a general symbolic coordinate;
4. checks the LHS and RHS have the same named axes directly;
5. asks the solver to prove equal extents;
6. asks it to prove that a valid LHS access maps to a valid RHS access; and
7. asks it to prove RHS validity and scalar equality.

The solver is given the negation of each implication. `unsat` is the proof;
`sat` supplies a counterexample model. A solver failure or timeout is not a
proof.

The implementation also diagnoses whether a precondition is stronger than
necessary and whether it admits at least one valid input. The latter is printed
as a diagnostic rather than being a hard failure in the current source. An
mltorch adaptation should be stricter: an unsatisfiable rule precondition must
make the rule verification fail as vacuous.

### 3.6 Reductions

Ordinary symbolic execution cannot expand a reduction over a symbolic extent.
TensorRight instead leaves a reduction as an uninterpreted reduction element.
It has a small normalization system for distributing scalar multiplication,
combining nested reductions, and combining products of reductions.

For two reduction elements, the user supplies a relation between the LHS and
RHS reduction indices. TensorRight asks SMT to prove that relation is a
bijection and that related elements are equal. This proves equality of the
reduced multisets under the mathematical reduction semantics.

This is the one manual proof hint in the system. The paper reports that 13 of 17
reduction rules were handled this way. The remaining cases include quantified
solver failures and rules whose two sides reduce sets of different cardinality.

There is an important paper/source distinction here. In
`Core/Verify.hs`, failure to prove uniqueness, totality, or accessibility of the
supplied reduction-index relation is printed as a `[WARNING]`; it does not throw
and therefore does not by itself make the overall rule result fail. The later
queries still prove equality for pairs satisfying the supplied relation. A
TensorRight reduction run should consequently be treated as a completed proof
only when these relation checks also succeed. An mltorch adaptation should make
an unproved bijection an `Unproved` rule, never a successful result with a
warning.

## 4. What TensorRight does and does not guarantee

When all obligations succeed, TensorRight proves a rule for every positive rank
of every `RClass` and every legal axis size, **within the denotational semantics
of the TensorRight DSL**.

That qualification is important:

- It verifies the handwritten DSL rule, not the XLA or mltorch matcher and
  builder that implement a pass.
- Operator semantics are a second implementation that can drift from the real
  compiler operation.
- `Real` uses symbolic algebraic-real arithmetic plus a separate abstract
  positive/negative-infinity case. It does not model IEEE float32 rounding,
  NaNs, signed zero, finite-width overflow, quantization, or the exact IEEE
  exceptional-value rules at every graph node.
- `Int` is unbounded mathematical integer, not a fixed-width tensor dtype.
- Aggregated axes are non-empty; rank-zero groups are outside the theorem.
- The rank theorem depends on the DSL restriction that map computation is
  pointwise and conditions are folds over axes. It must not be applied to a more
  general language without a new soundness argument.
- Axes are treated as unordered names for layout-insensitive operations.
  General `reshape` and `bitcast` are explicitly unsupported because their
  linearization depends on rank and physical order.
- Reduction equality is incomplete and uses user-provided relations.
- In the checked source revision, failed reduction-relation checks and
  semantically vacuous preconditions are warnings rather than hard failures;
  consumers must not count those runs as full proofs.
- Solver `unknown` and timeout are limitations, not counterexamples or proofs.

The source audit also found a concrete exceptional-value defect:
`Core/Tensor/TensorInt.hs:177-182` leaves `Inf lv` unchanged under `negate`,
although the same module defines `Inf True` as positive infinity and
`Inf False` as negative infinity. Thus `-(+inf)` remains `+inf`, and subtraction
uses this `Num` instance. This was confirmed from the source but not executed in
this workspace. Rules whose proof reaches subtraction or negation of the
abstract infinity case should not be trusted until that branch has a regression
test and is corrected to flip the sign.

The reported evaluation is nevertheless strong evidence for the approach: 121
of 175 XLA algebraic-simplifier rules were expressible, 118 were implemented,
and 115 were automatically verified. Most completed in under one second. The
unexpressed or unproved cases align with the limitations above: layout-sensitive
operators, missing operations, nonlinear/transcendental solver support, and
reductions.

The repository is Apache-2.0 while mltorch is MIT. Reimplementing the ideas is
straightforward; copying or closely porting TensorRight source would require
retaining its license, notices, and attribution for the copied portion.

## 5. What mltorch verifies today

mltorch's Native verifier starts from a different representation and guarantee:

- every tensor occupies the fixed ordered `N/T/D/H/W/C` frame;
- lower-rank tensors are embedded with unit extents;
- Native4D is a checked `N/H/W/C` subset with `T = D = 1`;
- graph signatures currently contain concrete `Vec6.shape` values;
- a pass returns the actual destination graph and a versioned `Graph_map`;
- every correspondence cluster states `Identical`, `Equivalent`,
  `Approximate`, or `Unverifiable`; and
- `Map_verify` checks the clusters of that actual map.

`Eval_symbolic` builds the same per-pixel expression from the same op `Compute`
functors used by direct evaluation. `Ground_eval` then evaluates indices at
each concrete output coordinate, expands producer stages iteratively, and
leaves input cells symbolic. Structural equality proves a local transfer
function for all assignments to those input cells.

The verifier includes facts TensorRight does not model:

- an explicit `Round` at every float32 materialization boundary;
- bitwise constants, including signed zero and NaN distinctions;
- formats and quantization when deciding whether a round can collapse;
- per-graph model constants rather than assuming they equal user inputs;
- many-to-many correspondence clusters and actual graph output positions;
- creation, deletion, fusion, and cross-dialect Native-to-Native4D mappings;
- deterministic counterexample probes where a frontier is fully expanded; and
- separate `Proved`, `Refuted`, `Tested`, and `Unproved` verdicts with budgets
  and coverage.

Its main limitation is that shapes and coordinates are concrete. Exhaustive
checking is proportional to output size and subject to a coordinate budget. A
proof establishes the concrete application, not every shape for which the
pass's matcher could fire. Structural mutation tests and selected fixtures are
currently what connect those concrete proofs to a pass's general reasoning.

## 6. Direct comparison

| Property | TensorRight | mltorch `Map_verify` |
|---|---|---|
| Checked object | handwritten rule schema | actual before/after graph and map |
| Rank | arbitrary through aggregated axes | fixed six-axis frame |
| Extents | symbolic, unbounded | concrete |
| Axis model | unordered named axes | ordered physical/logical frame |
| Layout-sensitive reshape | unsupported in general | central, with row-major semantics |
| Values | Bool, mathematical Int/Real | stored formats decoded to float, explicit f32 rounds |
| Preconditions | symbolic predicates | concrete matcher guards |
| Main proof method | SMT after symbolic execution | grounded structural equality and bounded expansion |
| Reductions | uninterpreted multiset plus bijection hint | ordered reduction expression, concretely expanded |
| Counterexample | solver model for sizes, attributes, indices, values | reproducible free-cell valuation at a concrete coordinate |
| Compiler implementation checked | no | yes, for the concrete application |
| Cross-dialect graph mapping | no | yes |

Neither subsumes the other.

## 7. Recommended mltorch architecture

### 7.1 Two independent proof layers

```text
parameterized rule schema
        |
        v
symbolic extents + arbitrary coordinate --SMT--> schema proof / counterexample
        |
        | names and documents the identity used by
        v
executable pass --applies to concrete graph--> Graph_map
                                               |
                                               v
                                  existing Map_verify
                                  exact graph/rounding/map check
```

The release claim is the conjunction:

1. the mathematical shape/index identity is proved for all admitted extents;
2. the pass's real matcher and builder produce a concrete map accepted by
   `Map_verify`; and
3. mutation tests demonstrate that both checks can fail for the intended defect
   classes.

The schema proof alone must never be printed as “the pass is verified.” It does
not prove that the pass implemented the schema. The concrete verifier alone must
not be printed as “the rule is verified for every shape.” It saw one shape.

### 7.2 Fixed frame first

The first schema language should use exactly `Axis.t` and a symbolic positive
extent for each of the six axes. Native4D schemas can add the assumptions
`T = 1` and `D = 1` or use `Axis4.t` through a thin adapter.

This immediately gives arbitrary-size checking for the representation mltorch
actually executes. It also retains axis order, so `Permute`, fixed-rank
`Reshape`, right-aligned `Mean keepdim=false`, and Native4D legality can be
represented in principle.

Aggregated axes should be a later, separate extension only if mltorch gains a
genuinely arbitrary-rank dialect. They should not be encoded as a clever use of
unit `Vec6` axes: a rank class and a fixed frame answer different questions.

### 7.3 Proposed modules

Keep the checker outside `lib/native`, `lib/core`, and the other JS-reachable
libraries. It is a developer/build tool and should not add an SMT dependency to
the inference runtime.

A minimal native-only library could be `lib/native_rule_verify/`:

| Module | Responsibility |
|---|---|
| `Index.t` | symbolic integer expressions for extents and coordinates |
| `Condition.t` | Boolean predicates and conjunctions |
| `Shape.t` | fixed-frame symbolic extents; a record in its own module |
| `Access.t` | one symbolic coordinate per frame axis |
| `Value.t` | opaque input cells, scalar applications, conditionals, and ordered reductions |
| `Tensor_expr.t` | tensor shape, validity condition, and per-access value function |
| `Rule.t` | inputs, parameters, precondition, LHS, RHS, and claim |
| `Smtlib` | deterministic SMT-LIB generation and model parsing |
| `Verify` | obligations, verdicts, and counterexample reporting |
| `Registry` | named schemas and their associated executable pass names |

Record types must follow the repository rule: each record is module `t`, rather
than sharing field labels in one module or silencing warning 30.

Use an external, pinned solver executable through `Unix.create_process` (no
shell), not an OCaml solver binding in the runtime dependency graph. Emit the
SMT-LIB query on failure so it is reproducible. Only `unsat` proves an
obligation; `sat` is a counterexample; `unknown`, timeout, malformed output, and
solver absence are `Unproved`/errors.

### 7.4 Semantics and claim levels

Do not start by asking SMT to solve arbitrary IEEE-754 graphs.

The useful initial split is:

- **shape and index semantics** use mathematical integers with extents `>= 1`;
- **data-moving `Identical` rules** treat decoded input cells as opaque values,
  retain explicit storage/materialization rounds, and carry any format premise
  needed to collapse a round;
- **pointwise `Identical` rules** treat a scalar operation as an uninterpreted
  function applied to ordered operands, which is sufficient when both sides
  invoke the same operation on the same cells;
- **ordered reductions** retain binder order and bounds, allowing structural
  alpha-equivalence and index-substitution proofs; and
- **`Equivalent` rules** may use mathematical real arithmetic, but the result
  remains `Equivalent` and must never be promoted to bit identity.

Keep explicit node `Round` boundaries in the schema value language even if the
first rules normalize them only structurally. This prevents the schema layer
from acquiring TensorRight's largest mismatch with mltorch.

In particular, removing a data-moving stage is not automatically identical for
an `I32`, `I64`, or quantized graph input: decoding and materializing it through
float32 can change the value. The schema must use the same round-trip premise as
`Ground_expr.normalise`, or decline the proof. An opaque-value abstraction may
hide arithmetic; it must not hide a storage conversion.

For reductions, distinguish two statements:

| Established relation | Maximum honest mltorch claim |
|---|---|
| same values in the same reduction order and same materialization boundaries | `Identical` |
| only a bijection/multiset equality under associative mathematical arithmetic | `Equivalent` |
| approximate floating equality without input ranges and an error bound | no proof |

TensorRight's bijection hint is valuable, but applying its multiset argument to
an mltorch float32 `sum` and labeling the result `Identical` would be unsound.

### 7.5 Obligations per rule

For one fixed-frame schema, generate separate queries for:

1. **precondition satisfiability** — reject a vacuous rule;
2. **LHS feasibility** — the rule must describe at least one valid LHS;
3. **RHS validity** — `Pre and Valid(LHS) imply Valid(RHS)`;
4. **shape equality** — all six extents match;
5. **access validity** — every valid LHS output access is valid on the RHS;
6. **value/index equality** — the two arbitrary output accesses select the same
   cells and scalar structure at the declared claim strength; and
7. **side-condition coverage** — every unchecked semantic assumption is named
   in the report, rather than silently becoming an axiom.

Counterexamples should print the smallest useful subset: symbolic extents,
operator attributes, the output coordinate, differing input coordinates or
branches, and the violated obligation.

### 7.6 Connecting a schema to real pass code

The connection is the hard part and must be explicit.

For the first iteration:

- each executable pass declares a stable `rule_id`;
- `Registry` maps that id to its schema and expected correspondence claim;
- pass code and schema share existing algebra helpers such as
  `Permute.Permute.compose`, `Permute.Permute.lookup`, and
  `Reduce.Mean.map_dims` wherever their types permit it;
- pass tests run with `Map_verify.Policy.Require_proved`; and
- a CI manifest requires both a successful schema proof and concrete mutation
  coverage before a rule is described as covered.

Do not generate the whole pass from the rule language initially. mltorch
patterns include fan-out, graph-output, group-placement, id, format,
quantization, and convex-region constraints that are deliberately outside a
tensor-value equation. Trying to make a first DSL own all of those would turn a
focused verifier into a second transformation framework.

Later, small identity-specific helpers can return both the concrete matcher
witness and the symbolic schema parameters. That narrows drift without asking
the schema language to represent graph traversal.

## 8. Recommended pilot rules

Use three rules that exercise progressively more of the checker.

### 8.1 Permutation composition

```text
P2(P1(x))  ==>  compose(P1, P2)(x)
```

This is the harness baseline. It checks full-bijection validation, the
output-to-input direction convention, shape transport, and arbitrary-coordinate
substitution. Mutations should reverse composition order and cross two axes.

### 8.2 Sinking a permutation through pointwise operations

```text
op(P(a), P(b))  ==>  P(op(a, b))
```

This is the first valuable arbitrary-extent proof. It must include mltorch's
broadcasting rule and ordered operands so `Sub` and `Div` cannot be accidentally
commuted. Unary and binary forms should share the tensor semantics but have
separate matcher obligations.

Mutations should:

- leave one operand unpermuted;
- use a different permutation on one operand;
- swap `Sub`/`Div` operands; and
- omit the broadcast-compatibility precondition.

### 8.3 Transporting permutation through `Mean keepdim=true`

```text
Mean_D(P(x))  ==>  P(Mean_{P(D)}(x))
```

This exercises symbolic extents, named-axis transport, and an ordered
reduction. It also maps directly to `sink_permute_mean.ml`, whose concrete
matcher already checks `keepdim`, precision, and a final shape round-trip.

Mutations should fail when dimensions are not mapped, when `keepdim=false` is
admitted under the same identity, or when the permutation direction is
reversed.

Do not use `reshape_to_permute` as the first pilot. It is important, but
symbolically proving row-major linearization through products, division, and
modulo is substantially harder and TensorRight itself excludes the general
case. Once the solver boundary and verdict discipline are established, the
fixed six-axis setting makes selected reshape rules feasible without needing
arbitrary-rank reasoning.

## 9. Staged implementation plan

### Stage 1 — symbolic shape/index core

- Implement fixed-frame symbolic extents, conditions, coordinates, and loads.
- Emit deterministic SMT-LIB for satisfiability, implication, and equality.
- Return `Proved`, `Refuted` with a model, or `Unproved`; never coerce solver
  uncertainty into success.
- Test an intentionally vacuous precondition and a simple out-of-bounds
  mutation.

Acceptance: one valid and one invalid hand-written index identity produce an
`unsat` proof and a replayable `sat` model respectively.

### Stage 2 — tensor expressions and exact data movement

- Add tensor variables, shapes, validity, arbitrary access, conditionals,
  opaque scalar cells, permutation, and broadcasting.
- Preserve `Round` boundaries in the value AST.
- Add permutation composition and pointwise-sink schemas.

Acceptance: every pilot mutation has first been observed failing, and the
correct schemas prove for arbitrary positive extents.

### Stage 3 — ordered reduction support

- Add reduction binders with symbolic bounds and alpha-normalization.
- Add the `Mean keepdim=true` schema.
- Permit a bijection hint only for `Equivalent` mathematical reductions unless
  the checker also proves order preservation.

Acceptance: the correct mean transport proves `Identical`; the unmapped-axis
and `keepdim=false` mutations do not.

### Stage 4 — pass registry and CI

- Associate the three schemas with their Native pass ids.
- Run existing concrete pass tests under `Require_proved`.
- Add `make native-rule-runtest`, gated on the pinned solver just as JS and PT2
  suites are gated on their external toolchains.
- Keep solver-backed code out of `make build` for JS-reachable libraries.

Acceptance: CI reports schema coverage and concrete `Map_verify` coverage
separately; a missing solver cannot produce a green proof result.

### Stage 5 — selected layout-sensitive rules

- Add fixed-rank row-major linearization terms.
- Attempt the restricted `reshape_to_permute` precondition, with a solver
  timeout reported as `Unproved`.
- Add Native-to-Native4D legality identities where shapes stay symbolic but the
  four-axis invariant is assumed.

Acceptance: a shape-preserving wrong reshape mapping produces a counterexample,
and the proof does not rely on enumerating concrete extents.

### Stage 6 — consider rank classes only if the IR needs them

If a future dialect is genuinely arbitrary-rank:

- define a restricted, layout-insensitive schema sublanguage;
- port the counterexample-projection bound only for that sublanguage;
- compute `conditions + choose(accesses, 2)` per input and rank class;
- enumerate the bounded rank product; and
- document or mechanize the extension's soundness assumptions.

This stage is not justified for current Native or Native4D.

## 10. Risks and controls

| Risk | Control |
|---|---|
| Proof semantics drift from executable ops | share algebra/index helpers; retain concrete `Map_verify`; add mutations |
| A schema proof is mistaken for an implementation proof | report the two layers separately and require both in the registry |
| Mathematical equality is mistaken for float identity | claim-aware semantics; explicit `Round`; ordered reduction rule |
| Solver timeout or `unknown` becomes success | three-way verdict; only `unsat` proves |
| Vacuous precondition proves anything | separate satisfiability obligation that fails the rule |
| Nonlinear reshape queries become unstable | defer them; use counted/pinned solver budgets and keep `Unproved` honest |
| SMT dependency reaches Node/JS builds | native-only library and external process boundary |
| Rank-bound theorem is applied outside its language | no rank classes initially; later eligibility checked by construction |
| Copied TensorRight code creates license ambiguity | reimplement concepts, or retain Apache-2.0 notices for any copied portion |

## 11. Final recommendation

Build the fixed-frame symbolic-extent checker through the three pilot rules and
stop to evaluate it before extending the language.

That milestone answers the important question cheaply: can mltorch prove the
axis, shape, broadcasting, and reduction identities behind its current passes
for every admitted extent while keeping its concrete, bit-aware transformation
verifier as the implementation check?

If yes, the next useful work is selected fixed-rank reshape and Native4D
legalization schemas. If no, the existing `Map_verify` and mutation-test system
remains intact; the experiment does not require a graph IR or runtime dependency
refactor.
