# Native4D — implementation plan and domain contract

## Status

Complete. All eight stages landed. This is the executable companion to
`.ai/native4d_design.md`, which holds the goal, the feasibility argument and the
per-operation legalization rationale; this file holds the stage sequence, the
decisions taken, the corrections found while planning, and the domain contract
table the tests pin.

## Decisions

| | |
|---|---|
| **Layout** | New library `lib/native4d/`, dune-default `(wrapped true)`, so modules are `Native4d.Op`, `Native4d.Shape4`, `Native4d.Lower`. |
| **Scope** | All eight stages, through the gated model reports. |
| **RMSNorm** | Native4D retains a fused `Rms_norm` delegating to Native's compute, so the legalization claims `Identical`. |
| **DCE** | Joins the canonical pipeline; the four PT2-gated cram goldens are re-promoted with it. |

`wrapped true` deviates from the repo convention, and the reason is the
convention's own reason. `wrapped false` exists so a library's modules can say
`Vec6`, `Axis`, `Dim` unqualified — and a wrapped `native4d` still can, because
those come from `native`, which stays unwrapped. What `wrapped false` cannot do
is host two dialects: `include_subdirs unqualified` makes every module name
globally visible, so a second `Op`/`Graph`/`Eval_op` would need a prefix per
file. It also avoids touching the two profile-gated `(library (name native))`
stanzas in `lib/native/dune`.

**The naming rule the wrapping forces.** A wrapped library compiles each unit
with its alias module opened, so a sibling shadows the same-named module from an
unwrapped dependency, and `native` is unwrapped. A `lib/native4d/eval_op.ml`
would make Native's `Eval_op` unreachable from *every* file in the library. Five
names collide and take a `4` suffix — `graph_shape4`, `eval_op4`,
`eval_direct4`, `eval_symbolic4`, `dialect4` — and only those. `Op`, `Graph`,
`Shape4`, `Axis4`, `Error`, `Builder`, `Domain`, `Lower`, `Framework` are free.

## Corrections to `.ai/native4d_design.md`

Found by measuring the design against the code. Each lands as a doc edit in the
commit that implements it.

| # | correction | where |
|---|---|---|
| C1 | `Mean keepdim=false` is not always legalizable: `Reduce.Mean.kept_map` packs survivors onto the innermost axes right-aligned, so reducing `H,W` of `[N,H,W,C]` puts `N`'s extent on `D`. Sound only when the packed output shape itself satisfies `T=D=1`, i.e. `N=1`. | §7.5 |
| C2 | §7.8 conflates two rewrites. SSA outputs are all-or-nothing: DCE deletes a node with *no* live outputs but cannot drop one dead edge from a live two-output node. Dropping dead max-pool indices is a separate node replacement. | §7.8 |
| C3 | Do not parameterize the tensor signature by shape type. `Stage_program`, `Expr`, `Symbolic`, `Ground_expr` are all typed on the concrete `Tensor_sig.t`; Native4D reuses it verbatim and `Shape4` guards only op payloads, shape inference and the builder. | §5.1 |
| C4 | Cross-dialect map verification precedes same-dialect transform functorization, not the reverse — the first milestone needs the former and not the latter. | §11 |
| C5 | Output completeness has no implementation behind it, and it belongs in `Graph_map`/`Map_verify`, not in the lowerer: it is a property of any map. | §11 |
| C6 | The conversion result must carry the destination **snapshot**, not a raw graph — the `'dst` brand exists nowhere else, and rebuilding a snapshot mints a fresh one. | §9.1 |
| C7 | The error row is built progressively, and `` `Shape `` must not be a tag. | §10 |
| C8 | "Every output in some non-vacuous cluster" is too weak to mean output equality: it admits crossed outputs and outputs paired with internal tensors. The rule is positional and two-sided. | `native_transform_verify.md` |

## Stage sequence

One commit per stage: code, tests and the `.ai/` delta together.

### Stage 0 — the domain check, without a destination *(done)*

`lib/native4d/{error,domain}.{ml,mli}`. `Domain.check : Graph_view.t -> (unit,
Error.t) Core.result`, reporting the first node outside the dialect, then the
shape rule. Exhaustive over `Graph_ir.op` with **no default arm**, the discipline
`Output_transfer.classify` already holds: a defaulting match would silently admit
the next Native op into a dialect that cannot represent it.

Node predicates run *before* the shape rule. The two overlap — a permutation
moving `C` onto `D` necessarily produces a tensor with extent on `D` — and the
op-level diagnostic is the actionable one.

**Acceptance**: every row of the table below pinned by `[%expect]`; the op match
exhaustive with no default arm.

### Stage 1 — DCE, index-dropping, one canonical pipeline *(done)*

`passes/dce.ml`, `passes/drop_pool_indices.ml`, `transform/pipeline.ml`.

`Dce` is a `Pass.per_node` whose predicate is **global reverse reachability from
`Graph.outputs`**, not local unusedness, and with **no `Pass.fixpoint`**. A local
predicate peels a dead chain one node per sweep, and `Pass.fixpoint`'s
`max_iters = 16` then fails with `` `Not_converged `` on a graph that was
converging — the identical failure `native_transform_design.md` records from the
`reuse_permute` review. It stays a `per_node` rather than a hand-built `Pass.t`
because `verified`, which calls `Map_verify.step` and mints the `Audit`, is
reachable only through the private `of_sweep`; `run_with` does not wrap what a
pass returns, so a custom pass would skip verification silently. `per_node`
collects every match in one sweep and merges the recipes, and deletions allocate
nothing, so one sweep suffices.

Both `fold` branches must be stated and tested — `fold=false` still runs
`Fold_batch_norm`, just without the `Fold_const` rounds either side:

| | sequence |
|---|---|
| common head | `Reshape_to_permute` → relayout fixpoint → `Dce` → `Drop_pool_indices` → `Dce` |
| `fold=true` | … → `Fold_const` fixpoint → `Fold_batch_norm` → `Fold_const` fixpoint |
| `fold=false` | … → `Fold_batch_norm` |

**Acceptance**: dead max-pool indices no longer reach `Domain.check`, and the
stage 0 table row flips visibly; `canonical` is idempotent on both branches; a
≥16-node dead chain is removed in one sweep, observed failing against a
locally-predicated implementation first; `run_reporting ~verify` returns a
non-empty audit naming `dce`, observed failing against a hand-built `Pass.t`.
The two verify crams join `pt2.runtest` — nothing runs them today.

### Stage 2 — `Axis4`, `Shape4`, the closed op variant *(done)*

`Axis4.t = C | H | N | W` is the enforcement §4.3 asks for, and is stronger than
validation: `Mean_keepdims {dims : Axis4.t list}` and `Permute4 {perm :
(Axis4.t * Axis4.t) list}` cannot name `T` or `D` at all.

Op set: `Add`, `Add_scalar`, `Avg_pool2d`, `Clamp`, `Conv2d`,
`Depthwise_conv2d`, `Div`, `Div_scalar`, `Hardtanh`, `Max_pool2d`,
`Mean_keepdims`, `Mul`, `Permute4`, `Relu`, `Reshape4`, `Rms_norm`, `Sqrt`,
`Sub`, `Transposed_conv2d`. **`Discard` is deliberately absent** — no Native4D op
is multi-output, and dead outputs are gone before conversion.

Reuses `Graph_ir.{Node_id, Group_id, Group, Input}`, `Tensor_id` and
`Tensor_sig.t` verbatim (C3); copies the `op_registry` idiom so JSON, `operands`,
`map_operands` and `pp` are one entry per op.

**Acceptance**: no non-4D `Shape4.t` constructible through the public API; every
`output_shape` arm returns `Shape4.t`; JSON round-trips per op.

### Stage 3 — evaluation through Native's compute *(done)*

`eval_op4` translates parameters and calls the *existing* `Compute (S)` functors:
`Conv2d` with `groups=1`, `Depthwise_conv2d` with `groups=in_channels`,
`Transposed_conv2d` through `Conv.Convolution` with `transposed=true`. Weight
layout is unchanged from Native — forward `[Cout,1,1,Kh,Kw,Cin/groups]`, and
`Linear`'s `[Out,1,1,1,1,In]` is already a 1×1 conv weight — so no permutation
is inserted around any shared compute call and no tensor is copied to cross the
boundary.

**Acceptance**: Direct and grounded Symbolic agree bitwise on every op.

### Stage 4 — the minimal shared extraction *(done)*

1. `graph_common.ml` — `'op Node.t`/`'op Graph.t`, plus `Node_id`, `Group_id`,
   `Group`, `Input` and the op-polymorphic `nodes`/`input_kind`, all moved down
   out of `Graph_ir` (they cannot stay there: `Graph_common` needs `Node_id.t`,
   and referring to it through `Graph_ir` closes a cycle). Re-exported
   monomorphically, keeping the record representation visible, so all 76 call
   sites are untouched.
2. `dialect.ml` — `module type S`, including `validate_sig`. Without that hook
   the four-axis invariant leaks: `Graph_view` constrains shapes only where an
   *op* produces them, and a graph input or captured constant is produced by no
   op.
3. `Graph_view.Make (D)` — plus two new steps: every live tensor id has a
   signature (a real hole today: validation checks only that an operand is
   *defined*, and the signature lookup lives inside per-node shape inference, so
   an input used directly as an output is never checked), then `D.validate_sig`
   on each. Error wrapped `` `Graph_shape ``, never `` `Shape `` (C7).
4. `Snapshot.Make (D)`.
5. `dialect4.ml` + `framework.ml` — each functor applied exactly once.
6. `Output_transfer.Make` over an inline reduced signature, not `Dialect.S`,
   which would be a cycle.
7. `side.ml` — the `SIDE` bundle a pair functor takes: `Snapshot`, `Transfer`,
   `symbolic`, `sig_of`, `group_root`, with `type op` shared by both submodules.
   A bare snapshot is not enough: `check_claim_closure` needs the destination
   dialect's transfer table, and `Map_verify` needs each side's symbolic
   evaluator. Native's bundle is its own unit, `native_side.ml`, since
   `Snapshot` is specialized *against* `Native_dialect`.
8. `Graph_map.Make_pair (Src : SIDE) (Dst : SIDE)` — `('src,'dst) t` stays
   **non-generative**, so `compose` works across the dialect boundary with no
   existential packaging.
9. `check_output_correspondence`, positional and two-sided per C8, in **`create`
   only**. Landed as `` `Graph_output_arity ``/`` `Graph_output_mismatch `` —
   `` `Output_arity `` was unavailable, `Graph_view.error` already owning it with
   a different payload. Not in `Map_verify.run`: unlike claim closure, positional
   correspondence *is* preserved by composition, and `identity` satisfies it
   trivially, so a check there could never fire — a vacuous check, which this
   codebase already judges worse than none.

**Acceptance**: every existing `test/native/` test unchanged, differing only in
module qualification, except `graph_view_test.ml` (the `` `Graph_shape `` wrapper)
and `graph_map_test.ml` (item 9's regressions); `version_safety.t`'s eight
negative compilations still fail *for their original reasons*.

### Stage 5 — the partial lowerer *(done)*

Returns the destination snapshot (C6), the map and the constants. Ids: an edge
whose value is preserved keeps its **source raw id**, which is forced rather than
chosen — `Graph_map.create` rejects a src id neither mentioned nor deleted but
absent from dst, so without preservation every edge becomes explicit and the
implicit-identity bulk disappears.

Must **materialize propagated downstream claims** before calling
`Graph_map.create`: one claim per legalized node is not enough, because any
legalization weaker than `Identical` makes every downstream edge weaker too, and
an edge left implicit reads as `Identical`. `Rewrite.apply`'s steps 9–10 are the
template.

Constants: *absence* is a per-legalization question (only batch-norm needs a
payload at conversion time), *validity* is a precondition checked over the whole
supplied map up front, against `Rewrite.origin`'s contract.

**Acceptance**: every produced graph passes Native4D view validation; output
completeness holds; conversion is deterministic in ids, node order and
diagnostics.

**NODE ids follow the same policy as edge ids**, which the plan did not say and
the implementation forced. Allocating destination nodes densely from zero makes
destination node 0 a *different node* from source node 0 the moment anything is
removed — the raw-id collision the design forbids for edges, reappearing for
nodes, and `Graph_map.create` catches it as `Unpaired_src`. So the first
destination node of a source node keeps that node's id and any extra takes a
fresh id above the source watermark.

**Known refinement, not a defect**: a batch-norm fold leaves the original mean
and variance constants declared in the destination graph, unread. They were read
in the source, so the unread-constant omission (which runs before the walk) does
not see them. Harmless — evaluation skips unused constants — but stage 8's
per-op counts will show them.

### Stage 6 — cross-dialect map verification *(done)*

`Map_verify.Make_pair` over the `SIDE` bundles. **The one genuinely uncertain
piece:** the verifier's internal side representation is monomorphic in the
snapshot module (`type 'v side = { …; snapshot : 'v Snapshot.t; … }`), so with two
dialects `('src,'dst) sides` does not typecheck and the rank-2 `resolve` record
must split in two. Fallback if that turns out to be restructuring rather than
generalizing: keep `Map_verify` monomorphic and add *two* explicit
specializations — Native→Native4D and Native4D→Native4D, both, since Stage 7
depends on the same-dialect one — sharing only the grounding and comparison
helpers. Temporary if taken; it duplicates the deepening driver.

**The feared restructuring did not happen.** `resolve` has exactly two users,
and only one of them (`shape_for`) touched `side.snapshot` — so replacing that
field with a signature LOOKUP makes the whole record dialect-free, and the
rank-2 `resolve` survives untouched. A `Tensor_sig.t` is a `Tensor_sig.t` in
either dialect. The version parameter is kept real by holding the edge
universe, which nothing reads; that is documented at the field.

`Group_path` also stayed outside the functor: it only touches shared structure,
so `index` and `producers` went op-polymorphic instead, with `producers` taking
the edge lookup rather than a snapshot.

**Result: cross-dialect verification works.** `relu`, `add`+`relu`, clone
removal, a lone permutation and `Mean keepdim=false` all prove *structurally* —
holding for every input, not one sample. That is the third architectural claim
§14 asks the milestone to prove.

**A verifier defect, found by the `Bmm` legalization and fixed separately.** The
output cluster refuted with an exhaustive counterexample while a numeric
cross-check showed both graphs computing the same values. The cause was in
`value_tiers`, not in the lowering: it handed the probe the *raw* projected
terms while giving the coefficient tier the *normalised* ones, so a `Round` that
`normalise` had legally collapsed was still in the term the probe evaluated —
and `Valuation.draw` assigns arbitrary doubles, so for an f32-stored cell it
drew a value f32 cannot hold and separated `f32(v)` from `v`.

`Bmm` is the first rewrite in the tree where one side reads an operand directly
and the other reads it through a *materialized stage* (a permuted convolution
weight), which is why nothing had exposed it before. Fixed by probing the
normalised terms; the ungated suite is unchanged and both gated ResNet-18 verify
crams pass byte-identical.

Residual: `Bmm` reports `tested (coefficients agree)` rather than `proved`,
because structural equality still fails on association — the source sums
`((0 + a) + b) + c` and the convolution `((((0+(0+(0+a))) + …) + 0)`. Those are
the same float sequence, `0 + x` being exact, but `normalise` does not simplify
additive identities. Teaching it to would upgrade this to a proof and is a
separate change to proof semantics.

Seven mutations, one per legalization family, each observed red first. Batch-norm
precomputation is the only `Equivalent` family and so cannot be `Refuted` —
`probe` formally refutes only `Identical`. Its test pins the
`Agrees → Disagrees` transition, with the correct lowering driven to `Agrees` on
a small fixture; "`Disagrees` or `Unproved`" would be a mutation test that cannot
go red, since `Unproved` is also what a correct lowering returns when the budget
stops the frontier settling.

### Stage 7 — Native4D transforms through shared functors *(done)*

`Pattern.Make`, `Region.Make`, `Recipe.Make`, `Rewrite.Make`, `Pass.Make`, Native
passes as specializations, one Native4D pass to prove the abstraction. Carries
nothing else: `rewrite.ml` holds the version-safety discipline.

Much lighter than the line counts suggest. `rewrite.ml` is 913 lines but its ten
dialect references are four distinct functions, and `input_kind` was already
op-polymorphic from stage 4. Pattern, Region and Recipe have one apiece.

**Two things the plan did not anticipate, both forced by the module system:**

`Recipe` and `Rewrite` take a **`Side.S`, not a `Dialect.S`** — `Rewrite.apply`
both rebuilds a graph (needing the operation table) and builds a map (needing
the snapshot and transfer table), and two separate arguments could disagree
about `op`. `Side.S` therefore gained a `Dialect` submodule, and declares its
`Snapshot` as `Snapshot.Make (Dialect)` rather than as something merely
signature-compatible: `Recipe` names that application too, and an abstract
snapshot would not be known equal to it.

**Do not alias the dialect inside a functor.** Writing `module D = S.Dialect`
and then `Graph_view.Make (D)` breaks applicative path equality against
`Graph_view.Make (S.Dialect)` written elsewhere — the two stop being the same
type. Use `S.Dialect` directly. This cost most of the debugging time and the
symptom (`Graph_view.Make(S.Dialect).t` vs `View.t`) does not name the cause.

`Id_supply.of_graph` and `Group_path.index`/`producers` went op-polymorphic
rather than into a functor: watermarks and group trees are about ids and shared
structure, neither of which is dialect-specific.

**Acceptance met**: `Trim_permute4` — a Native4D pass written against the same
`Recipe`/`Pattern`/`Pass` the nine Native passes use, its only dialect-specific
line being the op projector — trims an identity `Permute4`, produces the
`{t0, t1} -> {t0}` cluster §3 describes for a trim, and reports
`audit trim_permute4: 2 clusters: 2 proved (structural)` under
`Require_proved`. A Native4D rewrite is not merely expressible in the shared
framework; it is held to the same bar.

### Stage 8 — representative model gates *(done)*

`native_graph to4d`, reporting per model: conversion success or the first
unsupported operation, per-op counts, and the map verification summary. Gated
cram over resnet18 and mobilenet-v2, in `pt2.runtest`.

Cost is deliberately **not** in the golden: a timing in a cram is a flaky test,
not a measurement. Correctness and cost were to be reported separately, and the
separation here is that cost is not reported at all until there is something to
compare it against.

**Both models convert.**

| | canonical Native | Native4D | notable |
|---|---|---|---|
| ResNet-18 | 49 nodes | 49 nodes | 21 `Conv2D` — twenty convolutions plus the `Linear`, legalized params-only; one `MeanKeepDims` for the global average pool |
| MobileNet-v2 | 101 nodes | 100 nodes | 17 `DepthwiseConv2D`, which is why this model is here: grouping is a *constructor*, so the classification has to be right for the graph to convert at all |

**The cram pins the whole destination graph, with each edge's verdict beside
it** — counts say what a graph is made of, only the structure says how it is
wired, and only a per-edge verdict says which parts of the conversion are
justified. `Native4d.Graph.pp_with ~annot` takes the annotation the way
`Graph_ir.pp_with` takes its printer: the IR carries no verdicts, so anything a
reader wants beside an id has to arrive from outside.

**No refutations on either model.** ResNet-18's map is 92 clusters: 23 proved
structurally, 28 proved for these constants, 1 coefficient-agreed, 40 declined
as `too large`. A decline is the budget refusing an activation tensor before
doing any work, not a doubt about it.

That absence is asserted explicitly rather than left to be noticed in a
thousand-line dump — scanning is not checking.

## The domain contract

What `Domain.check` answers, pinned by `test/native4d/domain_test.ml`. Rows marked
† reject only because nothing has rewritten the op yet, and flip when stage 1's
pipeline runs ahead of the check.

| graph | outcome |
|---|---|
| every live tensor `T=D=1` | in the dialect |
| non-unit `T` / non-unit `D` | `Non_four_dimensional_tensor` |
| unread non-4D constant | in the dialect — the lowerer omits it |
| read non-4D constant | `Non_four_dimensional_tensor` |
| unused non-4D user input | `Non_four_dimensional_tensor` |
| `Mean`/`Rms_norm` over `C` | in the dialect |
| `Mean`/`Rms_norm` over `D` | `Axis_outside_dialect` |
| `Permute` fixing `T`/`D` | in the dialect |
| `Permute` moving `C` onto `D` | `Axis_outside_dialect` |
| `Batch_norm` on `C`, constant params | in the dialect |
| `Batch_norm` on `H` | `Axis_outside_dialect` |
| `Batch_norm` with a dynamic parameter | `Dynamic_batch_norm` |
| `Batch_norm` whose parameter is shorter than the normalized axis | `Batch_norm_extent` |
| `Mean keepdim=false`, `N=1` | in the dialect |
| `Mean keepdim=false`, `N=2` | `Non_four_dimensional_tensor` (C1) |
| conv `groups=1` / depthwise | in the dialect |
| conv `groups=2`, 2 per group | `Unsupported_grouped_conv` |
| transposed conv `groups=1` | in the dialect |
| transposed conv `groups=2` | `Unsupported_grouped_transposed_conv` |
| `Bmm` batch 1 / batch 2 | in the dialect / `Unsupported_bmm_batch` |
| `Bmm` whose `mat2` is not f32-exact (I8/I16/I32/I64) | `Lossy_bmm_operand` |
| max-pool indices discarded | `Unsupported_op` † |
| max-pool indices live | `Live_max_pool_indices` |
| `Linear` | in the dialect |

**Two rows are about neither the four-axis frame nor an op's own output shape**,
and both were added after a review found the legalization unsound without them.

`Lossy_bmm_operand` is not about shape at all. The legalization *materializes*
the permuted `mat2`, where Native's `Bmm` reads it directly, and every op output
in this engine is f32 — so an I64 operand holding 2^24+1 is silently rounded
while the map still claims `Identical`.

`Batch_norm_extent` *is* about shape, just not about a shape anything else
checks: `Norm.BatchNorm.output_shape` is a function of the input alone, so a
parameter's extent is never compared against the normalized axis and
`Graph_view` has nothing to validate. Native's own evaluation does not survive
such a graph either — `BatchNorm.Compute` reads each parameter at the *output's*
channel index and `Tensor.read` is strict — so this check does not make Native4D
stricter than Native; it makes the failure arrive as a typed error rather than
as an exception during evaluation.

The general shape: a legalization is sound only under a precondition, and the
precondition belongs in the domain check whether it concerns an operand's
*format*, an operand's *extent*, or the frame — the common thread is that none
of them is implied by the op's own output shape, which is all shape inference
looks at.

Two more properties of this table are worth stating, because both were nearly
got wrong:

- **`Mean keepdim=false` needs no special case in the node predicates.** The
  packed output shape *is* the reachable output tensor, so the shape rule catches
  the `N=2` case on its own. C1 is enforced structurally.
- **The accepting rows are not vacuous.** "Unread non-4D constant" accepts and
  "read non-4D constant" rejects, which is what distinguishes a working
  reachability rule from one that skips every constant.
