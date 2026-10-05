# Structured SSA computation IR — implemented design

Status: first slice implemented (typed IR, verifier, printer, reference
interpreter, direct source lowering for pointwise and ordered-sum kernels,
differential harness). The semantics, task list and per-task evidence live in the
SSA design, implementation plan and tracker in the sibling design repository;
this file records what the code in `lib/` actually does and where it deliberately
differs from the proposal.

## Libraries

| Library | Role | Depends on |
|---|---|---|
| `lib/ssa_ir` | Pure IR: ids, types, ops, regions, builder, verifier, printer, interpreter | `core`, `err_trace`, `expr`, `fmt` |
| `lib/ssa_lower` | Direct lowering of a `Fusion_plan.t`; runs a lowered program over bound tensors | `ssa_ir`, `native` |
| `lib/ssa_bridge` | Differential harness against `Kernel_eval` and the Loop interpreter; Loop ↔ SSA converters | `ssa_ir`, `ssa_lower`, `loop_ir`, `native` |

`ssa_ir` owns no tensor, kernel or graph type, so it cannot reach a native process
API or a physical target. Nothing in `loop_ir` or `native` knows SSA exists. A
module in `ssa_lower` may not be named `Ssa_lower`: a wrapped library's module of
its own name becomes its interface and hides its siblings (the lowering is
`Ssa_lower_plan`).

## Representation

- Values are `{id; ty}`; ids are `Core.Tagged_int` types, one per space (value,
  region, buffer, revision), so one cannot be passed for another. A buffer id is
  the tensor id it was lowered from.
- Operations are data (`Ssa_op.t`) with runtime type witnesses; `Ssa_typing` is
  the single typing rule, used by the builder to mint results and by the
  verifier to check declared ones. The builder adds phantom types on top; a
  transformed or hand-built program is checked by the verifier alone.
- Regions are `{id; params; body; yields}`; `Ssa_stmt.t` is parametrized by its
  region type so the region record can be named `t` in its own module.
- **Effects are explicit and linear along a path.** An effectful operation
  (`Index_add`, `Index_scale`, `Load`, `Store`, `Mark`) names the effect it
  consumes (`token`) and returns the next as its last result. `for`,
  `ordered_sum` and `if` each carry exactly one effect in their signature.
  `if` branches both start from the incoming effect (path-sensitive: only one
  runs); a loop body can reach the outer effect only through its own parameter.
  The verifier rejects forks, stale reuse and a lost chain.
- Loads check their coordinate and fail with the first axis outside, in
  `Expr.Axis.all` order. Checked index operations fail at the operation that
  leaves the 32-bit domain, with its operands, even if a later one would return
  to range. A flat access or a store outside its buffer is a defect
  (`Invalid_argument`), never a row.
- A failure ends the invocation (`Err.Escape` in the interpreter). There is no
  failure terminator yet; failure is a dynamic outcome of an operation.

## Verifier and interpreter bounds

`Ssa_verify.max_region_depth` (256) bounds region nesting; the interpreter
recurses once per level and verifies first, so execution depth is bounded
independently of source expression depth (expression trees are flattened into
statements by lowering). The inline suite runs the depth-200 case under node.

## Deviations from the proposal

- Buffers are declared objects named by id in an operation's attributes, not
  first-class `buffer<F>` values.
- Vector, mask and offset types are declared in `Ssa_type` but no operation uses
  them yet.
- The first lowering emits a checked access at every load and a checked
  operation at every index `Add`/`Scale` and does no range reasoning; removing
  checks is the range analysis' job, over a program that is already correct.
  Unsupported source constructs are typed refusals (`Ssa_unsupported`), distinct
  from every runtime failure.

## Testing

`test/ssa_ir` (verifier mutations built from records, interpreter control flow,
printer determinism) and `test/ssa_bridge` (differential against `Kernel_eval`,
logical-work marks against the Loop interpreter) run natively and under node.
A correctness-sensitive rule is considered tested only after reverting it and
watching a test go red.
