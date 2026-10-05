# Structured SSA computation IR — implemented design

Status: scalar surface implemented (typed IR, verifier, printer, reference
interpreter, direct source lowering of every pixel kernel the sweep walks except
region programs, differential harness). The semantics, task list and per-task evidence live in the
SSA design, implementation plan and tracker in the sibling design repository;
this file records what the code in `lib/` actually does and where it deliberately
differs from the proposal.

## Libraries

| Library | Role | Depends on |
|---|---|---|
| `lib/ssa_ir` | Pure IR: ids, types, ops, regions, builder, verifier, printer, interpreter | `core`, `err_trace`, `expr`, `fmt` |
| `lib/ssa_lower` | Direct lowering of a `Fusion_plan.t`; runs a lowered program over bound tensors | `ssa_ir`, `native` |
| `lib/ssa_bridge` | Differential harness against `Kernel_eval`; `Ssa_of_loop` and `Loop_of_ssa` converters | `ssa_ir`, `ssa_lower`, `loop_ir`, `native` |

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

## Scalar semantics

- **Index.** `Add` and `Scale` are checked and effectful; `min`, `max`,
  `clamp_low`, `floor_div` and `ceil_div` (positive literal divisor, a
  mathematical floor and ceiling for a negative numerator) are total and pure.
  A literal outside the 32-bit domain or a non-positive divisor is a typed
  refusal at lowering, never narrowed.
- **int64.** `add`, `sub` and `mul` are modular. `div` is checked (a zero
  divisor fails first, then `min_int / -1`) and `float_to_i64` is checked (NaN,
  an infinity or out of range). `i64_to_f64` rounds once at binary64 and
  `i64_to_f32` once at binary32: the integer's top 24 bits and a sticky bit, not
  a conversion through binary64 (checked against an integer-only oracle).
  Narrowing an int64 to an index is a defect outside the domain; a gather checks
  its raw index in the int64 domain before it normalizes and narrows.
- **Selection.** A source `Select` lowers to an `if`, so only the selected arm
  loads, fails or loops; the `select` operation only chooses between values
  already computed and exists for reductions and for passes that have proved an
  arm total. `pred.or` is not a short circuit.
- **Maxima.** A float `max` is `Expr.Max_op`'s `Float_max`; argmax advances value
  and index together under `pool_better`, whose ties keep the incumbent and whose
  NaN re-triggers (so the last NaN wins). The max-pool intrinsic is the
  rows-then-columns double loop over its clipped window, written with the same
  operations.
- **Formats.** A buffer is one of bf16, bool, f16, f32, f64, i16, i32, i64 or i8,
  the quantized two carrying `scale * (q - zero_point)` per tensor or per channel
  along C. A load names its decode; a flat offset cannot name a channel, so the
  verifier rejects a flat access on a per-channel buffer, and per-channel
  parameters must cover exactly the C extent. A `Filled` input is a scratch
  buffer that is never written: its read checks the coordinate against the shape
  and folds to the value a materialized fill decodes to.

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
- A gather, a division and a float-to-int conversion have no separate guard
  operation: the checked operation is the check. The Loop converter turns each
  Loop guard into the same check at its own site and relies on the later
  checked operation being unable to fail again.
- The first lowering emits a checked access at every load and a checked
  operation at every index `Add`/`Scale` and does no range reasoning; removing
  checks is the range analysis' job, over a program that is already correct.
  Unsupported source constructs are typed refusals (`Ssa_unsupported`), distinct
  from every runtime failure.

## Loop converters

Both are comparison instruments, never the permanent frontend or consumer.

- `Ssa_of_loop` turns mutable temporaries into values: a temporary assigned in a
  loop body and live before it becomes an iteration argument, one assigned in
  both arms of an `if` becomes its result, and a structured sum becomes an
  `ordered_sum`. A `Fail_if` is converted when it is one the Loop lowering writes
  (an index that may leave the domain, a coordinate outside its buffer, a gather,
  a division, a float-to-int conversion); any other guard is refused rather than
  dropped.
- `Loop_of_ssa` gives every value a Loop temporary, so an SSA program runs
  through the existing interpreter and the C, Wasm and JavaScript emitters. A
  loop's yields are all snapshotted before any parameter is overwritten; a
  checked index operation becomes the `Fail_if` of its own overflow at the same
  site; a coordinate load is preceded by the bounds check the SSA load performs.
  It has more locals and copies than a direct emitter would produce, which is why
  a representation-only measurement must not be read as a consumer result.

## Testing

`test/ssa_ir` (verifier mutations built from records, interpreter control flow,
printer determinism) and `test/ssa_bridge` (differential against `Kernel_eval`,
logical-work marks against the Loop interpreter, the two converters) run
natively and under node. `test/ssa_projection` runs one shared table of cases
(the source-direct and the bridge program, each projected to the Loop IR) through
generated C (native, part of `runtest`), the direct Wasm emitter (under node,
`make wasm.runtest`) and generated JavaScript (in process under node,
`make jsoo.inline-runtest`), against the reference.
A correctness-sensitive rule is considered tested only after reverting it and
watching a test go red.
