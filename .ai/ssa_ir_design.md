# Structured SSA computation IR — implemented design

Status: the source surface the op sweep walks is implemented (typed IR,
verifier, printer, reference interpreter, direct source lowering including
Region programs, locals, scans and groups, differential harness), with the
analyses and the exact optimizations over it. No vector form, CFG form or direct
emitter exists yet. The semantics, task list and per-task evidence live in the
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

## Regions, locals and scans

- **Locals are scratch objects.** `local.alloc` makes a fresh object of binary64
  cells, every cell unset; `local.write` and `local.read` address it by index. A
  read of an unset cell is a defect, never an undefined value. A Region program
  allocates each local inside its key's region, so what one key wrote is never
  visible to the next by construction; reusing an object across keys is a later,
  proved optimization. A read outside an object fails as the reference's unbound
  local when the object names a variable, and is a defect when it does not.
  `check_local` is the range check of a vector read or a scan's `prev`.
- **A trace** is written row 0 from the initializer and rows 1..steps from the
  update, each lane charged against the meter before its update runs; its cached
  read checks the row, then the lane, before either is used (`check_scan`), so
  the row wins a simultaneous failure. An **inline scan** reserves `2 * width`
  live state, fills two rolling rows, charges per lane update, copies the next
  row back, reads the lane and releases the state.
- **The meter** (`meter.reset|charge|reserve|release`) is invocation state with
  the program's scan limits. A Region key starts a fresh meter, and so does a
  pixel cell that reads it: whether a key or a cell reads the meter at all is
  found by lowering it once into a block that is thrown away
  (`Ssa_builder.probe`), because a reset the emitters then declare meter state
  for, and never use, does not compile.
- A group lowers to one nest over the canonical key: the shared locals once per
  key, then each member's emitter over its own Whole axes at its own physical
  key, each with its own conversion and store.

## Analyses and passes

- `Ssa_uses` records definitions, uses and the region tree of one revision and
  answers lexical dominance; it is bound to its revision. `Ssa_range` derives
  integer ranges (a missing fact is the whole domain, `Empty` is an unreachable
  definition; a checked operation's range is its wide range clipped to the
  domain, because an execution that left it failed first) and the claims they
  back: a sum or product stays in the domain, an access stays in its buffer, a
  loop runs a known number of times. `Ssa_effects` summarizes what a statement
  reads, writes, may fail on and counts, and states which buffers may overlap as
  a policy a caller chooses: `Conservative` (any pair may), or `Distinct_buffers`
  for a caller that has established it (a runner that allocates each buffer).
  No pass picks one.
- **Proofs are re-derived, never trusted.** An operation that carries a proof
  (`index.add_in_domain`, `index.scale_in_domain`, `load.in_bounds`) is accepted by
  the verifier only where `Ssa_range` re-derives the claim from the program; a
  second phase of `Ssa_verify.check` does it after the structural walk. This
  replaces a private proof constructor: the proof object would only have named a
  claim the verifier then had to check anyway.
- `Ssa_rewrite` rebuilds a program with a rule applied post-order and returns the
  next revision; a pass states what happens to one statement. `Ssa_scalar` is the
  single definition of the pure scalar operations: the interpreter and the
  constant folder both evaluate through it.
- **Relational bounds** (`Ssa_range`). A window loop's taps run over
  `[max (0, a), min (K, b))` and an access inside it reads `base + k`: inside
  the buffer only relationally, which intervals over `base` and `k` cannot show.
  An induction value keeps the facts its bounds give (`d * k >= a`,
  `d * k <= a`, from `clamp_low`, `max`, `min` and the ceiling and floor
  divisions of a dilation) and a coordinate's linear form, built through the
  additions, scalings and constants that define it, is proved by replacing
  `c * k` with the matching fact's side (depth two) and checking the rest by
  interval. The loop's own bounds are the only premise, and the verifier
  re-derives a proof through the same function. A vector access's coordinate the
  lanes do not move along takes the same proof.
- Passes (`Ssa_opt`): **simplify** (fold, pure CSE, dead pure values) never deletes
  a checked operation, a load, a mark or a meter operation for being unused and
  applies no algebraic identity; **guards** turns a checked operation the ranges
  prove cannot fail into its in-domain form (pure, its effect reconnected), a
  load proved in bounds into the unchecked form, deletes a check that cannot
  fire, removes a loop that cannot run and inlines one that runs once; **hoist**
  moves pure total operations out of any loop and a load only when it is in
  bounds, the loop runs, and nothing in it may write what it reads; **share**
  replaces a repeated load by the earlier one until something may write its
  buffer, and discards what a loop body may overwrite on entering it;
  **convert_ifs** (`Ssa_opt_if`, planner pipelines only, after guards) turns a
  branch whose arms hold only pure instructions and loads proved in bounds into a
  select over the arms' values: both arms run, which cannot fail or count, the
  then arm's loads precede the else arm's on the one effect chain, and an arm with
  a checked operation, mark, store or loop keeps the branch. A lowered clamp or
  guarded load is a branch, and the vectorizer holds no branch in a loop body.
  The driver verifies every revision a pass returns.
- **Independent-output blocking** (`Ssa_opt_block`, in the pipeline after hoist
  and before share). A loop with constant trips whose iterations are separate
  outputs around one ordered sum becomes full groups of G outputs (one reduction
  loop with an accumulator per output, each adding its own terms in the original
  order) plus the original loop over the remainder. Legality is checked and a
  refusal names the failing condition: the body cannot fail (so reordering
  outputs is unobservable), touches neither the meter nor locals, reads nothing
  it may write, the reduction's bounds do not vary with the output, and every
  store's coordinates are the iteration index or independent of it. Values and
  logical work are unchanged by construction and checked against the unblocked
  program; the loads the group repeats are merged by share. G is chosen from 8,
  4 and 2 by the trip count and a sixteen-value register estimate, and only if
  some read is shared across outputs; `Fixed` forces a size for the tests.
  `Ssa_clone` copies statements with fresh definitions and regions. A strided
  loop projects to the Loop IR as a trip counter with the induction value
  reconstructed from it.
- **Vectors** (`Vec`, `Mask`, `Ssa_type.Lanes`; at most 64 logical lanes, a width
  and never a register). A vector holds binary32 or binary64 lanes, a mask holds
  predicates; no int64 vector exists. Every pure float or predicate operation of
  the scalar surface lifts through one constructor, `Lanewise`, whose typing
  scalarizes the operands and applies the scalar rule, and whose interpreter
  evaluates each lane through the scalar function: lane semantics cannot drift
  from scalar semantics. `Vec_splat`, `Vec_iota` (lane `k` is the binary64 value
  of `base + k * step`, in int64), `Vec_extract` and `Vec_insert` are the only
  other pure vector operations. `Vec_load`/`Vec_store` address lane `k` at
  `at + k * steps` per axis, read or write every lane, and are accepted only where
  the range analysis proves every lane inside the buffer, so a mask can never
  hide a lane that would have failed: there is no masked memory operation.
  `Mark_lanes` counts what a vector iteration stands for, and an `Ordered_sum`
  or a loop may carry a vector, each lane its own left fold. `Ssa_vec_expand`
  rewrites a vector program into scalar lanes from the program alone (a memory
  operation becomes one access per lane with the proofs the vector one carried),
  which is the independent reference every vector result is checked against. The
  Loop converter refuses a vector operation: expand first.
- **Numerical policy and precision.** `Ssa_numerics` restates the three presets
  (`Reference_f64`, `Simd_fp32_ordered`, `Simd_fp32_relaxed`), their permissions
  and identities, the backends that accept them, and the admission rule (a
  dequantizing read has no binary32 decode), because this library sees no Loop
  type; the bridge suite checks every name, identity, target price and binary32
  helper against the Loop planner's. `Ssa_precision.to_f32` makes binary32
  explicit before vectorization: a read is narrowed once, a constant rounded
  once, an int64 or index converted in one rounding, a store or checked
  conversion widens first (exact), a scratch cell widens on write and narrows on
  read. Binary32 operations round once in the interpreter, `Float_unary` on a
  binary32 operand is the binary64 function rounded once (`erf` is `erf32`) and
  `Float_fma` is `fmaf`. The binary32 sweep (every walked plan against the Loop
  interpreter at binary32) agrees bitwise.
- **Targets.** `Ssa_target` is the Loop target description as data: legality
  (native or expanded per operation) kept apart from cost, per precision, with
  the logical width a `Lanes` value; `forced` zeroes every price so a test takes
  every legal loop.
- **Vectorization** (`Ssa_vectorize`, in the pipeline after hoisting and before
  scalar blocking when a target is given). A loop whose iterations are
  independent outputs, with the loops inside it, becomes full vector iterations
  plus the original loop as the remainder. Every value the body defines is
  classified uniform, affine in the induction value with a literal stride, or a
  vector; an access is a base coordinate and a step per axis; a carried value
  becomes a vector when anything feeding it is one, found by a fixed point.
  Refused, each with its reason named: a branch, an operation that can fail, a
  scratch or meter operation, a varying int64, a non-affine index, a varying
  inner bound, a store every lane would write, a loop that carries values, a
  buffer the loop writes and reads at different coordinates, an unproved
  overlap of buffers (unless the caller states them distinct), too few trips,
  and a body the target prices out. The vector program is checked against the
  reference, the scalar program and its scalar-lane expansion, and the sweep
  runs it for the cost model's target and for every legal loop.
- **Scheduled sums** (`Ssa_vector_sum`, relaxed policy only). A sum no enclosing
  loop's lanes took is rewritten along its own axis. The schedule is a
  definition, not an accident of the code: the terms split into `parts` lanes of
  `rounds` terms each, plus `extra` leftover terms; each lane folds its terms in
  order, the lanes combine by adjacent-pair trees, and the result is the seed plus
  (horizontal total plus the tail folded in order). The oracle evaluates exactly
  this definition scalar by scalar, so a regrouping error shows as a bitwise
  mismatch rather than a tolerance miss.
- **Contraction** (`Ssa_opt_contract`, relaxed policy, targets with a fused
  operation). `a + x * y` becomes one `Float_fma` with the product on the right
  taken first; a scalar flag selects scalar contraction for targets whose fused
  form is only scalar. `Float_fma` is a single rounding in the interpreter
  (`fma32` at binary32, `Float.fma` at binary64); `?fused:false` gives the
  unfused reading (round the product, then the sum) for a target whose
  multiply-add may or may not fuse, so `wasm128_relaxed` is accepted only where
  the result matches under the fused reading. `Float_fma` has no Loop form: the
  Loop converter refuses it, and a planned program runs through `Ssa_exec`.
- **The planner** (`Ssa_plan.resolve ?target ?alias ~numerics`). Binary32 is
  chosen only when the target vectorizes at least one loop or schedules a sum
  and admission passes; otherwise the plan stays binary64 and says why
  (`refusal`). Contraction runs before sum scheduling, otherwise it would fuse
  the scheduler's own accumulate and break the schedule definition. The plan
  records the precision, the program, the vectorized and scheduled counts and the
  contracted count, and `Ssa_plan.oracle` is the independent evaluation.
  `Ssa_check.run_planned` compares the plan with its own oracle bitwise, then,
  for an ordered policy, with the Loop binary32 interpreter, and for a relaxed
  one with the binary64 reference within 1e-4.
- **Row blocking** (`Ssa_opt_rows`, after vectorization, factor `Ssa_target.row_block`).
  A loop over rows around a vector loop that holds one ordered sum and its stores
  runs a block of rows per iteration: the rows' sums share one loop with an
  accumulator per row (the jam of `Ssa_opt_block`), so a load the rows share is
  issued once and their dependent add chains interleave. Each output cell does
  the same operations in the same order, so the result is bitwise the unblocked
  program's under every policy; the rows left over run as the original loop.
  Legality is the independent-output test on the row index: the vector loop
  cannot fail, touches no meter or scratch, reads nothing it may write, its
  and the sum's bounds do not vary with the row, every store's coordinates are
  the row index itself or independent of it. A vector loop with a remainder
  loop beside it, and a scalar column loop (the column blocking's), are left
  alone.
- Evidence beyond mutations: the op sweep through the optimizer has no
  disagreement, no change of logical work and no extra read; hand-built failing
  programs keep exactly the checks that report their failure; the optimized
  programs run through C, Wasm and JavaScript.

## Control-flow graph form

`Ssa_cfg_lower.program` turns a verified structured program into `Ssa_cfg.t`:
blocks (`Ssa_cfg_block.t`) with typed parameters (the only phi convention),
straight-line operations and one terminator (`Branch`, `Jump`, `Return`), and
edges (`Ssa_cfg_edge.t`) whose arguments bind the target's parameters
simultaneously. A `for` is a header (parameters: induction value and carried
values, the effect among them) that compares against the bound and branches to
the body or to an exit whose parameters are the loop's results; the body ends in
`index.add_in_domain iv, step` and the back edge. An `if` branches to two blocks
that meet in a join whose parameters are its results. An ordered sum is a loop
whose accumulator is a header parameter and whose terms are added on the back
edge (lane-wise for a vector), the effect threaded through the iterations. A
captured value is a dominating definition; no critical edge is made.

`Ssa_cfg_verify` checks reachability, one definition per value, dominance of
every use (`Ssa_cfg.immediate_dominators`), edge arguments against parameters,
predicate branch conditions, operation typing and buffer rules, and the effect
chain: a block with an effect parameter starts the chain there, one without
needs exactly one predecessor and continues its chain, an edge passes the chain
to the target's effect parameter and `Return` consumes it. Proof-carrying
operations are claims verified in the structured program; the lowering re-checks
the one it adds (the last increment stays in the index domain) and refuses
otherwise. `Ssa_cfg_interp` owns only control flow and runs every operation
through `Ssa_interp.Machine.exec`, so a disagreement with the structured
interpreter is a disagreement about control flow; `Ssa_exec.run ~engine:Cfg`
and `Ssa_check.run ~engine:Cfg` run whole plans that way (the graph sweeps run
every walked plan lowered, optimized and vectorized).

`Ssa_cfg_handoff` is the boundary to the native plan: values are machine data
(`Data ty`) or erased effects, precision and access layout are already explicit
in types, operations and accesses, and no register, spill or frame appears.
`copies` gives an edge's parallel copy without effects or self moves and
`sequentialize` orders it with one temporary per cycle.

## Direct C consumer

`Ssa_c` (library `lib/ssa_c`) emits C straight from a structured program, with no
Loop IR between: one `static int` function with `Loop_c`'s shape (error record,
scratch `double *local`, one typed pointer per buffer the program names), the
same runtime helpers and failure-record ABI (`Loop_c_runtime`, `Loop_js_failure`)
so every existing host runs it, and a failure-site table the record decoder
reads. Every value is a C variable declared at function scope and assigned where
it is defined, so a carried value or an `if` result is a plain assignment and an
iteration's transfer goes through temporaries unless a parameter is yielded to
itself. Types are the program's: binary32 values are `float` (the translation
unit asserts `FLT_EVAL_METHOD == 0`), `Float_fma` is `fma`/`fmaf`, vectors are
GCC/Clang generic vectors of the logical width (a power of two) with masks as
64-bit-lane vectors, a contiguous full-lane access is one `memcpy` and any other
stride goes lane by lane, and the operations C has no vector form for run lane by
lane through the scalar helper. Checked operations are explicit `if` + record at
their own site, with the operands the interpreter reports. Precision and
contraction are never decided here. Quantized buffers are a typed refusal; a
buffer no operation touches is no argument (the caller validates its binding).
Checks against the reference: the op sweep lowered and optimized, plans for every
policy on neon (against the structured interpreter on the same program),
hand-built control flow and every failure row, the vector surface, and mutations
of addressing, checks, transfers, masks and fused operations.

## Direct JavaScript consumer

`Ssa_js` (library `lib/ssa_js`) builds a `Js_ast.Program.t` straight from a
structured program, with the argument convention (one typed array per buffer the
program names) and the failure-record shape of `Loop_js`, so the same in-process
host runs it (`Loop_js_exec.compile_kernel` takes the failure-site table the
emitter made). Representations: an index is a `Number`, an int64 a `BigInt`
wrapped with `asIntN`, a float a `Number` (binary32 results rounded with
`Math.fround`), a predicate a boolean; every value is a `let` at function scope.
A loop runs a trip counter, `iv = lo + k * step`, so any step and the empty range
need no special case, and an index that came from `ceil`/`floor` is normalised
(`+ 0`) before it becomes a float because it can be `-0`. Checked operations
return their record at their own site. Refused, typed, not emitted: vectors and
masks, fused multiply-add, binary32 `erf` and an int64 to binary32 conversion
(each would lose its single rounding in a `Number`). Checks: the op sweep lowered
and optimized under node, hand-built control flow, every failure row, int64
wrap, `-0`, binary32 rounding and a stride that does not divide its range, and
mutations of each.

## Direct WebAssembly consumer

`Ssa_wasm` (library `lib/ssa_wasm`) emits one exported `loop_kernel` over linear
memory straight from a structured program, producing the same `Loop_wasm.kernel`
and `Loop_wasm.t` records as the Loop emitter (`Loop_wasm.lower_with` takes any
kernel producer), so the module layout, helper functions, `Math` imports,
manifest and failure-record ABI are one definition and `Loop_wasm_exec.exec_module`
runs it under node. A value is a local typed by the program's own types (index
`i32`, int64 `i64`, binary32 `f32`, predicate `i32`) and a vector or mask is a
group of `v128` registers, two `f64` lanes or four `f32` lanes each; a mask is
held as `i64x2` or `i32x4` lanes by the comparison that made it and re-held lane
by lane where it meets the other shape. A loop's carried values transfer through
the operand stack (every yield pushed, then popped in reverse), so the rebinding
is simultaneous with no temporary; an induction counter that could wrap an `i32`
(its last increment leaves the index domain) is a typed refusal. Contiguous
`f32` to `f64` vector loads are `v128.load64_zero` plus a promote, `f64` to `f32`
stores a demote plus a 64-bit lane store; any other stride goes lane by lane.
`Float_fma` is the one relaxed-SIMD `f32x4.relaxed_madd`, only where the plan was
made for relaxed SIMD; a scalar fused operation and a binary64 one are refused.
Checks: the op sweep lowered and optimized, planned programs for strict vectors,
ordered binary32 (standard SIMD) and relaxed binary32 (relaxed SIMD) against the
structured interpreter, hand-built control flow and every failure row, the vector
surface, and mutations of addressing, checks, transfers, selects, lanes and
accumulation.

## Whole-model bundles through SSA

The three bundle builders take an optional `kernel` hook that makes each
invocation's kernel instead of the Loop emitter: `Loop_bundle_c.build ?kernel`,
`Loop_bundle_wasm.build ?kernel`, `Loop_bundle_js.build ?kernel`, threaded through
`C_host.prepare`, `Wasm_host.prepare` and `Loop_bundle_exec.prepare`. Each
`Loop_bundle.invocation` carries `placed`, the placed kernel its program was
lowered from, so a consumer can lower plans itself; the storage plan, the payload
and workspace layout, the schedule, the argument convention (every program buffer,
positionally, bound through `edges`) and the failure-record ABI stay the bundle's.
`Ssa_backends` (library `lib/ssa_backends`) supplies the producers: it lowers
`placed`, runs a `Pipeline` (`Representation`, the exact passes only, or the
policy planner for a numerical policy and target) and emits with the invocation's
own buffer list. Passes take one invocation's buffers as distinct memory (the storage plan
allocates every output before releasing any operand, and the arena checker rejects an
overlap), the guarantee `Distinct_buffers` asks a caller to have established; a kernel's
own types and helper functions travel as a `Loop_c.t` `prelude`, guarded blocks that
the bundle emits once. A record names its site by the index of an entry of the Loop program's own
failure-site table (`Loop_failure.same_site`: the same kind of failure at the same
local variable), which is how the bundle hosts decode a failure from an SSA kernel
unchanged. A kernel the SSA path cannot make is `Kernel_refused`, never a silent
fallback; the unoptimized pipeline refuses an invocation whose lowering still
names a fusion scratch buffer the invocation does not have, which the exact passes
remove. A kernel's precision is its pipeline's (the plan's, for a planned one), not
the types its text uses: the strict pipelines hold binary32 buffers' values in
binary32 and compute in binary64, and report no binary32 kernel. Checks: the chain and a convolution, batch norm and relu bundle through
each of C, Wasm (node) and JavaScript (node) are bitwise the reference for the
representation, exact and strict-planned pipelines, and within 1e-4 under the
ordered and relaxed binary32 policies. On the model cohort (mobilenetv2_050,
regnetx_002, efficientnet_b0, fastvit_sa12, mobilenetv3_small_050, test_convnext2,
csatv2) every model runs bitwise against the reference through the exact and
strict-planned pipelines and within the frozen tolerance through the planned
performance policy. The performance policy is not yet as fast as the Loop path:
the planner vectorizes fewer kernels (mobilenetv2_050: 178 of 415 invocations in
binary32 against 205) and the run is 1.0 to 1.2x the Loop time (csatv2 equal),
because the Loop IR collapses dense nests that the SSA vectorizer meets nested;
`test/ssa_c/bench` measures generation,
compile, first and warm costs and source size per pipeline beside the Loop path.

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
