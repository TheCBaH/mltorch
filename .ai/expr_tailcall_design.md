# Expression recursion on JavaScript backends

## Result

The stack-depth concern is confirmed, with one important correction: it is not
mutual recursion by itself that limits js_of_ocaml.  js_of_ocaml 6.4.1 safely
trampolines statically known **mutual tail calls**.  Expression traversal still
overflows because most recursive child calls are **not in tail position**. The
proposed function-tag variant does fix mutually recursive tail transitions; it
does not by itself change the non-tail child calls.

Here, “constant space” must mean a constant **host call stack**. An arbitrary
immutable expression tree still needs somewhere to retain pending operands and
operators: the direct evaluator keeps that O(depth) state on the call stack,
while an iterative evaluator keeps it in explicit frames or an operand stack.
For a purely tail-recursive mutual group, however, the tag-and-arguments form
below needs only constant-size loop state and no per-transition allocation.

The source-tree oracle was checked first. `Expr.Eval.value` has non-tail
`Binary`, `Unary`/`Round_f32`, and `go -> guard -> go` paths; the existing
`depth_probe` JS suite passes at the configured 1536 limit. Temporary oracle
runs of all three shapes at depth 200,000 exhausted the stack when the real
evaluator was compiled by js_of_ocaml and when the same evaluator sources were
compiled in a Melange probe. That probe was validation scaffolding, not the
standalone deliverable; the production Melange closure does not currently
contain Expr.

`experiments/tailcall` then reproduces those three paths using an independent
AST and evaluator: it neither links nor copies Expr, and its Dune stanzas have
no library dependencies. The correspondence is structural:

| Source-tree path | Independent standalone path | Work left after the call |
|---|---|---|
| `Value.Binary` | `Binary` | apply the binary operator |
| `Value.Unary` / `Value.Round_f32` | `Unary` | apply the unary operation |
| `Value.Select` / `Bool.Value_lt` | `Select` / `Value_lt` | compare, then select |

At depth 200,000 on the repository toolchain (OCaml 4.14.3, js_of_ocaml 6.4.1,
Melange 5.1.0-414, Node 20.19.2), it records:

| Shape | native | OCaml bytecode | jsoo | jsoo `--effects=cps` | Melange |
|---|---:|---:|---:|---:|---:|
| self tail call | ok | ok | ok | ok | ok |
| mutual tail call | ok | ok | ok | ok | raises |
| mutual calls collapsed to one state variant | ok | ok | ok | ok | ok |
| mutual state tag plus separate arguments | ok | ok | ok | ok | ok |
| call through a ref | ok | ok | raises | ok | raises |
| direct binary/unary | ok | raises | raises | raises | raises |
| direct value/guard | raises | raises | raises | ok | raises |
| one-function variant, binary/unary | ok | raises | raises | raises | raises |
| one-function variant, value/guard | raises | raises | raises | raises | raises |
| eager closure trampoline, all three trees | ok | ok | ok | ok | ok |
| delayed closure trampoline, all three trees | ok | ok | ok | ok | ok |
| explicit machine, all three trees | ok | ok | ok | ok | ok |

“Raises” is deliberately backend-neutral: V8 stack exhaustion is represented
differently by the two JavaScript runtimes.  The important contract is that the
computation did not return its correct value.

The generated code confirms the mechanism, not merely the outcome. Melange
turns the direct evaluator's tail-recursive selected branch into a loop but
leaves its binary, unary, and guard operand calls as ordinary JavaScript calls.
It emits the proposed payload-variant rewrite, its tag-and-arguments refinement,
and the explicit machine as `while (true)` loops. The refined loop uses scalar
locals and allocates no state object per transition. js_of_ocaml likewise leaves
nested calls in the evaluator dispatcher, while emitting the state loops and
machine as loops. The original mutually recursive Melange control pair remains
two ordinary JavaScript functions calling one another, and V8 exhausts its
stack.

Primary references: js_of_ocaml's
[tail-call documentation](https://ocsigen.org/js_of_ocaml/latest/js_of_ocaml/tailcall.html)
defines the loop/trampoline/unknown-call boundary.  Melange's
[project documentation](https://melange.re/) does not make an equivalent mutual
tail-call guarantee; the claim here is deliberately limited to the pinned 5.1
compiler and its inspected generated JavaScript.

## Why Expr still overflows under js_of_ocaml

`Expr.Eval.value`'s `go`, `guard`, `intrinsic`, and `eval_scan_at` are already a
statically visible recursive group.  That helps only for a call that is the
caller's final operation.  Representative arms all retain work:

```ocaml
Value.Binary (op, a, b) -> apply_binary op (go a) (go b)
Bool.Value_lt (a, b) -> go a < go b
Value.Round_f32 a -> round (go a)
```

The same issue is broader than evaluation:

- `Eval.eval_index` evaluates operands before checked arithmetic;
- `Fold.walk`, measurement, scoped queries, and scan-cost traversal combine
  child results;
- `Rewrite.rebuild` reconstructs a parent after its children return;
- `Value.compare` and `Value.hash` combine child results;
- `Pp.at` resumes formatting after each child;
- `Check.duplicate_binder` resumes at siblings and unwinds scope.

This agrees with `test/native/depth_probe.ml`: under Node, real Expr traversals
already fail at depths where the checker itself can still succeed.  The hard
depth limits are therefore valid protection, but they are a restriction rather
than a constant-stack implementation.

The producer recursion in `Kernel_eval.value_at` is a second, independent
stack.  A virtual load calls `eval_value`, which calls `Expr.Eval.value`, whose
host load can call `eval_value` again.  The existing runtime
`Hard.eval_recursion` guard contains it.  `Kernel_eval.run` does not have this
problem because it materializes values in list order.

## Refactoring options

### 1. Collapse a mutually tail-recursive group to one state loop

This is the exact proposed transformation:

```ocaml
type state = Foo of foo_args | Bar of bar_args

let rec loop = function
  | Foo a -> ... loop (Bar b)
  | Bar b -> ... loop (Foo c)
```

It succeeds when each cross-function transition is in tail position and all
pending information is in the variant payload. The standalone `Foo | Bar`
case reaches depth 200,000 on Melange, while the equivalent two-function pair
overflows. It also succeeds on native, bytecode, and both js_of_ocaml modes.

For current Expr, this can consolidate tail edges such as `go -> intrinsic` and
`go -> eval_scan_at`. It does not make `Binary`, `Unary`, `Round_f32`, or
`Value_lt` constant-stack: those branches still have an operation to perform
after a child result returns. `Select` also needs the guard result before it can
choose its tail-called value branch.

All intended tail transitions should carry OCaml's call-site `[@tailcall]`
attribute, including both sides of a mutual pair and every transition in the
single state loop:

```ocaml
| Value.Intrinsic i -> (intrinsic [@tailcall]) reducers i
| Value.Scan_at (s, row, lane) ->
    (eval_scan_at [@tailcall]) reducers s row lane
| Foo a -> ... (loop [@tailcall]) (Bar b)
```

The attribute is a checked assertion, not an optimization switch. OCaml emits
warning 51 (`wrong-tailcall-expectation`) if the annotated application is not
in tail position; builds must promote warning 51 to an error for this to be a
hard contract. The standalone control calls are annotated accordingly. This
protects the native kernel and JavaScript state loops from accidentally losing
tail position during refactoring, but it cannot protect the genuinely non-tail
Expr branches listed above. Those still require the existing depth bound,
trampolining, or explicit pending-work representation.

The payload form constructs a state object at each transition. When the two
functions' arguments can use a common representation, a refinement keeps only
a nullary `Foo_tag | Bar_tag` variant and passes the arguments as separate loop
parameters. Melange then emits one `while (true)` over scalar locals.

Only the JavaScript implementations are candidates for this substitution; the
native build keeps the existing mutual functions through cppo. Three runs of
the JavaScript benchmark at 64 transitions gave these medians:

| JavaScript form | js_of_ocaml ns/eval | Melange ns/eval |
|---|---:|---:|
| two mutual functions | 23.5 | 34.9 |
| payload state variant | 198 | 180 |
| state tag plus separate arguments | 56.0 | 60.6 |

The tag-and-arguments form is the better JavaScript fallback, especially for
Melange, but js_of_ocaml's own mutual-tail handling is faster for this tiny
pair. A jsoo-specific driver may therefore retain the mutual functions while a
Melange driver uses the state loop; a single JS driver can use the state loop
when reduced implementation diversity matters more than this dispatch cost.

### 2. Eager and delayed closure trampolines

A trampoline can also handle the evaluator's non-tail branches by putting the
pending operation in a continuation. The eager form returns `More thunk` for
every logical transition and a root `drive` loop invokes thunks until it obtains
`Done value`. It has a constant host call stack, but closure creation and a root
bounce at every transition are expensive.

The delayed form carries a segment-depth counter through **every** evaluator,
guard, and continuation call. Calls proceed normally below a threshold. At the
threshold, the current operation returns `More` without invoking it. Because
all intermediate CPS calls return that value in tail position, it propagates
unchanged to the root driver, fully unwinding the current JavaScript stack. The
root invokes the suspension and the next segment starts with depth zero:

```ocaml
if depth >= threshold then
  More (fun () -> eval 0 expression continuation)
else
  eval (depth + 1) child continuation
```

The standalone result records the maximum depth reached and counts root
bounces. On a depth-64 binary expression there are 257 counted transitions:

| Strategy | Root bounces/eval | Observed maximum segment depth |
|---|---:|---:|
| eager, threshold 1 | 257 | 1 |
| delayed, threshold 32 | 8 | 32 |
| delayed, threshold 128 | 2 | 128 |

All eager and delayed binary, unary, and value/guard cases complete at depth
200,000 under both JavaScript compilers. This verifies that a suspension really
returns to the root before resumption rather than merely moving recursion into
another nested call.

The bounces are amortized, but the CPS continuations still allocate. Three
depth-64 JavaScript runs gave these medians:

| Evaluator | js_of_ocaml ns/eval | Melange ns/eval |
|---|---:|---:|
| direct recursion (unsafe at large depth) | 605 | 637 |
| one evaluator dispatcher (also unsafe) | 845 | 711 |
| eager trampoline, threshold 1 | 8,500 | 7,821 |
| delayed trampoline, threshold 32 | 3,570 | 3,593 |
| delayed trampoline, threshold 128 | 3,690 | 3,758 |
| explicit list frames | 1,575 | 983 |
| reusable array frames | 1,150 | 685 |
| hybrid: switch to reused frames at depth 32 | 865 | 593 |
| hybrid: cutoff 128, not reached at depth 64 | 710 | 691 |

These are directional microbenchmarks; absolute values and small differences
are sensitive to whole-program linking and V8 optimization.

Generated-code inspection explains the gap. Every `Binary` still creates two
JavaScript continuation closures, and continuation return goes through an
indirect call (`Curry._2` in Melange and the corresponding generic call helper
in js_of_ocaml). Each transition also checks/increments depth and updates the
instrumentation counters. The top-level prototype constructs its trampoline
driver closures and counter cells for every evaluation. Changing the threshold
only changes how often `More` and its resume thunk are created; it does not
remove any of those fixed CPS costs.

Delayed bouncing is materially cheaper than eager bouncing, but increasing the
threshold from 32 to 128 did not improve this microbenchmark: continuation
allocation and dispatch dominate once root bounces are infrequent. Saving six
root bounces out of 257 transitions is small, while the larger threshold permits
deeper real call/return segments and can interact less favorably with V8's JIT.
The 32-versus-128 difference is small enough that it should not be treated as a
monotonic law without a larger benchmark. Thresholds must remain below the
smallest measured stack frontier over all participating functions, with
headroom for runtime and error paths. The counter must include continuation
calls as well as the visibly recursive evaluator functions, or the claimed
bound is incomplete.

Some fixed trampoline overhead can itself be amortized by constructing a runner
and its counters once per kernel execution instead of once per value. The
per-node continuation closures remain; eliminating those closures is precisely
the defunctionalization into explicit frames described next.

### 3. Defunctionalize non-tail evaluator continuations

Turn calls into a `state` variant and pending work into a `frame` variant, then
run one self-tail-recursive loop.  `experiments/tailcall` proves this is safe in
both JavaScript compilers.  It is the most mechanical route for all structural
traversals and preserves their current ordering exactly.

The JavaScript measurements above make list frames faster than either closure
trampoline on this workload, at the cost of a larger mechanical rewrite. They
also provide predictable data representation and make maximum pending depth
directly enforceable.

Frame **allocation** can be amortized even though frame push/pop work cannot be
removed. The standalone reusable form keeps frame tags, child expressions,
operators, and partial values in parallel arrays with a mutable top index. The
arrays grow geometrically, and the benchmark warms them before timing; repeated
evaluations overwrite existing slots rather than allocate list cells and frame
variants. It completes every depth-200,000 shape on both JavaScript backends.

At depth 64, reuse improves js_of_ocaml from 1,575 to 1,150 ns/eval and Melange
from 983 to 685 ns/eval. The Melange result is about 8% slower than unsafe
direct recursion while retaining a constant host call stack. Jsoo retains more
dispatch/array overhead and is about 1.9x the direct baseline.

A production scratch stack should belong to an evaluator invocation or kernel
execution and be reused across output coordinates. Reentrant producer loads
need either separate scratch storage or a saved base/top range. Capacity can be
the checked maximum depth or grow and remain cached. Parallel arrays retain old
Expr references until overwritten, so a long-lived scratch object should clear
unused reference slots when expressions vary materially.

An additional hybrid can use already-computed depth metadata: shallow values
take the direct JavaScript evaluator, while only values above a conservative
threshold enter the reusable-frame evaluator. This amortizes the state-machine
cost over the cases that need it without trying to convert a live JavaScript
call stack after the threshold is reached.

If no classification is available, the standalone `eval_hybrid` instead
threads a depth counter through the otherwise-direct evaluator and switches the
current subtree to reusable frames at the cutoff. At depth 64, cutoff 128 is
never reached, so its 710 ns/eval under js_of_ocaml and 691 under Melange measure
the shallow-path counter/check overhead against the 605/637 direct baselines.
Cutoff 32 exercises the actual in-flight switch and remains much cheaper than a
closure trampoline.

An in-flight switch does leave the existing direct frames underneath it, but it
is still bounded: the reusable-frame evaluator adds only fixed host-stack
overhead, then returns and lets the direct prefix unwind. The cutoff must leave
headroom for that fixed overhead. Full unwinding to a root boundary is required
only if evaluation will later resume direct recursion with a reset depth, as in
the delayed trampoline.

Exact dynamic depth is not generally knowable in advance because `Select` and
callbacks determine the path at runtime. A conservative structural maximum is
knowable for the immutable Expr. `Kernel.create` already computes dependency
and cumulative evaluation-depth maps in its iterative validation sweep,
currently binds them as `_depths`, and then discards them. A production
JavaScript hybrid can retain the relevant per-value classification and select
its driver once when the converted value or region is prepared, rather than
walking the Expr for every output coordinate. If retaining that metadata is
undesirable, the in-flight counter is the appropriate fallback.

Producer recursion is a separate dimension. The cached expression depth can
choose the evaluator for one `Expr.Eval.value` call, but a chain of
`Kernel_eval.value_at` calls still consumes host stack between evaluator roots.
The existing `Hard.eval_recursion` runtime guard must remain unless that outer
producer traversal is also made iterative. Backend-specific thresholds should
therefore be calibrated with the complete linked call path, not just the
standalone AST depth.

For cold operations such as validation, pretty-printing, and rewrites, the
allocation tradeoff is easier to accept.  They can move independently after
evaluation because each has its own ordering and scope invariants.

### 4. One evaluator dispatcher without explicit continuations

This works only when all transitions are tail calls or the pending context can
be summarized in fixed state. The literal proposed rewrite still overflows on
the standalone binary, unary, and value/guard trees in both JavaScript backends
because it performs arithmetic or comparison after the recursive call returns.

For full Expr this option becomes option 3 as soon as variants are added for
pending binary operands, operators, select branches, reduction accumulators,
scan rows, and return values.  At that point the variant is useful, but it is a
defunctionalized continuation rather than merely a dispatch tag.

### 5. Compiler-provided trampolines

- js_of_ocaml's default trampoline already covers known mutual **tail** calls.
  Changing `tc_depth` trades speed for stack headroom but cannot help non-tail
  Expr traversal.
- `--effects=cps` made the experiment's ref call safe and happened to make the
  value/guard chain survive in the current linked test, but binary and unary
  non-tail recursion still overflow. It is therefore not a complete solution
  and imposes a whole-program performance/code-size choice.
- Melange 5.1 emits self-tail recursion as loops but does not trampoline the
  tested mutual pair. The proposed function-tag state loop is sufficient for
  those tail transitions.

### 6. Keep bounded recursion or change placement

The current hard limits are the smallest change and remain useful defense in
depth even after an iterative evaluator exists, because parsers, rewrites, and
printers still traverse the input.  For producer recursion specifically, the
JavaScript backend can prefer the already iterative/materialized `run` path and
decline deeply virtual `value_at` placement.  That trades buffers for call
stack and does not solve recursive Expr traversal within one value.

## Sharing code without changing the native evaluator

Cppo should select the recursive driver at compile time:

```text
Expr types + primitive semantics + error definitions
                       |
              backend-selected driver
                /                 \
 native: current recursion   JavaScript: direct fast path plus
                             state loop, delayed trampoline,
                             or reusable-frame slow path
```

The native branch must continue compiling the present evaluator and must not
contain the JavaScript state, trampoline, or frame machinery. Common code can
be moved into helpers for primitive operations and errors. That extraction is
the only expected native change; it can still affect inlining and call
boundaries, so the existing native region and inference benchmarks should
confirm that the factoring is neutral.

Use cppo while compiling separate `expr_native` and `expr_js` source instances.
The experiment proves this arrangement. Shared files should be generated or
factored once rather than forked, following the existing Melange mirror.

cppo cannot select native versus js_of_ocaml inside the current shared `expr`
bytecode library: `(modes js)` compiles that already-built bytecode later, and
no `JS_BACKEND` macro exists when cppo runs.  A real cppo cutover therefore
requires separate library instances or separate Dune contexts/profiles, not
only an `#if` added to `eval.ml`.  Melange already recompiles mirrored sources
in its own mode, so selection there is straightforward once Expr enters its
dependency closure.

Recommended sequence:

1. retain all current limits;
2. annotate every intended recursive tail transition with `[@tailcall]` and
   make warning 51 fatal in every native, js_of_ocaml, and Melange build;
3. use the function-tag state transform for genuinely mutual tail transitions
   that must also run under Melange;
4. keep the current native evaluator behind cppo and extract only genuinely
   common semantic helpers; benchmark that extraction for native regressions;
5. profile the state-loop, delayed-trampoline, and explicit-frame JavaScript
   implementations with `make tailcall.js-benchmark` and real Expr workloads;
6. retain validation's depth classification and dispatch shallow JavaScript
   values directly, using delayed trampolining or reusable frames for deeper
   paths while preserving lazy selection, error ordering, reductions, scans,
   and metering;
7. remove or raise a depth guard only after every downstream traversal named by
   `depth_probe` is stack-safe.
