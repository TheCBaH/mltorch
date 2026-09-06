# Expression tail-call implementation plan

Implementation checklist: [separate tracker](expr-tailcall-implementation-todo.md).

## Context and scope

Implement the recommended sequence in `.ai/expr_tailcall_design.md`, with
each stage independently buildable, testable, and committable. js_of_ocaml
already trampolines statically known mutual tail calls, but most recursive
child calls in `Expr.Eval.value` are not in tail position. The existing hard
depth limits provide protection; they do not make the evaluator use a
constant host stack. `experiments/tailcall/` supplies dependency-free
candidate implementations and depth-200,000 experiments for native,
bytecode, js_of_ocaml, and Melange.

Use full library mirrors for backend divergence. Keep native's evaluator
behavior and hard limits unchanged. Melange's scope is the isolated Expr
evaluator; the plan does not mirror the Bigarray-dependent `native` library
to Melange.

The guarantee is stack safety of the converted `Value`/`Bool` traversal for
`Index.t` subtrees whose depth is already bounded by caller validation
(`Check.value`'s optional `max_depth`) or Kernel admission. `Eval.eval_index`
remains recursive and out of scope. A shallow value containing an unbounded
index tree can still exhaust the stack. Stage 7 verifies this boundary.
Host callbacks that synchronously re-enter the evaluator still introduce
ordinary host recursion; each top-level evaluation owns separate storage.

`Kernel_eval.eval_value`'s producer recursion remains guarded by
`Hard.eval_recursion = 96` and is outside this conversion.

## Mirroring and namespaces

### js_of_ocaml dependency closure

Use `opam exec -- dune describe workspace js/jsoo --with-deps` to verify the
local-library closure. The source plan identifies these eleven libraries
for `_js` mirrors:

`expr_internal`, `expr`, `native`, `native4d`, `native_graph`,
`native_interp`, `native_op_walk`, `native_predict`, `model_explorer_export`,
`probes_native`, and `probes_pt2`.

Keep independent libraries shared: `core`, `err_trace`, `fmt`, `jsont`,
`jsont.bytesrw`, `walk_core`, `bytesrw`, `opickle`, `zipc`, `pytorch_types`,
`schema_runtime`, `pt2`, `model_explorer`, `infer_report`, and `probes_pure`.
Check every library's actual dependencies when creating the mirrors.

Switch the three executables in `js/jsoo/dune` to the appropriate `_js`
libraries. The resulting closure must contain zero ordinary occurrences of
all eleven mirrored names, in addition to passing the existing
`aten`/`ctypes`/`unix` exclusion check.

### Namespace rules

Preserve library wrapping. `expr_internal`, `expr`, `native4d`, and
`native_op_walk` are wrapped. `native`, `native_graph`, `native_interp`,
`native_predict`, `model_explorer_export`, `probes_native`, `probes_pt2`, and
`probes_pure` are unwrapped.

Keep original entry filenames when mirroring. Use `include` to re-export a
module through a shim; `module X = Y` inside `x.ml` would create `X.X`.

| Shim | Contents |
|---|---|
| `js/jsoo/expr_js/expr_internal.ml` | `include Expr_internal_js` |
| `js/jsoo/native_js/expr.ml` | `include Expr_js.Expr` |
| `js/jsoo/model_explorer_export_js/native4d.ml` | `include Native4d_js` |
| `js/jsoo/probes_native_js/native_op_walk.ml` | `include Native_op_walk_js.Native_op_walk` |

Unwrapped mirrors export their sibling modules unqualified. Never link an
ordinary library and its mirror into the same executable. `pt2_run` only
needs the `native_predict` to `native_predict_js` library substitution.

Each additional consumer of `Expr` needs its own `expr.ml` shim:
`probe_expr`, `order_probe`, and `expr_bench`, for both JS backends.
Benchmark consumers also need an `eval_candidates.ml` shim.

### Source tree layout

Inspect every source library's directory tree before writing its mirror.
Mirror `.ml` and `.mli` files with `copy_files`, retaining subdirectories and
`include_subdirs unqualified` where used by the source library.

For `native_js`, mirror the root, `ops/`, `transform/`, and
`transform/passes/`, with a copy stanza in each directory. For
`native4d_js`, include `passes4/`, including `trim_permute4.ml`.

### Melange libraries

Create only `js/melange/expr_internal_mel` and `js/melange/expr_mel` for the
Expr closure. Preserve wrapping and `expr_api`'s private status.

`expr_internal_mel` copies `lib/expr_internal` sources and uses:

```lisp
(library
 (name expr_internal_mel)
 (modes melange)
 (enabled_if (= %{profile} melange))
 (libraries core_mel err_trace_mel fmt_mel))
```

Add the cppo preprocessing described in Stage 3. `expr_mel` copies
`lib/expr` sources and uses:

```lisp
(library
 (name expr_mel)
 (modes melange)
 (enabled_if (= %{profile} melange))
 (private_modules expr_api)
 (libraries expr_internal_mel))
```

Its `expr_internal.ml` shim contains `include Expr_internal_mel`.
Consumers' `expr.ml` shims contain `include Expr_mel.Expr`.

### Standalone Expr probe

Keep `js/probe/probe_expr.ml` separate from `probes_pure`, whose shared
dependencies must remain Expr-free. The canonical source uses only public
`Expr` APIs, builds cases with `Expr.Builder`/`Expr.Rewrite`, evaluates them,
prints structural results, and invokes its own `run ()` entry point.

The native route in `js/probe/dune` is:

```lisp
(executable (name probe_expr) (modules probe_expr) (libraries expr))
```

`js/jsoo/probe_expr/dune` contains:

```lisp
(copy_files (files %{workspace_root}/js/probe/probe_expr.ml))
(executable
 (name probe_expr)
 (modes js)
 (modules probe_expr expr)
 (libraries expr_js))
```

Check in `expr.ml` with `include Expr_js.Expr` beside it.
`js/melange/probe_expr/dune` contains:

```lisp
(copy_files (files %{workspace_root}/js/probe/probe_expr.ml))
(melange.emit
 (target output)
 (alias expr-probe-melange)
 (module_systems commonjs)
 (enabled_if (= %{profile} melange))
 (modules probe_expr expr)
 (libraries expr_mel))
```

Check in the Melange `expr.ml` shim beside it. `make expr_probe.runtest`
builds and runs all three routes and diffs their shallow outputs. The JS
entry paths are:

- `_build/default/js/jsoo/probe_expr/probe_expr.bc.js`
- `_build/default/js/melange/probe_expr/output/js/melange/probe_expr/probe_expr.js`

## Stage 0 — Baselines and evaluation-order oracle

Add test infrastructure without changing production sources.

1. Build with `--profile melange -verbose` and confirm the compiler's
   `@49..57` warning range makes warning 51 fatal, as under `dev`.
2. Temporarily annotate a non-tail call in the experiment's `eval_direct`
   Binary arm with `[@tailcall]`. Confirm failure under both profiles and
   remove the temporary annotation.
3. Record `make benchmark.region_pixel` and `make benchmark.region_compute`
   as native performance baselines.
4. Save `test/native/depth_probe.ml`'s expect output and hard constants:
   `depth = 256`, `eval_depth = 1536`, `eval_recursion = 96`. The measured
   1536 frontier applies to that probe's binary shape, not every node kind.
5. Add `js/probe/order_probe.ml`, using logging `Env.load`/`Env.load_index`
   callbacks with the real evaluator. Record outcomes, values, load order,
   and first-error priority.

### Order cases

Measure seven rewritten call sites independently, with one successful
both-operands-load case and one both-operands-fail case for each:

| Site | Operands whose order must be preserved |
|---|---|
| `Binary` | Both child evaluations |
| `Bool.Value_lt` | Both child evaluations |
| `Bool.Index_eq` | Both `idx` calls |
| `Reduce` | `lo` and `hi` |
| Inline `Scan_at` | `row` and `lane` |
| `Local_scan_at` | `row` and `lane` |
| `Intrinsic.window` | Labeled `out_h` and `out_w` arguments |

Add twelve helper-preservation cases, a success/failure pair for each of
`Value.Load`, `Intrinsic.Max_pool`, `Index.Add`, `Index.Max`, `Index.Min`,
and nested `Index.Data`. Together these make 26 cases.

For `Value.Load`, make all six coordinate components independently logged
`Index.data` expressions with distinct sources and successful values; also
test all six failing. Check the final assembled value as well as the trace.
For max-pool, use four such axes (`N`, `T`, `D`, `C`); `H` and `W` use the
loop-local coordinates.

Use public smart constructors throughout: index constructors are private,
and `Index.data` produces a position whereas arithmetic needs deltas. For
example:

```ocaml
Value.value_of_index
  (Index.add
     (Index.of_position (Index.data src_a coord_a extent_a))
     (Index.of_position (Index.data src_b coord_b extent_b)))
```

Give `Index.max` and `Index.min` independent pairs. For nested `Index.data`,
make all six components of the outer node independently logged inner data
nodes; wrap the outer result with
`Value.value_of_index (Index.of_position (Index.data src coord extent))`.
Evaluate these wrappers through `Eval.value` in both the oracle and the
candidate corpus. Use small results so float conversion exactness checks
do not interfere.

All converted implementations must call `eval_index`, `Coord.map`, and
`Coord.of_fn` with their existing implementations, without incorporating
their traversal into machine transitions. Their internal unspecified-order
sites can nest arbitrarily. The 26 cases verify observable preservation
across compilation-unit and caller changes; source identity alone does not
prove compiled evaluation order is identical.

### Oracle build routes and goldens

In Stage 0, add this native/bytecode stanza to `js/probe/dune`:

```lisp
(executable
 (name order_probe)
 (modes exe byte)
 (modules order_probe)
 (libraries expr))
```

Add `js/jsoo/order_probe/dune` with a copy of the canonical source and:

```lisp
(executable
 (name order_probe)
 (modes js)
 (modules order_probe)
 (libraries expr))
```

Commit `js/probe/order_probe.expected.native`, `.bytecode`, and `.jsoo`.
`make expr_order.runtest` runs each artifact against its own golden from
this stage onward. Bytecode also serves as diagnostic reference for
native-versus-jsoo differences.

In Stage 3, switch the jsoo route to `expr_js`, add its checked-in
`expr.ml` shim (`include Expr_js.Expr`), and use `(modules order_probe expr)`.
Add `js/melange/order_probe/dune`:

```lisp
(copy_files (files %{workspace_root}/js/probe/order_probe.ml))
(melange.emit
 (target output)
 (alias expr-order-melange)
 (module_systems commonjs)
 (enabled_if (= %{profile} melange))
 (modules order_probe expr)
 (libraries expr_mel))
```

Add `expr.ml` with `include Expr_mel.Expr`. Run Melange via
`_build/default/js/melange/order_probe/output/js/melange/order_probe/order_probe.js`.
Capture and commit `.melange` only once this real mirror exists. Extend
`expr_order.runtest` to check it. Native, bytecode, and mirrored jsoo must
still match the original Stage 0 goldens exactly.

## Stage 1 — Tail-call annotations

In `lib/expr_internal/eval.ml`, annotate genuinely tail-position calls:
`go -> intrinsic`, `go -> eval_scan_at`, both selected `Select` branches,
the reduction fold's self-recursion, intrinsic `rows`/`cols`, and the scan's
inner `run`. Add a short design-reference comment at each site.

Do not annotate non-tail child calls in `Binary`, `Unary`, `Round_f32`,
`Value_lt`, `Local_at`, or a reduction body. Leave `fold.ml`, `rewrite.ml`,
`check.ml`, `value.ml`, and `pp.ml` unchanged.

Verify builds and existing suites, including `expr_order.runtest`, with no
golden changes.

## Stage 2 — Separate shared evaluator plumbing

Move definitions above `value` verbatim into new
`lib/expr_internal/eval_common.ml`, including the existing recursive
`eval_index` helper without changing its algorithm. Keep `value` in
`eval.ml` and add `open Eval_common` so its plumbing names resolve.

In `lib/expr/expr.ml`, compose the public module as:

```ocaml
module Eval = struct
  include Expr_internal.Eval_common
  include Expr_internal.Eval
end
```

Keep `expr_internal` wrapped. Verify the Stage 1 checks, all three depth
expect blocks, evaluation-order goldens, and native benchmark neutrality.

## Stage 3 — Build the mirrors

Create the eleven jsoo mirrors, the Melange Expr mirrors, all required
shims, and the standalone probes described above. Put identical copies of
Stage 2's `value` implementation on both sides of a cppo `JS_BACKEND`
conditional. This stage establishes wiring before algorithms diverge.

Use cppo with these definitions:

| Compilation | Definitions |
|---|---|
| Native `expr_internal` | None |
| `expr_internal_js` | `JS_BACKEND`, `JSOO_BACKEND` |
| `expr_internal_mel` | `JS_BACKEND`, `MELANGE_BACKEND` |

Use the shared symbol for JS-common code and backend-specific symbols for
measured operand order and eventual driver selection. For source comparison,
use `cppo -n` consistently or explicitly normalize its line markers.

Complete Stage 0's post-mirror oracle pass before Stage 4: preserve the three
original goldens and add Melange's first measurement. Recheck the eleven-name
dependency exclusion set, native benchmarks, `jsoo.runtest`, and
`expr_probe.runtest`.

Existing `jsoo.inline-runtest` suites still link ordinary `expr`/`native`;
`melange.runtest` exercises its existing subset. They remain regression
checks and do not verify the new Expr driver. `jsoo.runtest` now exercises
the mirrored executable closure; dedicated Expr cases run through
`expr_probe.runtest`.

Update `.ai/js_backends_design.md` to describe the mirrored jsoo closure
and the isolated Melange Expr scope.

## Stage 4 — Convert the mutual-tail intrinsic loop

In the JS branch, convert intrinsic `rows`/`cols` using the experiment's
`mutual_tag` technique. Keep native's implementation unchanged.

Add a public probe case exercising `Expr.Intrinsic.Max_pool` with
configuration-bounded window dimensions in the low hundreds. Run it through
`expr_probe.runtest` on native, mirrored jsoo, and Melange and require equal
results. Inspect generated JS and reconfirm native source neutrality.

The remaining `go -> intrinsic` and `go -> eval_scan_at` mutual tail edges
are not self-tail calls; Melange still relies on hard limits until the
complete driver conversion. Include nested inline scans, with `Select` or
another `Scan_at` in an update, in Stage 5's corpus.

## Stage 5 — Candidate drivers, oracle, and benchmarks

### Exposure and ownership

Create one canonical `js/probe/eval_candidates.ml`, copied into both JS
internal libraries, with `open Eval_common`. Keep benchmark-only APIs out
of the public `Expr_api.S` surface. Expose four candidates over `Value.t`:

- `eval_trampoline_delayed`, parameterized by threshold; threshold 1 is
  eager. Pending work uses CPS continuations and bounces.
- `eval_machine`, using a frame list and state variants.
- `eval_machine_reuse`, using parallel frame arrays allocated per call.
- `eval_hybrid`, a complete direct evaluator with its own cutoff and
  `eval_machine_reuse` fallback.

The tag/state-loop technique for mutual-tail transitions is not a separate
candidate for pending non-tail work. Port all constructors, including
reductions and scans, into each candidate's pending-work representation.

Frame storage is allocated fresh per top-level call and discarded on return.
Synchronous external re-entry gets independent storage. Cross-call pooling
is deferred.

### Reduction transitions

Carry `{ reducers; var; hi; combine; i; acc; body; on_reduction }`.
Evaluate bounds in the measured order. At entry, if `lo >= hi`, return the
seed (`0.` for Sum, negative infinity for Max) without pushing an iteration
frame, invoking `on_reduction`, or evaluating the body.

Otherwise start at `i = lo`. Before each body, call `on_reduction ()` and
rebuild the bound resolver from the saved outer `reducers`, variable, and
current `i`. Evaluate the body as a machine transition. Combine its result
with the accumulator; return if `i + 1 >= hi`, otherwise advance `i` and
repeat. Test empty and reversed Sum/Max ranges with exact event traces.

### Scan transitions

Row/lane filling belongs to the same driver loop. No `Array.init`, `for`,
or recursive callback may call the driver to evaluate a child subtree.

Carry the outer reducers, scan lane/step/prev identities, init/update
bodies, width/steps, requested row/lane, previous/current row arrays, fill
cursor, and saved/active local resolver.

1. Fill all lanes of row 0 using `init`, binding the lane over the outer
   reducers. Do not charge updates or rebind the local resolver. Once
   complete, let `r = 0` denote the completed row index.
2. On a fully filled row, if `r = requested_row`, return the requested
   lane and run that scan's cleanup. Otherwise retain the completed array
   as `prev_row`, rebind the local resolver to answer the scan's `prev`
   identity from it, and allocate a fresh `cur_row` for row `r + 1`.
3. Fill the new row's lanes with `update`, charging `Scan_meter.charge_update`
   before each body. Bind the lane and `step = r` over the outer reducers.
   After all lanes finish, advance the completed row index and repeat.

Never mutate a row while it is exposed as `prev_row`; drop arrays older
than the current previous row. Test row 0's lack of update events, row 1's
`step = 0`, complete-row visibility, and absence of reads from older rows.

At any hybrid cutoff, transfer the current reducers and active local
resolver into the machine. Preserve active cleanup state across the handoff.

### Stack-safety evidence

For explicit-frame candidates, require exactly one external driver entry
per evaluation and no internal transition that calls that entry again.
For delayed trampoline/hybrid candidates, require both a maximum segment
depth bounded by the configured threshold plus documented fixed overhead
and an active-root-driver count of one for the same evaluation. Independent
external re-entry remains a separate evaluation.

Use existing threaded depth instrumentation; do not add post-recursion
decrements that destroy tail position. Generated-JavaScript inspection is
a required gate for every candidate: verify flat loops and no hidden
recursive loop calls at child-evaluation sites. Counters and surviving
deep cases supplement this inspection; neither alone detects every lexical
loop re-entry. Do not use `Printexc.get_callstack` as JS stack evidence.

### Corpus and correctness harness

Add `test/expr_bench/corpus.ml`, entirely synthetic and Expr-only. Construct
deep cases iteratively and record depth in the generator; do not recover
it afterward through the still-recursive `Expr.Fold.depth`.

Each entry contains its `Expr.Value.t`, generated depth, per-backend
`Below_frontier`/`Above_frontier` classification, and expected outcome with
exact observable trace. Outcomes support successful values, structured
error tags/payloads, and `Raises of (exn -> bool)` for ordinary exceptions.
Trace loads, index loads, reduction callbacks, and scan events.

Order-independent expectations use `Shared (outcome, trace)`. All 26 order
cases use `Per_backend { native; jsoo; melange }`, populated from committed
oracle goldens. Run them through every candidate, including all helper
cases. Bytecode remains an oracle diagnostic and golden regression route,
without a separate benchmark expectation field.

Measure the direct evaluator's safe frontier independently for every shape
and backend. Do not infer one backend's frontier from another or reuse the
binary probe's 1536 figure for reductions or scans. Entries at or beyond an
unsafe boundary use generator-known, closed-form expectations.

`expr_bench_run.ml` must skip the reference evaluator entirely whenever an
entry is above that backend's frontier. On JS, still run every candidate
against the expected outcome and trace at those depths. On safe entries,
check both the reference evaluator and candidates. Any value, error, or
trace mismatch exits nonzero.

`--bench` uses `Sys.time` and suppresses correctness output. Measure both
representative shallow/production-shaped workloads and bounded ranges of
above-frontier depths. Time only candidates where the direct evaluator is
unsafe. Report allocation, bounce/frame growth, and scan/reduction scaling
as relevant to the selection. Benchmark the exact candidate/cutoff
composition proposed for production.

### Benchmark build routes

The native executable links ordinary `expr` and preprocesses its canonical
runner with cppo without definitions. Candidate calls are JS-gated.

In `test/expr_bench/jsoo`, copy `corpus.ml` and `expr_bench_run.ml`; check in
`expr.ml` containing `include Expr_js.Expr` and `eval_candidates.ml`
containing `include Expr_internal_js.Eval_candidates`. Use:

```lisp
(executable
 (name expr_bench_run)
 (modes js)
 (modules corpus expr_bench_run expr eval_candidates)
 (libraries expr_js expr_internal_js)
 (preprocess
  (action (run %{bin:cppo} -D JS_BACKEND -D JSOO_BACKEND %{input-file}))))
```

In `test/expr_bench/melange`, copy the same sources and add corresponding
shims with `include Expr_mel.Expr` and
`include Expr_internal_mel.Eval_candidates`. Use:

```lisp
(melange.emit
 (target output)
 (alias expr-bench-melange)
 (module_systems commonjs)
 (enabled_if (= %{profile} melange))
 (modules corpus expr_bench_run expr eval_candidates)
 (libraries expr_mel expr_internal_mel)
 (preprocess
  (action (run %{bin:cppo} -D JS_BACKEND -D MELANGE_BACKEND %{input-file}))))
```

Preprocessor definitions belong to each consumer stanza; linking a library
does not propagate its definitions. Add `expr_bench.runtest` and
`expr_bench.js-benchmark`. Their Melange entry path is
`_build/default/test/expr_bench/melange/output/test/expr_bench/melange/expr_bench_run.js`.

### Cleanup protocol

Use one non-lexical cleanup mechanism for every candidate, including the
hybrid's direct segment. A trampoline bounce is a host return but not scan
completion, so lexical `Fun.protect` around a segment would release state
too early.

Maintain a per-call mutable `(unit -> unit) list ref`. Immediately after a
successful `Scan_meter.reserve`, push a closure capturing the saved local
resolver, meter, and width. A failed reservation installs no cleanup.
At logical scan completion, pop and run its cleanup, restoring the resolver
and releasing the reservation. A bounce does neither.

Wrap the single top-level loop invocation with a handler inside the
`Err.Escape.with_escape` callback. On any exception:

1. Catch the original exception and immediately capture
   `Printexc.get_raw_backtrace ()` before cleanup touches runtime state.
2. Pop and run all pending cleanups in LIFO order.
3. Re-raise the same exception using `Printexc.raise_with_backtrace`.

The heap-visible stack survives loop unwinding. Keep the handler outside
the tail loop, and inside the structured-escape conversion boundary.
Prefer one shared unwind helper; otherwise review this ordering at every
implementation.

Test nested inline scans in three modes: success, structured failure partway
through the inner update after reservation, and an ordinary callback
exception at that point. On success, read outer `Local_at prev` after the
inner scan returns within the same evaluation to observe resolver
restoration.

For ordinary failure, bind one `expected_exn = Injected_fault (ref 0)`,
raise that exact value, and assert physical exception identity
`exn == expected_exn` on jsoo and Melange. Backtrace preservation remains
an implementation/review requirement; do not claim this identity test
verifies backtrace contents.

For either abort, reuse the meter in a second scan requesting row 0, which
uses no update budget. Let `O = 2 * outer_width`, `I = 2 * inner_width`,
and `S = 2 * second_width`. Choose the state budget to satisfy:

```text
O + I <= max_state
S <= max_state
max_state < S + min(O, I)
```

Thus the original reservations fit, full cleanup permits the second scan,
and leaking either reservation causes `State_over_limit`. Provide sufficient
update budget to reach the injected fault.

Expose a private test-only optional `skip_cleanup` selector for Outer/Inner
through `Eval_candidates`, using the same unwind helper. Run the check with
no fault (second scan succeeds), skipped outer cleanup (state failure), and
skipped inner cleanup (state failure). Keep fault injection outside the
public Expr API and disabled in production.

A separate evaluation cannot observe the failed call's discarded local
resolver ref. Abort-path resolver restoration is reviewed in code; the
same-call success case is the behavioral test of it.

### Operand order across the cutoff

In the JS direct path, explicitly sequence all seven rewritten sites to
their backend's measured order. Give each candidate's machine transitions
the identical order. Preserve jsoo and Melange's individual behavior even
if their goldens differ. Keep native's direct implementation unchanged.

Add both-load and both-fail cases on both sides of the cutoff, with shallow
and deep subtrees in one evaluation. Verify load/index-load events and the
winning error, along with the helper-preservation corpus.

## Stage 6 — Install the selected dispatcher

Before integration, record the selected candidate and cutoff per backend,
with reasons and shallow/deep benchmark evidence, in
`.ai/expr_tailcall_design.md`. Selection may differ between jsoo and Melange.

Install the chosen implementation entirely within `eval.ml`'s JS branch,
preserving `Expr.Eval.value`'s signature. No changes to Kernel, Kernel_eval,
Region_eval, Region_execution, or Schedule are needed in this stage.

If `eval_hybrid` wins, install its complete measured dispatcher directly.
For a standalone trampoline or machine, compose a direct shallow path and
the selected fallback, then measure that exact composition before shipping.
Preserve reducers, local resolver, and cleanup state at the switch.

Verify the cutoff using private instrumentation through
`expr_bench.runtest`, which links the internal libraries. Check a case just
below it remains direct and one above it uses the safe path and completes
beyond the direct frontier. Keep public `expr_probe` focused on observable
behavior. Re-run correctness and both benchmark workload classes.

## Stage 7 — Raise only the mirrored Kernel evaluation-depth limit

Begin after Stage 6 is verified on both JS backends beyond current ceilings.
Keep native's `Hard.depth`, `Hard.eval_depth`, and `Hard.eval_recursion`
unchanged, along with all shared non-evaluator traversal guards.

### Constant extraction

Create `lib/native/kernel_hard_shared.ml` containing every Hard constant
except `eval_depth`: `size`, `depth`, `values`, `dep_depth`, `inputs`,
`outputs`, `eval_recursion`, `extent`, `numel`, `max_local_slots`,
`max_scan_state`, `max_scan_updates_per_key`, and `max_scan_updates_total`.
Mirror this source unchanged.

Create flat `lib/native/kernel_hard.ml`:

```ocaml
include Kernel_hard_shared
let eval_depth = 1536
```

Inside `Kernel.Limits`, re-export `module Hard = Kernel_hard`, preserving
`Kernel.Limits.Hard.*`. Override only `js/jsoo/native_js/kernel_hard.ml`
with the measured JS `eval_depth` value, using the checked-in-source
override convention established by the Melange walk_core mirror. Keep
every other constant sourced from the shared file.

Add both extracted modules to `private_modules` in both native profile
stanzas and the mirrored library. Do not add cppo to `kernel.ml`; retain
compatibility with its landmarks PPX preprocessing. Verify
`dune build --profile landmarks`.

Choose the new constant from measured production compositions with roughly
2x headroom, rather than using the adversarial depth-200,000 figure directly.
Update `kernel.mli` documentation to distinguish shared `depth` and
`eval_recursion` from backend-specific `eval_depth`, naming the separate
native and mirrored tests. Preserve the public signature.

### Kernel admission and execution tests

Add a jsoo test linked to `native_js`; existing native inline tests do not
exercise this mirror. Independently verify three boundaries:

- `Eval_too_deep`: use a producer chain whose individual bodies obey
  `max_depth`, with cumulative depth determined by
  `e = e_prev_max + 1 + body_depth`. Test admission and `run` near the new
  boundary, plus the intended one-past rejection. `run` executes in
  topological order without producer recursion.
- `Body { ... Too_deep ... }`: use one expression just beyond a validly
  configured `max_depth`, whose largest admissible setting is 255 under
  `Hard.depth = 256`.
- `Recursion_too_deep`: exercise `value_at` at producer depth 96 (succeeds)
  and 97 (fails), using otherwise admissible topology. The guard is
  `depth > 96`; 97 admitted producer values imply 96 transitions.

For the cumulative-depth chain, configure `Kernel.Limits.create` with
sufficient `max_dep_depth` and `max_values` within unchanged hard ceilings
4096 and 65536. Defaults 1024/4096 may reject the setup too early. Confirm
the rejected boundary case reports `Eval_too_deep`, not another guard.
Measure constructed, bounded test body/chain depths with
`Region_program.Fold.max_depth` or an equivalent check before using them
as boundary fixtures. Do not make `value_at` reach a cumulative ceiling
that its separate producer guard prevents it from reaching.

### Deep isolated Expr tests

Add a runtime `--deep` flag to the shared `probe_expr.ml`. Default mode
keeps the shallow three-way diff. `expr_probe.deep-runtest` invokes only
jsoo and Melange with `--deep`, tests depth-200,000 Value/Bool cases against
closed forms, and exits nonzero on mismatch. It never runs native on these
inputs.

### Deep-index scope boundary

As Stage 7's first measurement, run an iteratively built chain of 200,000
`Index.add` operations adding delta 1, wrapped in a shallow
`Value.value_of_index`, under each JS backend. The mathematical result
200,000 must stay clear of integer overflow. Build before any observation
handler so construction failures cannot count as evaluator stack failures.

Record whether each backend's exhaustion is OCaml-catchable and the exact
exception, or its process exit status/signal and distinguishing stderr
diagnostic. Only that confirmed stack-exhaustion signal passes the negative
control; normal completion and unrelated errors fail.

Add hand-written `stack_fault.ml` shims to `js/probe/`,
`js/jsoo/probe_expr/`, and `js/melange/probe_expr/`, included in each route's
module list. Each exposes:

```ocaml
val run_in_process : bool
val is_exhausted : exn -> bool
```

For catchable faults, set the flag true and recognize the measured fault.
Native's default recognizes `Stack_overflow`. For uncaught process faults,
set it false and provide an unused false-returning predicate.

The shared `--deep` code runs the negative control only when the flag is
true, using `try ignore (Eval.value ...); false with e ->
Stack_fault.is_exhausted e`. When false, omit both evaluation and output
for this case and use the separate-process route below. Keep the source
identical across all builds without backend preprocessing.

For each backend requiring a separate process, create one canonical source
`js/probe/probe_expr_stack_fault.ml`, copied into its route. It builds the
same tree and evaluates with no `try`; normal completion prints
`UNEXPECTED_COMPLETION` and exits 0.

`js/jsoo/probe_expr_stack_fault/dune` contains:

```lisp
(copy_files (files %{workspace_root}/js/probe/probe_expr_stack_fault.ml))
(executable
 (name probe_expr_stack_fault)
 (modes js)
 (modules probe_expr_stack_fault expr)
 (libraries expr_js))
```

Add `expr.ml` with `include Expr_js.Expr`. Output is
`_build/default/js/jsoo/probe_expr_stack_fault/probe_expr_stack_fault.bc.js`.

`js/melange/probe_expr_stack_fault/dune` contains:

```lisp
(copy_files (files %{workspace_root}/js/probe/probe_expr_stack_fault.ml))
(melange.emit
 (target output)
 (alias expr-probe-stack-fault-melange)
 (module_systems commonjs)
 (enabled_if (= %{profile} melange))
 (modules probe_expr_stack_fault expr)
 (libraries expr_mel))
```

Add `expr.ml` with `include Expr_mel.Expr`. Output is
`_build/default/js/melange/probe_expr_stack_fault/output/js/melange/probe_expr_stack_fault/probe_expr_stack_fault.js`.
Create only the routes needed by measured non-catchable faults.

`expr_probe.stack-fault-runtest` must capture child status and stderr using
the `cmd || ec=$$?` recipe idiom, so the expected failure does not abort Make
before assertions run. For each applicable backend, reject the normal
completion marker, check its measured exit status/signal, and match its
measured diagnostic. A generic nonzero status is insufficient.

```make
expr_probe.stack-fault-runtest:
	@ec=0; node $(JSOO_PROBE_STACK_FAULT) >/tmp/expr_probe_stack_fault.jsoo.out \
	  2>/tmp/expr_probe_stack_fault.jsoo.err || ec=$$?; \
	  if grep -q UNEXPECTED_COMPLETION /tmp/expr_probe_stack_fault.jsoo.out; then \
	    echo "jsoo: unexpected completion"; exit 1; fi; \
	  test "$$ec" = "<measured jsoo exit status>" || exit 1; \
	  grep -q "<measured jsoo diagnostic>" /tmp/expr_probe_stack_fault.jsoo.err
```

Replace placeholders from the measurement and add the corresponding Melange
recipe when needed. Each backend uses exactly one assertion route. The
separate-process target is mandatory whenever any shim sets
`run_in_process = false`; otherwise the in-process deep target checks the
negative control.

## Deferred work

- Producer-to-producer recursion in `Kernel_eval.eval_value` and changes to
  `Hard.eval_recursion`.
- Conversion of `eval_index`, `Fold`, `Rewrite`, `Value.compare`,
  `Value.hash`, or `Pp`, and changes to native's `Hard.depth`.
- Cross-call scratch pooling, which needs an explicit base/ownership and
  mutable-current-top protocol for nested re-entry and restoration.
- Retaining Kernel admission's discarded depth classifications to select
  a driver once per value instead of using an in-flight counter.

## Critical files

- `lib/expr_internal/eval.ml`, new `eval_common.ml`, and `lib/expr/expr.ml`.
- Jsoo mirror libraries, recursive copy stanzas, namespace shims, and
  `js/jsoo/dune` executable dependencies.
- Melange Expr libraries and consumer shims.
- `js/probe/order_probe.ml`, its native/jsoo/Melange routes, and four
  `order_probe.expected.*` goldens.
- `js/probe/probe_expr.ml` and its three build routes.
- `js/probe/eval_candidates.ml` and `test/expr_bench/`, including backend
  shims and preprocessing.
- `lib/native/kernel.ml`, `kernel.mli`, new `kernel_hard.ml` and
  `kernel_hard_shared.ml`, associated private-module declarations, and
  the `native_js/kernel_hard.ml` override.
- New `native_js`-linked kernel tests; existing
  `test/native/depth_probe.ml` expectations remain unchanged.
- Three `stack_fault.ml` shims, canonical `probe_expr_stack_fault.ml`, and
  any required separate-process build routes and Expr shims.
- `Makefile` targets for oracle, probes, benchmarks, and stack-fault checks.
- `.ai/expr_tailcall_design.md` and `.ai/js_backends_design.md` for resulting
  decisions, scope, measurements, and deferred work.
- `experiments/tailcall/tailcall_cases.ml` as the candidate reference.

## Verification requirements

At each stage, run `make precommit`, `dune runtest` covering `test/expr` and
`test/native`, `make jsoo.runtest`, `make jsoo.inline-runtest`,
`make melange.runtest`, and `make tailcall.runtest`.

From Stage 0 onward, require `make expr_order.runtest`. From Stage 3 onward,
also require `make expr_probe.runtest` and the full mirrored dependency
closure check. From Stage 5 onward, require `make expr_bench.runtest`,
`make expr_bench.js-benchmark`, candidate stack instrumentation, generated
JS inspection, and demonstrated cleanup negative controls.

Stage 7 additionally requires the mirrored Kernel boundary tests,
`make expr_probe.deep-runtest`, and `make expr_probe.stack-fault-runtest`
whenever a backend requires the separate-process path. The landmarks build
must succeed. Native depth-probe output and every native `Hard.*` constant
must remain identical to the Stage 0 baseline.
