# Expr tail-call conversion: stack safety on the JS backends

Why `Expr.Eval.value` needed a different algorithm on jsoo/Melange than native, what
was tried, what was measured, and what is actually installed. See
`expr-tailcall-implementation-plan.md` (gitignored, `.ai/`) for the stage-by-stage
execution plan this doc's Stage 6 section closes out; see `.ai/js_backends_design.md`
for how the JS mirror libraries this work depends on are built.

## Why

Native's evaluator (`lib/expr_internal/eval.ml`) is ordinary, unbounded-looking OCaml
recursion (`go`/`guard`/`eval_scan_at`), safe there because native's own admission
limits (`Kernel.Limits.Hard.eval_depth = 1536`, `depth_probe.ml`'s measured frontier)
bound how deep a graph can ask it to go before the graph itself is rejected. Node's
stack is shallower, and — critically — jsoo and Melange do not fail the same way at
the same depth: jsoo's own compiler recognizes some unbounded recursive shapes and
wraps them in a runtime trampoline (`caml_trampoline`/`caml_trampoline_return`);
Melange does not, and additionally raises on a shape jsoo tolerates (a MUTUAL pair of
self-recursive functions — see Stage 4 below). Shipping native's `go`/`guard`
unmodified on either JS backend meant an admitted graph could still crash the
evaluator partway through, on one backend but not the other.

## Stages 0-4: baseline, then removing the one unsafe SHAPE

Stage 0 built `js/probe/order_probe.ml`, an oracle that measures which operand a
given backend evaluates FIRST at each of the language's seven order-sensitive call
sites (see "Operand order" below) — this is the source of truth every stack-safe
candidate had to preserve.

Stage 1 added `[@tailcall]` annotations at every genuine tail edge in `eval.ml`
(`go -> intrinsic`, `go -> eval_scan_at`, both `Select` branches, the reduction
fold's self-recursion, intrinsic `rows`/`cols`, the scan's inner `run`) — a build
error (warning 51), not a silent regression, if a marked call is ever not actually a
tail call. Stage 2 split the evaluator's shared plumbing into `eval_common.ml`.
Stage 3 duplicated `value` on both sides of a `#if defined JS_BACKEND` conditional in
`eval.ml` — byte-identical at that point — so the JS branch could diverge in later
stages while this file remains the one place both versions live.

Stage 4 found the one shape that was ACTUALLY broken on a JS backend before any
stack-safety redesign: `Max_pool`'s row/column sweep was two mutually tail-recursive
functions (`rows`/`cols`). jsoo already trampolines that safely; Melange raises at
depth on a mutual tail pair specifically (a self-recursive single function is fine on
both). Collapsed into one self-recursive `loop` over a nullary `pool_tag` state,
still in tail position on every backend.

## Stage 5: four candidates, measured

`js/probe/eval_candidates.ml` / `eval_trampoline_delayed.ml` / `eval_machine_reuse.ml`
/ `eval_hybrid.ml` (never linked natively, never exposed through the public `Expr`
API) implemented four independent designs for evaluating the ENTIRE grammar
(`Value.t`/`Bool.t`, including `Reduce` and `Scan_at`) without depending on any
backend's own tail-call handling:

- **`eval_machine`** — an explicit list-of-frames state machine. Every recursive
  call becomes a pushed frame and a `loop` iteration; O(depth) heap frames instead of
  O(depth) native stack frames.
- **`eval_trampoline_delayed ~threshold`** — CPS with a `depth` counter that bounces
  (returns a `Bounce` closure instead of calling through) once `threshold` real hops
  have accumulated, trading a bounded, DOCUMENTED amount of host stack for fewer
  bounces.
- **`eval_machine_reuse`** — `eval_machine`'s design, but with an array-backed
  pending-work stack (doubled on overflow, never shrunk) and a MUTABLE
  `reduce_progress` a `Reduce`'s own iterations update in place, instead of
  `eval_machine`'s frame-list cons and record-reallocation-per-iteration.
- **`eval_hybrid ~cutoff`** — a complete DIRECT evaluator, ordinary OCaml recursion
  shaped exactly like native's own `go`/`guard`, for the first `cutoff` levels of
  depth; at `cutoff`, hands the remaining subtree to `eval_machine_reuse`'s `run`,
  sharing the live call's own escape/cleanup/resolver state rather than starting
  fresh. Once handed off, a subtree stays in the machine for the rest of its own
  evaluation.

All four were verified, per candidate per backend, against: the full 18-then-21-case
hand-written correctness corpus (`test/expr_bench/corpus.ml`); a 20,000-deep smoke
case (well past the ~1536-node frontier `depth_probe.ml` measured for a comparable
binary-chain shape); and direct inspection of the compiled JS to confirm each
candidate's intended shape (a genuine flat loop for the machine-based candidates;
explicit `depth`/threshold guards, not an accidental automatic trampoline, doing the
real work for the CPS candidate).

### Cleanup protocol

Every candidate needs ONE non-lexical mechanism to release a `Scan_at`'s
`Scan_meter` reservation and restore its `local_at` resolver on ANY exit — success,
a structured `Err` failure, or an ordinary OCaml exception — including exits that
happen mid-bounce or mid-handoff, where native's `Fun.protect` does not apply (a
trampoline bounce or a machine handoff is a host RETURN, not a stack unwind, so
`~finally` would fire immediately rather than staying pending across it). Every
candidate instead pushes a release closure onto a mutable `cleanups : (unit -> unit)
list ref` on every reservation, and the ONE top-level `try`/`with` wrapping the whole
call sweeps every pending closure LIFO on any exception.

A previously-undetected bug surfaced by this protocol's own negative controls
(Stage 5 M7): `Printexc.raise_with_backtrace` throws `"caml_restore_raw_backtrace not
polyfilled by Melange yet"` at RUNTIME on Melange — no test before the negative
controls ever forced a genuine exception through a candidate's top-level handler on
that backend. Fixed with a `capture_backtrace`/`reraise` pair, backend-gated on
`MELANGE_BACKEND` (plain `raise`, no backtrace, there; `Printexc.raise_with_backtrace`
everywhere else).

The negative controls themselves (`test/expr_bench/cleanup_negative.ml`) prove the
protocol is not vacuous: a nested `Scan_at` sharing one `Scan_meter` with a second,
independent scan, run once with full cleanup (the second scan succeeds) and once with
a private test-only `skip_cleanup` hook withholding one release (the second scan then
fails with `State_over_limit`) — exercised against all four candidates.

### Operand order

Of the seven order-sensitive call sites `order_probe` measures, only TWO — `Binary`
and `Bool.Value_lt` — actually diverge between backends: jsoo evaluates the right
operand first, Melange the left (a genuine property of each backend's own compiler,
not a choice this project makes). The other five (`Bool.Index_eq`, `Reduce`,
inline `Scan_at`, `Local_scan_at`, `Intrinsic.window`) either use a `let ... and ...`
binding both backends already agree on, or are LEAF transitions in every candidate —
verbatim copies of `eval.ml`'s own call shape, inheriting whichever order the backend
naturally produces with no candidate-specific rewrite in the way.

Converting `Binary`/`Value_lt` into explicit frame or CPS transitions initially
hardcoded a single left-to-right order in every machine-based candidate — silently
correct for Melange, silently WRONG for jsoo (a both-fail case's winning error could
differ from the reference evaluator's). Fixed by backend-gating (`#if defined
MELANGE_BACKEND`) both the initial dispatch (which operand starts first) and the
final recombination (restoring the original `a`/`b` slot order before applying the
operator) in every affected candidate. `eval_hybrid`'s own direct segment needed no
fix — it is an unconverted copy of `eval.ml`'s implicit-order shape, so it already
inherits the correct order the same way the five non-divergent leaf sites do.
`test/expr_bench/order_check.ml` derives each backend's actual order from the
reference evaluator's own traced run at test time (rather than hardcoding
`order_probe`'s golden strings a second time) and checks every candidate's traced
order against it.

### Benchmark evidence

`expr_bench_run.ml --bench` (`make expr_bench.js-benchmark`) times every candidate
against 21 shallow, production-shaped corpus cases and three above-frontier
`deep_chain` depths (2,000 / 8,000 / 20,000; candidates only — see "Not yet built"
below for why the reference evaluator wasn't timed there before Stage 6). An initial
pass (200 iterations) was too coarse to compare candidates on the shallow cases —
`Sys.time`'s resolution under jsoo/node left many entries reading exactly `0.0`.
Retuned to 20,000 iterations × 5 trials (minimum-of-trials, standard noise
filtering), which produced a clean, consistent result on BOTH backends:

- **`eval_hybrid(cutoff=50)` is fastest or tied-fastest on nearly every shallow
  corpus case**, often markedly so on `Scan_at`-shaped cases (e.g. Melange
  `nested_scan_row2_lane1`: 1,685 ns/eval for `eval_hybrid(cutoff=50)` vs. 2,646
  ns/eval for plain `eval_machine`).
- **`eval_machine` is consistently ~10-20% faster at the three above-frontier deep
  depths** (e.g. Melange depth=20,000: 1,219,516 ns/eval for `eval_machine` vs.
  1,436,463 ns/eval for `eval_hybrid(cutoff=50)`).

Since production workloads are overwhelmingly shallow (the deep-chain depths are a
stack-safety stress case, not a representative shape), the shallow-case win is the
decisive evidence. `cutoff=50` is not a measured MAXIMUM safe depth for the direct
segment — see "Not yet built" below — it is the one value this benchmarking pass
actually measured, proven safe to the full 20,000-deep smoke case on both backends
before being installed.

## Stage 6: selection and installation

**Selected: `eval_hybrid` with `cutoff = 50`, the same choice on both jsoo and
Melange.** No benchmark evidence collected favored a different cutoff per backend,
and a uniform choice is simpler to reason about and re-measure later.

Installed entirely within `eval.ml`'s `#if defined JS_BACKEND` branch (native's
branch, and `Expr.Eval.value`'s public signature, are UNCHANGED):

- `lib/expr_internal/eval_js_machine.ml` (new file) carries `eval_machine_reuse`'s
  machinery — the shared "pending work" types and `run` — ported from
  `js/probe/eval_machine_reuse.ml` and wrapped entirely in
  `#if defined JS_BACKEND ... #endif`, so it compiles to an EMPTY module under
  native's own `expr_internal` library (which takes no `-D`). Depends only on
  `Expr_internal` (`Eval_common`), never on `Eval`, to avoid a circular module
  dependency with `eval.ml`, which now depends on THIS file for its handoff —
  `pool_tag` is therefore duplicated rather than shared, matching the policy the
  Stage 5 candidates already established for the same reason.
- `eval.ml`'s JS branch's `go`/`guard`/`eval_scan_at` gained a `depth` counter
  (direct recursion below `cutoff`, exactly like native's own shape) and a handoff
  to `Eval_js_machine.run` at `cutoff`, sharing the live call's own `esc`/
  `cleanups`/`local_at_ref`. The cleanup protocol (mutable `cleanups` list,
  replacing `Fun.protect`) and the `capture_backtrace`/`reraise` pair are installed
  directly in `eval.ml`'s own JS branch, both ported from the Stage 5 candidates.
  `js/jsoo/expr_internal_js/dune` and `js/melange/expr_internal_mel/dune` both
  needed `eval_js_machine` added to their explicit `lib/expr_internal` file lists
  (the same dune quirk `expr_internal_js`/`expr_internal_mel`'s own comments
  already document for `eval.ml` itself).

**Verified**, beyond the ordinary `make precommit`:

- `make jsoo.runtest` / `make js.runtest` (native-vs-jsoo/Melange probe diffs) —
  unaffected, since `probe_expr` observes VALUES, not internal control flow.
- `make expr_order.runtest` — `order_probe`'s own committed per-backend goldens are
  UNCHANGED: every one of `order_probe`'s cases is shallow (well under `cutoff = 50`),
  so they all still take the direct segment, which is byte-identical in shape to
  native's own `go`/`guard`.
- `make expr_bench.runtest` — all 21 correctness cases still agree; `deep_smoke`'s
  loop now includes `reference` (= `Eval.value` itself, no longer excluded, since
  the installed JS build genuinely is stack-safe now) and it survives depth 20,000
  on both backends — direct proof, against the real production entry point rather
  than a copy of it, that the installed dispatcher works.

`js/probe/eval_candidates.ml`/`eval_trampoline_delayed.ml`/`eval_machine_reuse.ml`/
`eval_hybrid.ml` remain in place, unchanged in role: `test/expr_bench`'s ongoing
comparison harness (`corpus.ml`, `cleanup_negative.ml`, `order_check.ml`,
`--bench`) still runs all four against the (now stack-safe) reference evaluator, so
a future re-measurement or reconsideration of the selection has the same
infrastructure available, not a one-shot decision that has to be rebuilt from
scratch.

## Not yet built

- **The generated deep/order corpus** the original plan calls for (all 26 of
  `order_probe`'s cases as `Per_backend`-classified `Corpus` entries with committed
  goldens, plus closed-form deep generators). Deliberately scoped down: the two
  sites that actually diverge between backends (`Binary`/`Value_lt`) are covered by
  `order_check.ml`; the other 24 exercise `eval_index`/`Coord.map` traversal that is
  UNCHANGED and un-machine-transitioned in every candidate, so they would add little
  candidate-specific coverage beyond what the 21-case corpus and `order_check.ml`
  already prove.
- **A measured, per-backend maximum-safe `cutoff`/`threshold` frontier.** `cutoff =
  50` is proven safe to 20,000 depth and is the one value actually benchmarked — it
  is not a bisected maximum. Finding that maximum needs real stack-fault
  characterization tooling, which is Stage 7's job, not retrofitted here.
- **Allocation and bounce/frame-growth benchmark instrumentation** — `--bench`
  reports wall-clock timing only.
- **Native's own benchmark build route** — low value while candidates stay JS-only
  and native's evaluator is untouched by this whole conversion.

## Stage 7 (mostly complete)

Per the implementation plan: raise `js/jsoo/native_js/kernel_hard.ml`'s
`eval_depth` override past native's `Hard.eval_depth = 1536` — now that the JS
evaluator is genuinely stack-safe well past that figure — chosen from measured
production compositions with headroom, not the adversarial 20,000-deep smoke figure
directly. Native's own `Hard.depth`, `Hard.eval_depth`, `Hard.eval_recursion`, and
every other shared non-evaluator traversal guard stay unchanged.

**Done so far:**

- **Deep-index exhaustion signal, measured** (the plan's first Stage 7 checklist
  item): a 200,000-deep `Index.add` chain wrapped in `value_of_index`, evaluated
  through the public `Expr` API. jsoo raises OCaml's own `Stack_overflow`,
  in-process catchable, exit 0. Melange raises `Js.Js_exn.Error` wrapping a
  `RangeError: Maximum call stack size exceeded`, also in-process catchable, exit
  0. **Both backends are in-process catchable**, so the plan's separate-process
  fault-detection route (`probe_expr_stack_fault`, its dune stanzas,
  `expr_probe.stack-fault-runtest`) is not needed for this boundary — a real scope
  reduction versus what the plan anticipated. `Stack_fault.run_in_process = true`
  on both routes once that shim is written.
- **Constant extraction**, mechanical and behavior-preserving: `Hard`'s body moved
  out of `kernel.ml` into `lib/native/kernel_hard_shared.ml` (every constant except
  `eval_depth`) and `lib/native/kernel_hard.ml` (`include Kernel_hard_shared` plus
  `eval_depth`, still `1536`, unchanged). `kernel.ml`'s `Limits.Hard` is now
  `module Hard = Kernel_hard`, preserving `Kernel.Limits.Hard.*` for every
  consumer. Both modules are `private_modules` in both native profile stanzas
  (plain and `landmarks`) and in `native_js`. Verified: plain build, `--profile
  landmarks`, `native_js`, `make precommit`, `make js.runtest` — all clean, no
  behavior change anywhere.

- **JS `eval_depth` override, chosen from real production-composition data**:
  `js/jsoo/native_js/kernel_hard.ml` now overrides `eval_depth` to `12288`
  (native's own `lib/native/kernel_hard.ml` is unchanged at `1536`). The
  measurement: `bin/pt2_json_model_support.exe`, run against the full
  `modules/devcontainer.pytorch-image-models` corpus (100 models,
  payload-free — no `.pt2` download needed) with `Kernel.create`'s
  `Eval_too_deep` guard temporarily disabled (a scratch build, reverted, not
  committed — same experimental discipline as the deep-index measurement
  above), printing the combined per-value depth (`kernel.ml`'s own `e`) the
  guard would otherwise have checked. Corpus-wide maximum: `6238`
  (`csatv2`; `mobilenetv5_base` next at `5313`; every other model under
  `4800`; `9191` individual values across the corpus already exceed
  native's `1536` today — consistent with `.ai/pt2_model_support.md`'s
  existing note that roughly half the corpus hits that ceiling). `12288` is
  ~2x that measured maximum, the same headroom multiple
  `lib/native/kernel_hard.ml`'s own comment used over resnet18's ~770 to
  reach native's `1536` — not the adversarial 20,000-deep smoke figure, per
  the plan's explicit instruction, and a small, deliberate fraction of the
  20,000 depth Stage 6 already proved the installed
  `eval_hybrid ~cutoff:50` driver survives on both JS backends. The
  `dune build` override mechanism follows `js/melange/walk_core/walk_core.ml`'s
  established convention (a hand-written mirror file with the same module
  name as one the copy_files stanzas deliberately exclude) rather than
  forking `kernel.ml`'s conditional: `js/jsoo/native_js/dune`'s root
  `copy_files` stanza switched from one wildcard glob to two brace-alternation
  globs (by `.mli` presence) excluding only `kernel_hard`, since dune's
  `copy_files` has no glob-exclusion syntax — mirroring `expr_internal_js`'s
  own explicit-file-list precedent, for a different reason (avoiding a
  checked-in-file collision, not a preprocessing-rule collision).
- `kernel.mli`'s `Hard` documentation now distinguishes the shared,
  `depth_probe.ml`-pinned constants (`depth`, `eval_recursion`) from the
  backend-specific `eval_depth`, and names the jsoo/`native_js`-linked test
  below as `eval_depth`'s own honesty check.
- `dune build --profile landmarks` reconfirmed clean after both dune changes
  above (still not wired into a Makefile target — pre-existing gap, and this
  environment's own `dlllandmark_stubs.so` is missing independently of this
  work, confirmed by reproducing the same error on a clean `git stash`).
- **Mirrored Kernel boundary tests**: new `test/native_js/`, linking
  `native_js` directly (existing native inline tests recompile ordinary
  `native` to JS under `modes best js` — they never exercise this mirror,
  so nothing previously proved the `12288` override actually reaches
  `Kernel.create`'s admission check). Three independent boundaries, matching
  `kernel.mli`'s own "four independent budget dimensions" table: a
  3072-value chain lands exactly on the mirror's own `12288` combined
  ceiling (`Eval_too_deep`, admits and `Kernel_eval.run`s; 3073 values
  rejects); single-value bodies of raw `Expr.Fold.depth` 255/256 accept/
  reject under the largest admissible `max_depth:255` (`Too_deep`, the
  unchanged per-body dimension — a genuinely separate check from the
  combined one, comparing raw depth with no result-conversion +1);
  `Kernel_eval.value_at` at producer depth 96/97 (`Recursion_too_deep`,
  `Hard.eval_recursion = 96`, unchanged). Getting the combined-depth chain's
  arithmetic right needed an empirical scratch probe of its own: naively
  reading `e = e_prev + 1 + max_depth(computation)` as contributing
  `d + 1` per value (matching a `d`-node body) is off by one, since
  `Expr.Fold.depth` counts a bare leaf as depth 1, not 0 — the real
  contribution is `d + 2`.

- **Deep isolated Expr tests**: `js/probe/probe_expr.ml` gained a `--deep`
  flag (default mode, the existing shallow three-way diff, is unchanged).
  `--deep` runs three depth-200,000 cases in-process and exits nonzero on
  any mismatch, never against native (confirmed directly — running the
  unmodified native route with `--deep` genuinely raises `Stack_overflow`,
  since `Eval.value` there is still ordinary recursion): `deep_value` (a
  200,000-deep `Value.add` chain against its closed-form sum), `deep_bool`
  (`Value.select` over `Bool.value_lt` on the same chain — `Eval.value` has
  no public `guard`, so `Select` is the only public route to the Bool
  entry point — against its closed-form branch choice), and `deep_index` (a
  200,000-deep `Index.add` chain, which `eval_index` — still ordinary
  recursion, out of scope — is expected to exhaust: a negative control, not
  a regression target). New per-route `stack_fault.ml` shims name each
  backend's own exhaustion shape (native/jsoo: plain `Stack_overflow`;
  Melange: `Js.Exn.Error` wrapping the underlying `RangeError`, detected via
  a `"call stack"` substring match on `Js.Exn.message`) — both are
  in-process catchable, matching the earlier deep-index measurement, so
  neither route needs the plan's separate-process fallback
  (`probe_expr_stack_fault.ml`, its dune stanzas, `stack-fault-runtest`).
  New `make expr_probe.deep-runtest` (jsoo + Melange only). Confirmed the
  `deep_index` negative control is non-vacuous: temporarily shrunk the chain
  to 10 adds, watched it correctly report `UNEXPECTED COMPLETION` instead of
  silently passing, reverted before committing.

**Not yet done:** wiring `dune build --profile landmarks` into a Makefile
target — this environment's own `dlllandmark_stubs.so` is independently
missing (confirmed pre-existing, unrelated to this work), so a new target
would fail here regardless of correctness; verified manually instead.
