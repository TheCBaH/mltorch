# Loop IR optimization

Status: implemented (2026-09-23). `Loop_opt.passes` runs unit-loop
elimination, index constant folding, range-aware index simplification,
proven-guard elimination (with the relational window-bounds proof), load
CSE, loop-invariant hoisting (re-implemented after a first attempt caught
its own bug) and loop collapsing over flat addressing; the emitter
peepholes are in `Loop_js` and `Js_build`. Strength reduction was built and
removed after measuring slower under V8.
Covers simple, exactness-preserving
rewrites of a lowered `Loop_program.t`, plus a few emitter-only peepholes in
`Loop_js`. Operator fusion, scheduling (tiling, vectorization) and the Phase 4
equivalence proofs are out of scope. They stay in the Kernel DSL design doc.

This is the "CSE/LICM on the Loop IR" item of that doc's Phase 3, and it answers
the `Num` simplifier open question in `loop_js_backend_design.md`: the
optimizations go on the Loop IR, not into the JS builder.

## Motivation: what the emitted JavaScript looks like today

The per-op goldens in `test/loop_ir/loop_js_{dense,pointwise}_test.ml` show the
lowering's output as is. Every inefficiency below appears in them:

| Pattern | Example golden | Cost |
|---|---|---|
| Unit-extent loops kept (`for (let i0 = 0; i0 < 1; i0++)`) | every op | nesting, index noise |
| Folded-away axes still summed into offsets: `(i0 + i1 + i2 + i3)` | `bmm`, `relu` | per-element adds |
| Full row-major offset recomputed per access: `3 * (4 * (4 * (...) + i3) + i4) + i5` | every op | 2–3 mul/add per axis per access |
| Six-deep nest over a dense elementwise tensor | `add`, `mul`, `relu`, `sigmoid`, `silu` | loop overhead; one flat loop would do |
| Identical load emitted twice: `b0[k] < 0 ? 0 : b0[k]` | `relu` | duplicate load and offset |
| Loop-invariant value recomputed: `a0[0] = Math.sqrt(1 / Math.sqrt(5))` per query row | `sdpa` | transcendental per row |
| Loop bounds recomputed: `Math.min(3, 8 + -1 + -1 * (i3 + -1) + 1)` | `conv2d` | per-iteration bound |
| Bounds check inside the innermost MAC, even though the clamped loop bounds already imply it | `conv2d` | per-element compare and branch |
| `Math.fround(x)` directly on a `Float32Array` store, which already rounds identically | every f32 op | redundant call |
| Copy temporary: `x4 = x3; ... b4[...] = Math.fround(x4)` | `sdpa` | noise |
| `float_max(a, b)` wrapping exactly `Math.max(a, b)` | `sdpa`, `softmax` | call (V8 probably inlines it) |

## Invariants every rewrite must keep

1. **Bitwise equality with the unoptimized program.** The Loop interpreter
   is the oracle for the JS backend, and `Kernel_eval` is the oracle for
   both. So floating-point operations may be neither reassociated nor
   re-rounded, nor rewritten with algebraic identities (`x + 0` and `x * 1`
   are not identities on binary64). Evaluating the *same* float expression
   fewer times (CSE, hoisting) is allowed. Changing its operands or their
   order is not.
2. **The same first failure.** `Loop_check` compares failures by kind *and*
   payload. `Fail_if` statements keep their relative order, and a
   rewrite may drop a `Fail_if` only when `Loop_range` proves it can never fire.
   A failure payload that carries a coordinate (`Load_out_of_range`,
   `coord_failure`) keeps the per-axis form, so addressing is flattened only
   for accesses whose guards were all removed.
3. **Index arithmetic stays in the proven domain.** `Loop_range.domain` is
   `[-2^31, 2^31 - 1]`. A fold is applied only when every intermediate it
   removes is proven in-domain. Removing a subexpression must not remove an
   overflow the original would have reported. This is the IR-level form of
   the `Idx.scale 0 a` rule in the JS builder. Aggregates are bounded with
   `Loop_range`'s saturating arithmetic, never by checking a folded result.
   No rewrite introduces a bitwise operator on an index.
4. **Loads never move above their guard, or out of a loop that may run
   zero times.** Under js_of_ocaml, a typed-array read out of range yields
   `undefined`. Natively, the interpreter's Bigarray access raises. A load
   is hoisted only from a loop whose extent `Loop_range` proves non-zero,
   and only past statements that contain no `Fail_if` guarding it.
5. **Effects stay put.** `Alloc`, `Charge_scan_update`,
   `Reserve_scan_state`/`Release_scan_state`, `Reset_meter`, `Mark` and
   `Store` are not reordered relative to each other. `Mark` is semantically
   empty but counted by tests, so its multiplicity is preserved too: a
   rewrite that removes a unit loop keeps the loop body's `Mark`s once.
6. **One program for every backend.** The interpreter, `Loop_js` and the
   jsoo executor consume the same optimized `Loop_program.t`. Emitter-only
   peepholes are limited to rewrites that change no value on any input
   (see below).

## Where it runs

A new `Loop_opt` in `lib/loop_ir` (one file per pass, `loop_opt_<pass>.ml`),
with `Loop_opt.run : Loop_program.t -> Loop_program.t`.

`Loop_lower.lower` is the single choke point. `Loop_node_program`,
`Loop_region_program` and `Loop_check` all reach programs through it, so it
applies `Loop_opt.run` by default. `Loop_lower.lower_unoptimized` keeps the
raw program for the differential and for inspecting a regression. The passes
are pure functions of the program and use no per-backend knowledge.

Pass order carries behavior: each pass exposes facts the next one uses. It is
therefore fixed in `Loop_opt.passes` and documented there. It is deliberately
not alphabetical. The sections below are numbered by design topic; the run
order is 1, 2, 2b, 3, 5 (CSE, which compares per-axis coordinates), 4
(hoisting, which can leave a loop perfectly nested), 6 (collapsing, which
flattens those coordinates).

1. **Unit-loop elimination** (`loop_opt_unit_loops.ml`, implemented). A `For`
   whose bounds are literal `Const`s exactly one apart becomes its body with
   `Var v := lo` substituted, via the shared `Loop_index_map` traversal. This
   must run first, because it turns `(i0 + i1 + i2 + i3)` into constants for
   folding. Scoped down from a full `Loop_range`-extent proof to a literal
   `Const`/`Const` check (the extent taken in `int64`: a 32-bit `hi - lo`
   wraps to `1` for the empty loop `[2^31 - 1, -2^31)`): every non-window loop's bounds are already literal
   `Const`s straight out of the axis nest, and a window loop's clamped
   bounds are deliberately left alone here (see "Relational bounds-check
   elimination" below).
2. **Index constant folding and simplification** (`loop_opt_fold.ml`,
   implemented, scoped down). Folds `Const` arithmetic, `Add (Const 0, a)`,
   `Scale (1, a)` and `Scale (k, Scale (m, a))` for `k <> 0` (with `k = 0` the
   combined `0 * a` drops the intermediate `m * a`'s overflow, the same hazard
   as `Scale (0, a) -> Const 0`), each accepted only when the
   new constant is proven, via `Loop_range`'s saturating `int64` arithmetic,
   to stay in `Loop_range.domain` — never by inspecting a value already
   computed with the host's own (32-bit-under-jsoo) `int`. **`Min`/`Max`/
   `Clamp_low` on proven ranges is not implemented**: it needs a
   `Loop_range.Env.t` of enclosing loop variables this context-free rewrite
   doesn't build; the next pass does it.
2b. **Range-aware index simplification** (`loop_opt_simplify.ml`,
   implemented). Walks the program top-down rebuilding the scope
   (`Loop_opt_scope`, shared with pass 3) and rewrites an index node, bottom
   up, only when `Loop_range.proven` holds for both it and its replacement
   and the replacement is cheaper: a point range becomes its constant
   (`floor (i / 64)` for `i < 64`); an ordered `Max`/`Min`/`Clamp_low` becomes
   its winning operand (`max (0, 2 * i)` is `2 * i`, `max (0, -2 * i)` is
   `0`); a division sheds the multiples of its divisor exactly
   (`floor ((d * q + r) / d) = q + floor (r / d)`, so `floor (8 * i / 4)` is
   `2 * i`); and an `Add`/`Scale` tree is re-summed through `Loop_linear` (an
   exact linear form over opaque atoms) when that is cheaper, which cancels
   `floor (i / 64) - floor (i / 64)`. The proof on the original is what keeps
   invariant 3: no removed `Add`/`Scale` could have overflowed. It turns the
   pools' and `adaptive_avg_pool2d`'s window bounds into plain affine ones
   (`avg_pool2d`'s become the constant `[0, 2)`), which is what later lets
   loop collapsing see them.
3. **Proven-guard elimination** (`loop_opt_guards.ml`, implemented). Walks
   the program top-down, rebuilding the `Loop_range.Env.t` of enclosing loop
   variables' and index temporaries' ranges the lowering had while
   constructing the program (and discarded once it finished) — recording an
   index temporary only when exactly one statement assigns it, since an
   argmax's `best_i` and a max-pool's `best_ix` are seeded and then reassigned
   in their loop — and drops a
   `Fail_if (Index_overflows _ | Out_of_range _)` — or an `Or` of only those
   two shapes, which is what a multi-axis bounds check actually emits — that
   it proves can never fire. In practice this pass's own (necessarily
   coarser, since it's a re-analysis of an already-lowered program rather
   than the live env lowering itself had) interval proof never finds
   anything the lowering hadn't already elided for the current op corpus; its
   relational proof (below) is what removes `conv2d`'s and `reshape`'s
   checks.
4. **Loop-invariant hoisting** (`loop_opt_hoist.ml`, implemented after a
   reverted first attempt; runs after CSE, before collapsing, since a loop
   emptied of its invariant statements may become perfectly nested). The
   first attempt checked that an *expression* was invariant and hoisted an
   accumulator's reset (`x0 = 0` before `x0 = x0 + ...`) out of the loop that
   re-initializes it per iteration; the differential caught it. What makes a
   *statement* invariant is where its target is read and written, so an
   `Assign` of a float temporary or an `Array_set` of a constant cell at a
   loop body's top level moves before the loop only when: the loop runs at
   least once (constant `lo < hi`, invariant 4); its value is closed
   (constants only — no load, temporary, array read, index, or anything that
   can fail); no other statement in the loop writes the target (another
   assignment of the temporary, an `Array_set` whose index range can reach
   the cell, an `Alloc` of the array); and nothing before it in the body
   reads the target (on the first iteration that read would see the pre-loop
   value). Then every read in the loop sees what it saw before, and the
   target holds the same value after it. Bottom-up, so `sdpa`'s
   `a0[0] = Math.sqrt(1 / Math.sqrt(5))` leaves both enclosing loops. The
   mutation proof replays the original bug (the write check off).

   A loop **bound** is not hoisted in the IR: the interpreter already
   evaluates `hi` once on entry, but a JavaScript `for` test runs per
   iteration, so `Loop_js` binds any `hi` other than a literal or a loop
   variable to `const nK` before the loop (`conv2d`'s `Math.min(3, 9 - i0)`).
   That is also the faithful form: a bound reading an index temporary the
   body reassigns would otherwise diverge from the interpreter.
5. **Load CSE** (`loop_opt_cse.ml`, implemented, scoped down). Within one
   straight-line `Store`/`Assign`'s own float value expression, an identical
   `Load (b, coord)` of a buffer no `Store` in the program targets is shared
   through one float temp, provided at least one occurrence is evaluated
   unconditionally: a read reached only through a `Select` branch or the
   right of an `Or` may be in range only because of that condition, and
   computing it before the statement would make it unconditional
   (invariant 4). **Scoped to a single statement's own expression
   tree, not also "across a statement's expression tree" more broadly**: a
   single pure-expression evaluation has nothing running between two
   occurrences of the same load, so sharing them needs no reasoning about
   staleness at all — sidestepping the exact hazard that sank pass 4. `relu`'s
   double load (plus `gelu`, `silu`, `layer_norm`'s variance term) is the
   target actually hit.
6. **Loop collapsing** (`loop_opt_collapse.ml`, implemented; runs last,
   after CSE, which compares per-axis coordinates). A perfectly nested pair
   `for v1 in [0, n1): for v2 in [0, n2): B` becomes one loop over
   `[0, n1 * n2)` when `v1` and `v2` reach `B` only through access offsets,
   and in each access's dense offset (`Loop_linear` of `Vec6.offset`'s
   linearization, or an already-flat offset) `v1`'s coefficient is `n2`
   times `v2`'s: then `c1 * v1 + c2 * v2 = c2 * g` with `g = n2 * v1 + v2`,
   and the iteration order is the nest's. Those accesses become flat, at
   an offset proven overflow-free where it is evaluated. Any other mention of
   `v1`/`v2` (a guard, a payload, an index temporary, `Value_of_index`, an
   inner loop's bound) refuses the pair, which is what keeps invariants 2
   and 5; a broadcast axis (`c1 = 0` beside `c2 <> 0`) or a transposed one
   fails the ratio; a per-channel quantized buffer is never flattened. Pairs
   collapse bottom-up, so the elementwise ops (`add`, `mul`, `relu`, `gelu`,
   `sigmoid`, `silu`) become one loop, and `layer_norm`, `softmax` and
   `batch_norm` collapse their outer axes around the reduced or broadcast
   one. The mutation — collapsing across a broadcast axis — reads the
   broadcast buffer past its end.
7. **Strength reduction of offsets — built, measured, removed.** In
   `for v in [lo, hi)`, an access whose dense offset is `s * v + rest` (`rest`
   fixed for the loop) read flat through an index temporary set before the
   loop and advanced by `s` after each iteration, with every value the
   temporary takes (the one after the last iteration included) proven in the
   domain; it passed the differential, its mutation proof, and the pt2
   tier-2 run. It was a net loss under V8: timing each walked op's generated
   JavaScript under node (best of 7, warm), `conv2d` took 59.7 µs with it
   and 53.6 µs without, and `linear`, `bmm`, `permute`, `softmax` and the
   pools were all slower with it too. V8 already reduces affine index
   arithmetic on a loop variable itself; a loop-carried offset temporary
   hides that induction variable from it. Revisit only for a backend that
   does no such analysis.

### IR extension: flat addressing — implemented

Offsets are linearized in the backends (`Loop_js.offset`, and `Vec6.offset`
in the interpreter) from a six-axis `Loop_index.coord`. Loop collapsing
needs the offset in the IR, so the IR has flat forms beside the per-axis
ones: `Loop_expr.Load_flat`/`Load_i64_flat` and `Loop_stmt.Store_flat`,
each over a `Loop_index.t` that is the dense row-major offset. They are
separate constructors rather than one `Coord | Flat` address type so every
per-axis path (and the many hand-built test programs) is untouched, and a
flat access is a visible arm in every exhaustive match. `Loop_js` indexes the
typed array directly; the interpreter peels the offset back into the
coordinate `Vec6.offset` would have linearized (divisions by each extent,
innermost first — no product of extents is formed, so none can wrap) and
reads and writes through the same path as a per-axis access, so an offset
outside the buffer is the same `Invalid_argument` defect as an unchecked
coordinate. `Loop_pp` prints `b0[@e]`. A per-channel quantized buffer is
never flat: its decode reads the C component (`Loop_js` refuses one as a
defect). Invariant 2 is kept because a failure payload lives on the
`Fail_if`, not on the access, so flattening an access never changes what a
failure reports.

### Relational bounds-check elimination (padding windows) — implemented

`conv2d`'s window loop is `for i4 in [max(0, 1 - i0), min(3, 9 - i0))`, and
the check `i0 - 1 + i4 ∈ [0, 8)` inside it follows from those bounds. An
interval analysis cannot see the correlation, so it is proven relationally,
inside pass 3: each loop variable's bounds become facts `d * k >= a` (one per
operand of a `max`/`clamp_low` lower bound; `k >= ceil (x / d) + c` gives
`d * k >= x + d * c`) and `d * k <= a` (one per operand of a `min` upper
bound, from `k <= hi - 1`; `floor (x / d) + c` likewise). A bound on the
guarded coordinate's `Loop_linear` form is proven by replacing `c * k`
(`c = m * d`) with `m * a` on the side the sign of `m` picks and checking
the rest by interval, up to two substitutions deep. Facts mention only loop
variables, which are immutable in the loop body, never index temporaries.
The same pass uses `Loop_linear.range`'s one relational fact,
`m * (x - d * floor (x / d)) ∈ m * [0, d - 1]`, which discharges an
unflattened `reshape`'s per-element check.

This deliberately does **not** use `Expr.Index.Assume_position`, which the
earlier plan proposed threading into `load_guard`: that node records an
op author's claim, not a proof (`Expr.Fold.assume_sites` exists to count
them), and it also wraps `repeat`/`resize`/`reshape` coordinates. Trusting
it would let a wrong claim drop a check `Kernel_eval` still makes, breaking
invariant 2; the loop's own bounds prove the same fact without trusting
anyone, and stay local to `lib/loop_ir`. A mutation that claims `k <= hi - 2`
(one tighter than the loop gives) turns the differential red on a window one
wider than its clamp.

## Emitter-only peepholes (`Loop_js`)

These are allowed because they change no value on any input, so the
interpreter needs no counterpart:

- **Omit `Math.fround` directly under a `Float32Array` store.**
  (Implemented.) ToFloat32 on store is the same round-to-nearest-even.
- **Index subtraction** (implemented, in `Js_build.Idx`): `a + -k * b`
  prints as `a - k * b`, `a + -k` as `a - k`, and `-1 * a` as `-a`. Every
  index is an integer within 2^31, exact in a `Number`, so both forms are
  equal (`-0` included); the overflow check is on the IR's `Scale` node, not
  the text. `Loop_linear` emits a re-summed form with its positive terms
  first so this reads naturally (`9 - i`, not `-1 * i + 9`).
- **`Num.of_idx` of a literal** (implemented): `2 + 0` is `2` — the `+ 0`
  only exists to turn an index `-0` into `+0`, and a literal other than `-0`
  needs no such mapping.
- **Emit `float_max` as `Math.max`** when the helper is exactly
  `Math.max`. (Implemented.) The helper stays for `Pool_better` and every
  other non-trivial comparison — the runtime prelude only emits a helper
  actually reachable from the program, so once nothing calls `float_max`
  its whole function definition drops out too, not just the call site.
- **Copy propagation of a temp read once** (`x4 = x3` in `sdpa`) and
  **dropping a temp's `= 0` initializer** — declined. The copy comes from a
  `Select` whose branch needs statements; a register allocator makes it free,
  so removing it is text only. The initializer is not dead weight: it types
  the slot as a `Number` from function entry (V8 never sees `undefined`
  there), and a read before any assignment — a defect the interpreter
  reports — would otherwise compute with `undefined` silently.

These change the `loop_js_test` text goldens. The `Math.fround` and
`float_max` peepholes live in `Loop_js`, outside Melange's reach; the index
subtraction and `of_idx` ones live in `Js_build`, which the Melange mirror
copies, so `make melange.runtest` covers them too.

## Out of scope, and why

- **Reassociating or splitting reductions** (flattening `conv2d`'s nested
  `x2 → x1 → x0` accumulators, pairwise sums, SIMD lanes). This breaks
  invariant 1.
- **Caching a value across sibling loops** (`sdpa`'s second
  `Math.exp(a0[1 + i] - a0[5])`). It is exact, but it is a memory/shape trade,
  so it belongs with fusion and cost modeling.
- **Unrolling and tiling.** They need a cost model and footprint legality,
  which is Phase 5.
- **`|0` / `>>>` integer hints.** Forbidden: no bitwise operator on an index.

## Verification

- **Optimized vs raw differential**
  (`test/loop_ir/loop_opt_differential_test.ml`). For every walked op
  (`Native_op_walk.all_walks`, the same sweep as `loop_node_check_test`),
  run the raw and optimized programs through `Loop_interp` and compare
  bitwise, with failures compared by kind and payload through
  `Loop_check.compare`'s semantics.
- **Existing gates unchanged:** `loop_node_check_test`/`loop_sweep_test`
  (optimized program vs `Kernel_eval`), the `loop-js-gate`,
  `make jsoo.inline-runtest`, and (for stages 4/6/9, not yet implemented;
  spot-checked anyway for the implemented ones) `make loop_js.node.pt2.runtest`.
- **Mutation proof per pass**, one per implemented pass, in
  `loop_opt_differential_test.ml` (stages 0-3) or the pass's own
  `loop_opt_<pass>_test.ml` (stage 8): a deliberately wrong variant must
  turn the differential red (or, once, raise the interpreter's own
  unchecked-access `Invalid_argument` directly — still red, just via a
  more direct channel). This is the check that the differential is not
  vacuous, and it caught a real bug once (the reverted hoisting pass,
  before any golden was promoted).
- **Goldens.** Each pass lands with its text diff in the per-op goldens
  and `loop_js_test`. These are the review surface.
- **Performance.** `make loop_js.bench` times a hand-lowered single loop
  that no pass changes, so it does not measure these passes. Instead each
  walked op's generated JavaScript at its walk's initial config was timed
  under node 20 (warm, best of 7, the raw lowering against the pipeline):

  | op | raw | optimized | |
  |---|---|---|---|
  | `reshape` | 1604 ns | 342 ns | 4.7x |
  | `adaptive_avg_pool2d` | 1020 ns | 337 ns | 3.0x |
  | `avg_pool2d` | 887 ns | 349 ns | 2.5x |
  | `max_pool2d` | 952 ns | 468 ns | 2.0x |
  | `conv2d` | 82.2 µs | 53.6 µs | 1.5x |
  | `relu` | 79 ns | 54 ns | 1.45x |
  | `add` | 81 ns | 72 ns | 1.1x |
  | `bmm`, `batch_norm`, `linear`, `layer_norm`, `softmax`, `permute`, `sdpa` | | | 0.98–1.1x |

  The wins are the bounds checks and divisions the relational proof and
  the simplifier remove (`reshape`, the pools, `conv2d`) and the collapsed
  elementwise loops; the rest are within noise. Strength reduction was cut
  on this measurement (pass 7).

## Open questions

- ~~Should `Loop_lower.lower` optimize by default, or should each caller
  opt in?~~ **Resolved: default-on.** `Loop_lower.lower` applies
  `Loop_opt.run`; `Loop_lower.lower_unoptimized` is the raw form, kept for
  the differential and for inspecting a regression.
- Does flat addressing need a quantized-channel variant? Per-channel
  quantized loads read the `C` coordinate for the scale. A collapsed loop
  loses it, so such buffers are excluded from collapsing for now.
- Should the relational window facts move into `Loop_range` as a small
  difference-bound domain? Only if a second producer of clamped bounds
  appears; for now they live in pass 3.
- Should hoisting reach non-closed values (a load of a never-stored buffer at
  an invariant coordinate, an invariant index expression into a temp)? That
  needs the same read/write analysis plus invariant 4's guard ordering; no
  current golden needs it.
