# PT2 inference-time profiling

Investigation into why the GitHub Actions "inference" step (`make inference`,
sweeping all 22 `PT2_MODELS_ALL`) took 20+ minutes, and what was actually
worth changing about it. Numbers below are from this devcontainer sandbox
(24 cores, no AVX2 — see below); absolute times differ on other machines,
but the *ratios* (which lever helped, and by how much) are the useful part.

## The `.pt2`/sample-loading suspicion did not hold

The initial suspicion was that `.pt2` archive loading or per-image sample
loading dominated inference time. `test/interp_run.ml` now times both
separately (via the `timed` helper, printed to stderr) instead of only
timing `Interp.run`. Measured on resnet18 (10 images):

| stage | time |
|---|---|
| `Pt2_archive.open_pt2` (once) | ~30-40 ms |
| `Pt2_archive.load_pt` (per image) | ~1.5-2 ms |
| `Interp.run` (per image, the ATen forward pass) | ~265-520 ms |

Loading (open + all 10 samples) is ~1% of total wall time; **compute
(`Interp.run`) is ~99%.** `Pt2_archive.open_pt2` was already called exactly
once outside the per-image loop before this investigation, so "N inferences,
one `.pt2` load" already held structurally — nothing to fix there.

The 20+ minute CI cost is compute, and it scales badly across the 22
models — e.g. efficientnet_b7 is ~270 ms/image at resnet18's low end vs
~1.6-1.9 s/image for the largest models tested (pre-`-O3`, several were
10-40x resnet18's per-image cost). Summed across 22 models this plausibly
accounted for the whole 20 minutes on its own.

## Root causes read from `lib/aten/build_archive.sh`

- The whole ATen core/kernel archive was compiled `-Os` (optimize for size,
  ~8 MB archive — see `.ai/aten_core_build.md`), not for speed.
- Every kernel was compiled `CPU_CAPABILITY=DEFAULT` (scalar only); no
  AVX2/AVX512 variant was built.
- `linear`/`addmm`/`mm` and `conv2d`'s im2col+gemm path bottom out in
  `BlasKernel.cpp`'s **reference (unblocked, unvectorized) gemm** — there is
  no external BLAS.
- `AT_PARALLEL_NATIVE=1` is compiled in (cpuinfo + `ParallelNative` linked),
  but empirically **`OMP_NUM_THREADS`/`MKL_NUM_THREADS` have zero effect on
  wall time** (tested 1/8/24 on resnet18: wall time and user≈real ratio
  unchanged in every case). The reference gemm in `BlasKernel.cpp` simply
  never calls `at::parallel_for` — this isn't a thread-pool-sizing config
  gap, it's architectural. Fixing it would mean editing the kernel itself
  (out of scope: the goal here was explicitly not to optimize the PyTorch
  kernel implementation or add dependencies like a real BLAS).

## What was changed

All three of these are build-flag/process-structure changes in files we own
(`lib/aten/build_archive.sh`, `Makefile`, CI YAML) — none touch
`modules/pytorch` source.

1. **`-Os` → `-O3`** — cut per-image time by **~44-45%** (resnet18:
   5.0s→2.79s total over 10 images; resnet152: 30.1s→16.45s). An AVX2
   build-time capability probe was also tried alongside this (see "Tried,
   reverted: AVX2 build-time capability probe" below) but doesn't ship —
   `-O3` is the change that actually landed.
2. **Parallelized `make inference` across models.** Every
   `inference.<model>` run is fully independent (own directory, own
   download, own process), but the sweep ran strictly serially — plain
   `make` doesn't parallelize a target's prerequisites without `-j`, and
   nothing passed `-j`. Restructured so `make -j` is safe:
   - `test/interp_run.exe` is now built once via a plain `dune build`
     (`$(PT2_RUN): opam exec -- dune build test/interp_run.exe`), never
     `dune exec` — concurrent `dune exec`/`dune build` invocations contend
     on `_build` locks (see `.ai/worktree_setup.md`), so per-model jobs must
     not each shell out to dune themselves.
   - `inference.%` uses a **pattern-specific variable**
     (`inference.%: PT2_MODEL = $*`) to resolve `$(PT2_INFER_ARGS)`/
     `$(PT2_MODEL_DIR)` per model without a nested `$(MAKE)`. `PT2_MODEL_DIR`/
     `PT2_ZIP`/`PT2_URL` had to change from `:=` (immediate expansion, frozen
     at parse time to the top-of-file default) to `=` (lazy, re-expanded at
     use time) for this to actually take effect.
   - One subtlety found empirically: a pattern-specific variable does *not*
     propagate into a separately-invoked prerequisite target's own recipe
     the way it propagates into the defining target's own recipe (confirmed
     via `make -n`: `pt2.download`'s recipe saw a blank `PT2_MODEL_DIR` when
     reached only as a plain prerequisite of `inference.%`). Fixed by having
     `inference.%`'s own recipe invoke `pt2.download` via a recursive
     `$(MAKE) pt2.download PT2_MODEL=$*` (explicit command-line override,
     the same mechanism the original code already used) — safe under `-j`
     since it's just curl/unzip, no dune involved.
   - CI runs `make -j"$(nproc)" inference` (queried inside the devcontainer).
     Verified with a mixed 4-model `-j4` run: all outputs matched
     `results.json` (no `PT2_MODEL` cross-talk between concurrent jobs).
   - Memory caveat: peak RSS at the high end is substantial even after
     `-O3` — efficientnet_b7 ~5.6 GB, efficientnet_v2_l ~3.7 GB, vit_l_32
     ~2.85 GB, resnet152 ~0.9 GB. This is memory-, not CPU-, bound at the top
     end; if a future runner's core count ever outscales its RAM (GH's
     standard tiers scale both together, ~4 GB/core), this may need an
     explicit `-j` cap rather than raw `nproc`.
3. **Fast per-push latency benchmark**: `make benchmark.inference` runs
   resnet18 (the smallest/fastest model, already `PT2_MODEL`'s default)
   once under `/usr/bin/time -v` (wall/user/sys + max RSS), wired into CI
   right after "pt2 load test" — so a latency regression on the smallest
   model shows up without waiting on the full 22-model sweep.

## Tried, no effect: flambda / OCaml-side optimization

As an alternate angle, tested whether the OCaml compiler itself (rather than
the C++ kernel archive) was a contributor: built a local, worktree-scoped
opam switch (`ocaml-variants.4.14.3+options` + `ocaml-option-flambda`,
project deps reinstalled) and rebuilt `test/interp_run.exe` against it in a
separate `--build-dir` (so it wouldn't disturb the default switch's `_build`).
Result: **no measurable difference** — resnet18 ~2.77s (vs ~2.79s baseline),
resnet152 ~16.39s (vs ~16.45s baseline), both within run-to-run noise. This
is the expected outcome given the earlier finding: `Interp.run` is dominated
by the C++ reference-gemm kernel, which flambda can't touch — it only
optimizes the OCaml glue/dispatch code around it, which was already a
negligible fraction of total time. (Aside: this sandbox is `aarch64`, where
AVX2 doesn't exist at all, confirmed via `/proc/cpuinfo` — see the AVX2
section below for why that mattered more than expected.)
Switch and build dir were removed after the experiment; no code changes came
of this.

## Tried, reverted: AVX2 build-time capability probe

`lib/aten/build_archive.sh`'s `-O3` change (above) originally shipped
alongside a build-time AVX2 probe (`clang++ -march=native -dM -E - | grep
__AVX2__`, setting `CAP=(-DCPU_CAPABILITY=AVX2 -DCPU_CAPABILITY_AVX2 -mavx2
-mfma)` when the build machine supports it, `DEFAULT` otherwise) — reasoned
to need no PyTorch-style dual-compile/runtime-dispatch since build machine ==
run machine here. This sandbox is `aarch64` (no AVX2 at all), so the probe
always took the `DEFAULT` branch here and the AVX2 code path was never
actually exercised or built — an untested assumption that turned out wrong.

**On a real x86 machine** (AVX2-capable, so the probe correctly picks the
`AVX2` branch), the build fails:

```
In file included from .../aten/src/ATen/cpu/vec/vec256/vec256.h:16:
.../aten/src/ATen/cpu/vec/vec256/vec256_float.h:10:10: fatal error: 'sleef.h' file not found
   10 | #include <sleef.h>
```

`vec256_float.h` (part of ATen's AVX2 `Vectorized<float>` implementation)
unconditionally `#include`s SLEEF (vectorized transcendental math) — this is
a *header-level* include, so it fails to even parse under `CPU_CAPABILITY=
AVX2` regardless of whether any op actually calls a transcendental. SLEEF is
a real PyTorch `third_party` submodule (`github.com/shibatch/sleef`,
`modules/pytorch/.gitmodules`) that this project doesn't vendor or build —
integrating it would need its own CMake build step (`build_archive.sh`
already does this for `cpuinfo`, so there's precedent, but it's a genuine new
dependency, not a flag). That directly contradicts the "not... increase size
of dependencies" goal this whole investigation started from, so: **reverted
the probe, `CAP` is unconditionally `DEFAULT` again.** `-O3` stays (no
dependency, no build-machine-specific behavior, and it's the change that
measurably helped).

Lesson: "the build machine is the run machine, so no dual-compile needed"
was true for the *compiler flags*, but AVX2 also gates which *headers* get
pulled in, which isn't something a same-machine assumption fixes — that part
needed testing on real AVX2 hardware to catch, not just reasoning about the
flag semantics.

## Native (pure-OCaml) engine's conv2d: a real but partial flambda win

Separate from the ATen-backed path above: `lib/native` (see
`.ai/native_compute_design.md`) is a pure-OCaml inference engine, unrelated to
`lib/aten`/`lib/interp`. Its `conv2d` (`lib/native/ops/conv.ml`) is excluded
from the full ATen-vs-native comparison sweep
(`test/pt2_node_bridge_cram.t`) and from `pt2_op_native_walk_cram.t`'s
real-scale walk because it takes **~27s even on resnet18's smallest/first
conv node** (`test/data/resnet18/000_convolution_default.json`:
`input=f32[1,3,224,224]`, `weight=f32[64,3,7,7]`, stride 2, pad 3 →
`[1,64,112,112]`, ~118M multiply-adds). Added a permanent regression
benchmark for this: `make benchmark.native_conv2d` (runs
`aten_spec_verify.exe` — default mode, ATen-vs-native — on that one fixture
under `/usr/bin/time -v`; ATen's own share of the time is negligible here).

Investigated with the same flambda-switch technique as above, since this
path (unlike the ATen one) is genuinely OCaml compute, not a C wrapper:

| build | time |
|---|---|
| current switch (no flambda, dune `dev` profile) | ~27.4s (baseline) |
| flambda switch, dune `dev` profile | ~25.7s (~6% faster) |
| flambda switch, dune `--profile release` | ~20.1s (~27% faster) |

flambda only pays off once paired with `--profile release` (dune's `dev`
profile doesn't pass the `-O2`/`-O3`/inlining flags flambda needs even when
the compiler supports it) — but even then it's a **partial** win, not the
order-of-magnitude a pure boxing fix would give. Read through the hot path to
see why:

- `Conv2d.Compute(S : SEMANTICS).pixel` is called once per output element
  (802,816 times for this fixture) and does a 3-deep `S.sum` nest (input
  channels × kernel h × kernel w, ~147 iterations here) — ~118M innermost
  iterations total.
- Each innermost iteration builds a fresh `x_idx`/`w_idx` closure (they close
  over `kh`/`kw`/`ic`/`oc`/`out`, defined inside the innermost `sum` body —
  see `conv.ml`'s `pixel`), then calls `S.load` twice (x and weight).
- `Direct.load` = `Tensor.read_at`, which builds a fresh `Vec6.coord` — a
  6-element `int array` — from the 6 per-axis index-closure calls, then
  `Tensor.read` re-derives bounds-check (`Vec6.in_bounds`) and linear offset
  (`Vec6.offset`) by `List.fold_left`/`List.for_all` over the 6-entry
  `Axis.all` list, each with its own closure capturing the coord/shape.
- So each of the ~118M inner iterations does on the order of 2 closure
  allocations (`x_idx`/`w_idx`) + 2 more `Vec6.coord` array allocations + the
  `in_bounds`/`offset` closures and list traversals for both reads — call it
  roughly half a billion+ small allocations for this one (smallest!) conv
  node. `Dim.t` itself is `private int` (so no boxing there — confirmed by
  reading `dim.ml`), but the *combinator/array* layer around it isn't free.

flambda's inliner clearly manages to remove *some* of this (the ~27%
release-profile win is real and reproducible), but not enough to fully
collapse the `SEMANTICS` functor + `Vec6`/`Tensor` indirection into a tight,
allocation-free loop over concrete floats/ints — likely because that would
require flambda to see through a functor application in one module
(`Eval_op.Make (Direct)`), invoked through another (`Schedule.evaluate`,
`Tensor.read_at`, `Vec6`), specializing every layer at once. That's the kind
of whole-program specialization flambda *can* sometimes do but isn't
guaranteed to, and isn't something you tune with a flag.

### Confirmed with memtrace: allocation hotspots, ranked

The above was inferred from reading the source; confirmed it directly by
profiling. `perf` isn't available in this sandbox and neither `memtrace`/
`memtrace_viewer` were installed, so: added a temporary
`Memtrace.trace_if_requested ()` call to `bin/aten_spec_verify.ml` + `memtrace`
to its `bin/dune` libraries, ran `MEMTRACE=trace.ctf
_build/default/bin/aten_spec_verify.exe test/data/resnet18/000_convolution_default.json`,
then wrote a ~40-line standalone script against `Memtrace.Trace.Reader` (no
`memtrace-hotspot`/`memtrace_viewer` available for this OCaml version) to
tally sampled allocation weight (`length * nsamples`) by allocation-site
source location. Both the instrumentation and the script were reverted after
— this section is the record of what they found, not a permanent addition.

Result (sampled-word share of total, five sites account for ~91%):

| share | site |
|---|---|
| 26.3% | `vec6.ml:27` `Vec6.coord` — the 6-element array built per tensor read |
| 18.9% | `vec6.ml:43` `Vec6.offset`'s internal `List.fold_left` closure |
| 16.9% | `conv.ml:367` `pixel`'s `x_idx` closure (rebuilt every innermost iteration) |
| 13.8% | `conv.ml:374` `pixel`'s `w_idx` closure (same) |
| 9.8% | `vec6.ml:48` `Vec6.in_bounds`'s internal `List.for_all` closure |
| 6.0% | `conv.ml:360` the innermost reduction-step closure itself |
| ~5% | long tail: `Direct.sum`/`Direct.mul` (float boxing at the module boundary), `Bigarray.Array1.create` (real tensor allocations), `Payload.get_float`, `Conv2d.validate_channels` (see below) |

This is exactly the five-site chain traced from source above, now with hard
numbers: `Vec6.coord` + its `offset`/`in_bounds` closures + the two per-axis
index closures are ~86% of all sampled allocation on their own.

One additional, independent finding from the long tail, unrelated to boxing:
`Conv2d.validate_channels` (conv.ml, channel/group-divisibility checks) shows
up because `pixel` calls it on **every one of the 802,816 output pixels**,
even though its result depends only on `p`/`weight_shape` — constant for the
whole op. Its allocation share is small (it mostly succeeds and returns
cheaply), but it's still 802,816 redundant calls doing real work each time.
Hoisting it to run once per op invocation (not once per pixel) is a small,
safe, no-design-risk cleanup independent of the allocation story — flagging
it here rather than fixing it inline, since it's still a change to
`lib/native/ops/conv.ml`'s hot path and this investigation was scoped to
profiling, not patching.

Two real options were on the table (neither a flag flip): switch the whole
project to a flambda variant + `--profile release` (~27%, but an infra
decision affecting everyone, not scoped to conv2d), or restructure the
per-element hot path directly. Went with the latter, one contained step at a
time, each validated against the existing native/aten_spec test suites and
measured with `benchmark.native_conv2d` before committing.

## Applied fix, step 1: unroll `Vec6.offset`/`Vec6.in_bounds`

Rewrote both (`lib/native/vec6.ml`) from `List.fold_left`/`List.for_all` over
`Axis.all` (a fresh closure per call, capturing the shape/coord) to
hand-unrolled straight-line code over the 6 fixed array indices — no API
change, no change to `Tensor`/`SEMANTICS`, scoped entirely to `vec6.ml`.
Validated with `dune build @test/native/runtest` (all native-engine inline
tests) plus `aten_spec_test`/`aten_spec_run_test`/`native_walk_test`/
`native_bridge_test` — all clean, no diffs.

Measured on the same fixture (`benchmark.native_conv2d`'s target):

| | time |
|---|---|
| before (baseline, this session) | ~25.5s |
| after step 1 | ~14.0s |

**~45% faster** — bigger than the ~28.7% allocation share
(`offset`+`in_bounds`'s closures) predicted, since removing the closures also
removes indirect-call overhead and GC-scan pressure beyond just the
allocation bookkeeping itself.

## Applied fix, step 2: stop allocating `Vec6.coord` in `Tensor.read_at`

Rewrote `Tensor.read_at` (`lib/native/tensor.ml`) to compute the bounds check
and linear offset directly from `idx`'s 6 results, with no intermediate
`Vec6.coord` array — the in-bounds case (the overwhelming majority) never
allocates; the rare out-of-range case still falls back to building the coord
and calling the original `read`, so its exact `Invalid_argument` message is
unchanged (`test/native/tensor_test.ml`'s expect tests assert that literal
string and passed unmodified). No `SEMANTICS`/`load` signature change.

First attempt regressed instead of improving: calling `Vec6.get t.shape axis`
once per axis for the bounds check *and again* for the offset sum (12 calls
total) measured **slower** (~16.0s) than step 1 alone (~14.0s) — `Vec6.get`
is a real cross-module function call here (no flambda to inline it away), so
12 of them cost more than the single array allocation they were replacing.
Fixed by reading each shape extent into a local exactly once (6 calls) and
reusing it for both the check and the offset arithmetic (inlined directly,
bypassing `Dim.lin`'s wrapper — the result only ever feeds
`Payload.get_float`'s raw `~i:int` anyway). Recorded here because it's a
real, measured lesson, not just a hypothesis: **"remove an allocation" isn't
automatically faster in non-flambda OCaml if the replacement costs more
cross-module calls than the allocation itself.**

Validated the same way as step 1 (all native/aten_spec test suites, no
diffs — including the exact-error-message expect test above).

| | time |
|---|---|
| after step 1 | ~14.0s |
| step 2, first attempt (12 `Vec6.get` calls) | ~16.0s (regression) |
| step 2, fixed (6 `Vec6.get` calls) | ~11.7s |

Cumulative: **~25.5s → ~11.7s, ~54% faster than the original baseline.**

## Re-profiled after steps 1+2: what's left, and why step 3 needs care

Re-ran the same memtrace procedure (temporary instrumentation, reverted
after) on the post-step-2 build. Total sampled allocation roughly halved
(47019 → 24152 words), and the remaining hotspots are now dominated by what
was already flagged as the harder part:

| share | site |
|---|---|
| 33.6% | `conv.ml:367` `pixel`'s `x_idx` closure |
| 27.2% | `conv.ml:374` `pixel`'s `w_idx` closure |
| 13.6% | `conv.ml:360` the innermost reduction-step closure |
| 9.6% | `vec6.ml:27` `Vec6.coord` — now only from `Schedule.evaluate`'s per-*output-pixel* `Vec6.iter` (802,816 calls), not per-MAC; steps 1+2 already removed the ~236M per-MAC calls that used to share this same site |
| ~16% | long tail (`Direct.sum`/`Direct.mul` float boxing, `Bigarray.Array1.create`, `Payload.get_float`, `validate_channels`) |

`x_idx`+`w_idx`+the reduction closure are now ~74% of remaining allocation —
the same three sites flagged before steps 1/2, just a larger share of a
smaller total.

The natural next step would be to stop reallocating `x_idx`/`w_idx` on every
innermost iteration — e.g. hoist them out of the `kw` loop and thread
`kh`/`kw`/`ic` through mutable refs instead of fresh closures each time.
Checked whether this is actually safe before attempting it: `Conv2d.Compute`
is a functor over `SEMANTICS`, shared by both `Direct` (`t = float`, eager)
and `Symbolic` (`t = Expr.t`, confirmed by reading `symbolic.ml` — `sum`/`mul`/
etc. build an AST node, they don't execute anything immediately). A
ref-mutation trick is sound for `Direct`'s eager, single-threaded evaluation
order, but would be **wrong** for `Symbolic`: building an expression tree
doesn't happen in the same order refs would be mutated, so a shared
mutation-based rewrite risks silently capturing stale index values in the
symbolic case. Fixing this properly means either changing the `SEMANTICS`/
`load` interface itself (bigger, cross-cutting — every op's `Compute` functor
uses it), or writing `Direct`-only and `Symbolic`-only versions of the
reduction (loses the "one op body, two interpretations" design the engine is
built around). Neither is a same-risk-class change as steps 1/2, so stopping
here to report back rather than guessing at a fix that could silently break
`Symbolic`.

## Applied fix, step 3: real CPU-time profiling found a bigger, safer target

memtrace only sees allocation, and `Vec6.get`/`Axis.to_int`/`Dim.to_int`
return plain `int`s — zero allocation, so entirely invisible to that profile
even though they're called on every single element access. Neither `perf`
nor `valgrind` are installed in this sandbox, so used `landmarks`/
`landmarks-ppx` instead (`--auto` mode on `lib/native`'s `dune`, temporarily,
gated the same way as the memtrace runs — reverted after): it reports actual
CPU cycles and call counts per function, not just bytes allocated.

That reprofile changed the picture completely. `Vec6.get` — called from
`Tensor.read_at` (6×/call) and `Schedule.coord_index` (1×/call) — was called
**1.83 billion times** for this one fixture and accounted for **~26% of
total cycles on its own** (1.83G of 7.03G cycles), even though it never
allocates: `Vec6.t`'s `Axis.t -> int -> array-index` dispatch (a real,
non-flambda-inlined cross-module function call per access) was costing more
raw time than any of the allocation sites steps 1/2 already fixed.

Fix: `Vec6.t` changed from an abstract array (`'d array`, forcing every
access through `get`'s `Axis.t` match + `Array.get`) to a **public record**
(`{ n; t; d; h; w; c }`, see `vec6.mli`) — `get`/`set` still exist for the
axis-generic API everywhere else, but `Tensor.read_at` and
`Schedule.coord_index` (the two highest-call-count paths) now read
`t.shape.n`/`.t`/`.d`/`.h`/`.w`/`.c` directly, a plain record-field
projection instead of a function call. Unlike the `x_idx`/`w_idx` closure
allocation above, this is a pure data-representation change — `Vec6.t` is
used identically by `Direct` and `Symbolic`, so there's no eager-vs-symbolic
safety concern the way there is for reduction-closure hoisting. Checked for
any code relying on the old array representation (`grep` across
`lib`/`test`/`bin` for non-public `Vec6` access) before changing it — none
found; everything already went through `get`/`set`/the named constructors.

Validated the same way as steps 1/2 (full native/`aten_spec_run` test suite,
no diffs) both under the temporary landmarks instrumentation and after
removing it.

| | (uninstrumented) time |
|---|---|
| after step 2 | ~11.7s |
| step 3 (`Vec6.t` record) | ~9.4s |

Under landmarks instrumentation itself, total cycles dropped 7.03G → 4.20G
(~40% fewer) and `Vec6.get` no longer appears at all in `Tensor.read_at`'s
call tree.

Cumulative: **~25.5s → ~9.4s, ~63% faster than the original baseline.**

`x_idx`/`w_idx` (conv.ml's per-iteration index closures) are very likely the
next-largest lever, per both profiles — but as established above, fixing
those safely needs either a `SEMANTICS`/`load` interface change or
`Direct`/`Symbolic`-specific reduction bodies, a different risk class than
the three steps applied so far. Stopping here again to report back.

## Applied fix, step 4: `Vec6.offset_of` — reuse a public function, not a copy

Two follow-up cleanups to step 3, both in `lib/native/vec6.ml`/`.mli`:

1. `Tensor.read_at` had hand-duplicated `Vec6.offset`'s linearization formula
   inline (to avoid the `coord` allocation calling the real `Vec6.offset`
   would need). Replaced that duplication with a new public function,
   `Vec6.offset_of : shape -> n:int -> t:int -> d:int -> h:int -> w:int ->
   c:int -> int` — `in_bounds`+`offset` fused, taking the 6 components
   directly instead of a `coord`, so it's still allocation-free but is now a
   single named, documented function `read_at` calls instead of a copy of
   the arithmetic living in the wrong module. Returns `-1` (never a valid
   offset) for out-of-range, avoiding an `option`'s `Some` allocation on the
   (overwhelmingly common) success path.
2. `Vec6.t` changed from a fully public record to a **private** one
   (`type 'd t = private { ... }`, same idiom `Dim.t` already uses) — fields
   stay readable from any module (the whole point of step 3), but
   construction/`{ v with ... }` updates are only possible inside `vec6.ml`
   itself, funneled through `shape`/`coord`/`deltas`/`set`. Checked before
   making it private: `dune build` (whole tree) still succeeds, confirming
   no code anywhere constructed or updated a `Vec6.t` via record syntax
   outside `vec6.ml`.

Validated the same way as the prior steps (full native/`aten_spec_run` test
suite, no diffs). Measured: ~9.2-9.4s, statistically the same as step 3 (no
regression from routing through the named function instead of inlining) —
this step is a code-quality change with the same allocation profile, not a
further speedup.

## Applied fix, step 5: thread `Dim.index Dim.t` all the way through `Direct`

Follow-up review asked: shouldn't `offset_of`'s `n`/`t`/`d`/`h`/`w`/`c` be
typed `Dim.index Dim.t`, not raw `int`, matching the `Dim.t` discipline used
everywhere else? First attempt did exactly that but only *locally*, at
`Tensor.read_at`: converted the raw `int`s from `idx` via 6 fresh
`Dim.index` calls right before calling `offset_of`. Correct, but **measured
slower** (~9.3s → ~10.3-10.6s, ~10-13%) — each `Dim.index` call, however
trivial its body, is a real un-inlined cross-module call (same lesson as
step 2's `Vec6.get` regression), and this added six of them on top of an
already-fast path.

The actual bug wasn't "the check is missing," it's "the check is redundant":
`idx`'s values (`x_idx`/`w_idx` in `conv.ml`) are already validated
*upstream*, at whichever of `Direct`'s three `position`-producing functions
built them (`index_zero`, `clamp_low`, `assume_index` — the *only* places a
`position index` value is ever created). `Direct`'s `'role index` was just
typed as plain `int`, so that guarantee was thrown away before it reached
`read_at`, which then had no choice but to re-derive it. Fixing `read_at`
alone can only ever re-pay a check that already happened once — the real fix
is to stop throwing the guarantee away in the first place:

1. **`dim.ml`/`.mli`**: added `Dim.succ : 'role t -> 'role t`, a
   role-preserving, unchecked increment (`x + 1`). Needed because
   `Direct.sum`'s reduction loop increments its position index on every one
   of ~118M iterations for this fixture; that increment is provably still
   valid (starts from a valid `lo`, the loop's own `>= hi` check stops it
   before it's used out of range), so re-validating it every step via
   `Dim.index` would just reintroduce the same class of regression at a
   different call site.
2. **`semantics.ml`**: `position`/`delta` (previously fresh local abstract
   marker types) now alias `Dim.index`/`Dim.delta`. A semantics whose index
   representation *is* a `Dim.t` (only `Direct`, so far) can then use
   `position index = Dim.index Dim.t` directly; a semantics that isn't (e.g.
   `Symbolic`, whose index is an `Expr.index_expr` AST node) is unaffected —
   `position`/`delta` stay purely phantom there either way, since `Symbolic`
   never inspects what they equal.
3. **`direct.ml`/`.mli`**: `'role index = 'role Dim.t` (was `int`).
   `index_zero`/`clamp_low`/`assume_index` (the three producers) now
   genuinely call `Dim.index`, doing the validation exactly once, at the
   true boundary. Every other operation (`index_add`, `index_scale`, the
   `sum`/`max_reduce` loops, …) threads `Dim.t` values through via free
   `:>` coercions to do its underlying int arithmetic, then re-wraps with
   `Dim.delta`/`Dim.succ` (never `Dim.index` again) — no redundant checks
   anywhere in the chain.
4. **`Vec6.offset_of`**/**`Tensor.read_at`**: reverted to `Dim.index Dim.t`
   parameters (the type-safe version), but this time `read_at`'s `idx`
   parameter is *itself* `Axis.t -> Dim.index Dim.t` — the values it
   receives are already-validated by construction, so there's no local
   `Dim.index` call left to cause the earlier regression.
5. **`Tensor.read_at_raw`** (new): `Tensor.read_at` is also called directly
   by `Expr.eval`'s grounding of a `Load` node (`expr.ml`) — a second,
   independent caller whose index function is naturally raw `int` (`Expr`'s
   own arithmetic interprets a `Symbolic` AST, unrelated to `Dim.t`/
   `Direct`'s representation). Rather than force one signature to serve two
   callers with genuinely different natural types, added `read_at_raw` (the
   original coord-building implementation) for `Expr.eval` to keep using,
   and reserved the fast `read_at` for `Direct.load`. Also added
   `Schedule.coord_index_dim` alongside the existing `coord_index`, for the
   same reason (`Schedule.evaluate` feeds `Direct`; `Schedule.ground` feeds
   `Expr.eval`).

Validated against the full native/`aten_spec_run` test suite; three tests
needed small, mechanical updates for the new types, not behavior changes
(`test/native/tensor_test.ml`'s `idx` closure now returns `Dim.index Dim.t`;
`test/native/pool_test.ml` coerces `Dim.t` values to `int` before `<`/
`List.init`; `test/native/symbolic_test.ml`'s `eval_direct` closures use
`coord_index_dim`). No diffs anywhere, including the exact-error-message
expect test.

Measured: **~9.4-9.7s — back to the fast baseline, with the type safety
this whole step set out to add and none of the ~10-13% cost the first,
narrower attempt paid.** Cumulative: still **~25.5s → ~9.5s, ~63% faster**
than the original baseline — this step traded nothing for real type safety
gained.

## Applied fix, step 6: `load6` — 6 explicit indices, no closure at all

Follow-up: since `x_idx`/`w_idx` (the `~61%`-of-remaining-allocation closures
from the memtrace profile) exist purely to satisfy `SEMANTICS.load : input ->
(Axis.t -> position index) -> t`'s closure-shaped interface, why not change
`load` to take the 6 indices directly? Considered and rejected a `Vec6`-typed
bundle as the alternative to a closure — passing a `position index Vec6.t`
would still allocate a record (same cost class as a closure, no `[@@unboxed]`
possible for 6 fields), just without a closure's environment-capture
overhead. The only genuinely zero-allocation option is 6 separate scalar
arguments, which OCaml passes without boxing at all.

Rather than change `load`'s signature (touching every op that calls it —
`conv.ml`, `pool.ml`, `linear.ml`, `matmul.ml`, `permute.ml`, `norm.ml`,
`pointwise.ml`, `reduce.ml`), added a **second** method, `load6`, alongside
the existing `load`:

```
val load6 :
  input -> n:position index -> t:position index -> d:position index ->
  h:position index -> w:position index -> c:position index -> t
```

- `semantics.ml`: added to the `SEMANTICS` module type.
- `direct.ml`: `load6 inp ~n ~t ~d ~h ~w ~c = Tensor.read_at6 inp ~n ~t ~d ~h ~w ~c`.
- `symbolic.ml`: `load6 s ~n ~t ~d ~h ~w ~c = Expr.Load (s, [| n; t; d; h; w; c |])`
  — trivially simpler than `load`'s closure-calling version, no `idx` calls
  needed at all.
- `tensor.ml`: `read_at6` is now the actual primitive (6 explicit
  `Dim.index Dim.t` args, same allocation-free bounds-check+offset logic as
  before); `read_at` (closure-based) becomes a thin wrapper around it
  (`read_at6 packed ~n:(idx N) ~t:(idx T) ...`), so there's one
  implementation, not two.
- `conv.ml`'s `Conv2d.Compute.pixel` (the hot innermost reduction): deleted
  `x_idx`/`w_idx` entirely, calling `S.load6 x ~n:on ~t:ot ~d:od
  ~h:(wh.src kh) ~w:(ww.src kw) ~c:ic` and `S.load6 weight ~n:oc
  ~t:S.index_zero ~d:S.index_zero ~h:kh ~w:kw ~c:local_ic` directly (`on`/
  `ot`/`od` — the N/T/D output-coord components, constant across the whole
  reduction — hoisted out of the loop once, same values `x_idx`'s `_ -> out a`
  fallback used to (re-)compute every call).

Scope deliberately narrow: only `conv.ml`'s two hottest `load` call sites
migrated to `load6`; the other 7 op files, and `conv.ml`'s bias load and
`Convolution.Compute.transposed_pixel`, still use the original closure-based
`load` — `load` isn't going away, `load6` is an additional fast path for
where it earns its keep. `Symbolic` implements both uniformly, so this
doesn't create any Direct-only capability gap.

Validated against the full native/`aten_spec_run` test suite (no diffs).

Measured: **~9.5s → ~6.1s, ~35% faster.** Cumulative: **~25.5s → ~6.1s, ~76%
faster than the original baseline.** This was the single largest individual
step in the whole investigation, confirming the closure-allocation hypothesis
from the very first memtrace profile.

## Applied fix, step 7: `Vec6.t` throughout — `load` itself, not just `load6`

Follow-up: `SEMANTICS.load`'s general (non-`load6`) form still took a
closure (`Axis.t -> position index`), used by ~16 call sites across the
other 7 op files plus `conv.ml`'s bias loads and `Convolution.Compute.
transposed_pixel`. Most of those only override one or two axes of an
existing coordinate (e.g. `linear.ml`'s bias load: all axes zero except
`C`), which a closure expresses awkwardly (a fresh `match` per call) — a
`Vec6.t` argument expresses it directly.

- `semantics.ml`: `load`'s signature changed from `input -> (Axis.t ->
  position index) -> t` to `input -> position index Vec6.t -> t`. `load6`
  is untouched (still the separate, zero-allocation, 6-explicit-scalars
  form for the actual hot path).
- `vec6.ml`/`.mli`: added `set_n`/`set_t`/`set_d`/`set_h`/`set_w`/`set_c`
  (**new value first, vec last** — `set_h h' base`, not `set base h'` — so
  `base |> set_h h' |> set_w w'` actually pipes: each partially applies to
  a `'d t -> 'd t` transformer ready for the next piped vec; `Vec6.set`'s
  existing vec-first order doesn't chain this way). Also added `of_fn :
  (Axis.t -> 'd) -> 'd t`, materializing a per-axis function into a `Vec6.t`
  once — for op code with genuinely dynamic per-axis logic (e.g. `norm.ml`'s
  runtime-length `dims` list), which still writes a closure but only pays
  for turning it into a `Vec6.t` once, not once per `load` call downstream.
- `direct.ml`: `load` now reads the `Vec6.t`'s fields directly (`v.Vec6.n`
  etc.) and calls `Tensor.read_at6` — no closure calling at all, same
  allocation profile as before (one `Vec6.t` per call) but simpler code and
  one fewer indirection.
- `symbolic.ml`: `load s v = Expr.Load (s, v)` — since `Expr.Load` already
  holds an `index_expr Vec6.t` (step 6's Vec6 cleanup), this is now a direct
  pass-through.
- Updated every call site: `linear.ml`, `matmul.ml`, `norm.ml`,
  `permute.ml`, `pointwise.ml`, `pool.ml`, `reduce.ml`, and `conv.ml`'s bias
  loads + `transposed_pixel`. Three patterns emerged:
  - **override 1-2 axes of the base `out` coordinate** (most common, e.g.
    `pool.ml`'s window read, `linear.ml`'s `x` read): `Vec6.of_fn out |>
    set_h ... |> set_w ...`, with the `of_fn` conversion hoisted out of any
    reduction loop (once per pixel, not once per iteration).
  - **build from scratch** (e.g. weight/bias reads, whose coordinate space
    is unrelated to `out`): `Vec6.make ~n:... ~t:... ...` directly — no
    pipe, since there's no meaningful "base" to override.
  - **genuinely dynamic per-axis logic** (`norm.ml`, `reduce.ml`'s
    runtime-length reduced-axis lists): kept the closure, wrapped in
    `Vec6.of_fn` at the `load` call site — correct and simple, just not
    the pipe style, since the axis set being overridden isn't known
    statically.

Validated against the full native/`aten_spec_run` test suite (no diffs —
every op's Direct-vs-Symbolic agreement is unchanged) and the conv2d
benchmark (unaffected, as expected: `load6`, not `load`, is on that path) —
still ~6.1-6.2s, ~76% cumulative.

This step didn't chase a number — it's a code-quality change enabled by
steps 3/6 (`Vec6.t` being cheap to construct/read and reusable for
non-`Dim.t` payloads). No allocation regression versus the closures it
replaced (a `Vec6.t` record and a closure are the same cost class), and
`load6`'s zero-allocation hot path is untouched.

## Applied fix, step 8: `Vec6.map`/`mapi`/`fold`/`foldi`/`copy`/`copy_axis` — revisit `of_fn` call sites

Follow-up to step 7's "kept the closure, wrapped in `Vec6.of_fn`" pattern:
revisited each of those sites to check whether direct `Vec6.t` manipulation
(now that `Vec6.t` carries `set_h`/`set_c`/etc.) would be simpler and avoid
per-pixel closure allocation, rather than materializing a closure into a
`Vec6.t` via `of_fn` every call.

- `vec6.ml`/`.mli`: added `map`/`mapi` (elementwise transform, optionally
  axis-aware) and `fold`/`foldi` (elementwise fold, same), all unrolled over
  the 6 fixed fields (no `Axis.all` traversal/closure-per-element). Also
  added two copy helpers for the "override some axes of one vec from
  another vec" pattern that shows up repeatedly in `output_shape`/`pixel`
  code:
  - `copy_axis : 'd t -> Axis.t -> 'd t -> 'd t` — same-axis copy (`src`'s
    value at `axis` into the piped vec's same `axis`), one dispatch. `src`
    is positional (not `~src`-labeled) so `copy_axis src axis` reads as a
    "pre-filled args first, piped vec last" transformer, matching
    `set_h`/`set_c`'s pipe shape — there's only one axis role here, so
    nothing needs disambiguating with a label.
  - `copy : 'd t -> src:Axis.t -> dst:Axis.t -> 'd t -> 'd t` — cross-axis
    copy (source vec's value at `~src` into the piped vec's `~dst`), two
    dispatches. The source vec stays positional for the same reason as
    `copy_axis`; `~src`/`~dst` label the two axis *roles*, which are what
    actually need disambiguating at a call site (e.g.
    `Vec6.copy weight_shape ~src:Axis.N ~dst:Axis.C x_shape`).
- `pointwise.ml`: `broadcast_coord` rewritten to take/return `Vec6.t`
  directly via `mapi`, instead of building an `Axis.t -> position index`
  closure per call.
- `norm.ml`, `reduce.ml` (`output_shape` and `Compute.pixel`), `linear.ml`,
  `matmul.ml`, `permute.ml` (`output_shape`): rewritten to precompute a base
  `Vec6.t` once (via `fold`/`copy_axis`/`copy`) instead of building a fresh
  per-axis closure inside a reduction loop or `List.fold_left` over
  `Axis.t * Axis.t` pairs.
- `permute.ml`'s `Compute.pixel` and `pointwise.ml`'s `Relu.Compute.pixel`
  were reviewed and left unchanged — both already call `of_fn` exactly once
  per pixel with a closure that's genuinely dynamic (an inverse-permutation
  lookup; a no-op pass-through), so there was nothing to hoist.

Validated against the full native/`aten_spec_run` test suite (no diffs) and
the conv2d benchmark (unaffected, as expected: none of these sites are on
`load6`'s hot path) — still ~6.1-6.2s, ~76% cumulative.

Like step 7, this didn't chase a number — `Vec6.t` records and closures are
the same allocation cost class, so this is a code-clarity change (fewer
per-axis `match` dispatches written out longhand, precomputed bases instead
of re-deriving a lookup inside a loop), not a further speedup.

## Permanent profiling infrastructure: `make profile.memtrace` / `make profile.landmarks`

Every profiling step in this investigation (memtrace for allocation, then
landmarks for real CPU cycles — see steps 3+ above) was done via ad hoc
instrumentation: add it, measure, revert. Made both permanently available
instead — `make profile.memtrace` and `make profile.landmarks` — targeting
the same fixture `benchmark.native_conv2d` measures, so a profile run's
hotspots map directly onto that number (`PROFILE_FIXTURE` overrides which
JSON op-spec either runs against). Both are wired into CI (`.github/
workflows/build.yml`, after "verify pristine") so a regression's *shape*, not
just its magnitude, is visible on every push without re-deriving
instrumentation by hand again.

- **memtrace**: `bin/aten_spec_verify.ml` unconditionally calls
  `Memtrace.trace_if_requested ()` at the top — self-gated on `MEMTRACE`
  being set, a no-op otherwise, so this is always compiled in and never
  affects the uninstrumented path (confirmed: benchmark timing unchanged
  after adding the call). `make profile.memtrace` sets `MEMTRACE=$(PROFILE_DIR)
  /native_conv2d.ctf` and runs the same binary `benchmark.native_conv2d`
  does — no separate build.

- **landmarks**: not safe to leave always-on the same way. First measured
  `landmarks-ppx --auto` compiled into `lib/native` (instrumenting every
  function) at ~2.3x slower on the conv2d benchmark (6.1s -> 14.4s)
  **even with profiling never started** — the ppx's per-function enter/exit
  calls have real cost on this codebase's hot loops even when idle, unlike
  memtrace's genuinely-free-when-disabled env check. So `lib/native/dune`
  now has two `enabled_if`-gated library stanzas (both `(name native)`,
  mutually exclusive on `%{profile}`): the plain one (identical to before),
  and one only enabled under `dune build --profile landmarks` that adds
  `(preprocess (pps landmarks-ppx --auto))` + the `landmarks` library.
  `bin/dune`'s `aten_spec_verify` executable stays a single stanza (no split
  needed there): it unconditionally links `landmarks` and calls
  `Landmark.start_profiling ()` when `LANDMARKS` is set, but with nothing
  instrumented under the plain profile, that's a harmless no-op (confirmed:
  a trivial one-node ROOT callgraph, no per-function breakdown, and
  benchmark timing unchanged). Only `dune build --profile landmarks`
  produces a binary with anything to actually record.
  - Dune-internals lesson hit while building this: an executable with a
    module whose only content is a side-effecting `let () = ...`, never
    referenced by name from another module, needs that module reachable
    from the stanza's own root module or the compiler treats it as dead
    code and silently omits it (no error — the binary just does nothing).
    First attempt split `aten_spec_verify.ml`'s logic from a separate
    profile-specific entry-point file; both `-linkall` and an `.mli` failed
    to fix it, tracing to dune's demand-driven compilation only reaching
    modules transitively required by the stanza's `(name ...)`-matching
    root module. Simplified instead to the single-stanza design above
    (no separate entry file at all) — the actual expensive part
    (`--auto`'s per-function instrumentation) lives entirely in
    `lib/native`, where the enabled_if split works cleanly since both
    stanzas already claim the *same* files, not a mutually-exclusive subset.
  - Active landmarks tracking (`LANDMARKS=1`, not just idle instrumentation)
    times every single instrumented call/exit — unlike memtrace's sampling,
    cost scales with call count, not just allocation volume. On
    `benchmark.native_conv2d`'s fixture (802,816 output pixels, ~1.83B inner
    calls) that's minutes, not seconds — killed a local run after 2.5+
    minutes still climbing. `make profile.landmarks` still defaults
    `PROFILE_FIXTURE` to that fixture for real local analysis, but CI
    overrides it to `test/data/profile/conv2d_small.json` (a tiny synthetic
    `1x4x8x8` conv, same code path, ~0.05s active) — same op, proportional
    hotspot shape, but small enough for every-push CI.
  - Output: `Landmark.start_profiling ()`'s defaults (`Channel stderr`,
    `Textual {threshold = 1.0}`) print a call-tree + aggregated table
    straight to stderr on exit — no temp file to manage or artifact to
    upload.

## Applied fix, step 9: `pixel`'s `out` becomes a `Vec6.t`, closing the loop step 7 left open

Follow-up to step 7 (`SEMANTICS.load` migrated from a closure to a `Vec6.t`)
and step 8 (revisiting `of_fn` call sites): `SEMANTICS.load` took `Vec6.t`
since step 7, but every op's `Compute.pixel` — and `Eval_op.Make.pixel`,
which dispatches to all of them — still took `out` as the *old* closure
(`Axis.t -> position index`). That closure was never anything but a wrapper
around a `Vec6.t` that already existed one level up:
`Schedule.evaluate`/`Tensor.materialize`'s per-pixel loop iterates
`Vec6.coord` values directly (`Vec6.iter`), and manually wrapped each one
into a closure (`coord_index_dim c`) just so `pixel` could unwrap it again —
almost always via `Vec6.of_fn out` at the top of the function, once per
pixel (802,816+ times on resnet18's smallest conv). `Symbolic`'s side had
the same shape for a different reason: `Symbolic.out_coord : Axis.t ->
Expr.index_expr` was a closure returning the same `Index_var` placeholder
regardless of which pixel or even which node was being built, so calling
`Vec6.of_fn out_coord` inside every op rebuilt an identical `Vec6.t` from
scratch on every single `pixel` call instead of it being a plain constant.

Changed `Eval_op.Make.pixel`'s (and by extension every op's `Compute.pixel`)
`out` parameter from `Axis.t -> Semantics.position S.index` to
`Semantics.position S.index Vec6.t` directly — this is exactly the
`load`-was-a-closure problem step 7 already fixed, one layer further out.
Verified type-compatible before touching anything: for `Direct`,
`position index = Dim.index Dim.t`, and `Vec6.coord` is already `Dim.index
Dim.t Vec6.t` — an exact match, no coercion. For `Symbolic`, `position index
= Expr.index_expr`, matching `Expr.Load`'s second argument already.

- `schedule.ml`: `evaluate`'s `pixel` parameter is now `Vec6.coord -> float`
  — exactly `Tensor.materialize`'s own parameter type, so `evaluate` is now
  a thin named seam over it (kept, not inlined away, since the module's
  whole purpose — per its header comment — is being the one place a future
  tiled/parallel/vectorised schedule plugs in). `coord_index_dim` loses this
  one call site but stays alive (and necessary) via `coord_index`, used by
  `ground`'s `Expr.eval` grounding — a genuinely different, still-closure
  consuming API (`Expr`'s own arithmetic is untyped `int`, unrelated to
  `Direct`'s `Dim.t`-typed indices).
- `symbolic.ml`/`.mli`: `out_coord : Axis.t -> Expr.index_expr` (a closure)
  replaced by `out_vec : Expr.index_expr Vec6.t` (a plain constant, built
  once at module load, not per call).
- `eval_symbolic.ml`: `identity_load` used to hand-build the exact same
  6-field `Vec6.make ~n:(Index_var N) ...` that `Symbolic.out_vec` now *is*
  — simplified to just use it directly instead of duplicating the
  construction.
- Every op file (`conv.ml` x3 `pixel` sites, `linear.ml`, `matmul.ml`,
  `norm.ml`, `pointwise.ml` x2, `pool.ml` x2, `reduce.ml`): dropped
  `Vec6.of_fn out` (now `out` is already the right type) and replaced
  `out Axis.X` reads with `Vec6.get out Axis.X`. Net: 8 of the 9 `of_fn`
  call sites the codebase had disappeared entirely.
- `permute.ml` is the one op that genuinely still needs `of_fn`: its
  `pixel` builds the *input* coordinate by looking up each input axis's
  value at a *different* (permuted) output axis
  (`Vec6.of_fn (fun in_ax -> Vec6.get out (List.assoc in_ax inv))`) — a
  real per-axis dynamic remap, not a pass-through, so there's no way to
  hand `load` `out` directly. This was already identified as the
  genuinely-dynamic case in step 8; only the inner closure body changed
  (`Vec6.get out ...` instead of calling the old closure form of `out`).
- `test/native/symbolic_test.ml`: ~20 call sites updated —
  `Symbolic.out_coord` → `Symbolic.out_vec` everywhere, and the 5 sites that
  wrapped a `Direct`-side pixel in `fun c -> ... (Schedule.coord_index_dim
  c)` (to adapt a real `Vec6.coord` into the old closure shape for
  comparison against `Symbolic`) simplified to a direct partial application
  (`~eval_direct:(Pd.pixel p ~x_shape ~x)` etc.) — the coord no longer needs
  adapting, since `pixel`'s type already matches what `compare_symbolic`
  wants. `test/native`'s split compute-test files and
  `graph_test.ml` needed zero changes: both already called `pixel` through
  `Schedule.evaluate` (or built inputs via `Tensor.materialize`, unrelated to
  `out`), so they transparently picked up the new type.

Validated against the full native/`aten_spec_run`/`native_walk`/
`native_bridge` test suites (no diffs) and the conv2d benchmark (unaffected,
as expected — `Conv2d.Compute.pixel`, the hot path this benchmark measures,
never used `of_fn`; it already read `out Axis.X` directly) — still
~6.1-6.2s.

Like steps 7/8, this wasn't chasing a number (a `Vec6.t` and a closure are
the same allocation cost class) — it's a correctness-preserving
simplification that also happens to remove real work: the old path built a
closure over `c` (`coord_index_dim c`) *and* a fresh `Vec6.t` via `of_fn`
for every pixel; the new path passes the *same* `Vec6.t` `Tensor.materialize`
already built straight through, with nothing rebuilt in between.

## Explicitly not done

- **Real external BLAS** for `cpublas::gemm` — would fix the reference-gemm
  bottleneck directly but is a new dependency, ruled out as a goal here.
- **AVX2** (both the build-time-probe form and PyTorch's usual dual-compile +
  runtime-dispatch form) — tried the probe form, reverted: it needs SLEEF,
  a real new `third_party` dependency, to even compile the AVX2 headers. See
  "Tried, reverted: AVX2 build-time capability probe" above.
- **Explicit intra-op thread-pool sizing** — moot per the `OMP_NUM_THREADS`
  finding above: there's no config gap to close without editing the kernel.
- **Per-model sample-image capping** for the slowest models in the smoke
  sweep — left for later if `-O3` + parallelization don't get CI under a
  reasonable target on their own; not added preemptively.
