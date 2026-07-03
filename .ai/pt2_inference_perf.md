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

**Stopping here to ask rather than guessing further** on the big-ticket fix —
there's no small, obvious change left in the "just flip a flag" category for
*that*; the two real options are both real commitments, not flag flips:

1. Switch the *whole project's* opam switch to a flambda variant
   (`ocaml-variants.4.14.3+options` + `ocaml-option-flambda`) and build with
   `--profile release` — gets the ~27% measured above for free everywhere,
   but is an infra decision (slower/different compiler bootstrap in CI unless
   cached, a real change other contributors need to know about), not scoped
   to conv2d.
2. Restructure the native engine's per-element hot path to stop allocating a
   `Vec6.coord` array (and its bounds-check/offset closures) per `Tensor`
   read — e.g. give `Tensor.read_at` a form that takes the 6 indices
   directly rather than an `Axis.t -> int` closure + intermediate array, and
   inline bounds-check/offset as straight-line arithmetic instead of a fold
   over `Axis.all`. This is the change that would actually close the gap
   flambda can't, but it touches `lib/native/tensor.ml`/`vec6.ml`'s core
   read primitive and the `SEMANTICS` interface's `load` signature — a
   deliberate, documented design (`.ai/native_tensor_design.md`), not a
   drive-by fix.

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
