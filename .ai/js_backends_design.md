# JavaScript backends: js_of_ocaml and Melange

How the pure-OCaml half of this repo is built and verified against a JavaScript
runtime, what that verification found, and which limits are structural rather than
unfinished work.

## Why

`lib/native`, `lib/core`, `lib/walk_core`, `lib/native_op_walk` and `lib/native4d` use
no ctypes, no libtorch and no C++. They are the only part of this project that could
run in a browser, which is what a viewer or playground would need. Nothing proved they
did until this went in.

**The scope now differs per backend, and the split is the design.** Running the tensor
engine under node was never the goal on its own — a viewer needs to *read a model* and
*run it*. js_of_ocaml does both: `lib/pt2`, `pytorch_types`, `lib/native_graph` and
`lib/native_interp` are all in its closure, so the jsoo CI job checks out submodules
(`pytorch_types` is generated from `schema.yaml` inside `modules/pytorch`). It still
needs no ccache and no C++: `lib/native_interp` depends on neither `aten` nor `interp`.

Melange stays at `walk_core` + `core`. That is not unfinished work — it is
[#1807][mel1807], below. Its CI job checks out only `vendored/err_trace` (`core` prints
`Err.Error.t`, so it needs that one submodule) and asserts no other gitlink was
initialized, which doubles as a check that no probe section has drifted across the
boundary.

`aten`, `interp` and anything ctypes remain out of scope for both, permanently.

## Layout

```
js/probe/      six probe modules + three entry points; built natively = the golden
js/jsoo/       (modes js) build of the same source
js/melange/    melange.emit of the pure half, plus the shims it needs
```

| Path | Role |
|---|---|
| `js/probe/probe_walk_core.ml` | seeded PCG draws, `Float32` hex + `enc_json` on awkward values |
| `js/probe/probe_core.ml` | `Err` error rendering and the stack-availability verdict |
| `js/probe/probe_tensor_json.ml` | `Tensor` → `Graph_json.encode_tensor` → decode, bit-exactness reported separately from the JSON text |
| `js/probe/probe_native.ml` | half/bfloat16 codecs, a conv2d+relu through `Eval_direct`, the op walk |
| `js/probe/probe_model_json.ml` | a real `model.json` through `ExportedProgram.jsont`; structural summary |
| `js/probe/probe_pt2.ml` | a real `.pt2` opened in memory, lowered, and run; full output payload |
| `js/probe/native_probe.ml` | ungated entry point (sections 1–5) |
| `js/probe/pt2_probe.ml` | gated entry point (section 6) |
| `js/probe/subset_probe.ml` | subset entry point (the two melange can reach) |

Three libraries, because each split *is* a real boundary rather than a grouping:
`probes_pure` is what melange reaches (`walk_core` needs `jsont` alone, `core` needs
`fmt` and `err_trace`); `probes_native` adds Bigarray and `Jsont_bytesrw`; `probes_pt2`
adds everything that reaches PyTorch, and so needs a submodule checkout — now not the
only one of the three, since melange checks out `err_trace` (see below).
Recording those lines in the build graph is what stops a section drifting across one
unnoticed.

Three entry points for the same reason, and they are not interchangeable:

- **Melange** reaches two sections; a two-section run can never diff clean against a
  five-section golden.
- **Tier 2** needs a ~12MB downloaded model; a gated run can never diff clean against an
  ungated golden. Tier 1's fixture is committed (resnet18's `model.json`, 224K), so it
  runs on every push.

Each backend is compared against a native binary built from *the same* source, so there
is no golden file that can drift: `jsoo.runtest` diffs `native_probe.exe` against
`native_probe.bc.js`, `jsoo.pt2.runtest` diffs `pt2_probe.exe` against `pt2_probe.bc.js`,
`melange.runtest` diffs `subset_probe.exe` against the emitted `subset_probe.js`.

**The diff answers "do the backends agree", never "is the answer right."** Sharing the
source is what makes it trustworthy for the first question and useless for the second:
a broken encoder or an op-walk mismatch prints the same text on both sides and diffs
clean. So every correctness verdict in the probe raises instead of printing —
`Err.or_raise` for an `Err.t`, `failwith` for the plain `result` Jsont returns,
and `Walk_core.Walk.run` returns whether every step verified so `probe_native` can fail
on it. A verdict that only prints is not a check; it was one here until a review caught
that a false `round-trip bit-exact` still exited 0.

## Commands

```sh
make js.build                # everything: jsoo + melange
make jsoo.runtest            # build both ways, run, diff (195 lines)
make jsoo.inline-runtest     # the expect suites under node
make melange.runtest         # same for the pure half (26 lines)
make js.runtest              # the three above
make melange.build.scaffold  # shim + fmt + jsont_base only; the diagnostic floor
make jsoo.pt2.runtest        # tier 2; gated on a downloaded model (133 lines)
```

Deliberately outside `make runtest`, same reasoning as `pt2.runtest`: they need node,
and linking js_of_ocaml on every local test run is not worth it. CI runs jsoo and
melange as two parallel jobs in `.github/workflows/js.yml`, and all three jsoo targets —
`jsoo.pt2.runtest` included — run in the jsoo job.

`js.build` is composed of `jsoo.build` and `melange.build` rather than being a bare
`dune build js`, which would defeat the profile gate that keeps `melange.emit` off
`@all`.

## Reading and running a model: what the two tiers cover

**Tier 1 — `model.json`, ungated.** `Jsont_bytesrw.decode_string ExportedProgram.jsont`
on resnet18's committed `model.json` (226,263 bytes, 70 nodes, 9 distinct targets), the
exact call `pt2_archive.ml` makes. It prints a *structural* summary rather than the
tree: 226KB of JSON in the diff would drown every other section, while a dropped node or
a mis-parsed target still moves the summary. Targets are sorted and de-duplicated so the
line cannot pick up an incidental traversal-order difference, only a decoding one.

**Tier 2 — the archive and inference, gated.** `Pt2_archive.of_string` → `lower_archive`
→ `run`, on `mobilenet_v3_small`. Everything goes through the **in-memory seam**
(`of_string`, `pt_of_string`, added for this) rather than the path-based entry points:
under node either would work, but a browser only has the in-memory one, so that is the
one worth proving.

Result: **bit-exact**, all 1000 logits, 488 lowered nodes. That confirms the FMA
prediction below — the product of two f32 values is exact in float64, so contracting
`acc +. (a *. b)` cannot change the accumulator either.

**Cost: 9.2s native, 44.9s under node (~4.9x).** This is why the fixture is
`mobilenet_v3_small` and not the resnet18 used everywhere else: resnet18 is 86.2s natively
(`native_inference_verify.md`), and node multiplies whatever the native cost is.

**The full payload is printed, not summarised, and that is load-bearing.** The only
cross-backend comparison that exists is the Makefile's `diff -u` of the two runs' stdout.
A "bit-exact" boolean computed *inside* one run would be comparing that run against
itself and proving nothing — which is exactly what makes `probe_tensor_json`'s flag
different: that one is a *round-trip* (encode → decode → compare) within a single run.
Inference has no round trip, so the bits themselves have to reach the diff. Shape and
top-5 are printed too, but they are a readable summary, never the check: both are
invariant to arbitrary low-order float differences. Hex rather than decimal so the
comparison is over bits instead of either backend's float formatting, and eight per line
so a divergence localizes to a handful of values — verified by flipping one ULP in one of
the 1000, which the diff pins to a single line.

**Tier 1's worker section — the whole browser path, ungated.** The request bytes go
through `Me_request.Request.jsont`'s staged decoder, `Me_export.handle` does the
projection, and the answer comes back through `Me_response.Wire`. That is exactly what a
worker does, and everything in it is pure OCaml in one library, so the two backends must
agree byte for byte. The section exists because the alternative is *asserting* that the
browser runs the same code rather than checking it.

It decodes a fixed request string rather than building one through the constructors:
building would test the constructors, decoding tests the staged decoder — which is the
half a browser exercises. It summarises the session document for tier 1's reason (a
253KB payload would drown the diff) but prints the response **metadata** in full,
including the byte count, which is measured from that document and therefore moves with
it. And it drives the failure path once, because a model the worker cannot read still
has to produce a well-formed response with a code from the closed vocabulary rather than
an exception escaping into the shell.

## Golden-based coverage: the expect suites under node

The differential probe answers *"do the backends agree"* and structurally cannot answer
*"is the answer right"* — a fault that reproduces on both sides prints identical text and
diffs clean. `(inline_tests (modes best js))` on `test/native`, `test/native4d` and
`pt2_test` closes that gap by running the existing expect blocks under node against their
committed goldens.

Two things about the dune stanza are easy to get wrong:

- **No `enabled_if`.** It is a field of the `inline_tests` stanza, not of a mode, so
  gating on `%{bin-available:node}` would disable `best` alongside `js` and silently
  empty the native suites on a node-less machine — strictly worse than the failure it
  guards against.
- **A root `dune` sets `(js_of_ocaml (runtest_alias runtest-js))`.** Without it the js
  runs attach to `runtest`, and every `make runtest` starts linking js_of_ocaml.

`ppx_expect` v0.16.2 needs no shims for this: it ships a `jsoo_runtime` implementing its
stdout-capture stubs over `caml_ml_channel_redirect`, and `ppx_inline_test` passes
`-diff-cmd -`, so no subprocess is spawned.

**One golden cannot hold two backends' answers**, which is the recurring constraint. Four
expect blocks asserted a native `Printexc` backtrace (two in `core_test`, one in
`shape4_test`); js_of_ocaml captures no callstack, so they now assert that *whichever*
branch was taken is well formed — reading the stack through `Err.Error.origin` and
accepting either `Err`'s own `stack unavailable` notice or the runtime's `not linked
with -g`. The two integer regressions in `pt2_test` do the same with an oracle: compare
against `Int32.unsigned_to_int` or the backend's own representable range and print only
whether they agree.

`~pos:__POS__` is the way out of this bind rather than around it. A position is compiled
in, so it reads identically on all three backends and is the *only* provenance that
exists on the two JavaScript ones — which is why the seams the browser path crosses
carry one, and why `core_test` pins that behaviour under a `Never`-backtrace policy and
runs the assertion under node too.

## The durable rule: never assume a 63-bit `int`

**js_of_ocaml's `int` is 32 bits** (`Sys.int_size = 32`, `Int64.to_int 0xFFFFFFFFL = -1`).
Two latent defects in the pure libraries turned on that assumption, and both are now
fixed:

- `Walk_core.Pcg.next` promised "a non-negative OCaml int in `[0, 2^32)`", which is not
  representable. Every draw ≥ 2^31 came back negative and `Walk.pick`'s `List.nth`
  raised, killing the whole random-walk framework under node. `next` now returns
  `int64`; callers narrow only after bounding (`top24` for float draws, `Walk.index_of`
  via `Int64.rem` for selection). Verified identical natively over 200k draws × 40 list
  lengths.
- `Half.f32_bits` built the f32 pattern as an `int` in `[0, 2^32)` with
  `land 0xFFFFFFFF`. The literal itself truncates to `-1` under jsoo, making the mask a
  no-op; natively the mask is load-bearing. Neither spelling is right for both, so the
  codecs now work from the two 16-bit halves and every intermediate stays under 2^23.

Extending the probe to the PT2 path added four more, all found by the same rule:

- **`Pt2_pickle.int_of`** narrowed an `int64` unchecked. Pickle integers are *signed*, so
  the bound is two-sided — checking only the top leaves half the defect. It gets its own
  `` `Int_out_of_range `` tag rather than reusing `` `Expected_int ``: the value in hand
  *is* an integer, and the wrong tag would send a reader hunting a type error in the
  pickle instead of a width limit.
- **`Pt2_tensor.numel` / `contiguous_strides`** multiplied individually-bounded sizes in
  `int`. **Per-value bounds do not cover aggregates** — this is the part of the rule that
  is easy to miss, and it now folds in `int64` and narrows after proving the product fits.
- **`Native_interp.storage_index`** sums `stride * coord` per element. Bounding it per
  element would cost the hot loop, so the *whole reachable index range* is bounded once in
  `int64` before materializing; that proves every coordinate between the extremes is in
  range, and incidentally hoists a per-element check out of the loop. Strides can be
  negative (channels-last inputs), hence both ends rather than just the maximum.
- **`Opickle.Src.uint4`** masked with `land 0xFFFF_FFFF` and so returned a *negative
  length* under jsoo. js_of_ocaml diagnosed the literal itself: `integer 0xffffffff
  (4294967295) truncated to 0xffffffff (-1)`. **Fixed upstream** — it now uses
  `Int32.unsigned_to_int` and raises where the value is unrepresentable, exactly as
  `uint8` already did for a 64-bit length that does not fit.

The first three are latent — reaching any of them needs storage past 2GB, which no model
in the zoo approaches. `uint4` was live but equally unreachable here: all seven call
sites are pickle string/bytes lengths, memo indices or an ext code, so triggering it
needs a single pickle object over 2GB, and a `.pt2` keeps tensor payloads in separate zip
entries, never in the pickle.

**Its regression test lives in *this* repo's `pt2_test`, not upstream**, and that is the
durable point rather than an accident of who fixed it: opickle's own suite has no js
mode, so the `None` branch is unreachable there and a test for it could never fail.
Coverage has to sit wherever the 32-bit backend actually runs. The test now spans the
full range including `0x80000000` and `0xffffffff`; before the fix those two could not be
asserted at all, because the backends' *correct* answers differed **because of the bug**
and one `[%expect]` cannot hold two spellings.

When adding to any of these libraries: any value that can reach 2^31 must be
`int32`/`int64`, not `int`, and aggregates need their own bound. `make jsoo.runtest`
catches a *divergence*; `make jsoo.inline-runtest` catches a wrong answer both backends
agree on.

## ocamlopt contracts FMA; bytecode and JS do not

`ocamlopt` compiles `a *. b -. c` into a fused multiply-add. The bytecode interpreter —
and therefore js_of_ocaml, which compiles the bytecode — does not:

```
                  ocamlopt              bytecode / node
14. *. 0.05 -. 0.7   0x1.8p-54              0x1p-53
```

Binding the product to a name first defeats the contraction and both agree.

**The engine is immune**: every tensor payload is f32, and a product of two f32 values
is exact in float64 (24+24 < 53 bits), so contraction cannot change what it computes.
Only code that manufactures values from inexact double arithmetic is exposed. The
probe's conv inputs are therefore `k/2^n`, exact in both float64 and float32 — an
earlier ramp of `float_of_int (i+1) *. scale -. 0.7` reported an "engine divergence" at
1e-16 that was entirely this.

**Triage rule.** `js/probe/dune` also builds `native_probe.bc`. js_of_ocaml compiles the
bytecode, so when the jsoo diff goes red, a difference that survives between `.bc` and
`.exe` is ocamlopt's, not jsoo's. That is how the above was identified in one run.

## What js_of_ocaml reproduces exactly

All 180 lines, with no shims and no source changes beyond the two int-width fixes:
`Float32` hex and `enc_json` spellings (including `9.9999997473787516e-06`, the
subnormal `1.4012984643248171e-45`, and the `-0.0`/`inf`/`nan` hex fallbacks); the
tensor payload round-trip, bit-exact; the half/bfloat16 codecs; a real conv2d+relu over
Bigarray storage including the printed tensor and its JSON; the full op walk.

A `(modes js)` executable links ordinary bytecode libraries, so `js/jsoo/` mirrors
nothing — it copies the entry module and links `probes_pure`/`probes_native` directly.

## Melange: three obstacles, all cleared

Melange cannot consume an opam library that lacks melange mode, so `fmt` and `jsont` are
rebuilt from sources copied out of the switch with `%{lib:...}`. That variable resolves
to whatever is installed; reproducibility comes from the devcontainer pins, not the
rule.

**1. No Bigarray.** Melange's stdlib omits the module entirely, and `jsont_base`,
`jsont` and `bytesrw` all name it in public signatures — so a shim has to come *first*,
not last. `js/melange/shim/bigarray.ml` supplies the closed set they touch:
`Array1.{create,get,unsafe_get,set,dim,init,kind,layout}`, the `kind`/`layout` families,
`c_layout`, and the kind constructors.

Its store is a plain OCaml array, which is correct for jsont (byte buffers, already in
range) and **wrong for `lib/native`'s `Payload`**, whose whole point is that storing
through a kind rounds and truncates.

**Do not grow this shim into that.** `Payload` needs real kind semantics across six
kinds — `float32`, `int64`, `int32`, `int16_unsigned`, `int16_signed`, `int8_signed` —
and `int64` in particular has no clean JS backing, since melange represents `Int64` as
a pair of 32-bit words rather than `BigInt`. That is the job of
[melange-re/melange#1807][mel1807], which adds Bigarray to melange's stdlib over
TypedArrays: every kind including `int64` (paired `Int32Array`, matching melange's
representation), Array0–3 and Genarray, both layouts, and `sub` with genuine shared
storage. Open as of writing. A hand-written TypedArray tier here would duplicate that,
get less testing, and be dead the day it lands — so this shim stays deliberately small
and jsont-shaped, and `native_mel` waits.

[mel1807]: https://github.com/melange-re/melange/pull/1807

**2. `jsont.ml` exports `Error` twice.** It declares `exception Error of error` beside
`module Error`; melange refuses to emit a unit exporting one name twice:

```
Error: Error are exported as twice
```

Not a version to upgrade past — 5.1.0-414 and 7.0.1-414 both reject it, and it
reproduces in a four-line module. The melange copy renames the exception to `Exn`. The
only site in this repo that catches it is `test/native4d/shape4_test.ml`, native-only.

**This must not be done with a regex.** `Error` names three different things in that
file — exception constructor, module, result constructor — and they are textually
identical. `jsont.ml:2058` is `| Error e ->` on a *result*; rewriting that form corrupts
it and still compiles. The recorded patch was produced by renaming the unambiguous
sites and letting the compiler name the remaining eight exception-position arms one at a
time.

**3. A recursive value definition crashes at runtime.** `let rec elt = Repr.Rec any and
… and any = lazy …` (`jsont.ml:1727`) becomes a `Caml_obj.update_dummy` over a `const`
declared later in the emitted file. JS's temporal dead zone makes that fatal the moment
anything requires jsont:

```
ReferenceError: Cannot access 'any$1' before initialization
```

It compiles perfectly — only running the output catches it, which is the argument for
`melange.runtest` existing at all rather than stopping at a build check. Deferring the
reference behind `lazy (Lazy.force any)` fixes it and is semantically a no-op, since
`Rec` is consumed through `Lazy.force`.

**This one is a backport, not a permanent workaround.** It is a melange codegen bug,
fixed upstream by [melange-re/melange#1735][mel1735] (superseding #1728) and released in
melange **7.0.0** (2026-06-21). Verified directly: on `7.0.1-414` the `Error` rename
*alone* produces JS that runs and matches the native golden exactly. Drop the hunk when
the pin can move — see the version table below for what that costs. Obstacle 2 stays
either way: `7.0.1-414` still rejects the duplicate export.

[mel1735]: https://github.com/melange-re/melange/pull/1735

Both edits live in `js/melange/vendor/jsont/*.patch`, applied by a rule that refuses to
run unless the upstream source matches a recorded sha256 — they are only correct for
jsont 0.2.0, and a later release could accept them while meaning something else.

No `Printexc` shim is needed, but melange's `Printexc` is only *partly* usable and the
difference bit us. `get_callstack` is declared and returns a value its own
`raw_backtrace_length` cannot read — the emitted code is `bt.length`, the value is
`undefined` — so asking for the length threw a TypeError out of a printer, i.e. a crash
in diagnostic code on the path taken when something has already gone wrong. This was
fixed upstream in the vendored `err_trace`: `Err.Stack` asks `backtrace_slots`, which
melange converts defensively and answers `None` from, the same branch js_of_ocaml takes.

**The rule that generalises**: a melange `Printexc` function that is *declared* is not
therefore *callable*. Reach the runtime through the conversion functions
(`backtrace_slots`), not the accessors (`raw_backtrace_length`,
`raw_backtrace_to_string`).

`err_trace` is mirrored into `js/melange/vendor/err_trace/` rather than linked, for the
usual reason plus one of its own: it ships a melange stanza gated on
`ERR_TRACE_TEST_MELANGE=true`, and that same variable *disables* its native `src/`
library, so no single tree can have both. Same copy-never-fork rule as `js/melange/core`.

### No OCaml crosses into the browser

Nothing in this repo links `js_of_ocaml` as a *library*: no OCaml handles `postMessage`,
returns a Promise, or touches `Js_error`. `web/src/session.js` is plain JavaScript
driving the upstream renderer against a session document the native CLI produced ahead
of time, and the jsoo build here is a differential probe, not a deployment.

So the external-stack adapters (`Err.Stack.of_external` fed from `Js_error.stack` /
`Js.Exn.stack`) are deliberately **not** written — they would have no call sites. When an
OCaml worker does appear, the adapter goes beside it and never in the base `Err`
library: a browser dependency there would land in melange's pure closure, which is
exactly what the scope split above exists to prevent.

### Profile gating

Every stanza under `js/melange/` carries `(enabled_if (= %{profile} melange))`. Without
it a `melange.emit` attaches to `@all` and plain `dune build` starts compiling melange
for everyone. A clean `dune build` produces no melange artifacts at all, not even the
vendored copies. Build with `dune build --profile melange @melange`.

`@melange-scaffold` (`js/melange/scaffold/`) builds only the shim, `fmt` and
`jsont_base` — the parts that hold regardless of the jsont patch — so there is always a
melange target that builds. It is a real `melange.emit` rather than an alias over file
paths, so it exercises the libraries through supported dune abstractions instead of
dune's private `.objs/melange` layout.

## Version constraints

| Package | Pin | Why |
|---|---|---|
| `melange` | `5.1.0-414` | newest that co-installs with `ppx_expect` on OCaml 4.14 — see below |
| `ppx_expect` | `v0.16.2` | pinned *with* melange |
| `js_of_ocaml{,-compiler,-ppx}` | `6.4.1` | measured toolchain |
| `jsont` | `0.2.0` | the patch's sha256 guard |
| `fmt` | `0.11.0` | vendored source |
| `@devcontainers/cli` | `0.88.0` | `--skip-post-create` is load-bearing |
| `nodejs` | `20.19.2+dfsg-1+deb13u2` | exact Debian package used by the JS and browser gates |
| `npm` | `9.2.0~ds1-3` | exact Debian package paired with nodejs |

### Why melange is pinned to 5.1.0 — it is not melange's fault

**Melange itself supports this switch.** `melange.7.0.1-414` is built for
`ocaml >= 4.14 & < 4.15` and installs here without complaint; the `-414` suffix is
melange's OCaml-series variant, and 7.0.1 also ships `-51`, `-52`, `-53`, `-54`, `-55`.
Nothing about the latest melange requires OCaml 5.

What cannot happen is **melange 7 and `ppx_expect` in the same opam switch on OCaml
4.14**:

```
melange >= 7.0.0         → ppxlib >= 0.36
ppx_expect               → ppx_inline_test
ppx_inline_test v0.16.1  → ppxlib < 0.36               (the cap; newest for OCaml 4.14)
ppx_inline_test v0.17.1  → ppxlib >= 0.36, ocaml >= 5.1
```

That cap is **real, not a stale upper bound**. ppxlib 0.36 changed the parsetree —
`Ptyp_alias` now carries a `string loc` rather than a `string` — and forcing the issue
with `--ignore-constraints-on ppxlib` fails to compile `ppx_globalize`, `ppx_optcomp`
and `ppx_sexp_conv` (verified). So an unpinned install resolves the conflict by
**removing `ppx_expect`**, silently taking every inline test in the repo with it. That
is not hypothetical: it happened twice while this was being built.

Two ways out, neither worth taking for this alone — the payoff is *one line* of
backported patch, since obstacle 2 persists on 7.0.1-414 regardless:

- **Move the switch to OCaml ≥ 5.1.** Then `ppx_inline_test v0.17.1` admits
  `ppxlib >= 0.36` and everything coexists. Every opam dependency here already has a
  5.x-compatible version with no upper bound on `ocaml`. The unverified part is this
  repo's own code — chiefly `lib/aten`'s ctypes + C++ archive, which nobody has built
  on OCaml 5. Worth revisiting whenever the switch moves for other reasons.
  Note what this does *not* buy: melange 7 has no Bigarray, so `native_mel` stays out
  of reach regardless — see **Not done**.
- **Give melange its own opam switch**, selected through a `dune-workspace` context and
  `(enabled_if (= %{context_name} melange))` instead of the profile gate. The two sides
  genuinely do not need each other — melange builds want no `ppx_expect`, and the
  native build wants no melange. Cost is a second switch in the devcontainer feature
  (which creates one today) and a correspondingly larger CI image.

`dune-project` is at `(lang dune 3.20)` because `(using melange 1.0)` is not accepted
below it.

## CI

`.github/workflows/js.yml`, three parallel jobs with **different** submodule needs and
different failure modes. The split is about attribution: a job that can go red for its own
reasons gets its own name in the job list.

The **jsoo** job checks submodules out (top level, non-recursive): `pytorch_types` comes
from `schema.yaml` in `modules/pytorch`, and the tier-1 fixture from
`modules/pytorch.models.pt2`. It runs `jsoo.runtest`, `jsoo.inline-runtest` and
`jsoo.pt2.runtest` as separate steps so a failure names the layer. It needs no ccache — it
builds no C++.

**Tier 2 runs here, not in `build.yml`.** It could run there for free: that job already
downloads `PT2_MODELS_CRAM`, which contains mobilenet_v3_small, so the step cost one
invocation and no download. It ran there first for exactly that reason, and the reason was
wrong twice over.

*Attribution.* A bit-exactness regression under node would have reddened `build` while the
workflow named `javascript` stayed green — the one failure mode the jsoo/melange job split
exists to prevent.

*Wall clock.* Node multiplies the native cost ~4.9x, so this is the most expensive JS step
by an order of magnitude, and it was being added to the longest job on the board. Measured
on one push (run 30779046699 / 30779046652, both at the same commit):

| job | duration | of which testing |
|---|---|---|
| `build` | 13+ min, serial | tier 2 was **1m31s** of it |
| `javascript` / `jsoo` | ~3 min | 21s — the rest is a 2m10s devcontainer |

The two jobs run concurrently and `build` sets the critical path, so the step is close to
free in the jsoo job and directly on the critical path in `build`. Moving it shortens the
push and costs ~99 MB of download in a job that had minutes of slack.

The **browser** job drives the real pinned Model Explorer element under Chromium. It needs
`modules/pytorch.models.pt2` for resnet18's committed `model.json`, because the Stage 1 gate
renders the session `native_graph visualize` actually produces from it — the only check
anywhere that our export is one the *element* accepts rather than one our own validator
accepts. Two caches, both keyed on `web/package-lock.json`: `node_modules` and the
downloaded Chromium. Keying on the lockfile rather than on a fixed string is what keeps the
renderer pin a claim about what was **rendered** — a version bump gets a fresh entry instead
of silently reusing the previous `dist`.

It fails for reasons the other two cannot — an npm lockfile, a downloaded browser, a
renderer version — which is exactly why burying it in either would make a browser regression
read as an OCaml one.

**Its cache key must not be `build.yml`'s.** `pt2.vars` describes an archive of every model
CI has downloaded this release; `jsoo.pt2.vars` describes one model, so it emits its own
key (`pt2-jsoo-<release>-<model>`) and its own glob. Sharing the key would not fail, which
is what makes it worth stating: Actions caches are immutable once saved, so whichever
workflow reached a cold key first would define its contents, and a one-model archive stored
under the all-models key silently makes every later save a no-op and every `build.yml` run
re-download the other four — forever, with nothing going red. `jsoo.pt2.vars` needs no
model-list hash, unlike `PT2_MODELS_HASH`, because the key names the single model it holds.

The **melange** job keeps a near-submodule-free path — one exception, `vendored/err_trace`,
which `core` (and its melange mirror, `core_mel`) needs because `Core.Pretty` prints
`Err.Error.t`. Checking out without `submodules:` is not enough on its own: the shared
devcontainer action runs `devcontainer up`, which runs this repo's `postCreateCommand`,
which runs `git submodule update --init`. The action's `post-create` input (default
`true`, so `build.yml` and `ocaml-images.yml` are unaffected) passes `--skip-post-create`,
so the job then initializes `vendored/err_trace` itself, by name, and nothing else. It
asserts *after* that step that no *other* gitlink was initialized — before it the check
passes trivially — and checks every gitlink but that one path.

**That assertion now carries more weight than a checkout flag.** Melange's reach is meant
to stop at `walk_core` + `core`, and everything past that line (besides `err_trace`) goes
through `pytorch_types`, which needs a submodule. If it goes red, the melange closure has
grown — most likely a probe section that drifted into the wrong library.

### Submodules: name them, never `--recursive`

`build.yml` initialised pytorch's nested submodules with `--recursive`, which is 37 repos.
`lib/aten/dune` documents the actual prerequisite as `third_party/fmt third_party/cpuinfo`,
and measured against `.git/modules/modules/pytorch`:

| | size |
|---|---|
| `modules/pytorch` itself, shallow | ~97 MB |
| its 37 nested submodules | **306 MB** |
| `modules/pytorch.models.pt2` | 1.1 MB |

So the recursion was 76% of the cost. Naming the two is equivalent for the build: nothing
in the tree references any other `third_party` path, neither `fmt` nor `cpuinfo` declares
submodules of its own, and `build_archive.sh` deliberately builds `CPU_CAPABILITY=DEFAULT`
so SLEEF — the one other plausible candidate — is never needed.

**No `.git` cache in `js.yml`.** With the recursion gone the jsoo job fetches ~98 MB, and a
~100 MB Actions cache restore is not reliably faster than a ~100 MB shallow fetch from
GitHub's own CDN; `build.yml`'s key is per-ref, so it would miss on every new branch anyway.

## Not done

**Melange's ceiling is Bigarray, not scope.** Now that jsoo reads and runs real models,
it is worth being precise about why melange does not: `Payload`'s whole point is that
storing through a kind rounds and truncates, `js/melange/shim/bigarray.ml` is
deliberately jsont-shaped and must not grow into that, and no melange release has
Bigarray. Everything below follows from that one fact.

The two remaining melange items have **different** gates. They are easy to conflate —
both sound like "newer melange" — but only one is available today:

- **Dropping the second jsont hunk** — fixed upstream by [#1735][mel1735] and released
  in melange **7.0.0**. Gated only on melange ≥ 7, hence `ppx_inline_test v0.17.1`,
  hence **OCaml ≥ 5.1**. Reachable now.
- **`native_mel`** — needs Bigarray in melange ([#1807][mel1807]), plus `bytesrw` and
  `jsont_bytesrw` vendored together. #1807 is **unmerged**, so *no* melange release has
  Bigarray, and melange 7 does not. An OCaml 5.1 move does **not** unlock this. Once it
  does land, `js/melange/shim/` goes away entirely rather than growing.

What OCaml floor #1807 will carry is not yet knowable. Its branch (melange `main`)
declares `ocaml >= 5.5 & < 5.6`, which taken alone would imply 5.5. But melange
publishes every release across the whole supported series — `6.0.1` shipped `-414`
and `-51`…`-54`; `7.0.0`/`7.0.1` shipped `-414` and `-51`…`-55` — so `main`'s
constraint governs development, not what a release makes available. On that history a
release containing #1807 would very likely also ship a `-414` variant. That is a
prediction from release practice, not a fact; check the actual variants when it lands.

The migration's unverified part is this repo's own `lib/aten` (ctypes + C++ archive),
not the opam solve — every dependency here already has a 5.x version.

A hand-written TypedArray shim would be the only way to get `native_mel` *before*
#1807 lands. It is a real option — the surface `Payload` needs is small, and the probe
would validate it against the native golden immediately — but it buys melange coverage
of the engine at the cost of carrying an implementation upstream is already replacing.
- **A browser target.** Both backends currently produce a node program that writes to
  stdout. `js/jsoo` takes no `js_of_ocaml` library dependency at all for that reason.
  The seam it needs already exists — `Pt2_archive.of_string` takes bytes, which is what a
  fetch or a file picker yields — so what is missing is the bindings and a way to hand
  the bytes in, not a restructuring of the reader.
- **Memory has not been engineered for a browser.** The archive is one OCaml string (a JS
  string under jsoo) and each weight is copied again by `Bytes.of_string`
  (`pt2_archive.ml`), so it roughly doubles. mobilenet_v3_small at 12 MB is comfortable;
  resnet18 at 46 MB is the shape of the real constraint.
- **The web dependency closure is lockfile-pinned and local.** `npm ci` installs
  `web/package-lock.json`; the Makefile invokes package scripts, not `npx`, so it cannot
  fetch a missing tool from the registry. `web/.npmrc` rejects a different Node/npm
  runtime, while the devcontainer pins the exact Debian packages above. Dependabot tracks
  `/web` as an npm ecosystem, so a deliberate dependency update changes the lockfile and
  browser-cache key together.
