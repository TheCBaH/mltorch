# Wasm backend: scalar contracts

Status: the contracts below are frozen for the scalar backend (W1). Code lives
in `lib/wasm_ir` (pure Wasm representation, checker, encoder) and, later, the
Loop IR lowering in `lib/loop_ir`. SIMD is a separate, later layer that must not
change anything stated here.

## Scope and refusals

The backend lowers exactly what `Loop_js`/`Loop_c` lower: a static
`Loop_program.t`. It adds no dynamic shapes, no operator `Loop_lower` refuses
(`Loop_unsupported`), no threads, no memory64, no relaxed or standard SIMD.
Core-only modules (MVP plus the non-trapping float-to-int conversions and
sign-extension, both in Node 20 / V8 11) must validate on a default Node.

A refusal is a value, never a half-lowered module: an unsupported buffer
format, a constant outside a Wasm bound, a program whose memory exceeds the
policy below.

## Census of `Loop_ir` constructors

Every constructor, its Wasm obligation and how it is checked. "Ref" is the
oracle: `Loop_interp` for values and failures, `Loop_js`/`Loop_c` as peers.

### `Loop_index.t` — index domain `[-2^31, 2^31)`, Wasm `i32`

| Constructor | Lowering |
|---|---|
| `Add`, `Scale (k, _)` | `i32.add`, `i32.mul` (`k` must fit `int32`, else refuse). Wrap is unobservable only because overflow is checked first, see `Index_overflows`. |
| `Ceil_div_pos`, `Floor_div_pos` | floor division by a positive constant `d`: `q = x / d` (`i32.div_s`), minus one when `x < 0` and `x mod d <> 0`; ceil is `-floor(-x / d)` computed on the same helper. `d` fits `int32` and is positive. |
| `Clamp_low` | `max(x, 0)` via `select`. |
| `Const` | `i32.const`; refuse outside `int32`. |
| `Max`, `Min` | compare and `select`. |
| `Temp`, `Var` | `i32` locals. |

### `Loop_expr.t`

| Constructor | Lowering |
|---|---|
| `Const`, `Binary`, `Unary` | `f64.const` (bit-exact incl. NaN and `-0`), `f64.add/sub/mul/div`; `Unary`: `sqrt`/`trunc` native, `exp log sin cos` host imports, `Erf` the project polynomial (`Expr.Value.erf_approx`) as a helper calling the `exp` import. |
| `Float_max` | `f64.max`: NaN-propagating, `+0 > -0`, exactly `Float.max`/`Math.max`. |
| `Round_f32` | `f32.demote_f64; f64.promote_f32`: round-to-nearest-even, the one rounding boundary. |
| `Value_of_index` | `f64.convert_i32_s` (never `-0`). |
| `Float_to_i64` | `i64.trunc_sat_f64_s`; the `Fail_if` that precedes it rejects NaN/inf/out-of-range, so saturation is unreachable. No trapping conversion is emitted. |
| `I64_binary` | `i64.add/sub/mul` modular; `I64_div` is `i64.div_s` only behind the existing zero and `min/-1` failure checks (a trap is never the failure path). |
| `I64_const`, `I64_of_index`, `I64_to_float` | `i64.const`, `i64.extend_i32_s`, `f64.convert_i64_s` (round-to-nearest-even = `Int64.to_float`). |
| `Load`, `Load_flat`, `Load_i64`, `Load_i64_flat` | typed loads by buffer format (table below). |
| `Array_get` | `f64.load` from the local-array region. |
| `Select` | `if`-expression (lazy branches), not Wasm `select`, because arms may fail or load. |
| `Temp` | locals, one per `(carrier, temp)`. |

### `Loop_bool.t`

`I64_eq/lt` → `i64.eq/lt_s`. `Index_eq/lt` → `i32.eq/lt_s`. `Not`, `Or` →
`i32.eqz`, short-circuit `if` (right operand is lazy). `Out_of_range (i, n)` →
`i32.ge_u i n` (one unsigned compare; `n ≥ 0`). `Value_eq/lt` → `f64.eq/lt`
(NaN false). `Pool_better (best, v)` → `v > best || v <> v`. `Index_overflows`
→ the per-node post-order checks of `Loop_js.overflow_nodes`, each node computed
in `i64` from already-checked operands so the check itself cannot wrap.

### `Loop_stored.t` and legal stores

Readable formats are wider than legal stores. A program stores only `Bool`
(canonical `v <> 0`, one byte), `F32` (`f32.demote_f64; f32.store`) and `I64`
(`i64.store`), each only into a buffer of the matching format; any other
pairing is a lowering defect (`Invalid_argument`), as in `Loop_interp.store`.

| Format | Load | Store |
|---|---|---|
| `f32` | `f32.load; f64.promote_f32` | `Loop_stored.F32` |
| `f64` | `f64.load` | not stored |
| `i32` | `i32.load; f64.convert_i32_s` | not stored |
| `i64` | `i64.load` | `Loop_stored.I64` |
| `bool` | `i32.load8_u`, `≠ 0` → `1.0 / 0.0` | `Loop_stored.Bool` |
| `f16`, `bf16` | `i32.load16_u` through the `Half` helpers | not stored |
| `i8`, `i16` per-tensor | `scale * (f64(q) - f64(zero))`, `q` sign-extended, one `f64` multiply | not stored |
| `i8`, `i16` per-channel | scale/zero tables as constant `f64` arrays in the data section, indexed by the `C` component (`Load` only; collapse never flattens it) | not stored |

### `Loop_stmt.t`

| Constructor | Lowering |
|---|---|
| `Alloc (a, n)` | a fixed region in the local-array area, reused across keys; nothing is allocated at run time. |
| `Array_set` | `f64.store` into it. |
| `Assign`, `Assign_index`, `Assign_index_of_i64` | `local.set`; the last bounds the `i64` into `int32` first (as `Idx.of_big_bounded`). |
| `For {var; lo; hi; body}` | `block`/`loop`, half-open; `hi` evaluated once on entry. |
| `If` | `if/else`. |
| `Fail_if (p, f)` | `if p { write record; return kind }`. Order of fields and the check sequence are the interpreter's. |
| `Store`, `Store_flat` | per the table above. |
| `Mark` | nothing (testable counting is a separate build mode, W4.3). |
| `Charge_scan_update`, `Reserve_scan_state`, `Release_scan_state`, `Reset_meter` | two `i64` globals/locals (`scan_remaining`, `scan_live`) and the same checks as `Loop_js`. |

### `Loop_failure.t`

All failure kinds keep `Loop_js_failure.Kind` as the closed vocabulary. The Wasm
error record is **the C record** (`Loop_c_runtime`'s `struct model_error`),
unchanged, so `Loop_c_failure.decode` decodes both: see the ABI.

## ABI v1

A kernel module is self-contained.

- Memory: one linear memory, **exported as `memory`**, defined by the module
  (min pages from the memory plan; no import, no growth by generated code).
- Exports: `loop_kernel(b0, b1, …, bN-1 : i32) -> i32`, one pointer per program
  buffer in program order (`Loop_js` order). Returns `0` on success and `1`
  after writing the failure record, whose `kind` is the index in
  `Loop_js_failure.Kind.all`.
- Error record: at the fixed address `Loop_wasm.error_address` (`0`). Layout is `struct model_error`: `i32 kind`, `i32 invocation` (`-1` from a
  kernel; the schedule overwrites it), `i64 v[12]` (96 bytes), 104 bytes total,
  8-aligned, little-endian. The kernel writes `kind` and `v` as the C helper
  `fail_set` does and zeroes `v` first.
- Imports (only the reachable ones, module `math`): `exp`, `log`, `sin`, `cos`,
  each `(f64) -> f64`. Nothing else is ever imported, and no import is called
  per execution mark.
- Whole-model entry (W6): `model_run(weights, inputs, workspace, outputs : i32)
  -> i32` plus the same record; the C `model_run` signature with 32-bit
  pointers. Versioned by an `abi` custom-section string `loop-wasm/1` that the
  host checks before instantiating.

Failure is a returned status plus the record, never a trap. A trap in generated
code is a defect: every division, conversion and address is guarded or proven
first, and linear memory bounds are an *additional* net, not the semantic check
(logical out-of-range inside allocated memory stays a typed failure).

## Numeric policy

- Working floats are `f64`; `Round_f32` is the only narrowing. No `f32`
  arithmetic, no FMA, no reassociation, no relaxed instruction.
- `i64` is modular two's complement. Index values are `i32` and every
  overflowing `Add`/`Scale` is a typed failure before it can wrap.
- NaN payloads are not compared (`Loop_check` policy); `-0` and infinities are.
- Transcendentals: `exp log sin cos` are imports bound to the host's `Math`, the
  same libm the JS backend uses, so a Wasm/JS difference is a bug, while a
  Wasm/native difference is the existing documented libm disagreement
  (`Loop_check` tolerance, unchanged). No tolerance is added.
- `Erf` is the polynomial, not a libm `erf`.

## Memory and size policy

- wasm32 only, `max` pages unset by generated code; total linear memory is
  capped at **2 GiB** (`32768` pages) so every `base + index * cell` stays in a
  positive `i32` and cannot wrap. The cap is enforced in checked `int64`
  arithmetic by the memory planner before any pointer is narrowed.
- A module exceeding the cap, or a buffer whose `numel * cell` does not fit, is
  refused with a typed error.
- Generated code never grows memory.
- Alignment: every buffer base is aligned to its cell size; the error record
  and local-array region to 8.

## Whole-model module

`Loop_bundle_wasm.build` composes a `Loop_bundle.t` the way `Loop_bundle_c`
does, and shares its planners: `C_payload_layout` (weights, inputs and outputs
files) and `C_workspace_plan` (arena pools, then a scratch region holding every
invocation's local arrays and carved buffers). The payload files are therefore
byte-identical to the C backend's except for the identity in their headers.

- Admitted configuration (frozen): `Separate` layout, `Borrowed` constants and
  inputs (`Loop_bundle_wasm.default_config`). Anything else is a typed refusal
  before any byte is made. There is no fallback in this backend: a model that
  builds has every scheduled invocation generated.
- Kernels are the same functions `Loop_wasm.kernel` makes, interned by complete
  body, taking `(local, b0, ..)` so storage offsets are arguments and two
  invocations of one op shape share a function (MobileNetV2: 415 invocations,
  123 kernels, as the C backend).
- `model_run(weights, inputs, workspace, outputs) -> i32` takes absolute
  addresses in the one linear memory. Per invocation it zeroes the scratch
  carves and workspace outputs it owns (`memory.fill`), fills synthetic
  operands (`Fill_f32`), calls the kernel and, on a nonzero status, stores the
  failing invocation's position at the record's `invocation` slot and returns
  `1`. After the schedule it `memory.copy`s each graph output (aliases,
  duplicates and forwarded inputs included) into the outputs region.
- The module's memory is sized for the default `Placement` (static bytes,
  weights, inputs, workspace, outputs, 64-aligned) and capped at 2 GiB; a host
  may place the regions elsewhere in memory it provides. The module's own bytes
  are the error record and per-channel constant tables.
- `identity` is the digest of the encoded module and keys the payload headers.
- Host: `Wasm_host` (native, `lib/loop_wasm_exec`) writes `model.wasm`,
  `weights.bin` and a header-only outputs template, runs a node runner
  (`runner.js`, embedded) that places the regions, runs `model_run` and
  optionally repeats it on the same instance (outputs must stay identical), and
  prints the failure record (exit 5) or a timing line (compile, instantiate,
  copy in, first run, warm run, copy out).

## In-process host (js_of_ocaml)

`js/loop_wasm_host` (mirrored in `js/jsoo/loop_wasm_host_js`) generates the
module with the JavaScript build of the compiler and runs it through the host's
own `WebAssembly`: no process, file or external tool. `prepare` compiles
synchronously (node, workers); `prepare_async` uses `WebAssembly.instantiate`
for a browser main thread and calls its continuation once, on a later turn.
Memory is allocated once and never grown; views are retaken every run and
dropped by `dispose`. Weights are placed once at preparation; each run places
the inputs, calls `model_run`, and copies outputs into fresh tensors. A trap is
`Js_exception`; a failure record decodes through `Loop_c_failure` (mirrored)
into the interpreter's row with the invocation position.

js_of_ocaml's stack is far shallower than a native one: every list in the Wasm
lowering that can run to thousands of elements uses tail-recursive append and
map (`Loop_wasm_ctx.( @ )`, `Wasm.Instr.map_calls`), and a 60,000-statement
kernel in `test/loop_ir/loop_wasm_test.ml` guards it (stock `@` overflows).

## Features and manifest

`Wasm_features.of_module` scans a module for post-MVP features (bulk memory,
non-trapping float-to-int, sign extension; SIMD is reserved) and `probe`
builds a tiny module using exactly one, which a host validates to detect the
extension (`test/wasm_ir/features.t`: all four validate under node 20.19.2, and
a truncated probe does not). The scalar backend needs `bulk-memory` only for a
whole-model module and `nontrapping-float-to-int` only where a program converts
a float to I64; it never needs SIMD or a relaxed instruction.

Every module carries an `abi` custom section (`loop-wasm/1`) and a `manifest`
section: the ABI, required features, imports, helper functions and the numeric
policy (`working=f64 f32=round-and-widen fma=none reassociation=none
i64=modular index=i32-checked`). The module's digest is its identity and covers
the manifest, so a change of ABI, helper set or policy changes the identity.

## C-to-Wasm baseline

`Wasm_c_host` compiles the C backend's whole-model unit (`model_infer.c`, plus a
three-export wrapper: `run`, `error_ptr`, `heap_base`) with Clang against a
wasm32 libc, links with `wasm-ld`, sizes the memory from the linked
`heap_base`, and runs it through the same node runner and payload files as the
direct route. Toolchain: the installed Clang 19.1.7 and `wasm-ld` plus the
Debian packages `wasi-libc` and `libclang-rt-19-dev-wasm32`
(`scripts/wasi-sysroot-userland.py` unpacks them without root; `WASI_SYSROOT=/usr`
after an apt install). Flags are strict scalar: `-O2 -ffp-contract=off
-fno-strict-aliasing -fno-vectorize -fno-slp-vectorize -mno-simd128`; the
generated assembly is scanned for vector mnemonics (0 in every model run) rather
than assuming the flags suffice. The link needs `-nodefaultlibs -lc -lm` and the
builtins archive named explicitly, since the installed Clang's resource
directory has none for wasm32. `make wasm.c.pt2.runtest` runs it over the CI
models; quantized programs, which the C emitter refuses, remain obligations of
the direct route only.

## Browser

`make wasm.browser.runtest` drives Chromium (playwright) over a page that loads
the JS build of the compiler. Evidence recorded for Chromium 141.0.7390.37
headless shell on Linux AArch64: fixtures generated in the page and compiled
through `WebAssembly.instantiate` match the reference; a real model (the native
compiler's `--export` artifacts, 43,695-byte module) runs twice on one instance
with a dirty workspace and its outputs are byte-identical to node's (first run
~343 ms, warm ~176 ms, memory 13.7 MB). Deployment limits observed:

- A Content-Security-Policy needs `'wasm-unsafe-eval'` in `script-src` for the
  module to compile; without it `prepare_async` returns the typed
  `Wasm_compile` error naming the refusal. Unlike the generated-JS backend, no
  `'unsafe-eval'` (`new Function`) is needed.
- This Chromium accepted a main-thread synchronous compile of the 43 KB module;
  the platform may refuse larger or any synchronous compiles on the main thread,
  so a browser must use `prepare_async`.
- Node passing is not browser evidence: the page is the only place these
  were observed.

## Measurements (scalar)

Matched model, input and storage plan; Linux AArch64 (NEON, no SVE), Node
20.19.2, gcc 14.2.0, Clang 19.1.7; one sample; warm = median of ten repeats on
one instance with a dirty workspace; times in ms. Warm figures include no
copying; the Wasm routes' first run includes V8 tiering and the host-side
`copy_in` (weights and input placement) is listed separately.

| route | model | module B | memory B | compile | instantiate | copy in | first run | warm |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| native C (`gcc -O2`) | mobilenetv2_050 | | | | | | 65.7 | 62.3 |
| C compiled to Wasm | mobilenetv2_050 | 95,476 | 14,734,272 | 0.25 | 0.04 | 4.4 | 176 | 77.2 |
| direct Wasm | mobilenetv2_050 | 42,471 | 13,685,696 | 0.21 | 0.04 | 5.2 | 235 | 88.2 |
| generated JavaScript | mobilenetv2_050 | | | | | | 279 | 149 |
| native C (`gcc -O2`) | fastvit_sa12 | | | | | | 944 | 935 |
| C compiled to Wasm | fastvit_sa12 | 144,871 | 57,762,624 | 0.38 | 0.26 | 25.0 | 2398 | 1119 |
| direct Wasm | fastvit_sa12 | 63,123 | 56,714,048 | 0.45 | 0.08 | 27.0 | 2247 | 1126 |
| generated JavaScript | fastvit_sa12 | | | | | | 1976 | 1703 |

Reading it: the direct emitter's module is 2.2x smaller than Clang's and, after
the second unit-loop round, within 14% (mobilenetv2_050) and 1% (fastvit_sa12)
of Clang-compiled scalar Wasm warm; direct Wasm is 1.4x (mobilenetv2_050) and
1.2x (fastvit_sa12) slower than native scalar C on the same host and 1.7x /
1.5x faster than the generated JavaScript. `compile` is `new WebAssembly.Module`:
V8 compiles lazily and tiers up, so it says little; `first run` is the honest
cold figure. Before the second unit-loop round the direct route's warm run was
164 ms (mobilenetv2_050). Timings were taken with nothing else running; a run
beside other work moved them by tens of percent.

## Reproduction

| Question | Command | Needs |
|---|---|---|
| Op table, fixtures, every `Loop_check` fixture and op sweep through emitted Wasm, C-compiled route, marks, feature probes, native/jsoo byte agreement | `make wasm.runtest` | node, wasm32 libc (`make wasm.toolchain` unpacks one without root) |
| In-process host under node, promise preparation | `make wasm.jsoo.runtest` | node |
| Does each deliberate defect turn the suite red? | `scripts/wasm-mutation-check.sh` | as `wasm.runtest` |
| Real models, direct emitter vs reference | `make wasm.pt2.runtest` | downloaded models, node |
| Real models, C compiled to Wasm vs reference | `make wasm.c.pt2.runtest` | as above plus the wasm32 libc |
| Model generated in JavaScript, byte-identical to native | `make wasm.jsoo.pt2.runtest` | one downloaded model, node |
| In a browser | `make wasm.browser.runtest` | playwright Chromium (its system libraries are in the devcontainer image) |
| Phases and warm repeats | `make wasm.pt2.bench` (direct), `loop_wasm_pt2 --via-c --bench=N` | downloaded models |

`loop_wasm_pt2` flags: `--strict` (release ranking), `--shadow` (bitwise vs the
per-node reference), `--poison` (workspace and outputs), `--bench=N`,
`--via-c [--cflags=FLAGS]`, `--export=DIR` (artifacts for another host),
`--wat=FILE` (readable dump of the generated module), `--keep=DIR`.

## Kernel module layout

The module defines and exports `memory`. Bytes `[0, heap_base)` are the
module's own: the 104-byte error record at 0, then per-channel quantization
tables (constant `f64` data segments), then local arrays (`Alloc`, zeroed by the
statement each call). A host places buffers at or above `heap_base`
(`Loop_wasm.t.heap_base`, 16-aligned) and passes their absolute addresses; the
static region is capped at 1 GiB, inside the 2 GiB memory policy.
`Loop_wasm.lower` returns the module with the smallest memory;
`Loop_wasm.with_pages` sets the host's.

Index arithmetic is `i32`. Overflow checks form each node's value in `i64` from
operands that already passed their own check. Calls in a kernel body are
symbolic (`Loop_wasm_runtime.Callee`) and renumbered once the reachable imports
and helpers are known, so unreached helpers are never emitted.

## Verification

- `make wasm.runtest` (needs node; outside `runtest`): the op table executed
  under node against a JavaScript reference (`test/wasm_ir`), a structural
  fixture, and every `Loop_check` fixture and op-sweep program run through the
  emitted module under node (`test/loop_wasm`, the same sources
  `test/loop_ir` and `test/loop_c` run, copied). A deliberate defect in the
  lowering (a swapped operator, a dropped `Round_f32`, a signed bounds compare,
  an off-by-one quantization zero point) turns the suite red.
- `test/loop_ir/loop_wasm_test.ml` pins module sizes and digests for one module
  per failure constructor and for whole-model fixtures, and runs under
  `make jsoo.inline-runtest` too, so native and 32-bit-`int` output agree byte
  for byte.
- `make wasm.jsoo.runtest`: the in-process host's fixtures under node, and the
  promise-based preparation gate. `make wasm.jsoo.pt2.runtest`: a real model
  generated and run inside the JS build, with the module it emitted
  byte-identical (same digest) to the native compiler's.
- `make wasm.pt2.runtest` (downloaded models): every CI model through
  `model_run` under node, every graph output bitwise equal to the per-node
  reference (`--shadow`), the release ranking (`--strict`), workspace and
  outputs poisoned first. `make wasm.pt2.run` is `fastvit_sa12` alone and
  `make wasm.pt2.bench` repeats the schedule on one instance.

## Where it lives

- `lib/wasm_ir`: `Wasm` types, `Wasm_op` table, `Wasm_check`, `Wasm_encode`,
  `Wasm_wat`. Pure,
  `fmt` + `err_trace` only, js_of_ocaml-safe (lengths and constants in
  `int32`/`int64`, never a 63-bit `int`).
- Loop-to-Wasm lowering (`Loop_wasm`, `Loop_wasm_value`, `Loop_wasm_fail`,
  `Loop_wasm_ctx`), helpers (`Loop_wasm_runtime`) and the record layout
  (`Loop_wasm_failure`) sit beside `Loop_js` in `lib/loop_ir`, native- and
  jsoo-reachable (mirrored in `js/jsoo/loop_ir_js`). Bundle composition follows.
- Host execution stays out of every pure library, as `lib/loop_c_exec` and
  `js/loop_js_exec` do: `lib/loop_wasm_exec` runs a module under a node
  subprocess (native only); an in-process `WebAssembly` host is separate.

## Binary32 kernels and relaxed SIMD (pointer)

The scalar and SIMD contracts above are the `Reference_f64` policy and stay the
default. Under a `Simd_fp32_*` policy a vectorized kernel uses `f32` locals and
`f32x4` registers (four per sixteen-lane vector), and `f32x4.relaxed_madd` when
the plan was made for `Loop_target.wasm128_relaxed` and the engine validates the
`relaxed-simd` probe. The manifest names the policy and the working precisions.
See the fp32 design record in this directory.
