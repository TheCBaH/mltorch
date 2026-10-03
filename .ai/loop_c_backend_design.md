# C backend — design record

The host-compiled C backend for the Loop IR, in two layers.

**Kernels.** `Loop_c` emits one kernel per `Loop_program`, `Loop_c_runtime` holds
the shared ABI declarations and helpers, `Loop_c_failure` decodes a failure
record. `lib/loop_c_exec` (native only, `unix`) compiles and runs emitted C as a
subprocess; `test/loop_c` runs the `Loop_check` fixtures and op sweep against it.

**Whole model.** From a `Loop_bundle.t` built with `Loop_bundle_c.default_config`
(separate layout, borrowed constants and inputs, output-only retention):
`C_payload_layout` (payload files), `C_workspace_plan` (arena pools plus a shared
scratch region), `Loop_bundle_c` (`model_infer.c`: interned kernels and
`model_run`, the fixed schedule), `C_main_gen` (`model_main.c`), and
`Loop_c_exec.Host` (generate, pack, compile, run, decode). Exactly two C files
are generated; the binary needs only libc and libm, no OCaml, JavaScript or
model JSON. Kernels are interned by complete text, so a shape seen many times is
one function; an invocation's buffers bind to fixed byte offsets in the weights
file, inputs file or workspace. `model_run` does no I/O and no allocation and
keeps no global state; a failure carries the invocation position, so a shared
kernel still names which call failed.

## Using it

Prerequisites: `gcc` (or another compiler taking the same flags), a 64-bit
little-endian POSIX host. Not part of `make runtest`'s model data, but
`make runtest` itself runs `test/loop_c` and needs the compiler.

| Command | What |
|---|---|
| `dune build @test/loop_c/runtest` | kernel differential suites, host process tests (in `runtest`) |
| `make c.runtest.o0`, `make c.runtest.san` | the same at `-O0` and under ASan+UBSan |
| `make c.pt2.runtest` | CI gate: the seven CI models (cram set + csatv2), all outputs bitwise vs the reference |
| `make c.pt2.run` | `fastvit_sa12` the same way (manual; reference alone ~100 s) |
| `make c.pt2.san` | both models, generated binary under ASan+UBSan |
| `make c.pt2.bench` | phase timings and 20 warm repeats |

Artifact directory (`--keep=DIR` on `bin/loop_c_pt2.exe`, or `dir` to
`Loop_c_exec.Host.prepare`): `model_infer.c`, `model_main.c`, `weights.bin`,
`model` (and `compile.log` after a failed compile, sources left in place). The
binary runs on its own: `model --weights W --inputs I --outputs O [--poison]
[--time] [--repeat N]`. `--poison` fills the workspace with 0xA5 first, to expose
a read of stale memory. Reuse: prepare once, then `Host.run` per input; only the
inputs file changes.

Limits: quantized formats (`i8`/`i16`) and any non-`Separate`/`Borrowed`
configuration are typed refusals before any C is written (`Loop_c.error`,
`C_workspace_plan.error`); dynamic shapes, other platforms, fusion, SIMD and
persistent processes are not attempted. There is no fallback path: a model that
is not fully admitted does not compile.

## Verified

Both intended models produce every output bitwise equal to the per-node
reference (`Eval_direct.run`), with the workspace poisoned, and run clean under
ASan+UBSan: `mobilenetv2_050` (415 invocations, 123 distinct kernels) and
`fastvit_sa12` (716 invocations, 132 distinct kernels, Region-authored ops).
Zero invocations fall back. Gates proven live: emitting `+` as `-` turns the
mobilenet shadow red; `floor_div` truncation turns 26 kernel comparisons red.

Measured (gcc 14.2, `-O2 -ffp-contract=off`, 8-core container, one thread, one
sample): `mobilenetv2_050` weights 7.9 MB, workspace 5.1 MB, source 182 KB,
generate+pack+compile ~1.05 s, inference warm ~63 ms, a fresh process run
~80 ms, reference evaluator ~16.5 s. `fastvit_sa12` weights 46.5 MB, workspace
9.6 MB, source 269 KB, generate+pack+compile ~1.9 s, warm ~947 ms, fresh run
~960 ms, reference ~103 s. The whole-model JS bundle recorded 196-212 ms and
2.45-2.50 s warm for the same models (Node 20, a different measurement, its copy
in/out inside the figure): the C figures are ~3x faster warm; this is a
comparison of one run each, not a benchmark suite. Pack of inputs and decoding
of outputs are each well under a millisecond for these models; the fresh-process
overhead beyond the inference unit is mapping and publication.

Reproduce: `make c.pt2.bench`.

## Frozen contracts

Changing any of these is a versioned change.

## Toolchain policy

`gcc` (or `clang`), `-std=c11 -O2 -ffp-contract=off -fno-strict-aliasing -Wall
-Wextra -Werror`, linked with `-lm`. No fast-math. Every generated unit must
compile warning-free at `-O0`, `-O2`, and under
`-fsanitize=address,undefined -fno-sanitize-recover=all`. The host is checked at
generation (64-bit little-endian, `sizeof(double) == 8`); a host that fails is
refused. `LOOP_C_CFLAGS` replaces `-O2` in the test executor only.

## Kernel ABI (`Loop_c`)

`static int kernel(struct model_error *err, double *local, <buffers>...)`.
Returns 0, or 1 after filling `*err`. One pointer per program buffer in
program order, cell type by format (`bf16`/`f16` `uint16_t`, `bool` `uint8_t`,
`f32` `float`, `f64` `double`, `i32` `int32_t`, `i64` `int64_t`); an `Input`
buffer is `const`. Quantized formats (`i8`, `i16`) are a typed refusal
(`Loop_c.error`). `local` provides `local_doubles` doubles: every `Alloc` gets
its own region in program order and is zeroed where the statement executes.

## Numeric contract

Floating carriers are `double`; `Round_f32` is `(double)(float)x`; an F32
store is `(float)x`; a Bool store is `x != 0.0`. Index arithmetic is `int64_t`;
the int32 index domain is enforced by explicit `Fail_if` nodes exactly as in the
interpreter, post-order, first node to leave the domain reported with its
operands. `Floor_div_pos`/`Ceil_div_pos` floor and ceil (never C truncation).
I64 add/sub/mul are modular through `uint64_t`; `i64_div` and `i64_from_float`
are total (a `Fail_if` precedes each in a valid program). `Float_max` is
`Math.max`'s rule (NaN propagates, +0 above -0); `pool_better` is `value > best
|| value is NaN`; `erf` is the Abramowitz-Stegun approximation of the reference,
not libm's. Transcendentals (`exp`, `log`, `sin`, `cos`, `sqrt`, `trunc`) are the
host libm's.

Comparison policy: values are compared bitwise (a NaN equals any NaN, as in
`Loop_check.tensors_equal`); integer and boolean results are exact. Library-math
differences are not presumed absent: none has been observed in the 372 programs
of the C2 differential suites, and any that appears is reported separately per
operation, never absorbed into a tolerance.

## Failure record

`struct model_error { int32_t kind; int32_t invocation; int64_t v[12]; }`.
`kind` is the position in `Loop_js_failure.Kind.all` (closed, alphabetical, hence
stable); `v` holds that kind's `Loop_js_failure.fields` in order, a `Coord`
taking six slots, a `Value` the `double`'s bit pattern, `Which`/`Op`/`Cached` a
0/1 (`Lane`/`Row`, `Add`/`Mul`, `State_over_limit`/`Updates_exhausted` are 0/1).
The kernel leaves `invocation` (-1) for the schedule to fill.
`Loop_c_failure.decode` maps a record back to the interpreter's row.

## Payload file, version 1

All multi-byte values little-endian. Header, 64 bytes:

| Offset | Size | Field |
|---|---|---|
| 0 | 8 | magic `"MLTCPAY\0"` |
| 8 | 4 | version = 1 |
| 12 | 4 | role: 1 weights, 2 inputs, 3 outputs |
| 16 | 8 | total file length in bytes |
| 24 | 8 | data alignment = 64 |
| 32 | 16 | identity: MD5 of the artifact's layout description |
| 48 | 16 | reserved, zero |

Tensor data starts at offset 64; every tensor offset is a multiple of 64 and
padding is zero. A zero-tensor set is exactly the 64-byte header. The identity is
an incompatibility check, not an integrity check.

## Host process contract

`model --weights W --inputs I --outputs O`. Exit codes: 0 success (`O` published
by rename); 2 command line; 3 payload/I-O (size, header, role, identity,
aliasing); 4 allocation; 5 inference failure. On 5 the last stderr line is
`model_error: <invocation> <kind> <v0> ... <v11>` (decimal). `O` is written to a
temporary sibling and renamed only on success. A signal exit is the host's to
classify.

## Refusals

`Loop_c.error`: `Unsupported_format` (quantized buffer). Bundle-level refusals
(quantized arenas, retained intermediates, non-`Separate` layout, a host that is
not 64-bit little-endian) are added in C3-C5 as typed errors.

## Model coverage

Bitwise equal to the reference, one sample, 2026-09-29: `csatv2` (1284
invocations), `edgenext_xx_small` (394), `efficientnet_b0` (583), `fastvit_sa12`
(716), `mobilenetv2_050` (415), `mobilenetv3_small_050` (387), `mvitv2_tiny`
(991), `regnetx_002` (368), `test_convnext2` (106). Not verified:
`mobilevitv2_175` (generates and compiles, 580 invocations; the reference
evaluator did not finish in 20 minutes, so no comparison) and
`vit_small_patch16_dinov3_qkvb` (not downloaded). CI runs the cram set plus
`csatv2`; the rest are manual.

## Binary32 kernels (pointer)

The numeric contract above (binary64 carriers, `Round_f32` as a cast pair) is the
`Reference_f64` policy and stays the default. Under `Simd_fp32_ordered` or
`Simd_fp32_relaxed` a kernel the planner vectorizes is emitted in `float`
(constants `f`-suffixed, `FLT_EVAL_METHOD == 0` asserted, float helpers,
sixteen-lane `v16sf` vectors, `fmaf` multiply-adds, local arrays packed two to a
scratch double); every other kernel is emitted exactly as before. See the fp32
design record in this directory, and `loop_c_pt2 --numerics`.
