# Inference comparison: variants, engines and the coverage manifest

How `make model.compare` (and `fixture_pair_run`) picks what to compare. The
coordinator layers never see an engine name (see the integration doc); this is
the layer that turns two words into two endpoints, a mapping and the initial
values, and the record of what that vocabulary covers.

## A variant is a graph view, an engine and a configuration

Spelling is `VIEW` or `VIEW:WORD,WORD...`. Views, with the rank that orders
composition: `aten` 0, `native` 1, `native.canonical` 2, `native4d` 3.

| Words | Valid on | Meaning |
|---|---|---|
| (none) | every view | the view's own evaluator |
| `sequential32` | native, native.canonical | binary32 sequential dots; the direct evaluator's policy only, with its own identity |
| `release`, `arena`, `arena=BYTES`, `separate`, `shared`, `scheduled` | native, native.canonical | storage and scheduling configurations of the Native evaluator over the same graph; `scheduled` needs a storage word; one storage word only |
| `symbolic` | native, native.canonical, native4d | the symbolic evaluator, grounded from the coordinator's bindings; takes no other word |
| `loop-c`, `ssa-c`, `loop-wasm`, `ssa-wasm` | native, native.canonical | generated C (host compiler) or Wasm (node); Loop emitter or SSA backend; takes no other word |
| `fp32` | with an engine word below | the engine's binary32 numerics (`Fp32_ordered`: every kernel binary32): `loop-c,fp32` is the provider `loop-c-fp32`; offered by `loop-c`, `ssa-c`, `loop-wasm`, `ssa-wasm`, `rivet*`, `compcert`, `compcert-rv32` and `compcert-coral-*`. Two fp32 engines are compared exactly unless `TOL` is set. For `rivet*` it is the configuration as deployed: planned for neon128, binary32 vector loops where the policy keeps each output's order |
| `fp32s` | with a `rivet*` engine word | the same policy scalar (`rivet,fp32s` is `rivet-fp32s`), so `rivet-fp32` against `rivet-fp32s` checks the vector path against the scalar one exactly |
| `rivet`, `rivet-table` (each with `-scanned`) | native, native.canonical | Rivet AArch64: the image route, or the table host, with the reference or the scanned allocator |
| `compcert` | native, native.canonical | the embedded CompCert, assembled and loaded by Rivet |
| `compcert-rv32` | native, native.canonical | the embedded F32 ABI riscv32 CompCert under qemu-riscv32 with D disabled; requires `fp32`, otherwise refuses before compilation |
| `compcert-coral-mpact-v2`, `compcert-coral-core-mini-highmem-axi` | native, native.canonical | the embedded F-only riscv32 CompCert on a Coral NPU simulator configuration (the archive `CORALNPU_DATA` names); `fp32` only, the word alone refuses (no D extension) |
| `loop-js`, `ssa-js`, `jsoo` | native, native.canonical | generated JavaScript, or the Native evaluator compiled by js_of_ocaml, run by node in a helper process |

An engine word is registered at run time, so the vocabulary is what the
executable was built with. A word whose engine was not built still parses and
refuses at instantiation naming the prerequisite (exit 3); it is never skipped
and never replaced by another engine. A combination a view cannot carry
(`native4d:arena`, `aten:symbolic`) is invalid input (exit 2).

## Layering

`lib/inference_compare_pt2` owns variant parsing, the capability rules, graph and
mapping construction, and `Provider`, the registry of engine words. Libraries
above it register providers: `inference_compare_generated` (C, Wasm),
`inference_compare_rivet` (image route, table host), `inference_compare_compiled`
(CompCert), `inference_compare_rv32` (riscv32 CompCert under qemu and on the
Coral simulators), `inference_compare_js` (JavaScript). Rivet is its own library, so a
Rivet-only install selects it without CompCert; CompCert needs the embedding for
the host ISA (`MLTORCH_COMPCERT`). `lib/inference_compare_host` holds the
production runner (`Pair_run`); the test directory `inference_compare_providers`
registers every provider for the driver and the manifest test.

A provider has `available` and `instantiate`. `instantiate` resolves the engine
identity (compiler banner or node and helper version, effective flags, numerics,
pipeline, allocator, ISA, CPU features) once, before any body is prepared, and
the runner closes over it, so the identity the coordinator reads is the one the
run used.

## Mappings and initial values

A pair's mapping is a chain of `Import_map` (ATen to Native), the canonicalize
pass map and the lower pass map, composed with `Compose.compose`. A pass map is
read in reverse by swapping source and destination, so any pair of views works
in either orientation except with ATen on the right (the importer's
correspondence is ATen to Native). A view paired with itself uses an identity
pass map over two copies of the graph. Initial values come from the surviving
importer bindings and the pass pins. Each node or cluster is a unit fed the
donor's inputs, so a difference shows only in the unit that holds it.

Views are built independently: a model Native4D refuses still has `native` and
`native.canonical`, and the `native4d` variant refuses naming the tensor.

## Policy and exit codes

The runner installs a policy per comparison site from the site's codec: exact
for bool and integer, the declared floating rule for floats (a single floating
policy is refused for a bool codec). Exit codes: 0 agreement, 1 disagreement, 2
invalid input, map or policy found before execution, 3 refused or failed
(including a missing prerequisite), 4 not established or interrupted. A
`prepare` error is a refusal with zero executions; a numerical disagreement is a
result, not a failure.

## The JavaScript helper

The JavaScript executors exist only under js_of_ocaml, so they run out of
process. `js/jsoo/compare_js_engine` holds them and the Native evaluator;
`lib/inference_compare_js` starts it under node and speaks the protocol of
`lib/inference_compare_js_wire` over stdin and stdout.

- Framed messages: a 4-byte length, then the magic, the version, a kind byte and
  the kind's fields. A message of another version, truncated, with trailing
  bytes, an unknown kind, or a length over the limit is rejected whole.
- Requests: hello, prepare (handle, engine, the graph as JSON), run (handle,
  inputs by tensor id), release. Answers: welcome (engines, runtime), prepared,
  refused, outputs in graph order, failed, released.
- Tensors carry a format tag, six extents, a byte count and the little-endian
  cells exactly as held (f32, f64, i32, i64, bool), so integers above 2^53 and
  NaN payloads survive. Other formats are refused when sent.
- A body is prepared once under a handle and run any number of times. One
  helper process serves every body of an engine and ends with the last release.
- The helper runs the graph it is given; it never relowers an archive and never
  falls back. An engine that cannot take a body refuses at prepare.
- Identity: the helper file's digest, the protocol version, the node command and
  the version the helper reports. The helper is found next to the build tree or
  by `MLTORCH_COMPARE_JS`.
- The generated-JavaScript engines refuse int64 values (their own limit); `jsoo`
  carries them.

JavaScript's math library is not the C library's, so GELU-like clusters differ
from the direct evaluator in the last bit: a comparison at zero tolerance
disagrees, within the default tolerance it agrees.

## The coverage manifest

`test/inference_compare_coverage/coverage.manifest` lists a representative pair
for every mode with its expected result on `test_convnext2`: `agree`,
`disagree`, `invalid`, or `refused TEXT`, with optional output checks.

- `manifest_test` runs on every `runtest`: ids are unique, every variant
  parses, every word of the vocabulary (the views, the configuration words and
  every registered engine, including `arena=BYTES`) appears in some row, and no
  row names a word outside it. Adding an engine word without a row fails it.
- `make model.compare.coverage` walks the rows against the real model. A row
  whose engine is not built or whose prerequisite is missing is reported
  pending with the reason, never passed; `MLTORCH_COVERAGE_STRICT=1` fails on
  pending. The rows assert, among others: view pairs adjacent and not, both
  orientations, both donors; `sequential32` agrees within the tolerance and
  differs bitwise; the arena row uses and reuses storage and an over-budget
  budget refuses with zero executions; Native4D storage words are invalid; the
  four-axis evaluator against each compiled engine (the 40 clusters are the
  four-axis units).

## Tests by cost

- Always (`make runtest`): parsing, mapping, capability, wire protocol and
  manifest tests; the C engines (a missing gcc fails them).
- `MLTORCH_WASM` (`make wasm.runtest`): Wasm engines.
- `make js.compare.runtest` (`MLTORCH_JS`): JavaScript engines and cram.
- `make rivet.compare.runtest`, `make compcert.compare.runtest`: AArch64 hosts.
- `make compcert.rv32.runtest`: the qemu engines against the host C engines,
  bitwise, and the Coral words' refusals and identity;
  `make coralnpu.compcert.runtest` the Coral engines against `loop-c-fp32`.
  A Coral identity names the configuration ID, the simulator's SHA-256, its
  command and the call deadline. Whole `test_convnext2` agrees on MPACT; the
  highmem RTL testbench (about 2e5 cycles a second) is too slow for a whole
  model, so its manifest rows are refusals and the hermetic bundles cover it.
  `compcert-rv32` without `fp32` refuses before compilation: the RISC-V
  CompCert integration uses only the F32 ABI extraction.
- `make model.compare.coverage`: the whole manifest, about four minutes.

## Not covered, and why

- x86-64 Rivet (table host, process under CPU or qemu-user): the wrappers are
  not written and this host cannot verify them.
- `native4d` with a compiled or JavaScript engine of its own: the compiled
  backends take Native operations only (`Loop_bundle.build` takes the Native
  graph); the cross-dialect pair `native4d` against a compiled Native view
  works. Native4D storage and scheduling words refuse.
- `system_libm` runtime, the `Simd_fp32_*` policies and the vectorized planned
  SSA pipelines: the generated engines expose the reference binary64 numerics
  with the exact SSA pipeline, and `fp32` (`Fp32_ordered`; vectorized on
  Rivet, `fp32s` scalar there).
- Loop and SSA interpreters and Machine IR stages as engines; C-generated Wasm
  (`--via-c`); GNU assembly coherence checks.
- Whole-graph calibration (segmented against whole run) for the variants: the
  CompCert/assembly pair has it in its own driver.
- Constant-role storage ownership: extracted bodies describe every source as an
  input, so no ownership option is offered.
- An output-perturbing endpoint wrapper for detection tests on real graphs;
  detection is shown with perturbed inputs and by the pair tests' flipped
  fixtures.
- Source-aware experiment export and replay of the new variants: the evidence
  engine specs name aten, native, native4d and synthetic only.
