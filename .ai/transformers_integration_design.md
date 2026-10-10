# Transformers artifact integration — design

Status (2026-10-09): proposed; assessment and planning are complete, consumer
implementation has not started. Working stages and evidence live in `ai/`:
[implementation plan](../ai/transformers_integration_implementation_plan.md)
and [tracker](../ai/transformers_integration_implementation_tracker.md).

## 1. Goal and delivery boundary

Consume the exported tensor graphs published by
`TheCBaH/devcontainer.transformers` through mltorch's existing PT2 and Native
pipeline. Obtain checkpoint captures from the producer's version-2 map, bind
named tensor inputs, and compare every named output with published cases.
The runtime remains OCaml; Transformers and torch are producer dependencies.

The first executable milestone is numerical replay of original-size
MobileViT forward and the SmolVLM connector from verified slim bundles.
The next milestone adds BERT/TinyCLIP and selected static generation components
as their importer and execution prerequisites become available. The admission
matrix records remaining failures; this design does not promise that all
producer artifacts fit Native, Native4D, or Kernel.

Full task execution is a later, explicit milestone. Tokenization,
preprocessing, embedding normalization, cache routing, sampling and stopping
belong to host adapters. Dynamic shape specialization, general empty tensors,
training, quantized checkpoints and a browser inference product are deferred.
Pure metadata/map logic must remain reachable by JavaScript; initial downloads
and numerical cohort runs use the Unix host.

## 2. Pinned baseline and evidence limits

The reviewed producer source is
`81feca91b3d3ad032cb3c1ef28f4d5e1751c1d55`; the measured consumer is
`1165d2b0` with its existing workspace state. Relative to producer `19d1a65`,
24 graph JSON files are byte-identical and six static decode graphs are new.

| Measurement | Result | Meaning |
|---|---|---|
| Committed corpus | 30 tiny, random-weight artifacts | Architecture fixtures, not checkpoint inference |
| Catalogue files checked | 210 hashes and sizes | Committed content integrity |
| Native construction | 2/30 | MobileViT and SmolVLM connector |
| Native4D conversion | 2/30 | Separate branch from canonical Native |
| Kernel conversion | 2/30 | Separate branch from canonical Native |
| First failures | 16 empty captures; 9 embedding; 3 other ops | First failure per graph, not a complete missing-op census |
| Empty captures | 56, all live, shape `[0]` | Valid producer values with consumers |
| Symbolic report | No regeneration diff; six unit tests pass | Metadata reporting only |

The inspected release is `checkpoint-003207ae59ed`, produced at
`003207ae59ed0555599da190d70cc3d2f15ad705`. It indexes 35 original-size
checkpoint artifacts. Independent byte reconstruction verified all 42 BERT
and 347 MobileViT captures in sampled released v2 maps; it did not execute
their graphs or cover every conversion/origin combination. All 13 committed
checkpoint binding reports say verified with no unmapped captures.

Producer CI for the new main was queued at the assessment's last observation.
Its status is historical evidence, not a continuing claim about current CI.
Keep inspected source commit, release producer commit, release tag, upstream
checkpoint revision, graph digest and consumer commit as separate facts.

An earlier v1 release connector probe is an unresolved numerical regression:
case-00 had 2/36,864 values outside `atol=1e-5`, `rtol=1e-4`; case-01 passed.
The two failing absolute errors were about `1.57e-5` and `1.56e-5`. This is
historical evidence at its recorded graph/weights/consumer pins, not a fresh
v2 result. Preserve the oracle and tolerances when investigating accumulation
and rounding. A successfully loaded capture is insufficient for parity.

## 3. Architecture and ownership

Use existing `Pt2_archive.of_parts` and its captured-storage loader seam.
Keep the v1 `Pt2_safetensors` API and consumers intact. Add a separate v2
implementation; a valid v2 artifact never falls back to v1 after a failure.

Proposed homes, subject to the repository's normal module-size rules:

| Home | Responsibility |
|---|---|
| `lib/pt2_checkpoint_map/` | Pure v2 vocabulary, Jsont codecs, structural checks, origins and capture preparation |
| `lib/pt2_checkpoint_map_unix/` | Verified local/cache/HTTPS sources, mmap ownership and archive assembly |
| `lib/pt2_fixture/` | Pure contract/case validation and logical tensor comparison |
| `lib/pt2_fixture_unix/` | Extracted fixture reading and structured replay reports |
| `lib/native_interp/` | Named input binding, constrained empty-value normalization and PT2 operator import |
| `lib/native/`, bridges and lowerings | Genuine operation semantics and backend-specific admission |
| `bin/`, `scripts/`, Makefile | Explicit acquisition, graph sweep and replay commands |

`Pt2_storage.t`, `Pt2_tensor.t`, safetensors views and the current PT2 JSON
decoders remain the common representation. The new libraries must not import
Transformers, torch or C++ ATen. An optional ATen comparison route may use the
existing bridge in tests without making it a loader dependency.

The map core takes verified source bytes and a SHA-256 capability. The host
owns I/O, cache configuration and budgets. SHA-256 is a byte hash, not an
ETag assertion or OCaml's MD5 `Digest`. Select a portable provider with known
vectors and native/JS parity before implementing capture preparation; avoid
per-tensor subprocesses and full-sized string copies.

## 4. Discovery and fixture acquisition

Treat committed catalogue entries and released checkpoint entries as separate
populations. Preserve the full artifact ID, including population, component,
dtype, policy, shape policy and checkpoint suffix. Model name alone is not
a cache key or a selection identity.

Pin a publication index's bytes/digest in a consumer manifest. Follow its
archive and manifest URLs; verify declared byte size and SHA-256, then verify
the exact member set and every member pin before opening the extracted graph.
Use an explicit download command; hermetic tests and replay never fetch.
Use atomic promotion of verified files into a content-addressed cache.

Initial replay accepts an extracted fixture directory. A download helper owns
tar extraction and rejects duplicate, absolute, escaping and linked members;
it enforces aggregate and per-member bounds before extraction. Do not add a
general tar implementation to `lib/pt2` for this integration.

An artifact may have no `model.pt2`. Slim checkpoint bundles contain graph,
weight/constant configs, captures, cases and `models/safetensors.v2.json`.
Tiny random fixtures retain their full program as the weight source. Neither
path synthesizes a program or obtains tiny weights from a reference checkpoint.

## 5. Version-2 map contract

The pinned producer [specification][map-spec] and [schema][map-schema] define
the wire format. Decode closed origin/conversion variants explicitly. Treat
record fields and identifiers as typed domains in public signatures, use
`Err` with data-bearing errors, and keep third-party codec text at a named seam.

Before source I/O:

1. Require schema 2, an empty `unmapped`, and matching artifact ID and raw graph
   SHA-256. Check the graph's contract/config digests independently.
2. Account for every `PARAMETER`, `BUFFER` and `CONSTANT_TENSOR` input spec,
   including unused captures. Validate inventory targets and payload-config
   names against graph signature and the two configs. Do not trust inventory
   completeness merely because its own hash matches.
3. Require exactly one tensor entry per capture target. Reject duplicate
   decoded keys, ambiguous payload bindings, missing and surplus entries.
4. Match final dtype and logical shape to graph metadata/configs. Check dense
   row-major storage, offset zero, and checked byte counts; keep scalar `[]`
   distinct from empty `[0]` and from singleton `[1]`.
5. Resolve each origin's file/key to a declared unique source. Validate source
   pins, conversion consistency and generated/inline byte lengths. A generated
   `empty` requires zero elements; `fill` requires exactly one dtype-sized
   element; inline decoded bytes may not exceed 64 KiB.

Final values are little-endian, contiguous bytes:

| Origin | Preparation |
|---|---|
| Checkpoint, `none` | Read the named safetensors key; verify stored dtype/shape; borrow its view |
| Checkpoint, `cast` | Check `from`, `to` and final dtype; convert once; own the result |
| Generated, `empty` | Produce zero bytes with the declared empty shape |
| Generated, `fill` | Repeat the exact element bits using checked counts |
| Inline | Decode bounded base64; preserve the exact element bits |
| Pack | Read the named key from the declared graph-owned file |

Verify whole-source size/SHA-256 before decoding its header or publishing
views, then verify every final tensor's dtype, shape and raw-value SHA-256.
The archive becomes executable only after all captures pass. Retain prepared
storage so repeated calls never fetch, recast, regenerate or rehash captures.
Shared checkpoint bytes may be borrowed; tied aliases are informational and
never create mutable shared inference state.

The initial conversion subset is `none`, BF16→F32 and F16→F32, covering the
first FP32 checkpoint cohort. Implement widening by exact bit conversion,
including signed zero, subnormals, infinities and NaNs; hashes decide the exact
required NaN encoding. Return a typed unsupported-conversion result for other
casts. Do not advertise all schema-permitted casts as implemented.

## 6. Sources, storage lifetime and bounds

Support caller-supplied local files keyed by declared source name, revision-
pinned Hub sources and release HTTPS sources. Local mode verifies the same
pins and never fetches. Converted checkpoints use their release URL and retain
the original-bin provenance; consumers do not reconvert `pytorch_model.bin`.
Name collisions or two incompatible pins for one source are errors.

Files fetched over HTTP are hashed from actual bytes, including cache hits.
Redirects do not change the expected pin. Publication and Hub URLs use the
same verification contract. Avoid using unchecked names as cache paths.
Keep mapped owners reachable for the lifetime of the prepared archive; owned
cast/generated bytes have the same lifetime. Check aggregate resident bytes
including conversion results, not just each source's size.

All size products, offsets and sums use bounded `int64` or existing typed
storage domains until proved representable on native and js_of_ocaml. Check
metadata bytes, source count, capture count, shape rank, inline bytes, source
bytes and prepared bytes before the corresponding allocation/work. Host
budgets may narrow platform ceilings, never raise them. Reuse existing PT2
limits where applicable; document new limits and their resource basis in S0/S1.

## 7. Named calls and reference replay

Add named-input execution alongside the current one-input API. The latter
delegates to the new validated binder and retains its existing behavior.
Use `GraphSignature.input_specs` to bind user inputs to SSA IDs; captures
remain loader-bound. Require exact input-name coverage, compatible dtype/rank
and static logical shape before execution. Reject missing/extra/duplicate
names and mutation-bearing signatures. Mixed masks, indices and floating
inputs keep their original dtypes.

Read `cases.json` and flat tensor files through existing
`Pt2_archive.load_pt_tensor_map`. This reader returns lexical name order;
reorder explicitly by the contract/case descriptors before binding or hashing.
Do not infer positional call order from a map traversal. Validate the graph's
input/output order against the contract and return outputs with their names.

Recompute the producer's tensor-content digests, including their ordered
name/dtype/shape preamble, using pinned cross-language test vectors. File
hashes alone do not check that logical tensors match a case descriptor.
Comparison uses logical coordinates and dtype-aware reads, so Native's
channels-last frame is never mistaken for the reference's row-major layout.
Integer and Boolean values compare exactly; floating outputs use the case's
declared `atol + rtol * abs(reference)` without relaxing tolerances. Explicitly
validate nonfinite values and document the pinned producer's comparison rule.

A replay result names artifact, graph, checkpoint sources, consumer revision,
case and backend; reports shape/dtype errors, mismatch count and representative
coordinates plus maximum absolute/relative error. Every output element is
checked. Rankings, greedy tokens and cosine similarity cannot substitute for
tensor parity. A per-artifact refusal is a structured result and never a
successful replay.

## 8. Empty captures and Native's positive extents

`Core.Dim.extent` and Native shapes require extents ≥1. Generalizing that
invariant would affect reductions, index domains, allocators and all backends;
this integration will not change it globally.

Use a constrained, explicitly reported normalization at the PT2 import
boundary for the evidenced `[0]` cache values. Keep the producer graph and
captures intact and digest-verified. Carry a typed empty value in the import
environment, outside Native tensors. Admit only proven empty-preserving
clone/identity chains and the ATen concat rule for one-dimensional `[0]`
operands. A concat must retain at least one nonempty operand and produce the
same logical shape, dtype, order and values. Nonempty concat behavior remains
the existing Native operation; a single survivor uses a proven identity.

Record source SSA IDs, captures, affected source nodes, retained operands and
the checked rule in provenance. If an empty escapes as a user output, has an
unrecognized reader, participates in mutation, or requires an unimplemented
empty result, return an unsupported-empty-use result. It is not malformed
producer data. Never perform an unrecorded deletion of a live capture.

Prove this rule with micrographs against ATen and full unchanged reference
cases, including reset histories. This is a deliberate boundary exception to
the one-source-op representation rule in [the Native operation design](native_add_op.md),
with its own source-normalization record. It is not a generic rewrite permission.
Other empty shapes and dynamic history remain explicit refusals.

## 9. Operators and backend admission

Add `embedding.default` as a genuine operation if existing gather semantics
cannot preserve the source node, rank and failure behavior. The initial scope
is inference with integer indices and the evidenced table/index ranks.
Preserve `padding_idx` inference behavior; do not zero the stored padding row
or clamp invalid indices. Training options do not imply training support.
Verify concrete, symbolic and ATen-bridge behavior and report independent
Native4D/Kernel admission. Do not decompose one embedding into several Native
nodes simply to reuse implementations.

Handle the observed `arange.start_step`, `abs.default` and `detach.default`
through their existing operator families where semantics match. Preserve
arange integer arithmetic, sign/step and dtype; abs signed-zero/integer rules;
detach's functional inference identity with provenance. Record restrictions.
Rerun the complete admission sweep after each change: solving a first blocker
may expose another operation, symbolic dimension or dialect limitation.

## 10. Static generation and full tasks

Start with the producer's static tensor cases: encoder/vision/connector,
prefill and fixed-history decode are independent call contracts. Validate all
self/cross-attention K/V outputs and resets, not only logits or tokens.
Published first-history/capacity snapshots are not a continuous generation
loop. Refuse a history for which no accepted graph exists; dynamic bounds or
example hints do not authorize specializing arbitrary dimensions.

Before promoting a bounded generation demo, supply an accepted component for
every executed history or implement a separately designed dynamic-shape path.
The host owns model/component compatibility, cache layout, masks, reset,
capacity checks, greedy sampling and stopping. Never cast index/mask inputs
to the floating model dtype. Include independent prompts and reset sequences.

Task adapters pin matching processor/tokenizer assets and the correct
population/component recipes. Existing metadata has shape gaps in seven
families; choose and verify explicit padding/resize policies against the
selected contract. TinyCLIP tower outputs need their documented normalization
and scoring. A passing component case does not establish image classification,
text generation, transcription or image-text task correctness.

## 11. Validation and compatibility policy

Maintain separate states for source integrity, capture preparation, graph
decode, Native import, Native4D conversion, Kernel conversion, numerical
replay and task execution. Numerical rows are also backend-specific. Native4D
and Kernel branch independently from canonical Native; neither is a required
predecessor for direct Native replay.

Hermetic tests exercise every supported origin, dtype widening, scalar/empty
distinction, multiple files, digest corruption, mismatched graphs/captures,
named mixed-dtype calls, overflow, and the normalization refusal boundary.
Use independent known byte vectors and ATen or producer oracles. Prove critical
corruption and semantic tests fail when their check is bypassed or computation
is changed. Large downloads are opt-in and digest-pinned; replay is offline.

Keep v1/PT2/image-model goldens passing. For JS-reachable changes run both
backend comparison and committed inline goldens. New stages are complete only
with measured evidence and the promised numerical checks, not merely a new
count of admitted graphs. No existing oracle, tolerance or unsupported status
is edited to make a result appear successful.

[map-spec]: https://github.com/TheCBaH/devcontainer.transformers/blob/81feca91b3d3ad032cb3c1ef28f4d5e1751c1d55/docs/checkpoint-map-v2.md
[map-schema]: https://github.com/TheCBaH/devcontainer.transformers/blob/81feca91b3d3ad032cb3c1ef28f4d5e1751c1d55/schemas/checkpoint-map-v2.schema.json

## 12. Consumer cohort, inventory and budgets

`data/transformers/cohort.json` is the consumer manifest, generated by
`scripts/transformers-cohort.py` from the release's `publication.json` asset.
It pins that index's exact bytes (71,174 bytes including the trailing newline;
an earlier saved copy without it hashes differently and is not the release
asset), then for each selected artifact: archive and manifest, graph and
contract digests, the map member's digest, the map-declared checkpoint files
and the released source assets. The initial cohort is MobileViT forward,
SmolVLM connector, BERT forward and the TinyCLIP text tower. Sources may be
release assets (converted BERT/MobileViT) or revision-pinned Hub files
(SmolVLM, TinyCLIP); the manifest keeps both.

`make transformers.admission` rebuilds the graph-only baseline from a producer
checkout at the pinned commit; its output is byte-identical to the retained
30-row sweep. Admission, capture preparation and numerical replay stay
separate commands and separate recorded states.

Inventory of all 35 released maps (every archive hash verified): 3,918
BF16→F32 casts, 3,659 uncast checkpoint tensors, 640 generated `empty`, 38
`fill`, 21 inline (≤4,096 base64 characters) and one `pack`. `F16`→`F32` is
not observed in this release; it stays in the supported subset because the
widening is exact and independently testable. Observed maxima: map 193 KB,
534 captures, rank 5, 28.4M elements per tensor, one checkpoint file per map
(largest 513 MB). These are observations, not ceilings. Native shapes are
rank-limited by `Vec6`, so rank above six is an unsupported-shape result.

Replay compares each output with `|actual - reference| <= atol + rtol *
|reference|` using the case's own tolerances (`1e-5`, `1e-4` in the cohort);
integer and Boolean outputs compare exactly.

## 13. Pure map library (implemented)

`lib/pt2_sha256` is the byte-hash provider: SHA-256 in pure OCaml over `Int32`
words with an `int64` length, incremental over strings and bigstrings, so one
implementation gives identical digests on native and js_of_ocaml (checked by
the published vectors, every chunk size and both backends). It replaces no
host facility: `hf-hub-unix`'s `sha256sum` shells out and stays the whole-file
path for large native sources. The OCaml custom operators `&%`/`^%` sit at
different precedence levels; the compression function parenthesizes every
mix of them (the first draft did not and produced wrong digests).

`lib/pt2_checkpoint_map` is the pure v2 core. `Document.of_string` decodes the
map with schema-closed Jsont objects (unknown members rejected where the
schema forbids them, tensor members kept in order so a repeated key is an
error rather than a silent overwrite) and checks the map against itself.
`Validate.check` reads the graph three independent ways -- signature, payload
configs, `captures.json` -- and requires the map to match all of them in a
fixed order, so the first reported fault is deterministic. Faults are
polymorphic variants with data payloads (`Fault`), not strings. Only the
conversions the host implements (none, BF16 to F32, F16 to F32) pass
`supported_conversions`; the others parse but are refused before any source
is opened.

`bin/transformers_map_check.exe` runs this against extracted bundles with no
source file read. All 35 released bundles of `checkpoint-003207ae59ed`
validate; 19 single-fault mutations of one bundle are each refused at the
intended check.

## 14. Source verification and capture preparation (implemented)

`Prepare.verify_sources` takes the bytes of every declared file as bigstrings
the host owns (`lib/pt2_checkpoint_map_unix` maps local files; later layers add
cache and download) and checks them in order: exact name set, size, whole-file
SHA-256, then the safetensors header. `Prepare.capture_set` then produces each
capture from its origin and proves it: an uncast checkpoint or pack tensor is a
zero-copy view after dtype and shape checks against the stored header; a
widening, `fill` or inline value is an owned copy; every result's byte length
and SHA-256 must equal the map's pin. The set exists only if every capture,
unused ones included, passed, and it returns the same storage on every lookup.
Allocations are bounded per buffer (a js_of_ocaml-safe ceiling) and in
aggregate with the mapped sources, before the allocation happens.

`Widen` converts BF16 and F16 to binary32 by integer bit manipulation only, so
signed zero, subnormals, infinities and NaN payloads cannot be altered by a
float round trip or differ on JavaScript. BF16 is the 16 bits followed by zero
bits. F16 renormalizes subnormals and makes a NaN quiet while keeping its sign
and payload, matching the hardware conversions torch uses; no released map
contains an F16 source, so that rule rests on tests, not a release digest.
All 65,536 patterns of both formats are checked against tables computed in
Python.

Measured against the released data (checkpoint-003207ae59ed): BERT 42 and
MobileViT 347 captures (uncast), the SmolVLM connector (a 28 MB BF16 to F32
cast from the 513 MB Hub checkpoint) and the 55 TinyCLIP text-tower captures
all reproduce the map's digests in OCaml; the hashing is pure OCaml at about
127 MB/s.

## 15. Acquisition chain (implemented)

`lib/pt2_fixture` (pure) holds the pin layers; `lib/pt2_fixture_unix` is the
native host. The chain, each link checked against the one above, is: the
cohort manifest (consumer, trusted) pins the publication index's bytes and each
artifact's archive, manifest, graph, contract, map and source files; the
publication must agree with those pins; the manifest must agree with the
cohort and lists every archive member by size and digest; the archive's members
must be exactly the manifest's and each must match its pin; the map must name
exactly the cohort's source files with identical pins; and every source and
capture is then verified as in the preparation stage. A failure names its layer
(`Fault.layer`), so a wrong index is never reported as a wrong archive.

Files live in a content-addressed cache: a blob's path is its SHA-256, never a
producer-supplied name. A download goes to a temporary file, is checked for
size and digest, and only then renamed into place; a cached file that fails its
pin is replaced when a transport exists and is an error offline. The transport
is a function writing one URL to one file (`curl`, HTTPS only, five redirects
at most); it is trusted with nothing, so a redirect or proxy changes the
digest, never the expected pin.

The `.tar.gz` reader is a deliberately small host helper, not a tar library: it
decompresses with `Zipc_deflate` under a size ceiling, checks the CRC and
length trailer, and accepts regular files only. Links, absolute or escaping
names, duplicate names and extended headers are refused. An archive is
extracted only after its member set and every member match the manifest in
memory, into a temporary directory that is verified once more and renamed to
`bundles/<archive digest>`; a bundle directory exists whole and verified or not
at all, and is re-verified (exact member set, no links, sizes, digests) every
time it is reopened. Missing `model.pt2` is the expected shape of a slim bundle.

`make transformers.download` is the only target that uses the network;
`make transformers.open` opens the cohort from the cache alone and proves every
capture. Measured on the four cohort artifacts of `checkpoint-003207ae59ed`
(about 650 MB): cold fetch 29 s, offline reopen 12 s (mostly SmolVLM's 513 MB
hashed twice, once for the cache check and once in memory).

## 16. Named calls and replay (implemented)

`Native_interp.run_named` binds user inputs by the exporter's names and checks
before evaluating anything: the signature has no mutation (every output spec is
a user output), the supplied names are exactly the graph's user inputs (none
missing, unexpected or repeated), and each tensor has the declared dtype and
static shape. A failure is an `` `Input_binding `` row; mixed float, int64 and
bool inputs keep their own dtypes. `run ~input` is the old one-input API and
keeps its behavior, including its `Not_exactly_one_user_input` refusal and the
absence of name and dtype checks; both share one run body. The transformed
route (`evaluate`) has no named variant yet; the replay harness executes the
direct route only and the report names the route it ran.

`lib/pt2_fixture` adds the pure replay vocabulary: `Contract` (plain functional
tensor call only: no positional arguments, no mutation, keyword order equal to
the input list), `Cases` (held to the contract: id sequence, tensor names and
order, tolerances), `Logical` (any strided reference tensor gathered into
row-major bytes after its reachable storage range is bounded in int64),
`Tensor_digest` (the producer's content digest, line for line), `Compare` and
`Report`. A reference output and an actual output meet only as `Logical`
tensors, so Native's channels-last storage and a `.pt` file's strides cannot be
mistaken for values.

Comparison follows `torch.testing.assert_close` with `equal_nan = false`:
integers and booleans exactly; floats equal (infinities and signed zeros
included) or within `|a - r| <= atol + rtol * |r|`, evaluated in binary32 with
`atol` and `rtol` first rounded to binary32, as torch does for float32; a NaN
never matches. Every element is checked and the first eight mismatches are kept
with their logical coordinates. Tolerances come from the contract and are never
relaxed.

`lib/pt2_fixture_replay` reads each case's `.pt` maps from the verified bundle,
reorders their lexical keys into contract order, recomputes both content
digests, checks the inputs against the contract, runs the case and compares each
output. A digest that disagrees with its descriptor, an input that disagrees
with the contract, and an engine refusal are recorded in the versioned JSON
report (`status` is `passed`, `failed` or `refused`) and are never a pass. A
dynamic-shape contract is refused outright.

First numbers (the initial cohort, Native direct, consumer at the S4 commit):
MobileViT-xxs original size, both cases, all 1000 logits within tolerance
(max absolute error 3.1e-4, relative 5.5e-5); the SmolVLM connector case-01
passes and case-00 fails the same two elements as the earlier v1 probe
(absolute errors 1.57e-5 and 1.56e-5 against allowances of 1.42e-5 and
1.33e-5); BERT-tiny and the TinyCLIP text tower are refused at
`aten.embedding.default`.

## 17. The connector's case-00 mismatch (diagnosed)

The SmolVLM connector is nine data-movement nodes and one bias-free linear:
36,864 dot products of length 12,288 per case. Replayed from the verified v2
bundle, case-00 fails exactly two elements (flat indices 4,235 and 31,987;
absolute errors 1.57e-5 and 1.56e-5 against allowances of 1.42e-5 and 1.33e-5)
and case-01 passes -- the same elements the earlier v1 probe failed, so the
v2 loader changed nothing about the answer.

Two independent oracles, run on the same captured input and the same BF16
weights widened to binary32:

1. Exact arithmetic (Python `fsum` over the exact binary64 products, rounded once
   to binary32). Native's output equals it bit for bit in all 36,864 elements of
   both cases. The producer's reference does not: it differs from the exact value
   by up to 2.7e-4 (mean 5.6e-6), is never closer to it than Native, and at the
   two failing elements is 1.57e-5 and 1.56e-5 away from the exact value --
   outside the producer's own tolerance of the exact answer -- while Native is
   1.7e-9 and 4.7e-10 away.
2. A sequential binary32 fused-multiply-add chain (`acc = fma(x[j], w[j], acc)`,
   `j` ascending, rounded to binary32 at each step) reproduces the reference
   bit for bit in every one of 32 sampled elements, including both failing ones;
   non-fused, reversed, interleaved and blocked variants reproduce at most 3.

So the reference is the accumulated rounding error of a sequential binary32
reduction, large enough at length 12,288 to leave its own tolerance. Native
direct compute is binary64 by design and rounds once on store (see the binary32
kernels design), so it is closer to the truth than the reference and cannot
match the reference's noise without emulating that reduction. A test pins this
contract for a linear of the same length.

This is a numerics-policy decision, not a defect: closing the case needs either a
reference computed with a higher-precision accumulator or an explicit opt-in
Native policy that emulates the binary32 sequential chain (and then a separate
backend row), neither of which is attempted here. The tolerance and the
reference are unchanged.

## 18. Small operator gaps (implemented)

`aten.detach.default` is the functional identity under inference and binds to
the existing `Clone` node, like `alias.default`; the node keeps its source op in
provenance. `aten.arange.start_step` extends the existing arange arm (importer
and ATen bridge): the overload whose `step` the schema requires, so an absent
step is an error rather than the default of one. Its dtype rule, exact int64
bounds, and the positive-step restriction are the other overloads' unchanged
(a descending range stays refused).

`aten.abs.default` is a genuine one-node `Abs`, dtype-preserving. The float form
is the three IEEE cases written out (`x < 0` negates as `0 - x`; `x == 0` yields
`+0`, so `|-0.| = +0.`; anything else, NaN included, passes through) and uses
only existing semantics primitives, so it needs no new expression constructor
and no `x + 0` that a simplifier could drop. The int64 form is exact: it reads
through `i64_load`, compares, and negates as `0 - x` in wrapping int64
arithmetic, so `abs min_int = min_int` as in ATen, with no float round trip
above 2^53. The builder threads the I64 output edge; Direct, Symbolic (an int64
stage) and Native4D carry the same dispatch. The ATen oracle gained bindings for
`abs`, `detach`, `embedding` and `arange.start_step`; `embedding` required
`native/Embedding.cpp` in the minimal archive.

Rerunning the 30-row admission sweep moved three first blockers and exposed
three others, as expected: `aten._to_copy.default` with a `layout` argument,
`aten.full.default` and `aten.squeeze.default` are now each a graph's first
refusal. They are recorded, not yet implemented.

## 19. Embedding lookup (implemented)

`aten.embedding.default` is a `Gather` over the table's leading axis: a `[V, D]`
float32 weight and an int64 index tensor of rank 1 or 2 give an output of the
indices' shape plus `D`. The indices' ATen rank is carried in the op's params
because the six-axis frame right-aligns and erases it. `padding_idx` only steers
ATen's backward pass, so it is kept for provenance and changes no value;
`scale_grad_by_freq` and `sparse` are read and dropped.

Index semantics are strict, as ATen's: an id outside `[0, V)`, negative ids
included, is an error, not a wrap. Direct enforces it in a pre-pass and reports a
typed `Embedding_index_out_of_range` row; the symbolic route wraps instead, a
difference documented where it is defined. Native4D rejects the op, so
admission rows for graphs that contain one stop at Native4D by design, while
Native and Kernel build. The ATen oracle cannot test the invalid-index side: the
kernel's bounds check aborts the process, so only valid indices are compared.

The importer arm lives in its own module to keep the compute-family dispatcher
under the file-size cap.

Admission after this change: embedding is no longer the first blocker of any
graph, and the replay of the released BERT-tiny and TinyCLIP text encoders now
stops further on (`ge.Scalar`, `le.Tensor`). So embedding is verified against
the ATen oracle and on micrographs, not yet on a released case.

The rest of the text/tower closure is a family, not one operator: measured across
the 30 cohort graphs, about 30 targets have no importer arm. Most are comparison
and logical forms (`ge`/`le`/`lt`/`gt`/`eq`/`ne` scalar and tensor, `__and__`,
`where` in three overloads), the factory forms (`new_ones`, `full`, `zeros_like`,
`full_like`), and a few pointwise and structural ones (`tanh`, `exp`, `log`,
`log1p`, `reciprocal`, `squeeze.default`, `diff`, `index_put`, ...). Comparisons
exist as Native ops reachable from the ATen bridge but not from the serialized
importer; the others need Native ops too.

Native keeps one node per ATen op, so none of these may be written as a
decomposition in the importer (`new_ones` as `zeros + 1` was tried and removed for
this reason): each needs its own op, or an importer arm onto an existing
one-node op. That is a separate program of work from the embedding gather and is
sequenced after the empty-cache normalization that blocks 16 graphs outright.

## 20. Empty-cache normalization (implemented)

The static decode graphs lift the initial key/value cache to a `Tensor_constant`
of shape `[0]`, clone it, and concatenate it with the step's new keys. ATen's
`cat` skips a 1-D size-0 operand whatever the other operands' rank (checked
against the oracle at the first, last and a negative dim), so the concatenation
is the new keys alone. `Native_interp.normalize_empty_caches` rewrites exactly
that and nothing else, on the PT2 program before lowering, so Native keeps its
positive extents and no empty value is ever a graph edge:

- an empty source is a `Tensor_constant` whose metadata is exactly `[0]`;
- `clone.default` of an empty is empty and is dropped;
- `cat.default` drops its empty operands, keeping the rest in order; one
  remaining operand becomes `clone.default` (ATen's `cat` of one tensor is a
  copy);
- refused with a typed `` `Empty_cache `` row: any other reader, an empty
  returned by the graph, a signature that mutates state, a cat whose operands
  are all empty, and a dtype different from the kept operands (ATen promotes
  over a skipped operand too, so that is not a pure drop). Zero-length tensors
  of any other shape stay refused by strict lowering.

The result is a new program value; the published graph, its digest and the
capture inventory are untouched, and the report names, per source, the clones
dropped and each cat site with the dropped operand positions (a typed ordinal).
It is opt-in (`run_named ?empty_caches`, which hands the report to a callback);
plain `lower` is unchanged, so no path that did not ask is affected. The replay
report (schema 2) records each rewrite in `normalizations`.

Measured on the 30 cohort graphs: all 16 graphs with empty captures (16, 24 or
60 sources each) normalize and move on. Their next blockers are recorded, not
solved here: `new_ones.default`, graphs whose attention masks have symbolic
dimensions, `eq.Scalar` and `full.default`. No producer case for those graphs
runs yet, so S7 is confirmed numerically only on micrographs and the oracle.

## 21. The attention-mask vocabulary (implemented)

BERT-tiny's published cases pass on the native path: both cases, all outputs
(`last_hidden_state`, `pooler_output`), max absolute error under 5e-6. It is the
first released case that runs through the embedding gather. Getting there took
the operators every text graph builds its attention mask from:

- Comparisons `ge.Scalar`, `lt.Scalar` and `le.Tensor` are new one-node ops, and
  the importer now also reaches `eq`/`ne`/`gt` (scalar and tensor forms) that
  Native already had. Each follows `Gt_scalar`'s split: the generic formula
  yields 0./1., and Direct lands genuine bool storage. NaN is neither above,
  below nor equal.
- `__and__.Tensor` is `Bitwise_and`, restricted to bool values: an int64 operand
  is refused, since the float-domain nonzero test is the bitwise `and` only for
  bool. `where.ScalarOther` is `Where_scalar_other`: the tensor where the
  condition is nonzero, the scalar elsewhere, broadcasting a rank-0 value over
  the mask. `tanh.default` is `Tanh`, with `tanh(-0) = -0` kept.
- `new_ones.default` is its own factory node, bool or float32, readable as a
  rank-0 `True`. It was first written as `zeros + 1` and removed: Native keeps one
  node per ATen op.
- `index.Tensor` with two live leading indices is `Index_pair`: the
  `mask[batch_idx, kv_idx]` and `h[arange, argmax]` forms. The result is the
  broadcast index shape followed by `self`'s remaining axes. The frame erases
  ranks, so the payload carries the three ATen ranks; with `t` trailing self
  axes an index axis sits `t` axes right of the result axis for the same logical
  dimension, which fixes both coordinate reads. The result keeps `self`'s element
  format (a gathered mask stays a mask). The single-live-index forms are
  untouched: the new family claims a node only when `indices` is a pair of live
  tensors.

Typing changes that make the graph's dtypes real rather than assumed:

- A user input is declared with its metadata dtype. It was always float32, which
  hid that an `input_ids` edge is int64 and made every integer-index consumer see
  a float edge.
- `slice` of an int64 tensor stays int64 and exact (a position-id buffer's
  `[:, :n]`), as `reshape` and `permute` already did. `add.Tensor` of an int64
  tensor and a whole scalar stays int64 through the exact add; a float scalar
  that is whole takes the same path and differs only in the dtype tag, which a
  float consumer refuses as mixed rather than misreading.
- Most data-movement ops do not keep a bool edge tag, so a bool commonly arrives
  as a float32 0./1. `Bitwise_and` and the where condition therefore accept
  float32 or bool and refuse int64.
- An explicit `dtype=float32` on `softmax.int` (and the other reduction-family
  ops that took the shared check) is accepted: float32 is the only float the
  engine holds, so it is the identity or a conversion it already performs. Any
  other dtype is still refused.

Native4D rejects all of the new ops; admission rows for graphs using them show a
Native4D stop by design while Native and Kernel build.

Still open on the text graphs: TinyCLIP's text tower needs `_to_copy` to int32,
`argmax`, `exp` and `t`; T5 needs `log`, `zeros_like`, `full_like`, `lt`/`min`;
the vision-language prefill graphs need `masked_scatter`, `index_put` and a bool
index; `squeeze.default`, `diff`, `full`, `conv_transpose2d`, weight norm and a
group-norm weight shape remain single-graph blockers. The graphs whose attention
masks have symbolic dimensions are dynamic-shape variants outside the static
scope. Several now lower (the SmolLM2 and Whisper decode/prefill graphs, the
SmolVLM text graphs) but have not been run against their cases.

## 22. Expanded cohort and the accumulation policy (measured)

The cohort grew from four artifacts to twelve: SegFormer-B0, YOLOS-tiny and
TinyCLIP's image tower (vision), the Whisper-tiny encoder and prefill, SmolLM2
prefill and decode at history 4, and SmolVLM decode at history 71. The last
shares the connector's checkpoint, the TinyCLIP image tower the text tower's, so
the Hub files are fetched once. The first four entries are byte-identical to
before.

**Dot accumulation is now a Direct policy.** `dot` is its own primitive beside
`sum` (symbolically the same expression) and the matrix-product ops use it.
`Direct.Binary64` (default) is the exact sum, rounded once on store;
`Direct.Binary32_sequential` rounds a sequential fused chain to binary32 at every
step. A replay under the second policy reports a different backend
(`native-direct+binary32-sequential-dots`) and writes a different report file:
it is a separate row, never a quiet variant. The policy is scoped to the call and
restored on a raise.

Measured, each row preserved as run (tolerances and references unchanged):

| artifact | exact binary64 | binary32 sequential |
|---|---|---|
| MobileViT-xxs | pass | not run |
| BERT-tiny | pass | not run |
| SegFormer-B0 | pass | not run |
| YOLOS-tiny | pass | not run |
| TinyCLIP image tower | pass | not run |
| SmolVLM connector | 2 of 36,864 elements over | **pass** |
| SmolVLM decode, history 71 | 1 of 49,280 logits over (1.4e-5 vs allowance 1.1e-5); all 60 K/V pass | 29 over |
| SmolLM2 decode, history 4 | case-01: 5 of 49,152 logits over (4.3e-5) | 20 over |
| SmolLM2 prefill | 174 and 1,880 of 196,608 logits over (max 7.4e-5) | 1,974 and 2,685 over, plus K/V |
| TinyCLIP text tower | pass (after `argmax` and the int32 cast, below) | not run |
| TinyCLIP forward (both towers + logits) | pass (adds `exp`, `t`) | not run |
| Whisper-tiny prefill | pass (after exact int64 `repeat`) | not run |
| T5-small encoder, forward, prefill, decode h4 | refused (cast of an infinite value) | not run; **pass** under `+saturating-casts` (see below) |
| Whisper-tiny encoder | 16 and 18 of 576,000 elements over | not run |

The reading: the sequential chain reproduces the reference only for the
connector's gemm. The decode and prefill references run a different kernel
(a row-vector product and a blocked product), whose order the chain does not
model, so under that policy they get worse. The exact sum stays within a few
1e-5 of every reference but crosses its allowance on a handful of elements.
This is the numerics-policy question T04 raised, now with a second and third
witness; it is not closed, and no policy that matches every reference exists
in the engine. What would close it is a reference computed with a
higher-precision accumulator, or per-kernel emulation of the producer's gemm
for each shape class.

A note on the Whisper encoder: a replay of it was once left running for over 30
minutes before being stopped, yet an earlier run of the same artifact had
finished and written the report above, so its cost is high (a 3,000-frame
convolution stem and 1,500-position attention in a pure evaluator) rather than
unbounded. Whisper prefill was not run; the matrix script
(`scripts/transformers-matrix.py`) states "not run" for it rather than inferring.

## 23. Last-token pooling: argmax and the int32 cast

TinyCLIP's text tower pools the hidden state at the end-of-text token:
`(input_ids.int() == eos).int().argmax(-1)` selects the position, and a
two-index `index.Tensor` reads it. Two small additions close it, and the tower
passes both published cases.

`argmax.default` over one axis is its own node (the schema's single int64
output, no values), sharing the index predicate of `max.dim`. The flattened
`dim=None` form is refused. A difference from ATen is known and tested: with
several NaNs in a row the predicate reports the last, ATen's reduction the first.

`_to_copy` to int32 is a `To_copy` target of its own. The engine has no 32-bit
integer edge, so the value rides in an int64 cell; Direct raises on a value
outside the int32 range instead of wrapping as ATen's cast does, since the cell
could not reproduce a wrap. The Symbolic route does not range-check, and
Native4D refuses the target.

## 24. Generation history: what a snapshot establishes

A decode artifact is exported at one history length (`variant.kind =
static-history`), and its published case is a single step at that length.
`Pt2_fixture.History` states the consequence and enforces it. A report for such an
artifact carries a `scope` sentence (schema 3): a snapshot at history N with the
artifact's capacity, other histories not covered. A feed at any other history is
refused, and one beyond the capacity is reported as exceeding it, never padded or
truncated. A prefill can feed a decode artifact only when its `present_*`
outputs correspond one to one, in order, with the decode's `past_*` inputs, agree
on batch, heads and head dimension, and have length equal to the decode's
history. The released SmolLM2 (4) and SmolVLM (71) prefill/decode pairs meet; no
chained run has been made, because it would have no producer reference to check
against and a measured result there would not be a gate. Dynamic-shape contracts
remain refused outright.

## 25. T5's relative-position buckets and the saturating cast

The T5 graphs compute a relative-position bucket in integers: absolute
distance, a logarithmic bucket for the far ones, a minimum against the last
bucket, and a `where` choosing between the near and far values. For distance 0
the far branch is `log 0 = -inf`, cast to int64, and then discarded by the
`where`. That cast is undefined in C++; aarch64, where the producer ran, saturates
it (NaN to 0, out of range to the nearest limit). The engine's default cast
rejects NaN, infinities and out-of-range values, so under it all four static T5
artifacts are refused with the cast's own message.

The cast is therefore a policy a caller opts into, in the manner of the dot
accumulation: `Direct.float_to_int`, `Checked` (default) or `Saturating`, scoped
to the call, named in the report's backend (`native-direct+saturating-casts`) and
in the report's file name. Under `Saturating` the T5 encoder, forward, prefill and
decode-at-history-4 artifacts pass both published cases. The graphs only ever use
the saturated value to be thrown away, which is why the answer does not depend
on the platform; a graph that used it would.

The integer vocabulary behind it is exact: `min.other` and `where.self` on int64
compare and select without a float, `full_like` (and `zeros_like`, the same node
with a zero fill) keeps the format of its operand, and an int64 tensor times a
whole-number scalar stays int64 (a wrapping product), as `add_scalar` already
did. A scalar spelled as a whole float takes the same path, which differs from
ATen only in the dtype tag. The replay boundary also changed: a failure the
evaluator raises mid-run is now recorded as the artifact's refusal, by kind, rather
than ending the process.

## 26. Exact int64 data movement, and a bounded text example

**Data movement keeps an int64 operand exact.** `clone`, `expand`, `repeat` and
`repeat_interleave` now join `reshape`, `permute`, `slice` and `unbind`: an int64
input gives an int64 edge and is copied through `Compute_i64`, never the float
domain. The failure that exposed it was Whisper prefill: a `repeat` of position
ids gave a float32 edge, and the next integer consumer refused it. With it fixed
Whisper prefill passes both published cases.

**A bounded text example, deliberately not a task-ready claim.** The plan asks for a
raw-input example that produces the expected tensors under pinned assets, and
the release cannot supply the expected tensors: its BERT-tiny cases hold random
vocabulary ids with a fixed mask, its vision cases one seeded normal stream. What
exists is a chain whose links are each checked as far as this environment allows.

- The pinned assets are `vocab.txt` at the checkpoint's revision (its sha256 is
  checked before every use) and `config.json`, whose digest equals the one in the
  artifact's contract.
- `lib/wordpiece` is BERT's basic tokenizer and WordPiece for ASCII text, with
  non-ASCII input and the special-token spellings refused rather than guessed.
  It agrees with an independent Python implementation of the same specification
  on 421 sentences against the real vocabulary. That is two implementations of a
  documented algorithm agreeing; it is not the reference tokenizer, which is not
  available here, and a disagreement with it could still exist.
- The model is the verified BERT-tiny graph (both published cases pass).
- The end-to-end embedding has no producer reference. BERT-tiny's pooler output
  also saturates near +-1, so cosines between pooled vectors are not a similarity
  measure; the example prints them only to show the pipeline runs.

So the example demonstrates the path from text to tensors to outputs, and the
claim stops there.
