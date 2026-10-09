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
