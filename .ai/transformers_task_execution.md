# Producer task execution and consumer tooling

The source dependency pins producer `b514afb61454fb15fd491826b475a6e3819ad641`.
Checkpoint publication remains at `003207ae59ed`; task selection independently
pins `task-fixtures-b514afb61454-c564bb4b0d83` and its index bytes. The earlier
diagnostic release remains a separate selection. No consumer runs producer
Python or computes PyTorch references.

## Checked boundaries

The task reader verifies index, manifest, archive and every named tensor member
against trusted digests, original contracts and cohort identities. Recipe kind,
ID and complete case list must match the pinned producer inventory. Adapter
assets are derived from actual pinned tokenizer, processor and configuration
bytes; a digest merely copied into a manifest is insufficient.

Task references are metadata-only bundles and can omit checkpoint sources.
Model/generation execution reopens the artifact through the full independently
pinned consumer cohort and checks the actual execution contract against the
task reference. Nested model reports retain checkpoint, graph-owned, graph,
manifest, archive, contract and map digests. A regression demonstrates that the
metadata-only route fails, while the full route checks bytes, executes complete
reference cases and reopens offline. Reference acquisition remains separate
from checkpoint acquisition. The initial CLIP/generation refusals and incomplete
loading investigation remain retained with their original identities.

`transformers_tasks adapters` compares all constructed model inputs exactly.
`models` executes only after those comparisons pass and checks every named
output with the original component tolerances. Reports retain all case failures
and refusals. Model execution uses Native direct, exact dot accumulation,
checked casts and explicitly requested empty-cache normalization. Actual rewrite
sources, clone names and removed concatenation operands are recorded per case.
Adapter comparisons identify their route as `consumer-adapter`.

Supported adapters are ASCII BERT WordPiece and Pillow TinyCLIP/MobileViT PPM
recipes. CLIP normalization rounds the mean/std and subtraction to binary32;
the producer pixel fixtures exposed the former double-precision subtraction.
In model mode the supported image recipes also require host boundary checks.
MobileViT ranks the actual complete logits, compares all top-five IDs and values,
and checks the producer labels against the pinned config before decoding IDs.
Nonfinite logits and ties within the top five or at its cutoff refuse rather
than guessing `topk`'s unspecified tie ordering.

TinyCLIP independently executes both selected towers using the constructed
inputs. Tower contracts must have the same model/checkpoint/config identity as
forward, and each execution reopens the full pinned cohort. Each feature tensor
is compared under its tower's original tolerances and retains its own execution
pins and normalization provenance. Host normalization and similarity use those
actual features and a scalar loaded from the verified forward checkpoint capture
whose origin is `logit_scale`; no host reference value supplies a computation.
The host route accumulates norm squares and score dots in binary64, rounds the
norm, division, exponential scale and scaled features to binary32, and retains
the producer's original host tolerances. All seven host tensors are compared.
Zero norms, nonfinite values, mismatched widths and nonscalar scale refuse.

Schema-2 task cases list required boundaries and all their reports. A failed,
omitted or duplicated required host/tower boundary prevents a passing task
result even when forward and input checks pass. Adapter-only commands explicitly
defer those executions; bounded generation keeps its separate host checks.
Tokenizer offsets/special-token masks, image resize/crop intermediates, Unicode
BERT, torchvision recipes and raw-text generation remain deferred. No result
promotes the complete published task recipe or general task readiness.

## Bounded generation

Two independent prompts use producer-published IDs. Each prefill's complete
actual K/V outputs feed its one history-4 decode; no history-five graph exists.
Input contracts are checked before indexing shapes or executing. The input boundary
requires positive user-input extents and bounded tensor elements before computing
contiguous strides. Zero extents cannot mask an overflowing trailing product;
the regression fails on the former construction and refuses after the correction.
The host requires identical model, checkpoint/config identity and cache dtype/shape/name
order across the two components. It compares inputs, derived positions, every
logit/K/V output, generated token/sequence, history and cache edges. Caller input
digests must remain unchanged. Fresh immutable state, first-maximum greedy ties,
EOS and two-token stop policy are tested; padded prompts and uncovered histories
are refused. Full-tensor failures remain failures even with token agreement.

## Provenance and CI

Each command writes an exclusive run directory with consumer/workspace,
source inventory, fixture pins, executable digest, every case outcome and a
completion digest. A changing workspace invalidates completion. Interrupted
runs remain incomplete rather than becoming passing evidence.

CI keeps the BERT/MobileViT checkpoint gate and adds source integrity, strict
and normalized admission, history/lifecycle tests and a restricted PATH without
Python after upstream build setup. Explicit guards reject an available Python
executable and dependency-scan failures; negation under shell errexit is not
used as an assertion. Manual workflow inputs enable T5 normalized
prefill with saturating casts, and a full 22-component default replay followed
by separately declared opt-in policy rows. Matrix validation retains numerical
failures and absent policy coverage. Hosted results require actual hosted runs.

## Python ownership and remaining migration

The five consumer PyTorch crosschecks and standalone tokenizer Python checker
are retired; published producer references replace their supported scope.
Missing original diagnostic cases retain the existing producer handoff.
OCaml discovers inline-test Dune stanzas, respecting comments, quoted atoms,
modes and library/runner Wasm or CompCert gates. The timing caller checks
discovery failure before cleaning/building. A shell WASI helper fetches package
metadata over HTTPS, validates cached/downloaded package size and SHA-256, and
extracts without root. Upstream ATen torchgen and its packages remain an accepted
build exception. All 87 Python command occurrences in fifteen project cram files
now use test-only OCaml fixtures/observations with their behavioral assertions
preserved. Forced migrated suites pass with Python unavailable, including the
payload-backed cases. The tooling migration is implemented; full numerical/task
scope and missing diagnostics remain separate. This is not a Python-free clean build.
