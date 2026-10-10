# Immutable Transformers replay evidence

`lib/transformers_reports` is native host tooling shared by the replay and
matrix CLIs. The tensor comparator and schema-3 report payload remain in the
pure fixture library; the host adds `run_id`, `run_manifest_sha256` and an
execution policy envelope. Acquisition errors receive a schema-3 failed report
with an explicit `acquisition_error` and no cases, rather than disappearing
from aggregation. Such a report cannot pass.

## Identity and pins

Identity includes the full consumer HEAD, actual file hashes/modes/link targets
and a binary diff hash for executable source/build rules/selected metadata.
The scope is `bin`, `lib`, `js`, `scripts`, `data/transformers`, `data/dune`,
root build/package rules and `.gitmodules`. Tracked and relevant untracked
files are included; hashing actual bytes detects assume-unchanged edits.
Documentation, tests and evidence directories are outside executable identity.
Each indexed submodule records its full gitlink and actual HEAD/diff, or explicit
uninitialized state. Vendored source bytes and the two upstream PyTorch
schema/operator generator inputs are hashed. Producer inventory and consumed
recipe/config/lock bytes are independently verified against the producer
gitlink by the shared metadata loader. The replay executable has its own hash.

The manifest includes the exact cohort byte digest, publication pin, release
producer/tag/repository, selected artifact pins including URLs/sizes, every
verified archive member digest/size, cases, tolerances, tensor input/output
specifications, checkpoint/config/recipe identity, producer package versions
and exporter/variant metadata. The source checkout lock belongs to the reviewed
source; reference package/lock provenance belongs to the released contract.
These remain distinct. Host architecture, kernel, stable CPU description hash,
OCaml/C compiler versions and relevant thread variables are recorded. Native
direct execution is single-threaded; no producer code is executed here.

## Lifecycle and validation

Every invocation creates a unique run directory. Manifest, each artifact report
and completion are written using exclusive creation and fsync; files are read
only, and completion makes the directory read only. Completion records report
digests and verifies that identity stayed unchanged during execution. Filesystem
permissions are an accidental-edit guard, not a signature/trust boundary. The
matrix verifies the content digests and compares against freshly computed
workspace identity and independently verified cached release contracts.

The matrix validates versions, duplicate JSON keys, run/policy/backend/consumer
identity, full selected metadata and all report pins, exact contract tolerances,
case IDs/order and complete ordered outputs. For numerical verdicts it checks
the expected element count, mismatch/sample bounds and passed/status consistency;
errors and digest flags participate in failure. Refusals and acquisition errors
cannot pass. No absent capture, output or case is inferred from another run.
Incomplete runs stay incomplete. A malformed run is quarantined with a reason
and makes the overall current matrix non-passing. Multiple current reports for
the same artifact/policy conflict, even if they agree. There is no last-writer
or directory-order preference; select separate evidence roots when investigating
repeated runs. Different consumer/source/cohort/executable/environment identities
are shown in the JSON historical section and excluded from current rows.
Legacy standalone reports are explicitly unverified historical evidence.

Default coverage lists all 22 cohort artifacts under four direct policies:
exact/checked, sequential-binary32/checked, exact/saturating and
sequential-binary32/saturating. Matrix completion is a coverage statement, not a
decision that all four policies must be promoted. Required numerical gate
policies and outstanding parity failures are tracked separately. The actual
route is `Native_interp.run_named`, with the existing opt-in empty-cache
normalizer requested and each applied normalization retained in the payload.
No transformed or Kernel execution is implied. The matrix compares retained
evidence; it does not independently rerun tensors or authenticate report authors.

Hermetic tests replay a real synthetic release through all four policies and
mutate stale report/manifest pins, tolerances, policies, cases, output sets,
element counts and digest flags. They cover duplicates, mixed consumers,
legacy reports, incomplete/corrupted runs, exclusive-write refusal and acquisition
errors. A temporary Git repository proves that identical changed-file counts
and assume-unchanged flags cannot hide differing source bytes.

## C++ ATen diagnostic route

`transformers_replay --backend aten-cpp` runs `Interp.run_named` through the
existing C++ bridge over the same pinned graph, verified checkpoint captures,
case inputs and producer references. The shared replay checks complete case and
output inventories, logical storage digests, actual output dtype/shape and the
unchanged contract tolerances. ATen outputs are copied in logical order after
materializing both strided and nonzero-offset views; int64 values remain exact.
The diagnostic introduces no Python torch or Transformers runtime.

Its backend is `aten-cpp`, with cast/dot labels `aten`, normalization `none`,
and route `Interp.run_named`. Native policy flags are rejected for this route.
Run creation, exclusive report writes, completion digests, executable/source
identity and before/after context checks are shared with Native replay. Store
ATen runs in their own evidence root: the Native policy parser and matrix
deliberately reject this diagnostic policy. A passing local ATen result cannot
promote a Native row, and a local ATen mismatch cannot alone establish a Native
implementation error. The local archive's scalar kernels and reference GEMM
are not the producer wheel's exact CPU implementation.

### Measured checkpoint comparison, 2026-10-10

At consumer `5d89674b937ad77d2af07ce2bc60c1f4f643a03c`, immutable C++ run
`run-4d0564` executes 11 components, 22 cases and all 526 output comparisons:
eight components pass, three fail numerically, and none refuse. Run identity
remains unchanged. Case/reference digests, report/completion digests, complete
output inventories and exact equality of pins/tolerances against the historical
Native run are independently checked. The command runs without Python on PATH.

| Component | C++ ATen mismatching elements | Earlier Native default |
|---|---:|---:|
| SmolVLM connector | 5 | 2 |
| Whisper encoder | 27 | 34 |
| SmolLM prefill | 4074 | 2054 |
| SmolLM history-4 decode | 0 (pass) | 5 |
| SmolVLM history-71 decode | 0 (pass) | 1 |
| T5 encoder, forward, prefill, history-4/history-8 decode | 0 (all five pass) | checked-cast refusal |
| Whisper forward | 0 (pass) | historical diff admission refusal |

The historical Native measurements are from consumer `f839b698`; these are
comparisons against the same producer outputs, not a freshly rerun Native
measurement. SmolLM C++ prefill has 4073 logits mismatches and one K/V mismatch
(`case-01`, `present_11_value`). The other K/V comparisons pass. Maximum
absolute errors in the three failed components are respectively
`2.86102294921875e-4`, `8.296966552734375e-5` and
`9.775161743164062e-5`. A passing verdict means within the frozen absolute/relative
tolerances, not bitwise equality. Forward and encoder cases are independent.

This measured run retains source gitlink
`b514afb61454fb15fd491826b475a6e3819ad641`, checkpoint publication
`checkpoint-003207ae59ed`, and local C++ PyTorch gitlink
`a3aa2c6ee42a053b5f761f4ff1f47974c39714fb`. The producer
records `torch 2.12.0+cpu` on aarch64; the consumer also runs on aarch64 but
its local source version is `2.13.0a0`, with a separate DEFAULT CPU/static
archive and reference GEMM. Local ATen
therefore reproduces three failures and resolves the tested decode/cast paths,
without establishing which individual kernel explains the remaining drift.
Native acceptance, including a current Native Whisper-forward measurement,
stays open.

The subsequent source update to `b2b5388` records producer torch 2.13, but
does not update these published checkpoint references or rerun the numerical
comparison. Its source admission evidence is separate; the retained run still
compares against producer 2.12 tensors.
