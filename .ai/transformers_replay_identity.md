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
