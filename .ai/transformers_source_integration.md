# Transformers source consumption

The consumer tracks `modules/devcontainer.transformers` as a source submodule.
Initialize only its top level with
`git submodule update --init modules/devcontainer.transformers`; nested producer
dependencies and producer Python are unnecessary for JSON admission.

`make transformers.admission` defaults to that checkout. The shell boundary
derives the expected commit from the consumer's indexed gitlink, rejects missing
or uninitialized checkouts, wrong HEAD and modified producer files, and compares
catalogue bytes to the pinned tree even with assume-unchanged/skip-worktree.
An explicit external source override receives identical checks. Offline
admission never fetches or substitutes another checkout.

`bin/transformers_source.ml` replaces inline Python. It checks schema version,
duplicate JSON members/IDs/paths, flattening collisions, safe relative paths,
regular file ancestry, required members, every declared size/SHA-256 and the
graph pin. Only after the entire inventory passes does it create the temporary
flat view consumed by the existing graph-support tool. The summary checks exact
model coverage and preserves separate Native, Native4D and Kernel counts.
`inventory.json` records the actual catalogue digest and artifact mappings.

Hermetic cram tests cover byte/size corruption, schema changes, escaping paths,
symlinks, duplicate members/IDs, flatten collisions, incomplete report coverage,
missing/uninitialized/dirty/wrong-pin checkouts and spaces in source paths.
The native-corpus CI suite runs the default admission command and uploads the
inventory, per-graph reports and summary, including on failure. This does not
establish numerical parity or hosted CI success.

Local evidence is retained under the integration evidence directory in `ai/`.
The current source has 30 tiny random artifacts and 210 verified member files;
strict admission gives Native 7/30, Native4D 2/30, Kernel 5/30. Its source identity
remains distinct from release producer/publication/checkpoint identities.
Admission passes after OCaml setup with Python absent from PATH. Upstream ATen
Python code generation remains the accepted build exception.

Source pin updates must include an inventory diff and regenerated reports.
The 2026-10-10 update pins `b514afb61454fb15fd491826b475a6e3819ad641`.
Its tiny catalogue and graph bytes are unchanged; all 22 checkpoint selections
and three derived adapter manifests still reproduce offline. Producer loading
also checks committed task recipes, request, trusted task pin, three schemas
and protocol documentation and records their hashes in the source identity.
Source updates do not implicitly change checkpoint or task release selections.
The CI source gate applies to Dependabot gitlink updates as to other changes.
Recipe/asset consumption, normalized admission and robust immutable numerical
run identities remain separate corrective work; the existing report's dirty
file count is not a content identity.
