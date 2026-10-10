# Transformers numerical acceptance and controlled diagnostics

The source catalogue, checkpoint publication and producer task-fixture release
have independent pins. The reviewed source gitlink remains
`81feca91b3d3ad032cb3c1ef28f4d5e1751c1d55`. Checkpoint references remain from
`003207ae59ed0555599da190d70cc3d2f15ad705`. A newer fixture generator can publish
controlled comparisons against that release without replacing its outputs.

`data/transformers/numerical-policy.json` binds the exact cohort bytes and names
a numerical policy and acceptance decision for every selected component.
The required core component gates are BERT and MobileViT, exact dots and checked
casts. `transformers.gate` checks the policy manifest and forces those policies.
All other entries remain deferred until their own unchanged-reference replay
requirements are met. A declared policy is not a numerical pass. Connector
sequential binary32 dots and T5 saturating casts are opt-in choices; their results
cannot replace default-policy failures or refusals. Dot accumulation order and
cast behavior belong in run identity, not architecture-dependent substitutions.

`data/transformers/diagnostic-selection.json` trusts the immutable index SHA,
size and URL and selects full fixture identities including recipe, generator
and request digests. The OCaml task reader verifies index/manifest/archive pins,
the complete bounded regular-file inventory, embedded contract/environment,
original publication assets, graph/map/contract/config identities, exact release
lock/packages, CPU/kernel/threading metadata, and every tensor name/dtype/shape
and logical storage hash. Checkpoint config hashes bind Hub bytes; the exporter's
modified runtime-config hash is a separate field. Task archives contain no
checkpoint weights. Flat torch-save files are parsed as data by existing OCaml
readers; no producer Python or consumer Python torch computation runs.

`transformers.diagnostics.fetch` explicitly acquires pinned data and verifies
it. `transformers.diagnostics.check` repeats integrity and tensor checks offline.
`transformers.diagnostics.compare` additionally compares all eager, re-exported
and published outputs, preserving all K/V tensors and original tolerances. It
checks original-case coverage, original input/output bytes and producer counts
and maximum errors. The producer's field called `bitwise` represents
`torch.equal`, which considers signed zeros equal. The consumer checks that
value-equality claim and reports raw storage equality separately. Producer
maximum absolute errors use binary32 subtraction; the schema-3 comparison also
retains its binary64 absolute-error measurement.

Each invocation writes an exclusive immutable `task-run-*` directory containing
execution context, fixture reports and completion digests. Context covers actual
consumer source/submodule bytes, full revisions, fixture selection, executable
hash and host environment. A changed context fails completion. Acquisition and
validation errors retain explicit failed reports. These directories carry
`consumer_model_execution: false` and do not populate the native replay matrix.
`diagnostic_compared` means a checked comparison, even when a pair differs;
individual pair reports decide pass/failure. A producer diagnostic cannot promote
a failed consumer component or end-to-end task to accepted status.

The published controlled scope is SmolLM2 static prefill, dynamic SmolVLM decode
and dynamic Whisper decode. Dynamic decode diagnostics do not establish static
history-71 SmolVLM consumer parity, and Whisper decode does not investigate the
Whisper encoder's additional error. SmolLM2 history-4 decode and Whisper encoder
still require producer evidence for their exact selected static contracts.
Keep those acceptance items open; retain prior scratch comparisons as historical
observations because their Transformers version differed from the release.
