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

## Localizing a divergence: the SmolVLM connector

The connector is a pixel shuffle (eight layout nodes, bitwise equal between ATen
and Native at zero tolerance) and one bias-free `linear` of 12288-term dots, so
the shared-frontier pair run (`fixture_pair_run`, ATen against Native, any
tolerance) names that single unit. Recomputing every output in compensated
binary64 (`fixture_linear_error`) attributes it: with Hub weights, the producer's
reference output is **bitwise equal** to Native's binary32 sequential
accumulation (36864 of 36864 elements, both cases), whereas Native's default
binary64 sum is the correctly rounded value of every element. The default-policy
failure is therefore the reference's own accumulation error (rms ~1e-5, up to
2.7e-4) on cancelling elements (sum|x*w| 120-200 against a result ~0.02) exceeding
the contract's atol 1e-5 / rtol 1e-4; local ATen lies between the two. This
explains why the sequential-dot opt-in passes. It does not change the default
acceptance row: the opt-in stays an explicit policy. Other failing components
have not been localized this way.

`make model.compare.<model> [LEFT=aten] [RIGHT=native|native-sequential] [TOL=]
[DONOR=] [CASE=] [SAMPLE=]` runs that pair over a model, wherever it lives: a
released Transformers fixture (an artifact id, or the dotted words that single
out one, `smolvlm-256m.connector`; fetched with its Hub checkpoint if absent) and prints each cluster as
`[k/N]` with its operations on both sides and the verdict. Exit status follows the
`inference_compare` command (0 agreement, 1 disagreement, 3 refused or failed, 4
not established). LEFT is ATen because the only supplied correspondence is the
ATen-to-Native import map; two Native variants cannot yet be paired.

`make model.download.<model>` fetches what a model needs (a timm model's release
assets and Hub checkpoint, or a fixture and its checkpoint) and prints where its
graph JSON is; `bin/model_source` resolves the name for both rules.

For a timm model (`make model.compare.mobilenetv2_050`) no `.pt2` is read: graph and weight
map from the producer submodule's model directory, weights from the pinned Hub
checkpoint, inputs from the release's `inputs.pt`. `pair_cram.t` pins that it
gives the `.pt2` run's cluster count and verdict (152 clusters on
`mobilenetv2_050`) and still disagrees at zero tolerance.
