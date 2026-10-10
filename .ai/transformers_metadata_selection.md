# Source-linked cohort and adapter metadata

`lib/transformers_metadata` owns native OCaml metadata tooling. It shares the
existing source-inventory helper and fixture/hash/bounded-bundle libraries.
Consumer commands never execute producer Python or convert tensors.

`data/transformers/selection.json` specifies the publication byte pin, release
identity/producer commit, and ordered artifact IDs with informational roles.
`transformers_cohort (generate|check|fetch)` consumes that explicit selection,
the verified source checkout and a content-addressed cache. Generate/check are
offline; fetch adds the existing curl transport. The generator verifies unique
IDs, schema, producer commits in selected publication rows and manifests, all
member hashes, graph/config/capture/map consistency, and map/contract/checkpoint
identity before writing a cohort. It only needs metadata bundles, not weights.
Check compares every derived field; serialization sorts object keys and keeps
declared array order. Errors use `Err` and typed metadata/fixture fault variants.

Source consumption checks the consumer's gitlink, producer HEAD, repository root,
checkout cleanliness, committed catalogue/task/candidate/recipe/schema/lock
bytes and all catalogue members. Reference configs are checked against committed
candidate hashes. Relevant recipe Python is read and hashed as source metadata;
it is never executed. The reviewed candidate manifest is JSON encoded even
though its filename has a YAML extension. Unsupported encodings/schemas refuse.

The source catalogue contains tiny random-weight graphs, while selected release
graphs use checkpoint populations. Their hashes and histories are independent.
The source report records the exact matching component, or a candidate's forward
recipe when the model has no tiny catalogue entry. Static histories match the
source recipe family, and each release's own history is checked against its
contract. No absent tiny graph is invented and no release graph is required to
share the tiny graph's digest. The report pins consumed source metadata and
records archive sizes and unique checkpoint source sizes separately.

`data/transformers/adapter-selection.json` declares bounded BERT WordPiece,
TinyCLIP and MobileViT adapters. `transformers_assets (generate|check|fetch)`
verifies source task metadata, model/revision/config identity, actual pinned
config/vocabulary/tokenizer/processor bytes and the complete static release
input contract. It derives sizes and processor fields deterministically rather
than believing fields in a consumer manifest. It checks the supported CLIP text
pipeline and special IDs, BERT vocabulary/special IDs, processor type, resize,
crop, resampling, normalization/BGR policy and label/config dimensions.
Old processor configs omit rescaling flags; the supported fixed defaults are
explicitly checked against any overrides. Raw image decoding remains binary PPM
in the examples; text remains bounded ASCII. These checks establish metadata and
recipe compatibility, not numerical or task parity.

Make targets expose offline `transformers.cohort.check`/`.generate` and
`transformers.assets.check`/`.generate`; `.fetch` variants explicitly acquire
missing metadata/assets. The default cache is `data/transformers-cache`.
The selected 22 archives occupy 90,463,444 bytes; unique checkpoint sources total
1,825,580,473 bytes. This is disk acquisition sizing, not runtime memory usage.
The old Python cohort generator is retired; old release/checkpoint pins remain
unchanged. Historical release/task evidence remains historical.

Hermetic tests build a synthetic pinned release and verify cold metadata-only
acquisition, offline/deterministic regeneration and corruption refusal. They
also cover publication/source/manifest producer mismatches, duplicate IDs,
schemas, missing map members, unsupported components, recipe/contract/pixel
mismatches and malformed adapter requests. Source checkout refusal tests remain
in the admission cram. Real full-cohort/three-adapter evidence and exact commands
are retained in the integration evidence directory under `ai/`.

The demos still need C5 to consume these verified assets at their own entry
points; metadata validation alone does not repair their runtime validation.
Producer-owned raw task/diagnostic fixtures, immutable numerical run identities,
numerical policy decisions and the remaining backend/lifecycle/CI work stay open.
