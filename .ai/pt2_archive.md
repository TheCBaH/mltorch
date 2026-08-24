# PT2 Archive Structure

A `.pt2` file is a ZIP archive. Contents after extraction (top-level model name dir is stripped):

```
models/<name>/
  archive_format           # format identifier string
  archive_version          # version string
  byteorder                # endianness
  models/
    model.json             # serialized graph (ExportedProgram)
  data/
    weights/
      weight_0 … weight_N  # raw tensor data (flat binary)
      model_weights_config.json   # weight index (PayloadConfig)
    constants/
      model_constants_config.json # constant tensor index (PayloadConfig)
    sample_inputs/
      model.pt             # sample inputs (pickled)
  .data/
    version
    serialization_id
```

## Schema source of truth

All JSON files in the archive are serialized from Python dataclasses defined in:

```
torch/_export/serde/schema.py        ← source of truth (dataclasses + IntEnums)
torch/_export/serde/schema.yaml      ← auto-generated, human-readable
torch/_export/serde/export_schema.thrift  ← auto-generated Thrift IDL
torch/include/torch/csrc/utils/generated_serialization_types.h  ← auto-generated C++
```

`schema_check.py` generates all three from `schema.py` via `_staged_schema()`.
The `# checksum<<...>>` at the top of each generated file is SHA-256 of the content,
used to detect out-of-sync regeneration.

Current version: `SCHEMA_VERSION = (8, 14)` (major, minor).
- Major bump = breaking change (field removed or added without default).
- Minor bump = compatible change (field added with default).

## Inference source binding

`GraphSignature.input_specs` classifies graph placeholders. For native
inference lowering, `USER_INPUT` remains caller-bound; `PARAMETER`, `BUFFER`,
and `CONSTANT_TENSOR` all lower to `Constant`. Their targets are resolved
once across `data/weights/model_weights_config.json` and
`data/constants/model_constants_config.json`, then reused for every SSA use of
that placeholder. Target names are importer provenance, not native IR. The
native IR intentionally does not retain trainability, buffer persistence, or
mutation semantics. Exact aliasing between separately named payloads is also
outside this inference-only boundary.

## Functional image-model release contract

Runnable fixtures come from `TheCBaH/devcontainer.pytorch-image-models` at
`a24c245fbd66214a7749806110186ef0a389391f` / `v0.0.4`. Downloaded assets live
under `data/pt2-functional/<model>/` and contain `<model>.pt2`,
`preprocessing.json`, `expected.json`, `contract.json`, `inputs.pt`, and
`outputs.pt`.

`inputs.pt` and `outputs.pt` are external `torch.save` `dict[str, Tensor]`
archives, not the single tensor in `data/sample_inputs/model.pt`.
`Pt2_archive.load_pt_tensor_map` is consequently separate from `load_pt`: it
accepts direct string-to-tensor maps, retains tensor layout metadata, rejects
duplicates and missing storage, and returns lexical key order. ZIP file,
per-entry, and aggregate bounds apply before pickle decoding on native and
js_of_ocaml alike.

`data/pt2-functional-manifest.json` is a vendored, byte-for-byte copy of the
producer's own release `manifest.json` (schema at
`modules/devcontainer.pytorch-image-models/schemas/manifest.schema.json`) --
not a hand-authored subset. It covers every release-tier model (a superset of
what mltorch selects), each with `archive.{url,sha256,bytes,members}`, plus a
`retired` section giving a reason and migration target for models the
producer has dropped. `pt2.download` reads the URL/digest/required-members
for `PT2_MODEL` straight out of it (failing with the producer's own
retirement message if the model isn't `current`), checks `PT2_RELEASE`
against the vendored file's `producer.tag` so the two pins can't drift apart,
and validates the canonicalized release graph against the pinned producer
graph. `pt2.runtest` is network-free and only preflights existing assets
before ordinary `dune runtest`; it never promotes downloaded output.

The producer also publishes `catalogue.json` (browser-facing model picker
metadata) and `compat-report.json` (per-model graph-vs-runnable
classification) as release assets. Neither is consumed here yet -- vendoring
`manifest.json` closes the archive-digest drift gap; catalogue/compat
consumption is a separate, not-yet-needed piece of work.
