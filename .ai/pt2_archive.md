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
