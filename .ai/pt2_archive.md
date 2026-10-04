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
`ad250479db4e42f77bad1c84c459c9a2731654a8` / `v0.0.6`. Downloaded assets live
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

## Safetensors weight source

A model can run without its `.pt2`. The model submodule commits, beside each
graph, `models/<name>/models/safetensors.json`: a pinned Hugging Face checkpoint
(`source.{repo_id,revision,filename,sha256,size,url}`), a map from each captured
tensor's config name to a checkpoint `{key,dtype,shape}`, and `unmapped`. With
the committed `model.json` and weights/constants configs that is the whole
archive minus the zip.

**The seam.** Runtimes touch an opened model only through `Pt2_archive.program`
and `Pt2_archive.load_captured_tensor`. `Pt2_archive.t` holds a `payload`: a
`Zip` (the `.pt2`) or a `Loader`, a function from a config name to the tensor's
storage. `Pt2_archive.of_parts` builds the second kind; a loader's own failure
surfaces as the `` `Payload `` row, named for the source rather than classified.
The JSON decoders (`program_of_json`, `weights_config_of_json`,
`constants_config_of_json`) are shared with the zip path.

**Storage is a bigstring.** `Pt2_tensor.data` is a `Pt2_storage.t`
(`(char, int8_unsigned_elt, c_layout) Bigarray.Array1.t`), the kind that
`Unix.map_file`, js_of_ocaml's `Typed_array.Bigstring` and ocaml-safetensors'
mapped views share, so a weight from a checkpoint is a view into the mapping
with no copy. The zip path copies each member once. Reads are little-endian and
bounds-checked (the compiler's own bigstring primitives, which js_of_ocaml
implements).

**Up-front validation** (`lib/pt2_safetensors`, pure, js_of_ocaml-reachable).
`check_graph` needs no checkpoint: nothing `unmapped` (a non-persistent buffer
the checkpoint lacks makes the model unrunnable from it — `csatv2`,
`vit_small_patch16_lingbot`, `mobilenetv4_conv_blur_medium` are refused), every
captured tensor mapped, and map dtype and shape equal to the graph's, with the
graph's tensor a dense row-major buffer at offset 0 (all a checkpoint tensor can
be). `of_parts` then checks the map against the checkpoint header (key present,
dtype, shape). So a loaded tensor can only be the one the graph asked for. The
network-free check over every committed `safetensors.json` is
`test/pt2_safetensors_maps.expected`.

**Native side** (`lib/pt2_safetensors_unix`). `open_dir model_dir` reads the
configs, resolves the pinned checkpoint at its commit through
`hf-hub-safetensors` and maps it read-only, then checks the pin: the cached
blob's etag (a git-LFS file's sha256) and its size against `source`. A commit
revision already in the cache needs no network. The cache is the
huggingface_hub layout, so Python tooling shares it.

**Vendored libraries.** `vendored/ocaml-safetensors` supplies `Memory.of_bigstring`,
`tensor_view` and `Safetensors_unix.Mmap`; `vendored/ocaml-hf-hub` is a sans-IO
download core (a step machine the driver runs) with a `curl`/filesystem driver.
Hashing shells out to `sha256sum` (or `shasum -a 256`), keeping the dependency
set to what the repo already installs. The mapping is safe because a cache blob
is content-addressed and never rewritten.

**Running it.** `make safetensors.run PT2_MODEL=<m>` runs the interpreter with
`--safetensors`, `--strict`. Inputs and reference outputs still come from the
release bundle, the only place they exist. `make pt2.runtest` also checks that
every captured tensor of `mobilenetv2_050` and `fastvit_sa12` from the
checkpoint is byte-equal to the `.pt2`'s.

**Under js_of_ocaml.** The mmap is native, but the weight source is not:
`Pt2_safetensors.of_parts` takes a `Safetensors.Memory.t`, which `Memory.of_string`
builds from file bytes (a copy, where the native path maps). The Hub download is
not native-only either: `js/pt2_safetensors_js` runs hf-hub's own JavaScript
driver (`javascript/shared` in the submodule, copied by dune, with its
js_of_ocaml `Runtime` binding) and fetches the pinned checkpoint itself, then
applies the same pin check as the native side (`Pt2_safetensors.check_source`:
etag against the pinned sha256, byte count against the pinned size). The
driver's host half is plain JavaScript loaded first -- `node-host.cjs` (a
huggingface_hub-layout cache on disk, shared with the native one) or
`browser-host.js` (in memory, Web Crypto), each given a `bytes` accessor.

The entry `safetensors_probe` takes the model files as strings, so one source
serves node (`node_run.cjs`) and the page. Both run `Probe_pt2`, the source of
the native golden (`pt2_probe --safetensors <model_dir> <checkpoint> <input>`,
the checkpoint path from `Pt2_safetensors_unix.checkpoint_path`), and are diffed
against it: `make jsoo.safetensors.runtest` for node, then checks the native
output equals the `.pt2` run's; `make safetensors.browser.runtest` for Chromium
via playwright (`web/scripts/safetensors-browser-check.mjs`), comparing the
page's console output line for line. The browser reaches the live Hub through
hf-hub's example proxy, because a browser cannot read the metadata headers of the
Hub's redirecting HEAD; that run needs the network. The browser buffers the whole
checkpoint and holds it in memory, so this suits the small models, not large
checkpoints.
