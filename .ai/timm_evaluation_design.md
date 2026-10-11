# Timm evaluation assets and input identity

The consumer probes `manifest-v2.json` before the existing v1 path. Only an
HTTP 404 permits legacy fallback. Request failures, malformed JSON, unsupported
versions, unresolved references and digest mismatches fail acquisition. A local
manifest override is strict and supports unpublished Actions dry runs.

V2 discovery references evaluation ZIPs and standalone inputs independently.
Evaluation ZIPs must contain exactly `cases.json`, `contract-v2.json`,
`expected.json`, `outputs.pt` and `preprocessing.json`. No full model archive
or `.pt2` is opened on this path. The graph and safetensors map are verified
against the manifest pins before loading the independently pinned Hub weights.
Checkout graph/map identity must agree; model names alone cannot select a
different producer revision. A model without a usable map refuses execution.

Files enter the existing digest-addressed cache only after size and SHA256
checks and atomic promotion. Corrupt offline entries fail; online acquisition
can replace them after verification. Images and their metadata share the
cache across models. Local dry-run asset copies use the same integrity checks.

Bindings follow the graph's user-input/output signatures and exclude captures.
Case dictionaries select tensor keys explicitly; lexical key ordering is not
binding order. The accepted transforms are input identity/leading unsqueeze
and output identity or leading index-zero selection followed by float32 cast.
Coverage, static shapes, dtypes, layouts and canonical hashes must agree.
Canonical tensor hashes cover logical row-major little-endian bytes, excluding
unused backing storage. Signed zero matters; nonfinite values are rejected.
Full outputs use the producer tolerances rather than the rounded top-five JSON.

Validation failures carry closed variants and identifiers; output mismatches
retain the element index and both values. Imported library errors keep a named
source, and JSON/native-layout messages remain separate third-party tags.

`TIMM_MODE=authoritative` loads standalone inputs. `TIMM_MODE=images` reconstructs
them from verified RGB CHW uint8 images and the published preprocessing recipe.
Every generated input must match its case descriptor before graph execution.
The comparison coordinator also checks its final outputs against the publication.

The portable resizer follows [Pillow 12.3.0's separable resampler](https://github.com/python-pillow/Pillow/blob/12.3.0/src/libImaging/Resample.c):
double-precision coefficient normalization, signed 22-bit coefficients,
horizontal then vertical uint8 rounding and clipping. Geometry follows the
pinned timm/torchvision shortest-edge, squash and border rules, including
truncation versus ties-to-even, asymmetric padding and observed event checks.
Normalization separately rounds float32 division by 255, subtraction of
float32 mean and division by float32 standard deviation.

Producer Actions run 38164099114 supplies the committed patterned references.
They cover both interpolations, all three crop modes, square/nonsquare targets,
portrait/landscape/small sources, odd offsets and padding. They require no
Python or Torch runtime in the consumer. Run 38163037026 supplies the full
68-model dry run at producer `1d58d21710376b2b1dafb4c68402c7aad964e141`.
Local acceptance reproduced all 680 published inputs exactly; MobileNetV2
passed all ten full-output cases through authoritative and image inputs,
and the image path passed all 152 ATen/Native comparison units.

`TIMM_TRANSFER_LOG` records actual acquired file-body bytes per URL, including
local artifact copies; it excludes cache hits and HTTP headers. Hub checkpoint
bytes and checkout graph/config metadata must be counted separately. An archive
total is not a claim that all its traffic disappears: Hub weights still need
acquisition, and the shared images cost is amortized across models.

## Measured acquisition sizes

Measurements use the dry-run release above, a cold fixture cache and the pinned
MobileNetV2 checkpoint. The public v0.0.6 archive was downloaded independently
(9,781,960 bytes; SHA256 `eed9209e3a38529073c248c9428d6eb385409645af42f3a23cffb0cea8c61240`).
The Hub blob acquired for replay is 7,976,780 bytes, matching the weight-map pin.
The shared path's transfer log records 9,775,951 acquired fixture/graph/map bytes;
its checkpoint is reused from the authoritative run. These are body bytes,
excluding transport headers/retries and Git object packing.

| Cold acquisition component | Legacy comparison | V2 authoritative | V2 shared images |
| --- | ---: | ---: | ---: |
| Discovery manifest | 41,859 | 168,569 | 168,569 |
| Model archive / evaluation + inputs | 9,781,960 | 6,064,250 | 39,669 |
| Decoded images + attribution/metadata | 0 | 0 | 9,272,824 |
| Graph JSON + map + weight/constant configs | 558,863 | 558,863 | 558,863 |
| Hub checkpoint | 7,976,780 | 7,976,780 | 7,976,780 |
| Total file payload | 18,359,462 | 14,768,462 | 18,016,705 |

The first shared-image model pays the one-time 9,272,824-byte shared asset cost;
additional models reuse it by digest. Warm-cache replay acquires no fixture,
graph, map or checkpoint files with an explicit local manifest. Remote discovery
may still request its manifest.

Across all 68 models, staged full archives total 2,374,437,804 bytes, standalone
inputs 403,404,756 bytes and evaluation ZIPs 2,846,966 bytes. Shared image and
metadata assets plus all evaluations total 12,119,790 bytes. These are measured
file sizes, not total network savings. The 67 available maps pin 2,285,379,706
checkpoint bytes; 65 usable maps pin 2,201,748,954 bytes. `rdnet_tiny` has no map,
and `csatv2` / `mobilenetv4_conv_blur_medium` have unmapped captures. Their
preprocessing verification still passes, but the independent Hub execution path
refuses those models. Graph/map/config bytes and Hub HTTP metadata remain costs.
