#!/usr/bin/env python3
"""Build data/transformers/cohort.json from a pinned release publication index.

Usage: transformers-cohort.py PUBLICATION_JSON OUT_JSON

Offline except for the four cohort manifests and archives, which are fetched
from the release URL the index names and verified against the index's size and
sha256. The archive's map member is checked against the manifest and its
checkpoint-file pins are recorded, so a source that is not a release asset
(an upstream Hub file) is pinned here too.
The output pins the index bytes, each selected artifact's archive and
manifest, its graph/contract digests and its source files, so replay can
check every layer independently.
"""
import hashlib
import io
import json
import sys
import tarfile
import urllib.request

COHORT = [
    ("mobilevit-xxs/image-classification/reference/forward/fp32/dynamo/static/ckpt-6703997f9e94", "first numerical milestone"),
    ("smolvlm-256m/image-text-generation/reference/connector/fp32/dynamo/static/ckpt-7e3e67edbbed", "first numerical milestone"),
    ("bert-tiny/text-encoder/reference/forward/fp32/dynamo/static/ckpt-6f75de8b60a9", "needs embedding"),
    ("tinyclip/image-text-embeddings/reference/text-encoder/fp32/dynamo/static/ckpt-a2a8c6eaa254", "text tower follow-up"),
    ("segformer-b0/semantic-segmentation/reference/forward/fp32/dynamo/static/ckpt-489d5cd81a0b", "second vision family"),
    ("yolos-tiny/object-detection/reference/forward/fp32/dynamo/static/ckpt-95a90f3c189f", "detection"),
    ("tinyclip/image-text-embeddings/reference/image-encoder/fp32/dynamo/static/ckpt-a2a8c6eaa254", "vision tower, shares the text tower's checkpoint"),
    ("whisper-tiny/audio-encoder-decoder/reference/encoder/fp32/dynamo/static/ckpt-169d4a4341b3", "audio encoder"),
    ("whisper-tiny/audio-encoder-decoder/reference/prefill/fp32/dynamo/static/ckpt-169d4a4341b3", "encoder-decoder prefill, self/cross K/V"),
    ("smollm2-135m/text-decoder/reference/prefill/fp32/dynamo/static/ckpt-12fd25f77366", "decoder prefill, K/V cache"),
    ("smollm2-135m/text-decoder/reference/decode/fp32/dynamo/static-h4/ckpt-12fd25f77366", "decoder decode at history 4"),
    ("smolvlm-256m/image-text-generation/reference/decode/fp32/dynamo/static-h71/ckpt-7e3e67edbbed", "VLM decode at history 71, shares the connector's checkpoint"),
]


def fetch(asset):
    data = urllib.request.urlopen(asset["url"]).read()
    assert len(data) == asset["size"], asset["name"]
    assert hashlib.sha256(data).hexdigest() == asset["sha256"], asset["name"]
    return data


def main(publication_path, out_path):
    raw = open(publication_path, "rb").read()
    index = json.loads(raw)
    by_id = {a["artifact_id"]: a for a in index["artifacts"]}
    artifacts = []
    for artifact_id, role in COHORT:
        a = by_id[artifact_id]
        manifest = json.loads(fetch(a["assets"]["manifest"]))
        sources = [v for k, v in a["assets"].items() if k.startswith("v2:")]
        archive = tarfile.open(fileobj=io.BytesIO(fetch(a["assets"]["archive"])))
        member = manifest["map_v2"]["member"]
        map_bytes = archive.extractfile(member).read()
        assert hashlib.sha256(map_bytes).hexdigest() == manifest["members"][member]["sha256"]
        v2 = json.loads(map_bytes)
        assert v2["artifact_id"] == artifact_id and v2["graph_sha256"] == a["graph_sha256"]
        artifacts.append({
            "artifact_id": artifact_id,
            "role": role,
            "archive": a["assets"]["archive"],
            "manifest": a["assets"]["manifest"],
            "graph_sha256": a["graph_sha256"],
            "contract_sha256": manifest["contract_sha256"],
            "map_member": manifest["map_v2"]["member"],
            "map_sha256": manifest["members"][manifest["map_v2"]["member"]]["sha256"],
            "cases": manifest["cases"],
            "released_sources": sources,
            "map_sources": v2["sources"],
            "capture_count": len(v2["tensors"]),
            "weight_source": a["weight_source"],
        })
    out = {
        "schema_version": 1,
        "release_tag": index["release_tag"],
        "repository": index["repository"],
        "release_producer_commit": index["artifacts"][0]["producer_commit"],
        "publication": {
            "size": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest(),
            "url": "https://github.com/%s/releases/download/%s/publication.json"
            % (index["repository"], index["release_tag"]),
        },
        "artifacts": artifacts,
    }
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, sort_keys=True)
        f.write("\n")


if __name__ == "__main__":
    main(*sys.argv[1:3])
