#!/bin/sh
# Graph-only admission sweep over a devcontainer.transformers checkout.
#
#   scripts/transformers-admission.sh PRODUCER_DIR OUT_DIR
#
# PRODUCER_DIR must be at the producer commit pinned in data/transformers.
# Writes OUT_DIR/admission.jsonl (one row per committed artifact) and
# OUT_DIR/run.json (pins, consumer commit and workspace state). Offline.
set -eu
producer=${1:?producer checkout}
out=${2:?output directory}
root=$(cd "$(dirname "$0")/.." && pwd)
pin=81feca91b3d3ad032cb3c1ef28f4d5e1751c1d55
have=$(git -C "$producer" rev-parse HEAD)
if [ "$have" != "$pin" ]; then
  echo "producer is at $have, expected $pin" >&2
  exit 1
fi
mkdir -p "$out"
flat=$(mktemp -d)
trap 'rm -rf "$flat"' EXIT
# The tool reads a flat directory of artifacts; nested producer IDs map to
# '--'-joined names.
python3 -I - "$producer" "$flat" <<'PY'
import json, sys
from pathlib import Path
source, flat = Path(sys.argv[1]).resolve(), Path(sys.argv[2])
for row in json.loads((source / 'catalogue.json').read_text())['artifacts']:
    (flat / row['artifact_id'].replace('/', '--')).symlink_to(
        source / row['path'], target_is_directory=True)
PY
cd "$root"
opam exec -- dune exec bin/pt2_json_model_support.exe -- "$flat" "$out/admission.jsonl"
rows=$(wc -l <"$out/admission.jsonl")
python3 -I - "$out/run.json" "$pin" "$(git rev-parse HEAD)" "$rows" "$(git status --short | wc -l)" "$out/admission.jsonl" <<'PY'
import json, sys, collections
path, pin, head, rows, dirty, adm = sys.argv[1:7]
r = [json.loads(l) for l in open(adm)]
count = lambda k: sum(1 for x in r if x.get(k) is True)
blockers = collections.Counter(x.get("native4d_blocker") or "ok" for x in r)
json.dump({
    "producer_commit": pin, "consumer_commit": head,
    "consumer_workspace_changes": int(dirty), "rows": int(rows),
    "native_builds": count("native_builds"),
    "native4d_converts": count("native4d_converts"),
    "kernel_converts": count("kernel_converts"),
    "first_blockers": dict(blockers),
}, open(path, "w"), indent=2, sort_keys=True)
open(path, "a").write("\n")
PY
cat "$out/run.json"
