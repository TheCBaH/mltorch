#!/usr/bin/env bash
# Fetch and verify one release-tier model archive from the vendored producer
# manifest (data/pt2-functional-manifest.json). Invoked by `make pt2.download`;
# see Makefile's PT2_* variables for the caller-side defaults.
#
# Usage: pt2-download.sh <model> <manifest.json> <model-dir> <release-tag>
set -euo pipefail

model=$1
manifest=$2
model_dir=$3
release=$4
submodule_graph="modules/devcontainer.pytorch-image-models/models/$model/models/model.json"

tag=$(jq -r '.producer.tag' "$manifest")
if [ "$tag" != "$release" ]; then
	echo "pt2.download: PT2_RELEASE=$release does not match $manifest producer.tag=$tag" >&2
	exit 1
fi

mkdir -p "$model_dir"
zip="$model_dir/$model.zip"

url=$(jq -r --arg m "$model" '.models[$m].archive.url // empty' "$manifest")
sha=$(jq -r --arg m "$model" '.models[$m].archive.sha256 // empty' "$manifest")
if [ -z "$url" ]; then
	reason=$(jq -r --arg m "$model" '.retired[$m].reason // "not a release-tier model"' "$manifest")
	migrate=$(jq -r --arg m "$model" '.retired[$m].migrate_to // empty' "$manifest")
	echo "pt2.download: $model is not in $manifest: $reason" >&2
	test -z "$migrate" || echo "pt2.download: migrate to $migrate" >&2
	exit 1
fi

test -f "$zip" || curl -fsSL -o "$zip" "$url"
echo "$sha  $zip" | sha256sum -c -

(cd "$model_dir" && unzip -o "$model.zip")

members=$(jq -r --arg m "$model" '.models[$m].archive.members[]' "$manifest")
for f in $members; do
	test -f "$model_dir/$f" || { echo "pt2.download: $model archive lacks $f" >&2; exit 1; }
done

release_graph=$(mktemp)
source_graph=$(mktemp)
trap 'rm -f "$release_graph" "$source_graph"' EXIT
unzip -p "$model_dir/$model.pt2" "$model/models/model.json" | jq -S . >"$release_graph"
jq -S . "$submodule_graph" >"$source_graph"
cmp -s "$release_graph" "$source_graph" || {
	echo "pt2.download: release graph differs from pinned producer graph for $model" >&2
	exit 1
}
