#!/bin/sh
# Offline graph admission from the consumer's pinned producer gitlink.
# scripts/transformers-admission.sh [PRODUCER_DIR [OUT_DIR]]
set -eu
root=$(cd "$(dirname "$0")/.." && pwd)
producer=${1:-"$root/modules/devcontainer.transformers"}
out=${2:-"$root/_build/transformers-admission"}
fail() { echo "transformers admission: $*" >&2; exit 2; }
pin=$(git -C "$root" ls-files --stage -- modules/devcontainer.transformers |
  awk '$1 == "160000" && $3 == "0" {print $2}')
[ -n "$pin" ] || fail "no producer gitlink in consumer index"
[ -d "$producer" ] || fail "missing checkout; run git submodule update --init modules/devcontainer.transformers"
producer=$(cd "$producer" && pwd -P)
top=$(git -C "$producer" rev-parse --show-toplevel 2>/dev/null) ||
  fail "uninitialized checkout; run git submodule update --init modules/devcontainer.transformers"
[ "$top" = "$producer" ] || fail "uninitialized checkout; run git submodule update --init modules/devcontainer.transformers"
have=$(git -C "$producer" rev-parse HEAD)
[ "$have" = "$pin" ] || fail "producer is at $have, expected gitlink $pin; run git submodule update --init modules/devcontainer.transformers"
dirty=$(git -C "$producer" status --porcelain --untracked-files=all)
[ -z "$dirty" ] || fail "modified producer inputs: $dirty"
mkdir -p "$out"
out=$(cd "$out" && pwd -P)
flat=$(mktemp -d)
trap 'rm -rf "$flat"' EXIT HUP INT TERM
# Compare actual catalogue bytes to the pinned tree even with assume-unchanged
# or skip-worktree. Its member pins then prove the consumed artifact files.
git -C "$producer" show "$pin:catalogue.json" > "$flat/catalogue.pinned"
cmp -s "$producer/catalogue.json" "$flat/catalogue.pinned" || fail "modified catalogue bytes"
rm "$flat/catalogue.pinned"
cd "$root"
opam exec -- dune exec bin/transformers_source.exe -- inventory "$producer" "$flat" "$out/inventory.json"
opam exec -- dune exec bin/pt2_json_model_support.exe -- "$flat" "$out/admission.jsonl"
opam exec -- dune exec bin/transformers_source.exe -- summary \
  "$out/inventory.json" "$out/admission.jsonl" "$out/run.json" "$pin" \
  "$(git rev-parse HEAD)" "$(git status --short | wc -l | tr -d ' ')"
cat "$out/run.json"
