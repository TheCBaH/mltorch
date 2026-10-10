#!/usr/bin/env bash
# Complete default coverage, then separately measured declared opt-in policies.
set -euo pipefail
root=$(cd "$(dirname "$0")/.." && pwd)
cd "$root"
cohort=${1:-data/transformers/cohort.json}
cache=${2:-data/transformers-cache}
reports=${3:-_build/transformers-reports}
source=${4:-modules/devcontainer.transformers}
opam exec -- dune build bin/transformers_policies.exe bin/transformers_fixture.exe bin/transformers_replay.exe bin/transformers_matrix.exe
bin=_build/default/bin
rows=$(mktemp)
trap 'rm -f "$rows"' EXIT
"$bin/transformers_policies.exe" "$cohort" data/transformers/numerical-policy.json --rows > "$rows"
"$bin/transformers_fixture.exe" fetch "$cohort" "$cache"
failed=0
"$bin/transformers_replay.exe" "$cohort" "$cache" --source "$source" --report-dir "$reports" --dots exact --casts checked || failed=1
while IFS=$'\t' read -r artifact dots casts; do
  [[ $dots == exact && $casts == checked ]] && continue
  "$bin/transformers_replay.exe" "$cohort" "$cache" --source "$source" --report-dir "$reports" --dots "$dots" --casts "$casts" "$artifact" || failed=1
done < "$rows"
"$bin/transformers_matrix.exe" "$cohort" "$cache" "$reports" --source "$source" --json "$reports/matrix.json" --markdown "$reports/matrix.md" || failed=1
exit "$failed"
