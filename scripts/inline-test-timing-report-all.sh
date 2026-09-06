#!/usr/bin/env bash
# Runs inline-test-timing-report.sh over every (inline_tests) library in the
# project (discovered by inline-test-libraries.py) for one mode: `dune
# clean`, one combined `dune build`, then time every partition individually.
# Clean+build up front keeps build time out of any test's measured time and
# forces every partition to actually execute.
set -euo pipefail

usage() {
  echo "usage: $0 <mode: best|js> [threshold-seconds]" >&2
  exit 1
}

[[ $# -ge 1 ]] || usage
mode=$1
threshold=${2:-10}

case "$mode" in
  best | js) ;;
  *) usage ;;
esac

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"

triples=()
while IFS=$'\t' read -r dir lib triple_mode; do
  [[ "$triple_mode" == "$mode" ]] || continue
  triples+=("$dir"$'\t'"$lib")
done < <(python3 scripts/inline-test-libraries.py)

(( ${#triples[@]} > 0 )) || {
  echo "inline-test-timing-report-all: no ($mode) inline_tests libraries found" >&2
  exit 1
}

echo "inline-test-timing-report-all: cleaning _build" >&2
dune clean --root .

targets=()
for triple in "${triples[@]}"; do
  IFS=$'\t' read -r dir lib <<< "$triple"
  build_dir="_build/default/$dir"
  inline_dir=".$lib.inline-tests"
  case "$mode" in
    js) runner_rel="$inline_dir/inline-test-runner.bc.js" ;;
    best) runner_rel="$inline_dir/inline-test-runner.exe" ;;
  esac
  targets+=("$build_dir/$runner_rel")
done

echo "inline-test-timing-report-all: building ${#triples[@]} ($mode) libraries" >&2
dune build --root . "${targets[@]}"

combined="_build/inline_timing_report.$mode.tsv"
mkdir -p "$(dirname "$combined")"
: > "$combined"

overall_status=0
for triple in "${triples[@]}"; do
  IFS=$'\t' read -r dir lib <<< "$triple"
  per_lib_report="_build/inline_timing_report.$lib.$mode.tsv"
  echo "inline-test-timing-report-all: running $lib ($mode)" >&2
  if ! scripts/inline-test-timing-report.sh --no-build "$dir" "$lib" "$mode" "$threshold" "$per_lib_report"; then
    overall_status=1
  fi
  sed "s/^/$lib\t/" "$per_lib_report" >> "$combined"
done

sort -t $'\t' -k2,2rn -o "$combined" "$combined"
echo >&2
echo "inline-test-timing-report-all: combined report ($mode, threshold=${threshold}s): $combined" >&2
cut -f1-4 "$combined" | column -t -s $'\t' >&2

if [[ "$overall_status" -eq 0 ]]; then
  echo "inline-test-timing-report-all: all ${#triples[@]} ($mode) libraries stayed under ${threshold}s per partition" >&2
else
  echo "inline-test-timing-report-all: FAIL -- see above" >&2
fi
exit "$overall_status"
