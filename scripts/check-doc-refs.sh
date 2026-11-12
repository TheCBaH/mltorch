#!/usr/bin/env bash
# Ratchet on references that rot: a file name inside `.ai/`, a reference to an
# untracked plan/tracker/census, and a numbered Gate or Stage. Documentation and
# comments outside `.ai/` name the topic instead (see CLAUDE.md).
#
# Counts matching lines in tracked files outside `.ai/`, `vendored/`, `modules/`.
# `scripts/doc-refs-ceiling.txt` holds one `name|count` line per pattern. A count
# above its ceiling fails (a new reference); a count below it fails too (lower the
# ceiling with the fix, so it only goes down).
#
#   scripts/check-doc-refs.sh          check against the ceilings
#   scripts/check-doc-refs.sh --list   print the current counts as ceiling lines
set -euo pipefail

root="$(git rev-parse --show-toplevel)"
ceilings="$root/scripts/doc-refs-ceiling.txt"

# name|ERE. The script and the ceiling file define the patterns, so they are skipped.
patterns=(
  'ai-file|\.ai/[A-Za-z0-9_.-]+\.md'
  'gate|Gate [0-9]'
  'stage|Stage [0-9]'
  'untracked-doc|[_-](plan|tracker|census)\.(md|tsv|txt|json|csv)'
)

count() {
  # grep exits 1 on no match, the goal state, so it must not fail the pipeline.
  { git -C "$root" grep -I -h -E -e "$1" -- . \
      ':!:.ai/*' ':!:vendored/*' ':!:modules/*' \
      ':!:scripts/check-doc-refs.sh' ':!:scripts/doc-refs-ceiling.txt' || true; } \
    | wc -l
}

if [[ "${1:-}" == "--list" ]]; then
  for entry in "${patterns[@]}"; do
    echo "${entry%%|*}|$(count "${entry#*|}")"
  done
  exit 0
fi

fail=0
for entry in "${patterns[@]}"; do
  name="${entry%%|*}"
  n=$(count "${entry#*|}")
  ceiling=$(awk -F'|' -v k="$name" '!/^#/ && $1 == k { print $2 }' "$ceilings")
  if [[ -z "$ceiling" ]]; then
    echo "FAIL: no ceiling for '$name' in scripts/doc-refs-ceiling.txt" >&2
    fail=1
  elif (( n > ceiling )); then
    echo "FAIL: $name has $n matching lines, ceiling is $ceiling; name the topic instead (git grep -n -E '${entry#*|}')" >&2
    fail=1
  elif (( n < ceiling )); then
    echo "FAIL: $name is down to $n lines, ceiling still $ceiling; lower it in scripts/doc-refs-ceiling.txt" >&2
    fail=1
  fi
done
exit "$fail"
