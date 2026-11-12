#!/usr/bin/env bash
# Every exception declared in `lib/` (top-level, nested in a module, or a
# `let exception`) must be listed in `scripts/exceptions-allowlist.txt` with a
# category. A private exception used to escape a deep walk is the defect
# `Err.Escape` removes (see CLAUDE.md), so a new one needs a reason.
#
#   path|Name|category
#   ceiling|escape-candidate|N
#
# Categories: escape-candidate (to become Err.Escape; counted, the ceiling only
# goes down), ffi, invariant, local-control, public-api. A listed entry with no
# matching declaration fails (stale), as does an escape-candidate count that is
# not equal to its ceiling.
#
#   scripts/check-exceptions.sh          check against the allowlist
#   scripts/check-exceptions.sh --list   print the current declarations as path|Name
set -euo pipefail

root="$(git rev-parse --show-toplevel)"
allow="$root/scripts/exceptions-allowlist.txt"

declared() {
  git -C "$root" grep -I -n -E -e '^[[:space:]]*(let[[:space:]]+)?exception[[:space:]]+[A-Z][A-Za-z0-9_]*' \
    -- 'lib/*.ml' 'lib/*.mli' ':!:lib/generated/*' \
    | sed -E 's/^([^:]+):[0-9]+:[[:space:]]*(let[[:space:]]+)?exception[[:space:]]+([A-Z][A-Za-z0-9_]*).*/\1|\3/' \
    | sort -u
}

if [[ "${1:-}" == "--list" ]]; then
  declared
  exit 0
fi

fail=0
entries=$(awk -F'|' '!/^#/ && NF >= 3 && $1 != "ceiling" { print $1 "|" $2 }' "$allow" | sort -u)
valid='escape-candidate ffi invariant local-control public-api'

while IFS='|' read -r path name cat extra; do
  [[ -z "$path" || "$path" == \#* || "$path" == ceiling ]] && continue
  if [[ " $valid " != *" $cat "* ]]; then
    echo "FAIL: $path|$name has category '$cat'; expected one of: $valid" >&2
    fail=1
  fi
done < "$allow"

while IFS= read -r d; do
  if ! grep -qxF -- "$d" <<< "$entries"; then
    echo "FAIL: exception ${d#*|} in ${d%%|*} is not in scripts/exceptions-allowlist.txt; use Err.Escape for a walk exit, or add an entry with a category" >&2
    fail=1
  fi
done < <(declared)

now=$(declared)
while IFS= read -r e; do
  [[ -z "$e" ]] && continue
  if ! grep -qxF -- "$e" <<< "$now"; then
    echo "FAIL: stale allowlist entry ${e}: no such declaration; remove it" >&2
    fail=1
  fi
done <<< "$entries"

n=$(awk -F'|' '!/^#/ && $3 == "escape-candidate"' "$allow" | wc -l)
ceiling=$(awk -F'|' '$1 == "ceiling" && $2 == "escape-candidate" { print $3 }' "$allow")
if [[ -z "$ceiling" ]]; then
  echo "FAIL: no 'ceiling|escape-candidate|N' line in scripts/exceptions-allowlist.txt" >&2
  fail=1
elif (( n > ceiling )); then
  echo "FAIL: $n escape-candidate entries, ceiling is $ceiling" >&2
  fail=1
elif (( n < ceiling )); then
  echo "FAIL: $n escape-candidate entries, ceiling still $ceiling; lower it" >&2
  fail=1
fi
exit "$fail"
