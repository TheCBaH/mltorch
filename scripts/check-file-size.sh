#!/usr/bin/env bash
# Prevent regression on authored .ml/.mli file size.
#
#   - Tree check: no tracked, non-exempt .ml/.mli file may exceed 1000 lines,
#     checked against the current tree (not a diff against any base ref).
#   - New-file check: a .ml/.mli file added by HEAD (relative to HEAD^) must
#     not exceed 750 lines, unless exempt.
#
# Cram (.t) snapshots are out of scope: they get a different, non-line-count
# exception (one command/model per file) that this script cannot check
# mechanically.
set -euo pipefail

root="$(git rev-parse --show-toplevel)"
exceptions="$root/scripts/file-size-exceptions.txt"
fail=0

is_exempt() {
  local path="$1"
  [[ -f "$exceptions" ]] || return 1
  awk -F'|' -v p="$path" '!/^#/ && NF >= 1 && $1 == p { found = 1 } END { exit !found }' \
    "$exceptions"
}

pathspec=('*.ml' '*.mli' ':!:vendored/*' ':!:modules/*' ':!:lib/generated/*')

echo "== file-size: tree check (>1000 lines) ==" >&2
while IFS= read -r f; do
  [[ -f "$root/$f" ]] || continue
  lines=$(wc -l < "$root/$f")
  if (( lines > 1000 )) && ! is_exempt "$f"; then
    echo "FAIL: $f has $lines lines (>1000 line cap), not listed in scripts/file-size-exceptions.txt" >&2
    fail=1
  fi
done < <(git -C "$root" ls-files -- "${pathspec[@]}")

echo "== file-size: new-file check (>750 lines) ==" >&2
if git -C "$root" rev-parse -q --verify HEAD^ >/dev/null 2>&1; then
  while IFS=$'\t' read -r status f; do
    [[ "$status" == "A" ]] || continue
    [[ -f "$root/$f" ]] || continue
    lines=$(wc -l < "$root/$f")
    if (( lines > 750 )) && ! is_exempt "$f"; then
      echo "FAIL: $f is new at $lines lines (>750 line cap for new files), not listed in scripts/file-size-exceptions.txt" >&2
      fail=1
    fi
  done < <(git -C "$root" diff-tree --no-commit-id --name-status -r HEAD^ HEAD -- "${pathspec[@]}")
else
  echo "note: HEAD has no parent commit, skipping new-file check" >&2
fi

exit "$fail"
