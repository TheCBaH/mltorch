#!/usr/bin/env bash
# The Loop optimizer, vectorizer, block recognizer and vector emitters are
# frozen (see the .ai/ role record for the Loop and SSA backends): they leave
# with the Loop retirement and gain no new files. The set below is a ceiling
# that only goes down: a file matching the frozen patterns must be listed in
# scripts/loop-freeze-files.txt, and a listed file that no longer exists must
# be deleted from the list.
set -euo pipefail

root="$(git rev-parse --show-toplevel)"
list="$root/scripts/loop-freeze-files.txt"
fail=0

pattern='^(loop_vector.*|loop_opt.*|loop_block.*|loop_(c|js|wasm)_vector.*)\.mli?$'

# Tracked and untracked-but-not-ignored, so a new file is caught before it is
# added.
found="$(git -C "$root" ls-files -co --exclude-standard -- '*.ml' '*.mli' \
  ':!:vendored/*' ':!:modules/*' |
  awk -F/ -v re="$pattern" '$NF ~ re' | LC_ALL=C sort)"
listed="$(grep -v '^#' "$list" | grep -v '^$' | LC_ALL=C sort)"

while IFS= read -r f; do
  [[ -n "$f" ]] || continue
  echo "FAIL: $f matches a frozen Loop pattern and is not in scripts/loop-freeze-files.txt" >&2
  fail=1
done < <(LC_ALL=C comm -23 <(printf '%s\n' "$found") <(printf '%s\n' "$listed"))

while IFS= read -r f; do
  [[ -n "$f" ]] || continue
  echo "FAIL: scripts/loop-freeze-files.txt lists $f, which no longer exists: delete the line" >&2
  fail=1
done < <(LC_ALL=C comm -13 <(printf '%s\n' "$found") <(printf '%s\n' "$listed"))

exit "$fail"
