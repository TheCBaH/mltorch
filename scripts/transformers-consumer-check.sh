#!/bin/sh
# Consumer checks after build setup, including the accepted upstream torchgen.
set -eu
root=$(cd "$(dirname "$0")/.." && pwd)
cd "$root"
fail() { echo "transformers consumer: $*" >&2; exit 2; }
# Producer sources and archived evidence are data; scan active consumer paths.
scan() {
  set -- bin/transformers* lib/transformers* Makefile .github/workflows
  for path in scripts/transformers*; do
    [ "$path" = scripts/transformers-consumer-check.sh ] || set -- "$@" "$path"
  done
  rg -n '(^|[[:space:]])(import torch|from torch|import transformers|from transformers)|pip.*(torch|transformers)|python[0-9]*.*modules/devcontainer.transformers' "$@"
}
scan_status=0
scan || scan_status=$?
if [ "$scan_status" -eq 0 ]; then
  fail 'active consumer ML Python dependency'
fi
[ "$scan_status" -eq 1 ] || fail 'cannot scan active consumer paths'
opam exec -- dune build bin/transformers_source.exe bin/pt2_json_model_support.exe bin/transformers_policies.exe
scratch=$(mktemp -d)
trap 'rm -rf "$scratch"' EXIT HUP INT TERM
# A flat directory of installed tools except Python; omit scripts whose shebang
# would bypass PATH with an absolute Python interpreter.
old_path=$PATH
printf '%s' "$old_path" | tr ':' '\n' > "$scratch/paths"
mkdir "$scratch/bin"
while IFS= read -r dir; do
  [ -d "$dir" ] || continue
  for file in "$dir"/*; do
    [ -f "$file" ] && [ -x "$file" ] || continue
    name=${file##*/}
    case "$name" in python*|pypy*|pip*) continue ;; esac
    [ ! -e "$scratch/bin/$name" ] || continue
    if head -c 128 "$file" 2>/dev/null | LC_ALL=C grep -aEq '^#!.*python'; then continue; fi
    ln -s "$file" "$scratch/bin/$name"
  done
done < "$scratch/paths"
PATH=$scratch/bin
export PATH
if command -v python || command -v python3; then fail 'Python remains available'; fi
make transformers.policies.check transformers.admission transformers.admission.normalized
opam exec -- dune runtest --force test/pt2_fixture test/transformers_metadata test/transformers_tasks test/transformers_source.t test/native_interp
# Optional pinned fixture command runs under the same restricted environment.
if [ "$#" -gt 0 ]; then "$@"; fi
