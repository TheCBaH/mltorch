#!/usr/bin/env bash
# Run torchgen to emit the ATen generated headers + core sources needed to
# build the ATen "core" subset. Two separate invocations on purpose: combining
# --generate headers and --generate sources in one call drops the headers.
# --per-operator-headers is intentionally omitted (keeps the 25 monolithic
# headers; no ops/ explosion). install_dir must be ABSOLUTE or headers are
# silently skipped.
#
#   $1 = path to the pytorch source root (the git submodule)
# Output: ./gen/ATen/...  (so <ATen/core/TensorBody.h> resolves with -Igen)
set -euo pipefail
PT=$(cd "$1" && pwd)
A="$(pwd)/gen/ATen"
mkdir -p "$A"
for what in headers sources; do
  env PYTHONPATH="$PT" python3 -m torchgen.gen \
    --source-path "$PT/aten/src/ATen" \
    --install_dir "$A" \
    --generate "$what"
done
