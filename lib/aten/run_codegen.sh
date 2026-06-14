#!/usr/bin/env bash
# Run torchgen to emit the ATen generated headers + sources needed to build the
# ATen "core" subset PLUS the real CPU operator kernels under STATIC DISPATCH.
# Two separate invocations on purpose: combining --generate headers and
# --generate sources in one call drops the headers. --per-operator-headers is
# intentionally omitted (keeps the 25 monolithic headers; no ops/ explosion).
# install_dir must be ABSOLUTE or headers are silently skipped.
#
# Static-dispatch flags (so at::add / at::mul compile to direct backend calls,
# no Dispatcher — see .ai/aten_core_build.md):
#   --static-dispatch-backend CPU    each op's ::call -> at::cpu::X directly
#   --skip-dispatcher-op-registration emits EMPTY TORCH_LIBRARY_IMPL blocks, so
#                                     no static-init force-links the kernels;
#                                     the bounded closure is reached on demand.
#
#   $1 = path to the pytorch source root (the git submodule)
# Output: ./gen/ATen/...  (so <ATen/core/TensorBody.h> resolves with -Igen)
#         ./gen/torch/headeronly/core/  (so <torch/headeronly/core/enum_tag.h>
#         resolves with -Igen; the PT source tree now has a redirect stub that
#         points there instead of emitting the content directly).
set -euo pipefail
PT=$(cd "$1" && pwd)
A="$(pwd)/gen/ATen"
H="$(pwd)/gen/torch/headeronly/core"
mkdir -p "$A" "$H"
# torchgen unconditionally emits the AOTInductor C-shim sources during
# `--generate sources`; --aoti-install-dir defaults to a CWD-RELATIVE
# `torch/csrc/inductor/aoti_torch/generated`, which would leak into the source
# tree. We do not use those shims, so pin the dir INSIDE the gen/ build target
# (an absolute path) — they become tracked build artifacts, cleaned with gen/.
for what in headers sources; do
  env PYTHONPATH="$PT" python3 -m torchgen.gen \
    --source-path "$PT/aten/src/ATen" \
    --install_dir "$A" \
    --aoti-install-dir "$A/aoti_unused" \
    --headeronly-install-dir "$H" \
    --static-dispatch-backend CPU \
    --skip-dispatcher-op-registration \
    --generate "$what"
done
