#!/usr/bin/env bash
# Generate the two configured headers a default CPU / static build needs:
#   c10/macros/cmake_macros.h  — every #cmakedefine left undefined.
#   ATen/Config.h              — every feature off, AT_PARALLEL_NATIVE=1.
# CMake normally configures these from the .in templates; we substitute by hand
# for a plain CPU build (no MKL/BLAS/OpenMP/CUDA). Config.h is required once the
# real native kernels (ParallelNative, TensorIterator, ...) are compiled in.
#
#   $1 = path to the pytorch source root (the git submodule)
# Output: ./inc/c10/macros/cmake_macros.h, ./inc/ATen/Config.h
set -euo pipefail
PT=$(cd "$1" && pwd)
mkdir -p inc/c10/macros inc/ATen
sed -E 's@#cmakedefine ([A-Za-z0-9_]+)@/* #undef \1 */@' \
  "$PT/c10/macros/cmake_macros.h.in" > inc/c10/macros/cmake_macros.h

# @VAR@ -> 0 for every feature flag, except AT_PARALLEL_NATIVE -> 1 (native
# threading; AT_PARALLEL_OPENMP stays 0). Test these with #if, not #ifdef.
sed -E 's|@AT_PARALLEL_NATIVE@|1|; s|@[A-Za-z0-9_]+@|0|g' \
  "$PT/aten/src/ATen/Config.h.in" > inc/ATen/Config.h
