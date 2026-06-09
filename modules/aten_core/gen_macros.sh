#!/usr/bin/env bash
# Generate c10/macros/cmake_macros.h for a default CPU / static build.
# CMake configures this from cmake_macros.h.in; for a plain CPU build every
# option is left undefined, so each #cmakedefine becomes an /* #undef */.
#
#   $1 = path to the pytorch source root (the git submodule)
# Output: ./inc/c10/macros/cmake_macros.h (relative to cwd = dune build dir)
set -euo pipefail
PT=$(cd "$1" && pwd)
mkdir -p inc/c10/macros
sed -E 's@#cmakedefine ([A-Za-z0-9_]+)@/* #undef \1 */@' \
  "$PT/c10/macros/cmake_macros.h.in" > inc/c10/macros/cmake_macros.h
