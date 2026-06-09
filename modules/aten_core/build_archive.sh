#!/usr/bin/env bash
# Compile the ATen "core" subset into a static archive libaten_core.a.
# Closure (CPU, static, no CUDA/MKL): c10/core + c10/util (75) + ATen/core (45)
# hand-written sources, plus the 2 generated core sources. Third-party deps are
# fmt (header-only) and cpuinfo (headers); both are pytorch submodules and must
# be checked out:  git submodule update --init third_party/fmt third_party/cpuinfo
#
#   $1 = path to the pytorch source root (the git submodule)
# Prereqs in cwd (produced by sibling rules): ./gen, ./inc
# Output: ./libaten_core.a
set -euo pipefail
PT=$(cd "$1" && pwd)

for need in "$PT/third_party/fmt/include/fmt/core.h" \
            "$PT/third_party/cpuinfo/include/cpuinfo.h"; do
  if [ ! -f "$need" ]; then
    echo "ERROR: missing $need" >&2
    echo "Run: git submodule update --init third_party/fmt third_party/cpuinfo" >&2
    exit 1
  fi
done

INC=(-I"$PT" -I"$PT/aten/src" -Igen -Iinc
     -I"$PT/third_party/fmt/include" -I"$PT/third_party/cpuinfo/include")
FLAGS=(-std=c++17 -Os -fPIC -DFMT_HEADER_ONLY=1)

# Source list: c10 core+util, ATen/core (excluding *test*), generated core srcs.
mapfile -t SRCS < <(
  find "$PT/c10/core" "$PT/c10/util" -maxdepth 1 -name '*.cpp' ! -name '*test*'
  find "$PT/aten/src/ATen/core" -name '*.cpp' ! -name '*test*'
  echo gen/ATen/core/ATenOpList.cpp
  echo gen/ATen/core/TensorMethods.cpp
)

rm -rf obj && mkdir -p obj
printf '%s\n' "${SRCS[@]}" | xargs -P"$(nproc)" -I{} \
  bash -c 'o="obj/$(echo "$1" | tr "/" _).o"; g++ "${@:2}" -c "$1" -o "$o"' \
       _ {} "${FLAGS[@]}" "${INC[@]}"

rm -f libaten_core.a
ar rcs libaten_core.a obj/*.o
echo "built libaten_core.a with $(ls obj/*.o | wc -l) objects"
