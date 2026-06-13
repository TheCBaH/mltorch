#!/usr/bin/env bash
# Compile the ATen "core" subset PLUS the real CPU add/mul kernels (static
# dispatch) into a static archive libaten_core.a.
#
# Layers:
#   1. c10 core+util+mobile (recursive) + ATen/core    — metadata/type system
#   2. generated static-dispatch glue: Operators_* (::call -> at::cpu::X),
#      RegisterCPU_* / RegisterComposite* (at::cpu:: wrappers + structured
#      kernels), UfuncCPU_add / UfuncCPUKernel_add (the add ufunc + its stub)
#   3. native runtime the kernels need: TensorIterator, Parallel*, factories
#      (empty*/empty_like), shape (as_strided/select), conversions, BinaryOps
#      (+ BinaryOpsKernel/FillKernel = mul_stub/fill_stub) — the bounded closure
#   4. atg_stubs.cpp — throwing leaves for the cold-path ops TensorIterator/structured
#      kernels reference straight-line but never run for dense float (Option A)
#   5. cpuinfo (built here via cmake) — ParallelNative needs it; bundled in
#
# Everything is compiled with -ffunction-sections -fdata-sections so the FINAL
# link (OCaml exe / dll) can --gc-sections away the unreached wrappers; the
# archive itself stays the bounded ~190-object set we actually compile.
# CPU kernels (cpu/*Kernel + UfuncCPUKernel + the native/ files that pull vec.h)
# are built scalar via -DCPU_CAPABILITY=DEFAULT -DCPU_CAPABILITY_DEFAULT.
#
#   $1 = path to the pytorch source root (the git submodule)
# Prereqs in cwd (produced by sibling rules): ./gen, ./inc
# Third-party submodules (must be checked out):
#   git submodule update --init third_party/fmt third_party/cpuinfo
# Output: ./libaten_core.a + ./dllaten_core.so
set -euo pipefail
PT=$(cd "$1" && pwd)

# Use ccache as a compiler launcher when it is on PATH. This is what makes the
# 194-object ATen compile cheap to rebuild: in CI the CCACHE_DIR is persisted
# through the GitHub Actions cache (see .github/workflows/build.yml), so an
# unchanged PyTorch checkout recompiles from cache. Harmless no-op when ccache
# is not installed: $CCACHE expands to nothing and clang++ runs directly.
CCACHE="$(command -v ccache || true)"
export CCACHE

for need in "$PT/third_party/fmt/include/fmt/core.h" \
            "$PT/third_party/cpuinfo/include/cpuinfo.h" \
            "$PT/third_party/cpuinfo/CMakeLists.txt"; do
  if [ ! -f "$need" ]; then
    echo "ERROR: missing $need" >&2
    echo "Run: git submodule update --init third_party/fmt third_party/cpuinfo" >&2
    exit 1
  fi
done

INC=(-I"$PT" -I"$PT/aten/src" -Igen -Iinc
     -I"$PT/third_party/fmt/include" -I"$PT/third_party/cpuinfo/include")
FLAGS=(-std=c++17 -Os -fPIC -DFMT_HEADER_ONLY=1)
# Per-function/data sections so the FINAL link can --gc-sections away the unused
# op wrappers. Applied ONLY to the generated glue + native closure: the c10 /
# ATen-core objects are monolithic and always fully reached, so splitting them
# gains nothing and makes section-GC over the whole archive pathologically slow.
SECT=(-ffunction-sections -fdata-sections)
# Extra flags for the SIMD-capable kernels: scalar (DEFAULT) capability only.
CAP=(-DCPU_CAPABILITY=DEFAULT -DCPU_CAPABILITY_DEFAULT)

# --- source lists -----------------------------------------------------------

# layer 1: full c10 (core incl. core/impl, util, mobile) + ATen/core + the
# generated op-name list. Monolithic, always needed -> NO section splitting.
mapfile -t SRCS_CORE < <(
  find "$PT/c10/core" "$PT/c10/util" "$PT/c10/mobile" -name '*.cpp' ! -name '*test*'
  find "$PT/aten/src/ATen/core" -name '*.cpp' ! -name '*test*'
  echo gen/ATen/core/ATenOpList.cpp
  echo "$PT/aten/src/ATen/TensorNames.cpp"
  echo "$PT/aten/src/ATen/AccumulateType.cpp"
)

# layer 2+3: static-dispatch glue + native runtime that does NOT pull vec.h.
# Section-split (SECT) so --gc-sections trims the unused wrappers. TensorMethods
# is here, not in core: under static dispatch every Tensor::method() routes
# directly to at::native::X, so its ~2600 methods must be GC-trimmable.
mapfile -t SRCS_GLUE < <(
  echo gen/ATen/core/TensorMethods.cpp
  for f in Operators_0 Operators_1 Operators_2 Operators_3 Operators_4 \
           RegisterCPU_0 RegisterCPU_1 RegisterCPU_2 RegisterCPU_3 \
           RegisterCompositeExplicitAutograd_0 \
           RegisterCompositeExplicitAutogradNonFunctional_0 \
           RegisterCompositeImplicitAutograd_0 \
           UfuncCPU_add; do
    echo "gen/ATen/$f.cpp"
  done
  for f in Context EmptyTensor ExpandUtils MemoryOverlap NamedTensorUtils \
           ParallelCommon ParallelNative ParallelThreadPoolNative \
           SequenceNumber record_function ScalarOps TensorUtils \
           TensorIterator DeviceAccelerator; do
    echo "$PT/aten/src/ATen/$f.cpp"
  done
  # avg_pool2d structured meta+impl (TORCH_META_FUNC/TORCH_IMPL_FUNC) and the
  # avg_pool2d_kernel DispatchStub declaration; no vec.h -> compiled here.
  echo "$PT/aten/src/ATen/native/AveragePool2d.cpp"
  # relu/relu_ (-> at::clamp_min/_), and the clamp_min structured meta+impl
  # lives in TensorCompare.cpp (already in the CAP list below). Activation.cpp
  # has no vec.h; its other activation stubs --gc-section away unreached.
  echo "$PT/aten/src/ATen/native/Activation.cpp"
  # max_pool2d: composite (Pooling.cpp) -> max_pool2d_with_indices structured
  # meta+impl (DilatedMaxPool2d.cpp); the MaxPoolKernel is in the CAP list.
  echo "$PT/aten/src/ATen/native/Pooling.cpp"
  echo "$PT/aten/src/ATen/native/DilatedMaxPool2d.cpp"
  # adaptive_avg_pool2d: composite (AdaptiveAveragePooling.cpp). The (1,1) case
  # used by resnet18 reduces via at::mean (-> ReduceOps below); other sizes use
  # the AdaptiveAvgPoolKernel in the CAP list.
  echo "$PT/aten/src/ATen/native/AdaptiveAveragePooling.cpp"
  # real copy_ (reductions / type-cast / contiguous paths call it); replaces
  # the former throwing stub. Kernel is cpu/CopyKernel.cpp in the CAP list.
  echo "$PT/aten/src/ATen/native/Copy.cpp"
  # linear -> at::addmm/mm/matmul (LinearAlgebra.cpp) -> cpublas::gemm host
  # dispatcher (CPUBlas.cpp) -> gemm_stub kernel (cpu/BlasKernel.cpp, CAP).
  # No external BLAS: BlasKernel ships a reference gemm fallback.
  echo "$PT/aten/src/ATen/native/Linear.cpp"
  echo "$PT/aten/src/ATen/native/LinearAlgebra.cpp"
  echo "$PT/aten/src/ATen/native/CPUBlas.cpp"
  # matmul/addmm pull in: mv/dot (Blas.cpp) and conj_physical/resolve_conj
  # (UnaryOps.cpp); their unreached siblings --gc-section away.
  echo "$PT/aten/src/ATen/native/Blas.cpp"
  echo "$PT/aten/src/ATen/native/UnaryOps.cpp"
  # gemv<T> / dot_impl<T> / blas_impl::fp16_gemv* (used by mv/dot); this is the
  # native/ BlasKernel (capability-agnostic), distinct from cpu/BlasKernel.
  echo "$PT/aten/src/ATen/native/BlasKernel.cpp"
  # batch_norm: composite -> _batch_norm_impl_index -> native_batch_norm
  # (Normalization.cpp); CPU kernel is cpu/batch_norm_kernel.cpp (CAP).
  echo "$PT/aten/src/ATen/native/Normalization.cpp"
  # conv2d -> convolution -> _convolution -> slow_conv2d (ConvolutionMM2d.cpp),
  # which im2col's (Unfold2d.cpp) then gemm's. cpu/Unfold2d kernel is in CAP.
  echo "$PT/aten/src/ATen/native/Convolution.cpp"
  echo "$PT/aten/src/ATen/native/ConvolutionMM2d.cpp"
  echo "$PT/aten/src/ATen/native/Unfold2d.cpp"
  # dropout/dropout_ (composite; identity at inference, train=false).
  echo "$PT/aten/src/ATen/native/Dropout.cpp"
  echo "$PT/aten/src/ATen/native/ReduceOps.cpp"
  echo "$PT/aten/src/ATen/native/ReduceAllOps.cpp"
  # TensorIteratorBase::parallel_reduce (separate from TensorIterator.cpp).
  echo "$PT/aten/src/ATen/native/TensorIteratorReduce.cpp"
  echo "$PT/aten/src/ATen/cpu/FlushDenormal.cpp"
  echo "$PT/aten/src/ATen/quantized/QTensorImpl.cpp"
  echo "$PT/aten/src/ATen/native/sparse/SparseTensor.cpp"
  find "$PT/aten/src/ATen/detail" -name '*HooksInterface.cpp'
  echo atg_stubs.cpp
  echo atg_shim.cpp
  echo atg_ops.cpp
)

# layer 2+3 WITH CPU_CAPABILITY (the add/mul ufunc + DispatchStub kernels, and
# the native/ files that include vec.h transitively). Also section-split.
mapfile -t SRCS_CAP < <(
  echo gen/ATen/UfuncCPUKernel_add.cpp
  for f in Fill Resize TypeProperties DispatchStub Scalar TensorCompare \
           TensorConversions TensorFactories TensorProperties TensorShape \
           BinaryOps; do
    echo "$PT/aten/src/ATen/native/$f.cpp"
  done
  echo "$PT/aten/src/ATen/native/cpu/BinaryOpsKernel.cpp"
  echo "$PT/aten/src/ATen/native/cpu/FillKernel.cpp"
  # clamp_min_scalar_stub (used by relu via clamp_min in TensorCompare.cpp).
  echo "$PT/aten/src/ATen/native/cpu/TensorCompareKernel.cpp"
  # activation kernels (silu_stub for silu_; siblings --gc-section away).
  echo "$PT/aten/src/ATen/native/cpu/Activation.cpp"
  # max_pool2d_kernel (the vectorized pooling kernel).
  echo "$PT/aten/src/ATen/native/cpu/MaxPoolKernel.cpp"
  # adaptive_avg_pool2d_kernel + the reduction kernels behind at::mean/sum
  # (mean_stub/sum_stub), needed by adaptive_avg_pool2d's global-average path.
  echo "$PT/aten/src/ATen/native/cpu/AdaptiveAvgPoolKernel.cpp"
  echo "$PT/aten/src/ATen/native/cpu/ReduceOpsKernel.cpp"
  echo "$PT/aten/src/ATen/native/cpu/SumKernel.cpp"
  echo "$PT/aten/src/ATen/native/cpu/CopyKernel.cpp"
  # conj_kernel / neg_kernel (CopyKernel uses them for conjugate/negative bits).
  echo "$PT/aten/src/ATen/native/cpu/UnaryOpsKernel.cpp"
  # cpublas_gemm_impl (the reference gemm registered to gemm_stub) for linear,
  # plus the half/bfloat16 gemv fast-path helpers BlasKernel references.
  echo "$PT/aten/src/ATen/native/cpu/BlasKernel.cpp"
  echo "$PT/aten/src/ATen/native/cpu/ReducedPrecisionFloatGemvFastPathKernel.cpp"
  # batch_norm_cpu_stub (the vectorized batch-norm kernel).
  echo "$PT/aten/src/ATen/native/cpu/batch_norm_kernel.cpp"
  # im2col/col2im kernel (unfolded_copy/unfolded_acc) for slow_conv2d.
  echo "$PT/aten/src/ATen/native/cpu/Unfold2d.cpp"
  # DispatchStub ::DEFAULT members Convolution.cpp references: cat_serial_stub
  # (grouped-conv at::cat) and the depthwise-3x3-winograd stub.
  echo "$PT/aten/src/ATen/native/cpu/CatKernel.cpp"
  echo "$PT/aten/src/ATen/native/cpu/DepthwiseConvKernel.cpp"
  # REGISTER_DISPATCH(avg_pool2d_kernel, ...) — the vectorized pooling kernel.
  echo "$PT/aten/src/ATen/native/cpu/AvgPoolKernel.cpp"
)

# --- compile ----------------------------------------------------------------
rm -rf obj && mkdir -p obj
compile_list() {  # extra flags as args; source paths via stdin
  xargs -P"$(nproc)" -I{} \
    bash -c 'o="obj/$(echo "$1" | tr "/" _).o"; $CCACHE clang++ "${@:2}" -c "$1" -o "$o"' \
         _ {} "$@"
}
printf '%s\n' "${SRCS_CORE[@]}" | compile_list "${FLAGS[@]}" "${INC[@]}"
printf '%s\n' "${SRCS_GLUE[@]}" | compile_list "${FLAGS[@]}" "${SECT[@]}" "${INC[@]}"
printf '%s\n' "${SRCS_CAP[@]}"  | compile_list "${FLAGS[@]}" "${SECT[@]}" "${CAP[@]}" "${INC[@]}"

# --- cpuinfo (static, PIC) --------------------------------------------------
CMAKE_LAUNCHER=()
if [ -n "$CCACHE" ]; then
  CMAKE_LAUNCHER=(-DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache)
fi
cmake -S "$PT/third_party/cpuinfo" -B cpuinfo-build "${CMAKE_LAUNCHER[@]}" \
  -DCPUINFO_LIBRARY_TYPE=static -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
  -DCPUINFO_BUILD_UNIT_TESTS=OFF -DCPUINFO_BUILD_MOCK_TESTS=OFF \
  -DCPUINFO_BUILD_BENCHMARKS=OFF -DCPUINFO_BUILD_TOOLS=OFF \
  -DCMAKE_BUILD_TYPE=Release >/dev/null
cmake --build cpuinfo-build -j"$(nproc)" --target cpuinfo >/dev/null

# --- archive + shared lib ---------------------------------------------------
rm -f libaten_core.a dllaten_core.so
ar rcs libaten_core.a obj/*.o
# Fold cpuinfo in via an MRI script: ADDLIB copies ALL members, preserving the
# duplicate object names cpuinfo ships (src/cache.c and src/arm/cache.c both
# compile to cache.c.o) that a filesystem `ar x` would clobber.
ar -M <<EOF
OPEN libaten_core.a
ADDLIB cpuinfo-build/libcpuinfo.a
SAVE
END
EOF
# Shared variant for ctypes' bytecode/toplevel path (foreign_archives needs a
# dll<name>.so alongside lib<name>.a). --whole-archive so every symbol is
# present; --gc-sections (via the fast lld linker) trims the unreached op
# wrappers — GNU ld is pathologically slow at this over the generated objects.
clang++ -shared -o dllaten_core.so -fuse-ld=lld \
  -Wl,--whole-archive libaten_core.a -Wl,--no-whole-archive \
  -Wl,--gc-sections -lpthread
echo "built libaten_core.a + dllaten_core.so with $(ls obj/*.o | wc -l) objects (+ cpuinfo)"
