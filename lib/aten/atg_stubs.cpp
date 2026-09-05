// Throwing stubs for ATen cold-path ops that TensorIterator / structured
// kernels reference UNCONDITIONALLY (straight-line, in error / type-check /
// quantized / sparse branches) but never execute for dense CPU float add/mul.
// Stubbing the leaf entry points lets --gc-sections drop the sparse /
// quantized / reduction / indexing machinery (and their SIMD DispatchStubs)
// behind them, keeping the static-dispatch archive bounded (Option A).
#include <ATen/core/Tensor.h>
#include <ATen/NativeFunctions.h>
#include <ATen/NativeMetaFunctions.h>
#include <ATen/core/operator_name.h>
#include <ATen/quantized/QTensorImpl.h>
#include <ATen/quantized/Quantizer.h>
#include <c10/util/Exception.h>

#define MINSTUB(name) \
  TORCH_CHECK(false, "ATen op '" name "' is not built in the minimal static-dispatch subset")

namespace at {
namespace native {

// --- layer 1: TensorIterator type-check / error-helper leaves ---
// (item / to_dense / coalesce / values_default / indices_default are provided
// by the real Scalar.cpp / TensorConversions.cpp / SparseTensor.cpp.)
bool is_nonzero(const at::Tensor&) { MINSTUB("is_nonzero"); }
at::Tensor dequantize_cpu_or_cuda(const at::Tensor&) { MINSTUB("dequantize"); }
at::Tensor col_indices_default(const at::Tensor&) { MINSTUB("col_indices"); }
// dropout's train=true mask path (Dropout.cpp); inference (train=false) is
// identity and never reaches it.
at::Tensor& bernoulli_(at::Tensor& self, double, std::optional<at::Generator>) { MINSTUB("bernoulli_"); return self; }
at::Tensor crow_indices_default(const at::Tensor&) { MINSTUB("crow_indices"); }
at::Tensor ccol_indices_default(const at::Tensor&) { MINSTUB("ccol_indices"); }
at::Tensor row_indices_default(const at::Tensor&) { MINSTUB("row_indices"); }

// --- layer 2: leaves pulled in by TensorFactories / TensorShape /
//     TensorConversions (output allocation + as_strided / select / to paths) ---
// nonzero_cpu / index_select_cpu_ are now REAL (native/TensorAdvancedIndexing.cpp,
// added for op8/scaled_dot_product_attention's masked_fill closure); they were
// stubbed only because nothing else pulled that file in.
at::Tensor sparse_compressed_tensor_with_dims(int64_t, int64_t, at::IntArrayRef, at::IntArrayRef, at::ScalarType, std::optional<at::ScalarType>, std::optional<at::Layout>, std::optional<at::Device>, std::optional<bool>) { MINSTUB("sparse_compressed_tensor_with_dims"); }
at::Tensor _sparse_compressed_tensor_unsafe_symint(const at::Tensor&, const at::Tensor&, const at::Tensor&, c10::SymIntArrayRef, std::optional<at::ScalarType>, std::optional<at::Layout>, std::optional<at::Device>, std::optional<bool>) { MINSTUB("_sparse_compressed_tensor_unsafe_symint"); }

// Copy.cpp dispatches to these for non-CPU source tensors (quantized / vulkan /
// metal); the dense-CPU copy path never reaches them.
at::Tensor& quantized_copy_from_float_(at::Tensor& self, const at::Tensor&) { MINSTUB("quantized_copy_from_float_"); return self; }

// matmul/linear mkldnn fast path (LinearAlgebra.cpp); never taken for dense CPU.
bool use_mkldnn_matmul(const at::Tensor&, const at::Tensor&, const at::Tensor&) { return false; }
at::Tensor mkldnn_matmul(const at::Tensor&, const at::Tensor&, const at::Tensor&, float, float) { MINSTUB("mkldnn_matmul"); }

// Convolution.cpp's backend switch references every conv variant straight-line.
// The 2D forward / dilated / transpose leaves are built; [slow_conv3d] is now
// REAL too (native/ConvolutionMM3d.cpp, added for aten.conv3d.default's own
// non-dilated forward path) -- only the transposed 3D leaf, which this engine
// never reaches, remains stubbed.
at::Tensor slow_conv_transpose3d_cpu(const at::Tensor&, const at::Tensor&, at::IntArrayRef, const std::optional<at::Tensor>&, at::IntArrayRef, at::IntArrayRef, at::IntArrayRef, at::IntArrayRef) { MINSTUB("slow_conv_transpose3d_cpu"); }
// constant_pad_nd is now REAL (native/PadNd.cpp, added for aten.pad.default);
// it was stubbed only because Convolution.cpp references it for `same` padding.

// ReflectionPad.cpp / ReplicationPadding.cpp carry quantized variants that call
// at::_empty_affine_quantized straight-line; under static dispatch that resolves
// to this factory. The dense-CPU float paths this engine takes never reach it,
// and PyTorch's own definition is itself only a better error message.
at::Tensor empty_affine_quantized_other_backends_stub(at::IntArrayRef, std::optional<at::ScalarType>, std::optional<at::Layout>, std::optional<at::Device>, std::optional<bool>, double, int64_t, std::optional<at::MemoryFormat>) { MINSTUB("_empty_affine_quantized"); }

// xnnpack mobile conv fast path (Convolution.cpp); never built/taken here.
namespace xnnpack {
at::Tensor convolution2d(const at::Tensor&, const at::Tensor&, const at::Tensor&, at::IntArrayRef, at::IntArrayRef, at::IntArrayRef, int64_t) { MINSTUB("xnnpack::convolution2d"); }
}  // namespace xnnpack

}  // namespace native

namespace vulkan {
at::Tensor& vulkan_copy_(at::Tensor& self, const at::Tensor&) { MINSTUB("vulkan_copy_"); return self; }
}  // namespace vulkan

namespace metal {
at::Tensor& metal_copy_(at::Tensor& self, const at::Tensor&) { MINSTUB("metal_copy_"); return self; }
}  // namespace metal

// --- quantizer leaves (TensorShape's is_quantized() branch in as_strided) ---
// The QTensorImpl ctor + vtable come from the real (tiny) QTensorImpl.cpp; only
// the affine-quantizer factory leaves are stubbed (they never cascade).
QTensorImpl* get_qtensorimpl(const TensorBase&) { MINSTUB("get_qtensorimpl"); }
QuantizerPtr make_per_tensor_affine_quantizer(double, int64_t, ScalarType) { MINSTUB("make_per_tensor_affine_quantizer"); }
QuantizerPtr make_per_channel_affine_quantizer(const Tensor&, const Tensor&, int64_t, ScalarType) { MINSTUB("make_per_channel_affine_quantizer"); }
void set_quantizer_(const Tensor&, ConstQuantizerPtr) { MINSTUB("set_quantizer_"); }
QuantizerPtr TensorBase::quantizer() const { MINSTUB("quantizer"); }

}  // namespace at

// --- TORCH_LIBRARY schema parser (cold path in ATen/core/library.cpp; unused
//     under --skip-dispatcher-op-registration). core is not section-split, so
//     this straight-line ref must resolve. ---
namespace torch {
namespace jit {
c10::OperatorName parseName(const std::string&) { MINSTUB("parseName"); }
}  // namespace jit
}  // namespace torch
