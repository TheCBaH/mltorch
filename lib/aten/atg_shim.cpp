#include "atg_shim.h"

#include <algorithm>
#include <sstream>
#include <string>

#include <ATen/core/Formatting.h>

#include <ATen/Functions.h>
#include <c10/core/CPUAllocator.h>
#include <c10/core/DefaultDtype.h>
#include <c10/core/ScalarType.h>
#include <c10/core/StorageImpl.h>
#include <c10/core/TensorImpl.h>

namespace {

at::Tensor make_cpu_tensor(const int64_t* sizes, size_t ndim,
                           c10::ScalarType dtype) {
  c10::IntArrayRef shape(sizes, ndim);
  int64_t numel = 1;
  for (size_t i = 0; i < ndim; ++i) numel *= sizes[i];
  auto storage = c10::make_intrusive<c10::StorageImpl>(
      c10::StorageImpl::use_byte_size_t(),
      static_cast<size_t>(numel) * c10::elementSize(dtype),
      c10::GetCPUAllocator(), /*resizable=*/true);
  auto t = at::detail::make_tensor<c10::TensorImpl>(
      std::move(storage), c10::DispatchKeySet(c10::DispatchKey::CPU),
      c10::scalarTypeToTypeMeta(dtype));
  t.unsafeGetTensorImpl()->set_sizes_contiguous(shape);
  return t;
}

}  // namespace

namespace atc_detail {
std::atomic<long> live{0};

/* Thread-local so each OCaml-calling thread reads its own last error / string. */
thread_local std::string last_error;
thread_local std::string last_string;
void set_error(const char* msg) { last_error = msg ? msg : "unknown error"; }
}  // namespace atc_detail

extern "C" {

int8_t atc_default_dtype(void) {
  return static_cast<int8_t>(c10::get_default_dtype_as_scalartype());
}

size_t atc_dtype_elem_size(int8_t scalar_type) {
  return c10::elementSize(static_cast<c10::ScalarType>(scalar_type));
}

atc_tensor atc_new(const int64_t* sizes, size_t ndim, atc_scalar_type dtype) {
  ATC_TRY(nullptr, {
    return atc_wrap(
        make_cpu_tensor(sizes, ndim, static_cast<c10::ScalarType>(dtype)));
  })
}

const char* atc_last_error(void) {
  return atc_detail::last_error.empty() ? nullptr
                                        : atc_detail::last_error.c_str();
}

const char* atc_to_string(atc_tensor t) {
  ATC_TRY(nullptr, {
    std::ostringstream oss;
    oss << *atc_to_ptr(t);
    atc_detail::last_string = oss.str();
    return atc_detail::last_string.c_str();
  })
}

int atc_allclose(atc_tensor a, atc_tensor b, double rtol, double atol,
                 int equal_nan) {
  ATC_TRY(-1, {
    return at::allclose(*atc_to_ptr(a), *atc_to_ptr(b), rtol, atol,
                        (bool)equal_nan)
               ? 1
               : 0;
  })
}

int atc_equal(atc_tensor a, atc_tensor b) {
  ATC_TRY(-1, {
    return at::equal(*atc_to_ptr(a), *atc_to_ptr(b)) ? 1 : 0;
  })
}

void atc_free(atc_tensor t) {
  if (!t) return;
  --atc_detail::live;
  delete atc_to_ptr(t);
}

int64_t atc_live_count(void) { return atc_detail::live.load(); }

int64_t atc_numel(atc_tensor t) { return atc_to_ptr(t)->numel(); }

size_t atc_dim(atc_tensor t) { return static_cast<size_t>(atc_to_ptr(t)->dim()); }

void atc_sizes(atc_tensor t, int64_t* out) {
  auto sizes = atc_to_ptr(t)->sizes();
  std::copy(sizes.begin(), sizes.end(), out);
}

void atc_strides(atc_tensor t, int64_t* out) {
  auto strides = atc_to_ptr(t)->strides();
  std::copy(strides.begin(), strides.end(), out);
}

atc_scalar_type atc_dtype(atc_tensor t) {
  return static_cast<atc_scalar_type>(atc_to_ptr(t)->scalar_type());
}

int64_t atc_element_size(atc_tensor t) { return atc_to_ptr(t)->element_size(); }

int atc_is_contiguous(atc_tensor t) {
  return atc_to_ptr(t)->is_contiguous() ? 1 : 0;
}

int atc_defined(atc_tensor t) { return atc_to_ptr(t)->defined() ? 1 : 0; }

int atc_is_cpu(atc_tensor t) { return atc_to_ptr(t)->is_cpu() ? 1 : 0; }

void* atc_data_ptr(atc_tensor t, atc_scalar_type dtype) {
  auto* a = atc_to_ptr(t);
  if (a->scalar_type() != static_cast<c10::ScalarType>(dtype)) return nullptr;
  return a->unsafeGetTensorImpl()->storage().mutable_data();
}

int atc_item_double(atc_tensor t, double* out) {
  ATC_TRY(0, {
    *out = atc_to_ptr(t)->item().toDouble();
    return 1;
  })
}

int atc_item_int64(atc_tensor t, int64_t* out) {
  ATC_TRY(0, {
    *out = atc_to_ptr(t)->item().toLong();
    return 1;
  })
}

}  // extern "C"
