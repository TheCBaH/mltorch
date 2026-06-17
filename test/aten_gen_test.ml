(* Generator output snapshots. Parse a schema string and emit the extern "C"
   shim + ctypes binding (or the skip reason). *)

let gen ?(style = `Function) s =
  match Func_schema.parse s with
  | Error e -> Printf.printf "PARSE ERROR: %s\n" e
  | Ok op -> (
      match Aten_gen.Gen.generate ~style op with
      | Skipped r -> Printf.printf "SKIPPED: %s\n" r
      | Generated g ->
          Printf.printf "%s\n---\n%s\n" g.Aten_gen.Gen.c_source
            g.Aten_gen.Gen.ctypes_line)

let%expect_test "add.Tensor" =
  gen "add.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor";
  [%expect
    {|
    atc_tensor atg_add_Tensor(atc_tensor self, atc_tensor other, const struct atc_scalar* alpha) {
      return atc_wrap(at::add(*atc_to_ptr(self), *atc_to_ptr(other), atc_to_c10_scalar(alpha)));
    }
    ---
    let add_Tensor = foreign "atg_add_Tensor" (atc_tensor @-> atc_tensor @-> scalar @-> returning atc_tensor) |}]

let%expect_test "mul.Tensor" =
  gen "mul.Tensor(Tensor self, Tensor other) -> Tensor";
  [%expect
    {|
    atc_tensor atg_mul_Tensor(atc_tensor self, atc_tensor other) {
      return atc_wrap(at::mul(*atc_to_ptr(self), *atc_to_ptr(other)));
    }
    ---
    let mul_Tensor = foreign "atg_mul_Tensor" (atc_tensor @-> atc_tensor @-> returning atc_tensor) |}]

let%expect_test "reshape (SymInt[])" =
  gen "reshape(Tensor(a) self, SymInt[] shape) -> Tensor(a)";
  [%expect
    {|
    atc_tensor atg_reshape(atc_tensor self, int64_t* shape_data, int shape_len) {
      return atc_wrap(at::reshape(*atc_to_ptr(self), at::IntArrayRef(shape_data, shape_len)));
    }
    ---
    let reshape = foreign "atg_reshape" (atc_tensor @-> ptr int64_t @-> int @-> returning atc_tensor) |}]

let%expect_test "narrow (SymInt scalar)" =
  gen
    "narrow(Tensor(a) self, int dim, SymInt start, SymInt length) -> Tensor(a)";
  [%expect
    {|
    atc_tensor atg_narrow(atc_tensor self, int64_t dim, int64_t start, int64_t length) {
      return atc_wrap(at::narrow(*atc_to_ptr(self), dim, start, length));
    }
    ---
    let narrow = foreign "atg_narrow" (atc_tensor @-> int64_t @-> int64_t @-> int64_t @-> returning atc_tensor) |}]

let%expect_test "avg_pool2d (int? divisor_override)" =
  gen
    "avg_pool2d(Tensor self, int[2] kernel_size, int[2] stride=[], int[2] \
     padding=0, bool ceil_mode=False, bool count_include_pad=True, int? \
     divisor_override=None) -> Tensor";
  [%expect
    {|
    atc_tensor atg_avg_pool2d(atc_tensor self, int64_t* kernel_size_data, int kernel_size_len, int64_t* stride_data, int stride_len, int64_t* padding_data, int padding_len, int ceil_mode, int count_include_pad, int64_t* divisor_override) {
      return atc_wrap(at::avg_pool2d(*atc_to_ptr(self), at::IntArrayRef(kernel_size_data, kernel_size_len), at::IntArrayRef(stride_data, stride_len), at::IntArrayRef(padding_data, padding_len), (bool)ceil_mode, (bool)count_include_pad, divisor_override ? std::make_optional(*divisor_override) : std::nullopt));
    }
    ---
    let avg_pool2d = foreign "atg_avg_pool2d" (atc_tensor @-> ptr int64_t @-> int @-> ptr int64_t @-> int @-> ptr int64_t @-> int @-> bool @-> bool @-> ptr int64_t @-> returning atc_tensor) |}]

let%expect_test "mean.dim (int[1]? + ScalarType?)" =
  gen
    "mean.dim(Tensor self, int[1]? dim, bool keepdim=False, *, ScalarType? \
     dtype=None) -> Tensor";
  [%expect
    {|
    atc_tensor atg_mean_dim(atc_tensor self, int64_t* dim_data, int dim_len, int keepdim, int dtype) {
      return atc_wrap(at::mean(*atc_to_ptr(self), dim_data ? at::OptionalIntArrayRef(at::IntArrayRef(dim_data, dim_len)) : at::OptionalIntArrayRef(std::nullopt), (bool)keepdim, dtype < 0 ? std::nullopt : std::make_optional(static_cast<at::ScalarType>(dtype))));
    }
    ---
    let mean_dim = foreign "atg_mean_dim" (atc_tensor @-> ptr int64_t @-> int @-> bool @-> scalar_type_opt @-> returning atc_tensor) |}]

let%expect_test "to.dtype_layout (Layout? + ScalarType?)" =
  gen
    "to.dtype_layout(Tensor(a) self, *, ScalarType? dtype=None, Layout? \
     layout=None) -> Tensor(a)";
  [%expect
    {|
    atc_tensor atg_to_dtype_layout(atc_tensor self, int dtype, int layout) {
      return atc_wrap(at::to(*atc_to_ptr(self), dtype < 0 ? std::nullopt : std::make_optional(static_cast<at::ScalarType>(dtype)), layout < 0 ? std::nullopt : std::make_optional(static_cast<at::Layout>(layout))));
    }
    ---
    let to_dtype_layout = foreign "atg_to_dtype_layout" (atc_tensor @-> scalar_type_opt @-> layout_opt @-> returning atc_tensor) |}]

let%expect_test "to.dtype (method-style, ScalarType + MemoryFormat?)" =
  gen ~style:`Method
    "to.dtype(Tensor(a) self, ScalarType dtype, bool non_blocking=False, bool \
     copy=False, MemoryFormat? memory_format=None) -> Tensor(a)";
  [%expect
    {|
    atc_tensor atg_to_dtype(atc_tensor self, int dtype, int non_blocking, int copy, int memory_format) {
      return atc_wrap(atc_to_ptr(self)->to(static_cast<at::ScalarType>(dtype), (bool)non_blocking, (bool)copy, memory_format < 0 ? std::nullopt : std::make_optional(static_cast<at::MemoryFormat>(memory_format))));
    }
    ---
    let to_dtype = foreign "atg_to_dtype" (atc_tensor @-> scalar_type @-> bool @-> bool @-> memory_format_opt @-> returning atc_tensor) |}]

let%expect_test "contiguous (method-style, MemoryFormat arg)" =
  gen ~style:`Method
    "contiguous(Tensor(a) self, *, MemoryFormat \
     memory_format=contiguous_format) -> Tensor(a)";
  [%expect
    {|
    atc_tensor atg_contiguous(atc_tensor self, int memory_format) {
      return atc_wrap(atc_to_ptr(self)->contiguous(static_cast<at::MemoryFormat>(memory_format)));
    }
    ---
    let contiguous = foreign "atg_contiguous" (atc_tensor @-> memory_format @-> returning atc_tensor) |}]

let%expect_test "clone (MemoryFormat? memory_format)" =
  gen "clone(Tensor self, *, MemoryFormat? memory_format=None) -> Tensor";
  [%expect
    {|
    atc_tensor atg_clone(atc_tensor self, int memory_format) {
      return atc_wrap(at::clone(*atc_to_ptr(self), memory_format < 0 ? std::nullopt : std::make_optional(static_cast<at::MemoryFormat>(memory_format))));
    }
    ---
    let clone = foreign "atg_clone" (atc_tensor @-> memory_format_opt @-> returning atc_tensor) |}]

let%expect_test "softmax.int (ScalarType? dtype)" =
  gen "softmax.int(Tensor self, int dim, ScalarType? dtype=None) -> Tensor";
  [%expect
    {|
    atc_tensor atg_softmax_int(atc_tensor self, int64_t dim, int dtype) {
      return atc_wrap(at::softmax(*atc_to_ptr(self), dim, dtype < 0 ? std::nullopt : std::make_optional(static_cast<at::ScalarType>(dtype))));
    }
    ---
    let softmax_int = foreign "atg_softmax_int" (atc_tensor @-> int64_t @-> scalar_type_opt @-> returning atc_tensor) |}]

let%expect_test "bool? (negative-sentinel view)" =
  gen "f(Tensor self, bool? flag=None) -> Tensor";
  [%expect
    {|
    atc_tensor atg_f(atc_tensor self, int flag) {
      return atc_wrap(at::f(*atc_to_ptr(self), flag < 0 ? std::nullopt : std::make_optional((bool)flag)));
    }
    ---
    let f = foreign "atg_f" (atc_tensor @-> bool_opt @-> returning atc_tensor) |}]

let%expect_test "Device? (struct pointer + view)" =
  gen "f(Tensor self, Device? device=None) -> Tensor";
  [%expect
    {|
    atc_tensor atg_f(atc_tensor self, const struct atc_device* device) {
      return atc_wrap(at::f(*atc_to_ptr(self), atc_to_device_opt(device)));
    }
    ---
    let f = foreign "atg_f" (atc_tensor @-> device_opt @-> returning atc_tensor) |}]

let%expect_test "tuple return (outputs 1.. via out-params)" =
  gen
    "max_pool2d_with_indices(Tensor self, int[2] kernel_size) -> (Tensor, \
     Tensor)";
  [%expect
    {|
    int atg_max_pool2d_with_indices(atc_tensor self, int64_t* kernel_size_data, int kernel_size_len, struct atc_tensors2* out) {
      ATC_TRY(-1, {
        auto __r = at::max_pool2d_with_indices(*atc_to_ptr(self), at::IntArrayRef(kernel_size_data, kernel_size_len));
        out->v0 = atc_wrap(std::get<0>(__r));
        out->v1 = atc_wrap(std::get<1>(__r));
        return 0;
      })
    }
    ---
    let max_pool2d_with_indices = foreign "atg_max_pool2d_with_indices" (atc_tensor @-> ptr int64_t @-> int @-> ptr tensors2_struct @-> returning int) |}]

let%expect_test "skipped: non-Tensor tuple element" =
  gen "frexp.Tensor(Tensor self) -> (Tensor mantissa, Scalar exponent)";
  [%expect {| SKIPPED: unsupported return shape |}]

let%expect_test "skipped: out= variant" =
  gen
    "add.out(Tensor self, Tensor other, *, Scalar alpha=1, Tensor(a!) out) -> \
     Tensor(a!)";
  [%expect {| SKIPPED: out= variant |}]

let%expect_test "skipped: unsupported arg (Dimname)" =
  gen "squeeze.dimname(Tensor(a) self, Dimname dim) -> Tensor(a)";
  [%expect {| SKIPPED: unsupported arg type: Dimname |}]
