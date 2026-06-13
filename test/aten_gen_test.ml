(* Generator output snapshots. Parse a schema string and emit the extern "C"
   shim + ctypes binding (or the skip reason). *)

let gen s =
  match Func_schema.parse s with
  | Error e -> Printf.printf "PARSE ERROR: %s\n" e
  | Ok op -> (
      match Aten_gen.Gen.generate op with
      | Skipped r -> Printf.printf "SKIPPED: %s\n" r
      | Generated g ->
          Printf.printf "%s\n---\n%s\n" g.Aten_gen.Gen.c_source
            g.Aten_gen.Gen.ctypes_line)

let%expect_test "add.Tensor" =
  gen "add.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor";
  [%expect
    {|
    atc_tensor atg_add_Tensor(atc_tensor self, atc_tensor other, double alpha) {
      return atc_wrap(at::add(*atc_to_ptr(self), *atc_to_ptr(other), c10::Scalar(alpha)));
    }
    ---
    let add_Tensor = foreign "atg_add_Tensor" (atc_tensor @-> atc_tensor @-> double @-> returning atc_tensor) |}]

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

let%expect_test "softmax (int + ScalarType?)" =
  gen "softmax.int(Tensor self, int dim, ScalarType? dtype=None) -> Tensor";
  [%expect {| SKIPPED: unsupported arg type: ScalarType? |}]

let%expect_test "skipped: out= variant" =
  gen
    "add.out(Tensor self, Tensor other, *, Scalar alpha=1, Tensor(a!) out) -> \
     Tensor(a!)";
  [%expect {| SKIPPED: out= variant |}]

let%expect_test "skipped: unsupported arg (Dimname)" =
  gen "squeeze.dimname(Tensor(a) self, Dimname dim) -> Tensor(a)";
  [%expect {| SKIPPED: unsupported arg type: Dimname |}]
