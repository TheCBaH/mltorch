(* Per-operation expect tests for the ATen bindings (Aten.C.Operations): each
   builds small float tensors, runs one op, and prints shape = values. These
   exercise the real CPU kernels end to end (the binding + the static-dispatch
   archive), so a value here is the op actually computing, not just linking.

   Tensors are not freed: each test is tiny and the runner is short-lived, so
   leaking a handful of small buffers is cheaper than threading frees through
   every case. *)

open Ctypes
module F = Aten.C.Functions
module O = Aten.C.Operations
module Stype = Aten.Scalar_type
module T = Aten.Tensor

(* A float32 tensor of [shape] filled row-major from [vals]. *)
let make shape vals =
  let sizes = CArray.of_list int64_t (List.map Int64.of_int shape) in
  let t =
    F.new_ (CArray.start sizes)
      (Unsigned.Size_t.of_int (List.length shape))
      Stype.Float
  in
  let ba = T.as_float32 t |> Option.get in
  List.iteri (fun i v -> ba.{i} <- v) vals;
  t

(* An int64 array argument (e.g. a kernel/stride/shape list) as a C pointer. *)
let arr xs = CArray.start (CArray.of_list int64_t (List.map Int64.of_int xs))

(* A null int64 pointer: an absent [int?] optional argument (-> nullopt). *)
let none_int = from_voidp int64_t null

(* An array of tensor handles for a Tensor[] argument (e.g. cat). *)
let tensor_arr ts =
  CArray.start (CArray.of_list Aten.Function_description.atc_tensor ts)

let pp_shape fmt s =
  Format.fprintf fmt "[%a]"
    (Format.pp_print_list
       ~pp_sep:(fun f () -> Format.pp_print_string f "x")
       Format.pp_print_int)
    (Array.to_list s)

(* Print a result tensor as "shape = values" (flushing for ppx_expect). *)
let show t =
  Format.printf "%a = %a@." pp_shape (T.shape t) T.pp_float32
    (T.as_float32 t |> Option.get)

(* Show a non-float result (e.g. a bool mask) by casting to float32 first, so
   the float32 view applies; bool true/false render as 1/0. *)
let show_as_float t = show (O.to_dtype t Stype.Float false false None)

(* Print an int64 tensor (e.g. max-pool indices) as "shape = values". *)
let show_i64 t =
  let ba = T.data Aten.Dtype.int64 t |> Option.get in
  let vals =
    List.init (Bigarray.Array1.dim ba) (fun i -> Int64.to_string ba.{i})
  in
  Format.printf "%a = [%s]@." pp_shape (T.shape t) (String.concat "; " vals)

(* Read output handle [field] from a filled multi-output result struct. *)
let tget = Aten.Operation_description.tensors_get

let%expect_test "tensor runtime defaults" =
  let dt = F.default_dtype () in
  Format.printf "default dtype = %d, elem size = %d bytes@." (Stype.to_int dt)
    (Unsigned.Size_t.to_int (F.dtype_elem_size dt));
  [%expect "default dtype = 6, elem size = 4 bytes"]

let%expect_test "add.Tensor" =
  let a = make [ 2; 3 ] [ 1.; 2.; 3.; 4.; 5.; 6. ] in
  let b = make [ 2; 3 ] [ 3.; 3.; 3.; 3.; 3.; 3. ] in
  show (O.add_Tensor a b (Aten.Scalar.Int 1L));
  [%expect "[2x3] = [4; 5; 6; 7; 8; 9]"]

let%expect_test "mul.Tensor" =
  let a = make [ 2; 3 ] [ 1.; 2.; 3.; 4.; 5.; 6. ] in
  let b = make [ 2; 3 ] [ 3.; 3.; 3.; 3.; 3.; 3. ] in
  show (O.mul_Tensor a b);
  [%expect "[2x3] = [3; 6; 9; 12; 15; 18]"]

let%expect_test "div.Tensor" =
  let a = make [ 2; 3 ] [ 2.; 4.; 6.; 8.; 10.; 12. ] in
  let b = make [ 2; 3 ] [ 2.; 2.; 2.; 4.; 5.; 6. ] in
  show (O.div_Tensor a b);
  [%expect "[2x3] = [1; 2; 3; 2; 2; 2]"]

let%expect_test "clamp (Scalar? min/max)" =
  let a = make [ 2; 3 ] [ -2.; -1.; 0.; 1.; 2.; 3. ] in
  (* both bounds *)
  show (O.clamp a (Some (Aten.Scalar.Float 0.)) (Some (Aten.Scalar.Float 2.)));
  [%expect "[2x3] = [0; 0; 0; 1; 2; 2]"];
  (* min only (max = None) *)
  show (O.clamp a (Some (Aten.Scalar.Float 0.)) None);
  [%expect "[2x3] = [0; 0; 0; 1; 2; 3]"]

let%expect_test "mul.Scalar" =
  show (O.mul_Scalar (make [ 3 ] [ 1.; 2.; 3. ]) (Aten.Scalar.Float 2.));
  [%expect "[3] = [2; 4; 6]"]

let%expect_test "eq.Scalar (bool mask)" =
  show_as_float (O.eq_Scalar (make [ 3 ] [ 1.; 2.; 2. ]) (Aten.Scalar.Float 2.));
  [%expect "[3] = [0; 1; 1]"]

let%expect_test "logical_not (bool mask)" =
  (* true where the element is zero *)
  show_as_float (O.logical_not (make [ 3 ] [ 0.; 1.; 2. ]));
  [%expect "[3] = [1; 0; 0]"]

let%expect_test "any.dim (bool reduction)" =
  (* any over dim 1: row [0,0] -> false, row [1,0] -> true *)
  show_as_float (O.any_dim (make [ 2; 2 ] [ 0.; 0.; 1.; 0. ]) 1L false);
  [%expect "[2] = [0; 1]"]

let%expect_test "where.self (select by mask)" =
  let cond = O.eq_Scalar (make [ 3 ] [ 1.; 0.; 1. ]) (Aten.Scalar.Float 1.) in
  show
    (O.where_self cond
       (make [ 3 ] [ 10.; 20.; 30. ])
       (make [ 3 ] [ 1.; 2.; 3. ]));
  [%expect "[3] = [10; 2; 30]"]

let%expect_test
    "full_like (full sig: dtype/layout/device/pin_memory/memory_format)" =
  (* self, fill_value, dtype, layout, device, pin_memory, memory_format. All the
     optionals None except device = Some CPU, exercising the Device view. *)
  show
    (O.full_like
       (make [ 2; 2 ] [ 1.; 2.; 3.; 4. ])
       (Aten.Scalar.Float 0.) None None (Some Aten.Device.cpu) None None);
  [%expect "[2x2] = [0; 0; 0; 0]"]

let%expect_test "add_.Tensor (in-place)" =
  let e = make [ 2; 3 ] [ 10.; 11.; 12.; 13.; 14.; 15. ] in
  let b = make [ 2; 3 ] [ 3.; 3.; 3.; 3.; 3.; 3. ] in
  show (O.add__Tensor e b (Aten.Scalar.Int 1L));
  [%expect "[2x3] = [13; 14; 15; 16; 17; 18]"]

let%expect_test "relu" =
  let g = make [ 2; 3 ] [ -2.; -1.; 0.; 1.; 2.; 3. ] in
  show (O.relu g);
  [%expect "[2x3] = [0; 0; 0; 1; 2; 3]"]

let%expect_test "relu_ (in-place)" =
  let h = make [ 2; 3 ] [ -3.; -2.; -1.; 0.; 1.; 2. ] in
  show (O.relu_ h);
  [%expect "[2x3] = [0; 0; 0; 0; 1; 2]"]

let%expect_test "reshape (SymInt[])" =
  let a = make [ 2; 3 ] [ 1.; 2.; 3.; 4.; 5.; 6. ] in
  show (O.reshape a (arr [ 3; 2 ]) 2);
  [%expect "[3x2] = [1; 2; 3; 4; 5; 6]"]

let%expect_test "flatten.using_ints" =
  let a = make [ 2; 3 ] [ 1.; 2.; 3.; 4.; 5.; 6. ] in
  show (O.flatten_using_ints a 0L (-1L));
  [%expect "[6] = [1; 2; 3; 4; 5; 6]"]

(* 1x1x4x4 of 1..16, the shared pooling input. *)
let img () = make [ 1; 1; 4; 4 ] (List.init 16 (fun i -> float_of_int (i + 1)))

let%expect_test "avg_pool2d" =
  (* self, kernel_size, stride, padding, ceil_mode, count_include_pad,
     divisor_override (None). *)
  show
    (O.avg_pool2d (img ())
       (arr [ 2; 2 ])
       2
       (arr [ 2; 2 ])
       2
       (arr [ 0; 0 ])
       2 false true none_int);
  [%expect "[1x1x2x2] = [3.5; 5.5; 11.5; 13.5]"]

let%expect_test "max_pool2d" =
  show
    (O.max_pool2d (img ())
       (arr [ 2; 2 ])
       2
       (arr [ 2; 2 ])
       2
       (arr [ 0; 0 ])
       2
       (arr [ 1; 1 ])
       2 false);
  [%expect "[1x1x2x2] = [6; 8; 14; 16]"]

let%expect_test "adaptive_avg_pool2d" =
  show (O.adaptive_avg_pool2d (img ()) (arr [ 1; 1 ]) 2);
  [%expect "[1x1x1x1] = [8.5]"]

let%expect_test "linear" =
  let x = make [ 1; 3 ] [ 1.; 2.; 3. ] in
  let w = make [ 2; 3 ] [ 1.; 0.; 0.; 0.; 1.; 1. ] in
  let b = make [ 2 ] [ 10.; 20. ] in
  show (O.linear x w b);
  [%expect "[1x2] = [11; 25]"]

let%expect_test "batch_norm (inference)" =
  let x = make [ 1; 2; 1; 2 ] [ 1.; 2.; 3.; 4. ] in
  let w = make [ 2 ] [ 2.; 2. ] and b = make [ 2 ] [ 1.; 1. ] in
  let mean = make [ 2 ] [ 0.; 0. ] and var = make [ 2 ] [ 1.; 1. ] in
  show (O.batch_norm x w b mean var false 0.1 0.0 false);
  [%expect "[1x2x1x2] = [3; 5; 7; 9]"]

let%expect_test "conv2d" =
  let x = make [ 1; 1; 3; 3 ] [ 1.; 2.; 3.; 4.; 5.; 6.; 7.; 8.; 9. ] in
  let w = make [ 1; 1; 2; 2 ] [ 1.; 0.; 0.; 1. ] in
  let b = make [ 1 ] [ 0. ] in
  show (O.conv2d x w b (arr [ 1; 1 ]) 2 (arr [ 0; 0 ]) 2 (arr [ 1; 1 ]) 2 1L);
  [%expect "[1x1x2x2] = [6; 8; 12; 14]"]

let%expect_test "dropout / dropout_ (inference)" =
  let t = make [ 2; 3 ] [ 1.; 2.; 3.; 4.; 5.; 6. ] in
  show (O.dropout t 0.5 false);
  show (O.dropout_ t 0.5 false);
  [%expect {|
    [2x3] = [1; 2; 3; 4; 5; 6]
    [2x3] = [1; 2; 3; 4; 5; 6] |}]

let%expect_test "sigmoid" =
  show (O.sigmoid (make [ 3 ] [ 0.; 0.; 0. ]));
  [%expect "[3] = [0.5; 0.5; 0.5]"]

let%expect_test "hardtanh_" =
  show
    (O.hardtanh_
       (make [ 3 ] [ -1.; 3.; 8. ])
       (Aten.Scalar.Float 0.0) (Aten.Scalar.Float 6.0));
  [%expect "[3] = [0; 3; 6]"]

let%expect_test "silu_" =
  show (O.silu_ (make [ 3 ] [ 0.; 2.; -2. ]));
  [%expect "[3] = [0; 1.76159; -0.238406]"]

let%expect_test "hardtanh (functional)" =
  show
    (O.hardtanh
       (make [ 3 ] [ -2.; 0.5; 2. ])
       (Aten.Scalar.Float 0.0) (Aten.Scalar.Float 1.0));
  [%expect "[3] = [0; 0.5; 1]"]

let%expect_test "permute" =
  (* permute returns a non-contiguous view; pp_float32 reads raw physical memory,
     so values appear in original storage order even though shape is transposed. *)
  show (O.permute (make [ 2; 3 ] [ 1.; 2.; 3.; 4.; 5.; 6. ]) (arr [ 1; 0 ]) 2);
  [%expect "[3x2] = [1; 2; 3; 4; 5; 6]"]

let%expect_test "view" =
  show (O.view (make [ 2; 3 ] [ 1.; 2.; 3.; 4.; 5.; 6. ]) (arr [ 3; 2 ]) 2);
  [%expect "[3x2] = [1; 2; 3; 4; 5; 6]"]

let%expect_test "expand" =
  (* broadcast a single row to two; clone to materialise the stride-0 view so
     pp_float32 reads real values rather than past the end of storage. *)
  show
    (O.clone
       (O.expand (make [ 1; 3 ] [ 1.; 2.; 3. ]) (arr [ 2; 3 ]) 2 false)
       None);
  [%expect "[2x3] = [1; 2; 3; 1; 2; 3]"]

let%expect_test "select.int" =
  (* row 1 of a 2x3 matrix; clone to materialise the offset view (pp_float32
     reads from the storage base, ignoring the view's storage offset). *)
  show
    (O.clone
       (O.select_int (make [ 2; 3 ] [ 1.; 2.; 3.; 4.; 5.; 6. ]) 0L 1L)
       None);
  [%expect "[3] = [4; 5; 6]"]

let%expect_test "squeeze.dims" =
  show (O.squeeze_dims (make [ 1; 3; 1 ] [ 1.; 2.; 3. ]) (arr [ 0; 2 ]) 2);
  [%expect "[3] = [1; 2; 3]"]

let%expect_test "unsqueeze" =
  show (O.unsqueeze (make [ 3 ] [ 1.; 2.; 3. ]) 0L);
  [%expect "[1x3] = [1; 2; 3]"]

let%expect_test "addmm" =
  (* out = beta * C + alpha * (A @ B): [1x3] = [10; 20; 30] + [1x2] @ [2x3] *)
  let c = make [ 1; 3 ] [ 10.; 20.; 30. ] in
  let a = make [ 1; 2 ] [ 1.; 2. ] in
  let b = make [ 2; 3 ] [ 1.; 0.; 0.; 0.; 1.; 0. ] in
  show (O.addmm c a b (Aten.Scalar.Int 1L) (Aten.Scalar.Int 1L));
  [%expect "[1x3] = [11; 22; 30]"]

let%expect_test "convolution" =
  let x = make [ 1; 1; 3; 3 ] [ 1.; 2.; 3.; 4.; 5.; 6.; 7.; 8.; 9. ] in
  let w = make [ 1; 1; 2; 2 ] [ 1.; 0.; 0.; 1. ] in
  let b = make [ 1 ] [ 0. ] in
  show
    (O.convolution x w b
       (arr [ 1; 1 ])
       2
       (arr [ 0; 0 ])
       2
       (arr [ 1; 1 ])
       2 false
       (arr [ 0; 0 ])
       2 1L);
  [%expect "[1x1x2x2] = [6; 8; 12; 14]"]

let%expect_test "bmm" =
  (* batched [1x2x2] @ [1x2x2]: [[1,2],[3,4]] @ [[5,6],[7,8]] = [[19,22],[43,50]] *)
  let a = make [ 1; 2; 2 ] [ 1.; 2.; 3.; 4. ] in
  let b = make [ 1; 2; 2 ] [ 5.; 6.; 7.; 8. ] in
  show (O.bmm a b);
  [%expect "[1x2x2] = [19; 22; 43; 50]"]

let%expect_test "_softmax" =
  (* softmax over a 2-vector of equal logits is uniform; half_to_float=false. *)
  show (O._softmax (make [ 2 ] [ 0.; 0. ]) 0L false);
  [%expect "[2] = [0.5; 0.5]"]

let%expect_test "gelu (exact, approximate=none)" =
  show (O.gelu (make [ 3 ] [ 0.; 1.; 2. ]) "none");
  [%expect "[3] = [0; 0.841345; 1.9545]"]

let%expect_test "cat (Tensor[])" =
  (* concat two [1x2] rows along dim 0 -> [2x2] *)
  let a = make [ 1; 2 ] [ 1.; 2. ] in
  let b = make [ 1; 2 ] [ 3.; 4. ] in
  show (O.cat (tensor_arr [ a; b ]) 2 0L);
  [%expect "[2x2] = [1; 2; 3; 4]"]

let%expect_test "mean.dim" =
  (* self, dim, keepdim, dtype (None): mean over dim 0, no keepdim: [2x3] → [3] *)
  show
    (O.mean_dim
       (make [ 2; 3 ] [ 1.; 2.; 3.; 4.; 5.; 6. ])
       (arr [ 0 ]) 1 false None);
  [%expect "[3] = [2.5; 3.5; 4.5]"]

let%expect_test "_native_batch_norm_legit_no_training (3-tuple out-struct)" =
  let x = make [ 1; 2; 1; 2 ] [ 1.; 2.; 3.; 4. ] in
  let w = make [ 2 ] [ 2.; 2. ] in
  let b = make [ 2 ] [ 1.; 1. ] in
  let mean = make [ 2 ] [ 0.; 0. ] in
  let var = make [ 2 ] [ 1.; 1. ] in
  let describe name t =
    if F.defined t = 0 then Format.printf "%s: undefined@." name
    else Format.printf "%s: %a@." name pp_shape (T.shape t)
  in
  (* op returns a 0/-1 status and fills the 3 outputs into the struct; the saved
     mean / invstd are empty in the no_training path (backward never runs). *)
  let out = Ctypes.make Aten.Types_generated.tensors3_struct in
  let status =
    O._native_batch_norm_legit_no_training x w b mean var 0.1 0.0 (addr out)
  in
  Printf.printf "status=%d\n" status;
  show (tget out Aten.Types_generated.tensors3_v0);
  describe "save_mean" (tget out Aten.Types_generated.tensors3_v1);
  describe "save_invstd" (tget out Aten.Types_generated.tensors3_v2);
  [%expect
    {|
    status=0
    [1x2x1x2] = [3; 5; 7; 9]
    save_mean: [0]
    save_invstd: [0] |}]

let%expect_test "native_layer_norm (3-tuple out-struct)" =
  (* layer-norm over the last dim of [1x2] [0,2]: mean=1, var=1, eps=0 ->
     normalized=[-1,1]; weight=1/bias=0 leave it unchanged. *)
  let x = make [ 1; 2 ] [ 0.; 2. ] in
  let w = make [ 2 ] [ 1.; 1. ] in
  let b = make [ 2 ] [ 0.; 0. ] in
  let describe name t = Format.printf "%s: %a@." name pp_shape (T.shape t) in
  let out = Ctypes.make Aten.Types_generated.tensors3_struct in
  let status = O.native_layer_norm x (arr [ 2 ]) 1 w b 0.0 (addr out) in
  Printf.printf "status=%d\n" status;
  show (tget out Aten.Types_generated.tensors3_v0);
  describe "mean" (tget out Aten.Types_generated.tensors3_v1);
  describe "rstd" (tget out Aten.Types_generated.tensors3_v2);
  [%expect
    {|
    status=0
    [1x2] = [-1; 1]
    mean: [1x1]
    rstd: [1x1] |}]

let%expect_test "max_pool2d_with_indices (2-tuple out-struct)" =
  (* output and the int64 index map, both filled into the out-struct. *)
  let out = Ctypes.make Aten.Types_generated.tensors2_struct in
  let status =
    O.max_pool2d_with_indices (img ())
      (arr [ 2; 2 ])
      2
      (arr [ 2; 2 ])
      2
      (arr [ 0; 0 ])
      2
      (arr [ 1; 1 ])
      2 false (addr out)
  in
  Printf.printf "status=%d\n" status;
  show (tget out Aten.Types_generated.tensors2_v0);
  show_i64 (tget out Aten.Types_generated.tensors2_v1);
  [%expect
    {|
    status=0
    [1x1x2x2] = [6; 8; 14; 16]
    [1x1x2x2] = [5; 7; 13; 15] |}]
