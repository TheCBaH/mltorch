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
      (Stype.to_int Stype.Float)
  in
  let ba = T.as_float32 t |> Option.get in
  List.iteri (fun i v -> ba.{i} <- v) vals;
  t

(* An int64 array argument (e.g. a kernel/stride/shape list) as a C pointer. *)
let arr xs = CArray.start (CArray.of_list int64_t (List.map Int64.of_int xs))

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

let%expect_test "tensor runtime defaults" =
  let dt = F.default_dtype () in
  Format.printf "default dtype = %d, elem size = %d bytes@." dt
    (Unsigned.Size_t.to_int (F.dtype_elem_size dt));
  [%expect "default dtype = 6, elem size = 4 bytes"]

let%expect_test "add.Tensor" =
  let a = make [ 2; 3 ] [ 1.; 2.; 3.; 4.; 5.; 6. ] in
  let b = make [ 2; 3 ] [ 3.; 3.; 3.; 3.; 3.; 3. ] in
  show (O.add_Tensor a b 1.0);
  [%expect "[2x3] = [4; 5; 6; 7; 8; 9]"]

let%expect_test "mul.Tensor" =
  let a = make [ 2; 3 ] [ 1.; 2.; 3.; 4.; 5.; 6. ] in
  let b = make [ 2; 3 ] [ 3.; 3.; 3.; 3.; 3.; 3. ] in
  show (O.mul_Tensor a b);
  [%expect "[2x3] = [3; 6; 9; 12; 15; 18]"]

let%expect_test "add_.Tensor (in-place)" =
  let e = make [ 2; 3 ] [ 10.; 11.; 12.; 13.; 14.; 15. ] in
  let b = make [ 2; 3 ] [ 3.; 3.; 3.; 3.; 3.; 3. ] in
  show (O.add__Tensor e b 1.0);
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
  show (O.avg_pool2d (img ()) (arr [ 2; 2 ]) 2);
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
  show (O.hardtanh_ (make [ 3 ] [ -1.; 3.; 8. ]) 0.0 6.0);
  [%expect "[3] = [0; 3; 6]"]

let%expect_test "silu_" =
  show (O.silu_ (make [ 3 ] [ 0.; 2.; -2. ]));
  [%expect "[3] = [0; 1.76159; -0.238406]"]

let%expect_test "hardtanh (functional)" =
  show (O.hardtanh (make [ 3 ] [ -2.; 0.5; 2. ]) 0.0 1.0);
  [%expect "[3] = [0; 0.5; 1]"]

let%expect_test "permute" =
  (* permute returns a non-contiguous view; pp_float32 reads raw physical memory,
     so values appear in original storage order even though shape is transposed. *)
  show (O.permute (make [ 2; 3 ] [ 1.; 2.; 3.; 4.; 5.; 6. ]) (arr [ 1; 0 ]) 2);
  [%expect "[3x2] = [1; 2; 3; 4; 5; 6]"]

let%expect_test "view" =
  show (O.view (make [ 2; 3 ] [ 1.; 2.; 3.; 4.; 5.; 6. ]) (arr [ 3; 2 ]) 2);
  [%expect "[3x2] = [1; 2; 3; 4; 5; 6]"]

let%expect_test "addmm" =
  (* out = beta * C + alpha * (A @ B): [1x3] = [10; 20; 30] + [1x2] @ [2x3] *)
  let c = make [ 1; 3 ] [ 10.; 20.; 30. ] in
  let a = make [ 1; 2 ] [ 1.; 2. ] in
  let b = make [ 2; 3 ] [ 1.; 0.; 0.; 0.; 1.; 0. ] in
  show (O.addmm c a b 1.0 1.0);
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

let%expect_test "mean.dim" =
  (* mean over dim 0, no keepdim: [2x3] → [3] *)
  show
    (O.mean_dim (make [ 2; 3 ] [ 1.; 2.; 3.; 4.; 5.; 6. ]) (arr [ 0 ]) 1 false);
  [%expect "[3] = [2.5; 3.5; 4.5]"]

let%expect_test "_native_batch_norm_legit_no_training" =
  let x = make [ 1; 2; 1; 2 ] [ 1.; 2.; 3.; 4. ] in
  let w = make [ 2 ] [ 2.; 2. ] in
  let b = make [ 2 ] [ 1.; 1. ] in
  let mean = make [ 2 ] [ 0.; 0. ] in
  let var = make [ 2 ] [ 1.; 1. ] in
  show (F._native_batch_norm_legit_no_training_default x w b mean var 0.1 0.0);
  [%expect "[1x2x1x2] = [3; 5; 7; 9]"]

let%expect_test "max_pool2d_with_indices" =
  show
    (F.max_pool2d_with_indices_default (img ())
       (arr [ 2; 2 ])
       2
       (arr [ 2; 2 ])
       2
       (arr [ 0; 0 ])
       2
       (arr [ 1; 1 ])
       2 0);
  [%expect "[1x1x2x2] = [6; 8; 14; 16]"]
