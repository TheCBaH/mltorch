(* Per-operation expect tests for the ATen bindings (Aten_c.Aten_operations): each
   builds small float tensors, runs one op, and prints shape = values. These
   exercise the real CPU kernels end to end (the binding + the static-dispatch
   archive), so a value here is the op actually computing, not just linking.

   Tensors are not freed: each test is tiny and the runner is short-lived, so
   leaking a handful of small buffers is cheaper than threading frees through
   every case. *)

open Ctypes
module F = Aten_c.Aten_functions
module O = Aten_c.Aten_operations
module Stype = Aten_scalar_type
module T = Aten_tensor

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

(* A null tensor handle / double pointer: an absent [Tensor?] / [float?]
   optional argument (-> nullopt). *)
let none_tensor = from_voidp Aten_types_generated.tensor_opaque null
let none_double = from_voidp double null

(* An array of tensor handles for a Tensor[] argument (e.g. cat). *)
let tensor_arr ts =
  CArray.start (CArray.of_list Aten_function_description.atc_tensor ts)

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
  let ba = T.data Aten_dtype.int64 t |> Option.get in
  let vals =
    List.init (Bigarray.Array1.dim ba) (fun i -> Int64.to_string ba.{i})
  in
  Format.printf "%a = [%s]@." pp_shape (T.shape t) (String.concat "; " vals)

(* Read output handle [field] from a filled multi-output result struct. *)
let tget = Aten_operation_description.tensors_get

let%expect_test "tensor runtime defaults" =
  let dt = F.default_dtype () in
  Format.printf "default dtype = %d, elem size = %d bytes@." (Stype.to_int dt)
    (Unsigned.Size_t.to_int (F.dtype_elem_size dt));
  [%expect "default dtype = 6, elem size = 4 bytes"]

let%expect_test "add.Tensor" =
  let a = make [ 2; 3 ] [ 1.; 2.; 3.; 4.; 5.; 6. ] in
  let b = make [ 2; 3 ] [ 3.; 3.; 3.; 3.; 3.; 3. ] in
  show (O.add_Tensor a b (Aten_scalar.Int 1L));
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
  show (O.clamp a (Some (Aten_scalar.Float 0.)) (Some (Aten_scalar.Float 2.)));
  [%expect "[2x3] = [0; 0; 0; 1; 2; 2]"];
  (* min only (max = None) *)
  show (O.clamp a (Some (Aten_scalar.Float 0.)) None);
  [%expect "[2x3] = [0; 0; 0; 1; 2; 3]"]

let%expect_test "mul.Scalar" =
  show (O.mul_Scalar (make [ 3 ] [ 1.; 2.; 3. ]) (Aten_scalar.Float 2.));
  [%expect "[3] = [2; 4; 6]"]

let%expect_test "eq.Scalar (bool mask)" =
  show_as_float (O.eq_Scalar (make [ 3 ] [ 1.; 2.; 2. ]) (Aten_scalar.Float 2.));
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
  let cond = O.eq_Scalar (make [ 3 ] [ 1.; 0.; 1. ]) (Aten_scalar.Float 1.) in
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
       (Aten_scalar.Float 0.) None None (Some Aten_device.cpu) None None);
  [%expect "[2x2] = [0; 0; 0; 0]"]

let%expect_test "add_.Tensor (in-place)" =
  let e = make [ 2; 3 ] [ 10.; 11.; 12.; 13.; 14.; 15. ] in
  let b = make [ 2; 3 ] [ 3.; 3.; 3.; 3.; 3.; 3. ] in
  show (O.add__Tensor e b (Aten_scalar.Int 1L));
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

(* An executable feasibility oracle for [aten.lstm.input]. The deliberately
   unequal sequence, batch, input, and hidden dimensions make accidental
   dimension swapping visible. Zero parameters leave the recurrence small
   enough to inspect, while nonzero [h0]/[c0] prove the initial-state path is
   live: i=f=o=0.5 and g=0, so c halves on each step. *)
let%expect_test "lstm.input (Tensor[] inputs, three outputs)" =
  let input = make [ 2; 3; 4 ] (List.init 24 (fun i -> float_of_int i)) in
  let h0 = make [ 1; 3; 5 ] (List.init 15 (fun i -> float_of_int (i + 1))) in
  let c0 =
    make [ 1; 3; 5 ] (List.init 15 (fun i -> float_of_int (i + 1) /. 10.))
  in
  let zeros shape =
    make shape (List.init (List.fold_left ( * ) 1 shape) (fun _ -> 0.))
  in
  let params =
    tensor_arr [ zeros [ 20; 4 ]; zeros [ 20; 5 ]; zeros [ 20 ]; zeros [ 20 ] ]
  in
  let hx = tensor_arr [ h0; c0 ] in
  let out = Ctypes.make Aten_types_generated.tensors3_struct in
  let status =
    O.lstm_input input hx 2 params 4 true 1L 0.0 false false false (addr out)
  in
  Printf.printf "status=%d outputs=3\n" status;
  show (tget out Aten_types_generated.tensors3_v0);
  show (tget out Aten_types_generated.tensors3_v1);
  show (tget out Aten_types_generated.tensors3_v2);
  [%expect
    {|
    status=0 outputs=3
    [2x3x5] = [0.0249792; 0.049834; 0.0744425; 0.0986877; 0.122459; 0.145656; 0.168188; 0.189974; 0.210949; 0.231059; 0.25026; 0.268525; 0.285835; 0.302184; 0.317574; 0.0124974; 0.0249792; 0.0374298; 0.049834; 0.0621765; 0.0744425; 0.0866176; 0.0986877; 0.110639; 0.122459; 0.134136; 0.145656; 0.15701; 0.168188; 0.179179]
    [1x3x5] = [0.0124974; 0.0249792; 0.0374298; 0.049834; 0.0621765; 0.0744425; 0.0866176; 0.0986877; 0.110639; 0.122459; 0.134136; 0.145656; 0.15701; 0.168188; 0.179179]
    [1x3x5] = [0.025; 0.05; 0.075; 0.1; 0.125; 0.15; 0.175; 0.2; 0.225; 0.25; 0.275; 0.3; 0.325; 0.35; 0.375] |}]

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

let%expect_test "conv2d (dilated)" =
  let x = make [ 1; 1; 1; 5 ] [ 0.; 1.; 2.; 3.; 4. ] in
  let w = make [ 1; 1; 1; 3 ] [ 1.; 1.; 1. ] in
  let b = make [ 1 ] [ 0. ] in
  show (O.conv2d x w b (arr [ 1; 1 ]) 2 (arr [ 0; 1 ]) 2 (arr [ 1; 2 ]) 2 1L);
  [%expect "[1x1x1x3] = [4; 6; 4]"]

let%expect_test "conv2d.padding" =
  let x = make [ 1; 1; 3; 3 ] [ 1.; 2.; 3.; 4.; 5.; 6.; 7.; 8.; 9. ] in
  let w = make [ 1; 1; 2; 2 ] [ 1.; 0.; 0.; 1. ] in
  let b = make [ 1 ] [ 0. ] in
  show (O.conv2d_padding x w b (arr [ 1; 1 ]) 2 "valid" (arr [ 1; 1 ]) 2 1L);
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
       (Aten_scalar.Float 0.0) (Aten_scalar.Float 6.0));
  [%expect "[3] = [0; 3; 6]"]

let%expect_test "silu_" =
  show (O.silu_ (make [ 3 ] [ 0.; 2.; -2. ]));
  [%expect "[3] = [0; 1.76159; -0.238406]"]

let%expect_test "silu (functional)" =
  show (O.silu (make [ 3 ] [ 0.; 2.; -2. ]));
  [%expect "[3] = [0; 1.76159; -0.238406]"]

let%expect_test "hardsigmoid" =
  show (O.hardsigmoid (make [ 3 ] [ -6.; 0.; 6. ]));
  [%expect "[3] = [0; 0.5; 1]"]

let%expect_test "hardsigmoid_" =
  show (O.hardsigmoid_ (make [ 3 ] [ -6.; 0.; 6. ]));
  [%expect "[3] = [0; 0.5; 1]"]

let%expect_test "hardswish" =
  show (O.hardswish (make [ 3 ] [ -6.; 0.; 6. ]));
  [%expect "[3] = [-0; 0; 6]"]

let%expect_test "hardswish_" =
  show (O.hardswish_ (make [ 3 ] [ -6.; 0.; 6. ]));
  [%expect "[3] = [-0; 0; 6]"]

let%expect_test "hardtanh (functional)" =
  show
    (O.hardtanh
       (make [ 3 ] [ -2.; 0.5; 2. ])
       (Aten_scalar.Float 0.0) (Aten_scalar.Float 1.0));
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
  show (O.addmm c a b (Aten_scalar.Int 1L) (Aten_scalar.Int 1L));
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

let%expect_test "convolution (dilated)" =
  let x = make [ 1; 1; 1; 5 ] [ 0.; 1.; 2.; 3.; 4. ] in
  let w = make [ 1; 1; 1; 3 ] [ 1.; 1.; 1. ] in
  let b = make [ 1 ] [ 0. ] in
  show
    (O.convolution x w b
       (arr [ 1; 1 ])
       2
       (arr [ 0; 1 ])
       2
       (arr [ 1; 2 ])
       2 false
       (arr [ 0; 0 ])
       2 1L);
  [%expect "[1x1x1x3] = [4; 6; 4]"]

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

let%expect_test "scaled_dot_product_attention (no mask, default scale)" =
  (* [D=1,H=1,Wq=1,C=1] q/k/v, no mask: single query/key, attention is
     trivially 1.0 on the only key, so out = value. Proves the archive
     closure holds end to end (real flash-CPU dispatch, not just a link) --
     op8-impl.md commit 0's spike, landed as this binding's coverage. *)
  let q = make [ 1; 1; 1; 1 ] [ 1. ] in
  let k = make [ 1; 1; 1; 1 ] [ 2. ] in
  let v = make [ 1; 1; 1; 1 ] [ 7. ] in
  show
    (O.scaled_dot_product_attention q k v none_tensor 0.0 false none_double
       false);
  [%expect "[1x1x1x1] = [7]"]

let%expect_test "scaled_dot_product_attention (explicit mask + explicit scale)"
    =
  (* q=[1,0], k0=[1,0] (dot=1), k1=[0,1] (dot=0); explicit scale=1 so
     score=dot; a rank-2 [Wq=1,Wk=2] additive mask of [0,-inf] excludes k1
     entirely, so the output is exactly value0=[10,20] -- exercises the
     binding's mask-tensor and scale-pointer arguments, which the no-mask
     case above never touches. *)
  let q = make [ 1; 1; 1; 2 ] [ 1.; 0. ] in
  let k = make [ 1; 1; 2; 2 ] [ 1.; 0.; 0.; 1. ] in
  let v = make [ 1; 1; 2; 2 ] [ 10.; 20.; 30.; 40. ] in
  let mask = make [ 1; 2 ] [ 0.; Float.neg_infinity ] in
  let scale = allocate double 1.0 in
  show (O.scaled_dot_product_attention q k v mask 0.0 false scale false);
  [%expect "[1x1x1x2] = [10; 20]"]

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
  let out = Ctypes.make Aten_types_generated.tensors3_struct in
  let status =
    O._native_batch_norm_legit_no_training x w b mean var 0.1 0.0 (addr out)
  in
  Printf.printf "status=%d\n" status;
  show (tget out Aten_types_generated.tensors3_v0);
  describe "save_mean" (tget out Aten_types_generated.tensors3_v1);
  describe "save_invstd" (tget out Aten_types_generated.tensors3_v2);
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
  let out = Ctypes.make Aten_types_generated.tensors3_struct in
  let status = O.native_layer_norm x (arr [ 2 ]) 1 w b 0.0 (addr out) in
  Printf.printf "status=%d\n" status;
  show (tget out Aten_types_generated.tensors3_v0);
  describe "mean" (tget out Aten_types_generated.tensors3_v1);
  describe "rstd" (tget out Aten_types_generated.tensors3_v2);
  [%expect
    {|
    status=0
    [1x2] = [-1; 1]
    mean: [1x1]
    rstd: [1x1] |}]

let%expect_test "rms_norm (float? eps, SymInt[] normalized_shape)" =
  (* RMS-norm over the last dim of [1x4] all-2s: mean(x^2)=4, +eps(12)=16,
     rsqrt=0.25, so x*0.25 = 0.5 each; weight=[1,2,3,4] scales it to
     [0.5,1,1.5,2]. eps is the float? optional (the new ptr-double binding),
     passed by an [allocate]d pointer. *)
  let x = make [ 1; 4 ] [ 2.; 2.; 2.; 2. ] in
  let w = make [ 4 ] [ 1.; 2.; 3.; 4. ] in
  let eps = allocate double 12.0 in
  show (O.rms_norm x (arr [ 4 ]) 1 w eps);
  [%expect "[1x4] = [0.5; 1; 1.5; 2]"]

let%expect_test "max_pool2d_with_indices (2-tuple out-struct)" =
  (* output and the int64 index map, both filled into the out-struct. *)
  let out = Ctypes.make Aten_types_generated.tensors2_struct in
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
  show (tget out Aten_types_generated.tensors2_v0);
  show_i64 (tget out Aten_types_generated.tensors2_v1);
  [%expect
    {|
    status=0
    [1x1x2x2] = [6; 8; 14; 16]
    [1x1x2x2] = [5; 7; 13; 15] |}]

let%expect_test "argmax" =
  let t = make [ 2; 3 ] [ 1.; 5.; 2.; 0.; 9.; 3. ] in
  (* flatten (dim=None): global max 9 is at flat index 4 *)
  Printf.printf "flat=%d\n" (T.item_int (O.argmax t none_int false));
  (* along dim 1: per-row argmax = [1; 1] *)
  let dim = allocate int64_t 1L in
  show_i64 (O.argmax t dim false);
  [%expect {|
    flat=4
    [2] = [1; 1] |}]

let%expect_test "topk" =
  let t = make [ 1; 5 ] [ 0.1; 0.5; 0.2; 0.9; 0.3 ] in
  (* top-3 over the last dim, largest first *)
  let out = Ctypes.make Aten_types_generated.tensors2_struct in
  let status = O.topk t 3L (-1L) true true (addr out) in
  Printf.printf "status=%d\n" status;
  show (tget out Aten_types_generated.tensors2_v0);
  show_i64 (tget out Aten_types_generated.tensors2_v1);
  [%expect {|
    status=0
    [1x3] = [0.9; 0.5; 0.3]
    [1x3] = [3; 1; 4] |}]

(* --- unbind.int: the only Tensor[]-returning binding ---------------------

   [show] reads flat off the storage base, and unbind returns VIEWS, so every
   value here goes through [T.materialize_for_raw_read] first. Reading a result
   directly would print result 0's numbers for every index. *)

let show_view t = show (T.materialize_for_raw_read t)

let%expect_test "unbind.int at dim 0" =
  let x = make [ 3; 2 ] [ 0.; 1.; 2.; 3.; 4.; 5. ] in
  let rows = Aten_tensor_list.to_list (O.unbind_int x 0L) in
  Printf.printf "n=%d\n" (List.length rows);
  List.iter show_view rows;
  [%expect {|
    n=3
    [2] = [0; 1]
    [2] = [2; 3]
    [2] = [4; 5] |}]

let%expect_test "unbind.int at dim -1" =
  (* Columns of a 2x3: the negative dim normalizes to 2's last axis, and each
     result is a STRIDED view, the other half of the view story. *)
  let x = make [ 2; 3 ] [ 0.; 1.; 2.; 3.; 4.; 5. ] in
  let cols = Aten_tensor_list.to_list (O.unbind_int x (-1L)) in
  Printf.printf "n=%d\n" (List.length cols);
  List.iter show_view cols;
  [%expect {|
    n=3
    [2] = [0; 3]
    [2] = [1; 4]
    [2] = [2; 5] |}]

let%expect_test "unbind.int of a zero-length dim is the empty list" =
  (* [make] can't build this one: a zero-element tensor has a null data pointer,
     so there is no Bigarray view to fill. *)
  let x = T.create [ 0; 2 ] in
  Printf.printf "n=%d\n"
    (List.length (Aten_tensor_list.to_list (O.unbind_int x 0L)));
  [%expect "n=0"]

(* The results are views onto the input's storage, and they outlive the
   container [to_list] frees. Mutating result 1 in place must therefore show up
   in the input's second row -- proving the extracted handle kept its view
   metadata (offset included) and its storage aliasing. Result 1, not 0: result
   0 is at offset 0 and would pass even if the offset were lost. *)
let%expect_test "unbind results alias their input and survive the container" =
  let x = make [ 3; 2 ] [ 0.; 1.; 2.; 3.; 4.; 5. ] in
  let rows = Aten_tensor_list.to_list (O.unbind_int x 0L) in
  let row1 = List.nth rows 1 in
  (* add_ returns Tensor(a!) -- a NEW owning handle over the same storage, so
     manage it or it leaks past the live-count assertions elsewhere. *)
  ignore
    (T.manage
       (O.add__Tensor row1 (make [ 2 ] [ 10.; 20. ]) (Aten_scalar.Int 1L)));
  show x;
  let expected = make [ 2 ] [ 12.; 23. ] in
  Printf.printf "row1 == [12;23]: %b\n"
    (T.equal (T.manage (O.select_int x 0L 1L)) expected);
  [%expect {|
    [3x2] = [0; 1; 12; 23; 4; 5]
    row1 == [12;23]: true |}]

let%expect_test "unbind.int with an out-of-range dim is a checked error" =
  let x = make [ 2; 3 ] [ 0.; 1.; 2.; 3.; 4.; 5. ] in
  (match Aten_tensor_list.to_list (O.unbind_int x 5L) with
  | exception T.Error _ -> print_endline "raised Tensor.Error"
  | l -> Printf.printf "no error, n=%d\n" (List.length l));
  [%expect "raised Tensor.Error"]
