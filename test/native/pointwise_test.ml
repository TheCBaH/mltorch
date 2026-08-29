(* Elementwise arithmetic (relu/add/div/mul/sqrt/pow/sub), clamp, hardtanh,
   clone, and the *_scalar/broadcast pointwise ops. Split from
   compute_test.ml. *)

open Compute_fixtures

let%expect_test "Direct: relu pointwise" =
  let module R = Pointwise.Relu.Compute (Direct) in
  let x_shape = s1c 4 in
  let x =
    Tensor.materialize x_shape (fun c -> [| -2.; -0.5; 1.; 3. |].(chan c))
  in
  Format.printf "%a@." (pp_result Tensor.pp)
    (eval_tensor (Pointwise.Relu.output_shape x_shape) (R.pixel x));
  [%expect {| tensor f32 [C=4] {0, 0, 1, 3} |}]

let%expect_test "Direct: add" =
  let module A = Pointwise.Add.Compute (Direct) in
  let x_shape = s1c 3 and y_shape = s1c 3 in
  let x = Tensor.materialize x_shape (fun c -> float_of_int (chan c)) in
  let y = Tensor.materialize y_shape (fun _ -> 10.) in
  Format.printf "%a@." (pp_result Tensor.pp)
    (eval_tensor
       (Pointwise.Add.output_shape x_shape y_shape)
       (A.pixel ~a_shape:x_shape ~b_shape:y_shape x y));
  [%expect {| tensor f32 [C=3] {10, 11, 12} |}]

let%expect_test "Direct: div" =
  let module D = Pointwise.Div.Compute (Direct) in
  let x_shape = s1c 3 and y_shape = s1c 3 in
  let x = Tensor.materialize x_shape (fun c -> float_of_int (chan c) +. 1.) in
  let y = Tensor.materialize y_shape (fun _ -> 4.) in
  Format.printf "%a@." (pp_result Tensor.pp)
    (eval_tensor
       (Pointwise.Div.output_shape x_shape y_shape)
       (D.pixel ~a_shape:x_shape ~b_shape:y_shape x y));
  [%expect {| tensor f32 [C=3] {0.25, 0.5, 0.75} |}]

let%expect_test "Direct: mul" =
  let module M = Pointwise.Mul.Compute (Direct) in
  let x_shape = s1c 3 and y_shape = s1c 3 in
  let x = Tensor.materialize x_shape (fun c -> float_of_int (chan c)) in
  let y = Tensor.materialize y_shape (fun _ -> 10.) in
  Format.printf "%a@." (pp_result Tensor.pp)
    (eval_tensor
       (Pointwise.Mul.output_shape x_shape y_shape)
       (M.pixel ~a_shape:x_shape ~b_shape:y_shape x y));
  [%expect {| tensor f32 [C=3] {0, 10, 20} |}]

let%expect_test "Direct: sqrt" =
  let module Q = Pointwise.Sqrt.Compute (Direct) in
  let x_shape = s1c 4 in
  let x =
    Tensor.materialize x_shape (fun c -> [| 0.; 1.; 4.; 2.25 |].(chan c))
  in
  Format.printf "%a@." (pp_result Tensor.pp)
    (eval_tensor (Pointwise.Sqrt.output_shape x_shape) (Q.pixel x));
  [%expect {| tensor f32 [C=4] {0, 1, 2, 1.5} |}]

let%expect_test "Direct: pow, special-cased and generic exponents" =
  let module Q = Pointwise.Pow.Compute (Direct) in
  let x_shape = s1c 3 in
  let x = Tensor.materialize x_shape (fun c -> [| 1.; 4.; 9. |].(chan c)) in
  let run scalar =
    Format.printf "%a@." (pp_result Tensor.pp)
      (eval_tensor (Pointwise.Pow.output_shape x_shape) (Q.pixel ~scalar x))
  in
  run 2.0 (* square *);
  run (-1.0) (* reciprocal *);
  run 1.5 (* generic: falls back to exp(scalar * log x) *);
  [%expect
    {|
    tensor f32 [C=3] {1, 16, 81}
    tensor f32 [C=3] {1, 0.25, 0.111111}
    tensor f32 [C=3] {1, 8, 27} |}]

let%expect_test "Direct: sub" =
  let module S = Pointwise.Sub.Compute (Direct) in
  let x_shape = s1c 3 and y_shape = s1c 3 in
  let x = Tensor.materialize x_shape (fun c -> float_of_int (chan c)) in
  let y = Tensor.materialize y_shape (fun _ -> 10.) in
  Format.printf "%a@." (pp_result Tensor.pp)
    (eval_tensor
       (Pointwise.Sub.output_shape x_shape y_shape)
       (S.pixel ~a_shape:x_shape ~b_shape:y_shape x y));
  [%expect {| tensor f32 [C=3] {-10, -9, -8} |}]

(* ---- scalar-operand and clamping pointwise ops ---------------------------- *)

let bounded = [| -3.; 0.; 2.5; 7. |]

let clamp_case (params : Pointwise.Clamp.params) =
  let module C = Pointwise.Clamp.Compute (Direct) in
  let x_shape = s1c 4 in
  let x = Tensor.materialize x_shape (fun c -> bounded.(chan c)) in
  Format.printf "%a@." (pp_result Tensor.pp)
    (eval_tensor
       (Pointwise.Clamp.output_shape params x_shape)
       (C.pixel params x))

let%expect_test "Direct: clamp both bounds" =
  clamp_case { min = Some 0.; max = Some 6. };
  [%expect {| tensor f32 [C=4] {0, 0, 2.5, 6} |}]

let%expect_test "Direct: clamp min only" =
  clamp_case { min = Some 0.; max = None };
  [%expect {| tensor f32 [C=4] {0, 0, 2.5, 7} |}]

let%expect_test "Direct: clamp max only" =
  clamp_case { min = None; max = Some 6. };
  [%expect {| tensor f32 [C=4] {-3, 0, 2.5, 6} |}]

(* ATen permits min > max and yields [max] everywhere, because its kernel is
   [min(max(a, lo), hi)] — the upper bound is applied last and wins. Reproduced
   here by applying the bounds in that same order. *)
let%expect_test "Direct: clamp reversed bounds collapse to max" =
  clamp_case { min = Some 6.; max = Some 0. };
  [%expect {| tensor f32 [C=4] {0, 0, 0, 0} |}]

(* [TORCH_IMPL_FUNC(clamp_out)] tests each scalar bound for NaN before
   dispatching to the kernel and fills the whole result with NaN if either is. *)
let%expect_test "Direct: clamp with a NaN bound is all NaN" =
  clamp_case { min = Some Float.nan; max = Some 6. };
  [%expect {| tensor f32 [C=4] {nan, nan, nan, nan} |}];
  clamp_case { min = Some 0.; max = Some Float.nan };
  [%expect {| tensor f32 [C=4] {nan, nan, nan, nan} |}]

(* A NaN *input* still propagates through the ordinary path: neither comparison
   is true for NaN, so the input flows through both selects untouched. *)
let%expect_test "Direct: clamp propagates a NaN input" =
  let module C = Pointwise.Clamp.Compute (Direct) in
  let params : Pointwise.Clamp.params = { min = Some 0.; max = Some 6. } in
  let x_shape = s1c 3 in
  let x =
    Tensor.materialize x_shape (fun c -> [| -1.; Float.nan; 9. |].(chan c))
  in
  Format.printf "%a@." (pp_result Tensor.pp)
    (eval_tensor
       (Pointwise.Clamp.output_shape params x_shape)
       (C.pixel params x));
  [%expect {| tensor f32 [C=3] {0, nan, 6} |}]

(* The both-absent configuration is rejected, as ATen's meta function rejects
   it, rather than silently behaving as the identity. *)
let%expect_test "Direct: clamp with no bounds is an error" =
  clamp_case { min = None; max = None };
  [%expect {| clamp: at least one of 'min' or 'max' must be given |}]

let%expect_test "Direct: hardtanh" =
  let module H = Pointwise.Hardtanh.Compute (Direct) in
  let x_shape = s1c 4 in
  let x = Tensor.materialize x_shape (fun c -> bounded.(chan c)) in
  let case (params : Pointwise.Hardtanh.params) =
    Format.printf "%a@." (pp_result Tensor.pp)
      (eval_tensor (Pointwise.Hardtanh.output_shape x_shape) (H.pixel params x))
  in
  (* the schema default, then MobileNet-v2's relu6 window *)
  case { min_val = -1.; max_val = 1. };
  [%expect {| tensor f32 [C=4] {-1, 0, 1, 1} |}];
  case { min_val = 0.; max_val = 6. };
  [%expect {| tensor f32 [C=4] {0, 0, 2.5, 6} |}]

let%expect_test "Direct: clone" =
  let module C = Pointwise.Clone.Compute (Direct) in
  let x_shape = s1c 4 in
  let x = Tensor.materialize x_shape (fun c -> bounded.(chan c)) in
  Format.printf "%a@." (pp_result Tensor.pp)
    (eval_tensor (Pointwise.Clone.output_shape x_shape) (C.pixel x));
  [%expect {| tensor f32 [C=4] {-3, 0, 2.5, 7} |}]

let%expect_test "Direct: add_scalar" =
  let module A = Pointwise.Add_scalar.Compute (Direct) in
  let x_shape = s1c 3 in
  let x = Tensor.materialize x_shape (fun c -> float_of_int (chan c)) in
  Format.printf "%a@." (pp_result Tensor.pp)
    (eval_tensor
       (Pointwise.Add_scalar.output_shape x_shape)
       (A.pixel ~scalar:3. x));
  [%expect {| tensor f32 [C=3] {3, 4, 5} |}]

let%expect_test "Direct: div_scalar" =
  let module D = Pointwise.Div_scalar.Compute (Direct) in
  let x_shape = s1c 3 in
  let x = Tensor.materialize x_shape (fun c -> float_of_int (chan c) +. 1.) in
  Format.printf "%a@." (pp_result Tensor.pp)
    (eval_tensor
       (Pointwise.Div_scalar.output_shape x_shape)
       (D.pixel ~scalar:6. x));
  [%expect {| tensor f32 [C=3] {0.166667, 0.333333, 0.5} |}]

(* No guard on a zero divisor, exactly as [Div] has none: the IEEE result
   stands rather than a substituted value. Avoid 0/0 here because the sign bit
   of its NaN result is platform-dependent. *)
let%expect_test "Direct: div_scalar by zero" =
  let module D = Pointwise.Div_scalar.Compute (Direct) in
  let x_shape = s1c 2 in
  let x = Tensor.materialize x_shape (fun c -> [| -1.; 1. |].(chan c)) in
  Format.printf "%a@." (pp_result Tensor.pp)
    (eval_tensor
       (Pointwise.Div_scalar.output_shape x_shape)
       (D.pixel ~scalar:0. x));
  [%expect {| tensor f32 [C=2] {-inf, inf} |}]

let%expect_test "Direct: mul_scalar" =
  let module M = Pointwise.Mul_scalar.Compute (Direct) in
  let x_shape = s1c 3 in
  let x = Tensor.materialize x_shape (fun c -> float_of_int (chan c) +. 1.) in
  Format.printf "%a@." (pp_result Tensor.pp)
    (eval_tensor
       (Pointwise.Mul_scalar.output_shape x_shape)
       (M.pixel ~scalar:3. x));
  [%expect {| tensor f32 [C=3] {3, 6, 9} |}]

(* Broadcast: [b] has an extent-1 axis (W) where [a] does not;
   [Pointwise.broadcast_coord] reads b at index 0 there, so its per-channel value
   fans out across W — and [load] is only ever handed in-bounds indices. *)
let%expect_test "Direct: add broadcasts an extent-1 axis" =
  let module A = Pointwise.Add.Compute (Direct) in
  let a_shape = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:1 ~w:2 ~c:3 in
  let b_shape = s1c 3 in
  let a =
    Tensor.materialize a_shape (fun c -> float_of_int ((col c * 3) + chan c))
  in
  let b = Tensor.materialize b_shape (fun c -> [| 10.; 20.; 30. |].(chan c)) in
  Format.printf "%a@." (pp_result Tensor.pp)
    (eval_tensor
       (Pointwise.Add.output_shape a_shape b_shape)
       (A.pixel ~a_shape ~b_shape a b));
  [%expect {| tensor f32 [W=2 C=3] {10, 21, 32, 13, 24, 35} |}]

(* Two extents that are neither equal nor 1 are incompatible — broadcasting is
   an error there, not silently the larger of the two. *)
let%expect_test "Direct: add of incompatible extents raises" =
  let a_shape = s1c 3 and b_shape = s1c 5 in
  (match Pointwise.Add.output_shape a_shape b_shape with
  | Ok _ -> print_string "no error"
  | Error e -> Format.printf "error: %a@." Shape_error.pp (Err.Error.kind e));
  [%expect {| error: incompatible broadcast extents on axis C: 3 vs 5 |}]
