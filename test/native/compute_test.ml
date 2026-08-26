let chan c = Dim.to_int (Vec6.get c Axis.C)
let row c = Dim.to_int (Vec6.get c Axis.H)
let col c = Dim.to_int (Vec6.get c Axis.W)
let bat c = Dim.to_int (Vec6.get c Axis.D)
let s1c n = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:1 ~w:1 ~c:n

let eval_tensor shape_result pixel =
  let open Err.Syntax in
  let* out_shape = shape_result in
  Err.return (Schedule.evaluate out_shape pixel)

let pp_shape_tensor ppf (out_shape, tensor) =
  Format.fprintf ppf "shape: %a@.%a" Vec6.pp_shape out_shape Tensor.pp tensor

let pp_named_shape_tensor name ppf (out_shape, tensor) =
  Format.fprintf ppf "%s: %a@.%a" name Vec6.pp_shape out_shape Tensor.pp tensor

let pp_result pp_ok = Core.Pretty.err_result ~ok:pp_ok ~error:Shape_error.pp

let conv_axis ?(pad_before = 0) ?pad_after ?(dilation = 1) ~kernel ~stride () :
    Conv.Conv2d.axis_window =
  {
    kernel = Dim.extent kernel;
    stride = Op_config.Pos.of_int stride;
    pad_before = Op_config.Nonneg.of_int pad_before;
    pad_after =
      Op_config.Nonneg.of_int (Option.value pad_after ~default:pad_before);
    dilation = Op_config.Pos.of_int dilation;
  }

let conv_params ?(groups = 1) ?(dilation = (1, 1)) ?(pad_after = None)
    ?(pad = (0, 0)) ~kernel ~stride ~in_channels () =
  let kh, kw = kernel
  and sh, sw = stride
  and ph, pw = pad
  and dh, dw = dilation in
  let pah, paw =
    match pad_after with None -> (None, None) | Some (h, w) -> (Some h, Some w)
  in
  {
    Conv.Conv2d.h =
      conv_axis ?pad_after:pah ~pad_before:ph ~dilation:dh ~kernel:kh ~stride:sh
        ();
    w =
      conv_axis ?pad_after:paw ~pad_before:pw ~dilation:dw ~kernel:kw ~stride:sw
        ();
    in_channels = Dim.extent in_channels;
    groups = Op_config.Pos.of_int groups;
  }

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

(* Deterministic value fixture shared by the Group 5 activation tests
   (op5-impl): large magnitudes for silu's sign/stability, the float32
   neighbours of both hardsigmoid/hardswish piecewise thresholds
   (x+3 = 0 and x+3 = 6), and signed zero. Finite values only. *)
let activation_fixture =
  [|
    -1e4;
    -6.;
    -3.0000005;
    -3.;
    -2.9999995;
    -0.5;
    -0.;
    0.;
    0.5;
    2.9999995;
    3.;
    3.0000005;
    6.;
    1e4;
  |]

(* [Tensor.pp] truncates after 8 elements; the 14-value activation fixture is
   deliberately wider than that, so print every element via [Tensor.read]. *)
let pp_activation x_shape ppf tensor =
  Format.fprintf ppf "{";
  let first = ref true in
  Vec6.iter x_shape (fun c ->
      if not !first then Format.fprintf ppf ", ";
      first := false;
      Format.fprintf ppf "%g" (Tensor.read tensor c));
  Format.fprintf ppf "}"

let activation_case name output_shape pixel =
  let x_shape = s1c (Array.length activation_fixture) in
  let x = Tensor.materialize x_shape (fun c -> activation_fixture.(chan c)) in
  let tensor =
    Schedule.evaluate
      (Err.or_raise ~pp_error:Shape_error.pp (output_shape x_shape))
      (pixel x)
  in
  Format.printf "%s: %a@." name (pp_activation x_shape) tensor

let%expect_test "Direct: silu" =
  let module S = Pointwise.Silu.Compute (Direct) in
  activation_case "silu" Pointwise.Silu.output_shape S.pixel;
  [%expect
    {| silu: {-0, -0.0148357, -0.142278, -0.142278, -0.142278, -0.18877, -0, 0, 0.31123, 2.85772, 2.85772, 2.85772, 5.98516, 10000} |}]

(* Mutation proof (op5.md's matrix, run and observed failing): dropping the
   leading [x] factor, [exp(x)] instead of [exp(-x)], and [1 - exp(-x)] instead
   of [1 + exp(-x)] each change this golden. *)

let%expect_test "Direct: sigmoid" =
  let module S = Pointwise.Sigmoid.Compute (Direct) in
  activation_case "sigmoid" Pointwise.Sigmoid.output_shape S.pixel;
  [%expect
    {| sigmoid: {0, 0.00247262, 0.0474259, 0.0474259, 0.0474259, 0.377541, 0.5, 0.5, 0.622459, 0.952574, 0.952574, 0.952574, 0.997527, 1} |}]

(* Saturation boundary fixture, the sigmoid counterpart of
   [gelu_boundary_fixture]: large +/- magnitude reaching into f32's denormal
   range at one tail and exact saturation at the other (Direct computes
   [exp(-x)] in OCaml's double, so nothing overflows before the f32-narrowing
   store -- -100/-1e4 land in the denormal and flush-to-zero regions
   respectively, which is the property this pins), plus near-zero and signed
   zero. Unlike gelu, sigmoid's formula has no piecewise/polynomial regime of
   its own, so this is the only boundary worth pinning here; ATen agreement
   at the same fixture is native_bridge_test.ml's job. *)
let sigmoid_boundary_fixture =
  [|
    -1e4;
    -100.;
    -88.;
    -50.;
    -20.;
    -1e-8;
    -0.;
    0.;
    1e-8;
    20.;
    50.;
    88.;
    100.;
    1e4;
  |]

let%expect_test "Direct: sigmoid boundary cases" =
  let module S = Pointwise.Sigmoid.Compute (Direct) in
  let x_shape = s1c (Array.length sigmoid_boundary_fixture) in
  let x =
    Tensor.materialize x_shape (fun c -> sigmoid_boundary_fixture.(chan c))
  in
  let tensor =
    Schedule.evaluate
      (Err.or_raise ~pp_error:Shape_error.pp
         (Pointwise.Sigmoid.output_shape x_shape))
      (S.pixel x)
  in
  Format.printf "sigmoid: %a@." (pp_activation x_shape) tensor;
  [%expect
    {| sigmoid: {0, 3.78351e-44, 6.0546e-39, 1.92875e-22, 2.06115e-09, 0.5, 0.5, 0.5, 0.5, 1, 1, 1, 1, 1} |}]

let%expect_test "Direct: hardsigmoid" =
  let module H = Pointwise.Hardsigmoid.Compute (Direct) in
  activation_case "hardsigmoid" Pointwise.Hardsigmoid.output_shape H.pixel;
  [%expect
    {| hardsigmoid: {0, 0, 0, 0, 7.94729e-08, 0.416667, 0.5, 0.5, 0.583333, 1, 1, 1, 1, 1} |}]

(* Both float32 threshold neighbourhoods (x+3 = 0 at index 2-4, x+3 = 6 at
   index 9-11) are pinned above. Mutation proof: moving either clamp bound,
   dividing before clamping, swapping the select arms, or using 5 instead of 6
   each change this golden. *)

let%expect_test "Direct: hardswish" =
  let module H = Pointwise.Hardswish.Compute (Direct) in
  activation_case "hardswish" Pointwise.Hardswish.output_shape H.pixel;
  [%expect
    {| hardswish: {-0, -0, -0, -0, -2.38419e-07, -0.208333, -0, 0, 0.291667, 3, 3, 3, 6, 10000} |}]

(* Mutation proof: dropping the [* x] factor changes this golden (observed).
   Reassociating to [x * (c / 6)] instead of [(x * c) / 6] does NOT change it
   at this fixture, checked by hand up to full double precision (op5-impl) —
   Direct's [Compute] functor runs entirely in OCaml [float] (double) with no
   per-node f32 narrowing between [clamp]/[div]/[mul], so the two association
   orders round identically here; op5.md's rounding hazard is about a
   DECOMPOSED 3-node graph (clamp, div_scalar, mul as separately materialized
   f32 nodes), which this retained fused op deliberately avoids being. The
   pinned association still matches the ATen kernel's literal source order —
   that is why it is pinned — but it is not this test's job to prove the
   order is load-bearing at Direct's precision, only that it matches ATen's
   formula, and [test/native_bridge_test.ml] holds it against real ATen. *)

let%expect_test "Direct: gelu" =
  let module G = Pointwise.Gelu.Compute (Direct) in
  activation_case "gelu" Pointwise.Gelu.output_shape
    (G.pixel Pointwise.Gelu.Exact);
  [%expect
    {| gelu: {-0, -5.94073e-09, -0.0040499, -0.0040499, -0.00404991, -0.154269, -0, 0, 0.345731, 2.99595, 2.99595, 2.99595, 6, 10000} |}]

let%expect_test "Direct: gelu tanh" =
  let module G = Pointwise.Gelu.Compute (Direct) in
  activation_case "gelu tanh" Pointwise.Gelu.output_shape
    (G.pixel Pointwise.Gelu.Tanh);
  [%expect
    {| gelu tanh: {-0, -8.43965e-11, -0.00363739, -0.00363739, -0.0036374, -0.154286, -0, 0, 0.345714, 2.99636, 2.99636, 2.99636, 6, 10000} |}]

(* [erf] saturates to +/-1 well before |x|=20, so the negative tail should
   read as (numerically) exact zero and the positive tail as (numerically)
   exact identity; signed zero and small |x| exercise the polynomial's [t]
   term near its own regime rather than the saturated one. *)
let gelu_boundary_fixture =
  [| -20.; -5.; -1.; -0.5; -0.1; -1e-8; -0.; 0.; 1e-8; 0.1; 0.5; 1.; 5.; 20. |]

let%expect_test "Direct: gelu boundary cases" =
  let module G = Pointwise.Gelu.Compute (Direct) in
  let x_shape = s1c (Array.length gelu_boundary_fixture) in
  let x =
    Tensor.materialize x_shape (fun c -> gelu_boundary_fixture.(chan c))
  in
  let tensor =
    Schedule.evaluate
      (Err.or_raise ~pp_error:Shape_error.pp
         (Pointwise.Gelu.output_shape x_shape))
      (G.pixel Pointwise.Gelu.Exact x)
  in
  Format.printf "gelu: %a@." (pp_activation x_shape) tensor;
  [%expect
    {| gelu: {-0, -1.43553e-06, -0.158655, -0.154269, -0.0460172, -5e-09, -0, 0, 5e-09, 0.0539828, 0.345731, 0.841345, 5, 20} |}]

let%expect_test "Direct: gelu tanh boundary cases" =
  let module G = Pointwise.Gelu.Compute (Direct) in
  let x_shape = s1c (Array.length gelu_boundary_fixture) in
  let x =
    Tensor.materialize x_shape (fun c -> gelu_boundary_fixture.(chan c))
  in
  let tensor =
    Schedule.evaluate
      (Err.or_raise ~pp_error:Shape_error.pp
         (Pointwise.Gelu.output_shape x_shape))
      (G.pixel Pointwise.Gelu.Tanh x)
  in
  Format.printf "gelu tanh: %a@." (pp_activation x_shape) tensor;
  [%expect
    {| gelu tanh: {-0, -2.2918e-07, -0.158808, -0.154286, -0.0460172, -5e-09, -0, 0, 5e-09, 0.0539828, 0.345714, 0.841192, 5, 20} |}]

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

let%expect_test "Direct: conv2d 2x2 box filter (stride 1, no pad)" =
  let module Cv = Conv.Conv2d.Compute (Direct) in
  (* 3x3 single-channel input, value(h,w) = h*3 + w *)
  let x_shape = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:3 ~w:3 ~c:1 in
  let x =
    Tensor.materialize x_shape (fun c -> float_of_int ((row c * 3) + col c))
  in
  (* Cout=1, 2x2 kernel, Cin=1, all ones -> sum of each 2x2 window *)
  let weight_shape = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:2 ~w:2 ~c:1 in
  let weight = Tensor.materialize weight_shape (fun _ -> 1.) in
  let bias = Tensor.materialize (s1c 1) (fun _ -> 0.) in
  let p = conv_params ~kernel:(2, 2) ~stride:(1, 1) ~in_channels:1 () in
  Format.printf "%a@." (pp_result Tensor.pp)
    (eval_tensor
       (Conv.Conv2d.output_shape ~x_shape ~weight_shape p)
       (Cv.pixel p ~x_shape ~weight_shape ~x ~weight ~bias));
  (* windows: 0+1+3+4=8, 1+2+4+5=12, 3+4+6+7=20, 4+5+7+8=24 *)
  [%expect {| tensor f32 [H=2 W=2 C=1] {8, 12, 20, 24} |}]

let%expect_test
    "Direct: conv2d 3x3 same padding (pad=1) — exercises bound clipping" =
  let module Cv = Conv.Conv2d.Compute (Direct) in
  (* 3x3 single-channel input, value(h,w) = h*3 + w; every output pixel's
     window runs off the edge on at least one side, so this only gets the
     right answer if the clipped kh/kw range — not a guarded read — is what
     keeps the padding region at 0. *)
  let x_shape = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:3 ~w:3 ~c:1 in
  let x =
    Tensor.materialize x_shape (fun c -> float_of_int ((row c * 3) + col c))
  in
  let weight_shape = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:3 ~w:3 ~c:1 in
  let weight = Tensor.materialize weight_shape (fun _ -> 1.) in
  let bias = Tensor.materialize (s1c 1) (fun _ -> 0.) in
  let p =
    conv_params ~kernel:(3, 3) ~stride:(1, 1) ~pad:(1, 1) ~in_channels:1 ()
  in
  Format.printf "%a@." (pp_result Tensor.pp)
    (eval_tensor
       (Conv.Conv2d.output_shape ~x_shape ~weight_shape p)
       (Cv.pixel p ~x_shape ~weight_shape ~x ~weight ~bias));
  (* hand-computed 3x3 zero-padded box sums over [0..8] laid out row-major *)
  [%expect {| tensor f32 [H=3 W=3 C=1] {8, 15, 12, 21, 36, 27, 20, 33, ...} |}]

let%expect_test "Direct: grouped conv2d reduces only within each channel group"
    =
  let module Cv = Conv.Conv2d.Compute (Direct) in
  let x_shape = s1c 4 in
  let x =
    Tensor.materialize x_shape (fun c -> [| 1.; 2.; 10.; 20. |].(chan c))
  in
  let weight_shape = Vec6.shape ~n:4 ~t:1 ~d:1 ~h:1 ~w:1 ~c:2 in
  let weight =
    Tensor.materialize weight_shape (fun c ->
        match (Dim.to_int (Vec6.get c Axis.N), chan c) with
        | 0, 0 | 0, 1 -> 1.
        | 1, 0 -> 10.
        | 2, 0 | 2, 1 -> 1.
        | 3, 1 -> 2.
        | _ -> 0.)
  in
  let bias =
    Tensor.materialize (s1c 4) (fun c -> [| 0.; 100.; 1000.; 10000. |].(chan c))
  in
  let p =
    conv_params ~groups:2 ~kernel:(1, 1) ~stride:(1, 1) ~in_channels:4 ()
  in
  Format.printf "%a@." (pp_result Tensor.pp)
    (eval_tensor
       (Conv.Conv2d.output_shape ~x_shape ~weight_shape p)
       (Cv.pixel p ~x_shape ~weight_shape ~x ~weight ~bias));
  [%expect {| tensor f32 [C=4] {3, 110, 1030, 10040} |}]

let%expect_test
    "Direct: depthwise conv2d as grouped conv with channel multiplier" =
  let module Cv = Conv.Conv2d.Compute (Direct) in
  let x_shape = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:1 ~w:2 ~c:2 in
  let x =
    Tensor.materialize x_shape (fun c ->
        match (col c, chan c) with
        | 0, 0 -> 1.
        | 0, 1 -> 10.
        | 1, 0 -> 2.
        | 1, 1 -> 20.
        | _ -> assert false)
  in
  let weight_shape = Vec6.shape ~n:4 ~t:1 ~d:1 ~h:1 ~w:1 ~c:1 in
  let weight =
    Tensor.materialize weight_shape (fun c ->
        [| 1.; 2.; 3.; 4. |].(Dim.to_int (Vec6.get c Axis.N)))
  in
  let bias = Tensor.materialize (s1c 4) (fun _ -> 0.) in
  let p =
    conv_params ~groups:2 ~kernel:(1, 1) ~stride:(1, 1) ~in_channels:2 ()
  in
  Format.printf "%a@." (pp_result Tensor.pp)
    (eval_tensor
       (Conv.Conv2d.output_shape ~x_shape ~weight_shape p)
       (Cv.pixel p ~x_shape ~weight_shape ~x ~weight ~bias));
  [%expect {| tensor f32 [W=2 C=4] {1, 2, 30, 40, 2, 4, 60, 80} |}]

let%expect_test "Direct: conv2d asymmetric padding with dilation" =
  let module Cv = Conv.Conv2d.Compute (Direct) in
  let x_shape = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:1 ~w:5 ~c:1 in
  let x = Tensor.materialize x_shape (fun c -> float_of_int (col c)) in
  let weight_shape = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:1 ~w:3 ~c:1 in
  let weight = Tensor.materialize weight_shape (fun _ -> 1.) in
  let bias = Tensor.materialize (s1c 1) (fun _ -> 0.) in
  let p =
    conv_params ~dilation:(1, 2) ~pad:(0, 1)
      ~pad_after:(Some (0, 2))
      ~kernel:(1, 3) ~stride:(1, 1) ~in_channels:1 ()
  in
  Format.printf "%a@." (pp_result Tensor.pp)
    (eval_tensor
       (Conv.Conv2d.output_shape ~x_shape ~weight_shape p)
       (Cv.pixel p ~x_shape ~weight_shape ~x ~weight ~bias));
  [%expect {| tensor f32 [W=4 C=1] {4, 6, 4, 6} |}]

(* [Conv2d_padding.same_padding] splits an ODD total unevenly — [total / 2]
   before and [total - total / 2] after — and nothing else pins which side gets
   the extra cell. Direct-vs-Symbolic agreement cannot: both resolve through the
   same function, so they agree on a reversed split as readily as on the right
   one. Neither can an output-shape check: reversing the split leaves the extent
   untouched. Nor can the ATen oracle, which reaches [constant_pad_nd] for this
   case and is not in this repository's minimal static-dispatch build.

   So the numbers here are hand-computed, and the fixture is built so they
   differ. A 1x2 kernel over a 1x3 row gives total = 1: pad_before = 0,
   pad_after = 1. With the weight [0; 1] each output reads the cell to its
   RIGHT, so y = [x1; x2; pad] = {2, 3, 0}. Reverse the split and every output
   reads itself instead, giving {1, 2, 3}. *)
let%expect_test "Direct: conv2d_padding same splits an odd total to the right" =
  let module Cv = Conv.Conv2d_padding.Compute (Direct) in
  let x_shape = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:1 ~w:3 ~c:1 in
  let x = Tensor.materialize x_shape (fun c -> float_of_int (col c + 1)) in
  let weight_shape = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:1 ~w:2 ~c:1 in
  let weight =
    Tensor.materialize weight_shape (fun c -> if col c = 0 then 0. else 1.)
  in
  let bias = Tensor.materialize (s1c 1) (fun _ -> 0.) in
  let p =
    {
      Conv.Conv2d_padding.stride =
        Op_config.Hw.{ h = Op_config.Pos.of_int 1; w = Op_config.Pos.of_int 1 };
      padding = Conv.Conv2d_padding.Same;
      dilation =
        Op_config.Hw.{ h = Op_config.Pos.of_int 1; w = Op_config.Pos.of_int 1 };
      groups = Op_config.Pos.of_int 1;
    }
  in
  Format.printf "%a@." (pp_result Tensor.pp)
    (eval_tensor
       (Conv.Conv2d_padding.output_shape ~x_shape ~weight_shape p)
       (Cv.pixel p ~x_shape ~weight_shape ~x ~weight ~bias));
  [%expect {| tensor f32 [W=3 C=1] {2, 3, 0} |}]

let%expect_test
    "Direct: max_pool2d 2x2 (stride 1, pad 1) — negative inputs catch a \
     padding=0 bug" =
  let module P = Pool.MaxPool2d.Compute (Direct) in
  (* All-negative input: if the padding region wrongly contributed 0 (as a
     naive guarded-load fallback would), 0 would beat every real value here
     and the result would be wrong everywhere a window touches the border —
     i.e. everywhere, since pad=1 on a 3x3 input with a 2x2 kernel. *)
  let x_shape = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:3 ~w:3 ~c:1 in
  let x =
    Tensor.materialize x_shape (fun c ->
        float_of_int (-((row c * 3) + col c) - 1))
  in
  let p =
    {
      Pool.MaxPool2d.kernel =
        Op_config.Hw.{ h = Dim.extent 2; w = Dim.extent 2 };
      stride =
        Op_config.Hw.{ h = Op_config.Pos.of_int 1; w = Op_config.Pos.of_int 1 };
      pad =
        Op_config.Hw.
          { h = Op_config.Nonneg.of_int 1; w = Op_config.Nonneg.of_int 1 };
    }
  in
  Format.printf "%a@." (pp_result Tensor.pp)
    (eval_tensor
       (Pool.MaxPool2d.output_shape ~x_shape p)
       (P.pixel p ~x_shape ~x));
  (* hand-computed: input(h,w) = -(h*3+w)-1, i.e. -1..-9 row-major; each
     output position's max over its real (non-padding) taps only *)
  [%expect {| tensor f32 [H=4 W=4 C=1] {-1, -1, -2, -3, -1, -1, -2, -3, ...} |}]

let%expect_test "Direct: max_pool2d_with_indices 2x2 stride 2 — values + argmax"
    =
  let module M = Pool.MaxPool2dWithIndices.Compute (Direct) in
  (* 4x4 input, value(h,w) = h*4 + w (0..15). Each 2x2 window's max is its
     bottom-right corner; the argmax flat index is ih*4 + iw of that corner. *)
  let x_shape = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:4 ~w:4 ~c:1 in
  let x =
    Tensor.materialize x_shape (fun c -> float_of_int ((row c * 4) + col c))
  in
  let p =
    {
      Pool.MaxPool2d.kernel =
        Op_config.Hw.{ h = Dim.extent 2; w = Dim.extent 2 };
      stride =
        Op_config.Hw.{ h = Op_config.Pos.of_int 2; w = Op_config.Pos.of_int 2 };
      pad =
        Op_config.Hw.
          { h = Op_config.Nonneg.of_int 0; w = Op_config.Nonneg.of_int 0 };
    }
  in
  Format.printf "values:  %a@." (pp_result Tensor.pp)
    (eval_tensor
       (Pool.MaxPool2dWithIndices.output_shape ~x_shape p)
       (M.value_pixel p ~x_shape ~x));
  Format.printf "indices: %a@." (pp_result Tensor.pp)
    (eval_tensor
       (Pool.MaxPool2dWithIndices.output_shape ~x_shape p)
       (M.index_pixel p ~x_shape ~x));
  [%expect
    {|
    values:  tensor f32 [H=2 W=2 C=1] {5, 7, 13, 15}
    indices: tensor f32 [H=2 W=2 C=1] {5, 7, 13, 15} |}]

let%expect_test
    "Direct: max_pool2d_with_indices ties choose smallest flat index" =
  let module M = Pool.MaxPool2dWithIndices.Compute (Direct) in
  let x_shape = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:2 ~w:2 ~c:1 in
  let x = Tensor.materialize x_shape (fun _ -> 7.) in
  let p =
    {
      Pool.MaxPool2d.kernel =
        Op_config.Hw.{ h = Dim.extent 2; w = Dim.extent 2 };
      stride =
        Op_config.Hw.{ h = Op_config.Pos.of_int 2; w = Op_config.Pos.of_int 2 };
      pad =
        Op_config.Hw.
          { h = Op_config.Nonneg.of_int 0; w = Op_config.Nonneg.of_int 0 };
    }
  in
  Format.printf "%a@." (pp_result Tensor.pp)
    (eval_tensor
       (Pool.MaxPool2dWithIndices.output_shape ~x_shape p)
       (M.index_pixel p ~x_shape ~x));
  [%expect {| tensor f32 [C=1] {0} |}]

(* ---- max-pool signed zero / NaN, against ATen -----------------------------

   [Tensor.pp] renders -0. as "-0" but every NaN as "nan", so it cannot show
   WHICH NaN survived — and these tests turn on exactly those bits. Print the
   f32 storage pattern instead. *)

let bits_of tensor =
  let (Tensor.Tensor t) = tensor in
  Vec6.fold_coords t.Tensor.shape ~init:[] ~f:(fun acc c ->
      Printf.sprintf "%08lx" (Int32.bits_of_float (Tensor.read tensor c)) :: acc)
  |> List.rev |> String.concat " "

let pp_bits ppf tensor = Format.pp_print_string ppf (bits_of tensor)

(* A 1xN window, so one output pixel reduces the whole row. *)
let row_pool n =
  {
    Pool.MaxPool2d.kernel = Op_config.Hw.{ h = Dim.extent 1; w = Dim.extent n };
    stride =
      Op_config.Hw.{ h = Op_config.Pos.of_int 1; w = Op_config.Pos.of_int n };
    pad =
      Op_config.Hw.
        { h = Op_config.Nonneg.of_int 0; w = Op_config.Nonneg.of_int 0 };
  }

let row_input values =
  let n = Array.length values in
  let x_shape = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:1 ~w:n ~c:1 in
  (x_shape, Tensor.materialize x_shape (fun c -> values.(col c)))

(* Two NaNs the same class but distinct payloads: identical payloads would make
   "the last NaN wins" unobservable. *)
let nan1 = Int32.float_of_bits 0x7FC00001l
let nan2 = Int32.float_of_bits 0x7FC00002l

let%expect_test "Direct: max_pool2d keeps the incumbent on a signed-zero tie" =
  (* ATen selects on [(val > maxval) || isnan(val)] (MaxPoolKernel.cpp:215), and
     [-0. > +0.] and [+0. > -0.] are both false, so a signed-zero tie keeps
     whichever came first. [Float.max] would answer +0. either way, which is the
     bug this pins. *)
  let module P = Pool.MaxPool2d.Compute (Direct) in
  List.iter
    (fun (name, values) ->
      let x_shape, x = row_input values in
      let p = row_pool 2 in
      Format.printf "%s: %a@." name (pp_result pp_bits)
        (eval_tensor
           (Pool.MaxPool2d.output_shape ~x_shape p)
           (P.pixel p ~x_shape ~x)))
    [ ("-0 then +0", [| -0.; 0. |]); ("+0 then -0", [| 0.; -0. |]) ];
  [%expect {|
    -0 then +0: 80000000
    +0 then -0: 00000000 |}]

let%expect_test
    "Direct: max_pool2d_with_indices — the last NaN wins, with its index" =
  (* Every NaN re-satisfies the predicate, so the last one in iteration order
     ends up in both the value and the index. Before [Max_op] the value
     propagated a NaN (via [Float.max]) while the index did not (strict [>]),
     so the two disagreed about which element won. *)
  let module M = Pool.MaxPool2dWithIndices.Compute (Direct) in
  let x_shape, x = row_input [| nan1; 5.; nan2; 7. |] in
  let p = row_pool 4 in
  let shape = Pool.MaxPool2dWithIndices.output_shape ~x_shape p in
  Format.printf "value: %a@." (pp_result pp_bits)
    (eval_tensor shape (M.value_pixel p ~x_shape ~x));
  Format.printf "index: %a@." (pp_result Tensor.pp)
    (eval_tensor shape (M.index_pixel p ~x_shape ~x));
  [%expect {|
    value: 7fc00002
    index: tensor f32 [C=1] {2} |}]

let%expect_test "Direct: reshape [H=2 W=3 C=1] -> [W=3 C=2] (contiguous)" =
  let module R = Reshape.Reshape.Compute (Direct) in
  (* row-major elements 0..5; a contiguous reshape reinterprets the same flat
     buffer, so the values are unchanged, only the shape differs. *)
  let x_shape = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:2 ~w:3 ~c:1 in
  let x =
    Tensor.materialize x_shape (fun c -> float_of_int ((row c * 3) + col c))
  in
  let p =
    { Reshape.Reshape.shape = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:1 ~w:3 ~c:2 }
  in
  Format.printf "%a@." (pp_result Tensor.pp)
    (eval_tensor
       (Reshape.Reshape.output_shape ~x_shape p)
       (R.pixel p ~x_shape ~x));
  [%expect {| tensor f32 [W=3 C=2] {0, 1, 2, 3, 4, 5} |}]

let%expect_test
    "Direct: reshape rejects a target that changes the element count" =
  (* Numel 6 either way for the valid case above; here the target's numel (8)
     disagrees with the source's (6). *)
  let x_shape = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:2 ~w:3 ~c:1 in
  let bad_target = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:1 ~w:4 ~c:2 in
  Format.printf "%a@." (pp_result Vec6.pp_shape)
    (Reshape.Reshape.output_shape ~x_shape
       { Reshape.Reshape.shape = bad_target });
  [%expect
    {| reshape target [W=4 C=2] does not preserve the element count of [H=2 W=3 C=1] |}]

let%expect_test "Direct: avg_pool2d 2x2 (stride 1, pad 0) — box-filter average"
    =
  let module P = Pool.AvgPool2d.Compute (Direct) in
  (* Same 3x3 input/kernel as the conv2d box-filter test, whose hand-computed
     window sums were {8,12,20,24}; dividing by the kernel area (4) is the
     cross-check here. *)
  let x_shape = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:3 ~w:3 ~c:1 in
  let x =
    Tensor.materialize x_shape (fun c -> float_of_int ((row c * 3) + col c))
  in
  let p =
    {
      Pool.AvgPool2d.kernel =
        Op_config.Hw.{ h = Dim.extent 2; w = Dim.extent 2 };
      stride =
        Op_config.Hw.{ h = Op_config.Pos.of_int 1; w = Op_config.Pos.of_int 1 };
      pad =
        Op_config.Hw.
          { h = Op_config.Nonneg.of_int 0; w = Op_config.Nonneg.of_int 0 };
    }
  in
  Format.printf "%a@." (pp_result Tensor.pp)
    (eval_tensor
       (Pool.AvgPool2d.output_shape ~x_shape p)
       (P.pixel p ~x_shape ~x));
  [%expect {| tensor f32 [H=2 W=2 C=1] {2, 3, 5, 6} |}]

let%expect_test
    "Direct: avg_pool2d 2x2 (stride 1, pad 1) — count_include_pad=true divides \
     by the full kernel area" =
  (* A constant-1 2x2 input: every real tap contributes 1, so the result at
     each output position is exactly (# real taps in its window) / 4 — this
     is what makes count_include_pad=true visible (a count_include_pad=false
     pooling would give 1 everywhere instead). *)
  let module P = Pool.AvgPool2d.Compute (Direct) in
  let x_shape = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:2 ~w:2 ~c:1 in
  let x = Tensor.materialize x_shape (fun _ -> 1.) in
  let p =
    {
      Pool.AvgPool2d.kernel =
        Op_config.Hw.{ h = Dim.extent 2; w = Dim.extent 2 };
      stride =
        Op_config.Hw.{ h = Op_config.Pos.of_int 1; w = Op_config.Pos.of_int 1 };
      pad =
        Op_config.Hw.
          { h = Op_config.Nonneg.of_int 1; w = Op_config.Nonneg.of_int 1 };
    }
  in
  Format.printf "%a@." (pp_result Tensor.pp)
    (eval_tensor
       (Pool.AvgPool2d.output_shape ~x_shape p)
       (P.pixel p ~x_shape ~x));
  [%expect
    {| tensor f32 [H=3 W=3 C=1] {0.25, 0.5, 0.25, 0.5, 1, 0.5, 0.25, 0.5, ...} |}]

let%expect_test
    "Direct: adaptive_avg_pool2d uses ATen's overlapping non-divisible bins" =
  let module P = Pool.AdaptiveAvgPool2d.Compute (Direct) in
  let x_shape = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:5 ~w:5 ~c:1 in
  let x =
    Tensor.materialize x_shape (fun c -> float_of_int ((row c * 5) + col c + 1))
  in
  let p =
    {
      Pool.AdaptiveAvgPool2d.output_size =
        Op_config.Hw.{ h = Op_config.Pos.of_int 3; w = Op_config.Pos.of_int 3 };
    }
  in
  Format.printf "%a@." (pp_result Tensor.pp)
    (eval_tensor
       (Pool.AdaptiveAvgPool2d.output_shape ~x_shape p)
       (P.pixel p ~x_shape ~x));
  [%expect
    {| tensor f32 [H=3 W=3 C=1] {4, 5.5, 7, 11.5, 13, 14.5, 19, 20.5, ...} |}]

let%expect_test "Direct: adaptive_avg_pool2d global [1,1]" =
  let module P = Pool.AdaptiveAvgPool2d.Compute (Direct) in
  let x_shape = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:4 ~w:4 ~c:1 in
  let x =
    Tensor.materialize x_shape (fun c -> float_of_int ((row c * 4) + col c + 1))
  in
  let p =
    {
      Pool.AdaptiveAvgPool2d.output_size =
        Op_config.Hw.{ h = Op_config.Pos.of_int 1; w = Op_config.Pos.of_int 1 };
    }
  in
  Format.printf "%a@." (pp_result Tensor.pp)
    (eval_tensor
       (Pool.AdaptiveAvgPool2d.output_shape ~x_shape p)
       (P.pixel p ~x_shape ~x));
  [%expect {| tensor f32 [C=1] {8.5} |}]

let%expect_test
    "adaptive_avg_pool2d rejects an index-scale aggregate before allocation" =
  let x_shape = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:(1 lsl 30) ~w:1 ~c:1 in
  let p =
    {
      Pool.AdaptiveAvgPool2d.output_size =
        Op_config.Hw.{ h = Op_config.Pos.of_int 2; w = Op_config.Pos.of_int 1 };
    }
  in
  Format.printf "%a@." (pp_result Vec6.pp_shape)
    (Pool.AdaptiveAvgPool2d.output_shape ~x_shape p);
  [%expect
    {| adaptive_avg_pool2d: input extent 1073741824 times output_size 2 on axis H is 2147483648, which must be below the engine maximum of 2147483648 |}]

let%expect_test "Direct: linear (addmm) — out_features mix in_features" =
  let module L = Linear.Linear.Compute (Direct) in
  let chan c = Dim.to_int (Vec6.get c Axis.C) in
  let x_shape = s1c 3 in
  let weight_shape = Vec6.shape ~n:2 ~t:1 ~d:1 ~h:1 ~w:1 ~c:3 in
  let bias_shape = s1c 2 in
  let x = Tensor.materialize x_shape (fun c -> [| 1.; 2.; 3. |].(chan c)) in
  (* weight[0,:] selects x0; weight[1,:] sums x1+x2 *)
  let weight =
    Tensor.materialize weight_shape (fun c ->
        match (Dim.to_int (Vec6.get c Axis.N), chan c) with
        | 0, 0 -> 1.
        | 1, 1 | 1, 2 -> 1.
        | _ -> 0.)
  in
  let bias =
    Tensor.materialize bias_shape (fun c -> [| 10.; 100. |].(chan c))
  in
  let p = { Linear.Linear.in_features = Dim.extent 3 } in
  Format.printf "%a@." (pp_result Tensor.pp)
    (eval_tensor
       (Linear.Linear.output_shape p ~x_shape ~weight_shape)
       (L.pixel p ~x ~weight ~bias));
  [%expect {| tensor f32 [C=2] {11, 105} |}]

let%expect_test
    "Direct: bmm — B=2 batch matrix multiply, exercises batch axis isolation" =
  let module B = Matmul.Bmm.Compute (Direct) in
  (* input[H=2, W=2, C=3]: two 2×3 matrices laid out H=batch, W=row, C=inner.
     batch 0: [[1,2,3],[4,5,6]]   batch 1: [[1,0,0],[0,1,0]] *)
  let input_shape = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:2 ~w:2 ~c:3 in
  let mat0 = [| [| 1.; 2.; 3. |]; [| 4.; 5.; 6. |] |] in
  let mat1 = [| [| 1.; 0.; 0. |]; [| 0.; 1.; 0. |] |] in
  let input =
    Tensor.materialize input_shape (fun c ->
        let b = Dim.to_int (Vec6.get c Axis.H) in
        let r = Dim.to_int (Vec6.get c Axis.W) in
        let k = Dim.to_int (Vec6.get c Axis.C) in
        [| mat0; mat1 |].(b).(r).(k))
  in
  (* mat2[H=2, W=3, C=2]: two 3×2 matrices laid out H=batch, W=inner, C=col.
     batch 0: [[1,0],[0,1],[1,1]]   batch 1: [[2,0],[0,2],[0,0]] *)
  let mat2_shape = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:2 ~w:3 ~c:2 in
  let m2_0 = [| [| 1.; 0. |]; [| 0.; 1. |]; [| 1.; 1. |] |] in
  let m2_1 = [| [| 2.; 0. |]; [| 0.; 2. |]; [| 0.; 0. |] |] in
  let mat2 =
    Tensor.materialize mat2_shape (fun c ->
        let b = Dim.to_int (Vec6.get c Axis.H) in
        let k = Dim.to_int (Vec6.get c Axis.W) in
        let j = Dim.to_int (Vec6.get c Axis.C) in
        [| m2_0; m2_1 |].(b).(k).(j))
  in
  Format.printf "%a@." (pp_result Tensor.pp)
    (eval_tensor
       (Matmul.Bmm.output_shape ~input_shape ~mat2_shape)
       (B.pixel ~input_shape ~input ~mat2));
  (* batch 0: [1,2,3]·[[1,0],[0,1],[1,1]] = [4,5]; [4,5,6]·same = [10,11]
     batch 1: [1,0,0]·[[2,0],[0,2],[0,0]] = [2,0]; [0,1,0]·same = [0,2] *)
  [%expect {| tensor f32 [H=2 W=2 C=2] {4, 5, 10, 11, 2, 0, 0, 2} |}]

let%expect_test "windowed axis: output_extent and window agree" =
  let module Wa = Window_axis.Compute (Direct) in
  (* The property that's actually guaranteed (not "the window becomes empty
     right at output_extent" — it can still have a stray valid tap one
     position past, depending on config, so that's not asserted here): for
     every out in [0, output_extent), the window names a genuinely non-empty
     range of real (non-padding) taps. If output_extent and the window
     disagreed about the windowed-axis relationship, some output position
     would have an empty window and this would fail. *)
  let check ~in_extent ~kernel ~stride ~pad =
    let kernel = Dim.extent kernel
    and stride = Op_config.Pos.of_int stride
    and pad = Op_config.Nonneg.of_int pad
    and in_extent = Dim.extent in_extent in
    let out_extent =
      Window_axis.output_extent ~kernel ~stride ~pad_before:pad ~pad_after:pad
        ~dilation:(Op_config.Pos.of_int 1) ~in_extent
      |> Err.or_raise ~pp_error:Shape_error.pp
    in
    let non_empty out =
      let w =
        Wa.window ~kernel ~stride ~pad_before:pad
          ~dilation:(Op_config.Pos.of_int 1) ~in_extent out
      in
      (w.lo :> int) < (w.hi :> int)
    in
    let all_in_range =
      List.for_all non_empty
        (List.init (out_extent :> int) (fun i -> Dim.index i))
    in
    Format.printf
      "in_extent=%d kernel=%d stride=%d pad=%d -> output_extent=%d, \
       all_in_range=%b@."
      (in_extent :> int)
      (kernel :> int)
      (stride :> int)
      (pad :> int)
      (out_extent :> int)
      all_in_range
  in
  check ~in_extent:3 ~kernel:2 ~stride:1 ~pad:0;
  check ~in_extent:3 ~kernel:3 ~stride:1 ~pad:1;
  check ~in_extent:7 ~kernel:3 ~stride:2 ~pad:1;
  check ~in_extent:5 ~kernel:1 ~stride:1 ~pad:0;
  [%expect
    {|
    in_extent=3 kernel=2 stride=1 pad=0 -> output_extent=2, all_in_range=true
    in_extent=3 kernel=3 stride=1 pad=1 -> output_extent=3, all_in_range=true
    in_extent=7 kernel=3 stride=2 pad=1 -> output_extent=4, all_in_range=true
    in_extent=5 kernel=1 stride=1 pad=0 -> output_extent=5, all_in_range=true
    |}]

let%expect_test "Direct: mean over spatial (H,W), per channel" =
  let module M = Reduce.Mean.Compute (Direct) in
  (* C passes through, H/W reduce: channel 0 is [1,2,3,4] (mean 2.5), channel 1
     is 10x that (mean 25). *)
  let x_shape = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:2 ~w:2 ~c:2 in
  let x =
    Tensor.materialize x_shape (fun c ->
        let base = [| 1.; 2.; 3.; 4. |].((row c * 2) + col c) in
        if chan c = 0 then base else base *. 10.)
  in
  let p = { Reduce.Mean.dims = [ Axis.H; Axis.W ]; keepdim = true } in
  Format.printf "%a@." (pp_result Tensor.pp)
    (eval_tensor (Reduce.Mean.output_shape ~x_shape p) (M.pixel p ~x_shape ~x));
  [%expect {| tensor f32 [C=2] {2.5, 25} |}]

let%expect_test
    "Mean output_shape: keepdim true collapses in place, false shifts" =
  (* input [H6 W7 C8] (rank 3); mean over W. keepdim=true leaves W at 1 in place;
     keepdim=false removes W and re-packs H's data onto W (§1d). *)
  let x_shape = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:6 ~w:7 ~c:8 in
  let shape ~keepdim =
    Reduce.Mean.output_shape ~x_shape { Reduce.Mean.dims = [ Axis.W ]; keepdim }
    |> Err.or_raise ~pp_error:Shape_error.pp
  in
  Format.printf "keepdim=true:  %a@." Vec6.pp_shape (shape ~keepdim:true);
  Format.printf "keepdim=false: %a@." Vec6.pp_shape (shape ~keepdim:false);
  [%expect {|
    keepdim=true:  [H=6 W=1 C=8]
    keepdim=false: [W=6 C=8] |}]

let%expect_test "Direct: mean over W, keepdim=false shifts H's data onto W" =
  let module M = Reduce.Mean.Compute (Direct) in
  (* [H2 W2 C1]; value(h,w): row H0 = [1,3] (mean 2), row H1 = [5,7] (mean 6). *)
  let x_shape = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:2 ~w:2 ~c:1 in
  let x =
    Tensor.materialize x_shape (fun c ->
        [| [| 1.; 3. |]; [| 5.; 7. |] |].(row c).(col c))
  in
  let run ~keepdim =
    let p = { Reduce.Mean.dims = [ Axis.W ]; keepdim } in
    Format.printf "%a@." (pp_result Tensor.pp)
      (eval_tensor
         (Reduce.Mean.output_shape ~x_shape p)
         (M.pixel p ~x_shape ~x))
  in
  run ~keepdim:true;
  run ~keepdim:false;
  (* same means {2, 6}; keepdim=true keeps them on H, keepdim=false moves them to W. *)
  [%expect
    {|
    tensor f32 [H=2 W=1 C=1] {2, 6}
    tensor f32 [W=2 C=1] {2, 6} |}]

let%expect_test "Direct: amax over spatial (H,W), per channel" =
  let module M = Reduce.Amax.Compute (Direct) in
  (* Same input as the mean fixture above: channel 0 is [1,2,3,4] (max 4),
     channel 1 is 10x that (max 40). *)
  let x_shape = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:2 ~w:2 ~c:2 in
  let x =
    Tensor.materialize x_shape (fun c ->
        let base = [| 1.; 2.; 3.; 4. |].((row c * 2) + col c) in
        if chan c = 0 then base else base *. 10.)
  in
  let p = { Reduce.Amax.dims = [ Axis.H; Axis.W ]; keepdim = true } in
  Format.printf "%a@." (pp_result Tensor.pp)
    (eval_tensor (Reduce.Amax.output_shape ~x_shape p) (M.pixel p ~x_shape ~x));
  [%expect {| tensor f32 [C=2] {4, 40} |}]

let%expect_test "Direct: amax over W, keepdim=false shifts H's data onto W" =
  let module M = Reduce.Amax.Compute (Direct) in
  (* [H2 W2 C1]; value(h,w): row H0 = [1,3] (max 3), row H1 = [5,7] (max 7). *)
  let x_shape = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:2 ~w:2 ~c:1 in
  let x =
    Tensor.materialize x_shape (fun c ->
        [| [| 1.; 3. |]; [| 5.; 7. |] |].(row c).(col c))
  in
  let run ~keepdim =
    let p = { Reduce.Amax.dims = [ Axis.W ]; keepdim } in
    Format.printf "%a@." (pp_result Tensor.pp)
      (eval_tensor
         (Reduce.Amax.output_shape ~x_shape p)
         (M.pixel p ~x_shape ~x))
  in
  run ~keepdim:true;
  run ~keepdim:false;
  (* same maxes {3, 7}; keepdim=true keeps them on H, keepdim=false moves them to W. *)
  [%expect
    {|
    tensor f32 [H=2 W=1 C=1] {3, 7}
    tensor f32 [W=2 C=1] {3, 7} |}]

let%expect_test "Direct: vector_norm over spatial (H,W), per channel" =
  let module M = Reduce.Vector_norm.Compute (Direct) in
  (* Same input as the mean/amax fixtures above: channel 0's L2 norm over
     [1,2,3,4] is sqrt(30), channel 1's over 10x that is sqrt(3000). *)
  let x_shape = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:2 ~w:2 ~c:2 in
  let x =
    Tensor.materialize x_shape (fun c ->
        let base = [| 1.; 2.; 3.; 4. |].((row c * 2) + col c) in
        if chan c = 0 then base else base *. 10.)
  in
  let p = { Reduce.Vector_norm.dims = [ Axis.H; Axis.W ]; keepdim = true } in
  Format.printf "%a@." (pp_result Tensor.pp)
    (eval_tensor
       (Reduce.Vector_norm.output_shape ~x_shape p)
       (M.pixel p ~x_shape ~x));
  [%expect {| tensor f32 [C=2] {5.47723, 54.7723} |}]

let%expect_test
    "Direct: vector_norm over W, keepdim=false shifts H's data onto W" =
  let module M = Reduce.Vector_norm.Compute (Direct) in
  (* [H2 W2 C1]; value(h,w): row H0 = [1,3] (norm sqrt(10)), row H1 = [5,7]
     (norm sqrt(74)). *)
  let x_shape = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:2 ~w:2 ~c:1 in
  let x =
    Tensor.materialize x_shape (fun c ->
        [| [| 1.; 3. |]; [| 5.; 7. |] |].(row c).(col c))
  in
  let run ~keepdim =
    let p = { Reduce.Vector_norm.dims = [ Axis.W ]; keepdim } in
    Format.printf "%a@." (pp_result Tensor.pp)
      (eval_tensor
         (Reduce.Vector_norm.output_shape ~x_shape p)
         (M.pixel p ~x_shape ~x))
  in
  run ~keepdim:true;
  run ~keepdim:false;
  (* same norms {sqrt(10), sqrt(74)}; keepdim=true keeps them on H, keepdim=false
     moves them to W. *)
  [%expect
    {|
    tensor f32 [H=2 W=1 C=1] {3.16228, 8.60233}
    tensor f32 [W=2 C=1] {3.16228, 8.60233} |}]

let%expect_test "Direct: rms_norm over C (channel-wise normalise)" =
  let module R = Norm.RmsNorm.Compute (Direct) in
  let run ~vals ~w ~eps =
    let x_shape = s1c (List.length vals) in
    let xa = Array.of_list vals and wa = Array.of_list w in
    let x = Tensor.materialize x_shape (fun c -> xa.(chan c)) in
    (* weight carries the normalised (C) shape; here all other axes are 1 so its
       shape coincides with x's. *)
    let weight = Tensor.materialize x_shape (fun c -> wa.(chan c)) in
    let p = { Norm.RmsNorm.dims = [ Axis.C ]; eps } in
    Format.printf "%a@." (pp_result Tensor.pp)
      (eval_tensor
         (Norm.RmsNorm.output_shape ~x_shape p)
         (R.pixel p ~x_shape ~x ~weight))
  in
  (* distinct |x| so the mean genuinely averages: mean(1²,7²)=mean(1,49)=25,
     sqrt=5, x/5 -> {0.2, 1.4} (eps=0, weight=1). *)
  run ~vals:[ 1.; 7. ] ~w:[ 1.; 1. ] ~eps:0.;
  (* eps + weight, sign preserved through the square: mean(x²)=4, +12=16,
     sqrt=4, x*0.25*w = {0.5, -1, 1.5, -2}. *)
  run ~vals:[ 2.; -2.; 2.; -2. ] ~w:[ 1.; 2.; 3.; 4. ] ~eps:12.;
  [%expect
    {|
    tensor f32 [C=2] {0.2, 1.4}
    tensor f32 [C=4] {0.5, -1, 1.5, -2} |}]

let%expect_test
    "Direct: batch_norm per-channel affine over C (broadcast over H)" =
  let module B = Norm.BatchNorm.Compute (Direct) in
  (* [H=2 C=2]: x[h,c] with c0 = [1,3] over h, c1 = [5,7]. The per-channel
     running stats (mean=[1,5], var=[4,4]) apply across the H axis. *)
  let x_shape = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:2 ~w:1 ~c:2 in
  let x =
    Tensor.materialize x_shape (fun c ->
        [| [| 1.; 3. |]; [| 5.; 7. |] |].(chan c).(row c))
  in
  let vec2 a0 a1 =
    Tensor.materialize (s1c 2) (fun c -> [| a0; a1 |].(chan c))
  in
  let p = { Norm.BatchNorm.channel = Axis.C; eps = 0. } in
  let run ~w0 ~w1 ~b0 ~b1 =
    Format.printf "%a@." (pp_result Tensor.pp)
      (eval_tensor
         (Norm.BatchNorm.output_shape ~x_shape)
         (B.pixel p ~x ~weight:(vec2 w0 w1) ~bias:(vec2 b0 b1)
            ~running_mean:(vec2 1. 5.) ~running_var:(vec2 4. 4.)))
  in
  (* mean=[1,5], 1/sqrt(4)=0.5, weight=1 bias=0: (x-mean)*0.5 per channel. *)
  run ~w0:1. ~w1:1. ~b0:0. ~b1:0.;
  (* per-channel weight/bias: c0 -> *2 + 1, c1 -> *10 - 1. *)
  run ~w0:2. ~w1:10. ~b0:1. ~b1:(-1.);
  [%expect
    {|
    tensor f32 [H=2 W=1 C=2] {0, 0, 1, 1}
    tensor f32 [H=2 W=1 C=2] {1, -1, 3, 9} |}]

let%expect_test "Direct: permute [W=2 C=3] — swap W and C gives [W=3 C=2]" =
  let module P = Permute.Permute.Compute (Direct) in
  (* [W=2 C=3] tensor: x[w,c] = w*3 + c, row-major values 0..5. *)
  let x_shape = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:1 ~w:2 ~c:3 in
  let x =
    Tensor.materialize x_shape (fun c -> float_of_int ((col c * 3) + chan c))
  in
  let perm =
    [
      (Axis.N, Axis.N);
      (Axis.T, Axis.T);
      (Axis.D, Axis.D);
      (Axis.H, Axis.H);
      (Axis.W, Axis.C);
      (Axis.C, Axis.W);
    ]
  in
  Format.printf "%a@." (pp_result Tensor.pp)
    (eval_tensor (Permute.Permute.output_shape ~x_shape perm) (P.pixel perm ~x));
  (* transpose of [[0,1,2],[3,4,5]] is [[0,3],[1,4],[2,5]]: row-major 0,3,1,4,2,5 *)
  [%expect {| tensor f32 [W=3 C=2] {0, 3, 1, 4, 2, 5} |}]

let%expect_test "Direct: permute identity — output equals input" =
  let module P = Permute.Permute.Compute (Direct) in
  let x_shape = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:2 ~w:3 ~c:4 in
  let x =
    Tensor.materialize x_shape (fun c -> float_of_int (row c + col c + chan c))
  in
  let perm = List.map (fun a -> (a, a)) Axis.all in
  let result =
    let open Err.Syntax in
    let* out_shape = Permute.Permute.output_shape ~x_shape perm in
    let tensor = Schedule.evaluate out_shape (P.pixel perm ~x) in
    Err.return (out_shape, tensor)
  in
  Format.printf "%a@." (pp_result pp_shape_tensor) result;
  [%expect
    {|
    shape: [H=2 W=3 C=4]
    tensor f32 [H=2 W=3 C=4] {0, 1, 2, 3, 1, 2, 3, 4, ...} |}]

let%expect_test "Direct: permute 3D [H=2 W=3 C=4] — cycle H->W->C->H" =
  let module P = Permute.Permute.Compute (Direct) in
  (* Cycle: output H <- input W, output W <- input C, output C <- input H.
     Input shape [H=2 W=3 C=4] -> output shape [H=3 W=4 C=2]. *)
  let x_shape = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:2 ~w:3 ~c:4 in
  let x =
    Tensor.materialize x_shape (fun c ->
        (* unique value per element: h*12 + w*4 + c *)
        float_of_int ((row c * 12) + (col c * 4) + chan c))
  in
  let perm =
    [
      (Axis.N, Axis.N);
      (Axis.T, Axis.T);
      (Axis.D, Axis.D);
      (Axis.H, Axis.W);
      (Axis.W, Axis.C);
      (Axis.C, Axis.H);
    ]
  in
  let result =
    let open Err.Syntax in
    let* out_shape = Permute.Permute.output_shape ~x_shape perm in
    let tensor = Schedule.evaluate out_shape (P.pixel perm ~x) in
    Err.return (out_shape, tensor)
  in
  Format.printf "%a@." (pp_result (pp_named_shape_tensor "out shape")) result;
  (* output[h,w,c] = input[c, h, w]  (inverse cycle: C->H->W->C)
     (h=0,w=0,c=0): input[c=0,h=0,w=0]=0; (h=0,w=0,c=1): input[c=1,h=0,w=0]=12
     (h=0,w=1,c=0): input[c=0,h=0,w=1]=1; (h=0,w=1,c=1): input[c=1,h=0,w=1]=13 *)
  [%expect
    {|
    out shape: [H=3 W=4 C=2]
    tensor f32 [H=3 W=4 C=2] {0, 12, 1, 13, 2, 14, 3, 15, ...} |}]

(* --- Unbind ---------------------------------------------------------------

   Hand-computed, not compared against another instantiation of the same
   functor: agreement between Direct and Symbolic would prove staging, not
   arithmetic. Value at (h,w,c) is h*100 + w*10 + c, so each printed slice can
   be read straight off its coordinates. *)

(* Unbind DROPS the axis it selects along, so the survivors re-pack
   right-aligned exactly as mean(keepdim=false) does. [H2 W3 C2] unbound on H
   gives [W3 C2]; on W gives [W2 C2] with H's data now on W; on C gives [W2 C3]
   with H on W and W on C. Every ordinal is printed, because a bug that only
   gets output 0 right is exactly what a singleton test cannot see. *)
let%expect_test "Direct: unbind along an outer, a middle and an inner axis" =
  let module U = Split.Unbind.Compute (Direct) in
  let x_shape = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:2 ~w:3 ~c:2 in
  let x =
    Tensor.materialize x_shape (fun c ->
        float_of_int ((row c * 100) + (col c * 10) + chan c))
  in
  let run axis =
    let p = { Split.Unbind.axis } in
    let shapes =
      Split.Unbind.output_shapes ~x_shape p
      |> Err.or_raise ~pp_error:Shape_error.pp
    in
    Format.printf "unbind %a -> %d outputs@." Axis.pp axis (List.length shapes);
    List.iteri
      (fun i sh ->
        Format.printf "  out%d %a %a@." i Vec6.pp_shape sh Tensor.pp
          (Schedule.evaluate sh (U.pixel p ~output:i ~x)))
      shapes
  in
  run Axis.H;
  run Axis.W;
  run Axis.C;
  [%expect
    {|
    unbind H -> 2 outputs
      out0 [W=3 C=2] tensor f32 [W=3 C=2] {0, 1, 10, 11, 20, 21}
      out1 [W=3 C=2] tensor f32 [W=3 C=2] {100, 101, 110, 111, 120, 121}
    unbind W -> 3 outputs
      out0 [W=2 C=2] tensor f32 [W=2 C=2] {0, 1, 100, 101}
      out1 [W=2 C=2] tensor f32 [W=2 C=2] {10, 11, 110, 111}
      out2 [W=2 C=2] tensor f32 [W=2 C=2] {20, 21, 120, 121}
    unbind C -> 2 outputs
      out0 [W=2 C=3] tensor f32 [W=2 C=3] {0, 10, 20, 100, 110, 120}
      out1 [W=2 C=3] tensor f32 [W=2 C=3] {1, 11, 21, 101, 111, 121} |}]

(* --- Split_with_sizes -------------------------------------------------

   Hand-computed, same attribution discipline as the Unbind block above.
   Unlike [Unbind], the axis is KEPT in every output, so each piece is
   exactly what [Slice] would give for its window -- windows of DIFFERENT
   width here, so a wrong per-output offset is visible in which values land
   in which piece, not only in the piece count. *)
let%expect_test "Direct: split_with_sizes divides W into windows of 2 and 3" =
  let module Sw = Split.Split_with_sizes.Compute (Direct) in
  let x_shape = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:2 ~w:5 ~c:1 in
  let x =
    Tensor.materialize x_shape (fun c -> float_of_int ((row c * 10) + col c))
  in
  let params = { Split.Split_with_sizes.axis = Axis.W; sizes = [ 2; 3 ] } in
  let shapes =
    Split.Split_with_sizes.output_shapes ~x_shape params
    |> Err.or_raise ~pp_error:Shape_error.pp
  in
  let _, offsets =
    List.fold_left
      (fun (acc, os) size -> (acc + size, os @ [ acc ]))
      (0, []) params.Split.Split_with_sizes.sizes
  in
  List.iteri
    (fun i (sh, offset) ->
      Format.printf "out%d %a %a@." i Vec6.pp_shape sh Tensor.pp
        (Schedule.evaluate sh (Sw.pixel ~offset params ~x)))
    (List.combine shapes offsets);
  [%expect
    {|
    out0 [H=2 W=2 C=1] tensor f32 [H=2 W=2 C=1] {0, 1, 10, 11}
    out1 [H=2 W=3 C=1] tensor f32 [H=2 W=3 C=1] {2, 3, 4, 12, 13, 14} |}]

(* The two shape faults, checked at [output_shapes] rather than
   [Compute.pixel] (which trusts the shapes it is given, like every other
   op): a non-positive size (Native has no empty extent, [Slice]'s [Empty]
   rule) and a sizes list that does not sum to the axis's extent. *)
let%expect_test
    "split_with_sizes output_shapes: a non-positive size and a bad sum" =
  let x_shape = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:1 ~w:5 ~c:1 in
  let at sizes =
    Format.printf "%a@."
      (pp_result (fun ppf shapes ->
           Format.fprintf ppf "%d outputs" (List.length shapes)))
      (Split.Split_with_sizes.output_shapes ~x_shape
         { Split.Split_with_sizes.axis = Axis.W; sizes })
  in
  at [ 2; 0; 3 ];
  at [ 2; -1; 4 ];
  at [ 2; 2 ];
  at [ 2; 4 ];
  at [ 2; 3 ];
  [%expect
    {|
    split_with_sizes of axis W over extent 5: size 0 at index 1 is not positive; the engine has no empty extent
    split_with_sizes of axis W over extent 5: size -1 at index 1 is not positive; the engine has no empty extent
    split_with_sizes of axis W: sizes sum to 4, not the axis's extent 5
    split_with_sizes of axis W: sizes sum to 6, not the axis's extent 5
    2 outputs |}]

(* Same ceiling [Unbind.output_shapes] checks, on the LIST LENGTH rather than
   a derived count. *)
let%expect_test
    "split_with_sizes output_shapes: the output-count ceiling is exclusive" =
  let limit = Kernel.Limits.Hard.outputs in
  let at n =
    let x_shape = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:n ~w:1 ~c:1 in
    Format.printf "%d -> %a@." n
      (Core.Pretty.err_result
         ~ok:(fun ppf shapes ->
           Format.fprintf ppf "%d outputs" (List.length shapes))
         ~error:Shape_error.pp)
      (Split.Split_with_sizes.output_shapes ~x_shape
         {
           Split.Split_with_sizes.axis = Axis.H;
           sizes = List.init n (fun _ -> 1);
         })
  in
  List.iter at [ limit - 1; limit; limit + 1 ];
  [%expect
    {|
    4095 -> 4095 outputs
    4096 -> 4096 outputs, above the maximum of 4095
    4097 -> 4097 outputs, above the maximum of 4095 |}]

(* --- Concat -----------------------------------------------------------

   Hand-computed, same attribution discipline as the Unbind block above: each
   operand's values carry a distinguishing base offset (a: none, b: 1000, c:
   2000) so the printed output can be read straight off which operand and
   local coordinate produced each entry. *)
let%expect_test "Direct: concat along C, three operands of different width" =
  let module Cc = Concat.Concat.Compute (Direct) in
  let a_shape = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:2 ~w:2 ~c:1 in
  let b_shape = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:2 ~w:2 ~c:2 in
  let c_shape = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:2 ~w:2 ~c:1 in
  let a =
    Tensor.materialize a_shape (fun c ->
        float_of_int ((row c * 100) + (col c * 10)))
  in
  let b =
    Tensor.materialize b_shape (fun c ->
        float_of_int (1000 + (row c * 100) + (col c * 10) + chan c))
  in
  let c_ =
    Tensor.materialize c_shape (fun c ->
        float_of_int (2000 + (row c * 100) + (col c * 10)))
  in
  let params = { Concat.Concat.axis = Axis.C } in
  let result =
    eval_tensor
      (Concat.Concat.output_shape
         ~xs_shapes:[ a_shape; b_shape; c_shape ]
         params)
      (Cc.pixel params ~xs:[ (a_shape, a); (b_shape, b); (c_shape, c_) ])
  in
  Format.printf "%a@." (pp_result Tensor.pp) result;
  [%expect
    {| tensor f32 [H=2 W=2 C=4] {0, 1000, 1001, 2000, 10, 1010, 1011, 2010, ...} |}]

(* Axis generality: the same op along an OUTER axis (H), two operands of
   different height. [a]'s single row lands first (output h=0); [b]'s two
   rows follow (output h=1,2). *)
let%expect_test "Direct: concat along H, two operands of different height" =
  let module Cc = Concat.Concat.Compute (Direct) in
  let a_shape = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:1 ~w:2 ~c:1 in
  let b_shape = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:2 ~w:2 ~c:1 in
  let a = Tensor.materialize a_shape (fun c -> float_of_int (col c * 10)) in
  let b =
    Tensor.materialize b_shape (fun c ->
        float_of_int (1000 + (row c * 100) + (col c * 10)))
  in
  let params = { Concat.Concat.axis = Axis.H } in
  let result =
    eval_tensor
      (Concat.Concat.output_shape ~xs_shapes:[ a_shape; b_shape ] params)
      (Cc.pixel params ~xs:[ (a_shape, a); (b_shape, b) ])
  in
  Format.printf "%a@." (pp_result Tensor.pp) result;
  [%expect {| tensor f32 [H=3 W=2 C=1] {0, 10, 1000, 1010, 1100, 1110} |}]

(* The overflow/mismatch faults, checked at [output_shape] rather than
   [Compute.pixel] (which trusts the shapes it is given, like every other
   op). *)
let%expect_test "concat output_shape: empty list and a non-concat-axis mismatch"
    =
  let a_shape = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:2 ~w:2 ~c:1 in
  let b_shape = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:3 ~w:2 ~c:1 in
  let params = { Concat.Concat.axis = Axis.C } in
  Format.printf "%a@." (pp_result Vec6.pp_shape)
    (Concat.Concat.output_shape ~xs_shapes:[] params);
  Format.printf "%a@." (pp_result Vec6.pp_shape)
    (Concat.Concat.output_shape ~xs_shapes:[ a_shape; b_shape ] params);
  [%expect
    {|
    concat: at least one tensor is required
    concat: axis H extent must agree across every tensor (it is not the concatenated axis): 2 vs 3 |}]

(* The ceiling is EXCLUSIVE, matching [Kernel.Limits.create]'s own [v >= hard]
   test, so 4095 is the largest accepted count. Checked here rather than in the
   builder because this is the boundary that runs BEFORE the list exists: a
   builder-side check would already have paid for the allocation it prevents. *)
let%expect_test "unbind output_shapes: the output-count ceiling is exclusive" =
  let limit = Kernel.Limits.Hard.outputs in
  let at n =
    let x_shape = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:n ~w:1 ~c:1 in
    Format.printf "%d -> %a@." n
      (Core.Pretty.err_result
         ~ok:(fun ppf shapes ->
           Format.fprintf ppf "%d outputs" (List.length shapes))
         ~error:Shape_error.pp)
      (Split.Unbind.output_shapes ~x_shape { Split.Unbind.axis = Axis.H })
  in
  List.iter at [ limit - 1; limit; limit + 1 ];
  [%expect
    {|
    4095 -> 4095 outputs
    4096 -> 4096 outputs, above the maximum of 4095
    4097 -> 4097 outputs, above the maximum of 4095 |}]

(* ---- Stack -----------------------------------------------------------

   The case that caught the shape/coordinate direction bug during
   development: [dim] is NOT the outermost insertion position,
   so the real per-operand extent moves onto a DIFFERENT native axis than the
   one the operand's own storage uses (here: native W -> native H), and a
   naive pass-through of the output coordinate into the operand would read
   the wrong, always-extent-1 slot. [a]'s values are row*10+col, [b]'s are
   offset by +100, so a swapped operand or a misrouted coordinate is legible
   rather than merely unequal. *)
let%expect_test "Direct: stack at a non-outermost dim, two rank-2 operands" =
  let module St = Concat.Stack.Compute (Direct) in
  (* Native shape for a rank-2 ATen tensor [3,4]: the innermost two axes,
     W (rows) and C (cols). *)
  let a_shape = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:1 ~w:3 ~c:4 in
  let a =
    Tensor.materialize a_shape (fun c -> float_of_int ((col c * 10) + chan c))
  in
  let b =
    Tensor.materialize a_shape (fun c ->
        float_of_int (100 + (col c * 10) + chan c))
  in
  (* ATen [torch.stack([a, b], dim=1)] on two [3,4] operands: axis_of_dim
     ~rank:3 1 = W, the SAME axis each operand's real row extent already
     occupies -- the case that exercises the direction bug. *)
  let params = { Concat.Stack.axis = Axis.W } in
  let result =
    eval_tensor
      (Concat.Stack.output_shape ~xs_shapes:[ a_shape; a_shape ] params)
      (St.pixel params ~xs:[ a; b ])
  in
  Format.printf "%a@." (pp_result Tensor.pp) result;
  [%expect
    {|
    tensor f32 [H=3 W=2 C=4] {0, 1, 2, 3, 100, 101, 102, 103, ...} |}]

(* The boundary case, dim=0: here [Concat]'s and [Stack]'s coordinate spaces
   coincide, since inserting the OUTERMOST axis relabels nothing -- this is
   the case both fold directions agree on, so it alone would NOT have caught
   the bug above; it is here to pin the boundary now that the general case is
   covered. *)
let%expect_test "Direct: stack at the outermost dim" =
  let module St = Concat.Stack.Compute (Direct) in
  let a_shape = s1c 3 in
  let a = Tensor.materialize a_shape (fun c -> float_of_int (chan c)) in
  let b = Tensor.materialize a_shape (fun c -> float_of_int (100 + chan c)) in
  let params = { Concat.Stack.axis = Axis.W } in
  let result =
    eval_tensor
      (Concat.Stack.output_shape ~xs_shapes:[ a_shape; a_shape ] params)
      (St.pixel params ~xs:[ a; b ])
  in
  Format.printf "%a@." (pp_result Tensor.pp) result;
  [%expect {| tensor f32 [W=2 C=3] {0, 1, 2, 100, 101, 102} |}]

(* ---- Pad: hand-computed, one rule per test ------------------------------- *)

(* A coordinate-coded tensor: element (h, w) is 10*h + w, so every value names
   the position it came from and a wrong source coordinate is legible rather
   than merely unequal. *)
let coord_hw h w =
  let shape = Vec6.shape ~n:1 ~t:1 ~d:1 ~h ~w ~c:1 in
  ( shape,
    Tensor.materialize shape (fun c -> float_of_int ((10 * row c) + col c)) )

(* [Tensor.pp] truncates after 8 elements and prints one flat list, which is the
   wrong shape of evidence here: a pad is judged by WHERE each value landed. Print
   the H x W grid in full, so a mirrored edge or a swapped before/after is legible
   rather than merely unequal. *)
let pp_grid shape ppf tensor =
  let h = Dim.to_int (Vec6.get shape Axis.H)
  and w = Dim.to_int (Vec6.get shape Axis.W) in
  Format.fprintf ppf "%a@," Vec6.pp_shape shape;
  for i = 0 to h - 1 do
    for j = 0 to w - 1 do
      let c = Vec6.coord ~n:0 ~t:0 ~d:0 ~h:i ~w:j ~c:0 in
      Format.fprintf ppf "%s%g"
        (if j = 0 then "" else " ")
        (Tensor.read tensor c)
    done;
    Format.fprintf ppf "@,"
  done

let pad_eval ~x_shape ~x (p : Pad.Pad.params) =
  let module P = Pad.Pad.Compute (Direct) in
  let open Err.Syntax in
  let r =
    let* out_shape = Pad.Pad.output_shape ~x_shape p in
    Err.return (out_shape, Schedule.evaluate out_shape (P.pixel p ~x_shape ~x))
  in
  Format.printf "@[<v>%a@]@."
    (pp_result (fun ppf (shape, t) -> pp_grid shape ppf t))
    r

let%expect_test "Direct: pad constant, one axis, asymmetric" =
  let x_shape, x = coord_hw 2 3 in
  (* W: 1 before, 2 after -> extent 3 + 3 = 6. Row 0 reads 0,1,2 into slots
     1..3; slots 0, 4, 5 are the fill. *)
  pad_eval ~x_shape ~x
    {
      Pad.Pad.pads = [ (Axis.W, { Pad.Pad.before = 1; after = 2 }) ];
      mode = Pad.Pad.Constant (-7.);
    };
  [%expect {|
    [H=2 W=6 C=1]
    -7 0 1 2 -7 -7
    -7 10 11 12 -7 -7 |}]

let%expect_test "Direct: pad constant, two axes at once" =
  let x_shape, x = coord_hw 2 2 in
  (* H by (1,0) and W by (0,1): 3 rows of 3. The interior is the 2x2 source in
     the LOWER-LEFT, which distinguishes a before/after swap on either axis. *)
  pad_eval ~x_shape ~x
    {
      Pad.Pad.pads =
        [
          (Axis.H, { Pad.Pad.before = 1; after = 0 });
          (Axis.W, { Pad.Pad.before = 0; after = 1 });
        ];
      mode = Pad.Pad.Constant 9.;
    };
  [%expect {|
    [H=3 W=3 C=1]
    9 9 9
    0 1 9
    10 11 9 |}]

let%expect_test "Direct: pad constant fill is used, not zero" =
  let x_shape, x = coord_hw 1 2 in
  (* A dropped [value] would print 0 in the pad slots. The source's own first
     element is 0 too, which is exactly why the fill must not be. *)
  pad_eval ~x_shape ~x
    {
      Pad.Pad.pads = [ (Axis.W, { Pad.Pad.before = 1; after = 1 }) ];
      mode = Pad.Pad.Constant 0.5;
    };
  [%expect {|
    [W=4 C=1]
    0.5 0 1 0.5 |}]

let%expect_test "Direct: pad reflect mirrors about the boundary, not through it"
    =
  let x_shape, x = coord_hw 1 4 in
  (* Source 0,1,2,3. Reflect by (2,2) gives 2,1 | 0,1,2,3 | 2,1 — the boundary
     element is NOT repeated, which is what separates reflect from replicate,
     and the left block is the mirror of the LEFT end rather than a copy of the
     right one. *)
  pad_eval ~x_shape ~x
    {
      Pad.Pad.pads = [ (Axis.W, { Pad.Pad.before = 2; after = 2 }) ];
      mode = Pad.Pad.Reflect;
    };
  [%expect {|
    [W=8 C=1]
    2 1 0 1 2 3 2 1 |}]

let%expect_test "Direct: pad reflect, two axes, asymmetric" =
  let x_shape, x = coord_hw 3 3 in
  (* H by (1,0), W by (0,2): the top row mirrors row 1 (values 10..12) and each
     row's tail mirrors its own columns 1 and 0. *)
  pad_eval ~x_shape ~x
    {
      Pad.Pad.pads =
        [
          (Axis.H, { Pad.Pad.before = 1; after = 0 });
          (Axis.W, { Pad.Pad.before = 0; after = 2 });
        ];
      mode = Pad.Pad.Reflect;
    };
  [%expect
    {|
    [H=4 W=5 C=1]
    10 11 12 11 10
    0 1 2 1 0
    10 11 12 11 10
    20 21 22 21 20 |}]

let%expect_test "Direct: negative pads crop" =
  let x_shape, x = coord_hw 3 4 in
  (* Crop one column from each side of W and the first row of H: a 2x2 window
     starting at (1,1). *)
  pad_eval ~x_shape ~x
    {
      Pad.Pad.pads =
        [
          (Axis.H, { Pad.Pad.before = -1; after = 0 });
          (Axis.W, { Pad.Pad.before = -1; after = -1 });
        ];
      mode = Pad.Pad.Constant 0.;
    };
  [%expect {|
    [H=2 W=2 C=1]
    11 12
    21 22 |}]

let%expect_test "Direct: pad and crop on different axes of one node" =
  let x_shape, x = coord_hw 2 3 in
  (* H padded by (1,0), W cropped by (-1,0): 3 rows of 2. *)
  pad_eval ~x_shape ~x
    {
      Pad.Pad.pads =
        [
          (Axis.H, { Pad.Pad.before = 1; after = 0 });
          (Axis.W, { Pad.Pad.before = -1; after = 0 });
        ];
      mode = Pad.Pad.Constant 5.;
    };
  [%expect {|
    [H=3 W=2 C=1]
    5 5
    1 2
    11 12 |}]

let%expect_test "Direct: pad rejects the configurations with no Native result" =
  let x_shape, _ = coord_hw 3 4 in
  let refuse (p : Pad.Pad.params) =
    Format.printf "%a@." (pp_result Vec6.pp_shape)
      (Pad.Pad.output_shape ~x_shape p)
  in
  (* A crop that consumes the axis: legal in ATen (size-0), unrepresentable
     here. *)
  refuse
    {
      Pad.Pad.pads = [ (Axis.H, { Pad.Pad.before = -2; after = -1 }) ];
      mode = Pad.Pad.Constant 0.;
    };
  refuse
    {
      Pad.Pad.pads = [ (Axis.H, { Pad.Pad.before = -5; after = 0 }) ];
      mode = Pad.Pad.Constant 0.;
    };
  (* Reflect needs each POSITIVE side below the extent it mirrors: H is 3, so 3
     is already too wide while 2 is fine. *)
  refuse
    {
      Pad.Pad.pads = [ (Axis.H, { Pad.Pad.before = 3; after = 0 }) ];
      mode = Pad.Pad.Reflect;
    };
  refuse
    {
      Pad.Pad.pads = [ (Axis.H, { Pad.Pad.before = 0; after = 3 }) ];
      mode = Pad.Pad.Reflect;
    };
  (* The same widths are ordinary in constant mode — the rule is reflect's, not
     padding's. *)
  refuse
    {
      Pad.Pad.pads = [ (Axis.H, { Pad.Pad.before = 3; after = 3 }) ];
      mode = Pad.Pad.Constant 0.;
    };
  (* Two entries for one axis: unreachable from either importer, so this is the
     guard on the builder and on JSON decoding. *)
  refuse
    {
      Pad.Pad.pads =
        [
          (Axis.W, { Pad.Pad.before = 1; after = 1 });
          (Axis.W, { Pad.Pad.before = 2; after = 2 });
        ];
      mode = Pad.Pad.Constant 0.;
    };
  [%expect
    {|
    pad of axis H by (-2, -1) over extent 3 leaves 0 elements; the engine has no empty extent
    pad of axis H by (-5, 0) over extent 3 leaves -2 elements; the engine has no empty extent
    reflect pad of axis H by (3, 0) needs each side below the extent 3
    reflect pad of axis H by (0, 3) needs each side below the extent 3
    [H=9 W=4 C=1]
    axis W has more than one pad entry |}]

(* ---- Slice ----------------------------------------------------------------

   Four mutations were applied to [Split.Slice] and observed failing here before
   being reverted, which is what makes these goldens evidence rather than
   description:

   - floor instead of the ceiling in [output_shape]: every strided extent drops
     by one, in this file and in graph_json_test.ml;
   - the step multiplication dropped from [Compute.pixel]: [0 2 4] becomes
     [0 1 2];
   - [start] replaced by 0: the selected window slides to the origin on both
     axes;
   - the upper range bound weakened from [stop <= extent]: an out-of-range slice
     produces a shape instead of the typed refusal, and [Compute]'s read goes
     out of bounds behind it.

   A fifth -- the WRONG AXIS -- is refuted by the two tests that run the same
   bounds on H and on W over a non-square fixture, which is why that pair exists
   rather than one test. *)

let slice_eval ~x_shape ~x (p : Split.Slice.params) =
  let module Sl = Split.Slice.Compute (Direct) in
  let open Err.Syntax in
  let r =
    let* out_shape = Split.Slice.output_shape ~x_shape p in
    Err.return (out_shape, Schedule.evaluate out_shape (Sl.pixel p ~x))
  in
  Format.printf "@[<v>%a@]@."
    (pp_result (fun ppf (shape, t) -> pp_grid shape ppf t))
    r

let pos = Op_config.Pos.of_int

let%expect_test "Direct: slice keeps the axis and narrows it" =
  let x_shape, x = coord_hw 3 4 in
  (* Columns 1 and 2 of every row. The values are 10*row + col, so a wrong start
     shows as a column shift and a wrong axis as a row selection. *)
  slice_eval ~x_shape ~x
    { Split.Slice.axis = Axis.W; start = 1; stop = 3; step = pos 1 };
  [%expect {|
    [H=3 W=2 C=1]
    1 2
    11 12
    21 22 |}]

let%expect_test "Direct: slice on the other axis selects rows" =
  let x_shape, x = coord_hw 3 4 in
  (* The same bounds on H. Distinguishable from the W case only because the
     fixture is 3x4 and the values encode both coordinates. *)
  slice_eval ~x_shape ~x
    { Split.Slice.axis = Axis.H; start = 1; stop = 3; step = pos 1 };
  [%expect {|
    [H=2 W=4 C=1]
    10 11 12 13
    20 21 22 23 |}]

let%expect_test "Direct: slice step selects every k-th element" =
  let x_shape, x = coord_hw 1 6 in
  (* [0,6) step 2 -> columns 0,2,4. An implementation that forgot the step
     multiplication would print 0 1 2. *)
  slice_eval ~x_shape ~x
    { Split.Slice.axis = Axis.W; start = 0; stop = 6; step = pos 2 };
  (* [1,6) step 2 -> columns 1,3,5: the span is 5, so the CEILING is what makes
     the count 3 rather than 2. *)
  slice_eval ~x_shape ~x
    { Split.Slice.axis = Axis.W; start = 1; stop = 6; step = pos 2 };
  (* [0,5) step 3 -> columns 0,3. Span 5 over step 3 is 1.67, and both a floor
     and a truncation would print one column. *)
  slice_eval ~x_shape ~x
    { Split.Slice.axis = Axis.W; start = 0; stop = 5; step = pos 3 };
  [%expect
    {|
    [W=3 C=1]
    0 2 4

    [W=3 C=1]
    1 3 5

    [W=2 C=1]
    0 3 |}]

let%expect_test "Direct: slice of the whole axis is the identity" =
  let x_shape, x = coord_hw 2 2 in
  (* The configuration a Default-tier walk would generate, and the reason
     slice.Tensor needs a walk_meta entry rather than the generated default:
     every implementation that returns its input passes this one. *)
  slice_eval ~x_shape ~x
    { Split.Slice.axis = Axis.W; start = 0; stop = 2; step = pos 1 };
  [%expect {|
    [H=2 W=2 C=1]
    0 1
    10 11 |}]

let%expect_test "Slice: the configurations with no Native result" =
  let x_shape = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:3 ~w:4 ~c:1 in
  let refuse (p : Split.Slice.params) =
    Format.printf "%a@." (pp_result Vec6.pp_shape)
      (Split.Slice.output_shape ~x_shape p)
  in
  (* Empty: legal in ATen, which returns a size-0 tensor, and unrepresentable
     here. Both the degenerate [start = stop] and a step wide enough to skip
     everything are the same fault -- the second cannot happen, since a
     non-empty span always yields at least one element under the ceiling, and
     the case is written out to record that rather than leave it implied. *)
  refuse { Split.Slice.axis = Axis.W; start = 2; stop = 2; step = pos 1 };
  refuse { Split.Slice.axis = Axis.W; start = 3; stop = 4; step = pos 9 };
  (* Out of range. Unreachable from either importer -- both build their bounds
     with [Aten_shape.resolve_slice], which clamps -- so these guard the builder
     and JSON decoding, and they are what keeps [Compute]'s read in bounds. *)
  refuse { Split.Slice.axis = Axis.W; start = 0; stop = 5; step = pos 1 };
  refuse { Split.Slice.axis = Axis.W; start = 3; stop = 1; step = pos 1 };
  refuse { Split.Slice.axis = Axis.W; start = -1; stop = 2; step = pos 1 };
  [%expect
    {|
    slice of axis W [2, 2) step 1 over extent 4 selects 0 elements; the engine has no empty extent
    [H=3 W=1 C=1]
    slice of axis W [0, 5) step 1 over extent 4 is not within 0 <= start <= stop <= extent
    slice of axis W [3, 1) step 1 over extent 4 is not within 0 <= start <= stop <= extent
    slice of axis W [-1, 2) step 1 over extent 4 is not within 0 <= start <= stop <= extent |}]

(* ---- Select ----------------------------------------------------------------

   [Select] DROPS the axis it picks along, unlike [Slice] which keeps the
   same rank -- so unlike [slice_eval]'s tests above, this must also confirm
   the surviving axes repack right-aligned exactly as [Unbind]'s do (it
   reuses the same [Aten_shape.repack_dropped] pairing). Value at (h,w,c) is
   h*100 + w*10 + c, so a wrong index reads a different row/column and a
   wrong axis reads the wrong pair of survivors. *)
let select_eval ~x_shape ~x (p : Split.Select.params) =
  let module Se = Split.Select.Compute (Direct) in
  eval_tensor (Split.Select.output_shape ~x_shape p) (Se.pixel p ~x)

let%expect_test "Direct: select along an outer, a middle and an inner axis" =
  let x_shape = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:2 ~w:3 ~c:2 in
  let x =
    Tensor.materialize x_shape (fun c ->
        float_of_int ((row c * 100) + (col c * 10) + chan c))
  in
  let run axis index =
    let p = { Split.Select.axis; index } in
    Format.printf "select %a=%d -> %a@." Axis.pp axis index
      (pp_result Tensor.pp)
      (select_eval ~x_shape ~x p)
  in
  run Axis.H 1;
  run Axis.W 2;
  run Axis.C 1;
  [%expect
    {|
    select H=1 -> tensor f32 [W=3 C=2] {100, 101, 110, 111, 120, 121}
    select W=2 -> tensor f32 [W=2 C=2] {20, 21, 120, 121}
    select C=1 -> tensor f32 [W=2 C=3] {1, 11, 21, 101, 111, 121} |}]

let%expect_test "Select: an out-of-range index has no Native result" =
  let x_shape = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:1 ~w:1 ~c:4 in
  let refuse index =
    Format.printf "%a@." (pp_result Vec6.pp_shape)
      (Split.Select.output_shape ~x_shape { Split.Select.axis = Axis.C; index })
  in
  refuse 4;
  refuse (-1);
  [%expect
    {|
    slice of axis C [4, 5) step 1 over extent 4 is not within 0 <= start <= stop <= extent
    slice of axis C [-1, 0) step 1 over extent 4 is not within 0 <= start <= stop <= extent |}]

(* LayerNorm's arithmetic, against values derived by hand rather than by a
   second run of the same functor. Direct-vs-Symbolic cannot see any of the
   mutations these cover, because both instantiate this same [Compute].

   Every case is chosen so a plausible WRONG implementation prints something
   visibly different:
     - skipping the centring (i.e. RMSNorm) changes every case with a non-zero
       mean, which is all of them;
     - dividing the variance by count - 1 changes every case with count > 2;
     - adding eps outside the sqrt changes the two eps cases;
     - reducing the wrong axis changes the multi-axis case into per-row work;
     - indexing the affine operands by the full output coordinate reads them
       out of range;
     - applying bias before weight changes the affine case's sign pattern.
   A 2-element normalised slice is deliberately NOT used as the main fixture:
   it normalises to +/-1 whatever the divisor is, so it cannot separate them. *)
let%expect_test "Direct: layer_norm over C, centred and scaled" =
  let module L = Norm.LayerNorm.Compute (Direct) in
  let ones n = List.init n (fun _ -> 1.)
  and zeros n = List.init n (fun _ -> 0.) in
  let run ?w ?b ~vals ~eps () =
    let n = List.length vals in
    let x_shape = s1c n in
    let arr l = Array.of_list l in
    let xa = arr vals in
    let wa = arr (Option.value w ~default:(ones n)) in
    let ba = arr (Option.value b ~default:(zeros n)) in
    let x = Tensor.materialize x_shape (fun c -> xa.(chan c)) in
    (* Both affine operands carry the normalised (C) shape; every other axis is
       1 here, so their shape coincides with x's. *)
    let weight = Tensor.materialize x_shape (fun c -> wa.(chan c)) in
    let bias = Tensor.materialize x_shape (fun c -> ba.(chan c)) in
    let p = { Norm.LayerNorm.dims = [ Axis.C ]; eps } in
    Format.printf "%a@." (pp_result Tensor.pp)
      (eval_tensor
         (Norm.LayerNorm.output_shape ~x_shape p)
         (L.pixel p ~x_shape ~x ~weight ~bias))
  in
  (* Non-zero mean AND non-unit variance, the two things RMSNorm gets wrong:
     mean(3,3,7,7) = 5, deviations (-2,-2,2,2), var = 16/4 = 4, sqrt = 2,
     so y = (-1,-1,1,1). The unbiased divisor 3 would give var = 16/3,
     sqrt = 2.3094, y = -+0.8660254 instead. *)
  run ~vals:[ 3.; 3.; 7.; 7. ] ~eps:0. ();
  (* The same slice with a NEGATIVE weight and a non-zero bias, and the affine
     step in ATen's order: y * weight + bias.
       (-1*-1)+10 = 11, (-1*2)+20 = 18, (1*-3)+30 = 27, (1*4)+40 = 44.
     Bias before weight would give (y + b) * w = (-9, 38, -81, 164). *)
  run ~vals:[ 3.; 3.; 7.; 7. ] ~w:[ -1.; 2.; -3.; 4. ] ~b:[ 10.; 20.; 30.; 40. ]
    ~eps:0. ();
  (* count = 1: the variance is 0 by construction, so y is 0 whatever eps is
     and only the bias survives. Legal in ATen, so it must be legal here. *)
  run ~vals:[ 7. ] ~w:[ 5. ] ~b:[ 3. ] ~eps:1e-5 ();
  (* A large offset with a small variance. mean = 8388608 = 2^23 and the
     deviations are -+1, both exactly representable in f32, so the answer is
     again (-1,-1,1,1).

     This pins the ANSWER at a large offset; it is not a demonstration that
     E[x^2] - mean^2 breaks, and it cannot be one here -- [Direct] accumulates
     in f64, whose 53-bit mantissa absorbs a 2^46 mean-square with room to
     spare, and no f32-representable input is far enough out to exhaust it. The
     case for two passes is settled against ATen in f32 by the walk, not here. *)
  run ~vals:[ 8388607.; 8388607.; 8388609.; 8388609. ] ~eps:0. ();
  [%expect
    {|
    tensor f32 [C=4] {-1, -1, 1, 1}
    tensor f32 [C=4] {11, 18, 27, 44}
    tensor f32 [C=1] {3}
    tensor f32 [C=4] {-1, -1, 1, 1} |}]

(* Multi-axis: the reduction nests one [sum] per normalised axis, and reducing
   over the wrong subset is a wrong answer that still has the right shape. *)
let%expect_test "Direct: layer_norm over W and C jointly" =
  let module L = Norm.LayerNorm.Compute (Direct) in
  (* [W=2 C=2] holding 1, 3, 5, 7 in (w, c) order. Over BOTH axes: mean = 4,
     deviations (-3,-1,1,3), var = 20/4 = 5, sqrt(5) = 2.2360680, so
     y = (-1.3416408, -0.4472136, 0.4472136, 1.3416408).

     Over C alone it would normalise each row separately -- (1,3) and (5,7) --
     and print (-1, 1, -1, 1). Over W alone, (-1, -1, 1, 1). Neither is close
     to the joint answer, which is the point of the fixture. *)
  let x_shape = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:1 ~w:2 ~c:2 in
  let vals = [| [| 1.; 3. |]; [| 5.; 7. |] |] in
  let x = Tensor.materialize x_shape (fun c -> vals.(col c).(chan c)) in
  let affine v = Tensor.materialize x_shape (fun _ -> v) in
  let p = { Norm.LayerNorm.dims = [ Axis.W; Axis.C ]; eps = 0. } in
  Format.printf "%a@." (pp_result Tensor.pp)
    (eval_tensor
       (Norm.LayerNorm.output_shape ~x_shape p)
       (L.pixel p ~x_shape ~x ~weight:(affine 1.) ~bias:(affine 0.)));
  [%expect {| tensor f32 [W=2 C=2] {-1.34164, -0.447214, 0.447214, 1.34164} |}]

(* Both epsilon values the corpus contains, on a slice whose variance is the
   same order as eps -- which is the only place the two are distinguishable,
   and also the only place "eps inside the sqrt" is distinguishable from "eps
   outside" it.

   x = (1, 1, 1 + 2^-10, 1 + 2^-10), all exact in f32. mean = 1 + 2^-11,
   deviations -+2^-11, var = 2^-22 = 2.384185791015625e-07. *)
let%expect_test "Direct: layer_norm eps is inside the sqrt" =
  let module L = Norm.LayerNorm.Compute (Direct) in
  let run ~eps =
    let x_shape = s1c 4 in
    let hi = 1. +. (2. ** -10.) in
    let xa = [| 1.; 1.; hi; hi |] in
    let x = Tensor.materialize x_shape (fun c -> xa.(chan c)) in
    let affine v = Tensor.materialize x_shape (fun _ -> v) in
    let p = { Norm.LayerNorm.dims = [ Axis.C ]; eps } in
    Format.printf "%a@." (pp_result Tensor.pp)
      (eval_tensor
         (Norm.LayerNorm.output_shape ~x_shape p)
         (L.pixel p ~x_shape ~x ~weight:(affine 1.) ~bias:(affine 0.)))
  in
  run ~eps:1e-6;
  run ~eps:1e-5;
  [%expect
    {|
    tensor f32 [C=4] {-0.438769, -0.438769, 0.438769, 0.438769}
    tensor f32 [C=4] {-0.1526, -0.1526, 0.1526, 0.1526} |}]

(* ---- group_norm -------------------------------------------------------------

   Hand-computed, same attribution discipline as the layer_norm block above.
   Unlike layer_norm, the reduction is over a WINDOW of C (one group's slice)
   plus the FULL extent of every other non-N axis -- so the fixture needs a
   real spatial extent (H > 1) to exercise the "full extent" half at all;
   layer_norm's single-axis-C fixtures would leave it untested. *)
let%expect_test "Direct: group_norm splits C into two groups, reducing H too" =
  let module G = Norm.GroupNorm.Compute (Direct) in
  (* [H=2 W=1 C=4], x[h,c] = h*10+c: group 0 is channels {0,1}, group 1 is
     {2,3}, and each group's window spans both rows of H. Group 0's four
     values {0,1,10,11} have mean 5.5, var 25.25; group 1's {2,3,12,13} have
     mean 7.5, the SAME variance (both groups are shifted copies of one
     another by 2), so a wrong group boundary that swapped which four values
     went together would still print a plausible-looking but different
     result. *)
  let x_shape = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:2 ~w:1 ~c:4 in
  let x =
    Tensor.materialize x_shape (fun c -> float_of_int ((row c * 10) + chan c))
  in
  let ones = Tensor.materialize x_shape (fun _ -> 1.) in
  let zeros = Tensor.materialize x_shape (fun _ -> 0.) in
  let p =
    {
      Norm.GroupNorm.channel = Axis.C;
      groups = Op_config.Pos.of_int 2;
      eps = 0.;
    }
  in
  Format.printf "%a@." (pp_result Tensor.pp)
    (eval_tensor
       (Norm.GroupNorm.output_shape ~x_shape p)
       (G.pixel p ~x_shape ~x ~weight:ones ~bias:zeros));
  [%expect
    {| tensor f32 [H=2 W=1 C=4] {-1.09454, -0.895533, -1.09454, -0.895533, 0.895533, 1.09454, 0.895533, 1.09454} |}]

(* The per-channel affine, on the trivial-spatial [H=1 W=1] case so the
   reduction itself is not also under test: with two channels per group and
   H=W=1, group 0 is {0,1} and group 1 is {2,3}, each normalising to
   [-1, 1]. weight/bias then apply PER CHANNEL (not per group, unlike
   layer_norm's per-normalised-shape affine) in ATen's [y * weight + bias]
   order. *)
let%expect_test "Direct: group_norm applies weight/bias per channel" =
  let module G = Norm.GroupNorm.Compute (Direct) in
  let x_shape = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:1 ~w:1 ~c:4 in
  let x = Tensor.materialize x_shape (fun c -> float_of_int (chan c)) in
  let wa = [| 10.; 20.; 30.; 40. |] in
  let ba = [| 100.; 200.; 300.; 400. |] in
  let weight = Tensor.materialize x_shape (fun c -> wa.(chan c)) in
  let bias = Tensor.materialize x_shape (fun c -> ba.(chan c)) in
  let p =
    {
      Norm.GroupNorm.channel = Axis.C;
      groups = Op_config.Pos.of_int 2;
      eps = 0.;
    }
  in
  Format.printf "%a@." (pp_result Tensor.pp)
    (eval_tensor
       (Norm.GroupNorm.output_shape ~x_shape p)
       (G.pixel p ~x_shape ~x ~weight ~bias));
  [%expect {| tensor f32 [C=4] {90, 220, 270, 440} |}]

(* [num_groups] must divide the channel count exactly -- ATen's own contract,
   checked at [output_shape] rather than assumed. *)
let%expect_test "group_norm output_shape: indivisible channel count" =
  let x_shape = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:1 ~w:1 ~c:5 in
  let p =
    {
      Norm.GroupNorm.channel = Axis.C;
      groups = Op_config.Pos.of_int 2;
      eps = 0.;
    }
  in
  Format.printf "%a@." (pp_result Vec6.pp_shape)
    (Norm.GroupNorm.output_shape ~x_shape p);
  [%expect {| group_norm: channel count 5 is not divisible by num_groups 2 |}]

(* ---- sdpa ------------------------------------------------------------------

   op8-impl.md commit 1 step 9: since [Direct] vs [Symbolic] cannot see a wrong
   formula (both run the same [Compute] functor), THIS is where the row is
   actually proven -- every case below is hand-computed against ATen's `math`
   backend structure (F3: split-sqrt scale, unconditional `_safe_softmax`),
   never against ATen itself.

   Mutation proof (CLAUDE.md: "prove the check can fail"), each applied to
   [Attention.Sdpa.Compute] in [lib/native/ops/attention.ml], run against this
   suite, OBSERVED changing at least one golden below, then reverted -- none
   of these mutations is left in the tree:
     - Q@K instead of Q@K^T (swapped which axis of [key] is the contraction
       index and which is the enumeration index);
     - scale applied after the softmax rather than split across the operands
       before the dot;
     - mask omitted;
     - mask sign reversed (subtracted instead of added);
     - the row max reduced over the wrong extent (E instead of Wk);
     - the denominator reduced over the wrong extent (E instead of Wk);
     - max-subtraction omitted (observed: [nan]/[inf] on the large-logits
       case, not merely a smaller numeric drift);
     - probabilities paired with the wrong value index (read [value] at the
       output's own row instead of the reduction variable);
     - D and H swapped;
     - the `_safe_softmax` guard removed (observed: [nan] on the all-`-inf`
       case, per F3 -- exactly the defect the guard exists to prevent).
   "scale applied once rather than split" is NOT a [%g]-rounded-printing
   mutation: [sqrt(scale)^2 = scale] in exact real arithmetic, so it agrees
   with the split form to 6 significant figures on ordinary inputs. It is
   proven separately below, bitwise, without touching [attention.ml] at all. *)

let run_sdpa params ~query_shape ~key_shape ~mask_shape ~query ~key ~value ~mask
    =
  let module S = Attention.Sdpa.Compute (Direct) in
  eval_tensor
    (Attention.Sdpa.output_shape ~query_shape ~key_shape ~value_shape:key_shape
       ~mask_shape:(Some mask_shape))
    (S.pixel params ~query_shape ~key_shape ~mask_shape ~query ~key ~value ~mask)

let no_mask = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:1 ~w:1 ~c:1
let zero_mask shape = Tensor.materialize shape (fun _ -> 0.)

let%expect_test "Direct: sdpa — one query, one key (trivial attention = value)"
    =
  let query_shape = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:1 ~w:1 ~c:2 in
  let key_shape = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:1 ~w:1 ~c:2 in
  let query = Tensor.materialize query_shape (fun c -> [| 1.; 2. |].(chan c)) in
  let key = Tensor.materialize key_shape (fun c -> [| 3.; 4. |].(chan c)) in
  let value = Tensor.materialize key_shape (fun c -> [| 7.; 8. |].(chan c)) in
  Format.printf "%a@." (pp_result Tensor.pp)
    (run_sdpa
       { Attention.Sdpa.scale = Attention.Sdpa.Scale.Default }
       ~query_shape ~key_shape ~mask_shape:no_mask ~query ~key ~value
       ~mask:(zero_mask no_mask));
  [%expect {| tensor f32 [C=2] {7, 8} |}]

let%expect_test "Direct: sdpa — two keys, unequal scores" =
  (* q=[1,0]; k0=[1,0] (dot=1), k1=[0,1] (dot=0); scale=1 so score = dot.
     m=1, z=1+exp(-1); p0=1/z, p1=exp(-1)/z. *)
  let query_shape = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:1 ~w:1 ~c:2 in
  let key_shape = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:1 ~w:2 ~c:2 in
  let query = Tensor.materialize query_shape (fun c -> [| 1.; 0. |].(chan c)) in
  let key =
    Tensor.materialize key_shape (fun c ->
        [| [| 1.; 0. |]; [| 0.; 1. |] |].(col c).(chan c))
  in
  let value =
    Tensor.materialize key_shape (fun c ->
        [| [| 10.; 20. |]; [| 30.; 40. |] |].(col c).(chan c))
  in
  Format.printf "%a@." (pp_result Tensor.pp)
    (run_sdpa
       { Attention.Sdpa.scale = Attention.Sdpa.Scale.Explicit 1.0 }
       ~query_shape ~key_shape ~mask_shape:no_mask ~query ~key ~value
       ~mask:(zero_mask no_mask));
  [%expect {| tensor f32 [C=2] {15.3788, 25.3788} |}]

let%expect_test "Direct: sdpa — default scale is 1/sqrt(head_dim)" =
  (* E=4: default scale = 0.5. q=[1,1,1,1], k0=q (dot=4 -> score=2), k1=0
     (dot=0 -> score=0). v0/v1 are constant vectors, so every output feature
     is the same softmax-weighted blend. *)
  let query_shape = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:1 ~w:1 ~c:4 in
  let key_shape = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:1 ~w:2 ~c:4 in
  let query = Tensor.materialize query_shape (fun _ -> 1.) in
  let key =
    Tensor.materialize key_shape (fun c -> if col c = 0 then 1. else 0.)
  in
  let value =
    Tensor.materialize key_shape (fun c -> if col c = 0 then 1. else 2.)
  in
  Format.printf "%a@." (pp_result Tensor.pp)
    (run_sdpa
       { Attention.Sdpa.scale = Attention.Sdpa.Scale.Default }
       ~query_shape ~key_shape ~mask_shape:no_mask ~query ~key ~value
       ~mask:(zero_mask no_mask));
  [%expect {| tensor f32 [C=4] {1.1192, 1.1192, 1.1192, 1.1192} |}]

let%expect_test "Direct: sdpa — explicit scale" =
  (* q=3, k0=1, k1=2, scale=2 (explicit) -> score = dot*scale = 6, 12. v0=5,
     v1=7. *)
  let query_shape = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:1 ~w:1 ~c:1 in
  let key_shape = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:1 ~w:2 ~c:1 in
  let query = Tensor.materialize query_shape (fun _ -> 3.) in
  let key =
    Tensor.materialize key_shape (fun c -> if col c = 0 then 1. else 2.)
  in
  let value =
    Tensor.materialize key_shape (fun c -> if col c = 0 then 5. else 7.)
  in
  Format.printf "%a@." (pp_result Tensor.pp)
    (run_sdpa
       { Attention.Sdpa.scale = Attention.Sdpa.Scale.Explicit 2.0 }
       ~query_shape ~key_shape ~mask_shape:no_mask ~query ~key ~value
       ~mask:(zero_mask no_mask));
  [%expect {| tensor f32 [C=1] {6.99505} |}]

(* Mutation proof: "scale applied once rather than split" (F3) is NOT a
   mutation [%g]-rounded printing can catch -- [sqrt(scale)^2 = scale] in
   EXACT real arithmetic, so the two forms agree to 6 significant figures for
   ordinary inputs. They are still genuinely different floats: rounding at the
   [sqrt] and at each of the two multiplies means [dot(q*sf, k*sf)] and
   [dot(q,k)*scale] are almost never bit-identical (observed here: a ULP-scale
   divergence, exactly the kind [.ai/native_walk_design.md]'s tolerance policy
   is about, not something a hand-picked test value hides by luck). Direct
   primitives only, bypassing [Compute] entirely, so this is independent of
   whatever attention.ml happens to do. *)
let%expect_test
    "Direct: sdpa — split-sqrt scale is not bit-identical to a single multiply"
    =
  let scale = 3.7 and q = 12.25 and k = -5.5 in
  let sf = Direct.sqrt (Direct.const scale) in
  let split =
    Direct.mul (Direct.mul (Direct.const q) sf) (Direct.mul (Direct.const k) sf)
  in
  let unsplit =
    Direct.mul
      (Direct.mul (Direct.const q) (Direct.const k))
      (Direct.const scale)
  in
  Format.printf "split=%h unsplit=%h bit_equal=%b@." split unsplit
    (Core.Float_bits.equal_exact split unsplit);
  [%expect
    {| split=-0x1.f293333333333p+7 unsplit=-0x1.f293333333334p+7 bit_equal=false |}]

let%expect_test "Direct: sdpa — additive mask excludes the largest raw score" =
  (* q=1, k0=1 (dot=1), k1=5 (dot=5, the larger raw score); mask=[0,-inf]
     rules k1 out entirely, so the output is exactly value0. *)
  let query_shape = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:1 ~w:1 ~c:1 in
  let key_shape = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:1 ~w:2 ~c:1 in
  let mask_shape = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:1 ~w:1 ~c:2 in
  let query = Tensor.materialize query_shape (fun _ -> 1.) in
  let key =
    Tensor.materialize key_shape (fun c -> if col c = 0 then 1. else 5.)
  in
  let value =
    Tensor.materialize key_shape (fun c -> if col c = 0 then 100. else 200.)
  in
  let mask =
    Tensor.materialize mask_shape (fun c ->
        if chan c = 0 then 0. else Float.neg_infinity)
  in
  Format.printf "%a@." (pp_result Tensor.pp)
    (run_sdpa
       { Attention.Sdpa.scale = Attention.Sdpa.Scale.Explicit 1.0 }
       ~query_shape ~key_shape ~mask_shape ~query ~key ~value ~mask);
  [%expect {| tensor f32 [C=1] {100} |}]

let%expect_test "Direct: sdpa — additive negative mask (finite)" =
  (* q=0, so the raw dot product is 0 for every key regardless of k -- the
     mask alone decides the weights: mask=[-1,-3], diff 2, same shape as the
     default-scale case above but distinguished by its own values. *)
  let query_shape = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:1 ~w:1 ~c:1 in
  let key_shape = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:1 ~w:2 ~c:1 in
  let mask_shape = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:1 ~w:1 ~c:2 in
  let query = Tensor.materialize query_shape (fun _ -> 0.) in
  let key =
    Tensor.materialize key_shape (fun c -> if col c = 0 then 7. else 9.)
  in
  let value =
    Tensor.materialize key_shape (fun c -> if col c = 0 then 9. else 11.)
  in
  let mask =
    Tensor.materialize mask_shape (fun c -> if chan c = 0 then -1. else -3.)
  in
  Format.printf "%a@." (pp_result Tensor.pp)
    (run_sdpa
       { Attention.Sdpa.scale = Attention.Sdpa.Scale.Explicit 1.0 }
       ~query_shape ~key_shape ~mask_shape ~query ~key ~value ~mask);
  [%expect {| tensor f32 [C=1] {9.23841} |}]

let%expect_test "Direct: sdpa — batch and head extents independently" =
  (* Wk=1: the softmax is trivially one-hot on the sole key, so the output IS
     the value at every (D,H), which proves those two axes thread through
     independently rather than being conflated with each other or with W/C. *)
  let query_shape = Vec6.shape ~n:1 ~t:1 ~d:2 ~h:2 ~w:1 ~c:1 in
  let key_shape = Vec6.shape ~n:1 ~t:1 ~d:2 ~h:2 ~w:1 ~c:1 in
  let query = Tensor.materialize query_shape (fun _ -> 0.) in
  let key = Tensor.materialize key_shape (fun _ -> 0.) in
  let value =
    Tensor.materialize key_shape (fun c ->
        float_of_int ((100 * bat c) + (10 * row c) + 1))
  in
  Format.printf "%a@." (pp_result Tensor.pp)
    (run_sdpa
       { Attention.Sdpa.scale = Attention.Sdpa.Scale.Default }
       ~query_shape ~key_shape ~mask_shape:no_mask ~query ~key ~value
       ~mask:(zero_mask no_mask));
  [%expect {| tensor f32 [D=2 H=2 W=1 C=1] {1, 11, 101, 111} |}]

let%expect_test "Direct: sdpa — large logits (max-subtraction stability)" =
  (* q=1, k0=1000, k1=999: raw scores 1000, 999. exp(1000) is not
     representable even in f64 (it is +inf); max-subtraction is what keeps
     this finite and correct: exp(0) + exp(-1). *)
  let query_shape = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:1 ~w:1 ~c:1 in
  let key_shape = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:1 ~w:2 ~c:1 in
  let query = Tensor.materialize query_shape (fun _ -> 1.) in
  let key =
    Tensor.materialize key_shape (fun c -> if col c = 0 then 1000. else 999.)
  in
  let value =
    Tensor.materialize key_shape (fun c -> if col c = 0 then 10. else 20.)
  in
  Format.printf "%a@." (pp_result Tensor.pp)
    (run_sdpa
       { Attention.Sdpa.scale = Attention.Sdpa.Scale.Explicit 1.0 }
       ~query_shape ~key_shape ~mask_shape:no_mask ~query ~key ~value
       ~mask:(zero_mask no_mask));
  [%expect {| tensor f32 [C=1] {12.6894} |}]

let%expect_test
    "Direct: sdpa — an all -inf row yields 0, not NaN (_safe_softmax)" =
  let query_shape = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:1 ~w:1 ~c:1 in
  let key_shape = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:1 ~w:2 ~c:1 in
  let mask_shape = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:1 ~w:1 ~c:2 in
  let query = Tensor.materialize query_shape (fun _ -> 1.) in
  let key = Tensor.materialize key_shape (fun _ -> 1.) in
  let value =
    Tensor.materialize key_shape (fun c -> if col c = 0 then 7. else 9.)
  in
  let mask = Tensor.materialize mask_shape (fun _ -> Float.neg_infinity) in
  Format.printf "%a@." (pp_result Tensor.pp)
    (run_sdpa
       { Attention.Sdpa.scale = Attention.Sdpa.Scale.Explicit 1.0 }
       ~query_shape ~key_shape ~mask_shape ~query ~key ~value ~mask);
  [%expect {| tensor f32 [C=1] {0} |}]

let%expect_test "Direct: sdpa — Wq <> Wk" =
  (* Two query positions, three keys, all-zero dot products: both query
     positions get the same uniform 1/3 blend of the three values. *)
  let query_shape = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:1 ~w:2 ~c:1 in
  let key_shape = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:1 ~w:3 ~c:1 in
  let query =
    Tensor.materialize query_shape (fun c -> if col c = 0 then 1. else 2.)
  in
  let key = Tensor.materialize key_shape (fun _ -> 0.) in
  let value =
    Tensor.materialize key_shape (fun c -> [| 3.; 6.; 9. |].(col c))
  in
  Format.printf "%a@." (pp_result Tensor.pp)
    (run_sdpa
       { Attention.Sdpa.scale = Attention.Sdpa.Scale.Explicit 1.0 }
       ~query_shape ~key_shape ~mask_shape:no_mask ~query ~key ~value
       ~mask:(zero_mask no_mask));
  [%expect {| tensor f32 [W=2 C=1] {6, 6} |}]

let%expect_test "Direct: sdpa — mask broadcast on Wq (2D-equivalent shape)" =
  (* Two query positions share one mask row: mask stored with W=1 (broadcast)
     against a real Wq=2. Dot products are 0 everywhere, so the mask alone
     decides, identically for both query positions. *)
  let query_shape = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:1 ~w:2 ~c:1 in
  let key_shape = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:1 ~w:2 ~c:1 in
  let mask_shape = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:1 ~w:1 ~c:2 in
  let query = Tensor.materialize query_shape (fun _ -> 0.) in
  let key = Tensor.materialize key_shape (fun _ -> 0.) in
  let value =
    Tensor.materialize key_shape (fun c -> if col c = 0 then 5. else 7.)
  in
  let mask =
    Tensor.materialize mask_shape (fun c -> if chan c = 0 then 0. else -2.)
  in
  Format.printf "%a@." (pp_result Tensor.pp)
    (run_sdpa
       { Attention.Sdpa.scale = Attention.Sdpa.Scale.Explicit 1.0 }
       ~query_shape ~key_shape ~mask_shape ~query ~key ~value ~mask);
  [%expect {| tensor f32 [W=2 C=1] {5.23841, 5.23841} |}]

let%expect_test "Direct: sdpa — mask broadcast on D (4D form)" =
  (* Two batch elements share one mask row: mask stored with D=1 (broadcast)
     against a real D=2. Values differ per batch (x10) to prove D itself
     still threads through correctly while the mask stays shared. *)
  let query_shape = Vec6.shape ~n:1 ~t:1 ~d:2 ~h:1 ~w:1 ~c:1 in
  let key_shape = Vec6.shape ~n:1 ~t:1 ~d:2 ~h:1 ~w:2 ~c:1 in
  let mask_shape = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:1 ~w:1 ~c:2 in
  let query = Tensor.materialize query_shape (fun _ -> 0.) in
  let key = Tensor.materialize key_shape (fun _ -> 0.) in
  let value =
    Tensor.materialize key_shape (fun c ->
        let base = if col c = 0 then 5. else 7. in
        if bat c = 0 then base else base *. 10.)
  in
  let mask =
    Tensor.materialize mask_shape (fun c -> if chan c = 0 then 0. else -2.)
  in
  Format.printf "%a@." (pp_result Tensor.pp)
    (run_sdpa
       { Attention.Sdpa.scale = Attention.Sdpa.Scale.Explicit 1.0 }
       ~query_shape ~key_shape ~mask_shape ~query ~key ~value ~mask);
  [%expect {| tensor f32 [D=2 H=1 W=1 C=1] {5.23841, 52.3841} |}]

(* op8-impl-review.md P1 (verified against source, fixed): [output_shape]
   checked D/H/C/W agreement across query/key/value but never N or T, and
   [Compute] reads key/value at the output coordinate's UNCHANGED N/T (never
   reduced through [broadcast_coord], exactly like D/H). A standalone or
   JSON-decoded graph with query.N=2, key.N=value.N=1 passed shape inference
   and then read key/value out of bounds evaluating output batch N=1; the
   reverse (key.N=2, query.N=1) silently ignored half of key/value. Both
   directions are now rejected here, the same as a D or H mismatch. *)
let%expect_test "Direct: sdpa — N and T must agree across query/key/value" =
  let ok = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:1 ~w:1 ~c:1 in
  let bad_n = Vec6.shape ~n:2 ~t:1 ~d:1 ~h:1 ~w:1 ~c:1 in
  let bad_t = Vec6.shape ~n:1 ~t:2 ~d:1 ~h:1 ~w:1 ~c:1 in
  Format.printf "query.N=2, key.N=1: %a@." (pp_result Vec6.pp_shape)
    (Attention.Sdpa.output_shape ~query_shape:bad_n ~key_shape:ok
       ~value_shape:ok ~mask_shape:None);
  Format.printf "key.N=2, query.N=1: %a@." (pp_result Vec6.pp_shape)
    (Attention.Sdpa.output_shape ~query_shape:ok ~key_shape:bad_n
       ~value_shape:ok ~mask_shape:None);
  Format.printf "value.N=2, query.N=1: %a@." (pp_result Vec6.pp_shape)
    (Attention.Sdpa.output_shape ~query_shape:ok ~key_shape:ok
       ~value_shape:bad_n ~mask_shape:None);
  Format.printf "query.T=2, key.T=1: %a@." (pp_result Vec6.pp_shape)
    (Attention.Sdpa.output_shape ~query_shape:bad_t ~key_shape:ok
       ~value_shape:ok ~mask_shape:None);
  [%expect
    {|
    query.N=2, key.N=1: sdpa: N extent must agree (query vs key): 2 vs 1
    key.N=2, query.N=1: sdpa: N extent must agree (query vs key): 1 vs 2
    value.N=2, query.N=1: sdpa: N extent must agree (query vs value): 1 vs 2
    query.T=2, key.T=1: sdpa: T extent must agree (query vs key): 2 vs 1
    |}]

(* The total-work bound now includes N/T too (op8-impl-review.md P1's second
   half): without them, a graph could inflate real work by N*T while the
   bound only ever saw D*H*Wq*Wk*E*E. N=1024 alone reaches the same
   6-factor product the F12 counterexample used ([D=H=1, Wq=Wk=E=Ev=1024]),
   so folding N in must reject it too. *)
let%expect_test "Direct: sdpa — total-work bound counts N and T" =
  let big = Vec6.shape ~n:1024 ~t:1 ~d:1 ~h:1 ~w:1024 ~c:1024 in
  let big_kv = Vec6.shape ~n:1024 ~t:1 ~d:1 ~h:1 ~w:1024 ~c:1024 in
  Format.printf "%a@." (pp_result Vec6.pp_shape)
    (Attention.Sdpa.output_shape ~query_shape:big ~key_shape:big_kv
       ~value_shape:big_kv ~mask_shape:None);
  [%expect
    {| sdpa: total work N*T*D*H*Wq*Wk*E*E (score, row max and denominator are recomputed per output feature) exceeds 2147483648 after folding in the head-dimension extent E (running product 1073741824, this factor 1024) |}]
