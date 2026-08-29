(* The Group 5 activations (silu, sigmoid, hardsigmoid, hardswish, gelu
   exact/tanh) evaluated directly, including their float32 boundary/threshold
   fixtures. Split from compute_test.ml. *)

open Compute_fixtures

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
