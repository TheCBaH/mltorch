(* Group 5 activations: silu, hardsigmoid, hardswish (op5-impl). Split from the former native_bridge_test.ml; promote with [dune promote test/native_bridge/activation_test.ml]. *)

open Helpers

(* ---- Group 5 activations: silu, hardsigmoid, hardswish (op5-impl) -------- *)

let%expect_test "dispatch: silu.default elementwise" =
  let a = float_tensor [ 2; 2 ] [ -6.; -0.5; 0.5; 6. ] in
  dispatch_print ~target:"torch.ops.aten.silu.default"
    ~bindings:[ ("self", a) ]
    ~inputs:[ in_tensor "self" ]
    ~noutputs:1;
  [%expect {| tensor f32 [W=2 C=2] {-0.0148357, -0.18877, 0.31123, 5.98516} |}]

let%expect_test "dispatch: hardsigmoid.default elementwise" =
  let a = float_tensor [ 2; 2 ] [ -6.; -0.5; 0.5; 6. ] in
  dispatch_print ~target:"torch.ops.aten.hardsigmoid.default"
    ~bindings:[ ("self", a) ]
    ~inputs:[ in_tensor "self" ]
    ~noutputs:1;
  [%expect {| tensor f32 [W=2 C=2] {0, 0.416667, 0.583333, 1} |}]

let%expect_test "dispatch: sigmoid.default elementwise" =
  let a = float_tensor [ 2; 2 ] [ -6.; -0.5; 0.5; 6. ] in
  dispatch_print ~target:"torch.ops.aten.sigmoid.default"
    ~bindings:[ ("self", a) ]
    ~inputs:[ in_tensor "self" ]
    ~noutputs:1;
  [%expect
    {| tensor f32 [W=2 C=2] {0.00247262, 0.377541, 0.622459, 0.997527} |}]

let%expect_test "dispatch: gelu.default elementwise" =
  let a = float_tensor [ 2; 2 ] [ -6.; -0.5; 0.5; 6. ] in
  dispatch_print ~target:"torch.ops.aten.gelu.default"
    ~bindings:[ ("self", a) ]
    ~inputs:[ in_tensor "self" ]
    ~noutputs:1;
  [%expect {| tensor f32 [W=2 C=2] {-5.94073e-09, -0.154269, 0.345731, 6} |}]

let%expect_test "dispatch: gelu.default elementwise (tanh)" =
  let a = float_tensor [ 2; 2 ] [ -6.; -0.5; 0.5; 6. ] in
  dispatch_print ~target:"torch.ops.aten.gelu.default"
    ~bindings:[ ("self", a) ]
    ~inputs:[ in_tensor "self"; in_string "approximate" "tanh" ]
    ~noutputs:1;
  [%expect {| tensor f32 [W=2 C=2] {-8.43965e-11, -0.154286, 0.345714, 6} |}]

let%expect_test "dispatch: leaky_relu.default elementwise" =
  let a = float_tensor [ 2; 2 ] [ -2.; -0.5; 0.; 3. ] in
  dispatch_print ~target:"torch.ops.aten.leaky_relu.default"
    ~bindings:[ ("self", a) ]
    ~inputs:[ in_tensor "self"; in_float "negative_slope" 0.2 ]
    ~noutputs:1;
  [%expect {| tensor f32 [W=2 C=2] {-0.4, -0.1, 0, 3} |}]

let%expect_test "dispatch: zeros.default preserves default and DOUBLE dtypes" =
  dispatch_print ~target:"torch.ops.aten.zeros.default" ~bindings:[]
    ~inputs:[ in_ints "size" [ 2; 3 ] ]
    ~noutputs:1;
  dispatch_print ~target:"torch.ops.aten.zeros.default" ~bindings:[]
    ~inputs:
      [ in_ints "size" [ 2; 3 ]; in_scalar_type "dtype" PT.ScalarType.DOUBLE ]
    ~noutputs:1;
  [%expect
    {|
    tensor f32 [W=2 C=3] {0, 0, 0, 0, 0, 0}
    tensor f64 [W=2 C=3] {0, 0, 0, 0, 0, 0} |}]

let%expect_test "verify: zeros.default against real ATen" =
  verify_print ~target:"torch.ops.aten.zeros.default" ~bindings:[]
    ~inputs:[ in_ints "size" [ 2; 3 ] ];
  [%expect {|
    aten and native agree |}]

let%expect_test "dispatch: arange preserves Long and Float factory dtypes" =
  dispatch_print ~target:"torch.ops.aten.arange.default" ~bindings:[]
    ~inputs:[ in_int "end" 5 ]
    ~noutputs:1;
  dispatch_print ~target:"torch.ops.aten.arange.start" ~bindings:[]
    ~inputs:
      [
        in_float "start" 0.5;
        in_int "end" 4;
        in_scalar_type "dtype" PT.ScalarType.FLOAT;
      ]
    ~noutputs:1;
  [%expect
    {|
    tensor i64 [C=5] {0, 1, 2, 3, 4}
    tensor f32 [C=4] {0.5, 1.5, 2.5, 3.5} |}]

let%expect_test "verify: arange.start Float against real ATen" =
  verify_print ~target:"torch.ops.aten.arange.start" ~bindings:[]
    ~inputs:[ in_float "start" 0.5; in_int "end" 4 ];
  [%expect {|
    aten and native agree |}]

let%expect_test "dispatch: mul.Tensor with a serialized scalar" =
  let a = float_tensor [ 2; 2 ] [ -6.; -0.5; 0.5; 6. ] in
  dispatch_print ~target:"torch.ops.aten.mul.Tensor"
    ~bindings:[ ("self", a) ]
    ~inputs:[ in_tensor "self"; in_int "other" 3 ]
    ~noutputs:1;
  dispatch_print ~target:"torch.ops.aten.mul.Tensor"
    ~bindings:[ ("self", a) ]
    ~inputs:[ in_tensor "self"; in_float "other" 0.1 ]
    ~noutputs:1;
  [%expect
    {|
    tensor f32 [W=2 C=2] {-18, -1.5, 1.5, 18}
    tensor f32 [W=2 C=2] {-0.6, -0.05, 0.05, 0.6} |}]

let%expect_test "dispatch: mul.Scalar" =
  let a = float_tensor [ 2; 2 ] [ -6.; -0.5; 0.5; 6. ] in
  dispatch_print ~target:"torch.ops.aten.mul.Scalar"
    ~bindings:[ ("self", a) ]
    ~inputs:[ in_tensor "self"; in_int "other" 3 ]
    ~noutputs:1;
  dispatch_print ~target:"torch.ops.aten.mul.Scalar"
    ~bindings:[ ("self", a) ]
    ~inputs:[ in_tensor "self"; in_float "other" 0.1 ]
    ~noutputs:1;
  [%expect
    {|
    tensor f32 [W=2 C=2] {-18, -1.5, 1.5, 18}
    tensor f32 [W=2 C=2] {-0.6, -0.05, 0.05, 0.6} |}]

let%expect_test "dispatch: hardswish.default elementwise" =
  let a = float_tensor [ 2; 2 ] [ -6.; -0.5; 0.5; 6. ] in
  dispatch_print ~target:"torch.ops.aten.hardswish.default"
    ~bindings:[ ("self", a) ]
    ~inputs:[ in_tensor "self" ]
    ~noutputs:1;
  [%expect {| tensor f32 [W=2 C=2] {-0, -0.208333, 0.291667, 6} |}]

(* Real-ATen agreement, functional AND in-place spellings both, over the
   float32-threshold-neighbourhood fixture (op5.md). [verify_print] runs the
   node through BOTH [Interp_dispatch] (real ATen) and [Op_bridge] +
   [Eval_direct] (native) and compares element-wise -- silence means
   agreement. The in-place spellings are the coverage that matters: efficientnet
   serialises [silu_], never [silu] (op5-impl F1), and getting this to agree
   required fixing [Interp_verify.dispatch] to capture native's read of [self]
   BEFORE the ATen in-place call mutates it through the same handle -- the
   previous order silently compared native's output against ATen's INPUT
   already overwritten with ATen's own output, invisible for relu_/hardtanh_
   only because clamping is idempotent (op5-impl). *)
let activation_fixture =
  [
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
  ]

let%expect_test "verify: silu against real ATen, functional and in-place" =
  let a = float_tensor [ 2; 7 ] activation_fixture in
  verify_print ~target:"torch.ops.aten.silu.default"
    ~bindings:[ ("self", a) ]
    ~inputs:[ in_tensor "self" ];
  verify_print ~target:"torch.ops.aten.silu_.default"
    ~bindings:[ ("self", a) ]
    ~inputs:[ in_tensor "self" ];
  [%expect {|
    aten and native agree
    aten and native agree |}]

let%expect_test "verify: hardsigmoid against real ATen, functional and in-place"
    =
  let a = float_tensor [ 2; 7 ] activation_fixture in
  verify_print ~target:"torch.ops.aten.hardsigmoid.default"
    ~bindings:[ ("self", a) ]
    ~inputs:[ in_tensor "self" ];
  verify_print ~target:"torch.ops.aten.hardsigmoid_.default"
    ~bindings:[ ("self", a) ]
    ~inputs:[ in_tensor "self" ];
  [%expect {|
    aten and native agree
    aten and native agree |}]

let%expect_test "verify: hardswish against real ATen, functional and in-place" =
  let a = float_tensor [ 2; 7 ] activation_fixture in
  verify_print ~target:"torch.ops.aten.hardswish.default"
    ~bindings:[ ("self", a) ]
    ~inputs:[ in_tensor "self" ];
  verify_print ~target:"torch.ops.aten.hardswish_.default"
    ~bindings:[ ("self", a) ]
    ~inputs:[ in_tensor "self" ];
  [%expect {|
    aten and native agree
    aten and native agree |}]

(* Saturation boundary fixture, the same ask [gelu_boundary_fixture] answers
   for gelu: large +/- magnitude values reaching into f32's denormal range
   at one tail and exact
   saturation at the other, plus near-zero and signed zero -- the same
   fixture [compute_test.ml]'s "Direct: sigmoid boundary cases" pins, here
   checked against real ATen instead of a hand-verified golden. Unlike gelu,
   sigmoid's formula has no piecewise/polynomial regime of its own --
   1/(1+exp(-x)) is one expression everywhere -- so this saturation/denormal
   boundary is the only one worth pinning. *)
let sigmoid_boundary_fixture =
  [
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
  ]

let%expect_test "verify: sigmoid against real ATen, functional" =
  let a = float_tensor [ 2; 7 ] activation_fixture in
  verify_print ~target:"torch.ops.aten.sigmoid.default"
    ~bindings:[ ("self", a) ]
    ~inputs:[ in_tensor "self" ];
  let b = float_tensor [ 2; 7 ] sigmoid_boundary_fixture in
  verify_print ~target:"torch.ops.aten.sigmoid.default"
    ~bindings:[ ("self", b) ]
    ~inputs:[ in_tensor "self" ];
  [%expect {|
    aten and native agree
    aten and native agree |}]

(* The boundary fixture (review point 8): large-magnitude positive and
   negative inputs (erf saturates to +/-1, so the negative tail loses relative
   precision), signed zero, and values near zero (the polynomial's [t] term
   near its own regime). Only ATen comparison proves the approximation's
   accuracy -- the Direct-vs-Symbolic walk only proves staging agreement,
   since both sides share one [erf] implementation. *)
let gelu_boundary_fixture =
  [ -20.; -5.; -1.; -0.5; -0.1; -1e-8; -0.; 0.; 1e-8; 0.1; 0.5; 1.; 5.; 20. ]

let%expect_test "verify: gelu against real ATen, functional" =
  let a = float_tensor [ 2; 7 ] activation_fixture in
  verify_print ~target:"torch.ops.aten.gelu.default"
    ~bindings:[ ("self", a) ]
    ~inputs:[ in_tensor "self" ];
  let b = float_tensor [ 2; 7 ] gelu_boundary_fixture in
  verify_print ~target:"torch.ops.aten.gelu.default"
    ~bindings:[ ("self", b) ]
    ~inputs:[ in_tensor "self" ];
  [%expect {|
    aten and native agree
    aten and native agree |}]

let%expect_test "verify: leaky_relu against real ATen" =
  let a = float_tensor [ 2; 7 ] activation_fixture in
  verify_print ~target:"torch.ops.aten.leaky_relu.default"
    ~bindings:[ ("self", a) ]
    ~inputs:[ in_tensor "self" ];
  verify_print ~target:"torch.ops.aten.leaky_relu.default"
    ~bindings:[ ("self", a) ]
    ~inputs:[ in_tensor "self"; in_float "negative_slope" 0.2 ];
  [%expect {|
    aten and native agree
    aten and native agree |}]

(* Same two fixtures, but [approximate="tanh"]: proves the tanh formula
   against ATen's own tanh kernel, not just against Direct/Symbolic agreeing
   with each other (native_op_walk's gelu walk only proves the latter). *)
let%expect_test "verify: gelu (tanh) against real ATen, functional" =
  let a = float_tensor [ 2; 7 ] activation_fixture in
  verify_print ~target:"torch.ops.aten.gelu.default"
    ~bindings:[ ("self", a) ]
    ~inputs:[ in_tensor "self"; in_string "approximate" "tanh" ];
  let b = float_tensor [ 2; 7 ] gelu_boundary_fixture in
  verify_print ~target:"torch.ops.aten.gelu.default"
    ~bindings:[ ("self", b) ]
    ~inputs:[ in_tensor "self"; in_string "approximate" "tanh" ];
  [%expect {|
    aten and native agree
    aten and native agree |}]

let%expect_test "dispatch: gelu.default rejects an unrecognized approximate" =
  let a = float_tensor [ 2; 2 ] [ -6.; -0.5; 0.5; 6. ] in
  dispatch_print ~target:"torch.ops.aten.gelu.default"
    ~bindings:[ ("self", a) ]
    ~inputs:[ in_tensor "self"; in_string "approximate" "unsupported" ]
    ~noutputs:1;
  [%expect
    {| error: gelu approximate=unsupported is not supported (only "none" or "tanh") |}]

let%expect_test "verify: mul.Tensor with a serialized scalar against real ATen"
    =
  let a = float_tensor [ 2; 3 ] [ -4.; -3.; 0.; 3.; 4.; 10. ] in
  verify_print ~target:"torch.ops.aten.mul.Tensor"
    ~bindings:[ ("self", a) ]
    ~inputs:[ in_tensor "self"; in_int "other" 3 ];
  verify_print ~target:"torch.ops.aten.mul.Tensor"
    ~bindings:[ ("self", a) ]
    ~inputs:[ in_tensor "self"; in_float "other" 0.1 ];
  [%expect {|
    aten and native agree
    aten and native agree |}]

let%expect_test "verify: mul.Scalar against real ATen" =
  let a = float_tensor [ 2; 3 ] [ -4.; -3.; 0.; 3.; 4.; 10. ] in
  verify_print ~target:"torch.ops.aten.mul.Scalar"
    ~bindings:[ ("self", a) ]
    ~inputs:[ in_tensor "self"; in_int "other" 3 ];
  verify_print ~target:"torch.ops.aten.mul.Scalar"
    ~bindings:[ ("self", a) ]
    ~inputs:[ in_tensor "self"; in_float "other" 0.1 ];
  [%expect {|
    aten and native agree
    aten and native agree |}]

let%expect_test "dispatch: view.default contiguous reshape (no permute)" =
  (* NCHW [1,1,2,3] (values 0..5) viewed as [3,2]. of_aten inputs are already
     ATen-row-major, so the graph is a single reshape (no surrounding permute);
     a contiguous reshape leaves the flat buffer unchanged. *)
  let x = float_tensor [ 1; 1; 2; 3 ] (List.init 6 float_of_int) in
  dispatch_print_with_graph ~print_graph:true
    ~target:"torch.ops.aten.view.default"
    ~bindings:[ ("self", x) ]
    ~inputs:[ in_tensor "self"; in_ints "size" [ 3; 2 ] ]
    ~noutputs:1;
  [%expect
    {|
    graph
    inputs: [t0 f32 [W=2 C=3] ->[n0]]
    nodes:
      n0: [t1 f32 [W=3 C=2]] = reshape x=t0 params={shape=[W=3 C=2]}
    outputs: [t1 f32 [W=3 C=2] <-n0]
    tensor f32 [W=3 C=2] {0, 1, 2, 3, 4, 5} |}]
