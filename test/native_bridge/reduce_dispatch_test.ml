(* Op_bridge.dispatch native compute for the reduction family
   (mean.dim/amax.default/sum.dim_IntList/linalg_vector_norm.default/
   softmax.int), evaluated directly. Split out of dispatch_test.ml (which
   crossed the tracked 1000-line ceiling, scripts/check-file-size.sh) --
   mirrors op_bridge_reduce.ml's own family split on the source side.
   Promote with [dune promote test/native_bridge/reduce_dispatch_test.ml]. *)

open Helpers

(* The bridge exposes the live values result of [max.dim] (its own [MaxDim]
   op, NOT [Amax]: both of [MaxDim]'s outputs fold with [Max_op.pool_better]
   rather than [Amax]'s [Float_max], so the two agree on ordinary numbers but
   would not on a NaN -- see op_bridge_reduce.ml's own comment); the
   serialized importer owns the fixed-tuple dead-index policy. Two modes pin
   [MaxDim]'s shared singleton-axis lowering and its packing rule -- like
   every other case here, against hand-derived values (Direct evaluation of
   the dispatched graph, not an ATen oracle: see helpers.ml's own comment). *)
let%expect_test "dispatch: max.dim values, keepdim true and false" =
  let x = float_tensor [ 2; 3 ] [ 0.; 2.; 1.; 5.; 3.; 4. ] in
  List.iter
    (fun keepdim ->
      dispatch_print ~target:"torch.ops.aten.max.dim"
        ~bindings:[ ("self", x) ]
        ~inputs:[ in_tensor "self"; in_int "dim" 1; in_bool "keepdim" keepdim ]
        ~noutputs:1)
    [ true; false ];
  [%expect {|
    tensor f32 [W=2 C=1] {2, 5}
    tensor f32 [C=2] {2, 5} |}]

let%expect_test "dispatch: max.dim with a tie along the reduced axis" =
  (* Row 0 ties at 5 (positions 0 and 2); row 1 has no tie -- pins that the
     tie does not perturb the reported VALUE. The dead index this bridge arm
     discards is never exposed here to compare; [Max_op.pool_better]'s
     "first index wins" tie contract is pinned directly by the Direct
     fixtures in reduce_test.ml instead. *)
  let x = float_tensor [ 2; 3 ] [ 5.; 3.; 5.; 1.; 4.; 2. ] in
  dispatch_print ~target:"torch.ops.aten.max.dim"
    ~bindings:[ ("self", x) ]
    ~inputs:[ in_tensor "self"; in_int "dim" 1; in_bool "keepdim" false ]
    ~noutputs:1;
  [%expect {| tensor f32 [C=2] {5, 4} |}]

let%expect_test "dispatch: mean.dim dim=[1] keepdim=true" =
  let x = float_tensor [ 2; 3 ] [ 0.; 1.; 2.; 3.; 4.; 5. ] in
  dispatch_print ~target:"torch.ops.aten.mean.dim"
    ~bindings:[ ("self", x) ]
    ~inputs:[ in_tensor "self"; in_ints "dim" [ 1 ]; in_bool "keepdim" true ]
    ~noutputs:1;
  [%expect {| tensor f32 [W=2 C=1] {1, 4} |}]

let%expect_test "dispatch: mean.dim dim=[1] keepdim=false" =
  let x = float_tensor [ 2; 3 ] [ 0.; 1.; 2.; 3.; 4.; 5. ] in
  dispatch_print ~target:"torch.ops.aten.mean.dim"
    ~bindings:[ ("self", x) ]
    ~inputs:[ in_tensor "self"; in_ints "dim" [ 1 ]; in_bool "keepdim" false ]
    ~noutputs:1;
  [%expect {| tensor f32 [C=2] {1, 4} |}]

let%expect_test "dispatch: mean.dim dim=[] reduces over all dims" =
  let x = float_tensor [ 2; 3 ] [ 0.; 1.; 2.; 3.; 4.; 5. ] in
  dispatch_print ~target:"torch.ops.aten.mean.dim"
    ~bindings:[ ("self", x) ]
    ~inputs:[ in_tensor "self"; in_ints "dim" []; in_bool "keepdim" false ]
    ~noutputs:1;
  [%expect {| tensor f32 [C=1] {2.5} |}]

let%expect_test "dispatch: mean.dim omitted dim reduces over all dims" =
  let x = float_tensor [ 2; 3 ] [ 0.; 1.; 2.; 3.; 4.; 5. ] in
  dispatch_print ~target:"torch.ops.aten.mean.dim"
    ~bindings:[ ("self", x) ]
    ~inputs:[ in_tensor "self"; in_bool "keepdim" false ]
    ~noutputs:1;
  [%expect {| tensor f32 [C=1] {2.5} |}]

(* [Aten_shape.axis_of_dim] asserts its range and raises; before commit 0 this
   escaped [Op_bridge.dispatch] as an uncaught [Invalid_argument] rather than
   the typed row every other bad-argument arm returns. *)
let%expect_test "dispatch: mean.dim rejects an out-of-range dim" =
  let x = float_tensor [ 2; 3 ] [ 0.; 1.; 2.; 3.; 4.; 5. ] in
  List.iter
    (fun d ->
      dispatch_print ~target:"torch.ops.aten.mean.dim"
        ~bindings:[ ("self", x) ]
        ~inputs:
          [ in_tensor "self"; in_ints "dim" [ d ]; in_bool "keepdim" false ]
        ~noutputs:1)
    [ 7; -3 ];
  [%expect
    {|
    error: mean.dim: invalid dimension 7 for rank 2
    error: mean.dim: invalid dimension -3 for rank 2 |}]

let%expect_test "dispatch: amax.default dim=[1] keepdim=true" =
  let x = float_tensor [ 2; 3 ] [ 0.; 1.; 2.; 3.; 4.; 5. ] in
  dispatch_print ~target:"torch.ops.aten.amax.default"
    ~bindings:[ ("self", x) ]
    ~inputs:[ in_tensor "self"; in_ints "dim" [ 1 ]; in_bool "keepdim" true ]
    ~noutputs:1;
  [%expect {| tensor f32 [W=2 C=1] {2, 5} |}]

let%expect_test "dispatch: amax.default dim=[1] keepdim=false" =
  let x = float_tensor [ 2; 3 ] [ 0.; 1.; 2.; 3.; 4.; 5. ] in
  dispatch_print ~target:"torch.ops.aten.amax.default"
    ~bindings:[ ("self", x) ]
    ~inputs:[ in_tensor "self"; in_ints "dim" [ 1 ]; in_bool "keepdim" false ]
    ~noutputs:1;
  [%expect {| tensor f32 [C=2] {2, 5} |}]

let%expect_test "dispatch: amax.default dim=[] reduces over all dims" =
  let x = float_tensor [ 2; 3 ] [ 0.; 1.; 2.; 3.; 4.; 5. ] in
  dispatch_print ~target:"torch.ops.aten.amax.default"
    ~bindings:[ ("self", x) ]
    ~inputs:[ in_tensor "self"; in_ints "dim" []; in_bool "keepdim" false ]
    ~noutputs:1;
  [%expect {| tensor f32 [C=1] {5} |}]

let%expect_test "dispatch: amax.default omitted dim reduces over all dims" =
  let x = float_tensor [ 2; 3 ] [ 0.; 1.; 2.; 3.; 4.; 5. ] in
  dispatch_print ~target:"torch.ops.aten.amax.default"
    ~bindings:[ ("self", x) ]
    ~inputs:[ in_tensor "self"; in_bool "keepdim" false ]
    ~noutputs:1;
  [%expect {| tensor f32 [C=1] {5} |}]

let%expect_test "dispatch: amax.default rejects an out-of-range dim" =
  let x = float_tensor [ 2; 3 ] [ 0.; 1.; 2.; 3.; 4.; 5. ] in
  List.iter
    (fun d ->
      dispatch_print ~target:"torch.ops.aten.amax.default"
        ~bindings:[ ("self", x) ]
        ~inputs:
          [ in_tensor "self"; in_ints "dim" [ d ]; in_bool "keepdim" false ]
        ~noutputs:1)
    [ 7; -3 ];
  [%expect
    {|
    error: amax.default: invalid dimension 7 for rank 2
    error: amax.default: invalid dimension -3 for rank 2 |}]

let%expect_test "dispatch: sum.dim_IntList dim=[1] keepdim=true" =
  let x = float_tensor [ 2; 3 ] [ 0.; 1.; 2.; 3.; 4.; 5. ] in
  dispatch_print ~target:"torch.ops.aten.sum.dim_IntList"
    ~bindings:[ ("self", x) ]
    ~inputs:[ in_tensor "self"; in_ints "dim" [ 1 ]; in_bool "keepdim" true ]
    ~noutputs:1;
  [%expect {| tensor f32 [W=2 C=1] {3, 12} |}]

let%expect_test "dispatch: sum.dim_IntList dim=[1] keepdim=false" =
  let x = float_tensor [ 2; 3 ] [ 0.; 1.; 2.; 3.; 4.; 5. ] in
  dispatch_print ~target:"torch.ops.aten.sum.dim_IntList"
    ~bindings:[ ("self", x) ]
    ~inputs:[ in_tensor "self"; in_ints "dim" [ 1 ]; in_bool "keepdim" false ]
    ~noutputs:1;
  [%expect {| tensor f32 [C=2] {3, 12} |}]

let%expect_test "dispatch: sum.dim_IntList dim=[] reduces over all dims" =
  let x = float_tensor [ 2; 3 ] [ 0.; 1.; 2.; 3.; 4.; 5. ] in
  dispatch_print ~target:"torch.ops.aten.sum.dim_IntList"
    ~bindings:[ ("self", x) ]
    ~inputs:[ in_tensor "self"; in_ints "dim" []; in_bool "keepdim" false ]
    ~noutputs:1;
  [%expect {| tensor f32 [C=1] {15} |}]

let%expect_test "dispatch: sum.dim_IntList omitted dim reduces over all dims" =
  let x = float_tensor [ 2; 3 ] [ 0.; 1.; 2.; 3.; 4.; 5. ] in
  dispatch_print ~target:"torch.ops.aten.sum.dim_IntList"
    ~bindings:[ ("self", x) ]
    ~inputs:[ in_tensor "self"; in_bool "keepdim" false ]
    ~noutputs:1;
  [%expect {| tensor f32 [C=1] {15} |}]

let%expect_test "dispatch: sum.dim_IntList rejects an out-of-range dim" =
  let x = float_tensor [ 2; 3 ] [ 0.; 1.; 2.; 3.; 4.; 5. ] in
  List.iter
    (fun d ->
      dispatch_print ~target:"torch.ops.aten.sum.dim_IntList"
        ~bindings:[ ("self", x) ]
        ~inputs:
          [ in_tensor "self"; in_ints "dim" [ d ]; in_bool "keepdim" false ]
        ~noutputs:1)
    [ 7; -3 ];
  [%expect
    {|
    error: sum.dim_IntList: invalid dimension 7 for rank 2
    error: sum.dim_IntList: invalid dimension -3 for rank 2 |}]

let%expect_test "dispatch: sum.dim_IntList rejects a supplied dtype" =
  let x = float_tensor [ 2; 3 ] [ 0.; 1.; 2.; 3.; 4.; 5. ] in
  dispatch_print ~target:"torch.ops.aten.sum.dim_IntList"
    ~bindings:[ ("self", x) ]
    ~inputs:
      [
        in_tensor "self";
        in_ints "dim" [ 1 ];
        in_bool "keepdim" false;
        PT.NamedArgument.make "dtype"
          (PT.Argument.Scalar_type PT.ScalarType.DOUBLE) None;
      ]
    ~noutputs:1;
  [%expect {| error: unsupported scalar_type argument "dtype" |}]

let%expect_test "dispatch: linalg_vector_norm.default dim=[1] keepdim=true" =
  let x = float_tensor [ 2; 3 ] [ 0.; 1.; 2.; 3.; 4.; 5. ] in
  dispatch_print ~target:"torch.ops.aten.linalg_vector_norm.default"
    ~bindings:[ ("self", x) ]
    ~inputs:[ in_tensor "self"; in_ints "dim" [ 1 ]; in_bool "keepdim" true ]
    ~noutputs:1;
  [%expect {| tensor f32 [W=2 C=1] {2.23607, 7.07107} |}]

let%expect_test "dispatch: linalg_vector_norm.default dim=[1] keepdim=false" =
  let x = float_tensor [ 2; 3 ] [ 0.; 1.; 2.; 3.; 4.; 5. ] in
  dispatch_print ~target:"torch.ops.aten.linalg_vector_norm.default"
    ~bindings:[ ("self", x) ]
    ~inputs:[ in_tensor "self"; in_ints "dim" [ 1 ]; in_bool "keepdim" false ]
    ~noutputs:1;
  [%expect {| tensor f32 [C=2] {2.23607, 7.07107} |}]

let%expect_test
    "dispatch: linalg_vector_norm.default omitted dim/ord reduces over all dims"
    =
  let x = float_tensor [ 2; 3 ] [ 0.; 1.; 2.; 3.; 4.; 5. ] in
  dispatch_print ~target:"torch.ops.aten.linalg_vector_norm.default"
    ~bindings:[ ("self", x) ]
    ~inputs:[ in_tensor "self"; in_bool "keepdim" false ]
    ~noutputs:1;
  [%expect {| tensor f32 [C=1] {7.4162} |}]

let%expect_test "dispatch: linalg_vector_norm.default rejects a non-2 ord" =
  let x = float_tensor [ 2; 3 ] [ 0.; 1.; 2.; 3.; 4.; 5. ] in
  dispatch_print ~target:"torch.ops.aten.linalg_vector_norm.default"
    ~bindings:[ ("self", x) ]
    ~inputs:
      [
        in_tensor "self";
        in_float "ord" 1.0;
        in_ints "dim" [ 1 ];
        in_bool "keepdim" false;
      ]
    ~noutputs:1;
  [%expect
    {| error: linalg_vector_norm.default: only ord=2 is supported, got 1 |}]

let%expect_test "dispatch: linalg_vector_norm.default rejects a supplied dtype"
    =
  let x = float_tensor [ 2; 3 ] [ 0.; 1.; 2.; 3.; 4.; 5. ] in
  dispatch_print ~target:"torch.ops.aten.linalg_vector_norm.default"
    ~bindings:[ ("self", x) ]
    ~inputs:
      [
        in_tensor "self";
        in_ints "dim" [ 1 ];
        in_bool "keepdim" false;
        PT.NamedArgument.make "dtype"
          (PT.Argument.Scalar_type PT.ScalarType.DOUBLE) None;
      ]
    ~noutputs:1;
  [%expect {| error: unsupported scalar_type argument "dtype" |}]

let%expect_test "dispatch: softmax.int dim=1" =
  let x = float_tensor [ 2; 3 ] [ 0.; 1.; 2.; 3.; 4.; 5. ] in
  dispatch_print ~target:"torch.ops.aten.softmax.int"
    ~bindings:[ ("self", x) ]
    ~inputs:[ in_tensor "self"; in_int "dim" 1 ]
    ~noutputs:1;
  [%expect
    {| tensor f32 [W=2 C=3] {0.0900306, 0.244728, 0.665241, 0.0900306, 0.244728, 0.665241} |}]

(* Negative [dim] normalizes the same as every other single-[dim] arm
   ([slice.Tensor], [unbind.int]): -1 is the last axis, here the same one
   [dim=1] names above -- same values, proving the normalization rather than
   just that some diagnostic fires. *)
let%expect_test "dispatch: softmax.int dim=-1 normalizes to the last axis" =
  let x = float_tensor [ 2; 3 ] [ 0.; 1.; 2.; 3.; 4.; 5. ] in
  dispatch_print ~target:"torch.ops.aten.softmax.int"
    ~bindings:[ ("self", x) ]
    ~inputs:[ in_tensor "self"; in_int "dim" (-1) ]
    ~noutputs:1;
  [%expect
    {| tensor f32 [W=2 C=3] {0.0900306, 0.244728, 0.665241, 0.0900306, 0.244728, 0.665241} |}]

let%expect_test "dispatch: softmax.int rejects an out-of-range dim" =
  let x = float_tensor [ 2; 3 ] [ 0.; 1.; 2.; 3.; 4.; 5. ] in
  List.iter
    (fun d ->
      dispatch_print ~target:"torch.ops.aten.softmax.int"
        ~bindings:[ ("self", x) ]
        ~inputs:[ in_tensor "self"; in_int "dim" d ]
        ~noutputs:1)
    [ 7; -3 ];
  [%expect
    {|
    error: softmax.int: invalid dimension 7 for rank 2
    error: softmax.int: invalid dimension -3 for rank 2 |}]

let%expect_test "dispatch: softmax.int rejects a supplied dtype" =
  let x = float_tensor [ 2; 3 ] [ 0.; 1.; 2.; 3.; 4.; 5. ] in
  dispatch_print ~target:"torch.ops.aten.softmax.int"
    ~bindings:[ ("self", x) ]
    ~inputs:
      [
        in_tensor "self";
        in_int "dim" 1;
        PT.NamedArgument.make "dtype"
          (PT.Argument.Scalar_type PT.ScalarType.DOUBLE) None;
      ]
    ~noutputs:1;
  [%expect {| error: unsupported scalar_type argument "dtype" |}]
