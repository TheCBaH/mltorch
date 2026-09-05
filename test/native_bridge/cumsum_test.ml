(* aten.cumsum.default: ATen as the oracle. Split out of dispatch_test.ml
   (which crossed the tracked 1000-line ceiling, scripts/check-file-size.sh)
   rather than folded into it. *)

open Helpers

let%expect_test "verify: cumsum.default dim=1" =
  let x = float_tensor [ 2; 3 ] [ 1.; 2.; 3.; 4.; 5.; 6. ] in
  verify_print ~target:"torch.ops.aten.cumsum.default"
    ~bindings:[ ("self", x) ]
    ~inputs:[ in_tensor "self"; in_int "dim" 1 ];
  [%expect {| aten and native agree |}]

(* Negative [dim] normalizes the same as every other single-[dim] arm
   ([softmax.int] in dispatch_test.ml, [slice.Tensor], [unbind.int]): -1 is
   the last axis, here the same one [dim=1] names above. *)
let%expect_test "verify: cumsum.default dim=-1 normalizes to the last axis" =
  let x = float_tensor [ 2; 3 ] [ 1.; 2.; 3.; 4.; 5.; 6. ] in
  verify_print ~target:"torch.ops.aten.cumsum.default"
    ~bindings:[ ("self", x) ]
    ~inputs:[ in_tensor "self"; in_int "dim" (-1) ];
  [%expect {| aten and native agree |}]

let%expect_test "verify: cumsum.default dim=0" =
  let x = float_tensor [ 2; 3 ] [ 1.; 2.; 3.; 4.; 5.; 6. ] in
  verify_print ~target:"torch.ops.aten.cumsum.default"
    ~bindings:[ ("self", x) ]
    ~inputs:[ in_tensor "self"; in_int "dim" 0 ];
  [%expect {| aten and native agree |}]

let%expect_test "dispatch: cumsum.default rejects an out-of-range dim" =
  let x = float_tensor [ 2; 3 ] [ 1.; 2.; 3.; 4.; 5.; 6. ] in
  List.iter
    (fun d ->
      dispatch_print ~target:"torch.ops.aten.cumsum.default"
        ~bindings:[ ("self", x) ]
        ~inputs:[ in_tensor "self"; in_int "dim" d ]
        ~noutputs:1)
    [ 7; -3 ];
  [%expect
    {|
    error: cumsum.default: invalid dimension 7 for rank 2
    error: cumsum.default: invalid dimension -3 for rank 2 |}]

(* Unlike [softmax.int] (which rejects ANY supplied [dtype]), a FLOAT target
   is accepted -- Native's compute domain is always float, so this is the
   identity, exactly [_to_copy.default]'s own [Float] case: EdgeNeXt's own
   corpus occurrence casts a bool mask to FLOAT this way.
   [dispatch_print], not [verify_print]: the
   generic (schema-driven) real-ATen decode this repo's [Interp_dispatch]
   uses for every [ScalarType?] argument raises on ANY present value, the
   same [scalar_type_opt_arg] restriction [softmax.int] hits -- so a
   present-and-accepted dtype cannot go through the real-ATen comparison
   [verify_print] runs; the identity itself is grounded by the dim=1 test
   above, which computes the same values with [dtype] omitted. *)
let%expect_test "dispatch: cumsum.default accepts a FLOAT dtype" =
  let x = float_tensor [ 2; 3 ] [ 1.; 2.; 3.; 4.; 5.; 6. ] in
  dispatch_print ~target:"torch.ops.aten.cumsum.default"
    ~bindings:[ ("self", x) ]
    ~inputs:
      [
        in_tensor "self";
        in_int "dim" 1;
        in_scalar_type "dtype" PT.ScalarType.FLOAT;
      ]
    ~noutputs:1;
  [%expect {| tensor f32 [W=2 C=3] {1, 3, 6, 4, 9, 15} |}]

let%expect_test "dispatch: cumsum.default rejects a non-FLOAT dtype" =
  let x = float_tensor [ 2; 3 ] [ 1.; 2.; 3.; 4.; 5.; 6. ] in
  dispatch_print ~target:"torch.ops.aten.cumsum.default"
    ~bindings:[ ("self", x) ]
    ~inputs:
      [
        in_tensor "self";
        in_int "dim" 1;
        in_scalar_type "dtype" PT.ScalarType.DOUBLE;
      ]
    ~noutputs:1;
  [%expect {| error: cumsum.default: only default/FLOAT dtype are supported |}]
