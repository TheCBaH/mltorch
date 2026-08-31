(* copy.default, through the ATen-linked bridge. Binds directly to the
   existing Expand node -- see op_bridge_shape.ml's own comment for why.
   Promote with [dune promote test/native_bridge/copy_test.ml]. *)

open Helpers

let dispatch_copy ?(non_blocking = false) ~self ~src () =
  dispatch_print ~target:"torch.ops.aten.copy.default"
    ~bindings:[ ("self", self); ("src", src) ]
    ~inputs:
      [ in_tensor "self"; in_tensor "src"; in_bool "non_blocking" non_blocking ]
    ~noutputs:1

(* The corpus's own shape: self and src share one shape exactly, so the
   result is just src's own values -- self contributes only its shape, never
   read as data (its values here are deliberately DIFFERENT from src's, so a
   bridge that read self's data by mistake would show up immediately). *)
let%expect_test "dispatch: copy with matching shapes returns src's values" =
  let self = float_tensor [ 3 ] [ 9.; 9.; 9. ] in
  let src = float_tensor [ 3 ] [ 0.; 1.; 2. ] in
  dispatch_copy ~self ~src ();
  [%expect {| tensor f32 [C=3] {0, 1, 2} |}]

(* [non_blocking] carries no computational effect -- both spellings must
   build the identical result, the same [implicit] proof [expand_test.ml]
   makes for [expand.default]. *)
let%expect_test "dispatch: copy ignores non_blocking" =
  let self = float_tensor [ 3 ] [ 9.; 9.; 9. ] in
  let src = float_tensor [ 3 ] [ 0.; 1.; 2. ] in
  dispatch_copy ~self ~src ~non_blocking:false ();
  dispatch_copy ~self ~src ~non_blocking:true ();
  [%expect {|
    tensor f32 [C=3] {0, 1, 2}
    tensor f32 [C=3] {0, 1, 2} |}]

(* The genuine broadcast case: self's shape has extent 2 on W where src has
   1 -- no corpus occurrence exercises this, but the translation is the
   fully general broadcast, not an overfit to the observed equal-shape
   case. *)
let%expect_test "dispatch: copy broadcasts src to self's shape" =
  let self = float_tensor [ 2; 3 ] [ 9.; 9.; 9.; 9.; 9.; 9. ] in
  let src = float_tensor [ 1; 3 ] [ 0.; 1.; 2. ] in
  dispatch_copy ~self ~src ();
  [%expect {| tensor f32 [W=2 C=3] {0, 1, 2, 0, 1, 2} |}]

(* Not [Aten_shape]'s business, the same split [expand_test.ml]'s own
   non-broadcastable case draws -- [self]'s axis is 5, [src]'s is 2,
   incompatible on W. [Pointwise.Expand.output_shape]'s own axis-wise check
   catches it. *)
let%expect_test "dispatch: copy rejects a non-broadcastable src" =
  let self =
    float_tensor [ 5; 3 ]
      [ 0.; 0.; 0.; 0.; 0.; 0.; 0.; 0.; 0.; 0.; 0.; 0.; 0.; 0.; 0. ]
  in
  let src = float_tensor [ 2; 3 ] [ 0.; 1.; 2.; 3.; 4.; 5. ] in
  dispatch_copy ~self ~src ();
  [%expect {| error: incompatible broadcast extents on axis W: 2 vs 5 |}]
