(* squeeze.dim, through the ATen-linked bridge. Binds directly to the
   existing Reshape node -- see op_bridge_shape.ml's own comment for why.
   Promote with [dune promote test/native_bridge/squeeze_test.ml]. *)

open Helpers

let dispatch_squeeze ~x ~dim () =
  dispatch_print ~target:"torch.ops.aten.squeeze.dim"
    ~bindings:[ ("self", x) ]
    ~inputs:[ in_tensor "self"; in_int "dim" dim ]
    ~noutputs:1

let%expect_test "dispatch: squeeze drops a unit axis" =
  let x = float_tensor [ 2; 1; 3 ] [ 0.; 1.; 2.; 3.; 4.; 5. ] in
  dispatch_squeeze ~x ~dim:1 ();
  [%expect {| tensor f32 [W=2 C=3] {0, 1, 2, 3, 4, 5} |}]

(* The corpus's own configuration: negative [dim] resolving to the same
   unit axis as above. *)
let%expect_test "dispatch: squeeze, negative dim" =
  let x = float_tensor [ 2; 1; 3 ] [ 0.; 1.; 2.; 3.; 4.; 5. ] in
  dispatch_squeeze ~x ~dim:(-2) ();
  [%expect {| tensor f32 [W=2 C=3] {0, 1, 2, 3, 4, 5} |}]

(* Real ATen leaves [self] UNCHANGED (same rank) when the named axis's LIVE
   extent is not 1, rather than raising -- squeeze is a no-op here, not a
   rejection. *)
let%expect_test "dispatch: squeeze is a no-op on a non-unit axis" =
  let x = float_tensor [ 2; 3 ] [ 0.; 1.; 2.; 3.; 4.; 5. ] in
  dispatch_squeeze ~x ~dim:0 ();
  [%expect {| tensor f32 [W=2 C=3] {0, 1, 2, 3, 4, 5} |}]

(* Squeezing every axis of a rank-1 unit tensor down to rank 0. *)
let%expect_test "dispatch: squeeze to rank 0" =
  let x = float_tensor [ 1 ] [ 7. ] in
  dispatch_squeeze ~x ~dim:0 ();
  [%expect {| tensor f32 [C=1] {7} |}]
