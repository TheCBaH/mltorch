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

(* An I64 operand, real ATen as the oracle -- the [squeeze.dims] twin of
   shape_ops_test.ml's I64 unsqueeze fixture, added when [op_bridge_shape.ml]
   dropped [require_f32] from this arm (see that file's own comment on the
   [squeeze.dims] arm). [squeeze.dims] (plural), not [squeeze.dim]: the real
   ATen side of this harness (lib/interp/interp_dispatch.ml, generated) has no
   binding at all for singular [squeeze.dim] -- confirmed by trying it first,
   an "unhandled op" report unrelated to dtype -- so only the plural form can
   be verified against real ATen here. [Verify.verify_node] compares I64
   outputs by exact [Int64.equal], not a float tolerance, so this genuinely
   proves the whole bridge -> [Reshape] -> [Eval_direct]'s [Compute_i64] path
   stays exact past 2^53, not merely that native's own standalone graph does. *)
let%expect_test "verify: squeeze.dims on an I64 operand past 2^53" =
  let x =
    i64_tensor [ 2; 1; 3 ]
      [
        9_007_199_254_740_993L;
        9_007_199_254_740_994L;
        9_007_199_254_740_995L;
        9_007_199_254_740_996L;
        9_007_199_254_740_997L;
        9_007_199_254_740_998L;
      ]
  in
  verify_print ~target:"torch.ops.aten.squeeze.dims"
    ~bindings:[ ("self", x) ]
    ~inputs:[ in_tensor "self"; in_ints "dim" [ 1 ] ];
  [%expect {| aten and native agree |}]
