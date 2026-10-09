(* Source kernels as emulated x86-64 processes: every case runs on the physical
   interpreter and under qemu-user over the same bound bytes. *)

open Loop_ir_test
open Ssa_bridge_test.Ssa_fixtures
module Src = Machine_source_test.Mir_source
module Cases = Machine_source_test.Mir_source_test
module H = Native_harness

let data_bind ?(shape = Loop_fixtures.shape_w 4) data = bind_data ~shape data

let check kernel ~bind =
  match Src.case_of_plan (Fusion_plan.default kernel) ~bind with
  | Error e -> Fmt.pr "%s@." e
  | Ok case -> Fmt.pr "%s@." (H.compare case)

let%expect_test "pointwise: signed zero, NaN, binary32 boundaries" =
  check Loop_programs.kernel ~bind:(data_bind [| -0.; 1.5; nan; 3. |]);
  check Loop_programs.kernel ~bind:(data_bind [| 1e30; -1e30; 0.1; 16777217. |]);
  check Cases.noncommutative ~bind:(data_bind [| 7.; -0.; 1e-40; 3.4e38 |]);
  [%expect
    {|
    ok [emulated: agree]
    ok [emulated: agree]
    ok [emulated: agree] |}]

let%expect_test "failures: coordinates and competing axes" =
  let zeros = data_bind [| 0.; 0.; 0.; 0. |] in
  check Loop_programs.shifted_kernel ~bind:zeros;
  check (Cases.shifted ~dh:0 ~dw:(-1)) ~bind:zeros;
  check (Cases.shifted ~dh:1 ~dw:4) ~bind:zeros;
  [%expect
    {|
    coord_out_of_range(t0, W) [emulated: agree]
    coord_out_of_range(t0, W) [emulated: agree]
    coord_out_of_range(t0, H) [emulated: agree] |}]

let%expect_test "matmul, odd shapes" =
  List.iter
    (fun (m, k, n) ->
      let a = operand 3 (m * k) and b = operand 5 (k * n) in
      Fmt.pr "%dx%dx%d: " m k n;
      check (matmul_kernel ~m ~k ~n) ~bind:(matmul_bind ~m ~k ~n ~a ~b))
    [ (1, 1, 1); (1, 3, 2); (3, 1, 4); (5, 7, 3); (2, 9, 1) ];
  [%expect
    {|
    1x1x1: ok [emulated: agree]
    1x3x2: ok [emulated: agree]
    3x1x4: ok [emulated: agree]
    5x7x3: ok [emulated: agree]
    2x9x1: ok [emulated: agree] |}]

let%expect_test "an exp kernel needs a math runtime this image does not carry" =
  check
    (Loop_programs.unary_kernel Expr.Value.Exp)
    ~bind:(data_bind [| 1.; 2.; 3.; 4. |]);
  [%expect {| emulated: helper exp has no implementation in this image |}]
