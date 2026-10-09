(* Source kernels natively through typed Rivet modules: every case runs on the
   physical interpreter and on the CPU over the same bound bytes, and the
   verdict names the status the two agree on. *)

open Loop_ir_test
open Ssa_bridge_test.Ssa_fixtures
module Src = Machine_source_test.Mir_source
module Cases = Machine_source_test.Mir_source_test
module H = Native_harness

let data_bind ?(shape = Loop_fixtures.shape_w 4) data = bind_data ~shape data

let check ?mutation kernel ~bind =
  match Src.case_of_plan (Fusion_plan.default kernel) ~bind with
  | Error e -> Fmt.pr "%s@." e
  | Ok case -> Fmt.pr "%s@." (H.compare ?mutation case)

let%expect_test "pointwise: signed zero, NaN, binary32 boundaries" =
  check Loop_programs.kernel ~bind:(data_bind [| -0.; 1.5; nan; 3. |]);
  check Loop_programs.kernel ~bind:(data_bind [| 1e30; -1e30; 0.1; 16777217. |]);
  check Cases.noncommutative ~bind:(data_bind [| 7.; -0.; 1e-40; 3.4e38 |]);
  [%expect
    {|
    ok [native: agree]
    ok [native: agree]
    ok [native: agree] |}]

let%expect_test "failures: coordinates and competing axes" =
  let zeros = data_bind [| 0.; 0.; 0.; 0. |] in
  check Loop_programs.shifted_kernel ~bind:zeros;
  check (Cases.shifted ~dh:0 ~dw:(-1)) ~bind:zeros;
  check (Cases.shifted ~dh:1 ~dw:4) ~bind:zeros;
  [%expect
    {|
    coord_out_of_range(t0, W) [native: agree]
    coord_out_of_range(t0, W) [native: agree]
    coord_out_of_range(t0, H) [native: agree] |}]

let%expect_test "matmul, odd shapes" =
  List.iter
    (fun (m, k, n) ->
      let a = operand 3 (m * k) and b = operand 5 (k * n) in
      Fmt.pr "%dx%dx%d: " m k n;
      check (matmul_kernel ~m ~k ~n) ~bind:(matmul_bind ~m ~k ~n ~a ~b))
    [ (1, 1, 1); (1, 3, 2); (3, 1, 4); (5, 7, 3); (2, 9, 1) ];
  [%expect
    {|
    1x1x1: ok [native: agree]
    1x3x2: ok [native: agree]
    3x1x4: ok [native: agree]
    5x7x3: ok [native: agree]
    2x9x1: ok [native: agree] |}]

let%expect_test "an exp kernel through its libm helper" =
  check
    (Loop_programs.unary_kernel Expr.Value.Exp)
    ~bind:(data_bind [| 1.; 2.; 3.; 4. |]);
  [%expect {| ok [native: agree] |}]

(* Each mapping defect below leaves a program that still assembles, loads and
   runs; only the comparison with the interpreter can tell. *)
let%expect_test "mapping mutations are detected natively" =
  let module M = Machine_rivet_aarch64.Rivet_a64_form.Mutation in
  let a = operand 3 35 and b = operand 5 21 in
  let matmul = matmul_kernel ~m:5 ~k:7 ~n:3 in
  let bind = matmul_bind ~m:5 ~k:7 ~n:3 ~a ~b in
  Fmt.pr "page only: ";
  check ~mutation:M.Dropped_lo12 matmul ~bind;
  Fmt.pr "branch sense: ";
  check ~mutation:M.Branch_sense Loop_programs.shifted_kernel
    ~bind:(data_bind [| 0.; 0.; 0.; 0. |]);
  Fmt.pr "commuted sub: ";
  check ~mutation:M.Commuted_sub Cases.noncommutative
    ~bind:(data_bind [| 7.; -0.; 1e-40; 3.4e38 |]);
  Fmt.pr "narrow spill: ";
  check ~mutation:M.Narrow_spill matmul ~bind;
  [%expect
    {|
    page only: ok [native: DISAGREE output t2[0]: -0x1.fc2c3ep+2:f32 vs 0x0p+0:f32]
    branch sense: coord_out_of_range(t0, N) [native: DISAGREE failure row: coord_out_of_range(t0, W)(0:i64, 0:i64, 0:i64, 0:i64, 4:i64, 0:i64) vs coord_out_of_range(t0, N)(0:i64, 0:i64, 0:i64, 0:i64, 1:i64, 0:i64)]
    commuted sub: ok [native: DISAGREE output t1[0]: 0x1.aaaaaap-1:f32 vs -0x1.aaaaaap-1:f32]
    narrow spill: native: generated code killed by SIGSEGV |}]
