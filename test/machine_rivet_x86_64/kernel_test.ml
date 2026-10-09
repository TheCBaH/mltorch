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

(* The same modules through GNU: assembled and linked at Rivet's addresses, the
   loadable bytes and global symbol addresses are Rivet's. *)
let gnu ?tamper_gnu ?tamper_text kernel ~bind =
  match Src.case_of_plan (Fusion_plan.default kernel) ~bind with
  | Error e -> Fmt.pr "%s@." e
  | Ok case -> Fmt.pr "%s@." (H.gnu ?tamper_gnu ?tamper_text case)

let%expect_test "GNU assembles what Rivet encodes" =
  gnu Loop_programs.kernel ~bind:(data_bind [| -0.; 1.5; nan; 3. |]);
  gnu Loop_programs.shifted_kernel ~bind:(data_bind [| 0.; 0.; 0.; 0. |]);
  let a = operand 3 35 and b = operand 5 21 in
  gnu (matmul_kernel ~m:5 ~k:7 ~n:3) ~bind:(matmul_bind ~m:5 ~k:7 ~n:3 ~a ~b);
  [%expect
    {|
    agree (2 segments, 328 bytes, 4 symbols)
    agree (2 segments, 4148 bytes, 4 symbols)
    agree (2 segments, 938 bytes, 5 symbols) |}]

(* A difference between the two is seen: GNU is handed source with one
   instruction changed. *)
let%expect_test "a tampered GNU source disagrees" =
  let replace_first ~sub ~by s =
    match Str.search_forward (Str.regexp_string sub) s 0 with
    | i ->
        String.sub s 0 i ^ by
        ^ String.sub s
            (i + String.length sub)
            (String.length s - i - String.length sub)
    | exception Not_found -> s
  in
  gnu
    ~tamper_gnu:(replace_first ~sub:"\tsubsd" ~by:"\taddsd")
    Cases.noncommutative
    ~bind:(data_bind [| 7.; -0.; 1e-40; 3.4e38 |]);
  gnu
    ~tamper_gnu:(replace_first ~sub:"\tret" ~by:"\tnop")
    Loop_programs.kernel
    ~bind:(data_bind [| -0.; 1.5; nan; 3. |]);
  gnu
    ~tamper_text:(replace_first ~sub:"\tsubsd" ~by:"\taddsd")
    Cases.noncommutative
    ~bind:(data_bind [| 7.; -0.; 1e-40; 3.4e38 |]);
  [%expect
    {|
    .text+240: rivet 5c, gnu 58
    .text+79: rivet c3, gnu 90
    reparsed .text+240: typed 5c, text 58 |}]
