(* Source kernels natively through typed Rivet modules: every case runs on the
   physical interpreter and on the CPU over the same bound bytes, and the
   verdict names the status the two agree on. *)

open Loop_ir_test
open Ssa_bridge_test.Ssa_fixtures
module Src = Machine_source_test.Mir_source
module Cases = Machine_source_test.Mir_source_test
module H = Native_harness

let data_bind ?(shape = Loop_fixtures.shape_w 4) data = bind_data ~shape data

let check ?mutation ?frame_mutation ?probe kernel ~bind =
  match Src.case_of_plan (Fusion_plan.default kernel) ~bind with
  | Error e -> Fmt.pr "%s@." e
  | Ok case -> Fmt.pr "%s@." (H.compare ?mutation ?frame_mutation ?probe case)

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
  gnu
    (Loop_programs.unary_kernel Expr.Value.Exp)
    ~bind:(data_bind [| 1.; 2.; 3.; 4. |]);
  [%expect
    {|
    agree (2 segments, 352 bytes, 4 symbols)
    agree (2 segments, 2800 bytes, 4 symbols)
    agree (2 segments, 740 bytes, 5 symbols)
    agree (2 segments, 360 bytes, 5 symbols) |}]

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
    ~tamper_gnu:(replace_first ~sub:"\tfsub" ~by:"\tfadd")
    Cases.noncommutative
    ~bind:(data_bind [| 7.; -0.; 1e-40; 3.4e38 |]);
  gnu
    ~tamper_gnu:(replace_first ~sub:"\tret" ~by:"\tnop")
    Loop_programs.kernel
    ~bind:(data_bind [| -0.; 1.5; nan; 3. |]);
  gnu
    ~tamper_text:(replace_first ~sub:"\tfsub" ~by:"\tfadd")
    Cases.noncommutative
    ~bind:(data_bind [| 7.; -0.; 1e-40; 3.4e38 |]);
  [%expect
    {|
    .text+237: rivet 3a, gnu 2a
    .text+92: rivet c0, gnu 1f
    reparsed .text+237: typed 3a, text 2a |}]

(* AAPCS64 around the kernel: the callee-saved registers, FPCR and the stack
   pointer come back as they went in, whichever way the kernel exits, and its
   arithmetic ignores the caller's rounding, flush-to-zero and default-NaN
   controls. *)
let%expect_test "the kernel keeps the ABI and ignores the caller's FP controls"
    =
  let probe = Abi_probe.altered_fpcr in
  let zeros = data_bind [| 0.; 0.; 0.; 0. |] in
  check ~probe Loop_programs.kernel ~bind:(data_bind [| -0.; 1.5; nan; 3. |]);
  check ~probe Cases.noncommutative
    ~bind:(data_bind [| 7.; -0.; 1e-40; 3.4e38 |]);
  check ~probe Loop_programs.shifted_kernel ~bind:zeros;
  check ~probe
    (Loop_programs.unary_kernel Expr.Value.Exp)
    ~bind:(data_bind [| 1.; 2.; 3.; 4. |]);
  let a = operand 3 35 and b = operand 5 21 in
  check ~probe
    (matmul_kernel ~m:5 ~k:7 ~n:3)
    ~bind:(matmul_bind ~m:5 ~k:7 ~n:3 ~a ~b);
  [%expect
    {|
    ok [native: agree]
    ok [native: agree]
    coord_out_of_range(t0, W) [native: agree]
    ok [native: agree]
    ok [native: agree] |}]

(* A frame that drops one of its duties leaves a kernel that still computes:
   only the probe sees it. *)
let%expect_test "frame defects are seen by the probe" =
  let module Fm = Machine_alloc.Mir_frame.Mutation in
  let probe = Abi_probe.altered_fpcr in
  let zeros = data_bind [| 0.; 0.; 0.; 0. |] in
  Fmt.pr "no control restore: ";
  check ~probe ~frame_mutation:Fm.No_control_restore Loop_programs.kernel
    ~bind:(data_bind [| -0.; 1.5; nan; 3. |]);
  Fmt.pr "epilogue once, failing exit: ";
  check ~probe ~frame_mutation:Fm.Epilogue_once Loop_programs.shifted_kernel
    ~bind:zeros;
  [%expect
    {|
    no control restore: ok [native: agree] ABI broken: fpcr
    epilogue once, failing exit: native: generated code killed by SIGSEGV |}]
