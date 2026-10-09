(* Source kernels as emulated x86-64 processes: every case runs on the physical
   interpreter and under qemu-user over the same bound bytes. *)

open Loop_ir_test
open Ssa_bridge_test.Ssa_fixtures
module Src = Machine_source_test.Mir_source
module Cases = Machine_source_test.Mir_source_test
module H = Native_harness

let data_bind ?(shape = Loop_fixtures.shape_w 4) data = bind_data ~shape data

let check ?probe ?entry_mutation kernel ~bind =
  match Src.case_of_plan (Fusion_plan.default kernel) ~bind with
  | Error e -> Fmt.pr "%s@." e
  | Ok case -> Fmt.pr "%s@." (H.compare ?probe ?entry_mutation case)

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

(* System V around the kernel: the callee-saved registers, MXCSR and the stack
   pointer come back as they went in, whichever way the kernel exits; its
   arithmetic ignores the caller's rounding, flush-to-zero and denormals-are-
   zero settings; and a helper is entered with the stack aligned. *)
let%expect_test "the kernel keeps the ABI and ignores the caller's FP controls"
    =
  let abi kernel ~bind =
    match Src.case_of_plan (Fusion_plan.default kernel) ~bind with
    | Error e -> Fmt.pr "%s@." e
    | Ok case -> Fmt.pr "%s@." (H.abi case)
  in
  let zeros = data_bind [| 0.; 0.; 0.; 0. |] in
  check ~probe:true Loop_programs.kernel
    ~bind:(data_bind [| -0.; 1.5; nan; 3. |]);
  check ~probe:true Cases.noncommutative
    ~bind:(data_bind [| 7.; -0.; 1e-40; 3.4e38 |]);
  check ~probe:true Loop_programs.shifted_kernel ~bind:zeros;
  abi
    (Loop_programs.unary_kernel Expr.Value.Exp)
    ~bind:(data_bind [| 1.; 2.; 3.; 4. |]);
  [%expect
    {|
    ok [emulated: agree]
    ok [emulated: agree]
    coord_out_of_range(t0, W) [emulated: agree]
    ABI kept |}]

(* Each wrapper defect below leaves a program that still assembles and runs;
   only the probe can tell. *)
let%expect_test "wrapper defects are detected" =
  let module E = Machine_rivet_x86_64.Rivet_x64_module.Entry_mutation in
  let pointwise = data_bind [| 1.0000001; 1.5; nan; 3. |] in
  Fmt.pr "rbx clobbered: ";
  check ~probe:true ~entry_mutation:E.Clobber_rbx Loop_programs.kernel
    ~bind:pointwise;
  Fmt.pr "caller's MXCSR kept: ";
  (let a = operand 3 35 and b = operand 5 21 in
   check ~probe:true ~entry_mutation:E.Keep_caller_mxcsr
     (matmul_kernel ~m:5 ~k:7 ~n:3)
     ~bind:(matmul_bind ~m:5 ~k:7 ~n:3 ~a ~b));
  Fmt.pr "MXCSR not restored: ";
  check ~probe:true ~entry_mutation:E.Skip_mxcsr_restore Loop_programs.kernel
    ~bind:pointwise;
  Fmt.pr "stack misaligned: ";
  (match
     Src.case_of_plan
       (Fusion_plan.default (Loop_programs.unary_kernel Expr.Value.Exp))
       ~bind:(data_bind [| 1.; 2.; 3.; 4. |])
   with
  | Ok case -> Fmt.pr "%s@." (H.abi ~entry_mutation:E.Misalign_stack case)
  | Error e -> Fmt.pr "%s@." e);
  [%expect
    {|
    rbx clobbered: ok [emulated: agree] ABI broken: rbx changed from b0b0b0b0b0b0b0b to 1
    caller's MXCSR kept: ok [emulated: DISAGREE output t2[2]: -0x1.02779cp+4:f32 vs -0x1.02779ap+4:f32]
    MXCSR not restored: ok [emulated: agree] ABI broken: mxcsr changed from ffc0 to 1f80
    stack misaligned: ABI broken: a helper was entered with a misaligned stack |}]

(* Frames that put their slots far from the stack pointer. *)
let%expect_test "large frames" =
  let ck pad kernel ~bind =
    match Src.case_of_plan (Fusion_plan.default kernel) ~bind with
    | Error e -> Fmt.pr "%s@." e
    | Ok case -> Fmt.pr "%s@." (H.compare ~pad case)
  in
  let a = operand 3 35 and b = operand 5 21 in
  ck 40_000L Cases.noncommutative ~bind:(data_bind [| 7.; -0.; 1e-40; 3.4e38 |]);
  ck 1_048_576L Loop_programs.kernel ~bind:(data_bind [| -0.; 1.5; nan; 3. |]);
  ck 1_048_576L
    (matmul_kernel ~m:5 ~k:7 ~n:3)
    ~bind:(matmul_bind ~m:5 ~k:7 ~n:3 ~a ~b);
  [%expect
    {|
    ok [emulated: agree]
    ok [emulated: agree]
    ok [emulated: agree] |}]

(* Call-frame information describes the code: at every instruction the CFA
   offset GNU's decoding of .eh_frame gives equals the one the instructions
   have made. *)
let%expect_test "call-frame information" =
  let cfi ?pad kernel ~bind =
    match Src.case_of_plan (Fusion_plan.default kernel) ~bind with
    | Error e -> Fmt.pr "%s@." e
    | Ok case -> Fmt.pr "%s@." (H.cfi ?pad case)
  in
  let a = operand 3 35 and b = operand 5 21 in
  cfi Loop_programs.kernel ~bind:(data_bind [| -0.; 1.5; nan; 3. |]);
  cfi Loop_programs.shifted_kernel ~bind:(data_bind [| 0.; 0.; 0.; 0. |]);
  cfi (matmul_kernel ~m:5 ~k:7 ~n:3) ~bind:(matmul_bind ~m:5 ~k:7 ~n:3 ~a ~b);
  cfi ~pad:1_048_576L Loop_programs.kernel
    ~bind:(data_bind [| -0.; 1.5; nan; 3. |]);
  [%expect
    {|
    ok (1 functions, 64 instructions); ok (2 functions, 75 instructions)
    ok (1 functions, 664 instructions); ok (2 functions, 675 instructions)
    ok (1 functions, 149 instructions); ok (2 functions, 160 instructions)
    ok (1 functions, 66 instructions); ok (2 functions, 77 instructions) |}]

(* A wrong description is seen: the first stack adjustment says 8 more bytes. *)
let%expect_test "wrong call-frame information is seen" =
  let tamper s =
    Str.replace_first
      (Str.regexp "\t.cfi_adjust_cfa_offset\t\\([0-9]+\\)")
      "\t.cfi_adjust_cfa_offset\t\\1000" s
  in
  (match
     Src.case_of_plan
       (Fusion_plan.default Loop_programs.kernel)
       ~bind:(data_bind [| -0.; 1.5; nan; 3. |])
   with
  | Ok case -> Fmt.pr "%s@." (H.cfi ~tamper case)
  | Error e -> Fmt.pr "%s@." e);
  [%expect
    {| 4 mov $0x0,%edi: CFA offset 120008, instructions say 128; 4 mov $0x0,%edi: CFA offset 120008, instructions say 128 |}]
