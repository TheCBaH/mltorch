(* Source kernels natively through typed Rivet modules: every case runs on the
   physical interpreter and on the CPU over the same bound bytes, and the
   verdict names the status the two agree on. *)

open Loop_ir_test
open Ssa_bridge_test.Ssa_fixtures
module Src = Machine_source_test.Mir_source
module Cases = Machine_source_test.Mir_source_test
module H = Native_harness

let data_bind ?(shape = Loop_fixtures.shape_w 4) data = bind_data ~shape data

let check ?mutation ?runtime ?frame_mutation ?pad ?probe kernel ~bind =
  match Src.case_of_plan (Fusion_plan.default kernel) ~bind with
  | Error e -> Fmt.pr "%s@." e
  | Ok case ->
      Fmt.pr "%s@."
        (H.compare ?mutation ?runtime ?frame_mutation ?pad ?probe case)

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

(* The default mode carries the project's own exp instead of linking one; the
   other C library helpers stay refused. *)
let%expect_test "an exp kernel through the owned exp, and a refused log" =
  let owned = Machine_rivet_aarch64.Rivet_a64_runtime.Dependency_free in
  check ~runtime:owned
    (Loop_programs.unary_kernel Expr.Value.Exp)
    ~bind:(data_bind [| 1.; -87.5; 88.7; -1e30 |]);
  check ~runtime:owned
    (Loop_programs.unary_kernel Expr.Value.Log)
    ~bind:(data_bind [| 1.; 2.; 3.; 4. |]);
  [%expect
    {|
    ok [native: agree]
    native: helper log needs the system math library, which the mode forbids |}]

(* The same cases against the source oracle itself, the structured SSA
   interpreter on the plan, with no machine-level interpreter between. *)
let%expect_test "native agrees with the source oracle" =
  let oracle ?mutation kernel ~bind =
    match Src.case_of_plan (Fusion_plan.default kernel) ~bind with
    | Error e -> Fmt.pr "%s@." e
    | Ok case -> Fmt.pr "%s@." (H.oracle ?mutation case)
  in
  oracle Loop_programs.kernel ~bind:(data_bind [| -0.; 1.5; nan; 3. |]);
  oracle Cases.noncommutative ~bind:(data_bind [| 7.; -0.; 1e-40; 3.4e38 |]);
  oracle Loop_programs.shifted_kernel ~bind:(data_bind [| 0.; 0.; 0.; 0. |]);
  oracle (Cases.shifted ~dh:1 ~dw:4) ~bind:(data_bind [| 0.; 0.; 0.; 0. |]);
  let a = operand 3 35 and b = operand 5 21 in
  oracle (matmul_kernel ~m:5 ~k:7 ~n:3) ~bind:(matmul_bind ~m:5 ~k:7 ~n:3 ~a ~b);
  oracle
    (Loop_programs.unary_kernel Expr.Value.Exp)
    ~bind:(data_bind [| 1.; -87.5; 88.7; -1e30 |]);
  Fmt.pr "commuted subtraction: ";
  oracle ~mutation:Machine_rivet_aarch64.Rivet_a64_form.Mutation.Commuted_sub
    Cases.noncommutative
    ~bind:(data_bind [| 7.; -0.; 1e-40; 3.4e38 |]);
  [%expect
    {|
    ok [native vs source oracle: agree]
    ok [native vs source oracle: agree]
    coord_out_of_range(t0, W) [native vs source oracle: agree]
    coord_out_of_range(t0, H) [native vs source oracle: agree]
    ok [native vs source oracle: agree]
    ok [native vs source oracle: agree]
    commuted subtraction: ok [native vs source oracle: DISAGREE output t1[0]: 0x1.aaaaaap-1:f32 vs -0x1.aaaaaap-1:f32] |}]

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
let gnu ?pad ?tamper_gnu ?tamper_text kernel ~bind =
  match Src.case_of_plan (Fusion_plan.default kernel) ~bind with
  | Error e -> Fmt.pr "%s@." e
  | Ok case -> Fmt.pr "%s@." (H.gnu ?pad ?tamper_gnu ?tamper_text case)

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
    agree (2 segments, 256 bytes, 4 symbols)
    agree (2 segments, 2704 bytes, 4 symbols)
    agree (2 segments, 824 bytes, 5 symbols)
    agree (2 segments, 264 bytes, 5 symbols) |}]

(* Selection folds an index extension, scale and addition into the access, and
   a float load or store into a floating-point access with no move between
   register files; the kernel still agrees with the interpreter, and GNU
   assembles what Rivet encodes. *)
let%expect_test "register-offset and floating-point accesses" =
  let m, k, n = (5, 7, 3) in
  let a = operand 3 (m * k) and b = operand 5 (k * n) in
  let report name kernel bind =
    match Src.case_of_plan (Fusion_plan.default kernel) ~bind with
    | Error e -> Fmt.pr "%s@." e
    | Ok case -> (
        Fmt.pr "%s: %s@." name (H.compare case);
        match H.build case ~sites:[||] with
        | Error e -> Fmt.pr "%s@." e
        | Ok b -> (
            match
              Err.payload
                (Machine_rivet_aarch64.Rivet_a64_module.of_artifact b.H.artifact)
            with
            | Error _ -> Fmt.pr "refused@."
            | Ok modul ->
                let text =
                  Machine_rivet_aarch64_gnu.Gnu_coherence.assembly modul
                in
                let count re =
                  let n = ref 0 and from = ref 0 in
                  (try
                     while true do
                       from := Str.search_forward (Str.regexp re) text !from + 1;
                       incr n
                     done
                   with Not_found -> ());
                  !n
                in
                Fmt.pr
                  "  register-offset accesses: %b; float loads: %b, moves from \
                   a general register: %d@."
                  (count "sxtw #[0-9]\\]" > 0)
                  (count "ldr s[0-9]" > 0)
                  (count "fmov s[0-9]+, w")))
  in
  report "matmul" (matmul_kernel ~m ~k ~n) (matmul_bind ~m ~k ~n ~a ~b);
  report "shifted" Loop_programs.shifted_kernel (data_bind [| 0.; 0.; 0.; 0. |]);
  report "pointwise" Loop_programs.kernel (data_bind [| 1.; 2.; 3.; 4. |]);
  [%expect
    {|
    matmul: ok [native: agree]
      register-offset accesses: false; float loads: true, moves from a general register: 0
    shifted: coord_out_of_range(t0, W) [native: agree]
      register-offset accesses: true; float loads: true, moves from a general register: 0
    pointwise: ok [native: agree]
      register-offset accesses: true; float loads: true, moves from a general register: 0 |}]

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
    .text+189: rivet 3a, gnu 2a
    .text+92: rivet c0, gnu 1f
    reparsed .text+189: typed 3a, text 2a |}]

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

(* A frame beyond what an immediate offset reaches: the allocator's stores and
   reloads go through x16, computed by a late form. 40,000 bytes and 1 MiB run;
   larger frames are encoded and compared with GNU but would overflow a stack,
   so they are not run, and the frame stage itself refuses one beyond its code
   model: a stack step is an ADD or SUB of an immediate and an immediate shifted
   by twelve, so a frame past 16 MiB. *)
let%expect_test "large frames, run and encoded" =
  let bind = data_bind [| -0.; 1.5; nan; 3. |] in
  let probe = Abi_probe.altered_fpcr in
  check ~pad:40_000L ~probe Cases.noncommutative
    ~bind:(data_bind [| 7.; -0.; 1e-40; 3.4e38 |]);
  check ~pad:1_048_576L ~probe Loop_programs.kernel ~bind;
  let a = operand 3 35 and b = operand 5 21 in
  check ~pad:1_048_576L
    (matmul_kernel ~m:5 ~k:7 ~n:3)
    ~bind:(matmul_bind ~m:5 ~k:7 ~n:3 ~a ~b);
  gnu ~pad:40_000L Loop_programs.kernel ~bind;
  gnu ~pad:0x00FE_0000L Loop_programs.kernel ~bind;
  gnu ~pad:0x0100_0000L Loop_programs.kernel ~bind;
  gnu ~pad:0x2000_0000L Loop_programs.kernel ~bind;
  [%expect
    {|
    ok [native: agree]
    ok [native: agree]
    ok [native: agree]
    agree (2 segments, 416 bytes, 4 symbols)
    agree (2 segments, 416 bytes, 4 symbols)
    not built: frame: fn0: a frame beyond the supported code model
    not built: frame: fn0: a frame beyond the supported code model |}]

(* Planned binary32 vector kernels on NEON: the same cases as the selected-form
   tests, now on the CPU. *)
let planned ?(numerics = Ssa_ir.Ssa_numerics.Simd_fp32_ordered) ?probe kernel
    ~bind =
  match
    Src.case_of_planned ~target:Ssa_ir.Ssa_target.neon128 ~numerics
      (Fusion_plan.default kernel)
      ~bind
  with
  | Error e -> Fmt.pr "%s@." e
  | Ok case -> Fmt.pr "%s@." (H.compare ?probe case)

let vdata n = Array.init n (fun i -> (float_of_int (i * 7 mod 11) -. 5.) /. 4.)

let%expect_test "vector pointwise and matmul on NEON, natively" =
  let x = Loop_fixtures.load_t0 in
  List.iter
    (fun (name, body) ->
      List.iter
        (fun n ->
          Fmt.pr "%s, w=%d: " name n;
          let d = vdata n in
          if n > 2 then (
            d.(0) <- Float.nan;
            d.(1) <- -0.;
            d.(2) <- 3.4e38);
          let shape = Loop_fixtures.shape_w n in
          planned
            (Loop_fixtures.pixel_kernel ~shape body)
            ~bind:(bind_data ~shape d))
        [ 16; 37 ])
    [
      ("x / 3 - 1.5", Expr.Value.(sub (div x (const 3.)) (const 1.5)));
      ("sqrt x", Expr.Value.sqrt x);
      ("x * x + x", Expr.Value.(add (mul x x) x));
    ];
  List.iter
    (fun (m, k, n) ->
      let a = operand 3 (m * k) and b = operand 5 (k * n) in
      Fmt.pr "%dx%dx%d: " m k n;
      planned (matmul_kernel ~m ~k ~n) ~bind:(matmul_bind ~m ~k ~n ~a ~b))
    [ (2, 3, 16); (3, 5, 17); (5, 7, 33) ];
  [%expect
    {|
    x / 3 - 1.5, w=16: ok [native: agree]
    x / 3 - 1.5, w=37: ok [native: agree]
    sqrt x, w=16: ok [native: agree]
    sqrt x, w=37: ok [native: agree]
    x * x + x, w=16: ok [native: agree]
    x * x + x, w=37: ok [native: agree]
    2x3x16: ok [native: agree]
    3x5x17: ok [native: agree]
    5x7x33: ok [native: agree] |}]

let vector_gnu ?(numerics = Ssa_ir.Ssa_numerics.Simd_fp32_ordered) kernel ~bind
    =
  match
    Src.case_of_planned ~target:Ssa_ir.Ssa_target.neon128 ~numerics
      (Fusion_plan.default kernel)
      ~bind
  with
  | Error e -> Fmt.pr "%s@." e
  | Ok case ->
      (match H.mnemonics case with
      | Ok t ->
          let n k = Option.value ~default:0 (Hashtbl.find_opt t k) in
          Fmt.pr "%d vector instructions, %d fmla, %d fmadd, %d ld1r; "
            (n "<vector>") (n "fmla") (n "fmadd") (n "ld1r")
      | Error e -> Fmt.pr "%s; " e);
      Fmt.pr "%s@." (H.gnu case)

let%expect_test "vector kernels: the forms they use, and GNU's reading of them"
    =
  let a = operand 3 (3 * 5) and b = operand 5 (5 * 17) in
  vector_gnu
    (matmul_kernel ~m:3 ~k:5 ~n:17)
    ~bind:(matmul_bind ~m:3 ~k:5 ~n:17 ~a ~b);
  vector_gnu ~numerics:Ssa_ir.Ssa_numerics.Simd_fp32_relaxed
    (matmul_kernel ~m:3 ~k:5 ~n:17)
    ~bind:(matmul_bind ~m:3 ~k:5 ~n:17 ~a ~b);
  let shape = Loop_fixtures.shape_w 37 in
  vector_gnu
    (Loop_fixtures.pixel_kernel ~shape (Expr.Value.sqrt Loop_fixtures.load_t0))
    ~bind:(bind_data ~shape (vdata 37));
  [%expect
    {|
    210 vector instructions, 0 fmla, 0 fmadd, 0 ld1r; agree (2 segments, 2404 bytes, 5 symbols)
    210 vector instructions, 0 fmla, 0 fmadd, 0 ld1r; agree (2 segments, 2404 bytes, 5 symbols)
    156 vector instructions, 0 fmla, 0 fmadd, 0 ld1r; agree (2 segments, 1324 bytes, 4 symbols) |}]

let%expect_test "contracted vector kernels, natively and with the ABI probe" =
  let relaxed = Ssa_ir.Ssa_numerics.Simd_fp32_relaxed in
  List.iter
    (fun (m, k, n) ->
      let a = operand 3 (m * k) and b = operand 5 (k * n) in
      Fmt.pr "%dx%dx%d: " m k n;
      planned ~numerics:relaxed ~probe:Abi_probe.altered_fpcr
        (matmul_kernel ~m ~k ~n)
        ~bind:(matmul_bind ~m ~k ~n ~a ~b))
    [ (2, 3, 16); (3, 5, 17); (5, 7, 33) ];
  [%expect
    {|
    2x3x16: ok [native: agree]
    3x5x17: ok [native: agree]
    5x7x33: ok [native: agree] |}]

(* FMLA: the relaxed policy contracts a multiply feeding an add into the fused
   vector form, which the CPU must execute as the physical interpreter models
   it (one rounding). *)
let%expect_test "fmla on the CPU" =
  let relaxed = Ssa_ir.Ssa_numerics.Simd_fp32_relaxed in
  let x = Loop_fixtures.load_t0 in
  List.iter
    (fun n ->
      let shape = Loop_fixtures.shape_w n in
      let d = vdata n in
      if n > 2 then (
        d.(0) <- Float.nan;
        d.(1) <- -0.;
        d.(2) <- 3.4e38);
      Fmt.pr "w=%d: " n;
      vector_gnu ~numerics:relaxed
        (Loop_fixtures.pixel_kernel ~shape Expr.Value.(add (mul x x) x))
        ~bind:(bind_data ~shape d);
      Fmt.pr "w=%d: " n;
      planned ~numerics:relaxed ~probe:Abi_probe.altered_fpcr
        (Loop_fixtures.pixel_kernel ~shape Expr.Value.(add (mul x x) x))
        ~bind:(bind_data ~shape d))
    [ 16; 37 ];
  [%expect
    {|
    w=16: 308 vector instructions, 4 fmla, 0 fmadd, 0 ld1r; agree (2 segments, 2068 bytes, 4 symbols)
    w=16: ok [native: agree]
    w=37: 308 vector instructions, 4 fmla, 1 fmadd, 0 ld1r; agree (2 segments, 2252 bytes, 4 symbols)
    w=37: ok [native: agree] |}]

(* Strided lane expansion: the input read transposed, so a vector of outputs
   gathers one lane at a time (LD1 and INS) and the stores scatter (ST1). *)
let transposed =
  let at a = Expr.Index.output a in
  Expr.Value.load
    (Expr_bridge.source_of_id (tid 0))
    (Expr.Coord.set
       (Expr.Coord.set
          (Expr_bridge.coord_of_vec6 Symbolic.out_vec)
          Expr.Axis.W (at Expr.Axis.H))
       Expr.Axis.H (at Expr.Axis.W))

let%expect_test "lane gathers and every legal loop vectorized, natively" =
  let square = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:17 ~w:17 ~c:1 in
  let target = Ssa_ir.Ssa_target.forced Ssa_ir.Ssa_target.neon128 in
  let numerics = Ssa_ir.Ssa_numerics.Simd_fp32_ordered in
  let bind = bind_data ~shape:square (vdata (17 * 17)) in
  let kernel = Loop_fixtures.pixel_kernel ~shape:square transposed in
  match
    Src.case_of_planned ~target ~numerics (Fusion_plan.default kernel) ~bind
  with
  | Error e -> Fmt.pr "%s@." e
  | Ok case ->
      (match H.mnemonics case with
      | Ok t ->
          let n k = Option.value ~default:0 (Hashtbl.find_opt t k) in
          Fmt.pr "ld1 %d, st1 %d, ins %d, ld1r %d, dup %d@." (n "ld1") (n "st1")
            (n "ins") (n "ld1r") (n "dup")
      | Error e -> Fmt.pr "%s@." e);
      Fmt.pr "%s@." (H.compare ~probe:Abi_probe.altered_fpcr case);
      Fmt.pr "%s@." (H.gnu case);
      [%expect
        {|
        ld1 12, st1 0, ins 8, ld1r 4, dup 16
        ok [native: agree]
        agree (2 segments, 1984 bytes, 4 symbols) |}]

(* A call per lane: exp has no vector form, so the lanes go out one at a time,
   with the vector held across each call in a full-width spill. *)
let%expect_test "vector exp, a helper call per lane, natively" =
  let x = Loop_fixtures.load_t0 in
  List.iter
    (fun n ->
      let shape = Loop_fixtures.shape_w n in
      let d = vdata n in
      if n > 2 then (
        d.(0) <- Float.nan;
        d.(1) <- -0.;
        d.(2) <- 3.4e38);
      Fmt.pr "w=%d: " n;
      planned ~probe:Abi_probe.altered_fpcr
        (Loop_fixtures.pixel_kernel ~shape (Expr.Value.exp x))
        ~bind:(bind_data ~shape d))
    [ 16; 17; 37 ];
  [%expect
    {|
    w=16: ok [native: agree]
    w=17: ok [native: agree]
    w=37: ok [native: agree] |}]

(* Masks: a compare makes all-ones or all-zero lanes and a select chooses by
   bits. Less-than is the compare with its operands swapped; a NaN is false for
   every ordered compare. *)
let%expect_test "compares and selects on masks, natively" =
  let x = Loop_fixtures.load_t0 in
  let zero = Expr.Value.const 0. in
  let kernels =
    [
      ( "x < 0 ? -x : x / 3",
        Expr.Value.(
          select (Expr.Bool.value_lt x zero) (sub zero x) (div x (const 3.))) );
      ( "x == 0 ? 0 : 1",
        Expr.Value.(select (Expr.Bool.value_eq x zero) (const 0.) (const 1.)) );
      ( "x < 1 ? x * x : x + 1",
        Expr.Value.(
          select (Expr.Bool.value_lt x (const 1.)) (mul x x) (add x (const 1.)))
      );
    ]
  in
  List.iter
    (fun (name, body) ->
      List.iter
        (fun n ->
          let shape = Loop_fixtures.shape_w n in
          let d = vdata n in
          if n > 3 then (
            d.(0) <- Float.nan;
            d.(1) <- -0.;
            d.(2) <- 3.4e38;
            d.(3) <- 0.);
          Fmt.pr "%s, w=%d: " name n;
          let kernel = Loop_fixtures.pixel_kernel ~shape body in
          (match
             Src.case_of_planned ~target:Ssa_ir.Ssa_target.neon128
               ~numerics:Ssa_ir.Ssa_numerics.Simd_fp32_ordered
               (Fusion_plan.default kernel)
               ~bind:(bind_data ~shape d)
           with
          | Ok case -> (
              match H.mnemonics case with
              | Ok t ->
                  let c k = Option.value ~default:0 (Hashtbl.find_opt t k) in
                  Fmt.pr "%d fcmgt, %d fcmeq, %d bit; " (c "fcmgt") (c "fcmeq")
                    (c "bit")
              | Error e -> Fmt.pr "%s; " e)
          | Error _ -> ());
          planned ~probe:Abi_probe.altered_fpcr kernel
            ~bind:(bind_data ~shape d))
        [ 16; 37 ])
    kernels;
  [%expect
    {|
    x < 0 ? -x : x / 3, w=16: 4 fcmgt, 0 fcmeq, 4 bit; ok [native: agree]
    x < 0 ? -x : x / 3, w=37: 4 fcmgt, 0 fcmeq, 4 bit; ok [native: agree]
    x == 0 ? 0 : 1, w=16: 0 fcmgt, 4 fcmeq, 4 bit; ok [native: agree]
    x == 0 ? 0 : 1, w=37: 0 fcmgt, 4 fcmeq, 4 bit; ok [native: agree]
    x < 1 ? x * x : x + 1, w=16: 4 fcmgt, 0 fcmeq, 4 bit; ok [native: agree]
    x < 1 ? x * x : x + 1, w=37: 4 fcmgt, 0 fcmeq, 4 bit; ok [native: agree] |}]

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
  cfi ~pad:40_000L Loop_programs.kernel
    ~bind:(data_bind [| -0.; 1.5; nan; 3. |]);
  [%expect
    {|
    ok (1 functions, 64 instructions); ok (2 functions, 73 instructions)
    ok (1 functions, 676 instructions); ok (2 functions, 685 instructions)
    ok (1 functions, 206 instructions); ok (2 functions, 215 instructions)
    ok (1 functions, 104 instructions); ok (2 functions, 113 instructions) |}]

(* A wrong description is seen: the first stack adjustment says 8 more bytes. *)
let%expect_test "wrong call-frame information is seen" =
  let tamper s =
    Str.replace_first
      (Str.regexp "\t.cfi_adjust_cfa_offset\t\\([0-9]+\\)")
      "\t.cfi_adjust_cfa_offset\t1000" s
  in
  (match
     Src.case_of_plan
       (Fusion_plan.default Loop_programs.kernel)
       ~bind:(data_bind [| -0.; 1.5; nan; 3. |])
   with
  | Ok case -> Fmt.pr "%s@." (H.cfi ~tamper case)
  | Error e -> Fmt.pr "%s@." e);
  [%expect
    {| 4 mrs x17, fpcr: CFA offset 1000, instructions say 112; 4 mrs x17, fpcr: CFA offset 1000, instructions say 112 |}]
