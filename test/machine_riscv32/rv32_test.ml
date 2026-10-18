open Loop_ir_test
open Ssa_bridge_test.Ssa_fixtures
module H = Rv32_harness
module A64T = Machine_aarch64_test.A64_select_test
module B = Ssa_ir.Ssa_builder

(* RV32, interpreted on this host: C3 (selected), C4 (allocated, checked) and
   C5 (realized frames) on binary32 kernels and on integer programs, whose
   i64 values become pairs of words. *)

let data_bind = A64T.data_bind
let zeros = data_bind [| 0.; 0.; 0.; 0. |]
let floats xs = Ssa_ir.Ssa_memory.Floats (Array.of_list xs)
let i64s xs = Ssa_ir.Ssa_memory.Int64s (Array.of_list xs)

(* a max reduction is IEEE maximum: a NaN in a window is the window's result *)
let pool =
  ( Loop_programs.pool_kernel ~input:4 ~out:2 ~kernel:2 ~stride:2 ~pad:0
      ~result:Expr.Intrinsic.Max_pool.Value,
    bind_data ~shape:(Loop_programs.hw 4 4)
      [|
        1.;
        Float.nan;
        3.;
        -0.;
        0.;
        -1.;
        2.;
        5.;
        -0.;
        0.;
        7.;
        7.5;
        2.;
        1.;
        9.;
        Float.nan;
      |] )

let kernels =
  [
    ("max pool", fst pool, snd pool);
    ("pointwise", Loop_programs.kernel, data_bind [| -0.; 1.5; nan; 3. |]);
    ( "pointwise bounds",
      Loop_programs.kernel,
      data_bind [| 1e30; -1e30; 0.1; 16777217. |] );
    ("shifted", Loop_programs.shifted_kernel, zeros);
    ("negative shift", A64T.shifted ~dh:0 ~dw:(-1), zeros);
    ("competing axes", A64T.shifted ~dh:1 ~dw:4, zeros);
    ("index overflow", Loop_programs.overflow_kernel, zeros);
    ( "matmul 5x7x3",
      matmul_kernel ~m:5 ~k:7 ~n:3,
      matmul_bind ~m:5 ~k:7 ~n:3 ~a:(operand 3 35) ~b:(operand 5 21) );
  ]

let programs () =
  let div = A64T.i64_program (fun bld x y -> B.i64_div bld x y) in
  [
    ("div 7/-2", div, [ (0, i64s [ 7L; -2L ]) ]);
    ("div by zero", div, [ (0, i64s [ 7L; 0L ]) ]);
    ("to_i64 nan", A64T.to_i64, [ (0, floats [ Float.nan ]) ]);
    ( "recurrences",
      Machine_alloc_test.Alloc_test.recurrences 5,
      Machine_alloc_test.Alloc_test.inputs );
  ]

let binary32 = Ssa_ir.Ssa_numerics.Fp32_ordered
let scalar = Ssa_ir.Ssa_target.scalar

let all_stages label stage =
  Fmt.pr "== %s@." label;
  List.iter
    (fun (n, k, bind) ->
      Fmt.pr "%s: %s@." n
        (H.planned ~stage ~target:scalar ~numerics:binary32 k ~bind))
    kernels;
  List.iter
    (fun (n, p, inputs) -> Fmt.pr "%s: %s@." n (H.program ~stage p ~inputs))
    (programs ())

let%expect_test "selected (C3)" =
  all_stages "selected" H.Selected;
  [%expect
    {|
    == selected
    max pool: ok
    pointwise: ok
    pointwise bounds: ok
    shifted: coord_out_of_range(t0, W)
    negative shift: coord_out_of_range(t0, W)
    competing axes: coord_out_of_range(t0, H)
    index overflow: index_overflow(mul)
    matmul 5x7x3: ok
    div 7/-2: refused: an i64 division is not selected for riscv32
    div by zero: refused: an i64 division is not selected for riscv32
    to_i64 nan: refused: type f64 has no riscv32 register in this slice
    recurrences: refused: type f64 has no riscv32 register in this slice |}]

let%expect_test "allocated and checked (C4)" =
  all_stages "allocated" H.Allocated;
  [%expect
    {|
    == allocated
    max pool: ok
    pointwise: ok
    pointwise bounds: ok
    shifted: coord_out_of_range(t0, W)
    negative shift: coord_out_of_range(t0, W)
    competing axes: coord_out_of_range(t0, H)
    index overflow: index_overflow(mul)
    matmul 5x7x3: ok
    div 7/-2: refused: an i64 division is not selected for riscv32
    div by zero: refused: an i64 division is not selected for riscv32
    to_i64 nan: refused: type f64 has no riscv32 register in this slice
    recurrences: refused: type f64 has no riscv32 register in this slice |}]

let%expect_test "realized frames (C5)" =
  all_stages "realized" H.Realized;
  [%expect
    {|
    == realized
    max pool: ok
    pointwise: ok
    pointwise bounds: ok
    shifted: coord_out_of_range(t0, W)
    negative shift: coord_out_of_range(t0, W)
    competing axes: coord_out_of_range(t0, H)
    index overflow: index_overflow(mul)
    matmul 5x7x3: ok
    div 7/-2: refused: an i64 division is not selected for riscv32
    div by zero: refused: an i64 division is not selected for riscv32
    to_i64 nan: refused: type f64 has no riscv32 register in this slice
    recurrences: refused: type f64 has no riscv32 register in this slice |}]

(* i64 arithmetic on pairs of words: carries, borrows, the high word of a
   product, and compares decided by the high words, at the boundaries where a
   half overflows. *)
let i64_cases () =
  let arith op = A64T.i64_program (fun bld x y -> B.i64_arith bld op x y) in
  let select c =
    A64T.i64_program (fun bld x y -> B.select bld (B.i64_compare bld c x y) x y)
  in
  let ops =
    Ssa_ir.Ssa_op.
      [
        ("add", arith I64_op.Add);
        ("sub", arith I64_op.Sub);
        ("mul", arith I64_op.Mul);
        ("min by lt", select Compare.Lt);
        ("eq picks", select Compare.Eq);
      ]
  in
  let operands =
    [
      (0xFFFF_FFFFL, 1L);
      (0L, 1L);
      (0x1_0000_0000L, -1L);
      (Int64.max_int, 1L);
      (Int64.min_int, Int64.max_int);
      (-3L, 0x7FFF_FFFFL);
      (0x1234_5678_9ABC_DEF0L, 0x0FED_CBA9_8765_4321L);
      (-0x8000_0000L, -0x8000_0000L);
    ]
  in
  List.concat_map
    (fun (n, p) ->
      List.map
        (fun (x, y) -> (Fmt.str "%s %Lx %Lx" n x y, p, [ (0, i64s [ x; y ]) ]))
        operands)
    ops

let%expect_test "i64 pairs, every stage" =
  List.iter
    (fun stage ->
      List.iter
        (fun (n, p, inputs) ->
          match H.program ~stage p ~inputs with
          | "ok" -> ()
          | s -> Fmt.pr "%s: %s@." n s)
        (i64_cases ()))
    [ H.Selected; H.Allocated; H.Realized ];
  Fmt.pr "done@.";
  [%expect {| done |}]

let%expect_test "selection mutations are caught" =
  let x = Loop_fixtures.load_t0 in
  let pixel body =
    let shape = Loop_fixtures.shape_w 5 in
    ( Loop_fixtures.pixel_kernel ~shape body,
      bind_data ~shape [| Float.nan; -0.5; 0.; 1e30; 9e6 |] )
  in
  let kernel name (k, bind) mutation =
    Fmt.pr "%s: %s@." name
      (H.planned ~mutation ~target:scalar ~numerics:binary32 k ~bind)
  in
  let open Machine_target_riscv32.Rv32_select.Mutation in
  let square = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:37 ~w:1 ~c:1 in
  kernel "contract, x * x + x"
    ( Loop_fixtures.pixel_kernel ~shape:square Expr.Value.(add (mul x x) x),
      bind_data ~shape:square
        (Array.init 37 (fun i -> 1. +. (float_of_int i *. 0.0123))) )
    Contract;
  let reduce_max =
    Loop_programs.reduction_kernel Expr.Reduction.Max
      ~lo:(Expr.Index.assume_position (Expr.Index.const 0))
      ~hi:(Expr.Index.const 3)
  in
  let max_case data =
    (reduce_max, bind_data ~shape:(Loop_programs.s1c 3) data)
  in
  kernel "pruned live, max reduction"
    (max_case [| 1.; Float.nan; 3. |])
    Pruned_live;
  kernel "missing failure word, index overflow"
    (Loop_programs.overflow_kernel, zeros)
    Missing_failure_word;
  kernel "number max, max reduction"
    (max_case [| 1.; Float.nan; 3. |])
    Number_max;
  kernel "unguarded trunc" (pixel (Expr.Value.trunc x)) Unguarded_trunc;
  let carry =
    A64T.i64_program (fun bld x y ->
        B.i64_arith bld Ssa_ir.Ssa_op.I64_op.Add x y)
  in
  Fmt.pr "dropped carry: %s@."
    (H.program ~mutation:Dropped_carry carry
       ~inputs:[ (0, i64s [ 0xFFFF_FFFFL; 1L ]) ]);
  let lt =
    A64T.i64_program (fun bld x y ->
        B.select bld (B.i64_compare bld Ssa_ir.Ssa_op.Compare.Lt x y) x y)
  in
  Fmt.pr "signed compare: %s@."
    (H.program ~mutation:Signed_compare lt
       ~inputs:[ (0, i64s [ 0xFFFF_FFFFL; 1L ]) ]);
  [%expect
    {|
    contract, x * x + x: ok DISAGREE generic vs riscv32: output t1[1]: 0x1.04be18p+1:f32 vs 0x1.04be1ap+1:f32; oracle vs riscv32: output t1[1]: 0x1.04be18p+1:f32 vs 0x1.04be1ap+1:f32
    pruned live, max reduction: refused: selection defect: selected fn0 bb1: %8 is never defined
    missing failure word, index overflow: defect(uninitialized) DISAGREE generic vs riscv32: inconclusive: failure(index_overflow(mul)) vs defect(uninitialized); oracle vs riscv32: inconclusive: failure(index_overflow(mul)) vs defect(uninitialized)
    number max, max reduction: ok DISAGREE generic vs riscv32: output t1[0]: nan:f32 vs 0x1.8p+1:f32; oracle vs riscv32: output t1[0]: nan:f32 vs 0x1.8p+1:f32
    unguarded trunc: ok DISAGREE generic vs riscv32: output t1[0]: nan:f32 vs 0x1p+31:f32; oracle vs riscv32: output t1[0]: nan:f32 vs 0x1p+31:f32
    dropped carry: ok DISAGREE generic vs riscv32: output t1[0]: 4294967296:i64 vs 0:i64; structured vs riscv32: output t1[0]: 4294967296:i64 vs 0:i64
    signed compare: ok DISAGREE generic vs riscv32: output t1[0]: 1:i64 vs 4294967295:i64; structured vs riscv32: output t1[0]: 1:i64 vs 4294967295:i64 |}]

(* A full i64 to binary32, one rounding, at the cases that matter: words,
   ties on either side, sticky bits far below, the largest magnitudes. A
   function of one i64 returning its binary32, generic and selected (where the
   argument arrives as two words). *)
let i64_f32_values =
  [
    0L;
    1L;
    -1L;
    0x7FFF_FFFFL;
    0x8000_0000L;
    0xFFFF_FFFFL;
    0x1_0000_0000L;
    0x1_0000_0001L;
    0x100_0000L;
    0x100_0001L;
    0x100_0003L;
    0xFFFF_FF80L;
    0xFFFF_FF7FL;
    0x8000_0080_0000_0000L;
    0x0123_4567_89AB_CDEFL;
    0x7FFF_FFFF_FFFF_FFFFL;
    Int64.min_int;
    -0x0123_4567_89AB_CDEFL;
    0x0000_0080_0000_0001L;
    0x0000_0100_0000_0001L;
    0x4000_0040_0000_0000L;
    0x4000_0040_0000_0001L;
    0x4000_00C0_0000_0000L;
    -0x4000_0040_0000_0001L;
    0x0000_0001_0000_0080L;
    0x0000_0001_0000_0180L;
    0x00FF_FFFF_8000_0000L;
    0x0000_0100_0001_0001L;
    0x0000_0100_0003_0001L;
    -0x0000_0100_0001_0001L;
    0x4000_0080_0000_0001L;
  ]

let to_f32 () =
  let open Machine_ir in
  let b = Mir_builder.create () in
  let blk = Mir_builder.new_block b [ Mir_type.i64 ] in
  let x = List.hd (Mir_builder.param blk) in
  let f =
    Mir_builder.emit b blk (Mir_op.Fconvert (Mir_op.Fconvert.S64_to_f32, x))
  in
  Mir_builder.return blk [ f ];
  let fn =
    Mir_builder.func b ~id:(Mir_id.Func.of_int 0) ~name:"to_f32" ~entry:blk
      ~results:[ Mir_type.F32 ]
  in
  Result.get_ok
    (Err.payload
       (Mir_verify.generic
          (Mir_builder.program [ fn ] ~main:(Mir_id.Func.of_int 0))))

let%expect_test "i64 to binary32" =
  let open Machine_interp in
  let g = to_f32 () in
  let sel =
    match Err.payload (Machine_target_riscv32.Rv32_select.program g) with
    | Ok r -> r.Machine_target_riscv32.Rv32_select.selected
    | Error e ->
        Fmt.failwith "%a" Machine_target_riscv32.Rv32_select.Refusal.pp e
  in
  let gp = Machine_ir.Mir_verify.Generic.program g in
  let bad = ref 0 in
  List.iter
    (fun x ->
      let memory = Mir_memory.create () in
      let binding =
        Result.get_ok (Mir_interp.instantiate gp memory ~bound:(fun _ -> None))
      in
      let want =
        (Mir_interp.run g memory binding ~args:[ Mir_datum.Bits x ])
          .Mir_interp.outcome
      in
      let s = Machine_target_riscv32.Rv32_stage.Sel.Verified.selected sel in
      let memory = Mir_memory.create () in
      let binding =
        Result.get_ok
          (Mir_interp.instantiate
             s.Machine_target_riscv32.Rv32_stage.Sel.program memory
             ~bound:(fun _ -> None))
      in
      let got =
        (Machine_target_riscv32.Rv32_stage.Interp.run sel memory binding
           ~args:
             [
               Mir_datum.Bits (Int64.logand x 0xFFFF_FFFFL);
               Mir_datum.Bits (Int64.shift_right_logical x 32);
             ])
          .Machine_target_riscv32.Rv32_stage.Interp.outcome
      in
      let show o = Fmt.str "%a" Mir_interp.Outcome.pp o in
      let want = show want and got = show got in
      (* the selected function returns its status after the value *)
      if
        not
          (String.starts_with
             ~prefix:(String.sub want 0 (String.length want - 1))
             got)
      then (
        incr bad;
        Fmt.pr "%Lx: %s vs %s@." x want got))
    i64_f32_values;
  Fmt.pr "%d values, %d disagree@." (List.length i64_f32_values) !bad;
  [%expect {| 31 values, 0 disagree |}]
