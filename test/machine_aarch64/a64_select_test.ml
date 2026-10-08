open Loop_ir_test
open Ssa_bridge_test.Ssa_fixtures

(* C3 for AArch64: the source kernels and integer/conversion programs selected
   and interpreted, agreeing with the generic route and the SSA oracle on
   outputs, failure records (stored, then decoded) and marks. *)

let show s = Fmt.pr "%s@." s
let data_bind data = bind_data ~shape:(Loop_fixtures.shape_w 4) data

let shifted ~dh ~dw =
  let out a d =
    Expr.Index.assume_position
      (Expr.Index.add
         (Expr.Index.of_position (Expr.Index.output a))
         (Expr.Index.const d))
  in
  Loop_fixtures.pixel_kernel
    (Expr.Value.load
       (Expr_bridge.source_of_id (tid 0))
       (Expr.Coord.set
          (Expr.Coord.set
             (Expr_bridge.coord_of_vec6 Symbolic.out_vec)
             Expr.Axis.W (out Expr.Axis.W dw))
          Expr.Axis.H (out Expr.Axis.H dh)))

let%expect_test "source kernels: pointwise, failures, matmul" =
  let zeros = data_bind [| 0.; 0.; 0.; 0. |] in
  show
    (A64_harness.plan Loop_programs.kernel
       ~bind:(data_bind [| -0.; 1.5; nan; 3. |]));
  show
    (A64_harness.plan Loop_programs.kernel
       ~bind:(data_bind [| 1e30; -1e30; 0.1; 16777217. |]));
  show (A64_harness.plan Loop_programs.shifted_kernel ~bind:zeros);
  show (A64_harness.plan (shifted ~dh:0 ~dw:(-1)) ~bind:zeros);
  show (A64_harness.plan (shifted ~dh:1 ~dw:4) ~bind:zeros);
  show (A64_harness.plan Loop_programs.overflow_kernel ~bind:zeros);
  List.iter
    (fun (m, k, n) ->
      let a = operand 3 (m * k) and b = operand 5 (k * n) in
      show
        (A64_harness.plan (matmul_kernel ~m ~k ~n)
           ~bind:(matmul_bind ~m ~k ~n ~a ~b)))
    [ (1, 1, 1); (3, 1, 4); (5, 7, 3); (2, 9, 1) ];
  [%expect
    {|
    ok
    ok
    coord_out_of_range(t0, W)
    coord_out_of_range(t0, W)
    coord_out_of_range(t0, H)
    index_overflow(mul)
    ok
    ok
    ok
    ok |}]

module B = Ssa_ir.Ssa_builder
module F = Ssa_ir_test.Ssa_fixtures

let one fmt id role = F.buffer id ~h:1L ~w:1L fmt role
let cell bld = F.at bld ~h:(B.index bld 0L) ~w:(B.index bld 0L)
let at bld w = F.at bld ~h:(B.index bld 0L) ~w:(B.index bld w)
let i64s xs = Ssa_ir.Ssa_memory.Int64s (Array.of_list xs)
let floats xs = Ssa_ir.Ssa_memory.Floats (Array.of_list xs)

let i64_program f =
  F.build
    ~buffers:
      [
        F.buffer 0 ~h:1L ~w:2L Ssa_ir.Ssa_format.I64 Ssa_ir.Ssa_buffer.Input;
        one Ssa_ir.Ssa_format.I64 1 Ssa_ir.Ssa_buffer.Output;
      ]
    (fun bld ->
      let x = B.load_i64 bld (F.buf 0) (at bld 0L)
      and y = B.load_i64 bld (F.buf 0) (at bld 1L) in
      B.store_i64 bld (F.buf 1) (cell bld) (f bld x y))

(* [5 - x]: a constant only the subtraction's left operand can be *)
let sub_from =
  i64_program (fun bld x _ ->
      B.i64_arith bld Ssa_ir.Ssa_op.I64_op.Sub (B.i64 bld 5L) x)

let to_i64 =
  F.build
    ~buffers:
      [
        one Ssa_ir.Ssa_format.F64 0 Ssa_ir.Ssa_buffer.Input;
        one Ssa_ir.Ssa_format.I64 1 Ssa_ir.Ssa_buffer.Output;
      ]
    (fun bld ->
      let x =
        B.load_f64 bld (F.buf 0) ~decode:Ssa_ir.Ssa_op.Decode.F64_to_f64
          (cell bld)
      in
      B.store_i64 bld (F.buf 1) (cell bld) (B.float_to_i64 bld x))

(* out[0] <- x * y + z (not contracted), out[1] <- (x < y) as 1 or 0,
   out[2] <- max (x, y), out[3] <- i64_to_f32 (float_to_i64 z) *)
let mixed =
  F.build
    ~buffers:
      [
        F.buffer 0 ~h:1L ~w:3L Ssa_ir.Ssa_format.F64 Ssa_ir.Ssa_buffer.Input;
        F.buffer 1 ~h:1L ~w:4L Ssa_ir.Ssa_format.F32 Ssa_ir.Ssa_buffer.Output;
      ]
    (fun bld ->
      let ld w =
        B.load_f64 bld (F.buf 0) ~decode:Ssa_ir.Ssa_op.Decode.F64_to_f64
          (at bld w)
      in
      let x = ld 0L and y = ld 1L and z = ld 2L in
      let st w v =
        B.store_f64 bld (F.buf 1) ~encode:Ssa_ir.Ssa_op.Encode.F32_round
          (at bld w) v
      in
      st 0L
        (B.f64_binary bld Expr.Value.Add
           (B.f64_binary bld Expr.Value.Mul x y)
           z);
      st 1L
        (B.select bld
           (B.float_compare bld Ssa_ir.Ssa_op.Compare.Lt x y)
           (B.f64 bld 1.) (B.f64 bld 0.));
      st 2L (B.f64_max bld x y);
      let n =
        B.select bld
          (B.float_compare bld Ssa_ir.Ssa_op.Compare.Lt (B.f64 bld 1e18) z)
          (B.f64 bld 0.) z
      in
      st 3L (B.f32_to_f64 bld (B.i64_to_f32 bld (B.float_to_i64 bld n))))

(* Branches on float compares: [x = y] and [x < y] each choose which constant
   is stored. *)
let branches =
  F.build
    ~buffers:
      [
        F.buffer 0 ~h:1L ~w:2L Ssa_ir.Ssa_format.F64 Ssa_ir.Ssa_buffer.Input;
        F.buffer 1 ~h:1L ~w:2L Ssa_ir.Ssa_format.F32 Ssa_ir.Ssa_buffer.Output;
      ]
    (fun bld ->
      let ld w =
        B.load_f64 bld (F.buf 0) ~decode:Ssa_ir.Ssa_op.Decode.F64_to_f64
          (at bld w)
      in
      let x = ld 0L and y = ld 1L in
      let st bld w k =
        B.store_f64 bld (F.buf 1) ~encode:Ssa_ir.Ssa_op.Encode.F32_round
          (at bld (Int64.of_int w))
          (B.f64 bld k);
        B.Nil
      in
      List.iteri
        (fun w c ->
          let B.Nil =
            B.if_ bld
              (B.float_compare bld c x y)
              ~then_:(fun bld -> st bld w 1.)
              ~else_:(fun bld -> st bld w 0.)
          in
          ())
        Ssa_ir.Ssa_op.Compare.[ Eq; Lt ])

let%expect_test "integer and conversion programs" =
  let p = i64_program (fun bld x y -> B.i64_div bld x y) in
  List.iter
    (fun (x, y) -> show (A64_harness.program p ~inputs:[ (0, i64s [ x; y ]) ]))
    [ (7L, -2L); (7L, 0L); (Int64.min_int, -1L); (Int64.min_int, 1L) ];
  List.iter
    (fun x -> show (A64_harness.program to_i64 ~inputs:[ (0, floats [ x ]) ]))
    [
      -0.5;
      9223372036854774784.;
      9223372036854775808.;
      Float.nan;
      Float.neg_infinity;
    ];
  let eps = Float.ldexp 1. (-27) in
  List.iter
    (fun xs -> show (A64_harness.program mixed ~inputs:[ (0, floats xs) ]))
    [
      [ 1. +. eps; 1. -. eps; -1. ];
      [ Float.nan; 1.; 3e15 ];
      [ -0.; 0.; -2.5 ];
      [ 2.; 1.; 4611686293305294849. ];
    ];
  List.iter
    (fun xs -> show (A64_harness.program branches ~inputs:[ (0, floats xs) ]))
    [ [ 1.; 1. ]; [ 1.; 2. ]; [ Float.nan; Float.nan ]; [ -0.; 0. ] ];
  [%expect
    {|
    ok
    i64_division_by_zero
    i64_division_overflow
    ok
    ok
    ok
    i64_from_float_out_of_range
    i64_from_float_nan
    i64_from_float_infinite
    ok
    ok
    ok
    ok
    ok
    ok
    ok
    ok |}]

let%expect_test "selection mutations are detected" =
  let open Machine_target_aarch64.A64_select.Mutation in
  let eps = Float.ldexp 1. (-27) in
  Fmt.pr "commuted subtraction: %s@."
    (A64_harness.program ~mutation:Commuted_sub sub_from
       ~inputs:[ (0, i64s [ 7L; 0L ]) ]);
  Fmt.pr "lost NaN branch, fused: %s@."
    (A64_harness.program ~mutation:Fcmp_lt_cond branches
       ~inputs:[ (0, floats [ Float.nan; 1. ]) ]);
  Fmt.pr "contraction: %s@."
    (A64_harness.program ~mutation:Contract mixed
       ~inputs:[ (0, floats [ 1. +. eps; 1. -. eps; -1. ]) ]);
  Fmt.pr "lost NaN branch: %s@."
    (A64_harness.program ~mutation:Fcmp_lt_cond mixed
       ~inputs:[ (0, floats [ Float.nan; 1.; 3. ]) ]);
  Fmt.pr "missing failure word: %s@."
    (A64_harness.plan ~mutation:Missing_failure_word
       Loop_programs.shifted_kernel
       ~bind:(data_bind [| 0.; 0.; 0.; 0. |]));
  Fmt.pr "pruned live: %s@."
    (A64_harness.plan ~mutation:Pruned_live Loop_programs.kernel
       ~bind:(data_bind [| 1.; 2.; 3.; 4. |]));
  [%expect
    {|
    commuted subtraction: ok DISAGREE generic vs aarch64: output t1[0]: -2:i64 vs 2:i64; structured vs aarch64: output t1[0]: -2:i64 vs 2:i64
    lost NaN branch, fused: ok DISAGREE generic vs aarch64: output t1[1]: 0x0p+0:f32 vs 0x1p+0:f32; structured vs aarch64: output t1[1]: 0x0p+0:f32 vs 0x1p+0:f32
    contraction: ok DISAGREE generic vs aarch64: output t1[0]: 0x0p+0:f32 vs -0x1p-54:f32; structured vs aarch64: output t1[0]: 0x0p+0:f32 vs -0x1p-54:f32
    lost NaN branch: ok DISAGREE generic vs aarch64: output t1[1]: 0x0p+0:f32 vs 0x1p+0:f32; structured vs aarch64: output t1[1]: 0x0p+0:f32 vs 0x1p+0:f32
    missing failure word: defect(uninitialized) DISAGREE generic vs aarch64: inconclusive: failure(coord_out_of_range(t0, W)) vs defect(uninitialized); structured vs aarch64: inconclusive: failure(coord_out_of_range(t0, W)) vs defect(uninitialized)
    pruned live: refused: selection defect: selected fn0 bb1: %6 is never defined |}]
