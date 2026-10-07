open Ssa_ir
module B = Ssa_builder
module F = Ssa_ir_test.Ssa_fixtures

(* M4.1: checked i64 division, float-to-i64, one-rounding conversions, floor
   and ceiling division, maximum and the pool predicate, gather checks and a
   contracted multiply-add, each through the structured SSA interpreter and
   the generic route; two injected lowering defects are detected. *)

let one fmt id role = F.buffer id ~h:1L ~w:1L fmt role
let cell bld = F.at bld ~h:(B.index bld 0L) ~w:(B.index bld 0L)
let i64s xs = Ssa_memory.Int64s (Array.of_list xs)
let floats xs = Ssa_memory.Floats (Array.of_list xs)

let run ?mutation ?fma p inputs =
  Fmt.pr "%s@." (Mir_source.check_program ?mutation ?fma p ~inputs)

(* out <- f (x, y) over i64 cells *)
let i64_program f =
  F.build
    ~buffers:
      [
        F.buffer 0 ~h:1L ~w:2L Ssa_format.I64 Ssa_buffer.Input;
        one Ssa_format.I64 1 Ssa_buffer.Output;
      ]
    (fun bld ->
      let x =
        B.load_i64 bld (F.buf 0)
          (F.at bld ~h:(B.index bld 0L) ~w:(B.index bld 0L))
      in
      let y =
        B.load_i64 bld (F.buf 0)
          (F.at bld ~h:(B.index bld 0L) ~w:(B.index bld 1L))
      in
      B.store_i64 bld (F.buf 1) (cell bld) (f bld x y))

let%expect_test "checked i64 division" =
  let p = i64_program (fun bld x y -> B.i64_div bld x y) in
  List.iter
    (fun (x, y) -> run p [ (0, i64s [ x; y ]) ])
    [
      (7L, -2L);
      (-7L, 2L);
      (7L, 0L);
      (Int64.min_int, -1L);
      (Int64.min_int, 0L);
      (Int64.min_int, 1L);
    ];
  [%expect
    {|
    ok [-3:i64]
    ok [-3:i64]
    i64_division_by_zero
    i64_division_overflow
    i64_division_by_zero
    ok [-9223372036854775808:i64] |}]

(* out <- float_to_i64 x, x an f64 cell *)
let to_i64 =
  F.build
    ~buffers:
      [
        one Ssa_format.F64 0 Ssa_buffer.Input;
        one Ssa_format.I64 1 Ssa_buffer.Output;
      ]
    (fun bld ->
      let x =
        B.load_f64 bld (F.buf 0) ~decode:Ssa_op.Decode.F64_to_f64 (cell bld)
      in
      B.store_i64 bld (F.buf 1) (cell bld) (B.float_to_i64 bld x))

let%expect_test "float to i64: domain, order and limits" =
  List.iter
    (fun x -> run to_i64 [ (0, floats [ x ]) ])
    [
      -0.5;
      -0.;
      4.9e-324;
      9007199254740993.;
      -9223372036854775808.;
      9223372036854774784.;
      9223372036854775808.;
      -9223372036854777856.;
      Float.nan;
      Float.infinity;
      Float.neg_infinity;
    ];
  [%expect
    {|
    ok [0:i64]
    ok [0:i64]
    ok [0:i64]
    ok [9007199254740992:i64]
    ok [-9223372036854775808:i64]
    ok [9223372036854774784:i64]
    i64_from_float_out_of_range
    i64_from_float_out_of_range
    i64_from_float_nan
    i64_from_float_infinite
    i64_from_float_infinite |}]

(* out[0] <- i64_to_f32 n (as binary32), out[1] <- i64_to_f64 n rounded back
   to i64, out[2] <- floor_div 3, out[3] <- ceil_div 3 of the index n *)
let conversions =
  F.build
    ~buffers:
      [
        one Ssa_format.I64 0 Ssa_buffer.Input;
        F.buffer 1 ~h:1L ~w:1L Ssa_format.F32 Ssa_buffer.Output;
        F.buffer 2 ~h:1L ~w:3L Ssa_format.I64 Ssa_buffer.Output;
      ]
    (fun bld ->
      let n = B.load_i64 bld (F.buf 0) (cell bld) in
      let at w = F.at bld ~h:(B.index bld 0L) ~w:(B.index bld w) in
      B.store_f64 bld (F.buf 1) ~encode:Ssa_op.Encode.F32_round (at 0L)
        (B.f32_to_f64 bld (B.i64_to_f32 bld n));
      B.store_i64 bld (F.buf 2) (at 0L)
        (B.float_to_i64 bld (B.i64_to_f64 bld n));
      let small =
        B.select bld
          (B.i64_compare bld Ssa_op.Compare.Lt n (B.i64 bld 1000L))
          (B.select bld
             (B.i64_compare bld Ssa_op.Compare.Lt (B.i64 bld (-1000L)) n)
             n (B.i64 bld 0L))
          (B.i64 bld 0L)
      in
      let ix = B.index_of_i64 bld small in
      B.store_i64 bld (F.buf 2) (at 1L)
        (B.index_to_i64 bld (B.index_floor_div bld 3L ix));
      B.store_i64 bld (F.buf 2) (at 2L)
        (B.index_to_i64 bld (B.index_ceil_div bld 3L ix)))

let%expect_test "one-rounding conversions and floor/ceiling division" =
  List.iter
    (fun n -> run conversions [ (0, i64s [ n ]) ])
    [
      0L;
      -7L;
      7L;
      -9L;
      16777217L;
      0x20000020000001L;
      Int64.add (Int64.shift_left 1L 62) (Int64.add (Int64.shift_left 1L 38) 1L);
      Int64.max_int;
      Int64.min_int;
    ];
  [%expect
    {|
    ok [0x0p+0:f32 0:i64 0:i64 0:i64]
    ok [-0x1.cp+2:f32 -7:i64 -3:i64 -2:i64]
    ok [0x1.cp+2:f32 7:i64 2:i64 3:i64]
    ok [-0x1.2p+3:f32 -9:i64 -3:i64 -3:i64]
    ok [0x1p+24:f32 16777217:i64 0:i64 0:i64]
    ok [0x1.000002p+53:f32 9007199791611904:i64 0:i64 0:i64]
    ok [0x1.000002p+62:f32 4611686293305294848:i64 0:i64 0:i64]
    i64_from_float_out_of_range
    ok [-0x1p+63:f32 -9223372036854775808:i64 0:i64 0:i64] |}]

(* out <- max (a, b), and pool_better a b as 1 or 0 *)
let maxima =
  F.build
    ~buffers:
      [
        F.buffer 0 ~h:1L ~w:2L Ssa_format.F32 Ssa_buffer.Input;
        F.buffer 1 ~h:1L ~w:2L Ssa_format.F32 Ssa_buffer.Output;
      ]
    (fun bld ->
      let at w = F.at bld ~h:(B.index bld 0L) ~w:(B.index bld w) in
      let a =
        B.load_f64 bld (F.buf 0) ~decode:Ssa_op.Decode.F32_to_f64 (at 0L)
      in
      let b =
        B.load_f64 bld (F.buf 0) ~decode:Ssa_op.Decode.F32_to_f64 (at 1L)
      in
      B.store_f64 bld (F.buf 1) ~encode:Ssa_op.Encode.F32_round (at 0L)
        (B.f64_max bld a b);
      B.store_f64 bld (F.buf 1) ~encode:Ssa_op.Encode.F32_round (at 1L)
        (B.select bld (B.pool_better bld a b) (B.f64 bld 1.) (B.f64 bld 0.)))

let%expect_test "maximum and the pool predicate: NaN and signed zero" =
  List.iter
    (fun (a, b) -> run maxima [ (0, floats [ a; b ]) ])
    [
      (-0., 0.);
      (0., -0.);
      (Float.nan, 1.);
      (1., Float.nan);
      (2., 1.);
      (1., 2.);
      (1e-45, 0.);
    ];
  [%expect
    {|
    ok [0x0p+0:f32 0x0p+0:f32]
    ok [0x0p+0:f32 0x0p+0:f32]
    ok [nan:f32 0x0p+0:f32]
    ok [nan:f32 0x1p+0:f32]
    ok [0x1p+1:f32 0x0p+0:f32]
    ok [0x1p+1:f32 0x1p+0:f32]
    ok [0x1p-149:f32 0x0p+0:f32] |}]

let gather =
  F.build
    ~buffers:
      [
        one Ssa_format.I64 0 Ssa_buffer.Input;
        one Ssa_format.I64 1 Ssa_buffer.Output;
      ]
    (fun bld ->
      let raw = B.load_i64 bld (F.buf 0) (cell bld) in
      B.check_gather bld raw ~extent:4L;
      B.store_i64 bld (F.buf 1) (cell bld) raw)

let%expect_test "gather index checks" =
  List.iter
    (fun n -> run gather [ (0, i64s [ n ]) ])
    [ -4L; 3L; -5L; 4L; Int64.min_int ];
  [%expect
    {|
    ok [-4:i64]
    ok [3:i64]
    gather_index_out_of_range
    gather_index_out_of_range
    gather_index_out_of_range |}]

(* out <- x * y + z, contracted to one fused operation *)
let fma =
  let p =
    F.build
      ~buffers:
        [
          F.buffer 0 ~h:1L ~w:3L Ssa_format.F64 Ssa_buffer.Input;
          one Ssa_format.F32 1 Ssa_buffer.Output;
        ]
      (fun bld ->
        let at w = F.at bld ~h:(B.index bld 0L) ~w:(B.index bld w) in
        let ld w =
          B.load_f64 bld (F.buf 0) ~decode:Ssa_op.Decode.F64_to_f64 (at w)
        in
        let x = ld 0L and y = ld 1L and z = ld 2L in
        B.store_f64 bld (F.buf 1) ~encode:Ssa_op.Encode.F32_round (at 0L)
          (B.f64_binary bld Expr.Value.Add z
             (B.f64_binary bld Expr.Value.Mul x y)))
  in
  fst (Ssa_opt_contract.pass ~scalar:true p)

let%expect_test "a contracted multiply-add rounds once, and needs permission" =
  Fmt.pr "fused operations: %d@." (Ssa_opt_contract.contracted fma);
  let eps = Float.ldexp 1. (-27) in
  (* (1+e)(1-e) - 1 = -e^2, which a separately rounded product loses *)
  let inputs = [ (0, floats [ 1. +. eps; 1. -. eps; -1. ]) ] in
  run ~fma:Machine_ir.Mir_planning.Fma.Exact fma inputs;
  run fma inputs;
  [%expect
    {|
    fused operations: 1
    ok [-0x1p-54:f32]
    refused: a fused multiply-add under fma=forbidden |}]

let%expect_test "lowering mutations: conversion order, double rounding" =
  let open Machine_lower.Mir_lower.Mutation in
  run ~mutation:Conversion_order to_i64 [ (0, floats [ Float.nan ]) ];
  run ~mutation:Conversion_order to_i64 [ (0, floats [ Float.infinity ]) ];
  run ~mutation:Double_rounding conversions
    [
      ( 0,
        i64s
          [
            Int64.add (Int64.shift_left 1L 62)
              (Int64.add (Int64.shift_left 1L 38) 1L);
          ] );
    ];
  [%expect
    {|
    i64_from_float_out_of_range DISAGREE structured vs generic: failure row: i64_from_float_nan() vs i64_from_float_out_of_range(nan:f64)
    i64_from_float_out_of_range DISAGREE structured vs generic: failure row: i64_from_float_infinite() vs i64_from_float_out_of_range(infinity:f64)
    ok [0x1p+62:f32 4611686293305294848:i64 0:i64 0:i64] DISAGREE structured vs generic: output t1[0]: 0x1.000002p+62:f32 vs 0x1p+62:f32 |}]
