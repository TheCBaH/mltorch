open Loop_ir_test
open Ssa_bridge_test.Ssa_fixtures
open Machine_ir
open Machine_target_x86_64
module H = X64_harness
module A64T = Machine_aarch64_test.A64_select_test
module B = Ssa_ir.Ssa_builder
module F = Ssa_ir_test.Ssa_fixtures

(* x86-64, interpreted on this host: C3 (selected), C4 (allocated, checked)
   and C5 (realized frames) on the source kernels and the integer and
   conversion programs, the baseline's refusals, and the target mutations. *)

let show s = Fmt.pr "%s@." s
let data_bind = A64T.data_bind
let zeros = data_bind [| 0.; 0.; 0.; 0. |]
let floats xs = Ssa_ir.Ssa_memory.Floats (Array.of_list xs)
let i64s xs = Ssa_ir.Ssa_memory.Int64s (Array.of_list xs)
let eps = Float.ldexp 1. (-27)

let kernels =
  [
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
    ("div overflow", div, [ (0, i64s [ Int64.min_int; -1L ]) ]);
    ("to_i64 2^63-1024", A64T.to_i64, [ (0, floats [ 9223372036854774784. ]) ]);
    ("to_i64 nan", A64T.to_i64, [ (0, floats [ Float.nan ]) ]);
    ("mixed", A64T.mixed, [ (0, floats [ 1. +. eps; 1. -. eps; -1. ]) ]);
    ("mixed nan", A64T.mixed, [ (0, floats [ Float.nan; 1.; 3e15 ]) ]);
    ("mixed zeros", A64T.mixed, [ (0, floats [ -0.; 0.; -2.5 ]) ]);
    ("branches", A64T.branches, [ (0, floats [ 1.; 2. ]) ]);
    ("branches nan", A64T.branches, [ (0, floats [ Float.nan; Float.nan ]) ]);
    ("branches zeros", A64T.branches, [ (0, floats [ -0.; 0. ]) ]);
    ( "recurrences",
      Machine_alloc_test.Alloc_test.recurrences 5,
      Machine_alloc_test.Alloc_test.inputs );
  ]

let all_stages label stage =
  Fmt.pr "== %s@." label;
  List.iter
    (fun (n, k, bind) -> Fmt.pr "%s: %s@." n (H.plan ~stage k ~bind))
    kernels;
  List.iter
    (fun (n, p, inputs) -> Fmt.pr "%s: %s@." n (H.program ~stage p ~inputs))
    (programs ())

let%expect_test "selected (C3)" =
  all_stages "selected" H.Selected;
  [%expect
    {|
    == selected
    pointwise: ok
    pointwise bounds: ok
    shifted: coord_out_of_range(t0, W)
    negative shift: coord_out_of_range(t0, W)
    competing axes: coord_out_of_range(t0, H)
    index overflow: index_overflow(mul)
    matmul 5x7x3: ok
    div 7/-2: ok
    div by zero: i64_division_by_zero
    div overflow: i64_division_overflow
    to_i64 2^63-1024: ok
    to_i64 nan: i64_from_float_nan
    mixed: ok
    mixed nan: ok
    mixed zeros: ok
    branches: ok
    branches nan: ok
    branches zeros: ok
    recurrences: ok |}]

let%expect_test "allocated and checked (C4)" =
  all_stages "allocated" H.Allocated;
  [%expect
    {|
    == allocated
    pointwise: ok
    pointwise bounds: ok
    shifted: coord_out_of_range(t0, W)
    negative shift: coord_out_of_range(t0, W)
    competing axes: coord_out_of_range(t0, H)
    index overflow: index_overflow(mul)
    matmul 5x7x3: ok
    div 7/-2: ok
    div by zero: i64_division_by_zero
    div overflow: i64_division_overflow
    to_i64 2^63-1024: ok
    to_i64 nan: i64_from_float_nan
    mixed: ok
    mixed nan: ok
    mixed zeros: ok
    branches: ok
    branches nan: ok
    branches zeros: ok
    recurrences: ok |}]

let%expect_test "realized frames (C5)" =
  all_stages "realized" H.Realized;
  let mm = matmul_kernel ~m:5 ~k:3 ~n:3
  and bind = matmul_bind ~m:5 ~k:3 ~n:3 ~a:(operand 3 15) ~b:(operand 5 9) in
  Fmt.pr "large frame: %s@." (H.plan ~stage:H.Realized ~pad:40_000L mm ~bind);
  [%expect
    {|
    == realized
    pointwise: ok
    pointwise bounds: ok
    shifted: coord_out_of_range(t0, W)
    negative shift: coord_out_of_range(t0, W)
    competing axes: coord_out_of_range(t0, H)
    index overflow: index_overflow(mul)
    matmul 5x7x3: ok
    div 7/-2: ok
    div by zero: i64_division_by_zero
    div overflow: i64_division_overflow
    to_i64 2^63-1024: ok
    to_i64 nan: i64_from_float_nan
    mixed: ok
    mixed nan: ok
    mixed zeros: ok
    branches: ok
    branches nan: ok
    branches zeros: ok
    recurrences: ok
    large frame: ok |}]

(* trunc(x), stored as binary32 *)
let trunc_program =
  F.build
    ~buffers:
      [
        F.buffer 0 ~h:1L ~w:1L Ssa_ir.Ssa_format.F64 Ssa_ir.Ssa_buffer.Input;
        F.buffer 1 ~h:1L ~w:1L Ssa_ir.Ssa_format.F32 Ssa_ir.Ssa_buffer.Output;
      ]
    (fun bld ->
      let at = F.at bld ~h:(B.index bld 0L) ~w:(B.index bld 0L) in
      let x =
        B.load_f64 bld (F.buf 0) ~decode:Ssa_ir.Ssa_op.Decode.F64_to_f64 at
      in
      B.store_f64 bld (F.buf 1) ~encode:Ssa_ir.Ssa_op.Encode.F32_round at
        (B.f64_unary bld Expr.Value.Trunc x))

let%expect_test "baseline refusals and extensions" =
  let fma_inputs = [ (0, floats [ 1. +. eps; 1. -. eps; -1. ]) ] in
  let fma = Machine_source_test.Mir_integer_test.fma in
  Fmt.pr "fma, SSE2: %s@."
    (H.program ~fma:Mir_planning.Fma.Exact fma ~inputs:fma_inputs);
  Fmt.pr "fma, FMA: %s@."
    (H.program
       ~features:Mir_target.Feature.[ Fma; Sse2 ]
       ~fma:Mir_planning.Fma.Exact fma ~inputs:fma_inputs);
  Fmt.pr "trunc, SSE2: %s@."
    (H.program trunc_program ~inputs:[ (0, floats [ -2.75 ]) ]);
  Fmt.pr "trunc, SSE4.1: %s@."
    (H.program
       ~features:Mir_target.Feature.[ Sse2; Sse41 ]
       trunc_program
       ~inputs:[ (0, floats [ -2.75 ]) ]);
  [%expect
    {|
    fma, SSE2: refused: needs fma, which the program's features do not include
    fma, FMA: ok
    trunc, SSE2: refused: needs sse4.1, which the program's features do not include
    trunc, SSE4.1: ok |}]

(* (x == y) as 1 or 0, stored as binary32 *)
let equal_program =
  F.build
    ~buffers:
      [
        F.buffer 0 ~h:1L ~w:2L Ssa_ir.Ssa_format.F64 Ssa_ir.Ssa_buffer.Input;
        F.buffer 1 ~h:1L ~w:1L Ssa_ir.Ssa_format.F32 Ssa_ir.Ssa_buffer.Output;
      ]
    (fun bld ->
      let at w = F.at bld ~h:(B.index bld 0L) ~w:(B.index bld w) in
      let ld w =
        B.load_f64 bld (F.buf 0) ~decode:Ssa_ir.Ssa_op.Decode.F64_to_f64 (at w)
      in
      let x = ld 0L and y = ld 1L in
      B.store_f64 bld (F.buf 1) ~encode:Ssa_ir.Ssa_op.Encode.F32_round (at 0L)
        (B.select bld
           (B.float_compare bld Ssa_ir.Ssa_op.Compare.Eq x y)
           (B.f64 bld 1.) (B.f64 bld 0.)))

let%expect_test "x86-64 selection mutations are detected" =
  let open X64_select.Mutation in
  let div = A64T.i64_program (fun bld x y -> B.i64_div bld x y) in
  Fmt.pr "division operand swap: %s@."
    (H.program ~mutation:Division_swap div ~inputs:[ (0, i64s [ 7L; -2L ]) ]);
  Fmt.pr "scaled address: %s@."
    (H.plan ~mutation:Scaled_address Loop_programs.kernel
       ~bind:(data_bind [| 1.; 2.; 3.; 4. |]));
  Fmt.pr "equality without parity: %s@."
    (H.program ~mutation:No_parity equal_program
       ~inputs:[ (0, floats [ Float.nan; Float.nan ]) ]);
  Fmt.pr "fused float equality: %s@."
    (H.program ~mutation:Fused_float_eq A64T.branches
       ~inputs:[ (0, floats [ Float.nan; Float.nan ]) ]);
  Fmt.pr "maximum without NaN repair: %s@."
    (H.program ~mutation:Max_no_nan A64T.mixed
       ~inputs:[ (0, floats [ Float.nan; 1.; 3e15 ]) ]);
  Fmt.pr "commuted subtraction: %s@."
    (H.program ~mutation:Commuted_sub A64T.sub_from
       ~inputs:[ (0, i64s [ 7L; 0L ]) ]);
  Fmt.pr "contraction: %s@."
    (H.program
       ~features:Mir_target.Feature.[ Fma; Sse2 ]
       ~mutation:Contract A64T.mixed
       ~inputs:[ (0, floats [ 1. +. eps; 1. -. eps; -1. ]) ]);
  Fmt.pr "missing failure word: %s@."
    (H.plan ~mutation:Missing_failure_word Loop_programs.shifted_kernel
       ~bind:zeros);
  Fmt.pr "pruned live: %s@."
    (H.plan ~mutation:Pruned_live Loop_programs.kernel
       ~bind:(data_bind [| 1.; 2.; 3.; 4. |]));
  [%expect
    {|
    division operand swap: ok DISAGREE generic vs x86_64: output t1[0]: -3:i64 vs 0:i64; structured vs x86_64: output t1[0]: -3:i64 vs 0:i64
    scaled address: defect(bad_access) DISAGREE generic vs x86_64: inconclusive: success vs defect(bad_access); structured vs x86_64: inconclusive: success vs defect(bad_access)
    equality without parity: ok DISAGREE generic vs x86_64: output t1[0]: 0x0p+0:f32 vs 0x1p+0:f32; structured vs x86_64: output t1[0]: 0x0p+0:f32 vs 0x1p+0:f32
    fused float equality: ok DISAGREE generic vs x86_64: output t1[0]: 0x0p+0:f32 vs 0x1p+0:f32; structured vs x86_64: output t1[0]: 0x0p+0:f32 vs 0x1p+0:f32
    maximum without NaN repair: ok DISAGREE generic vs x86_64: output t1[2]: nan:f32 vs 0x1p+0:f32; structured vs x86_64: output t1[2]: nan:f32 vs 0x1p+0:f32
    commuted subtraction: ok DISAGREE generic vs x86_64: output t1[0]: -2:i64 vs 2:i64; structured vs x86_64: output t1[0]: -2:i64 vs 2:i64
    contraction: ok DISAGREE generic vs x86_64: output t1[0]: 0x0p+0:f32 vs -0x1p-54:f32; structured vs x86_64: output t1[0]: 0x0p+0:f32 vs -0x1p-54:f32
    missing failure word: defect(uninitialized) DISAGREE generic vs x86_64: inconclusive: failure(coord_out_of_range(t0, W)) vs defect(uninitialized); structured vs x86_64: inconclusive: failure(coord_out_of_range(t0, W)) vs defect(uninitialized)
    pruned live: refused: selection defect: selected fn0 bb1: %6 is never defined |}]

let%expect_test "allocation and frame mutations on x86-64" =
  let mm = matmul_kernel ~m:5 ~k:3 ~n:3
  and bind = matmul_bind ~m:5 ~k:3 ~n:3 ~a:(operand 3 15) ~b:(operand 5 9) in
  Fmt.pr "two-address tie broken: %s@."
    (H.plan ~stage:H.Allocated
       ~alloc_mutation:Machine_alloc.Mir_ref_alloc.Mutation.Tie mm ~bind);
  Fmt.pr "cycle scratch: %s@."
    (H.program ~stage:H.Allocated
       ~alloc_mutation:Machine_alloc.Mir_ref_alloc.Mutation.Cycle_scratch
       (Machine_alloc_test.Alloc_test.recurrences 5)
       ~inputs:Machine_alloc_test.Alloc_test.inputs);
  Fmt.pr "misaligned frame: %s@."
    (H.plan ~stage:H.Realized
       ~frame_mutation:Machine_alloc.Mir_frame.Mutation.Misalign mm ~bind);
  [%expect
    {|
    two-address tie broken: rejected: physical verifier: allocated fn0 bb4 i103: target constraint: a tied result not in its use's register
    cycle scratch: rejected: checker: fn0 bb5: [slot13:8] does not hold %13
    misaligned frame: rejected: physical verifier: allocated fn0: target constraint: a frame size that breaks stack alignment |}]
