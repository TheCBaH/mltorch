open Loop_ir_test
open Ssa_bridge_test.Ssa_fixtures

(* M11.1: planned binary32 vector kernels as generic Machine IR vectors,
   against the plan's lane-by-lane oracle. *)

let neon = Ssa_ir.Ssa_target.neon128
let ordered = Ssa_ir.Ssa_numerics.Simd_fp32_ordered

let show ?mutation ?split_mutation ?(target = neon) ?(numerics = ordered) kernel
    ~bind =
  Fmt.pr "%s@."
    (Mir_source.check_planned ?mutation ?split_mutation ~target ~numerics
       (Fusion_plan.default kernel)
       ~bind)

let data n = Array.init n (fun i -> (float_of_int (i * 7 mod 11) -. 5.) /. 4.)
let x = Loop_fixtures.load_t0

let pointwise =
  [
    ("x / 3 - 1.5", Expr.Value.(sub (div x (const 3.)) (const 1.5)));
    ("sqrt x", Expr.Value.sqrt x);
    ("x * x + x", Expr.Value.(add (mul x x) x));
    ("exp x", Expr.Value.exp x);
  ]

let%expect_test "pointwise: lanes, tails and special values" =
  List.iter
    (fun (name, body) ->
      List.iter
        (fun n ->
          Fmt.pr "%s, w=%d: " name n;
          let d = data n in
          if n > 2 then (
            d.(0) <- Float.nan;
            d.(1) <- -0.;
            d.(2) <- 3.4e38);
          let shape = Loop_fixtures.shape_w n in
          show
            (Loop_fixtures.pixel_kernel ~shape body)
            ~bind:(bind_data ~shape d))
        [ 1; 16; 17; 37 ])
    pointwise;
  [%expect
    {|
    x / 3 - 1.5, w=1: ok (f64, 0 vector operations)
    x / 3 - 1.5, w=16: ok (f32, 10 vector operations)
    x / 3 - 1.5, w=17: ok (f32, 10 vector operations)
    x / 3 - 1.5, w=37: ok (f32, 10 vector operations)
    sqrt x, w=1: ok (f64, 0 vector operations)
    sqrt x, w=16: ok (f32, 7 vector operations)
    sqrt x, w=17: ok (f32, 7 vector operations)
    sqrt x, w=37: ok (f32, 7 vector operations)
    x * x + x, w=1: ok (f64, 0 vector operations)
    x * x + x, w=16: ok (f32, 14 vector operations)
    x * x + x, w=17: ok (f32, 14 vector operations)
    x * x + x, w=37: ok (f32, 14 vector operations)
    exp x, w=1: ok (f64, 0 vector operations)
    exp x, w=16: ok (f32, 38 vector operations)
    exp x, w=17: ok (f32, 38 vector operations)
    exp x, w=37: ok (f32, 38 vector operations) |}]

let matmul (m, k, n) =
  let a = operand 3 (m * k) and b = operand 5 (k * n) in
  Fmt.pr "%dx%dx%d: " m k n;
  show (matmul_kernel ~m ~k ~n) ~bind:(matmul_bind ~m ~k ~n ~a ~b)

let%expect_test "matmul: vector sums, odd extents" =
  List.iter matmul [ (1, 1, 1); (2, 3, 16); (3, 5, 17); (5, 7, 33) ];
  [%expect
    {|
    1x1x1: ok (f64, 0 vector operations)
    2x3x16: ok (f32, 19 vector operations)
    3x5x17: ok (f32, 10 vector operations)
    5x7x33: ok (f32, 10 vector operations) |}]

let%expect_test "another target's lane count" =
  List.iter
    (fun (m, k, n) ->
      let a = operand 3 (m * k) and b = operand 5 (k * n) in
      Fmt.pr "wasm128 %dx%dx%d: " m k n;
      show ~target:Ssa_ir.Ssa_target.wasm128 (matmul_kernel ~m ~k ~n)
        ~bind:(matmul_bind ~m ~k ~n ~a ~b))
    [ (2, 3, 8); (3, 5, 9) ];
  [%expect
    {|
    wasm128 2x3x8: ok (f64, 10 vector operations)
    wasm128 3x5x9: ok (f64, 10 vector operations) |}]

let%expect_test "a lane stride doubled is caught" =
  let m, k, n = (3, 5, 17) in
  let a = operand 3 (m * k) and b = operand 5 (k * n) in
  show ~mutation:Machine_lower.Mir_lower.Mutation.Vector_stride
    (matmul_kernel ~m ~k ~n)
    ~bind:(matmul_bind ~m ~k ~n ~a ~b);
  [%expect
    {| defect(bad_access) (f32, 10 vector operations) DISAGREE oracle vs generic: inconclusive: success vs defect(bad_access); oracle vs split: inconclusive: success vs defect(bad_access) |}]

let%expect_test "the relaxed policy's plan" =
  List.iter
    (fun (m, k, n) ->
      let a = operand 3 (m * k) and b = operand 5 (k * n) in
      Fmt.pr "%dx%dx%d: " m k n;
      show ~numerics:Ssa_ir.Ssa_numerics.Simd_fp32_relaxed
        (matmul_kernel ~m ~k ~n)
        ~bind:(matmul_bind ~m ~k ~n ~a ~b))
    [ (2, 3, 16); (3, 9, 17) ];
  [%expect
    {|
    2x3x16: ok (f32, 19 vector operations)
    3x9x17: ok (f32, 10 vector operations) |}]

let%expect_test "16-byte slices: swapped or stale slices are caught" =
  let m, k, n = (3, 5, 17) in
  let a = operand 3 (m * k) and b = operand 5 (k * n) in
  List.iter
    (fun mutation ->
      show ~split_mutation:mutation (matmul_kernel ~m ~k ~n)
        ~bind:(matmul_bind ~m ~k ~n ~a ~b))
    Machine_ir.Mir_vsplit.Mutation.[ Stale_slice; Swapped_slices ];
  [%expect
    {|
    ok (f32, 10 vector operations) DISAGREE oracle vs split: output t2[12]: -0x1.6ffcb8p+3:f32 vs -0x1.eaa644p+1:f32
    ok (f32, 10 vector operations) DISAGREE oracle vs split: output t2[0]: -0x1.eaa644p+1:f32 vs -0x1.55b3d2p+2:f32 |}]
