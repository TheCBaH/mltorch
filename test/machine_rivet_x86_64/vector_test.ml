(* Planned binary32 vector kernels selected as SSE2 packed forms, run as
   emulated x86-64 processes against the same selected program on the
   interpreter. *)

open Loop_ir_test
open Ssa_bridge_test.Ssa_fixtures
module H = Native_harness

let show ?(target = Ssa_ir.Ssa_target.neon128) kernel ~bind =
  Fmt.pr "%s@."
    (H.planned ~target ~numerics:Ssa_ir.Ssa_numerics.Simd_fp32_ordered kernel
       ~bind)

let data n = Array.init n (fun i -> (float_of_int (i * 7 mod 11) -. 5.) /. 4.)
let x = Loop_fixtures.load_t0

let%expect_test "pointwise and matmul as packed SSE2" =
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
        [ 16; 37 ])
    [
      ("x / 3 - 1.5", Expr.Value.(sub (div x (const 3.)) (const 1.5)));
      ("sqrt x", Expr.Value.sqrt x);
      ("x * x + x", Expr.Value.(add (mul x x) x));
      ("0 - x", Expr.Value.(sub (const 0.) x));
    ];
  List.iter
    (fun (m, k, n) ->
      let a = operand 3 (m * k) and b = operand 5 (k * n) in
      Fmt.pr "%dx%dx%d: " m k n;
      show (matmul_kernel ~m ~k ~n) ~bind:(matmul_bind ~m ~k ~n ~a ~b))
    [
      (2, 3, 16);
      (3, 5, 17);
      (5, 7, 33);
      (1, 7, 33);
      (5, 1, 33);
      (5, 2, 33);
      (5, 7, 32);
      (5, 7, 34);
    ];
  [%expect
    {|
    x / 3 - 1.5, w=16: ok [emulated: agree]
    x / 3 - 1.5, w=37: ok [emulated: agree]
    sqrt x, w=16: ok [emulated: agree]
    sqrt x, w=37: ok [emulated: agree]
    x * x + x, w=16: ok [emulated: agree]
    x * x + x, w=37: ok [emulated: agree]
    0 - x, w=16: ok [emulated: agree]
    0 - x, w=37: ok [emulated: agree]
    2x3x16: ok [emulated: agree]
    3x5x17: ok [emulated: agree]
    5x7x33: ok [emulated: agree]
    1x7x33: ok [emulated: agree]
    5x1x33: ok [emulated: agree]
    5x2x33: ok [emulated: agree]
    5x7x32: ok [emulated: agree]
    5x7x34: ok [emulated: agree] |}]
