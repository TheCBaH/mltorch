(* Planned binary32 vector kernels selected as SSE2 packed forms, run as
   x86-64 processes against the same selected program on the
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
    x / 3 - 1.5, w=16: ok [process: agree]
    x / 3 - 1.5, w=37: ok [process: agree]
    sqrt x, w=16: ok [process: agree]
    sqrt x, w=37: ok [process: agree]
    x * x + x, w=16: ok [process: agree]
    x * x + x, w=37: ok [process: agree]
    0 - x, w=16: ok [process: agree]
    0 - x, w=37: ok [process: agree]
    2x3x16: ok [process: agree]
    3x5x17: ok [process: agree]
    5x7x33: ok [process: agree]
    1x7x33: ok [process: agree]
    5x1x33: ok [process: agree]
    5x2x33: ok [process: agree]
    5x7x32: ok [process: agree]
    5x7x34: ok [process: agree] |}]

(* x * x + x under the relaxed policy is one fused multiply-add per lane: it
   needs the FMA feature, and the process runs it on this CPU. *)
let%expect_test "relaxed binary32 as fused multiply-add" =
  let shape = Loop_fixtures.shape_w 37 in
  let d = data 37 in
  let kernel = Loop_fixtures.pixel_kernel ~shape Expr.Value.(add (mul x x) x) in
  let show features =
    Fmt.pr "%s@."
      (H.planned ?features ~target:Ssa_ir.Ssa_target.neon128
         ~numerics:Ssa_ir.Ssa_numerics.Simd_fp32_relaxed kernel
         ~bind:(bind_data ~shape d))
  in
  show None;
  show (Some Machine_ir.Mir_target.Feature.[ Sse2; Fma ]);
  [%expect
    {|
    not built: selection: needs fma, which the program's features do not include
    ok [process: agree] |}]

(* the input read transposed: lane k of an output row reads down a column *)
let transposed =
  let at a = Expr.Index.output a in
  Expr.Value.load
    (Expr_bridge.source_of_id (tid 0))
    (Expr.Coord.set
       (Expr.Coord.set
          (Expr_bridge.coord_of_vec6 Symbolic.out_vec)
          Expr.Axis.W (at Expr.Axis.H))
       Expr.Axis.H (at Expr.Axis.W))

(* Masks and strided lanes in the process, on the CPU: packed compares, bitwise
   selects, and lane inserts built from SHUFPS. *)
let%expect_test "masks and strided access as processes" =
  let zero = Expr.Value.const 0. in
  List.iter
    (fun (name, body) ->
      List.iter
        (fun n ->
          Fmt.pr "%s, w=%d: " name n;
          let d = data n in
          d.(0) <- Float.nan;
          d.(1) <- -0.;
          d.(2) <- 0.;
          d.(3) <- 3.4e38;
          let shape = Loop_fixtures.shape_w n in
          show
            (Loop_fixtures.pixel_kernel ~shape body)
            ~bind:(bind_data ~shape d))
        [ 16; 37 ])
    [
      ( "x < 0 ? -x : x / 3",
        Expr.Value.(
          select (Expr.Bool.value_lt x zero) (sub zero x) (div x (const 3.))) );
      ( "x == 0 ? 0 : 1",
        Expr.Value.(select (Expr.Bool.value_eq x zero) (const 0.) (const 1.)) );
    ];
  let square = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:17 ~w:17 ~c:1 in
  Fmt.pr "transposed read: ";
  show
    ~target:(Ssa_ir.Ssa_target.forced Ssa_ir.Ssa_target.neon128)
    (Loop_fixtures.pixel_kernel ~shape:square transposed)
    ~bind:(bind_data ~shape:square (data (17 * 17)));
  [%expect
    {|
    x < 0 ? -x : x / 3, w=16: ok [process: agree]
    x < 0 ? -x : x / 3, w=37: ok [process: agree]
    x == 0 ? 0 : 1, w=16: ok [process: agree]
    x == 0 ? 0 : 1, w=37: ok [process: agree]
    transposed read: ok [process: agree] |}]
