open Loop_ir_test
open Ssa_bridge_test.Ssa_fixtures

(* M11.2: planned binary32 vector kernels selected as SSE2 packed forms, their
   selected run against the generic route and the plan's oracle. *)

let show ?features ?stage ?(target = Ssa_ir.Ssa_target.neon128) kernel ~bind =
  Fmt.pr "%s@."
    (X64_harness.planned ?features ?stage ~target
       ~numerics:Ssa_ir.Ssa_numerics.Simd_fp32_ordered kernel ~bind)

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
    [ (2, 3, 16); (3, 5, 17); (5, 7, 33) ];
  [%expect
    {|
    x / 3 - 1.5, w=16: ok
    x / 3 - 1.5, w=37: ok
    sqrt x, w=16: ok
    sqrt x, w=37: ok
    x * x + x, w=16: ok
    x * x + x, w=37: ok
    0 - x, w=16: ok
    0 - x, w=37: ok
    2x3x16: ok
    3x5x17: ok
    5x7x33: ok |}]

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

let%expect_test "a strided access has no packed SSE2 form: a typed refusal" =
  let square = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:17 ~w:17 ~c:1 in
  show
    ~target:(Ssa_ir.Ssa_target.forced Ssa_ir.Ssa_target.neon128)
    (Loop_fixtures.pixel_kernel ~shape:square transposed)
    ~bind:(bind_data ~shape:square (data (17 * 17)));
  [%expect {| refused: vector vload.f32 is not selected for x86_64 |}]

let%expect_test "reference allocation and frames: XMM registers, 16-byte slots"
    =
  let x = Loop_fixtures.load_t0 in
  let shape = Loop_fixtures.shape_w 37 in
  List.iter
    (fun stage ->
      Fmt.pr "x * x + x: ";
      show ~stage
        (Loop_fixtures.pixel_kernel ~shape Expr.Value.(add (mul x x) x))
        ~bind:(bind_data ~shape (data 37));
      let m, k, n = (3, 5, 17) in
      Fmt.pr "matmul: ";
      show ~stage (matmul_kernel ~m ~k ~n)
        ~bind:
          (matmul_bind ~m ~k ~n ~a:(operand 3 (m * k)) ~b:(operand 5 (k * n))))
    X64_harness.[ Allocated; Realized ];
  [%expect
    {|
    x * x + x: ok
    matmul: ok
    x * x + x: ok
    matmul: ok |}]
