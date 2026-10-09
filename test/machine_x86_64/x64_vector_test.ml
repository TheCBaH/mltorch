open Loop_ir_test
open Ssa_bridge_test.Ssa_fixtures

(* M11.2: planned binary32 vector kernels selected as SSE2 packed forms, their
   selected run against the generic route and the plan's oracle. *)

let show ?mutation ?features ?stage ?(target = Ssa_ir.Ssa_target.neon128) kernel
    ~bind =
  Fmt.pr "%s@."
    (X64_harness.planned ?mutation ?features ?stage ~target
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

(* A strided access is one scalar access per lane, in lane order; a loaded
   vector is built by lane inserts, with SHUFPS alone. *)
let strided_kernel () =
  let square = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:17 ~w:17 ~c:1 in
  ( square,
    Loop_fixtures.pixel_kernel ~shape:square transposed,
    bind_data ~shape:square (data (17 * 17)) )

let%expect_test "a strided access as lane loads and inserts" =
  let square, kernel, bind = strided_kernel () in
  ignore square;
  show ~target:(Ssa_ir.Ssa_target.forced Ssa_ir.Ssa_target.neon128) kernel ~bind;
  [%expect {| ok |}]

let%expect_test "a lane insert on the wrong lane is caught" =
  let _, kernel, bind = strided_kernel () in
  show ~mutation:Machine_target_x86_64.X64_select.Mutation.Insert_next_lane
    ~target:(Ssa_ir.Ssa_target.forced Ssa_ir.Ssa_target.neon128)
    kernel ~bind;
  [%expect
    {| ok DISAGREE generic vs x86_64: output t1[0]: -0x1.4p+0:f32 vs 0x0p+0:f32; oracle vs x86_64: output t1[0]: -0x1.4p+0:f32 vs 0x0p+0:f32 |}]

(* Masks: a packed compare makes a register of all-ones and all-zero lanes, a
   select chooses bit by bit, and a NaN is false for every ordered compare. *)
let mask_kernels =
  let zero = Expr.Value.const 0. in
  [
    ( "x < 0 ? -x : x / 3",
      Expr.Value.(
        select (Expr.Bool.value_lt x zero) (sub zero x) (div x (const 3.))) );
    ( "x == 0 ? 0 : 1",
      Expr.Value.(select (Expr.Bool.value_eq x zero) (const 0.) (const 1.)) );
  ]

let mask_data n =
  let d = data n in
  d.(0) <- Float.nan;
  d.(1) <- -0.;
  d.(2) <- 0.;
  d.(3) <- 3.4e38;
  d

let%expect_test "masks as packed compares" =
  List.iter
    (fun (name, body) ->
      List.iter
        (fun n ->
          Fmt.pr "%s, w=%d: " name n;
          let shape = Loop_fixtures.shape_w n in
          show
            (Loop_fixtures.pixel_kernel ~shape body)
            ~bind:(bind_data ~shape (mask_data n)))
        [ 16; 37 ])
    mask_kernels;
  [%expect
    {|
    x < 0 ? -x : x / 3, w=16: ok
    x < 0 ? -x : x / 3, w=37: ok
    x == 0 ? 0 : 1, w=16: ok
    x == 0 ? 0 : 1, w=37: ok |}]

let%expect_test "mask selection mutations are caught" =
  let name, body = List.hd mask_kernels in
  let shape = Loop_fixtures.shape_w 37 in
  Fmt.pr "%s: " name;
  show
    (Loop_fixtures.pixel_kernel ~shape body)
    ~bind:(bind_data ~shape (mask_data 37));
  List.iter
    (fun (m, mutation) ->
      Fmt.pr "  %s: " m;
      show ~mutation
        (Loop_fixtures.pixel_kernel ~shape body)
        ~bind:(bind_data ~shape (mask_data 37)))
    Machine_target_x86_64.X64_select.Mutation.
      [
        ("arms exchanged", Mask_arms); ("less-than as less-or-equal", Mask_lt_le);
      ];
  [%expect
    {|
    x < 0 ? -x : x / 3: ok
      arms exchanged: ok DISAGREE generic vs x86_64: output t1[1]: -0x0p+0:f32 vs 0x0p+0:f32; oracle vs x86_64: output t1[1]: -0x0p+0:f32 vs 0x0p+0:f32
      less-than as less-or-equal: ok DISAGREE generic vs x86_64: output t1[1]: -0x0p+0:f32 vs 0x0p+0:f32; oracle vs x86_64: output t1[1]: -0x0p+0:f32 vs 0x0p+0:f32 |}]

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
