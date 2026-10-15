open Loop_ir_test
open Ssa_bridge_test.Ssa_fixtures

(* M11.2: planned binary32 vector kernels selected as Advanced SIMD forms,
   their selected run against the generic route and the plan's oracle. *)

let show ?mutation ?(target = Ssa_ir.Ssa_target.neon128)
    ?(numerics = Ssa_ir.Ssa_numerics.Simd_fp32_ordered) kernel ~bind =
  Fmt.pr "%s@." (A64_harness.planned ?mutation ~target ~numerics kernel ~bind)

let data n = Array.init n (fun i -> (float_of_int (i * 7 mod 11) -. 5.) /. 4.)
let x = Loop_fixtures.load_t0

let%expect_test "pointwise and matmul on NEON" =
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

let square = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:17 ~w:17 ~c:1

let%expect_test "vector selection mutations are caught" =
  let cases =
    [
      ( "x * x + x",
        Loop_fixtures.pixel_kernel ~shape:(Loop_fixtures.shape_w 37)
          Expr.Value.(add (mul x x) x),
        bind_data ~shape:(Loop_fixtures.shape_w 37) (data 37) );
      ( "matmul 3x5x17",
        matmul_kernel ~m:3 ~k:5 ~n:17,
        matmul_bind ~m:3 ~k:5 ~n:17 ~a:(operand 3 15) ~b:(operand 5 85) );
      ( "transposed, every legal loop vectorized",
        Loop_fixtures.pixel_kernel ~shape:square transposed,
        bind_data ~shape:square (data (17 * 17)) );
    ]
  in
  let target = Ssa_ir.Ssa_target.forced Ssa_ir.Ssa_target.neon128 in
  List.iter
    (fun (name, kernel, bind) ->
      Fmt.pr "%s: " name;
      show ~target kernel ~bind;
      (* the binary32 plans read and write binary32 vectors directly; the
         narrowing of two binary64 halves is a binary64 plan's binary32
         store *)
      List.iter
        (fun (m, mutation, numerics) ->
          Fmt.pr "  %s: " m;
          show ~target ~mutation ~numerics kernel ~bind)
        Machine_target_aarch64.A64_select.Mutation.
          [
            ( "contiguous lanes",
              Contiguous_lanes,
              Ssa_ir.Ssa_numerics.Simd_fp32_ordered );
            ("contract", Contract, Ssa_ir.Ssa_numerics.Simd_fp32_ordered);
            ( "dropped half, binary64 plan",
              Dropped_half,
              Ssa_ir.Ssa_numerics.Reference_f64 );
          ])
    cases;
  [%expect
    {|
    x * x + x: ok
      contiguous lanes: ok
      contract: ok
      dropped half, binary64 plan: ok DISAGREE generic vs aarch64: output t1[2]: -0x1p-2:f32 vs 0x0p+0:f32; oracle vs aarch64: output t1[2]: -0x1p-2:f32 vs 0x0p+0:f32
    matmul 3x5x17: ok
      contiguous lanes: ok
      contract: ok DISAGREE generic vs aarch64: output t2[33]: -0x1.29e508p+1:f32 vs -0x1.29e5p+1:f32; oracle vs aarch64: output t2[33]: -0x1.29e508p+1:f32 vs -0x1.29e5p+1:f32
      dropped half, binary64 plan: ok
    transposed, every legal loop vectorized: ok
      contiguous lanes: ok DISAGREE generic vs aarch64: output t1[1]: 0x1p+0:f32 vs 0x1p-1:f32; oracle vs aarch64: output t1[1]: 0x1p+0:f32 vs 0x1p-1:f32
      contract: ok
      dropped half, binary64 plan: ok DISAGREE generic vs aarch64: output t1[2]: 0x1p-1:f32 vs 0x0p+0:f32; oracle vs aarch64: output t1[2]: 0x1p-1:f32 vs 0x0p+0:f32 |}]

(* Masks: a vector compare makes a Q register of all-ones and all-zero lanes,
   a select chooses bit by bit, and a NaN is false for every ordered compare. *)
let mask_kernels =
  let zero = Expr.Value.const 0. in
  [
    ( "x < 0 ? -x : x / 3",
      Expr.Value.(
        select (Expr.Bool.value_lt x zero) (sub zero x) (div x (const 3.))) );
    ( "x == 0 ? 0 : 1",
      Expr.Value.(select (Expr.Bool.value_eq x zero) (const 0.) (const 1.)) );
  ]

let%expect_test "masks on NEON" =
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
  let d = data 37 in
  d.(0) <- Float.nan;
  d.(2) <- 0.;
  Fmt.pr "%s: " name;
  show (Loop_fixtures.pixel_kernel ~shape body) ~bind:(bind_data ~shape d);
  List.iter
    (fun (m, mutation) ->
      Fmt.pr "  %s: " m;
      show ~mutation
        (Loop_fixtures.pixel_kernel ~shape body)
        ~bind:(bind_data ~shape d))
    Machine_target_aarch64.A64_select.Mutation.
      [
        ("arms exchanged", Mask_arms); ("less-than unswapped", Mask_lt_operands);
      ];
  [%expect
    {|
    x < 0 ? -x : x / 3: ok
      arms exchanged: ok DISAGREE generic vs aarch64: output t1[1]: 0x1.555556p-3:f32 vs -0x1p-1:f32; oracle vs aarch64: output t1[1]: 0x1.555556p-3:f32 vs -0x1p-1:f32
      less-than unswapped: ok DISAGREE generic vs aarch64: output t1[1]: 0x1.555556p-3:f32 vs -0x1p-1:f32; oracle vs aarch64: output t1[1]: 0x1.555556p-3:f32 vs -0x1p-1:f32 |}]

(* Binary32 dequantization as integer lanes, selected: the integer register
   loads, the sign extensions from it and the int32 conversion, against the
   generic route and the plan's oracle. *)
let%expect_test "binary32 dequantization as integer lanes, selected" =
  let f32 x = fst (Quant.round_scale x) in
  List.iter
    (fun (name, fmt, quant, n) ->
      let cells =
        Array.init n (fun i -> Int64.of_int ((i * 37 mod 256) - 128))
      in
      let bind id =
        if Tensor_id.equal id (Loop_fixtures.tid 0) then
          Some (Loop_programs.raw_tensor fmt ~quant n cells)
        else None
      in
      Fmt.pr "%s: %s@." name
        (A64_harness.planned ~target:Ssa_ir.Ssa_target.neon128
           ~numerics:Ssa_ir.Ssa_numerics.Fp32_ordered
           (Loop_programs.format_kernel ~fmt ~quant n)
           ~bind))
    [
      ( "i8 per tensor, 32 lanes",
        Payload.Fmt Payload.I8,
        Quant.per_tensor ~scale:(f32 0.37) ~zero_point:(-3.),
        32 );
      ( "i8 per tensor, 37 lanes",
        Payload.Fmt Payload.I8,
        Quant.per_tensor ~scale:(f32 0.05) ~zero_point:1.25,
        37 );
      ( "i16 per tensor, 32 lanes",
        Payload.Fmt Payload.I16,
        Quant.per_tensor ~scale:(f32 1e-3) ~zero_point:7.,
        32 );
    ];
  [%expect
    {|
    i8 per tensor, 32 lanes: ok
    i8 per tensor, 37 lanes: ok
    i16 per tensor, 32 lanes: ok |}]

(* Binary16 and bfloat16 as integer lanes, selected: every encoding through
   the zero extensions, lane shifts and bitwise forms, against the generic
   route and the plan's oracle. *)
let%expect_test "binary16 and bfloat16 as integer lanes, selected" =
  List.iter
    (fun (name, fmt) ->
      let n = 65536 in
      let cells = Array.init n Int64.of_int in
      let bind id =
        if Tensor_id.equal id (Loop_fixtures.tid 0) then
          Some (Loop_programs.raw_tensor fmt n cells)
        else None
      in
      Fmt.pr "%s: %s@." name
        (A64_harness.planned ~target:Ssa_ir.Ssa_target.neon128
           ~numerics:Ssa_ir.Ssa_numerics.Fp32_ordered
           (Loop_programs.format_kernel ~fmt n)
           ~bind))
    [ ("f16", Payload.Fmt Payload.F16); ("bf16", Payload.Fmt Payload.BF16) ];
  [%expect {|
    f16: ok
    bf16: ok |}]
