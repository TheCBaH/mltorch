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
    x / 3 - 1.5, w=16: ok (f32, 6 vector operations)
    x / 3 - 1.5, w=17: ok (f32, 6 vector operations)
    x / 3 - 1.5, w=37: ok (f32, 6 vector operations)
    sqrt x, w=1: ok (f64, 0 vector operations)
    sqrt x, w=16: ok (f32, 3 vector operations)
    sqrt x, w=17: ok (f32, 3 vector operations)
    sqrt x, w=37: ok (f32, 3 vector operations)
    x * x + x, w=1: ok (f64, 0 vector operations)
    x * x + x, w=16: ok (f32, 6 vector operations)
    x * x + x, w=17: ok (f32, 6 vector operations)
    x * x + x, w=37: ok (f32, 6 vector operations)
    exp x, w=1: ok (f64, 0 vector operations)
    exp x, w=16: ok (f32, 66 vector operations)
    exp x, w=17: ok (f32, 66 vector operations)
    exp x, w=37: ok (f32, 66 vector operations) |}]

let matmul (m, k, n) =
  let a = operand 3 (m * k) and b = operand 5 (k * n) in
  Fmt.pr "%dx%dx%d: " m k n;
  show (matmul_kernel ~m ~k ~n) ~bind:(matmul_bind ~m ~k ~n ~a ~b)

let%expect_test "matmul: vector sums, odd extents" =
  List.iter matmul [ (1, 1, 1); (2, 3, 16); (3, 5, 17); (5, 7, 33) ];
  [%expect
    {|
    1x1x1: ok (f64, 0 vector operations)
    2x3x16: ok (f32, 11 vector operations)
    3x5x17: ok (f32, 6 vector operations)
    5x7x33: ok (f32, 6 vector operations) |}]

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
    {| defect(bad_access) (f32, 6 vector operations) DISAGREE oracle vs generic: inconclusive: success vs defect(bad_access); oracle vs split: inconclusive: success vs defect(bad_access) |}]

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
    2x3x16: ok (f32, 11 vector operations)
    3x9x17: ok (f32, 6 vector operations) |}]

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
    ok (f32, 6 vector operations) DISAGREE oracle vs split: output t2[12]: -0x1.6ffcb8p+3:f32 vs -0x1.eaa644p+1:f32
    ok (f32, 6 vector operations) DISAGREE oracle vs split: output t2[0]: -0x1.eaa644p+1:f32 vs -0x1.55b3d2p+2:f32 |}]

(* Thirty-two logical lanes ([neon128_x32]): eight 16-byte slices a vector;
   the split mutations are caught there too. *)
let%expect_test "32 lanes in 16-byte slices" =
  let m, k, n = (3, 5, 37) in
  let a = operand 3 (m * k) and b = operand 5 (k * n) in
  let target = Ssa_ir.Ssa_target.forced Ssa_ir.Ssa_target.neon128_x32 in
  show ~target (matmul_kernel ~m ~k ~n) ~bind:(matmul_bind ~m ~k ~n ~a ~b);
  List.iter
    (fun mutation ->
      show ~target ~split_mutation:mutation (matmul_kernel ~m ~k ~n)
        ~bind:(matmul_bind ~m ~k ~n ~a ~b))
    Machine_ir.Mir_vsplit.Mutation.[ Stale_slice; Swapped_slices ];
  [%expect
    {|
    ok (f32, 6 vector operations)
    ok (f32, 6 vector operations) DISAGREE oracle vs split: output t2[28]: -0x1.ad5184p+3:f32 vs 0x1.c9cb2ap+4:f32
    ok (f32, 6 vector operations) DISAGREE oracle vs split: output t2[0]: 0x1.c9cb2ap+4:f32 vs 0x1.1a8fc4p+4:f32 |}]

(* A bfloat16 result stored from binary32 lanes: the rounding in int32 lanes
   and the 16-bit store, against the plan's lane-by-lane oracle, over ties
   either way, subnormals, the overflow to infinity and NaNs whose payload
   sits in the low half or carries the sign; the canonical-NaN select dropped
   is caught. *)
let bf16_kernel ~shape =
  let open Loop_fixtures in
  Err.or_raise ~pp_error:Kernel.pp_error
    (Kernel.create
       ~inputs:
         [
           {
             Kernel.Input.id = tid 0;
             sg = sg 0 shape f32;
             binding = Kernel.Binding.Caller;
           };
         ]
       ~values:
         [
           {
             Kernel.Value.id = tid 1;
             sg = sg 1 shape (Payload.Fmt Payload.BF16);
             computation = Region_group.Ref.Solo (Region_program.pixel x);
             result = Kernel.Result_conversion.Round_bf16;
           };
         ]
       ~outputs:[ tid 1 ]
       ())

let bf16_inputs n =
  let special =
    [|
      0x3F808000l;
      0x3F818000l;
      0x3F808001l;
      0x00008000l;
      0x80018000l;
      0x7F7FFFFFl;
      0xFF800000l;
      0x7FFFFFFFl;
      0xFFC00001l;
      0x7F800001l;
    |]
  in
  Array.init n (fun i ->
      if i < Array.length special then Int32.float_of_bits special.(i)
      else (float_of_int (i * 7 mod 11) -. 5.) /. 4.)

let%expect_test "bfloat16 results: rounding and canonical NaN in lanes" =
  List.iter
    (fun (tname, target) ->
      List.iter
        (fun n ->
          Fmt.pr "%s, w=%d: " tname n;
          let shape = Loop_fixtures.shape_w n in
          show ~target (bf16_kernel ~shape)
            ~bind:(bind_data ~shape (bf16_inputs n)))
        [ 16; 17; 37 ])
    [
      ("neon128", neon);
      ("neon128_x32", Ssa_ir.Ssa_target.forced Ssa_ir.Ssa_target.neon128_x32);
    ];
  let shape = Loop_fixtures.shape_w 16 in
  show ~mutation:Machine_lower.Mir_lower.Mutation.Bf16_nan_dropped
    (bf16_kernel ~shape)
    ~bind:(bind_data ~shape (bf16_inputs 16));
  List.iter
    (fun mutation ->
      show ~split_mutation:mutation (bf16_kernel ~shape)
        ~bind:(bind_data ~shape (bf16_inputs 16)))
    Machine_ir.Mir_vsplit.Mutation.[ Stale_slice; Swapped_slices ];
  [%expect
    {|
    neon128, w=16: ok (f32, 16 vector operations)
    neon128, w=17: ok (f32, 16 vector operations)
    neon128, w=37: ok (f32, 16 vector operations)
    neon128_x32, w=16: ok (f64, 9 vector operations)
    neon128_x32, w=17: ok (f64, 9 vector operations)
    neon128_x32, w=37: ok (f32, 16 vector operations)
    ok (f32, 13 vector operations) DISAGREE oracle vs generic: output t1[7]: 32704:i64 vs 32768:i64; oracle vs split: output t1[7]: 32704:i64 vs 32768:i64
    ok (f32, 16 vector operations) DISAGREE oracle vs split: output t1[8]: 32704:i64 vs 16256:i64
    ok (f32, 16 vector operations) DISAGREE oracle vs split: output t1[0]: 16256:i64 vs 32770:i64 |}]
