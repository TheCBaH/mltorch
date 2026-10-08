open Loop_ir_test
open Ssa_bridge_test.Ssa_fixtures

(* C2: fresh source kernels through every route, and the lowering mutations
   the comparison detects. The status column is the generic route's; the
   bracket is the reference's own verdict against the structured SSA route. *)

let pointwise body = Loop_fixtures.pixel_kernel body
let data_bind ?(shape = Loop_fixtures.shape_w 4) data = bind_data ~shape data

let show ?mutation ?offsets_mutation ?narrow_mutation kernel ~bind =
  Fmt.pr "%s@."
    (Mir_source.check ?mutation ?offsets_mutation ?narrow_mutation
       (Fusion_plan.default kernel)
       ~bind)

(* the input read at output coordinate [w + dw] on W and [h] on H *)
let shifted ~dh ~dw =
  let out a d =
    Expr.Index.assume_position
      (Expr.Index.add
         (Expr.Index.of_position (Expr.Index.output a))
         (Expr.Index.const d))
  in
  pointwise
    (Expr.Value.load
       (Expr_bridge.source_of_id (tid 0))
       (Expr.Coord.set
          (Expr.Coord.set
             (Expr_bridge.coord_of_vec6 Symbolic.out_vec)
             Expr.Axis.W (out Expr.Axis.W dw))
          Expr.Axis.H (out Expr.Axis.H dh)))

let noncommutative =
  pointwise
    (Expr.Value.sub
       (Expr.Value.div Loop_fixtures.load_t0 (Expr.Value.const 3.))
       (Expr.Value.const 1.5))

let%expect_test
    "pointwise: signed zero, NaN, binary32 boundaries, noncommutative" =
  show Loop_programs.kernel ~bind:(data_bind [| -0.; 1.5; nan; 3. |]);
  show Loop_programs.kernel ~bind:(data_bind [| 1e30; -1e30; 0.1; 16777217. |]);
  show noncommutative ~bind:(data_bind [| 7.; -0.; 1e-40; 3.4e38 |]);
  [%expect
    {|
    ok [reference: agree]
    ok [reference: agree]
    ok [reference: agree] |}]

let%expect_test "failures: coordinates, competing axes, index overflow" =
  let zeros = data_bind [| 0.; 0.; 0.; 0. |] in
  show Loop_programs.shifted_kernel ~bind:zeros;
  show (shifted ~dh:0 ~dw:(-1)) ~bind:zeros;
  show (shifted ~dh:1 ~dw:4) ~bind:zeros;
  show Loop_programs.overflow_kernel ~bind:zeros;
  [%expect
    {|
    coord_out_of_range(t0, W) [reference: agree on failure: coord_out_of_range]
    coord_out_of_range(t0, W) [reference: agree on failure: coord_out_of_range]
    coord_out_of_range(t0, H) [reference: agree on failure: coord_out_of_range]
    index_overflow(mul) [reference: DISAGREE: only ssa failed: index_overflow] |}]

let matmul ?offsets_mutation ?narrow_mutation (m, k, n) =
  let a = operand 3 (m * k) and b = operand 5 (k * n) in
  Fmt.pr "%dx%dx%d: " m k n;
  show ?offsets_mutation ?narrow_mutation (matmul_kernel ~m ~k ~n)
    ~bind:(matmul_bind ~m ~k ~n ~a ~b)

let%expect_test "matmul, odd and empty reductions" =
  List.iter matmul [ (1, 1, 1); (1, 3, 2); (3, 1, 4); (5, 7, 3); (2, 9, 1) ];
  List.iter
    (fun (lo, hi) ->
      let plan =
        Loop_programs.reduction_kernel Expr.Reduction.Sum
          ~lo:(Expr.Index.assume_position (Expr.Index.const lo))
          ~hi:(Expr.Index.const hi)
      in
      Fmt.pr "sum [%d,%d): " lo hi;
      show plan ~bind:(fun id ->
          if Tensor_id.equal id (tid 0) then
            Some
              (Loop_fixtures.f32_tensor (Loop_programs.s1c 3) (fun c ->
                   0.25 +. float_of_int (Dim.to_int (Vec6.get c Axis.C))))
          else None))
    [ (0, 0); (2, 1); (0, 3) ];
  [%expect
    {|
    1x1x1: ok [reference: agree]
    1x3x2: ok [reference: agree]
    3x1x4: ok [reference: agree]
    5x7x3: ok [reference: agree]
    2x9x1: ok [reference: agree]
    sum [0,0): ok [reference: agree]
    sum [2,1): ok [reference: agree]
    sum [0,3): ok [reference: agree] |}]

(* Every route above also runs the program with its loop-invariant address
   terms hoisted; dropping a term that varies in the loop reads the wrong
   elements. *)
let%expect_test "a hoisted address missing a varying term is detected" =
  matmul ~offsets_mutation:Machine_ir.Mir_offsets.Mutation.Dropped_term (5, 7, 3);
  [%expect
    {| 5x7x3: ok [reference: agree] DISAGREE structured vs offsets: output t2[0]: -0x1.fc2c3ep+2:f32 vs -0x1.eaa64ep+5:f32 |}]

(* ... and with its in-domain index arithmetic at 32 bits; a narrowed
   constant one larger steps a loop past outputs it should write. *)
let%expect_test "a narrowed constant one larger is detected" =
  matmul ~narrow_mutation:Machine_ir.Mir_narrow.Mutation.Shifted_constant
    (5, 7, 3);
  [%expect
    {| 5x7x3: ok [reference: agree] DISAGREE structured vs narrowed: output t2[0]: -0x1.fc2c3ep+2:f32 vs 0x1.185f0ap+1:f32 |}]

let%expect_test "lowering mutations are detected" =
  let zeros = data_bind [| 0.; 0.; 0.; 0. |] in
  let open Machine_lower.Mir_lower.Mutation in
  let m x = Some x in
  Fmt.pr "zero extension: ";
  show ?mutation:(m Zero_extend) (shifted ~dh:0 ~dw:(-1)) ~bind:zeros;
  Fmt.pr "byte scaling: ";
  show ?mutation:(m Scale_bytes) Loop_programs.kernel
    ~bind:(data_bind [| 1.; 2.; 3.; 4. |]);
  Fmt.pr "operand order: ";
  show ?mutation:(m Operand_order) noncommutative
    ~bind:(data_bind [| 7.; 1.; 2.; 3. |]);
  Fmt.pr "guard order: ";
  show ?mutation:(m Guard_order) (shifted ~dh:1 ~dw:4) ~bind:zeros;
  Fmt.pr "eager load: ";
  show ?mutation:(m Eager_load) Loop_programs.shifted_kernel ~bind:zeros;
  [%expect
    {|
    zero extension: defect(domain) [reference: agree on failure: coord_out_of_range] DISAGREE structured vs generic: inconclusive: failure(coord_out_of_range(t0, W)) vs defect(domain); structured vs offsets: inconclusive: failure(coord_out_of_range(t0, W)) vs defect(domain); structured vs narrowed: failure row: coord_out_of_range(t0, W)(0:i64, 0:i64, 0:i64, 0:i64, -1:i64, 0:i64) vs coord_out_of_range(t0, W)(0:i64, 0:i64, 0:i64, 0:i64, 4294967295:i64, 0:i64)
    byte scaling: defect(bad_access) [reference: agree] DISAGREE structured vs generic: inconclusive: success vs defect(bad_access); structured vs offsets: inconclusive: success vs defect(bad_access); structured vs narrowed: inconclusive: success vs defect(bad_access)
    operand order: ok [reference: agree] DISAGREE structured vs generic: output t1[0]: 0x1.aaaaaap-1:f32 vs 0x1.124924p+0:f32; structured vs offsets: output t1[0]: 0x1.aaaaaap-1:f32 vs 0x1.124924p+0:f32; structured vs narrowed: output t1[0]: 0x1.aaaaaap-1:f32 vs 0x1.124924p+0:f32
    guard order: coord_out_of_range(t0, W) [reference: agree on failure: coord_out_of_range] DISAGREE structured vs generic: failure row: coord_out_of_range(t0, H)(0:i64, 0:i64, 0:i64, 1:i64, 4:i64, 0:i64) vs coord_out_of_range(t0, W)(0:i64, 0:i64, 0:i64, 1:i64, 4:i64, 0:i64); structured vs offsets: failure row: coord_out_of_range(t0, H)(0:i64, 0:i64, 0:i64, 1:i64, 4:i64, 0:i64) vs coord_out_of_range(t0, W)(0:i64, 0:i64, 0:i64, 1:i64, 4:i64, 0:i64); structured vs narrowed: failure row: coord_out_of_range(t0, H)(0:i64, 0:i64, 0:i64, 1:i64, 4:i64, 0:i64) vs coord_out_of_range(t0, W)(0:i64, 0:i64, 0:i64, 1:i64, 4:i64, 0:i64)
    eager load: defect(bad_access) [reference: agree on failure: coord_out_of_range] DISAGREE structured vs generic: inconclusive: failure(coord_out_of_range(t0, W)) vs defect(bad_access); structured vs offsets: inconclusive: failure(coord_out_of_range(t0, W)) vs defect(bad_access); structured vs narrowed: inconclusive: failure(coord_out_of_range(t0, W)) vs defect(bad_access) |}]

module B = Ssa_ir.Ssa_builder
module F = Ssa_ir_test.Ssa_fixtures

let row_buffer id n role = F.buffer id ~h:1L ~w:n Ssa_ir.Ssa_format.F32 role
let idx bld n = B.index bld (Int64.of_int n)

let store bld i x =
  B.store_f64 bld (F.buf 1) ~encode:Ssa_ir.Ssa_op.Encode.F32_round
    (F.at bld ~h:(idx bld 0) ~w:(idx bld i))
    x

(* (a, b) <- (b, a + b) [n] times from (0, 1), and a plain swap [n] times: each
   back edge rebinds both parameters at once. *)
let recurrences n =
  F.build
    ~buffers:
      [
        row_buffer 0 4L Ssa_ir.Ssa_buffer.Input;
        row_buffer 1 4L Ssa_ir.Ssa_buffer.Output;
      ]
    (fun bld ->
      let (B.Cons (a, B.Cons (b, B.Nil))) =
        B.for_ bld ~lo:(idx bld 0) ~hi:(idx bld n)
          ~init:(B.Cons (B.f64 bld 0., B.Cons (B.f64 bld 1., B.Nil)))
          (fun bld _ (B.Cons (a, B.Cons (b, B.Nil))) ->
            B.Cons (b, B.Cons (B.f64_binary bld Expr.Value.Add a b, B.Nil)))
      in
      store bld 0 a;
      store bld 1 b;
      let (B.Cons (x, B.Cons (y, B.Nil))) =
        B.for_ bld ~lo:(idx bld 0) ~hi:(idx bld n)
          ~init:(B.Cons (B.f64 bld 1., B.Cons (B.f64 bld 2., B.Nil)))
          (fun _ _ (B.Cons (x, B.Cons (y, B.Nil))) ->
            B.Cons (y, B.Cons (x, B.Nil)))
      in
      store bld 2 x;
      store bld 3 y)

let%expect_test "recurrences and simultaneous transfers" =
  let inputs = [ (0, Ssa_ir.Ssa_memory.Floats [| 0.; 0.; 0.; 0. |]) ] in
  List.iter
    (fun n ->
      Fmt.pr "n=%d: %s@." n (Mir_source.check_program (recurrences n) ~inputs))
    [ 0; 1; 5; 10 ];
  Fmt.pr "sequential: %s@."
    (Mir_source.check_program
       ~mutation:Machine_lower.Mir_lower.Mutation.Sequential_transfer
       (recurrences 5) ~inputs);
  [%expect
    {|
    n=0: ok [0x0p+0:f32 0x1p+0:f32 0x1p+0:f32 0x1p+1:f32]
    n=1: ok [0x1p+0:f32 0x1p+0:f32 0x1p+1:f32 0x1p+0:f32]
    n=5: ok [0x1.4p+2:f32 0x1p+3:f32 0x1p+1:f32 0x1p+0:f32]
    n=10: ok [0x1.b8p+5:f32 0x1.64p+6:f32 0x1p+0:f32 0x1p+1:f32]
    sequential: ok [0x1.4p+2:f32 0x1p+3:f32 0x1p+1:f32 0x1p+1:f32] DISAGREE structured vs generic: output t1[3]: 0x1p+0:f32 vs 0x1p+1:f32 |}]

let%expect_test "an exp kernel through its libm helper" =
  show
    (Loop_programs.unary_kernel Expr.Value.Exp)
    ~bind:(data_bind [| 1.; 2.; 3.; 4. |]);
  [%expect {| ok [reference: agree] |}]

let%expect_test "a vector built directly: splat, then extract" =
  let module B = Ssa_ir.Ssa_builder in
  let out =
    {
      Ssa_ir.Ssa_buffer.id = Ssa_ir.Ssa_id.Buffer.of_int 0;
      extents = Expr.Coord.make ~n:1L ~t:1L ~d:1L ~h:1L ~w:1L ~c:1L;
      format = Ssa_ir.Ssa_format.F32;
      role = Ssa_ir.Ssa_buffer.Output;
    }
  in
  let p =
    Err.or_raise ~pp_error:Ssa_ir.Ssa_verify.pp_error
      (B.program ~buffers:[ out ] (fun bld ->
           let v =
             B.vec_splat bld
               ~lanes:(Ssa_ir.Ssa_type.Lanes.of_int 2)
               (B.f64 bld 1. :> Ssa_ir.Ssa_value.t)
           in
           let x =
             B.as_f64
               (B.vec_extract bld ~lane:(Ssa_ir.Ssa_type.Lane.of_int 0) v)
           in
           B.store_f64 bld out.Ssa_ir.Ssa_buffer.id
             ~encode:Ssa_ir.Ssa_op.Encode.F32_round
             (B.Flat (B.index bld 0L))
             x))
  in
  print_endline (Mir_source.check_program p ~inputs:[]);
  [%expect {| ok [0x1p+0:f32] |}]
