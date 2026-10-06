open Ssa_ir
open Ssa_ir_test.Ssa_fixtures
module B = Ssa_builder

(* The vector surface: operations, the lane-wise interpreter and the scalar-lane
   reference. Every program here is run twice, directly and after expanding it to
   scalar lanes, and the two must agree on every cell, on the failure, and on the
   loads, stores and marks counted. *)

let lanes = Ssa_type.Lanes.of_int
let lane = Ssa_type.Lane.of_int
let row n format role id = buffer id ~h:1L ~w:n format role
let idx bld n = B.index bld (Int64.of_int n)

let bufs =
  [
    row 8L Ssa_format.F32 Ssa_buffer.Input 0;
    row 8L Ssa_format.F32 Ssa_buffer.Output 1;
  ]

(* A coordinate along W at [w], and the lane step along W. *)
let raw (type a) (x : a B.value) : Ssa_value.t = (x :> Ssa_value.t)

let at_w bld w =
  let z = raw (idx bld 0) in
  Expr.Coord.make ~n:z ~t:z ~d:z ~h:z ~w:(raw w) ~c:z

let steps_w k = Expr.Coord.make ~n:0L ~t:0L ~d:0L ~h:0L ~w:k ~c:0L

let vload bld ?(buffer = 0) ?(n = 4) ?(step = 1L) w =
  B.vec_load bld (buf buffer) ~decode:Ssa_op.Decode.F32_to_f64 ~lanes:(lanes n)
    ~steps:(steps_w step) (at_w bld w)

let vstore bld ?(n = 4) ?(step = 1L) w v =
  B.vec_store bld (buf 1) ~encode:Ssa_op.Encode.F32_round ~lanes:(lanes n)
    ~steps:(steps_w step) (at_w bld w) v

let op1 bld op = B.lanewise bld op
let bin bld o a b = op1 bld (Ssa_op.Float_binary (o, a, b))
let f64v bld x = raw (B.f64 bld x)
let program f = build ~buffers:bufs f

(* Generated C against the interpreter, on the same programs the vector suite
   runs: every cell and the failure must agree bit for bit, for the program as
   built and as the optimizer leaves it. *)
module Lf = Loop_ir_test.Loop_fixtures

type outcome = { result : string; cells : float array }

let shape = Lf.shape_w 8

let structured p input =
  let out = Array.make 8 0. in
  let memory = memory [ (0, floats (Array.copy input)); (1, floats out) ] in
  let result =
    match run_result p ~memory with
    | Ok () -> "ok"
    | Error e -> Fmt.str "%a" Ssa_interp.pp_error e
  in
  { result; cells = out }

let c ?(relaxed = false) p input =
  let lowered =
    match Ssa_wasm.lower ~relaxed_madd:relaxed p with
    | Ok k -> k
    | Error e -> Fmt.failwith "%a" Ssa_wasm.pp_error e
  in
  let loop =
    Lf.program
      ~buffers:
        (List.map Ssa_bridge.Loop_of_ssa.loop_buffer (Ssa_wasm.arguments p))
      []
  in
  let bind id =
    if Tensor_id.equal id (Lf.tid 0) then
      Some (Lf.f32_tensor shape (fun c -> input.((Vec6.offset shape c :> int))))
    else None
  in
  match Err.payload (Loop_wasm_exec.exec_module ~lowered loop ~bind) with
  | Error e -> { result = Fmt.str "%a" Loop_wasm_exec.pp_error e; cells = [||] }
  | Ok outputs ->
      let t = Tensor_id.Map.find (Lf.tid 1) outputs in
      { result = "ok"; cells = Array.of_list (Lf.cells t 8) }

let execute = c

let same a b =
  a.result = b.result
  && Array.length a.cells = Array.length b.cells
  && Array.for_all2 Core.Float_bits.equal_portable a.cells b.cells

(* A NaN's sign is the host's: x86 makes -nan where arm64 makes nan. *)
let pp_cell ppf x =
  if Float.is_nan x then Fmt.string ppf "nan" else Fmt.float ppf x

let show ?(relaxed = false)
    ?(input = [| 1.5; -2.; 3.25; 0.; 5.; -6.5; 7.; 8. |]) p =
  let reference = structured p input in
  let direct = c ~relaxed p input in
  let optimized =
    c ~relaxed (fst (Ssa_opt.run ~alias:Ssa_effects.Distinct_buffers p)) input
  in
  Fmt.pr
    "%s | %s | Wasm agrees with the interpreter: %b | optimized Wasm agrees: \
     %b@."
    direct.result
    (Fmt.str "%a" Fmt.(array ~sep:(any " ") pp_cell) direct.cells)
    (same reference direct) (same reference optimized)

let%expect_test "lane-wise arithmetic, conversion and iota" =
  show
    (program (fun bld ->
         let x = vload bld (idx bld 0) in
         let two = B.vec_splat bld ~lanes:(lanes 4) (f64v bld 2.) in
         let i = B.vec_iota bld ~lanes:(lanes 4) ~step:1L (idx bld 3) in
         let y = bin bld Expr.Value.Add (bin bld Expr.Value.Mul x two) i in
         let narrow = op1 bld (Ssa_op.Convert (Ssa_op.Convert.F64_to_f32, y)) in
         let wide =
           op1 bld (Ssa_op.Convert (Ssa_op.Convert.F32_to_f64, narrow))
         in
         vstore bld (idx bld 0) wide;
         (* a strided read: every second cell *)
         vstore bld (idx bld 4) (vload bld ~step:2L (idx bld 0))));
  (* iota with a stride and a negative stride: 10, 7, 4, 1 and 2, 5, 8, 11 *)
  show
    (program (fun bld ->
         vstore bld (idx bld 0)
           (B.vec_iota bld ~lanes:(lanes 4) ~step:(-3L) (idx bld 10));
         vstore bld (idx bld 4)
           (B.vec_iota bld ~lanes:(lanes 4) ~step:3L (idx bld 2))));
  [%expect
    {|
    ok | 6 0 11.5 6 1.5 3.25 5 7 | Wasm agrees with the interpreter: true | optimized Wasm agrees: true
    ok | 10 7 4 1 2 5 8 11 | Wasm agrees with the interpreter: true | optimized Wasm agrees: true |}]

let%expect_test "a mask selects between values both already computed" =
  show
    ~input:[| 1.5; -2.; nan; -0.; 5.; -6.5; infinity; 8. |]
    (program (fun bld ->
         let x = vload bld (idx bld 0) in
         let zero = B.vec_splat bld ~lanes:(lanes 4) (f64v bld 0.) in
         let negative =
           op1 bld (Ssa_op.Float_compare (Ssa_op.Compare.Lt, x, zero))
         in
         let flipped = bin bld Expr.Value.Sub zero x in
         vstore bld (idx bld 0) (op1 bld (Ssa_op.Select (negative, flipped, x)));
         (* mask algebra: not-or, and the pooled-argmax predicate *)
         let positive = op1 bld (Ssa_op.Pred_not negative) in
         let either = op1 bld (Ssa_op.Pred_or (negative, positive)) in
         let better = op1 bld (Ssa_op.Pool_better (zero, x)) in
         let both = op1 bld (Ssa_op.Pred_or (either, better)) in
         vstore bld (idx bld 4) (op1 bld (Ssa_op.Select (both, x, zero)))));
  [%expect
    {| ok | 1.5 2 nan -0 1.5 -2 nan -0 | Wasm agrees with the interpreter: true | optimized Wasm agrees: true |}]

let%expect_test "extract and insert address one lane" =
  show
    (program (fun bld ->
         let x = vload bld (idx bld 0) in
         let nine = f64v bld 9. in
         let y = B.vec_insert bld ~lane:(lane 2) x nine in
         vstore bld (idx bld 0) y;
         let e = B.vec_extract bld ~lane:(lane 1) y in
         let back = B.vec_splat bld ~lanes:(lanes 4) e in
         vstore bld (idx bld 4) back));
  [%expect
    {| ok | 1.5 -2 9 0 -2 -2 -2 -2 | Wasm agrees with the interpreter: true | optimized Wasm agrees: true |}]

let%expect_test "an ordered sum over vectors folds each lane on its own" =
  show
    (program (fun bld ->
         let seed = B.vec_splat bld ~lanes:(lanes 4) (f64v bld 0.) in
         let sum =
           B.ordered_sum_dyn bld ~lo:(idx bld 0) ~hi:(idx bld 3) ~seed
             (fun bld k ->
               B.mark_lanes bld Ssa_mark.Reduction ~lanes:(lanes 4);
               let x = vload bld k in
               bin bld Expr.Value.Mul x x)
         in
         vstore bld (idx bld 0) sum));
  [%expect
    {| ok | 16.8125 14.5625 35.5625 67.25 0 0 0 0 | Wasm agrees with the interpreter: true | optimized Wasm agrees: true |}]

let%expect_test "a loop carries a vector" =
  show
    (program (fun bld ->
         let zero = B.vec_splat bld ~lanes:(lanes 4) (f64v bld 0.) in
         match
           B.for_dyn bld ~lo:(idx bld 0) ~hi:(idx bld 3) ~init:[ zero ]
             (fun bld i carried ->
               let acc = List.hd carried in
               [ bin bld Expr.Value.Add acc (vload bld i) ])
         with
         | [ acc ] -> vstore bld (idx bld 0) acc
         | _ -> assert false));
  [%expect
    {| ok | 2.75 1.25 8.25 -1.5 0 0 0 0 | Wasm agrees with the interpreter: true | optimized Wasm agrees: true |}]

(* A binary32 vector rounds every lane after every operation, as the scalar
   operation does: the products and the running sum below differ from binary64's
   in the low bits, and the expectation is computed independently by rounding
   explicitly. *)
let%expect_test "binary32 lanes round per operation" =
  let input = [| 0.1; 0.2; 0.3; 1.0000001; 0.7; 0.9; 1.1; 1.3 |] in
  let p =
    program (fun bld ->
        let narrow bld v =
          op1 bld (Ssa_op.Convert (Ssa_op.Convert.F64_to_f32, v))
        in
        let widen bld v =
          op1 bld (Ssa_op.Convert (Ssa_op.Convert.F32_to_f64, v))
        in
        let seed = B.vec_splat bld ~lanes:(lanes 4) (raw (B.f32 bld 0.)) in
        let sum =
          B.ordered_sum_dyn bld ~lo:(idx bld 0) ~hi:(idx bld 3) ~seed
            (fun bld k ->
              let x = narrow bld (vload bld k) in
              bin bld Expr.Value.Mul x x)
        in
        vstore bld (idx bld 0) (widen bld sum))
  in
  let out = (execute p input).cells in
  let r = Ssa_const.round_f32 in
  let expected =
    Array.init 4 (fun j ->
        let acc = ref 0. in
        for k = 0 to 2 do
          let x = r input.(k + j) in
          acc := r (!acc +. r (x *. x))
        done;
        r !acc)
  in
  Fmt.pr "matches the hand-rounded fold: %b@."
    (Array.for_all2 Core.Float_bits.equal_portable (Array.sub out 0 4) expected);
  show ~input p;
  [%expect
    {|
    matches the hand-rounded fold: true
    ok | 0.14 1.13 1.58 2.3 0 0 0 0 | Wasm agrees with the interpreter: true | optimized Wasm agrees: true |}]

let%expect_test "lane-wise functions, fused multiply-add and binary32 vectors" =
  let input = [| 1.5; -2.; nan; -0.; 0.; 6.5; infinity; 8. |] in
  show ~input
    (program (fun bld ->
         let x = vload bld (idx bld 0) in
         let y = vload bld (idx bld 4) in
         vstore bld (idx bld 0) (op1 bld (Ssa_op.Float_max (x, y)));
         vstore bld (idx bld 4) (op1 bld (Ssa_op.Float_max (y, x)))));
  List.iter
    (fun uop ->
      show
        (program (fun bld ->
             let x = vload bld (idx bld 0) in
             vstore bld (idx bld 0) (op1 bld (Ssa_op.Float_unary (uop, x)));
             let narrow =
               op1 bld (Ssa_op.Convert (Ssa_op.Convert.F64_to_f32, x))
             in
             let r = op1 bld (Ssa_op.Float_unary (uop, narrow)) in
             vstore bld (idx bld 4)
               (op1 bld (Ssa_op.Convert (Ssa_op.Convert.F32_to_f64, r))))))
    [
      Expr.Value.Cos;
      Expr.Value.Erf;
      Expr.Value.Exp;
      Expr.Value.Log;
      Expr.Value.Sin;
      Expr.Value.Sqrt;
      Expr.Value.Trunc;
    ];
  (* a fused multiply-add is the relaxed-SIMD one, over binary32 lanes only *)
  show ~relaxed:true
    (program (fun bld ->
         let x = vload bld (idx bld 0) in
         let y = vload bld (idx bld 4) in
         let z = B.vec_splat bld ~lanes:(lanes 4) (f64v bld 0.1) in
         let n v = op1 bld (Ssa_op.Convert (Ssa_op.Convert.F64_to_f32, v)) in
         let w v = op1 bld (Ssa_op.Convert (Ssa_op.Convert.F32_to_f64, v)) in
         vstore bld (idx bld 4) (w (op1 bld (Ssa_op.Float_fma (n x, n y, n z))))));
  [%expect
    {|
    ok | 1.5 6.5 nan 8 1.5 6.5 nan 8 | Wasm agrees with the interpreter: true | optimized Wasm agrees: true
    ok | 0.0707372 -0.416147 -0.99413 1 0.0707372 -0.416147 -0.99413 1 | Wasm agrees with the interpreter: true | optimized Wasm agrees: true
    ok | 0.966105 -0.995322 0.999996 1e-09 0.966105 -0.995322 0.999996 0 | Wasm agrees with the interpreter: true | optimized Wasm agrees: true
    ok | 4.48169 0.135335 25.7903 1 4.48169 0.135335 25.7903 1 | Wasm agrees with the interpreter: true | optimized Wasm agrees: true
    ok | 0.405465 nan 1.17866 -inf 0.405465 nan 1.17866 -inf | Wasm agrees with the interpreter: true | optimized Wasm agrees: true
    ok | 0.997495 -0.909297 -0.108195 0 0.997495 -0.909297 -0.108195 0 | Wasm agrees with the interpreter: true | optimized Wasm agrees: true
    ok | 1.22474 nan 1.80278 0 1.22474 nan 1.80278 0 | Wasm agrees with the interpreter: true | optimized Wasm agrees: true
    ok | 1 -2 3 0 1 -2 3 0 | Wasm agrees with the interpreter: true | optimized Wasm agrees: true
    ok | 0 0 0 0 7.6 13.1 22.85 0.1 | Wasm agrees with the interpreter: true | optimized Wasm agrees: true |}]
