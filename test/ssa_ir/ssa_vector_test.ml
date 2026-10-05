open Ssa_ir
open Ssa_fixtures
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

type outcome = {
  result : string;
  cells : float array;
  loads : int;
  stores : int;
  marks : int list;
}

let execute p input =
  let out = Array.make 8 0. in
  let counters = Ssa_interp.Counters.create () in
  let memory = memory [ (0, floats (Array.copy input)); (1, floats out) ] in
  let result =
    match run_result ~counters p ~memory with
    | Ok () -> "ok"
    | Error e -> Fmt.str "%a" Ssa_interp.pp_error e
  in
  {
    result;
    cells = out;
    loads = Ssa_interp.Counters.loads counters;
    stores = Ssa_interp.Counters.stores counters;
    marks = List.map (Ssa_interp.Counters.mark counters) Ssa_mark.all;
  }

let same a b =
  a.result = b.result && a.loads = b.loads && a.stores = b.stores
  && a.marks = b.marks
  && Array.for_all2 Core.Float_bits.equal_portable a.cells b.cells

let show ?(input = [| 1.5; -2.; 3.25; 0.; 5.; -6.5; 7.; 8. |]) p =
  let direct = execute p input in
  let expanded = execute (Ssa_vec_expand.program p) input in
  let optimized =
    execute (fst (Ssa_opt.run ~alias:Ssa_effects.Distinct_buffers p)) input
  in
  Fmt.pr
    "%s | %s | loads %d stores %d reductions %d | expansion agrees: %b | \
     optimized agrees: %b@."
    direct.result
    (String.concat " " (Array.to_list (Array.map (Fmt.str "%g") direct.cells)))
    direct.loads direct.stores (List.nth direct.marks 3) (same direct expanded)
    (same direct optimized)

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
    ok | 6 0 11.5 6 1.5 3.25 5 7 | loads 8 stores 8 reductions 0 | expansion agrees: true | optimized agrees: true
    ok | 10 7 4 1 2 5 8 11 | loads 0 stores 8 reductions 0 | expansion agrees: true | optimized agrees: true |}]

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
    {| ok | 1.5 2 nan -0 1.5 -2 nan -0 | loads 4 stores 8 reductions 0 | expansion agrees: true | optimized agrees: true |}]

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
    {| ok | 1.5 -2 9 0 -2 -2 -2 -2 | loads 4 stores 8 reductions 0 | expansion agrees: true | optimized agrees: true |}]

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
    {| ok | 16.8125 14.5625 35.5625 67.25 0 0 0 0 | loads 12 stores 4 reductions 12 | expansion agrees: true | optimized agrees: true |}]

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
    {| ok | 2.75 1.25 8.25 -1.5 0 0 0 0 | loads 12 stores 4 reductions 0 | expansion agrees: true | optimized agrees: true |}]

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
    ok | 0.14 1.13 1.58 2.3 0 0 0 0 | loads 12 stores 4 reductions 0 | expansion agrees: true | optimized agrees: true |}]

(* ---- what the verifier refuses ----------------------------------------------- *)

let refused name build =
  match build () with
  | exception Invalid_argument m -> Fmt.pr "%s: %s@." name m
  | Ok _ -> Fmt.pr "%s: accepted@." name
  | Error e -> Fmt.pr "%s: %a@." name Ssa_verify.pp_error e

let attempt f = fun () -> Err.payload (B.program ~buffers:bufs f)

let%expect_test "malformed vector programs are refused where they are built" =
  refused "lane counts differ"
    (attempt (fun bld ->
         let a = vload bld (idx bld 0) in
         let b = vload bld ~n:2 (idx bld 0) in
         ignore (bin bld Expr.Value.Add a b)));
  refused "a scalar operand"
    (attempt (fun bld ->
         let a = vload bld (idx bld 0) in
         ignore (bin bld Expr.Value.Add a (f64v bld 1.))));
  refused "lane outside the vector"
    (attempt (fun bld ->
         ignore (B.vec_extract bld ~lane:(lane 4) (vload bld (idx bld 0)))));
  refused "wider than the IR names"
    (attempt (fun bld ->
         ignore (B.vec_splat bld ~lanes:(lanes 65) (f64v bld 0.))));
  refused "an int64 vector"
    (attempt (fun bld ->
         ignore (B.vec_splat bld ~lanes:(lanes 4) (raw (B.i64 bld 1L)))));
  refused "an operation that cannot be lifted"
    (attempt (fun bld ->
         let a = vload bld (idx bld 0) in
         ignore (op1 bld (Ssa_op.Float_to_i64 a))));
  refused "a step outside the index domain"
    (attempt (fun bld -> ignore (vload bld ~step:0x1_0000_0000L (idx bld 0))));
  [%expect
    {|
    lane counts differ: Ssa_builder: lane counts differ: x4 and x2
    a scalar operand: Ssa_builder: operand1: expected a vector or mask, found f64
    lane outside the vector: Ssa_builder: lane4 is outside a vector of x4 lanes
    wider than the IR names: Ssa_builder: x65 lanes is outside 1..64
    an int64 vector: Ssa_builder: a vector holds floats and masks, never int64
    an operation that cannot be lifted: Ssa_builder: float.to_i64 cannot be applied lane by lane
    a step outside the index domain: Ssa_builder: scale literal 4294967296 is outside the index domain |}]

let%expect_test "every lane must be provably inside the buffer" =
  (* the last lane reads cell 8 of 8: a mask could hide it, so it is refused *)
  refused "last lane past the end"
    (attempt (fun bld ->
         let x = vload bld (idx bld 5) in
         let zero = B.vec_splat bld ~lanes:(lanes 4) (f64v bld 0.) in
         let keep =
           op1 bld (Ssa_op.Float_compare (Ssa_op.Compare.Lt, zero, zero))
         in
         vstore bld (idx bld 0) (op1 bld (Ssa_op.Select (keep, x, zero)))));
  refused "a negative step walking off the front"
    (attempt (fun bld -> ignore (vload bld ~step:(-1L) (idx bld 2))));
  refused "a data-dependent start"
    (attempt (fun bld ->
         let start =
           B.index_of_i64 bld
             (B.float_to_i64 bld
                (B.load_f64 bld (buf 0) ~decode:Ssa_op.Decode.F32_to_f64
                   (at bld ~h:(idx bld 0) ~w:(idx bld 0))))
         in
         ignore (vload bld start)));
  refused "inside, edge to edge"
    (attempt (fun bld -> ignore (vload bld ~n:8 (idx bld 0))));
  refused "last lane stored past the end"
    (attempt (fun bld -> vstore bld (idx bld 5) (vload bld (idx bld 0))));
  refused "a store to an input"
    (attempt (fun bld ->
         B.vec_store bld (buf 0) ~encode:Ssa_op.Encode.F32_round
           ~lanes:(lanes 4) ~steps:(steps_w 1L)
           (at_w bld (idx bld 0))
           (vload bld (idx bld 0))));
  [%expect
    {|
    last lane past the end: r0, stmt2: a proof of every lane of a vector access staying in its buffer cannot be re-derived
    a negative step walking off the front: r0, stmt2: a proof of every lane of a vector access staying in its buffer cannot be re-derived
    a data-dependent start: r0, stmt7: a proof of every lane of a vector access staying in its buffer cannot be re-derived
    inside, edge to edge: accepted
    last lane stored past the end: r0, stmt5: a proof of every lane of a vector access staying in its buffer cannot be re-derived
    a store to an input: r0, stmt5: b0 is an input and is never written |}]
