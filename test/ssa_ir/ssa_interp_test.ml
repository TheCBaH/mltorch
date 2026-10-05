open Ssa_ir
open Ssa_fixtures

let%expect_test "matmul matches an independently computed sum" =
  let m = 3 and k = 5 and n = 4 in
  let p = matmul ~m ~k ~n in
  let a =
    Array.init (m * k) (fun i ->
        Ssa_const.round_f32 (0.1 *. float_of_int (i + 1)))
  in
  let b =
    Array.init (k * n) (fun i ->
        Ssa_const.round_f32 (1.5 -. (0.07 *. float_of_int i)))
  in
  let out = Array.make (m * n) 0. in
  let memory = memory [ (0, floats a); (1, floats b); (2, floats out) ] in
  let counters = Ssa_interp.Counters.create () in
  run ~counters p ~memory;
  let expected = matmul_expected ~m ~k ~n a b in
  Fmt.pr "equal=%b reductions=%d loads=%d stores=%d@."
    (Array.for_all2 Core.Float_bits.equal_portable out expected)
    (Ssa_interp.Counters.mark counters Ssa_mark.Reduction)
    (Ssa_interp.Counters.loads counters)
    (Ssa_interp.Counters.stores counters);
  [%expect {| equal=true reductions=60 loads=120 stores=12 |}]

module B = Ssa_builder

let row_buffer id n format role = buffer id ~h:1L ~w:n format role
let idx bld n = B.index bld (Int64.of_int n)

let load_at bld id i =
  B.load_f64 bld (buf id) ~decode:Ssa_op.Decode.F32_to_f64
    (at bld ~h:(idx bld 0) ~w:i)

let store_at bld id i x =
  B.store_f64 bld (buf id) ~encode:Ssa_op.Encode.F32_round
    (at bld ~h:(idx bld 0) ~w:i)
    x

let show_failure = function
  | Ok () -> Fmt.pr "ok@."
  | Error e -> Fmt.pr "%a@." Ssa_interp.pp_error e

(* A program is run on one output buffer of [n] cells and an input buffer 0. *)
let run_row ?(input = [||]) p =
  let out = Array.make 8 0. in
  let memory = memory [ (0, floats input); (1, floats out) ] in
  (run_result p ~memory, out)

let%expect_test "carried values transfer simultaneously" =
  (* (a, b) <- (b, a + b), ten times: the Fibonacci pair, written to the output.
     A sequential rebind would read the already-overwritten [a]. *)
  let bufs =
    [
      row_buffer 0 8L Ssa_format.F32 Ssa_buffer.Input;
      row_buffer 1 8L Ssa_format.F32 Ssa_buffer.Output;
    ]
  in
  let p =
    build ~buffers:bufs (fun bld ->
        let a0 = B.f64 bld 0. and b0 = B.f64 bld 1. in
        let (B.Cons (a, B.Cons (b, B.Nil))) =
          B.for_ bld ~lo:(idx bld 0) ~hi:(idx bld 10)
            ~init:(B.Cons (a0, B.Cons (b0, B.Nil)))
            (fun bld _ (B.Cons (a, B.Cons (b, B.Nil))) ->
              B.Cons (b, B.Cons (B.f64_binary bld Expr.Value.Add a b, B.Nil)))
        in
        store_at bld 1 (idx bld 0) a;
        store_at bld 1 (idx bld 1) b)
  in
  let result, out = run_row p in
  show_failure result;
  Fmt.pr "%g %g@." out.(0) out.(1);
  (* the swap alone, an odd number of times *)
  let swap =
    build ~buffers:bufs (fun bld ->
        let a0 = B.f64 bld 1. and b0 = B.f64 bld 2. in
        let (B.Cons (a, B.Cons (b, B.Nil))) =
          B.for_ bld ~lo:(idx bld 0) ~hi:(idx bld 5)
            ~init:(B.Cons (a0, B.Cons (b0, B.Nil)))
            (fun _ _ (B.Cons (a, B.Cons (b, B.Nil))) ->
              B.Cons (b, B.Cons (a, B.Nil)))
        in
        store_at bld 1 (idx bld 0) a;
        store_at bld 1 (idx bld 1) b)
  in
  let result, out = run_row swap in
  show_failure result;
  Fmt.pr "%g %g@." out.(0) out.(1);
  [%expect {|
    ok
    55 89
    ok
    2 1
    |}]

let%expect_test "a zero-trip loop and an empty sum never run their body" =
  let bufs =
    [
      row_buffer 0 4L Ssa_format.F32 Ssa_buffer.Input;
      row_buffer 1 8L Ssa_format.F32 Ssa_buffer.Output;
    ]
  in
  (* the body reads far outside the input: only an executed iteration fails *)
  let program ~lo ~hi =
    build ~buffers:bufs (fun bld ->
        let seed = B.f64 bld 7. in
        let sum =
          B.ordered_sum bld ~lo:(idx bld lo) ~hi:(idx bld hi) ~seed
            (fun bld _ ->
              B.mark bld Ssa_mark.Reduction;
              load_at bld 0 (idx bld 1000))
        in
        let (B.Cons (carried, B.Nil)) =
          B.for_ bld ~lo:(idx bld lo) ~hi:(idx bld hi)
            ~init:(B.Cons (B.f64 bld 3., B.Nil))
            (fun bld _ (B.Cons (x, B.Nil)) ->
              B.Cons
                ( B.f64_binary bld Expr.Value.Add x
                    (load_at bld 0 (idx bld 1000)),
                  B.Nil ))
        in
        store_at bld 1 (idx bld 0) sum;
        store_at bld 1 (idx bld 1) carried)
  in
  List.iter
    (fun (lo, hi) ->
      let counters = Ssa_interp.Counters.create () in
      let out = Array.make 8 0. in
      let memory = memory [ (0, floats (Array.make 4 0.)); (1, floats out) ] in
      show_failure (run_result ~counters (program ~lo ~hi) ~memory);
      Fmt.pr "[%d,%d) sum=%g carried=%g marks=%d@." lo hi out.(0) out.(1)
        (Ssa_interp.Counters.mark counters Ssa_mark.Reduction))
    [ (0, 0); (5, 2) ];
  show_failure (fst (run_row ~input:(Array.make 4 0.) (program ~lo:0 ~hi:1)));
  [%expect
    {|
    ok
    [0,0) sum=7 carried=3 marks=0
    ok
    [5,2) sum=7 carried=3 marks=0
    t0: coordinate W = 1000 is outside the buffer, at (0,0,0,0,1000,0)
    |}]

let%expect_test "only the selected branch runs, and the first failure leaves" =
  let bufs =
    [
      row_buffer 0 4L Ssa_format.F32 Ssa_buffer.Input;
      row_buffer 1 8L Ssa_format.F32 Ssa_buffer.Output;
    ]
  in
  let program cond =
    build ~buffers:bufs (fun bld ->
        let c = B.pred bld cond in
        let (B.Cons (x, B.Nil)) =
          B.if_ bld c
            ~then_:(fun bld -> B.Cons (load_at bld 0 (idx bld 100), B.Nil))
            ~else_:(fun bld -> B.Cons (B.f64 bld 5., B.Nil))
        in
        store_at bld 1 (idx bld 0) x)
  in
  let attempt cond =
    let result, out = run_row ~input:(Array.make 4 1.) (program cond) in
    show_failure result;
    Fmt.pr "out0=%g@." out.(0)
  in
  (* the untaken branch would fail; the taken one does *)
  attempt false;
  attempt true;
  [%expect
    {|
    ok
    out0=5
    t0: coordinate W = 100 is outside the buffer, at (0,0,0,0,100,0)
    out0=0
    |}]

let%expect_test
    "failure rows: the first axis outside, and checked index operations" =
  let bufs =
    [
      row_buffer 0 4L Ssa_format.F32 Ssa_buffer.Input;
      row_buffer 1 8L Ssa_format.F32 Ssa_buffer.Output;
    ]
  in
  let attempt f =
    let p = build ~buffers:bufs f in
    show_failure (fst (run_row ~input:(Array.make 4 1.) p))
  in
  (* H and W are both outside: the axis order of Expr.Axis.all reports H *)
  attempt (fun bld ->
      let x =
        B.load_f64 bld (buf 0) ~decode:Ssa_op.Decode.F32_to_f64
          (at bld ~h:(idx bld 3) ~w:(idx bld 9))
      in
      store_at bld 1 (idx bld 0) x);
  (* a negative component is outside too *)
  attempt (fun bld ->
      let x =
        B.load_f64 bld (buf 0) ~decode:Ssa_op.Decode.F32_to_f64
          (at bld ~h:(idx bld 0) ~w:(B.index bld (-1L)))
      in
      store_at bld 1 (idx bld 0) x);
  (* the first operation to leave the domain fails, though a later one returns *)
  attempt (fun bld ->
      let top = B.index bld 0x7FFF_FFFFL in
      let over = B.index_add bld top (idx bld 1) in
      let back = B.index_add bld over (B.index bld (-1L)) in
      store_at bld 1 back (B.f64 bld 1.));
  attempt (fun bld ->
      let scaled = B.index_scale bld 2L (B.index bld 0x4000_0000L) in
      store_at bld 1 scaled (B.f64 bld 1.));
  attempt (fun bld ->
      let low =
        B.index_add bld (B.index bld (-0x8000_0000L)) (B.index bld (-1L))
      in
      store_at bld 1 low (B.f64 bld 1.));
  [%expect
    {|
    t0: coordinate H = 3 is outside the buffer, at (0,0,0,3,9,0)
    t0: coordinate W = -1 is outside the buffer, at (0,0,0,0,-1,0)
    index add overflows on 2147483647 and 1
    index mul overflows on 2 and 1073741824
    index add overflows on -2147483648 and -1
    |}]

let%expect_test "rounding, conversion and order are part of the value" =
  let bufs =
    [
      row_buffer 0 4L Ssa_format.F32 Ssa_buffer.Input;
      row_buffer 1 8L Ssa_format.F32 Ssa_buffer.Output;
      row_buffer 2 4L Ssa_format.Bool Ssa_buffer.Output;
    ]
  in
  let p =
    build ~buffers:bufs (fun bld ->
        (* store rounds to binary32 *)
        store_at bld 1 (idx bld 0) (B.f64 bld 0.1);
        (* a left fold: 1e16 + 1 - 1e16 is 0, a regrouped sum would give 1 *)
        let sum =
          B.ordered_sum bld ~lo:(idx bld 0) ~hi:(idx bld 3) ~seed:(B.f64 bld 0.)
            (fun bld i -> load_at bld 0 i)
        in
        store_at bld 1 (idx bld 1) sum;
        (* f64 -> f32 -> f64 keeps the rounded value *)
        let r = B.f32_to_f64 bld (B.f64_to_f32 bld (B.f64 bld 16777217.)) in
        store_at bld 1 (idx bld 2) r;
        (* a Bool store writes a canonical byte, NaN and -0 included *)
        List.iteri
          (fun k x ->
            B.store_f64 bld (buf 2) ~encode:Ssa_op.Encode.Bool_nonzero
              (at bld ~h:(idx bld 0) ~w:(idx bld k))
              (B.f64 bld x))
          [ nan; -0.; 2.5; 0. ])
  in
  let out = Array.make 8 0. and bits = Array.make 4 9. in
  let memory =
    memory
      [
        (0, floats [| 1e16; 1.; -1e16; 0. |]); (1, floats out); (2, floats bits);
      ]
  in
  show_failure (run_result p ~memory);
  Fmt.pr "round_f32(0.1) stored: %b@."
    (out.(0) = Ssa_const.round_f32 0.1 && out.(0) <> 0.1);
  Fmt.pr "sum=%g@." out.(1);
  Fmt.pr "16777217 -> %g@." out.(2);
  Array.iter (Fmt.pr "%g ") bits;
  Fmt.pr "@.";
  [%expect
    {|
    ok
    round_f32(0.1) stored: true
    sum=0
    16777217 -> 1.67772e+07
    1 0 1 0
    |}]

(* [depth] nested regions, each guarding the next: the verifier bounds it. *)
let nested ~depth =
  let bufs = [ row_buffer 1 8L Ssa_format.F32 Ssa_buffer.Output ] in
  Err.payload
    (B.program ~buffers:bufs (fun bld ->
         let rec go bld n =
           if n = 0 then store_at bld 1 (idx bld 0) (B.f64 bld 42.)
           else
             let B.Nil =
               B.if_ bld (B.pred bld true)
                 ~then_:(fun bld ->
                   go bld (n - 1);
                   B.Nil)
                 ~else_:(fun _ -> B.Nil)
             in
             ()
         in
         go bld depth))

let%expect_test "deep nesting is bounded by the verifier and runs inside it" =
  (match nested ~depth:200 with
  | Ok p ->
      let out = Array.make 8 0. in
      let memory = memory [ (1, floats out) ] in
      show_failure (run_result p ~memory);
      Fmt.pr "out0=%g@." out.(0)
  | Error e -> Fmt.pr "%a@." Ssa_verify.pp_error e);
  (match nested ~depth:300 with
  | Ok _ -> Fmt.pr "accepted@."
  | Error e -> Fmt.pr "%a@." Ssa_verify.pp_error e);
  [%expect {|
    ok
    out0=42
    r88, stmt1: region nesting is too deep |}]

(* ---- int64 ------------------------------------------------------------------ *)

(* The binary32 nearest to an int64, ties to even, from the integer alone: the
   magnitude's top 24 bits, then the discarded bits against a half. *)
let nearest_f32 n =
  let neg = Int64.compare n 0L < 0 in
  if Int64.equal n 0L then 0.
  else
    (* min_int's magnitude is 2^63, which is a power of two and exact *)
    let mag = if neg then Int64.neg n else n in
    let bits = ref 0 in
    (let m = ref mag in
     while not (Int64.equal !m 0L) do
       incr bits;
       m := Int64.shift_right_logical !m 1
     done);
    let value =
      if !bits <= 24 then Int64.to_float mag
      else
        let drop = !bits - 24 in
        let q = Int64.shift_right_logical mag drop in
        let rem = Int64.logand mag (Int64.pred (Int64.shift_left 1L drop)) in
        let half = Int64.shift_left 1L (drop - 1) in
        let c = Int64.unsigned_compare rem half in
        let q =
          if c > 0 || (c = 0 && Int64.equal (Int64.logand q 1L) 1L) then
            Int64.succ q
          else q
        in
        Int64.to_float q *. Float.pow 2. (float_of_int drop)
    in
    if neg then -.value else value

let%expect_test
    "an int64 converts to binary32 with one rounding, in the interpreter" =
  let samples =
    [|
      16777217L;
      0x0020000020000001L;
      0x0020000020000000L;
      0x7FFFFF7FFFFFFFFFL;
      Int64.add (Int64.shift_left 1L 53) 1L;
      Int64.max_int;
      Int64.min_int;
      -16777217L;
      Int64.neg 0x0020000020000001L;
      3L;
      0L;
      Int64.add (Int64.shift_left 1L 62) 1L;
    |]
  in
  let n = Array.length samples in
  let bufs =
    [ row_buffer 1 (Int64.of_int n) Ssa_format.F32 Ssa_buffer.Output ]
  in
  let p =
    build ~buffers:bufs (fun bld ->
        Array.iteri
          (fun k x ->
            let f = B.f32_to_f64 bld (B.i64_to_f32 bld (B.i64 bld x)) in
            store_at bld 1 (idx bld k) f)
          samples)
  in
  let out = Array.make n 0. in
  let memory = memory [ (1, floats out) ] in
  show_failure (run_result p ~memory);
  let wrong =
    List.filter
      (fun k ->
        not (Core.Float_bits.equal_portable out.(k) (nearest_f32 samples.(k))))
      (List.init n Fun.id)
  in
  Fmt.pr "disagreements with the integer rounding: %d@." (List.length wrong);
  (* a conversion through binary64 would round twice on the second sample *)
  Fmt.pr "binary64 first would differ: %b@."
    (not
       (Core.Float_bits.equal_portable
          (Ssa_const.round_f32 (Int64.to_float samples.(1)))
          (nearest_f32 samples.(1))));
  [%expect
    {|
    ok
    disagreements with the integer rounding: 0
    binary64 first would differ: true |}]

let%expect_test "narrowing an int64 outside the index domain is a defect" =
  let bufs = [ row_buffer 1 4L Ssa_format.F32 Ssa_buffer.Output ] in
  let p =
    build ~buffers:bufs (fun bld ->
        let narrowed = B.index_of_i64 bld (B.i64 bld 0x1_0000_0000L) in
        store_at bld 1 narrowed (B.f64 bld 1.))
  in
  let memory = memory [ (1, floats (Array.make 4 0.)) ] in
  (match run_result p ~memory with
  | exception Invalid_argument _ -> Fmt.pr "defect@."
  | Ok () -> Fmt.pr "no failure@."
  | Error e -> Fmt.pr "row: %a@." Ssa_interp.pp_error e);
  [%expect {| defect |}]
