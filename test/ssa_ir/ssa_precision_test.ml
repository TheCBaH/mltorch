open Ssa_ir
open Ssa_fixtures
module B = Ssa_builder

(* The rewrite to binary32, on programs small enough to compute by hand. *)

let bufs =
  [
    buffer 0 ~h:1L ~w:8L Ssa_format.F64 Ssa_buffer.Input;
    buffer 1 ~h:1L ~w:8L Ssa_format.F32 Ssa_buffer.Output;
  ]

let idx bld n = B.index bld (Int64.of_int n)

let load_at bld i =
  B.load_f64 bld (buf 0) ~decode:Ssa_op.Decode.F64_to_f64
    (at bld ~h:(idx bld 0) ~w:i)

(* Stored as binary32: a binary32 working value is stored as itself. *)
let store_at bld i x =
  B.store_f64 bld (buf 1) ~encode:Ssa_op.Encode.F32_round
    (at bld ~h:(idx bld 0) ~w:i)
    x

let outputs ?(input = [||]) p =
  let out = Array.make 8 0. in
  let memory = memory [ (0, floats (Array.copy input)); (1, floats out) ] in
  match run_result p ~memory with
  | Ok () -> out
  | Error e -> Fmt.failwith "%a" Ssa_interp.pp_error e

let verified p =
  match Err.payload (Ssa_verify.check p) with Ok () -> true | Error _ -> false

let%expect_test "an int64 reaches binary32 in one rounding" =
  (* 2^53 + 2^29 + 1 is just above the binary32 midpoint 2^53 + 2^29. Through
     binary64 it first rounds to the midpoint itself, then to even, 2^53; the
     correct single rounding is 2^53 + 2^30. *)
  let n =
    Int64.add (Int64.shift_left 1L 53) (Int64.add (Int64.shift_left 1L 29) 1L)
  in
  let p =
    build ~buffers:bufs (fun bld ->
        store_at bld (idx bld 0) (B.i64_to_f64 bld (B.i64 bld n)))
  in
  let q = Ssa_precision.to_f32 p in
  Fmt.pr "verifies: %b@." (verified q);
  Fmt.pr "through binary64: %.0f@." (outputs p).(0);
  Fmt.pr "in binary32:      %.0f (expected %.0f)@."
    (outputs q).(0)
    (Ssa_const.round32_of_i64 n);
  [%expect
    {|
    verifies: true
    through binary64: 9007199254740992
    in binary32:      9007200328482816 (expected 9007200328482816) |}]

let%expect_test "constants, loads, sums and stores round once and stay in order"
    =
  let input = [| 0.1; 0.2; 0.3; 1e-3; 1.5; 2.25; 3.; 4. |] in
  let p =
    build ~buffers:bufs (fun bld ->
        let seed = B.f64 bld 0.1 in
        let sum =
          B.ordered_sum bld ~lo:(idx bld 0) ~hi:(idx bld 4) ~seed (fun bld k ->
              B.f64_binary bld Expr.Value.Mul (load_at bld k) (B.f64 bld 3.3))
        in
        store_at bld (idx bld 0) sum;
        store_at bld (idx bld 1)
          (B.f64_unary bld Expr.Value.Sqrt (load_at bld (idx bld 2)));
        store_at bld (idx bld 2) (B.index_to_f64 bld (idx bld 16777217)))
  in
  let q = Ssa_precision.to_f32 p in
  let r = Ssa_const.round_f32 in
  let acc = ref (r 0.1) in
  for k = 0 to 3 do
    acc := r (!acc +. r (r input.(k) *. r 3.3))
  done;
  let got = outputs ~input q in
  Fmt.pr "verifies: %b@." (verified q);
  Fmt.pr "sum matches the hand-rounded fold: %b@."
    (Core.Float_bits.equal_portable got.(0) !acc);
  Fmt.pr "sqrt rounded once: %b@."
    (Core.Float_bits.equal_portable got.(1) (r (sqrt (r input.(2)))));
  Fmt.pr "index 16777217 -> %.0f@." got.(2);
  [%expect
    {|
    verifies: true
    sum matches the hand-rounded fold: true
    sqrt rounded once: true
    index 16777217 -> 16777216 |}]

let%expect_test "a checked float-to-int64 conversion widens its operand" =
  let p =
    build ~buffers:bufs (fun bld ->
        let n =
          B.float_to_i64 bld
            (B.f64_binary bld Expr.Value.Mul
               (load_at bld (idx bld 0))
               (B.f64 bld 1000.))
        in
        store_at bld (idx bld 0) (B.i64_to_f64 bld n))
  in
  let q = Ssa_precision.to_f32 p in
  Fmt.pr "verifies: %b@." (verified q);
  Fmt.pr "%.0f@." (outputs ~input:[| 2.7182818 |] q).(0);
  [%expect {|
    verifies: true
    2718 |}]

let%expect_test "vectors and unadmitted payloads are refused" =
  let refuse f =
    try
      ignore (f ());
      Fmt.pr "accepted@."
    with Invalid_argument m -> Fmt.pr "%s@." m
  in
  refuse (fun () ->
      Ssa_precision.to_f32
        (build ~buffers:bufs (fun bld ->
             ignore
               (B.vec_splat bld ~lanes:(Ssa_type.Lanes.of_int 4)
                  (B.f64 bld 1. :> Ssa_value.t)))));
  refuse (fun () ->
      let quantized =
        {
          (buffer 0 ~h:1L ~w:8L
             (Ssa_format.I8
                (Ssa_format.Per_tensor { scale = 0.5; zero_point = 0 }))
             Ssa_buffer.Input)
          with
          Ssa_buffer.id = buf 0;
        }
      in
      Ssa_precision.to_f32
        (build
           ~buffers:[ quantized; List.nth bufs 1 ]
           (fun bld ->
             store_at bld (idx bld 0)
               (B.load_f64 bld (buf 0) ~decode:Ssa_op.Decode.I8_dequant
                  (at bld ~h:(idx bld 0) ~w:(idx bld 0))))));
  [%expect
    {|
    Ssa_precision: choose the precision before vectorizing
    Ssa_precision: b0: float read of a i8 payload |}]
