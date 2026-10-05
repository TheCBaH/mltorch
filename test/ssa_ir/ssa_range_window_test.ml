open Ssa_ir
open Ssa_fixtures
module B = Ssa_builder

(* A padded window: for each position [i] of a row, the taps [k] in
   [max (0, 1 - i), min (3, bound - i)) read [i - 1 + k]. The access is inside
   the row only relationally: [k >= 1 - i] and [k <= bound - i - 1] together
   bound [i - 1 + k] on both sides, which intervals over [i] and [k] separately
   cannot. With [bound = 9] on a row of 8 the check can never fire and is
   removed; with [bound = 10] the last position reads cell 8, and the check must
   stay and report it. *)

let width = 8

let bufs =
  [
    buffer 0 ~h:1L ~w:(Int64.of_int width) Ssa_format.F32 Ssa_buffer.Input;
    buffer 1 ~h:1L ~w:(Int64.of_int width) Ssa_format.F32 Ssa_buffer.Output;
  ]

let idx bld n = B.index bld (Int64.of_int n)

let window ~bound =
  build ~buffers:bufs (fun bld ->
      let zero = idx bld 0 and one = idx bld 1 in
      let B.Nil =
        B.for_ bld ~lo:zero ~hi:(idx bld width) ~init:B.Nil (fun bld i B.Nil ->
            let neg_i = B.index_scale bld (-1L) i in
            let lo = B.index_clamp_low bld (B.index_add bld one neg_i) in
            let hi =
              B.index_min bld (idx bld 3)
                (B.index_add bld (idx bld bound) neg_i)
            in
            let sum =
              B.ordered_sum bld ~lo ~hi ~seed:(B.f64 bld 0.) (fun bld k ->
                  let at_ =
                    B.index_add bld (B.index_add bld i (idx bld (-1))) k
                  in
                  B.load_f64 bld (buf 0) ~decode:Ssa_op.Decode.F32_to_f64
                    (at bld ~h:zero ~w:at_))
            in
            B.store_f64 bld (buf 1) ~encode:Ssa_op.Encode.F32_round
              (at bld ~h:zero ~w:i) sum;
            B.Nil)
      in
      ())

(* The same with a dilation of two and a stride of two: the taps start at
   [ceil ((1 - 2i) / 2)] and stop below [floor ((limit - 2i) / 2) + 1], and the
   access is [2i - 1 + 2k]. The division bounds give [2k >= 1 - 2i] and
   [2k <= limit - 2i] directly. [limit = 8] stays inside the row, [limit = 10]
   reads cell 9 for the last position. *)
let dilated ~limit =
  build ~buffers:bufs (fun bld ->
      let zero = idx bld 0 in
      let B.Nil =
        B.for_ bld ~lo:zero ~hi:(idx bld 4) ~init:B.Nil (fun bld i B.Nil ->
            let two_i = B.index_scale bld 2L i in
            let neg_two_i = B.index_scale bld (-2L) i in
            let lo =
              B.index_clamp_low bld
                (B.index_ceil_div bld 2L
                   (B.index_add bld (idx bld 1) neg_two_i))
            in
            let hi =
              B.index_min bld (idx bld 3)
                (B.index_add bld
                   (B.index_floor_div bld 2L
                      (B.index_add bld (idx bld limit) neg_two_i))
                   (idx bld 1))
            in
            let sum =
              B.ordered_sum bld ~lo ~hi ~seed:(B.f64 bld 0.) (fun bld k ->
                  let at_ =
                    B.index_add bld
                      (B.index_add bld two_i (idx bld (-1)))
                      (B.index_scale bld 2L k)
                  in
                  B.load_f64 bld (buf 0) ~decode:Ssa_op.Decode.F32_to_f64
                    (at bld ~h:zero ~w:at_))
            in
            B.store_f64 bld (buf 1) ~encode:Ssa_op.Encode.F32_round
              (at bld ~h:zero ~w:i) sum;
            B.Nil)
      in
      ())

(* Programs that read outside the row where a careless proof would say they do
   not. The first has a division bound with an offset, [k >= ceil (x / 2) - 1],
   which gives [2k >= x - 2], not [x - 1]: the access [2i - 1 + 2k] then reaches
   -1 at the first position. The second reads [3k + i + 1] against a fact on
   [2k]: a coefficient that is not a multiple of the fact's divisor takes no
   substitution, and the program fails at the second position. *)
let trap ~lo_of ~access_of =
  build ~buffers:bufs (fun bld ->
      let zero = idx bld 0 in
      let B.Nil =
        B.for_ bld ~lo:zero ~hi:(idx bld 4) ~init:B.Nil (fun bld i B.Nil ->
            let lo = lo_of bld i and hi = idx bld 1 in
            let sum =
              B.ordered_sum bld ~lo ~hi ~seed:(B.f64 bld 0.) (fun bld k ->
                  B.load_f64 bld (buf 0) ~decode:Ssa_op.Decode.F32_to_f64
                    (at bld ~h:zero ~w:(access_of bld i k)))
            in
            B.store_f64 bld (buf 1) ~encode:Ssa_op.Encode.F32_round
              (at bld ~h:zero ~w:i) sum;
            B.Nil)
      in
      ())

let offset_division =
  trap
    ~lo_of:(fun bld i ->
      B.index_clamp_low bld
        (B.index_add bld
           (B.index_ceil_div bld 2L
              (B.index_add bld (idx bld 2) (B.index_scale bld (-2L) i)))
           (idx bld (-1))))
    ~access_of:(fun bld i k ->
      B.index_add bld
        (B.index_add bld (B.index_scale bld 2L i) (idx bld (-1)))
        (B.index_scale bld 2L k))

let indivisible_coefficient =
  trap
    ~lo_of:(fun bld i ->
      B.index_ceil_div bld 2L
        (B.index_add bld (B.index_scale bld (-1L) i) (idx bld (-1))))
    ~access_of:(fun bld i k ->
      B.index_add bld (B.index_add bld (B.index_scale bld 3L k) i) (idx bld 1))

let optimize p =
  fst
    (Ssa_opt.run ~alias:Ssa_effects.Distinct_buffers
       ~passes:Ssa_opt.[ simplify; guards; simplify ]
       p)

let run_on p =
  let input = Array.init width (fun i -> float_of_int (i + 1)) in
  let out = Array.make width 0. in
  let memory = memory [ (0, floats input); (1, floats out) ] in
  let result = run_result p ~memory in
  (Result.map_error (fun e -> Fmt.str "%a" Ssa_interp.pp_error e) result, out)

let report name p =
  let q = optimize p in
  let r, out = run_on p and r', out' = run_on q in
  Fmt.pr "%s: checked operations %d -> %d; same outcome: %b; %s@." name
    (Ssa_stats.of_program p).Ssa_stats.checked
    (Ssa_stats.of_program q).Ssa_stats.checked
    (r = r' && Array.for_all2 Core.Float_bits.equal_portable out out')
    (match r' with Ok () -> "ok" | Error e -> e)

let%expect_test "a padded window's check goes only where the bounds prove it" =
  report "bound 9 (inside)" (window ~bound:9);
  report "bound 10 (reads cell 8)" (window ~bound:10);
  report "dilated, limit 8 (inside)" (dilated ~limit:8);
  report "dilated, limit 10 (reads cell 9)" (dilated ~limit:10);
  report "offset division bound" offset_division;
  report "coefficient not a multiple of the divisor" indivisible_coefficient;
  [%expect
    {|
    bound 9 (inside): checked operations 6 -> 0; same outcome: true; ok
    bound 10 (reads cell 8): checked operations 6 -> 1; same outcome: true; t0: coordinate W = 8 is outside the buffer, at (0,0,0,0,8,0)
    dilated, limit 8 (inside): checked operations 9 -> 0; same outcome: true; ok
    dilated, limit 10 (reads cell 9): checked operations 9 -> 1; same outcome: true; t0: coordinate W = 9 is outside the buffer, at (0,0,0,0,9,0)
    offset division bound: checked operations 8 -> 1; same outcome: true; t0: coordinate W = -1 is outside the buffer, at (0,0,0,0,-1,0)
    coefficient not a multiple of the divisor: checked operations 6 -> 1; same outcome: true; t0: coordinate W = -1 is outside the buffer, at (0,0,0,0,-1,0) |}]
