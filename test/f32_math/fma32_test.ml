(* [F32_math.fma32] against an independent oracle, natively and under
   js_of_ocaml. The oracle decides the rounding of the exact [s + e] (TwoSum
   of the exact binary64 product and the addend) by the one place it can
   differ from rounding [s]: [s] exactly a binary32 midpoint, where [e]'s sign
   picks the neighbour. A different argument from round to odd, so agreement
   is evidence. The hand-built triples are binary64 double-rounding cases:
   [round32 (Float.fma a b c)] is wrong on them. *)

let round32 = F32_math.round32

(* The binary32 neighbours of a binary64 [s] that is not binary32: the one
   below and the one above, an overflow's upper neighbour as 2^128. *)
let neighbours s =
  let r = round32 s in
  let step r up =
    let bits = Int32.bits_of_float r in
    (* toward larger magnitude when [up] has the sign of [r] *)
    let larger = up = (r > 0. || (r = 0. && up)) in
    if r = 0. then if up then ldexp 1. (-149) else -.ldexp 1. (-149)
    else if Float.is_integer r && Float.abs r = Float.infinity then r
    else
      Int32.float_of_bits (if larger then Int32.succ bits else Int32.pred bits)
  in
  let lo, hi = if r < s then (r, step r true) else (step r false, r) in
  let fix x =
    if x = Float.infinity then ldexp 1. 128
    else if x = Float.neg_infinity then -.ldexp 1. 128
    else x
  in
  (fix lo, fix hi)

let oracle a b c =
  if not (Float.is_finite a && Float.is_finite b && Float.is_finite c) then
    round32 ((a *. b) +. c)
  else
    let p = a *. b in
    let s = p +. c in
    let bb = s -. p in
    let e = p -. (s -. bb) +. (c -. bb) in
    if round32 s = s || e = 0. then round32 s
    else
      let lo, hi = neighbours s in
      let mid = (lo /. 2.) +. (hi /. 2.) in
      let pick x =
        if Float.abs x = ldexp 1. 128 then Float.copy_sign Float.infinity x
        else x
      in
      if s = mid then pick (if e > 0. then hi else lo) else round32 s

let same x y = Int64.equal (Int64.bits_of_float x) (Int64.bits_of_float y)
let double_rounded a b c = round32 (Float.fma a b c)
let p2 k = ldexp 1. k

(* [a * b] is [2^-24 + 2^-57] or [2^-24 - 2^-57]: [(1 + u)(1 - u + u^2) = 1 +
   u^3] with [u = 2^-11]. Added to 1 (or to [1 + 2^-23]) it rounds in binary64
   onto the binary32 midpoint, whose tie then goes the wrong way. *)
let midpoint_cases =
  let u = p2 (-11) in
  [
    ("1 + (2^-24 + 2^-57)", 1. +. u, p2 (-24) *. (1. -. u +. (u *. u)), 1.);
    ( "(1 + 2^-23) + (2^-24 - 2^-57)",
      1. -. u,
      p2 (-24) *. (1. +. u +. (u *. u)),
      1. +. p2 (-23) );
    ( "-(1 + (2^-24 + 2^-57))",
      -.(1. +. u),
      p2 (-24) *. (1. -. u +. (u *. u)),
      -1. );
  ]

let%expect_test "midpoint triples: one rounding, not two" =
  List.iter
    (fun (name, a, b, c) ->
      assert (round32 a = a && round32 b = b && round32 c = c);
      Fmt.pr "%s: fma32 %h, oracle %h, binary64 then binary32 %h@." name
        (F32_math.fma32 a b c) (oracle a b c) (double_rounded a b c))
    midpoint_cases;
  [%expect
    {|
    1 + (2^-24 + 2^-57): fma32 0x1.000002p+0, oracle 0x1.000002p+0, binary64 then binary32 0x1p+0
    (1 + 2^-23) + (2^-24 - 2^-57): fma32 0x1.000002p+0, oracle 0x1.000002p+0, binary64 then binary32 0x1.000004p+0
    -(1 + (2^-24 + 2^-57)): fma32 -0x1.000002p+0, oracle -0x1.000002p+0, binary64 then binary32 -0x1p+0 |}]

(* Random binary32 triples: bit patterns from a 30-bit generator (so the 32-bit
   [int] of js_of_ocaml is never exceeded), biased toward exponents that make
   [a * b] and [c] overlap, where rounding is decided. *)
let%expect_test "random triples against the oracle" =
  let st = Random.State.make [| 20261024 |] in
  let word () =
    Int32.logor
      (Int32.shift_left (Int32.of_int (Random.State.bits st land 0xFFFF)) 16)
      (Int32.of_int (Random.State.bits st land 0xFFFF))
  in
  let near e =
    (* sign and mantissa random, exponent [e] plus a small offset *)
    let w = word () in
    let ex = e + (Random.State.bits st mod 9) - 4 in
    Int32.float_of_bits
      (Int32.logor
         (Int32.logand w 0x807FFFFFl)
         (Int32.shift_left (Int32.of_int (max 1 (min 254 ex))) 23))
  in
  let n = 200_000 in
  let differ = ref 0 and double = ref 0 and nonfinite = ref 0 in
  for k = 1 to n do
    let a, b, c =
      if k mod 4 = 0 then
        ( Int32.float_of_bits (word ()),
          Int32.float_of_bits (word ()),
          Int32.float_of_bits (word ()) )
      else
        let ea = 100 + (Random.State.bits st mod 60) in
        let eb = 100 + (Random.State.bits st mod 60) in
        (near ea, near eb, near (ea + eb - 127 + (Random.State.bits st mod 30)))
    in
    let r = F32_math.fma32 a b c in
    if not (Float.is_finite r) then incr nonfinite;
    let o = oracle a b c in
    if not (same r o || (Float.is_nan r && Float.is_nan o)) then incr differ;
    let d = double_rounded a b c in
    if not (same d o || (Float.is_nan d && Float.is_nan o)) then incr double
  done;
  Fmt.pr "%d triples: fma32 differs from the oracle on %d@." n !differ;
  Fmt.pr "nonfinite results: %d; double rounding differs on %d@." !nonfinite
    !double;
  [%expect
    {|
    200000 triples: fma32 differs from the oracle on 0
    nonfinite results: 6735; double rounding differs on 0 |}]
