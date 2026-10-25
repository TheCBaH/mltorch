(* [F32_math.add_odd32] against the exact sum, as an integer times a power of
   two, rounded to odd in binary32 by counting bits: a different argument from
   the binary64 sticky bit, so agreement is evidence. The operands are what the
   matrix engine's bfloat16 pair adds: two products of bfloat16 values, over
   exponents wide enough for cancellation, a large gap and subnormal results. *)

let round32 = F32_math.round32

(* [x = m * 2^e] with [m] an integer *)
let decompose x =
  if x = 0. then (Z.zero, 0)
  else
    let f, e = Float.frexp x in
    (Z.of_float (Float.ldexp f 53), e - 53)

(* the exact sum of [xs] as [(m, e)] *)
let exact_sum xs =
  let parts = List.map decompose xs in
  let e = List.fold_left (fun acc (_, e) -> min acc e) max_int parts in
  let e = if e = max_int then 0 else e in
  ( List.fold_left
      (fun acc (m, e') -> Z.add acc (Z.shift_left m (e' - e)))
      Z.zero parts,
    e )

(* [m * 2^e] rounded to odd in binary32: truncated to 24 bits, then, when
   anything was cut off, the last bit set *)
let round_exact (m, e) =
  if Z.equal m Z.zero then 0.
  else
    let neg = Z.sign m < 0 in
    let m = Z.abs m in
    let bits = Z.numbits m in
    (* the value's top bit is at 2^(bits + e - 1); a binary32 keeps 24 bits
       from there, but nothing below 2^-149 *)
    let lsb = max (bits + e - 24) (-149) in
    let drop = lsb - e in
    let q =
      if drop <= 0 then Z.shift_left m (-drop)
      else
        let q = Z.shift_right m drop in
        let cut = not (Z.equal m (Z.shift_left q drop)) in
        if cut && not (Z.is_odd q) then Z.succ q else q
    in
    let x = Float.ldexp (Z.to_float q) lsb in
    if neg then -.x else x

let oracle a b = round_exact (exact_sum [ a; b ])
let same x y = Int64.equal (Int64.bits_of_float x) (Int64.bits_of_float y)

(* a random product of two bfloat16 values: 8 significant bits each, exponents
   around [e] *)
let product st e =
  let bf16 e =
    let m = 128 + Random.State.int st 128 in
    let x = Float.ldexp (float_of_int m) (e - 7) in
    if Random.State.bool st then -.x else x
  in
  bf16 (e + Random.State.int st 5 - 2) *. bf16 (Random.State.int st 5 - 2)

let%expect_test "random pairs against the exact value" =
  let st = Random.State.make [| 20261025 |] in
  let n = 400_000 in
  let differ = ref 0 and nearest = ref 0 in
  for k = 1 to n do
    let e = -60 + Random.State.int st 120 in
    let p0 = product st e in
    (* a gap from none to past a binary32, and subnormal results *)
    let e1 =
      match k mod 4 with
      | 0 -> -140 + Random.State.int st 10
      | 1 -> e + Random.State.int st 60 - 30
      | _ -> e + Random.State.int st 12 - 6
    in
    let p1 = product st e1 in
    let o = oracle p0 p1 in
    if not (same (F32_math.add_odd32 p0 p1) o) then incr differ;
    if not (same (round32 (p0 +. p1)) o) then incr nearest
  done;
  Fmt.pr "%d pairs, %d differ (to nearest: %d)@." n !differ !nearest;
  [%expect {| 400000 pairs, 0 differ (to nearest: 143346) |}]

(* The cases the matrix engine probe uses, added to 1: two products of 2^-24
   make 2^-23, and 2^-24 with 2^-48 is a bit past the midpoint of 1's
   neighbours. Any bit below the pair's last, however far (2^-60), sets it. *)
let%expect_test "the probe's cases" =
  List.iter
    (fun (name, p0, p1) ->
      Fmt.pr "%s: pair %h (by counting %h, to nearest %h)@." name
        (F32_math.add_odd32 p0 p1) (oracle p0 p1)
        (round32 (p0 +. p1)))
    [
      ("2^-24 twice", ldexp 1. (-24), ldexp 1. (-24));
      ("2^-24, 2^-48", ldexp 1. (-24), ldexp 1. (-48));
      ("2^-24, 2^-60", ldexp 1. (-24), ldexp 1. (-60));
    ];
  [%expect
    {|
    2^-24 twice: pair 0x1p-23 (by counting 0x1p-23, to nearest 0x1p-23)
    2^-24, 2^-48: pair 0x1.000002p-24 (by counting 0x1.000002p-24, to nearest 0x1p-24)
    2^-24, 2^-60: pair 0x1.000002p-24 (by counting 0x1.000002p-24, to nearest 0x1p-24) |}]
