(* Machine IR's own primitive numeric semantics over exact bits. Integers are
   canonical zero-extended bits of their width; binary32 is the low 32 bits;
   binary64 is its bits; predicates 0/1.

   Host environment (documented and tested): OCaml native code computes
   binary64 [+ - * /], [sqrt] and conversions with round-to-nearest-even,
   gradual underflow and no traps. Each primitive below is one host operation
   on values already rounded, so no host expression lets the compiler fuse a
   product into a sum — OCaml's arm64 backend does that for [x *. y +. z].
   Shared low-level primitives with the SSA oracle: binary64 arithmetic,
   [Float.fma] (correctly rounded) and the binary64-to-binary32 rounding of
   [Int32.bits_of_float]. Binary32 arithmetic, binary32 FMA, i64-to-binary32,
   maximum and the checked conversions are derived independently here. *)

open Machine_ir

let f64 bits = Int64.float_of_bits bits
let of_f64 x = Int64.bits_of_float x
let f32 bits = Int32.float_of_bits (Int64.to_int32 bits)

(* Binary64 to binary32 bits, one rounding (the shared host primitive). *)
let round32 x =
  Int64.logand (Int64.of_int32 (Int32.bits_of_float x)) 0xFFFF_FFFFL

(* A binary32 operation on binary32 operands: the exact binary64 result of
   [+ - * / sqrt] on two binary32 values rounds once more to binary32 without
   double-rounding error (53 >= 2 * 24 + 2). *)
let lift32_2 op a b = round32 (op (f32 a) (f32 b))
let is_nan x = Float.is_nan x

(* IEEE 754-2019 maximum: a NaN operand gives NaN; +0 is above -0. *)
let fmax x y =
  if is_nan x || is_nan y then Float.nan
  else if x = 0. && y = 0. then if Float.sign_bit x then y else x
  else if x > y then x
  else y

(* The next binary64 toward zero from a nonzero finite [x]. *)
let toward_zero x =
  let b = Int64.bits_of_float x in
  Int64.float_of_bits (Int64.pred b)

(* [x] with its last significand bit set: round-to-odd's sticky bit. *)
let make_odd x = Int64.float_of_bits (Int64.logor (Int64.bits_of_float x) 1L)

(* Binary32 [a * b + c] rounded once. The product of two binary32 values is
   exact in binary64; the sum is formed rounded-to-odd in binary64 (its error
   recovered by TwoSum), and round-to-odd at 53 bits followed by
   round-to-nearest at 24 bits is a single correct rounding. *)
let fma32 a b c =
  let a = f32 a and b = f32 b and c = f32 c in
  let p = Sys.opaque_identity (a *. b) in
  let s = p +. c in
  if not (Float.is_finite s && Float.is_finite p && Float.is_finite c) then
    round32 (Float.fma a b c)
  else
    (* TwoSum: s + e = p + c exactly *)
    let bb = s -. p in
    let e = p -. (s -. bb) +. (c -. bb) in
    if e = 0. || s = 0. then round32 s
    else
      (* truncate toward zero, then set the sticky bit *)
      let trunc = if e > 0. = (s > 0.) then s else toward_zero s in
      round32 (make_odd trunc)

(* i64 to binary64 or binary32, one rounding each. *)
let s64_to_f64 n = of_f64 (Int64.to_float n)

(* Binary32 of an int64, rounded once: the magnitude is first shortened to 53
   significant bits with round-to-odd (any dropped bit sets the last kept
   one), which is exact in binary64 and then rounds correctly to binary32. *)
let s64_to_f32 n =
  if Int64.equal n Int64.min_int then round32 (-9223372036854775808.)
  else
    let neg = Int64.compare n 0L < 0 in
    let m = if neg then Int64.neg n else n in
    let rec bits k =
      if Int64.equal (Int64.shift_right_logical m k) 0L then k else bits (k + 1)
    in
    let width = bits 0 in
    let v =
      if width <= 53 then Int64.to_float m
      else
        let drop = width - 53 in
        let kept = Int64.shift_right_logical m drop in
        let lost = Int64.logand m (Int64.pred (Int64.shift_left 1L drop)) in
        let kept = if Int64.equal lost 0L then kept else Int64.logor kept 1L in
        Float.ldexp (Int64.to_float kept) drop
    in
    round32 (if neg then -.v else v)

(* The truncated int64 of a binary64 inside [-2^63, 2^63); [None] outside,
   for a NaN or an infinity. *)
let f64_to_s64 x =
  if Float.is_nan x || not (Float.is_finite x) then None
  else if x >= 9223372036854775808. || x < -9223372036854775808. then None
  else Some (Int64.of_float x)

let int_binary (op : Mir_op.Iarith.t) w a b =
  let n = Mir_width.normalize w in
  match op with
  | Mir_op.Iarith.Add -> n (Int64.add a b)
  | Mir_op.Iarith.And -> Int64.logand a b
  | Mir_op.Iarith.Mul -> n (Int64.mul a b)
  | Mir_op.Iarith.Or -> Int64.logor a b
  | Mir_op.Iarith.Shl -> n (Int64.shift_left a (Int64.to_int b))
  | Mir_op.Iarith.Shr_s ->
      n (Int64.shift_right (Mir_width.signed w a) (Int64.to_int b))
  | Mir_op.Iarith.Shr_u -> Int64.shift_right_logical a (Int64.to_int b)
  | Mir_op.Iarith.Sub -> n (Int64.sub a b)
  | Mir_op.Iarith.Xor -> Int64.logxor a b

let int_compare (c : Mir_op.Icmp.t) w a b =
  let s = Mir_width.signed w in
  match c with
  | Mir_op.Icmp.Eq -> Int64.equal a b
  | Mir_op.Icmp.Ne -> not (Int64.equal a b)
  | Mir_op.Icmp.Sle -> Int64.compare (s a) (s b) <= 0
  | Mir_op.Icmp.Slt -> Int64.compare (s a) (s b) < 0
  | Mir_op.Icmp.Ule -> Int64.unsigned_compare a b <= 0
  | Mir_op.Icmp.Ult -> Int64.unsigned_compare a b < 0

(* Signed division and remainder on their defined domain; [None] outside. *)
let int_div (op : Mir_op.Idiv.t) w a b =
  let x = Mir_width.signed w a and y = Mir_width.signed w b in
  if Int64.equal y 0L then None
  else if Int64.equal x (Mir_width.min_signed w) && Int64.equal y (-1L) then
    None
  else
    Some
      (Mir_width.normalize w
         (match op with
         | Mir_op.Idiv.Div_s -> Int64.div x y
         | Mir_op.Idiv.Rem_s -> Int64.rem x y))

let float_binary (op : Mir_op.Fbinary.t) =
  match op with
  | Mir_op.Fbinary.Add -> ( +. )
  | Mir_op.Fbinary.Div -> ( /. )
  | Mir_op.Fbinary.Max -> fmax
  | Mir_op.Fbinary.Mul -> ( *. )
  | Mir_op.Fbinary.Sub -> ( -. )

let float_compare (c : Mir_op.Fcmp.t) x y =
  match c with
  | Mir_op.Fcmp.Eq -> x = y
  | Mir_op.Fcmp.Le -> x <= y
  | Mir_op.Fcmp.Lt -> x < y
  | Mir_op.Fcmp.Unordered -> is_nan x || is_nan y

let float_unary (u : Mir_op.Funary.t) x =
  match u with
  | Mir_op.Funary.Neg -> Float.neg x
  | Mir_op.Funary.Sqrt -> Float.sqrt x
  | Mir_op.Funary.Trunc -> Float.trunc x
