(* The two 16-bit float formats, which no Bigarray kind holds: a stored cell is
   a raw 16-bit integer. Decoding is exact, and is the definition the Loop IR's
   own decode is compared against. Every intermediate stays below 2^24, so the
   arithmetic is exact under a 32-bit [int] too. *)

(* bfloat16 is the high 16 bits of a binary32. *)
let bf16_to_float bits =
  Int32.float_of_bits (Int32.shift_left (Int32.of_int (bits land 0xFFFF)) 16)

(* The bfloat16 a value stores as: rounded to binary32, then to nearest even on
   the high half of its bit pattern, the [Attention_block.bf16_round] rule. *)
let float_to_bf16 x =
  let r = Attention_block.bf16_round (F32_math.round32 x) in
  Int32.to_int (Int32.shift_right_logical (Int32.bits_of_float r) 16)
  land 0xFFFF

(* IEEE binary16: 1 sign, 5 exponent (bias 15), 10 mantissa. *)
let f16_to_float h =
  let sign = (h lsr 15) land 1 in
  let exp = (h lsr 10) land 0x1F in
  let mant = h land 0x3FF in
  let m =
    if exp = 0 then ldexp (float_of_int mant) (-24)
    else if exp = 0x1F then if mant = 0 then infinity else nan
    else ldexp (float_of_int (mant lor 0x400)) (exp - 25)
  in
  if sign = 1 then -.m else m
