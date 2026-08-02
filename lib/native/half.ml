(* Software codecs for the two 16-bit float formats, which have no Bigarray kind
   (native_tensor_design.md §2). A stored cell is a raw 16-bit [int]; [to_float]
   decodes it to an OCaml float (the read path used when loading reduced-precision
   weights/activations), [of_float] encodes (the rarer store path).

   [Int32.bits_of_float] / [Int32.float_of_bits] give the IEEE-754 *single*
   precision (float32) bit pattern, which both formats are derived from. *)

(* The float32 pattern as its two 16-bit halves, rather than one [0, 2^32) [int].

   That whole range is not representable as a non-negative [int] on a
   32-bit-[int] runtime — js_of_ocaml has [Sys.int_size = 32] — so the former
   [Int32.to_int … land 0xFFFFFFFF] could not work there: the literal 0xFFFFFFFF
   truncates to -1, making the mask a silent no-op (js_of_ocaml warns about
   exactly this). It only appeared to work because [lsr] on a jsoo [int] is an
   unsigned 32-bit shift. Natively the mask is load-bearing, since
   [Int32.to_int] is negative whenever the sign bit is set.

   Halves sidestep the dilemma: every intermediate below stays under 2^23, so
   the arithmetic is exact on both. *)
let f32_halves f =
  let b = Int32.bits_of_float f in
  let hi = Int32.to_int (Int32.shift_right_logical b 16) in
  let lo = Int32.to_int (Int32.logand b 0xFFFFl) in
  (hi, lo)

module Bf16 = struct
  (* bfloat16 is simply the high 16 bits of a float32. *)
  let to_float (bits : int) : float =
    Int32.float_of_bits (Int32.shift_left (Int32.of_int (bits land 0xFFFF)) 16)

  let of_float (f : float) : int =
    let hi, lo = f32_halves f in
    (* Round to nearest even on the 16-bit boundary. Adding 0x7FFF + lsb to the
       full pattern only ever perturbs [hi] through the carry out of [lo], so
       propagating that carry is the same computation without the wide value. *)
    let lsb = hi land 1 in
    let carry = (lo + 0x7FFF + lsb) lsr 16 in
    (hi + carry) land 0xFFFF

  let pp fmt bits = Format.fprintf fmt "0x%04x(%g)" bits (to_float bits)
end

module Half = struct
  (* IEEE-754 binary16: 1 sign, 5 exponent (bias 15), 10 mantissa. *)
  let to_float (h : int) : float =
    let sign = (h lsr 15) land 1 in
    let exp = (h lsr 10) land 0x1F in
    let mant = h land 0x3FF in
    let m =
      if exp = 0 then ldexp (float_of_int mant) (-24) (* subnormal / zero *)
      else if exp = 0x1F then if mant = 0 then infinity else nan
      else ldexp (float_of_int (mant lor 0x400)) (exp - 25)
    in
    if sign = 1 then -.m else m

  (* Encode the normal range with round-to-nearest-even; overflow -> inf,
     underflow -> 0 (subnormals flushed). Sufficient for inference outputs. *)
  let of_float (f : float) : int =
    let hi, lo = f32_halves f in
    (* Sign and exponent live entirely in [hi]; the 23-bit mantissa is [hi]'s low
       7 bits above all of [lo]. Widest intermediate is the mantissa, < 2^23. *)
    let sign = hi land 0x8000 in
    let exp = (hi lsr 7) land 0xFF in
    let mant = ((hi land 0x7F) lsl 16) lor lo in
    if exp = 0xFF then sign lor 0x7C00 lor if mant <> 0 then 0x200 else 0
      (* inf / nan *)
    else begin
      let e = exp - 127 + 15 in
      if e >= 0x1F then sign lor 0x7C00 (* overflow -> inf *)
      else if e <= 0 then sign (* underflow -> signed zero *)
      else begin
        let h = (e lsl 10) lor (mant lsr 13) in
        let rem = mant land 0x1FFF in
        let h =
          if rem > 0x1000 || (rem = 0x1000 && h land 1 = 1) then h + 1 else h
        in
        sign lor h
      end
    end

  let pp fmt bits = Format.fprintf fmt "0x%04x(%g)" bits (to_float bits)
end
