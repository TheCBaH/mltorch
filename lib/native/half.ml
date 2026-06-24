(* Software codecs for the two 16-bit float formats, which have no Bigarray kind
   (native_tensor_design.md §2). A stored cell is a raw 16-bit [int]; [to_float]
   decodes it to an OCaml float (the read path used when loading reduced-precision
   weights/activations), [of_float] encodes (the rarer store path).

   [Int32.bits_of_float] / [Int32.float_of_bits] give the IEEE-754 *single*
   precision (float32) bit pattern, which both formats are derived from. *)

let f32_bits f = Int32.to_int (Int32.bits_of_float f) land 0xFFFFFFFF
let f32_of_bits b = Int32.float_of_bits (Int32.of_int b)

module Bf16 = struct
  (* bfloat16 is simply the high 16 bits of a float32. *)
  let to_float (bits : int) : float = f32_of_bits ((bits land 0xFFFF) lsl 16)

  let of_float (f : float) : int =
    let b = f32_bits f in
    (* round to nearest even on the 16-bit boundary *)
    let lsb = (b lsr 16) land 1 in
    let b = b + 0x7FFF + lsb in
    (b lsr 16) land 0xFFFF

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
    let b = f32_bits f in
    let sign = (b lsr 16) land 0x8000 in
    let exp = (b lsr 23) land 0xFF in
    let mant = b land 0x7FFFFF in
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
