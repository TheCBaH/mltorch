(* A cell's codec as generic Machine IR: the decode of every storage format to
   its working value (binary16 and bfloat16 by integer primitives, a quantized
   read by its explicit operations) and the encode of a working value. Split
   from {!Mir_lower}, which holds the instruction-by-instruction walk. *)

open Ssa_ir
open Machine_ir
open Mir_lower_core

(* binary16 bits (in an i32) to binary64 by primitives: a normal or an
   infinity/NaN by its binary32 bit pattern, a subnormal or zero as its
   significand times 2^-24, the sign applied last. *)
let f16_decode ?(f32 = false) st h =
  let i32 x = i32k st x in
  let iar o a b = emit st (Mir_op.Iarith (o, a, b)) in
  let s = iar Mir_op.Iarith.Shr_u h (i32 15L) in
  let e =
    iar Mir_op.Iarith.And (iar Mir_op.Iarith.Shr_u h (i32 10L)) (i32 0x1FL)
  in
  let m = iar Mir_op.Iarith.And h (i32 0x3FFL) in
  let sign = iar Mir_op.Iarith.Shl s (i32 31L) in
  let mant = iar Mir_op.Iarith.Shl m (i32 13L) in
  let f32_of bits =
    let v = emit st (Mir_op.Bitcast (Mir_type.F32, bits)) in
    if f32 then v else emit st (Mir_op.Fconvert (Mir_op.Fconvert.F32_to_f64, v))
  in
  let normal =
    f32_of
      (iar Mir_op.Iarith.Or sign
         (iar Mir_op.Iarith.Or
            (iar Mir_op.Iarith.Shl
               (iar Mir_op.Iarith.Add e (i32 112L))
               (i32 23L))
            mant))
  in
  let special =
    f32_of
      (iar Mir_op.Iarith.Or sign (iar Mir_op.Iarith.Or (i32 0x7F80_0000L) mant))
  in
  let tiny =
    (* exact in either precision: [m] below 2^10, times a power of two *)
    if f32 then
      emit st
        (Mir_op.Fbinary
           ( Mir_op.Fbinary.Mul,
             emit st (Mir_op.Fconvert (Mir_op.Fconvert.S64_to_f32, sext st m)),
             const st (Mir_const.f32 (Float.ldexp 1. (-24))) ))
    else
      emit st
        (Mir_op.Fbinary
           ( Mir_op.Fbinary.Mul,
             emit st (Mir_op.Fconvert (Mir_op.Fconvert.S64_to_f64, sext st m)),
             f64k st (Float.ldexp 1. (-24)) ))
  in
  let tiny =
    select st
      (icmp st Mir_op.Icmp.Eq s (i32 0L))
      tiny
      (emit st (Mir_op.Funary (Mir_op.Funary.Neg, tiny)))
  in
  select st
    (icmp st Mir_op.Icmp.Eq e (i32 0L))
    tiny
    (select st (icmp st Mir_op.Icmp.Eq e (i32 0x1FL)) special normal)

let zext32 st v = emit st (Mir_op.Iext (Mir_op.Iext.Zext, Mir_width.W32, v))

let decode ?entry ?channel st op addr (d : Ssa_op.Decode.t) =
  with_role st Mir_origin.Role.Decode;
  let load width align =
    emit st (Mir_op.Load { Mir_op.Access.width; addr; align })
  in
  match d with
  | Ssa_op.Decode.Bf16_to_f32 ->
      emit st
        (Mir_op.Bitcast
           ( Mir_type.F32,
             emit st
               (Mir_op.Iarith
                  ( Mir_op.Iarith.Shl,
                    zext32 st (load Mir_width.W16 2L),
                    i32k st 16L )) ))
  | Ssa_op.Decode.Bool_to_f32 ->
      let b = zext32 st (load Mir_width.W8 1L) in
      select st
        (icmp st Mir_op.Icmp.Ne b (i32k st 0L))
        (const st (Mir_const.f32 1.))
        (const st (Mir_const.f32 0.))
  | Ssa_op.Decode.F16_to_f32 ->
      f16_decode ~f32:true st (zext32 st (load Mir_width.W16 2L))
  | Ssa_op.Decode.I32_to_f32 ->
      (* an int32 is exact in int64, so one rounding to binary32 *)
      emit st
        (Mir_op.Fconvert
           (Mir_op.Fconvert.S64_to_f32, sext st (load Mir_width.W32 4L)))
  | Ssa_op.Decode.Bf16_to_f64 ->
      (* the high half of a binary32 *)
      let bits =
        emit st
          (Mir_op.Iarith
             (Mir_op.Iarith.Shl, zext32 st (load Mir_width.W16 2L), i32k st 16L))
      in
      emit st
        (Mir_op.Fconvert
           ( Mir_op.Fconvert.F32_to_f64,
             emit st (Mir_op.Bitcast (Mir_type.F32, bits)) ))
  | Ssa_op.Decode.Bool_to_f64 ->
      let b = zext32 st (load Mir_width.W8 1L) in
      select st
        (icmp st Mir_op.Icmp.Ne b (i32k st 0L))
        (f64k st 1.) (f64k st 0.)
  | Ssa_op.Decode.F16_to_f64 ->
      f16_decode st (zext32 st (load Mir_width.W16 2L))
  | Ssa_op.Decode.I32_to_f64 ->
      emit st
        (Mir_op.Fconvert
           (Mir_op.Fconvert.S64_to_f64, sext st (load Mir_width.W32 4L)))
  | ( Ssa_op.Decode.I16_dequant | Ssa_op.Decode.I16_dequant32
    | Ssa_op.Decode.I8_dequant | Ssa_op.Decode.I8_dequant32 ) as d -> (
      match entry with
      | None -> unsupported st op
      | Some e ->
          let f32 =
            match d with
            | Ssa_op.Decode.I16_dequant32 | Ssa_op.Decode.I8_dequant32 -> true
            | _ -> false
          in
          let q =
            match d with
            | Ssa_op.Decode.I8_dequant | Ssa_op.Decode.I8_dequant32 ->
                load Mir_width.W8 1L
            | _ -> load Mir_width.W16 2L
          in
          Mir_lower_f32.dequantize st op e q ~channel ~f32)
  | Ssa_op.Decode.F32_exact ->
      emit st (Mir_op.Bitcast (Mir_type.F32, load Mir_width.W32 4L))
  | Ssa_op.Decode.F32_to_f64 ->
      let bits = load Mir_width.W32 4L in
      let f = emit st (Mir_op.Bitcast (Mir_type.F32, bits)) in
      emit st (Mir_op.Fconvert (Mir_op.Fconvert.F32_to_f64, f))
  | Ssa_op.Decode.F64_to_f64 ->
      emit st (Mir_op.Bitcast (Mir_type.F64, load Mir_width.W64 8L))
  | Ssa_op.Decode.I64 -> load Mir_width.W64 8L
  | Ssa_op.Decode.I64_to_f64 ->
      emit st
        (Mir_op.Fconvert (Mir_op.Fconvert.S64_to_f64, load Mir_width.W64 8L))

let encode st addr (enc : Ssa_op.Encode.t) v =
  with_role st Mir_origin.Role.Encode;
  let store width align x =
    emit_unit st (Mir_op.Store ({ Mir_op.Access.width; addr; align }, x))
  in
  match enc with
  | Ssa_op.Encode.F32_exact ->
      store Mir_width.W32 4L (emit st (Mir_op.Bitcast (Mir_type.i32, v)))
  | Ssa_op.Encode.F32_round ->
      let r = emit st (Mir_op.Fconvert (Mir_op.Fconvert.F64_to_f32, v)) in
      store Mir_width.W32 4L (emit st (Mir_op.Bitcast (Mir_type.i32, r)))
  | Ssa_op.Encode.I64 -> store Mir_width.W64 8L v
  | Ssa_op.Encode.Bf16_round ->
      (* to nearest even on the high half of the binary32's bits *)
      let b = emit st (Mir_op.Bitcast (Mir_type.i32, v)) in
      let ar op x y = emit st (Mir_op.Iarith (op, x, y)) in
      let lsb =
        ar Mir_op.Iarith.And
          (ar Mir_op.Iarith.Shr_u b (i32k st 16L))
          (i32k st 1L)
      in
      let rounded =
        ar Mir_op.Iarith.Add b (ar Mir_op.Iarith.Add (i32k st 0x7FFFL) lsb)
      in
      store Mir_width.W16 2L
        (emit st
           (Mir_op.Itrunc
              (Mir_width.W16, ar Mir_op.Iarith.Shr_u rounded (i32k st 16L))))
  | Ssa_op.Encode.Bool_nonzero ->
      (* [v <> 0.]: a NaN stores 1 *)
      let zero = emit st (Mir_op.Fcmp (Mir_op.Fcmp.Eq, v, f64k st 0.)) in
      let b = select st zero (i32k st 0L) (i32k st 1L) in
      store Mir_width.W8 1L (emit st (Mir_op.Itrunc (Mir_width.W8, b)))
