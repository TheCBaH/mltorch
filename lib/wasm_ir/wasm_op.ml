(* The numeric instructions of the supported subset as one closed table: each
   carries its spec encoding and its stack signature, so the validator and the
   encoder read the same row and cannot drift. Alphabetical; the encoding is the
   spec's, not the order. Float-to-int conversions are the non-trapping
   saturating forms only: generated code never traps on a value. The 128-bit
   operations are the standard SIMD ones with no immediate; lane access, memory
   access and constants are separate instructions ([Wasm.Instr]). No relaxed
   operation is representable. *)
type t =
  | F32_abs
  | F32_add
  | F32_convert_i32_s
  | F32_convert_i32_u
  | F32_convert_i64_s
  | F32_demote_f64
  | F32_div
  | F32_eq
  | F32_ge
  | F32_gt
  | F32_le
  | F32_lt
  | F32_max
  | F32_min
  | F32_mul
  | F32_ne
  | F32_neg
  | F32_reinterpret_i32
  | F32_sqrt
  | F32_sub
  | F32_trunc
  | F32x4_abs
  | F32x4_add
  | F32x4_demote_f64x2_zero
  | F32x4_div
  | F32x4_eq
  | F32x4_ge
  | F32x4_gt
  | F32x4_le
  | F32x4_lt
  | F32x4_max
  | F32x4_min
  | F32x4_mul
  | F32x4_ne
  | F32x4_neg
  | F32x4_relaxed_madd
  | F32x4_splat
  | F32x4_sqrt
  | F32x4_sub
  | F32x4_trunc
  | F64_abs
  | F64_add
  | F64_ceil
  | F64_convert_i32_s
  | F64_convert_i32_u
  | F64_convert_i64_s
  | F64_convert_i64_u
  | F64_copysign
  | F64_div
  | F64_eq
  | F64_floor
  | F64_ge
  | F64_gt
  | F64_le
  | F64_lt
  | F64_max
  | F64_min
  | F64_mul
  | F64_ne
  | F64_nearest
  | F64_neg
  | F64_promote_f32
  | F64_reinterpret_i64
  | F64_sqrt
  | F64_sub
  | F64_trunc
  | F64x2_abs
  | F64x2_add
  | F64x2_ceil
  | F64x2_convert_low_i32x4_s
  | F64x2_convert_low_i32x4_u
  | F64x2_div
  | F64x2_eq
  | F64x2_floor
  | F64x2_ge
  | F64x2_gt
  | F64x2_le
  | F64x2_lt
  | F64x2_max
  | F64x2_min
  | F64x2_mul
  | F64x2_ne
  | F64x2_nearest
  | F64x2_neg
  | F64x2_promote_low_f32x4
  | F64x2_splat
  | F64x2_sqrt
  | F64x2_sub
  | F64x2_trunc
  | I32_add
  | I32_and
  | I32_clz
  | I32_ctz
  | I32_div_s
  | I32_div_u
  | I32_eq
  | I32_eqz
  | I32_extend16_s
  | I32_extend8_s
  | I32_ge_s
  | I32_ge_u
  | I32_gt_s
  | I32_gt_u
  | I32_le_s
  | I32_le_u
  | I32_lt_s
  | I32_lt_u
  | I32_mul
  | I32_ne
  | I32_or
  | I32_popcnt
  | I32_reinterpret_f32
  | I32_rem_s
  | I32_rem_u
  | I32_shl
  | I32_shr_s
  | I32_shr_u
  | I32_sub
  | I32_trunc_sat_f64_s
  | I32_trunc_sat_f64_u
  | I32_wrap_i64
  | I32_xor
  | I32x4_splat
  | I32x4_trunc_sat_f64x2_s_zero
  | I32x4_trunc_sat_f64x2_u_zero
  | I64_add
  | I64_and
  | I64_div_s
  | I64_div_u
  | I64_eq
  | I64_eqz
  | I64_extend_i32_s
  | I64_extend_i32_u
  | I64_ge_s
  | I64_ge_u
  | I64_gt_s
  | I64_gt_u
  | I64_le_s
  | I64_le_u
  | I64_lt_s
  | I64_lt_u
  | I64_mul
  | I64_ne
  | I64_or
  | I64_reinterpret_f64
  | I64_rem_s
  | I64_rem_u
  | I64_shl
  | I64_shr_s
  | I64_shr_u
  | I64_sub
  | I64_trunc_sat_f64_s
  | I64_trunc_sat_f64_u
  | I64_xor
  | I64x2_add
  | I64x2_eq
  | I64x2_ge_s
  | I64x2_gt_s
  | I64x2_le_s
  | I64x2_lt_s
  | I64x2_mul
  | I64x2_ne
  | I64x2_shl
  | I64x2_shr_s
  | I64x2_shr_u
  | I64x2_splat
  | I64x2_sub
  | V128_and
  | V128_andnot
  | V128_any_true
  | V128_bitselect
  | V128_not
  | V128_or
  | V128_xor

let all =
  [
    F32_abs;
    F32_add;
    F32_convert_i32_s;
    F32_convert_i32_u;
    F32_convert_i64_s;
    F32_demote_f64;
    F32_div;
    F32_eq;
    F32_ge;
    F32_gt;
    F32_le;
    F32_lt;
    F32_max;
    F32_min;
    F32_mul;
    F32_ne;
    F32_neg;
    F32_reinterpret_i32;
    F32_sqrt;
    F32_sub;
    F32_trunc;
    F32x4_abs;
    F32x4_add;
    F32x4_demote_f64x2_zero;
    F32x4_div;
    F32x4_eq;
    F32x4_ge;
    F32x4_gt;
    F32x4_le;
    F32x4_lt;
    F32x4_max;
    F32x4_min;
    F32x4_mul;
    F32x4_ne;
    F32x4_neg;
    F32x4_relaxed_madd;
    F32x4_splat;
    F32x4_sqrt;
    F32x4_sub;
    F32x4_trunc;
    F64_abs;
    F64_add;
    F64_ceil;
    F64_convert_i32_s;
    F64_convert_i32_u;
    F64_convert_i64_s;
    F64_convert_i64_u;
    F64_copysign;
    F64_div;
    F64_eq;
    F64_floor;
    F64_ge;
    F64_gt;
    F64_le;
    F64_lt;
    F64_max;
    F64_min;
    F64_mul;
    F64_ne;
    F64_nearest;
    F64_neg;
    F64_promote_f32;
    F64_reinterpret_i64;
    F64_sqrt;
    F64_sub;
    F64_trunc;
    F64x2_abs;
    F64x2_add;
    F64x2_ceil;
    F64x2_convert_low_i32x4_s;
    F64x2_convert_low_i32x4_u;
    F64x2_div;
    F64x2_eq;
    F64x2_floor;
    F64x2_ge;
    F64x2_gt;
    F64x2_le;
    F64x2_lt;
    F64x2_max;
    F64x2_min;
    F64x2_mul;
    F64x2_ne;
    F64x2_nearest;
    F64x2_neg;
    F64x2_promote_low_f32x4;
    F64x2_splat;
    F64x2_sqrt;
    F64x2_sub;
    F64x2_trunc;
    I32_add;
    I32_and;
    I32_clz;
    I32_ctz;
    I32_div_s;
    I32_div_u;
    I32_eq;
    I32_eqz;
    I32_extend16_s;
    I32_extend8_s;
    I32_ge_s;
    I32_ge_u;
    I32_gt_s;
    I32_gt_u;
    I32_le_s;
    I32_le_u;
    I32_lt_s;
    I32_lt_u;
    I32_mul;
    I32_ne;
    I32_or;
    I32_popcnt;
    I32_reinterpret_f32;
    I32_rem_s;
    I32_rem_u;
    I32_shl;
    I32_shr_s;
    I32_shr_u;
    I32_sub;
    I32_trunc_sat_f64_s;
    I32_trunc_sat_f64_u;
    I32_wrap_i64;
    I32_xor;
    I32x4_splat;
    I32x4_trunc_sat_f64x2_s_zero;
    I32x4_trunc_sat_f64x2_u_zero;
    I64_add;
    I64_and;
    I64_div_s;
    I64_div_u;
    I64_eq;
    I64_eqz;
    I64_extend_i32_s;
    I64_extend_i32_u;
    I64_ge_s;
    I64_ge_u;
    I64_gt_s;
    I64_gt_u;
    I64_le_s;
    I64_le_u;
    I64_lt_s;
    I64_lt_u;
    I64_mul;
    I64_ne;
    I64_or;
    I64_reinterpret_f64;
    I64_rem_s;
    I64_rem_u;
    I64_shl;
    I64_shr_s;
    I64_shr_u;
    I64_sub;
    I64_trunc_sat_f64_s;
    I64_trunc_sat_f64_u;
    I64_xor;
    I64x2_add;
    I64x2_eq;
    I64x2_ge_s;
    I64x2_gt_s;
    I64x2_le_s;
    I64x2_lt_s;
    I64x2_mul;
    I64x2_ne;
    I64x2_shl;
    I64x2_shr_s;
    I64x2_shr_u;
    I64x2_splat;
    I64x2_sub;
    V128_and;
    V128_andnot;
    V128_any_true;
    V128_bitselect;
    V128_not;
    V128_or;
    V128_xor;
  ]

let name = function
  | F32_abs -> "f32.abs"
  | F32_add -> "f32.add"
  | F32_convert_i32_s -> "f32.convert_i32_s"
  | F32_convert_i32_u -> "f32.convert_i32_u"
  | F32_convert_i64_s -> "f32.convert_i64_s"
  | F32_demote_f64 -> "f32.demote_f64"
  | F32_div -> "f32.div"
  | F32_eq -> "f32.eq"
  | F32_ge -> "f32.ge"
  | F32_gt -> "f32.gt"
  | F32_le -> "f32.le"
  | F32_lt -> "f32.lt"
  | F32_max -> "f32.max"
  | F32_min -> "f32.min"
  | F32_mul -> "f32.mul"
  | F32_ne -> "f32.ne"
  | F32_neg -> "f32.neg"
  | F32_reinterpret_i32 -> "f32.reinterpret_i32"
  | F32_sqrt -> "f32.sqrt"
  | F32_sub -> "f32.sub"
  | F32_trunc -> "f32.trunc"
  | F32x4_abs -> "f32x4.abs"
  | F32x4_add -> "f32x4.add"
  | F32x4_demote_f64x2_zero -> "f32x4.demote_f64x2_zero"
  | F32x4_div -> "f32x4.div"
  | F32x4_eq -> "f32x4.eq"
  | F32x4_ge -> "f32x4.ge"
  | F32x4_gt -> "f32x4.gt"
  | F32x4_le -> "f32x4.le"
  | F32x4_lt -> "f32x4.lt"
  | F32x4_max -> "f32x4.max"
  | F32x4_min -> "f32x4.min"
  | F32x4_mul -> "f32x4.mul"
  | F32x4_ne -> "f32x4.ne"
  | F32x4_neg -> "f32x4.neg"
  | F32x4_relaxed_madd -> "f32x4.relaxed_madd"
  | F32x4_splat -> "f32x4.splat"
  | F32x4_sqrt -> "f32x4.sqrt"
  | F32x4_sub -> "f32x4.sub"
  | F32x4_trunc -> "f32x4.trunc"
  | F64_abs -> "f64.abs"
  | F64_add -> "f64.add"
  | F64_ceil -> "f64.ceil"
  | F64_convert_i32_s -> "f64.convert_i32_s"
  | F64_convert_i32_u -> "f64.convert_i32_u"
  | F64_convert_i64_s -> "f64.convert_i64_s"
  | F64_convert_i64_u -> "f64.convert_i64_u"
  | F64_copysign -> "f64.copysign"
  | F64_div -> "f64.div"
  | F64_eq -> "f64.eq"
  | F64_floor -> "f64.floor"
  | F64_ge -> "f64.ge"
  | F64_gt -> "f64.gt"
  | F64_le -> "f64.le"
  | F64_lt -> "f64.lt"
  | F64_max -> "f64.max"
  | F64_min -> "f64.min"
  | F64_mul -> "f64.mul"
  | F64_ne -> "f64.ne"
  | F64_nearest -> "f64.nearest"
  | F64_neg -> "f64.neg"
  | F64_promote_f32 -> "f64.promote_f32"
  | F64_reinterpret_i64 -> "f64.reinterpret_i64"
  | F64_sqrt -> "f64.sqrt"
  | F64_sub -> "f64.sub"
  | F64_trunc -> "f64.trunc"
  | F64x2_abs -> "f64x2.abs"
  | F64x2_add -> "f64x2.add"
  | F64x2_ceil -> "f64x2.ceil"
  | F64x2_convert_low_i32x4_s -> "f64x2.convert_low_i32x4_s"
  | F64x2_convert_low_i32x4_u -> "f64x2.convert_low_i32x4_u"
  | F64x2_div -> "f64x2.div"
  | F64x2_eq -> "f64x2.eq"
  | F64x2_floor -> "f64x2.floor"
  | F64x2_ge -> "f64x2.ge"
  | F64x2_gt -> "f64x2.gt"
  | F64x2_le -> "f64x2.le"
  | F64x2_lt -> "f64x2.lt"
  | F64x2_max -> "f64x2.max"
  | F64x2_min -> "f64x2.min"
  | F64x2_mul -> "f64x2.mul"
  | F64x2_ne -> "f64x2.ne"
  | F64x2_nearest -> "f64x2.nearest"
  | F64x2_neg -> "f64x2.neg"
  | F64x2_promote_low_f32x4 -> "f64x2.promote_low_f32x4"
  | F64x2_splat -> "f64x2.splat"
  | F64x2_sqrt -> "f64x2.sqrt"
  | F64x2_sub -> "f64x2.sub"
  | F64x2_trunc -> "f64x2.trunc"
  | I32_add -> "i32.add"
  | I32_and -> "i32.and"
  | I32_clz -> "i32.clz"
  | I32_ctz -> "i32.ctz"
  | I32_div_s -> "i32.div_s"
  | I32_div_u -> "i32.div_u"
  | I32_eq -> "i32.eq"
  | I32_eqz -> "i32.eqz"
  | I32_extend16_s -> "i32.extend16_s"
  | I32_extend8_s -> "i32.extend8_s"
  | I32_ge_s -> "i32.ge_s"
  | I32_ge_u -> "i32.ge_u"
  | I32_gt_s -> "i32.gt_s"
  | I32_gt_u -> "i32.gt_u"
  | I32_le_s -> "i32.le_s"
  | I32_le_u -> "i32.le_u"
  | I32_lt_s -> "i32.lt_s"
  | I32_lt_u -> "i32.lt_u"
  | I32_mul -> "i32.mul"
  | I32_ne -> "i32.ne"
  | I32_or -> "i32.or"
  | I32_popcnt -> "i32.popcnt"
  | I32_reinterpret_f32 -> "i32.reinterpret_f32"
  | I32_rem_s -> "i32.rem_s"
  | I32_rem_u -> "i32.rem_u"
  | I32_shl -> "i32.shl"
  | I32_shr_s -> "i32.shr_s"
  | I32_shr_u -> "i32.shr_u"
  | I32_sub -> "i32.sub"
  | I32_trunc_sat_f64_s -> "i32.trunc_sat_f64_s"
  | I32_trunc_sat_f64_u -> "i32.trunc_sat_f64_u"
  | I32_wrap_i64 -> "i32.wrap_i64"
  | I32_xor -> "i32.xor"
  | I32x4_splat -> "i32x4.splat"
  | I32x4_trunc_sat_f64x2_s_zero -> "i32x4.trunc_sat_f64x2_s_zero"
  | I32x4_trunc_sat_f64x2_u_zero -> "i32x4.trunc_sat_f64x2_u_zero"
  | I64_add -> "i64.add"
  | I64_and -> "i64.and"
  | I64_div_s -> "i64.div_s"
  | I64_div_u -> "i64.div_u"
  | I64_eq -> "i64.eq"
  | I64_eqz -> "i64.eqz"
  | I64_extend_i32_s -> "i64.extend_i32_s"
  | I64_extend_i32_u -> "i64.extend_i32_u"
  | I64_ge_s -> "i64.ge_s"
  | I64_ge_u -> "i64.ge_u"
  | I64_gt_s -> "i64.gt_s"
  | I64_gt_u -> "i64.gt_u"
  | I64_le_s -> "i64.le_s"
  | I64_le_u -> "i64.le_u"
  | I64_lt_s -> "i64.lt_s"
  | I64_lt_u -> "i64.lt_u"
  | I64_mul -> "i64.mul"
  | I64_ne -> "i64.ne"
  | I64_or -> "i64.or"
  | I64_reinterpret_f64 -> "i64.reinterpret_f64"
  | I64_rem_s -> "i64.rem_s"
  | I64_rem_u -> "i64.rem_u"
  | I64_shl -> "i64.shl"
  | I64_shr_s -> "i64.shr_s"
  | I64_shr_u -> "i64.shr_u"
  | I64_sub -> "i64.sub"
  | I64_trunc_sat_f64_s -> "i64.trunc_sat_f64_s"
  | I64_trunc_sat_f64_u -> "i64.trunc_sat_f64_u"
  | I64_xor -> "i64.xor"
  | I64x2_add -> "i64x2.add"
  | I64x2_eq -> "i64x2.eq"
  | I64x2_ge_s -> "i64x2.ge_s"
  | I64x2_gt_s -> "i64x2.gt_s"
  | I64x2_le_s -> "i64x2.le_s"
  | I64x2_lt_s -> "i64x2.lt_s"
  | I64x2_mul -> "i64x2.mul"
  | I64x2_ne -> "i64x2.ne"
  | I64x2_shl -> "i64x2.shl"
  | I64x2_shr_s -> "i64x2.shr_s"
  | I64x2_shr_u -> "i64x2.shr_u"
  | I64x2_splat -> "i64x2.splat"
  | I64x2_sub -> "i64x2.sub"
  | V128_and -> "v128.and"
  | V128_andnot -> "v128.andnot"
  | V128_any_true -> "v128.any_true"
  | V128_bitselect -> "v128.bitselect"
  | V128_not -> "v128.not"
  | V128_or -> "v128.or"
  | V128_xor -> "v128.xor"

let bytes = function
  | F32_abs -> [ 0x8B ]
  | F32_add -> [ 0x92 ]
  | F32_convert_i32_s -> [ 0xB2 ]
  | F32_convert_i32_u -> [ 0xB3 ]
  | F32_convert_i64_s -> [ 0xB4 ]
  | F32_demote_f64 -> [ 0xB6 ]
  | F32_div -> [ 0x95 ]
  | F32_eq -> [ 0x5B ]
  | F32_ge -> [ 0x60 ]
  | F32_gt -> [ 0x5E ]
  | F32_le -> [ 0x5F ]
  | F32_lt -> [ 0x5D ]
  | F32_max -> [ 0x97 ]
  | F32_min -> [ 0x96 ]
  | F32_mul -> [ 0x94 ]
  | F32_ne -> [ 0x5C ]
  | F32_neg -> [ 0x8C ]
  | F32_reinterpret_i32 -> [ 0xBE ]
  | F32_sqrt -> [ 0x91 ]
  | F32_sub -> [ 0x93 ]
  | F32_trunc -> [ 0x8F ]
  | F32x4_abs -> [ 0xFD; 0xE0; 0x01 ]
  | F32x4_add -> [ 0xFD; 0xE4; 0x01 ]
  | F32x4_demote_f64x2_zero -> [ 0xFD; 0x5E ]
  | F32x4_div -> [ 0xFD; 0xE7; 0x01 ]
  | F32x4_eq -> [ 0xFD; 0x41 ]
  | F32x4_ge -> [ 0xFD; 0x46 ]
  | F32x4_gt -> [ 0xFD; 0x44 ]
  | F32x4_le -> [ 0xFD; 0x45 ]
  | F32x4_lt -> [ 0xFD; 0x43 ]
  | F32x4_max -> [ 0xFD; 0xE9; 0x01 ]
  | F32x4_min -> [ 0xFD; 0xE8; 0x01 ]
  | F32x4_mul -> [ 0xFD; 0xE6; 0x01 ]
  | F32x4_ne -> [ 0xFD; 0x42 ]
  | F32x4_neg -> [ 0xFD; 0xE1; 0x01 ]
  | F32x4_relaxed_madd -> [ 0xFD; 0x85; 0x02 ]
  | F32x4_splat -> [ 0xFD; 0x13 ]
  | F32x4_sqrt -> [ 0xFD; 0xE3; 0x01 ]
  | F32x4_sub -> [ 0xFD; 0xE5; 0x01 ]
  | F32x4_trunc -> [ 0xFD; 0x69 ]
  | F64_abs -> [ 0x99 ]
  | F64_add -> [ 0xA0 ]
  | F64_ceil -> [ 0x9B ]
  | F64_convert_i32_s -> [ 0xB7 ]
  | F64_convert_i32_u -> [ 0xB8 ]
  | F64_convert_i64_s -> [ 0xB9 ]
  | F64_convert_i64_u -> [ 0xBA ]
  | F64_copysign -> [ 0xA6 ]
  | F64_div -> [ 0xA3 ]
  | F64_eq -> [ 0x61 ]
  | F64_floor -> [ 0x9C ]
  | F64_ge -> [ 0x66 ]
  | F64_gt -> [ 0x64 ]
  | F64_le -> [ 0x65 ]
  | F64_lt -> [ 0x63 ]
  | F64_max -> [ 0xA5 ]
  | F64_min -> [ 0xA4 ]
  | F64_mul -> [ 0xA2 ]
  | F64_ne -> [ 0x62 ]
  | F64_nearest -> [ 0x9E ]
  | F64_neg -> [ 0x9A ]
  | F64_promote_f32 -> [ 0xBB ]
  | F64_reinterpret_i64 -> [ 0xBF ]
  | F64_sqrt -> [ 0x9F ]
  | F64_sub -> [ 0xA1 ]
  | F64_trunc -> [ 0x9D ]
  | F64x2_abs -> [ 0xFD; 0xEC; 0x01 ]
  | F64x2_add -> [ 0xFD; 0xF0; 0x01 ]
  | F64x2_ceil -> [ 0xFD; 0x74 ]
  | F64x2_convert_low_i32x4_s -> [ 0xFD; 0xFE; 0x01 ]
  | F64x2_convert_low_i32x4_u -> [ 0xFD; 0xFF; 0x01 ]
  | F64x2_div -> [ 0xFD; 0xF3; 0x01 ]
  | F64x2_eq -> [ 0xFD; 0x47 ]
  | F64x2_floor -> [ 0xFD; 0x75 ]
  | F64x2_ge -> [ 0xFD; 0x4C ]
  | F64x2_gt -> [ 0xFD; 0x4A ]
  | F64x2_le -> [ 0xFD; 0x4B ]
  | F64x2_lt -> [ 0xFD; 0x49 ]
  | F64x2_max -> [ 0xFD; 0xF5; 0x01 ]
  | F64x2_min -> [ 0xFD; 0xF4; 0x01 ]
  | F64x2_mul -> [ 0xFD; 0xF2; 0x01 ]
  | F64x2_ne -> [ 0xFD; 0x48 ]
  | F64x2_nearest -> [ 0xFD; 0x94; 0x01 ]
  | F64x2_neg -> [ 0xFD; 0xED; 0x01 ]
  | F64x2_promote_low_f32x4 -> [ 0xFD; 0x5F ]
  | F64x2_splat -> [ 0xFD; 0x14 ]
  | F64x2_sqrt -> [ 0xFD; 0xEF; 0x01 ]
  | F64x2_sub -> [ 0xFD; 0xF1; 0x01 ]
  | F64x2_trunc -> [ 0xFD; 0x7A ]
  | I32_add -> [ 0x6A ]
  | I32_and -> [ 0x71 ]
  | I32_clz -> [ 0x67 ]
  | I32_ctz -> [ 0x68 ]
  | I32_div_s -> [ 0x6D ]
  | I32_div_u -> [ 0x6E ]
  | I32_eq -> [ 0x46 ]
  | I32_eqz -> [ 0x45 ]
  | I32_extend16_s -> [ 0xC1 ]
  | I32_extend8_s -> [ 0xC0 ]
  | I32_ge_s -> [ 0x4E ]
  | I32_ge_u -> [ 0x4F ]
  | I32_gt_s -> [ 0x4A ]
  | I32_gt_u -> [ 0x4B ]
  | I32_le_s -> [ 0x4C ]
  | I32_le_u -> [ 0x4D ]
  | I32_lt_s -> [ 0x48 ]
  | I32_lt_u -> [ 0x49 ]
  | I32_mul -> [ 0x6C ]
  | I32_ne -> [ 0x47 ]
  | I32_or -> [ 0x72 ]
  | I32_popcnt -> [ 0x69 ]
  | I32_reinterpret_f32 -> [ 0xBC ]
  | I32_rem_s -> [ 0x6F ]
  | I32_rem_u -> [ 0x70 ]
  | I32_shl -> [ 0x74 ]
  | I32_shr_s -> [ 0x75 ]
  | I32_shr_u -> [ 0x76 ]
  | I32_sub -> [ 0x6B ]
  | I32_trunc_sat_f64_s -> [ 0xFC; 0x02 ]
  | I32_trunc_sat_f64_u -> [ 0xFC; 0x03 ]
  | I32_wrap_i64 -> [ 0xA7 ]
  | I32_xor -> [ 0x73 ]
  | I32x4_splat -> [ 0xFD; 0x11 ]
  | I32x4_trunc_sat_f64x2_s_zero -> [ 0xFD; 0xFC; 0x01 ]
  | I32x4_trunc_sat_f64x2_u_zero -> [ 0xFD; 0xFD; 0x01 ]
  | I64_add -> [ 0x7C ]
  | I64_and -> [ 0x83 ]
  | I64_div_s -> [ 0x7F ]
  | I64_div_u -> [ 0x80 ]
  | I64_eq -> [ 0x51 ]
  | I64_eqz -> [ 0x50 ]
  | I64_extend_i32_s -> [ 0xAC ]
  | I64_extend_i32_u -> [ 0xAD ]
  | I64_ge_s -> [ 0x59 ]
  | I64_ge_u -> [ 0x5A ]
  | I64_gt_s -> [ 0x55 ]
  | I64_gt_u -> [ 0x56 ]
  | I64_le_s -> [ 0x57 ]
  | I64_le_u -> [ 0x58 ]
  | I64_lt_s -> [ 0x53 ]
  | I64_lt_u -> [ 0x54 ]
  | I64_mul -> [ 0x7E ]
  | I64_ne -> [ 0x52 ]
  | I64_or -> [ 0x84 ]
  | I64_reinterpret_f64 -> [ 0xBD ]
  | I64_rem_s -> [ 0x81 ]
  | I64_rem_u -> [ 0x82 ]
  | I64_shl -> [ 0x86 ]
  | I64_shr_s -> [ 0x87 ]
  | I64_shr_u -> [ 0x88 ]
  | I64_sub -> [ 0x7D ]
  | I64_trunc_sat_f64_s -> [ 0xFC; 0x06 ]
  | I64_trunc_sat_f64_u -> [ 0xFC; 0x07 ]
  | I64_xor -> [ 0x85 ]
  | I64x2_add -> [ 0xFD; 0xCE; 0x01 ]
  | I64x2_eq -> [ 0xFD; 0xD6; 0x01 ]
  | I64x2_ge_s -> [ 0xFD; 0xDB; 0x01 ]
  | I64x2_gt_s -> [ 0xFD; 0xD9; 0x01 ]
  | I64x2_le_s -> [ 0xFD; 0xDA; 0x01 ]
  | I64x2_lt_s -> [ 0xFD; 0xD8; 0x01 ]
  | I64x2_mul -> [ 0xFD; 0xD5; 0x01 ]
  | I64x2_ne -> [ 0xFD; 0xD7; 0x01 ]
  | I64x2_shl -> [ 0xFD; 0xCB; 0x01 ]
  | I64x2_shr_s -> [ 0xFD; 0xCC; 0x01 ]
  | I64x2_shr_u -> [ 0xFD; 0xCD; 0x01 ]
  | I64x2_splat -> [ 0xFD; 0x12 ]
  | I64x2_sub -> [ 0xFD; 0xD1; 0x01 ]
  | V128_and -> [ 0xFD; 0x4E ]
  | V128_andnot -> [ 0xFD; 0x4F ]
  | V128_any_true -> [ 0xFD; 0x53 ]
  | V128_bitselect -> [ 0xFD; 0x52 ]
  | V128_not -> [ 0xFD; 0x4D ]
  | V128_or -> [ 0xFD; 0x50 ]
  | V128_xor -> [ 0xFD; 0x51 ]

let signature = function
  | F32_abs -> ([ Wasm_type.F32 ], [ Wasm_type.F32 ])
  | F32_add -> ([ Wasm_type.F32; Wasm_type.F32 ], [ Wasm_type.F32 ])
  | F32_convert_i32_s -> ([ Wasm_type.I32 ], [ Wasm_type.F32 ])
  | F32_convert_i32_u -> ([ Wasm_type.I32 ], [ Wasm_type.F32 ])
  | F32_convert_i64_s -> ([ Wasm_type.I64 ], [ Wasm_type.F32 ])
  | F32_demote_f64 -> ([ Wasm_type.F64 ], [ Wasm_type.F32 ])
  | F32_div -> ([ Wasm_type.F32; Wasm_type.F32 ], [ Wasm_type.F32 ])
  | F32_eq -> ([ Wasm_type.F32; Wasm_type.F32 ], [ Wasm_type.I32 ])
  | F32_ge -> ([ Wasm_type.F32; Wasm_type.F32 ], [ Wasm_type.I32 ])
  | F32_gt -> ([ Wasm_type.F32; Wasm_type.F32 ], [ Wasm_type.I32 ])
  | F32_le -> ([ Wasm_type.F32; Wasm_type.F32 ], [ Wasm_type.I32 ])
  | F32_lt -> ([ Wasm_type.F32; Wasm_type.F32 ], [ Wasm_type.I32 ])
  | F32_max -> ([ Wasm_type.F32; Wasm_type.F32 ], [ Wasm_type.F32 ])
  | F32_min -> ([ Wasm_type.F32; Wasm_type.F32 ], [ Wasm_type.F32 ])
  | F32_mul -> ([ Wasm_type.F32; Wasm_type.F32 ], [ Wasm_type.F32 ])
  | F32_ne -> ([ Wasm_type.F32; Wasm_type.F32 ], [ Wasm_type.I32 ])
  | F32_neg -> ([ Wasm_type.F32 ], [ Wasm_type.F32 ])
  | F32_reinterpret_i32 -> ([ Wasm_type.I32 ], [ Wasm_type.F32 ])
  | F32_sqrt -> ([ Wasm_type.F32 ], [ Wasm_type.F32 ])
  | F32_sub -> ([ Wasm_type.F32; Wasm_type.F32 ], [ Wasm_type.F32 ])
  | F32_trunc -> ([ Wasm_type.F32 ], [ Wasm_type.F32 ])
  | F32x4_abs -> ([ Wasm_type.V128 ], [ Wasm_type.V128 ])
  | F32x4_add -> ([ Wasm_type.V128; Wasm_type.V128 ], [ Wasm_type.V128 ])
  | F32x4_demote_f64x2_zero -> ([ Wasm_type.V128 ], [ Wasm_type.V128 ])
  | F32x4_div -> ([ Wasm_type.V128; Wasm_type.V128 ], [ Wasm_type.V128 ])
  | F32x4_eq -> ([ Wasm_type.V128; Wasm_type.V128 ], [ Wasm_type.V128 ])
  | F32x4_ge -> ([ Wasm_type.V128; Wasm_type.V128 ], [ Wasm_type.V128 ])
  | F32x4_gt -> ([ Wasm_type.V128; Wasm_type.V128 ], [ Wasm_type.V128 ])
  | F32x4_le -> ([ Wasm_type.V128; Wasm_type.V128 ], [ Wasm_type.V128 ])
  | F32x4_lt -> ([ Wasm_type.V128; Wasm_type.V128 ], [ Wasm_type.V128 ])
  | F32x4_max -> ([ Wasm_type.V128; Wasm_type.V128 ], [ Wasm_type.V128 ])
  | F32x4_min -> ([ Wasm_type.V128; Wasm_type.V128 ], [ Wasm_type.V128 ])
  | F32x4_mul -> ([ Wasm_type.V128; Wasm_type.V128 ], [ Wasm_type.V128 ])
  | F32x4_ne -> ([ Wasm_type.V128; Wasm_type.V128 ], [ Wasm_type.V128 ])
  | F32x4_neg -> ([ Wasm_type.V128 ], [ Wasm_type.V128 ])
  | F32x4_relaxed_madd ->
      ([ Wasm_type.V128; Wasm_type.V128; Wasm_type.V128 ], [ Wasm_type.V128 ])
  | F32x4_splat -> ([ Wasm_type.F32 ], [ Wasm_type.V128 ])
  | F32x4_sqrt -> ([ Wasm_type.V128 ], [ Wasm_type.V128 ])
  | F32x4_sub -> ([ Wasm_type.V128; Wasm_type.V128 ], [ Wasm_type.V128 ])
  | F32x4_trunc -> ([ Wasm_type.V128 ], [ Wasm_type.V128 ])
  | F64_abs -> ([ Wasm_type.F64 ], [ Wasm_type.F64 ])
  | F64_add -> ([ Wasm_type.F64; Wasm_type.F64 ], [ Wasm_type.F64 ])
  | F64_ceil -> ([ Wasm_type.F64 ], [ Wasm_type.F64 ])
  | F64_convert_i32_s -> ([ Wasm_type.I32 ], [ Wasm_type.F64 ])
  | F64_convert_i32_u -> ([ Wasm_type.I32 ], [ Wasm_type.F64 ])
  | F64_convert_i64_s -> ([ Wasm_type.I64 ], [ Wasm_type.F64 ])
  | F64_convert_i64_u -> ([ Wasm_type.I64 ], [ Wasm_type.F64 ])
  | F64_copysign -> ([ Wasm_type.F64; Wasm_type.F64 ], [ Wasm_type.F64 ])
  | F64_div -> ([ Wasm_type.F64; Wasm_type.F64 ], [ Wasm_type.F64 ])
  | F64_eq -> ([ Wasm_type.F64; Wasm_type.F64 ], [ Wasm_type.I32 ])
  | F64_floor -> ([ Wasm_type.F64 ], [ Wasm_type.F64 ])
  | F64_ge -> ([ Wasm_type.F64; Wasm_type.F64 ], [ Wasm_type.I32 ])
  | F64_gt -> ([ Wasm_type.F64; Wasm_type.F64 ], [ Wasm_type.I32 ])
  | F64_le -> ([ Wasm_type.F64; Wasm_type.F64 ], [ Wasm_type.I32 ])
  | F64_lt -> ([ Wasm_type.F64; Wasm_type.F64 ], [ Wasm_type.I32 ])
  | F64_max -> ([ Wasm_type.F64; Wasm_type.F64 ], [ Wasm_type.F64 ])
  | F64_min -> ([ Wasm_type.F64; Wasm_type.F64 ], [ Wasm_type.F64 ])
  | F64_mul -> ([ Wasm_type.F64; Wasm_type.F64 ], [ Wasm_type.F64 ])
  | F64_ne -> ([ Wasm_type.F64; Wasm_type.F64 ], [ Wasm_type.I32 ])
  | F64_nearest -> ([ Wasm_type.F64 ], [ Wasm_type.F64 ])
  | F64_neg -> ([ Wasm_type.F64 ], [ Wasm_type.F64 ])
  | F64_promote_f32 -> ([ Wasm_type.F32 ], [ Wasm_type.F64 ])
  | F64_reinterpret_i64 -> ([ Wasm_type.I64 ], [ Wasm_type.F64 ])
  | F64_sqrt -> ([ Wasm_type.F64 ], [ Wasm_type.F64 ])
  | F64_sub -> ([ Wasm_type.F64; Wasm_type.F64 ], [ Wasm_type.F64 ])
  | F64_trunc -> ([ Wasm_type.F64 ], [ Wasm_type.F64 ])
  | F64x2_abs -> ([ Wasm_type.V128 ], [ Wasm_type.V128 ])
  | F64x2_add -> ([ Wasm_type.V128; Wasm_type.V128 ], [ Wasm_type.V128 ])
  | F64x2_ceil -> ([ Wasm_type.V128 ], [ Wasm_type.V128 ])
  | F64x2_convert_low_i32x4_s -> ([ Wasm_type.V128 ], [ Wasm_type.V128 ])
  | F64x2_convert_low_i32x4_u -> ([ Wasm_type.V128 ], [ Wasm_type.V128 ])
  | F64x2_div -> ([ Wasm_type.V128; Wasm_type.V128 ], [ Wasm_type.V128 ])
  | F64x2_eq -> ([ Wasm_type.V128; Wasm_type.V128 ], [ Wasm_type.V128 ])
  | F64x2_floor -> ([ Wasm_type.V128 ], [ Wasm_type.V128 ])
  | F64x2_ge -> ([ Wasm_type.V128; Wasm_type.V128 ], [ Wasm_type.V128 ])
  | F64x2_gt -> ([ Wasm_type.V128; Wasm_type.V128 ], [ Wasm_type.V128 ])
  | F64x2_le -> ([ Wasm_type.V128; Wasm_type.V128 ], [ Wasm_type.V128 ])
  | F64x2_lt -> ([ Wasm_type.V128; Wasm_type.V128 ], [ Wasm_type.V128 ])
  | F64x2_max -> ([ Wasm_type.V128; Wasm_type.V128 ], [ Wasm_type.V128 ])
  | F64x2_min -> ([ Wasm_type.V128; Wasm_type.V128 ], [ Wasm_type.V128 ])
  | F64x2_mul -> ([ Wasm_type.V128; Wasm_type.V128 ], [ Wasm_type.V128 ])
  | F64x2_ne -> ([ Wasm_type.V128; Wasm_type.V128 ], [ Wasm_type.V128 ])
  | F64x2_nearest -> ([ Wasm_type.V128 ], [ Wasm_type.V128 ])
  | F64x2_neg -> ([ Wasm_type.V128 ], [ Wasm_type.V128 ])
  | F64x2_promote_low_f32x4 -> ([ Wasm_type.V128 ], [ Wasm_type.V128 ])
  | F64x2_splat -> ([ Wasm_type.F64 ], [ Wasm_type.V128 ])
  | F64x2_sqrt -> ([ Wasm_type.V128 ], [ Wasm_type.V128 ])
  | F64x2_sub -> ([ Wasm_type.V128; Wasm_type.V128 ], [ Wasm_type.V128 ])
  | F64x2_trunc -> ([ Wasm_type.V128 ], [ Wasm_type.V128 ])
  | I32_add -> ([ Wasm_type.I32; Wasm_type.I32 ], [ Wasm_type.I32 ])
  | I32_and -> ([ Wasm_type.I32; Wasm_type.I32 ], [ Wasm_type.I32 ])
  | I32_clz -> ([ Wasm_type.I32 ], [ Wasm_type.I32 ])
  | I32_ctz -> ([ Wasm_type.I32 ], [ Wasm_type.I32 ])
  | I32_div_s -> ([ Wasm_type.I32; Wasm_type.I32 ], [ Wasm_type.I32 ])
  | I32_div_u -> ([ Wasm_type.I32; Wasm_type.I32 ], [ Wasm_type.I32 ])
  | I32_eq -> ([ Wasm_type.I32; Wasm_type.I32 ], [ Wasm_type.I32 ])
  | I32_eqz -> ([ Wasm_type.I32 ], [ Wasm_type.I32 ])
  | I32_extend16_s -> ([ Wasm_type.I32 ], [ Wasm_type.I32 ])
  | I32_extend8_s -> ([ Wasm_type.I32 ], [ Wasm_type.I32 ])
  | I32_ge_s -> ([ Wasm_type.I32; Wasm_type.I32 ], [ Wasm_type.I32 ])
  | I32_ge_u -> ([ Wasm_type.I32; Wasm_type.I32 ], [ Wasm_type.I32 ])
  | I32_gt_s -> ([ Wasm_type.I32; Wasm_type.I32 ], [ Wasm_type.I32 ])
  | I32_gt_u -> ([ Wasm_type.I32; Wasm_type.I32 ], [ Wasm_type.I32 ])
  | I32_le_s -> ([ Wasm_type.I32; Wasm_type.I32 ], [ Wasm_type.I32 ])
  | I32_le_u -> ([ Wasm_type.I32; Wasm_type.I32 ], [ Wasm_type.I32 ])
  | I32_lt_s -> ([ Wasm_type.I32; Wasm_type.I32 ], [ Wasm_type.I32 ])
  | I32_lt_u -> ([ Wasm_type.I32; Wasm_type.I32 ], [ Wasm_type.I32 ])
  | I32_mul -> ([ Wasm_type.I32; Wasm_type.I32 ], [ Wasm_type.I32 ])
  | I32_ne -> ([ Wasm_type.I32; Wasm_type.I32 ], [ Wasm_type.I32 ])
  | I32_or -> ([ Wasm_type.I32; Wasm_type.I32 ], [ Wasm_type.I32 ])
  | I32_popcnt -> ([ Wasm_type.I32 ], [ Wasm_type.I32 ])
  | I32_reinterpret_f32 -> ([ Wasm_type.F32 ], [ Wasm_type.I32 ])
  | I32_rem_s -> ([ Wasm_type.I32; Wasm_type.I32 ], [ Wasm_type.I32 ])
  | I32_rem_u -> ([ Wasm_type.I32; Wasm_type.I32 ], [ Wasm_type.I32 ])
  | I32_shl -> ([ Wasm_type.I32; Wasm_type.I32 ], [ Wasm_type.I32 ])
  | I32_shr_s -> ([ Wasm_type.I32; Wasm_type.I32 ], [ Wasm_type.I32 ])
  | I32_shr_u -> ([ Wasm_type.I32; Wasm_type.I32 ], [ Wasm_type.I32 ])
  | I32_sub -> ([ Wasm_type.I32; Wasm_type.I32 ], [ Wasm_type.I32 ])
  | I32_trunc_sat_f64_s -> ([ Wasm_type.F64 ], [ Wasm_type.I32 ])
  | I32_trunc_sat_f64_u -> ([ Wasm_type.F64 ], [ Wasm_type.I32 ])
  | I32_wrap_i64 -> ([ Wasm_type.I64 ], [ Wasm_type.I32 ])
  | I32_xor -> ([ Wasm_type.I32; Wasm_type.I32 ], [ Wasm_type.I32 ])
  | I32x4_splat -> ([ Wasm_type.I32 ], [ Wasm_type.V128 ])
  | I32x4_trunc_sat_f64x2_s_zero -> ([ Wasm_type.V128 ], [ Wasm_type.V128 ])
  | I32x4_trunc_sat_f64x2_u_zero -> ([ Wasm_type.V128 ], [ Wasm_type.V128 ])
  | I64_add -> ([ Wasm_type.I64; Wasm_type.I64 ], [ Wasm_type.I64 ])
  | I64_and -> ([ Wasm_type.I64; Wasm_type.I64 ], [ Wasm_type.I64 ])
  | I64_div_s -> ([ Wasm_type.I64; Wasm_type.I64 ], [ Wasm_type.I64 ])
  | I64_div_u -> ([ Wasm_type.I64; Wasm_type.I64 ], [ Wasm_type.I64 ])
  | I64_eq -> ([ Wasm_type.I64; Wasm_type.I64 ], [ Wasm_type.I32 ])
  | I64_eqz -> ([ Wasm_type.I64 ], [ Wasm_type.I32 ])
  | I64_extend_i32_s -> ([ Wasm_type.I32 ], [ Wasm_type.I64 ])
  | I64_extend_i32_u -> ([ Wasm_type.I32 ], [ Wasm_type.I64 ])
  | I64_ge_s -> ([ Wasm_type.I64; Wasm_type.I64 ], [ Wasm_type.I32 ])
  | I64_ge_u -> ([ Wasm_type.I64; Wasm_type.I64 ], [ Wasm_type.I32 ])
  | I64_gt_s -> ([ Wasm_type.I64; Wasm_type.I64 ], [ Wasm_type.I32 ])
  | I64_gt_u -> ([ Wasm_type.I64; Wasm_type.I64 ], [ Wasm_type.I32 ])
  | I64_le_s -> ([ Wasm_type.I64; Wasm_type.I64 ], [ Wasm_type.I32 ])
  | I64_le_u -> ([ Wasm_type.I64; Wasm_type.I64 ], [ Wasm_type.I32 ])
  | I64_lt_s -> ([ Wasm_type.I64; Wasm_type.I64 ], [ Wasm_type.I32 ])
  | I64_lt_u -> ([ Wasm_type.I64; Wasm_type.I64 ], [ Wasm_type.I32 ])
  | I64_mul -> ([ Wasm_type.I64; Wasm_type.I64 ], [ Wasm_type.I64 ])
  | I64_ne -> ([ Wasm_type.I64; Wasm_type.I64 ], [ Wasm_type.I32 ])
  | I64_or -> ([ Wasm_type.I64; Wasm_type.I64 ], [ Wasm_type.I64 ])
  | I64_reinterpret_f64 -> ([ Wasm_type.F64 ], [ Wasm_type.I64 ])
  | I64_rem_s -> ([ Wasm_type.I64; Wasm_type.I64 ], [ Wasm_type.I64 ])
  | I64_rem_u -> ([ Wasm_type.I64; Wasm_type.I64 ], [ Wasm_type.I64 ])
  | I64_shl -> ([ Wasm_type.I64; Wasm_type.I64 ], [ Wasm_type.I64 ])
  | I64_shr_s -> ([ Wasm_type.I64; Wasm_type.I64 ], [ Wasm_type.I64 ])
  | I64_shr_u -> ([ Wasm_type.I64; Wasm_type.I64 ], [ Wasm_type.I64 ])
  | I64_sub -> ([ Wasm_type.I64; Wasm_type.I64 ], [ Wasm_type.I64 ])
  | I64_trunc_sat_f64_s -> ([ Wasm_type.F64 ], [ Wasm_type.I64 ])
  | I64_trunc_sat_f64_u -> ([ Wasm_type.F64 ], [ Wasm_type.I64 ])
  | I64_xor -> ([ Wasm_type.I64; Wasm_type.I64 ], [ Wasm_type.I64 ])
  | I64x2_add -> ([ Wasm_type.V128; Wasm_type.V128 ], [ Wasm_type.V128 ])
  | I64x2_eq -> ([ Wasm_type.V128; Wasm_type.V128 ], [ Wasm_type.V128 ])
  | I64x2_ge_s -> ([ Wasm_type.V128; Wasm_type.V128 ], [ Wasm_type.V128 ])
  | I64x2_gt_s -> ([ Wasm_type.V128; Wasm_type.V128 ], [ Wasm_type.V128 ])
  | I64x2_le_s -> ([ Wasm_type.V128; Wasm_type.V128 ], [ Wasm_type.V128 ])
  | I64x2_lt_s -> ([ Wasm_type.V128; Wasm_type.V128 ], [ Wasm_type.V128 ])
  | I64x2_mul -> ([ Wasm_type.V128; Wasm_type.V128 ], [ Wasm_type.V128 ])
  | I64x2_ne -> ([ Wasm_type.V128; Wasm_type.V128 ], [ Wasm_type.V128 ])
  | I64x2_shl -> ([ Wasm_type.V128; Wasm_type.I32 ], [ Wasm_type.V128 ])
  | I64x2_shr_s -> ([ Wasm_type.V128; Wasm_type.I32 ], [ Wasm_type.V128 ])
  | I64x2_shr_u -> ([ Wasm_type.V128; Wasm_type.I32 ], [ Wasm_type.V128 ])
  | I64x2_splat -> ([ Wasm_type.I64 ], [ Wasm_type.V128 ])
  | I64x2_sub -> ([ Wasm_type.V128; Wasm_type.V128 ], [ Wasm_type.V128 ])
  | V128_and -> ([ Wasm_type.V128; Wasm_type.V128 ], [ Wasm_type.V128 ])
  | V128_andnot -> ([ Wasm_type.V128; Wasm_type.V128 ], [ Wasm_type.V128 ])
  | V128_any_true -> ([ Wasm_type.V128 ], [ Wasm_type.I32 ])
  | V128_bitselect ->
      ([ Wasm_type.V128; Wasm_type.V128; Wasm_type.V128 ], [ Wasm_type.V128 ])
  | V128_not -> ([ Wasm_type.V128 ], [ Wasm_type.V128 ])
  | V128_or -> ([ Wasm_type.V128; Wasm_type.V128 ], [ Wasm_type.V128 ])
  | V128_xor -> ([ Wasm_type.V128; Wasm_type.V128 ], [ Wasm_type.V128 ])
