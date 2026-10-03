(* The numeric instructions of the scalar subset, as one closed table: each
   carries its spec encoding and its stack signature, so the validator and the
   encoder read the same row and cannot drift. Alphabetical; the encoding is the
   spec's, not the order. Float-to-int conversions are the non-trapping
   saturating forms only: generated code never traps on a value. *)
type t =
  | F32_demote_f64
  | F32_reinterpret_i32
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

let all =
  [
    F32_demote_f64;
    F32_reinterpret_i32;
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
  ]

let name = function
  | F32_demote_f64 -> "f32.demote_f64"
  | F32_reinterpret_i32 -> "f32.reinterpret_i32"
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

let bytes = function
  | F32_demote_f64 -> [ 0xB6 ]
  | F32_reinterpret_i32 -> [ 0xBE ]
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

let signature = function
  | F32_demote_f64 -> ([ Wasm_type.F64 ], [ Wasm_type.F32 ])
  | F32_reinterpret_i32 -> ([ Wasm_type.I32 ], [ Wasm_type.F32 ])
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
