(** The numeric instructions of the supported subset: scalar, and the standard
    128-bit SIMD operations that take no immediate (lane access, memory access
    and constants are separate instructions in {!Wasm.Instr}). A closed table:
    the validator and the encoder read the same row for an operation, so its
    stack signature and its bytes cannot drift. Conversions from float to
    integer are the non-trapping saturating forms only. The only relaxed SIMD
    operation is [F32x4_relaxed_madd], reported by {!Wasm_features} so a host
    can refuse it. *)

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
      (** [a * b + c], fused or not at the engine's choice: the one relaxed-SIMD
          operation, which needs the [relaxed-simd] feature *)
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

val all : t list
(** Every operation, in declaration order. *)

val name : t -> string
(** The text-format mnemonic, e.g. ["i32.add"]. *)

val bytes : t -> int list
(** The spec encoding: one opcode byte, [0xFC] and a sub-opcode, or [0xFD] and a
    LEB128 sub-opcode. *)

val signature : t -> Wasm_type.t list * Wasm_type.t list
(** Operand types (deepest first) and result types. *)
