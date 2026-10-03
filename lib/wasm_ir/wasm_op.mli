(** The numeric instructions of the scalar subset. A closed table: the validator
    and the encoder read the same row for an operation, so its stack signature
    and its bytes cannot drift. Conversions from float to integer are the
    non-trapping saturating forms only. *)

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

val all : t list
(** Every operation, in declaration order. *)

val name : t -> string
(** The text-format mnemonic, e.g. ["i32.add"]. *)

val bytes : t -> int list
(** The spec encoding: one opcode byte, or [0xFC] and a sub-opcode. *)

val signature : t -> Wasm_type.t list * Wasm_type.t list
(** Operand types (deepest first) and result types. *)
