type t = Bulk_memory | Non_trapping_float_to_int | Sign_extension | Simd128

let all = [ Bulk_memory; Non_trapping_float_to_int; Sign_extension; Simd128 ]

let name = function
  | Bulk_memory -> "bulk-memory"
  | Non_trapping_float_to_int -> "nontrapping-float-to-int"
  | Sign_extension -> "sign-extension"
  | Simd128 -> "simd128"

let of_op : Wasm_op.t -> t option = function
  | Wasm_op.F32x4_demote_f64x2_zero | Wasm_op.F32x4_splat | Wasm_op.F64x2_abs
  | Wasm_op.F64x2_add | Wasm_op.F64x2_ceil | Wasm_op.F64x2_convert_low_i32x4_s
  | Wasm_op.F64x2_convert_low_i32x4_u | Wasm_op.F64x2_div | Wasm_op.F64x2_eq
  | Wasm_op.F64x2_floor | Wasm_op.F64x2_ge | Wasm_op.F64x2_gt | Wasm_op.F64x2_le
  | Wasm_op.F64x2_lt | Wasm_op.F64x2_max | Wasm_op.F64x2_min | Wasm_op.F64x2_mul
  | Wasm_op.F64x2_ne | Wasm_op.F64x2_nearest | Wasm_op.F64x2_neg
  | Wasm_op.F64x2_promote_low_f32x4 | Wasm_op.F64x2_splat | Wasm_op.F64x2_sqrt
  | Wasm_op.F64x2_sub | Wasm_op.F64x2_trunc | Wasm_op.I32x4_splat
  | Wasm_op.I32x4_trunc_sat_f64x2_s_zero | Wasm_op.I32x4_trunc_sat_f64x2_u_zero
  | Wasm_op.I64x2_add | Wasm_op.I64x2_eq | Wasm_op.I64x2_ge_s
  | Wasm_op.I64x2_gt_s | Wasm_op.I64x2_le_s | Wasm_op.I64x2_lt_s
  | Wasm_op.I64x2_mul | Wasm_op.I64x2_ne | Wasm_op.I64x2_shl
  | Wasm_op.I64x2_shr_s | Wasm_op.I64x2_shr_u | Wasm_op.I64x2_splat
  | Wasm_op.I64x2_sub | Wasm_op.V128_and | Wasm_op.V128_andnot
  | Wasm_op.V128_any_true | Wasm_op.V128_bitselect | Wasm_op.V128_not
  | Wasm_op.V128_or | Wasm_op.V128_xor ->
      Some Simd128
  | Wasm_op.I32_extend16_s | Wasm_op.I32_extend8_s -> Some Sign_extension
  | Wasm_op.I32_trunc_sat_f64_s | Wasm_op.I32_trunc_sat_f64_u
  | Wasm_op.I64_trunc_sat_f64_s | Wasm_op.I64_trunc_sat_f64_u ->
      Some Non_trapping_float_to_int
  | Wasm_op.F32_demote_f64 | Wasm_op.F32_reinterpret_i32 | Wasm_op.F64_abs
  | Wasm_op.F64_add | Wasm_op.F64_ceil | Wasm_op.F64_convert_i32_s
  | Wasm_op.F64_convert_i32_u | Wasm_op.F64_convert_i64_s
  | Wasm_op.F64_convert_i64_u | Wasm_op.F64_copysign | Wasm_op.F64_div
  | Wasm_op.F64_eq | Wasm_op.F64_floor | Wasm_op.F64_ge | Wasm_op.F64_gt
  | Wasm_op.F64_le | Wasm_op.F64_lt | Wasm_op.F64_max | Wasm_op.F64_min
  | Wasm_op.F64_mul | Wasm_op.F64_ne | Wasm_op.F64_nearest | Wasm_op.F64_neg
  | Wasm_op.F64_promote_f32 | Wasm_op.F64_reinterpret_i64 | Wasm_op.F64_sqrt
  | Wasm_op.F64_sub | Wasm_op.F64_trunc | Wasm_op.I32_add | Wasm_op.I32_and
  | Wasm_op.I32_clz | Wasm_op.I32_ctz | Wasm_op.I32_div_s | Wasm_op.I32_div_u
  | Wasm_op.I32_eq | Wasm_op.I32_eqz | Wasm_op.I32_ge_s | Wasm_op.I32_ge_u
  | Wasm_op.I32_gt_s | Wasm_op.I32_gt_u | Wasm_op.I32_le_s | Wasm_op.I32_le_u
  | Wasm_op.I32_lt_s | Wasm_op.I32_lt_u | Wasm_op.I32_mul | Wasm_op.I32_ne
  | Wasm_op.I32_or | Wasm_op.I32_popcnt | Wasm_op.I32_reinterpret_f32
  | Wasm_op.I32_rem_s | Wasm_op.I32_rem_u | Wasm_op.I32_shl | Wasm_op.I32_shr_s
  | Wasm_op.I32_shr_u | Wasm_op.I32_sub | Wasm_op.I32_wrap_i64 | Wasm_op.I32_xor
  | Wasm_op.I64_add | Wasm_op.I64_and | Wasm_op.I64_div_s | Wasm_op.I64_div_u
  | Wasm_op.I64_eq | Wasm_op.I64_eqz | Wasm_op.I64_extend_i32_s
  | Wasm_op.I64_extend_i32_u | Wasm_op.I64_ge_s | Wasm_op.I64_ge_u
  | Wasm_op.I64_gt_s | Wasm_op.I64_gt_u | Wasm_op.I64_le_s | Wasm_op.I64_le_u
  | Wasm_op.I64_lt_s | Wasm_op.I64_lt_u | Wasm_op.I64_mul | Wasm_op.I64_ne
  | Wasm_op.I64_or | Wasm_op.I64_reinterpret_f64 | Wasm_op.I64_rem_s
  | Wasm_op.I64_rem_u | Wasm_op.I64_shl | Wasm_op.I64_shr_s | Wasm_op.I64_shr_u
  | Wasm_op.I64_sub | Wasm_op.I64_xor ->
      None

let rec of_instr acc (i : Wasm.Instr.t) =
  match i with
  | Wasm.Instr.Block (_, l) | Wasm.Instr.Loop (_, l) ->
      List.fold_left of_instr acc l
  | Wasm.Instr.If (_, a, b) ->
      List.fold_left of_instr (List.fold_left of_instr acc a) b
  | Wasm.Instr.Memory_copy | Wasm.Instr.Memory_fill -> Bulk_memory :: acc
  | Wasm.Instr.Numeric op -> (
      match of_op op with Some f -> f :: acc | None -> acc)
  | Wasm.Instr.Simd_lane _ | Wasm.Instr.Simd_load _ | Wasm.Instr.Simd_store _
  | Wasm.Instr.V128_const _ ->
      Simd128 :: acc
  | Wasm.Instr.Br _ | Wasm.Instr.Br_if _ | Wasm.Instr.Call _ | Wasm.Instr.Drop
  | Wasm.Instr.F32_const _ | Wasm.Instr.F64_const _ | Wasm.Instr.Global_get _
  | Wasm.Instr.Global_set _ | Wasm.Instr.I32_const _ | Wasm.Instr.I64_const _
  | Wasm.Instr.Load _ | Wasm.Instr.Local_get _ | Wasm.Instr.Local_set _
  | Wasm.Instr.Local_tee _ | Wasm.Instr.Return | Wasm.Instr.Select
  | Wasm.Instr.Store _ | Wasm.Instr.Unreachable ->
      acc

let of_module (m : Wasm.Module.t) =
  let used =
    List.fold_left
      (fun acc (f : Wasm.Func.t) ->
        List.fold_left of_instr acc f.Wasm.Func.body)
      [] m.Wasm.Module.funcs
  in
  List.filter (fun f -> List.mem f used) all

(* A module of one function whose body is the feature's instruction, over a
   one-page memory where it needs one. *)
let probe f =
  let memory = Some { Wasm.Memory.min_pages = 1; max_pages = None } in
  let func params results body =
    { Wasm.Func.type_ = { Wasm.Func_type.params; results }; locals = []; body }
  in
  let body =
    match f with
    | Bulk_memory ->
        func [] []
          [
            Wasm.Instr.I32_const 0l;
            Wasm.Instr.I32_const 0l;
            Wasm.Instr.I32_const 0l;
            Wasm.Instr.Memory_fill;
          ]
    | Non_trapping_float_to_int ->
        func [ Wasm_type.F64 ] [ Wasm_type.I32 ]
          [
            Wasm.Instr.Local_get 0;
            Wasm.Instr.Numeric Wasm_op.I32_trunc_sat_f64_s;
          ]
    | Sign_extension ->
        func [ Wasm_type.I32 ] [ Wasm_type.I32 ]
          [ Wasm.Instr.Local_get 0; Wasm.Instr.Numeric Wasm_op.I32_extend8_s ]
    | Simd128 ->
        (* No vector instruction is representable yet: the smallest module that
           needs the extension is [v128.const] dropped, written as bytes. *)
        func [] [] []
  in
  let m = { Wasm.Module.empty with funcs = [ body ]; memory } in
  match f with
  | Simd128 ->
      (* type () -> (); one function; body: v128.const 0 (0xFD 0x0C and 16
         bytes), drop, end. *)
      "\000asm\001\000\000\000\001\004\001\096\000\000\003\002\001\000\n\
       \023\001\021\000\253\012" ^ String.make 16 '\000' ^ "\026\011"
  | Bulk_memory | Non_trapping_float_to_int | Sign_extension -> (
      match Err.payload (Wasm_encode.module_ m) with
      | Ok s -> s
      | Error _ ->
          invalid_arg "Wasm_features.probe: the probe module is invalid")
