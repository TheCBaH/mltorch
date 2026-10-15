module Role = struct
  type t = Arithmetic | Conversion | Helper_call | Memory_move

  let all = [ Arithmetic; Conversion; Helper_call; Memory_move ]

  let name = function
    | Arithmetic -> "arithmetic"
    | Conversion -> "conversion"
    | Helper_call -> "helper_call"
    | Memory_move -> "memory_move"
end

(* Every operation is listed, so a new one is a compile error until it is
   classified. An operation touches [f64] exactly when its mnemonic says so;
   the test suite checks that against each signature. *)
let role_of_op : Wasm_op.t -> Role.t option = function
  | Wasm_op.F64_abs | Wasm_op.F64_add | Wasm_op.F64_ceil | Wasm_op.F64_copysign
  | Wasm_op.F64_div | Wasm_op.F64_eq | Wasm_op.F64_floor | Wasm_op.F64_ge
  | Wasm_op.F64_gt | Wasm_op.F64_le | Wasm_op.F64_lt | Wasm_op.F64_max
  | Wasm_op.F64_min | Wasm_op.F64_mul | Wasm_op.F64_ne | Wasm_op.F64_nearest
  | Wasm_op.F64_neg | Wasm_op.F64_sqrt | Wasm_op.F64_sub | Wasm_op.F64_trunc
  | Wasm_op.F64x2_abs | Wasm_op.F64x2_add | Wasm_op.F64x2_ceil
  | Wasm_op.F64x2_div | Wasm_op.F64x2_eq | Wasm_op.F64x2_floor
  | Wasm_op.F64x2_ge | Wasm_op.F64x2_gt | Wasm_op.F64x2_le | Wasm_op.F64x2_lt
  | Wasm_op.F64x2_max | Wasm_op.F64x2_min | Wasm_op.F64x2_mul | Wasm_op.F64x2_ne
  | Wasm_op.F64x2_nearest | Wasm_op.F64x2_neg | Wasm_op.F64x2_sqrt
  | Wasm_op.F64x2_sub | Wasm_op.F64x2_trunc ->
      Some Role.Arithmetic
  | Wasm_op.F32_demote_f64 | Wasm_op.F32x4_demote_f64x2_zero
  | Wasm_op.F64_convert_i32_s | Wasm_op.F64_convert_i32_u
  | Wasm_op.F64_convert_i64_s | Wasm_op.F64_convert_i64_u
  | Wasm_op.F64_promote_f32 | Wasm_op.F64x2_convert_low_i32x4_s
  | Wasm_op.F64x2_convert_low_i32x4_u | Wasm_op.F64x2_promote_low_f32x4
  | Wasm_op.I32_trunc_sat_f64_s | Wasm_op.I32_trunc_sat_f64_u
  | Wasm_op.I32x4_trunc_sat_f64x2_s_zero | Wasm_op.I32x4_trunc_sat_f64x2_u_zero
  | Wasm_op.I64_trunc_sat_f64_s | Wasm_op.I64_trunc_sat_f64_u ->
      Some Role.Conversion
  | Wasm_op.F64_reinterpret_i64 | Wasm_op.F64x2_splat
  | Wasm_op.I64_reinterpret_f64 ->
      Some Role.Memory_move
  | Wasm_op.F32_abs | Wasm_op.F32_add | Wasm_op.F32_convert_i32_s
  | Wasm_op.F32_convert_i32_u | Wasm_op.F32_convert_i64_s | Wasm_op.F32_div
  | Wasm_op.F32_eq | Wasm_op.F32_ge | Wasm_op.F32_gt | Wasm_op.F32_le
  | Wasm_op.F32_lt | Wasm_op.F32_max | Wasm_op.F32_min | Wasm_op.F32_mul
  | Wasm_op.F32_ne | Wasm_op.F32_neg | Wasm_op.F32_reinterpret_i32
  | Wasm_op.F32_sqrt | Wasm_op.F32_sub | Wasm_op.F32_trunc | Wasm_op.F32x4_abs
  | Wasm_op.F32x4_add | Wasm_op.F32x4_div | Wasm_op.F32x4_eq | Wasm_op.F32x4_ge
  | Wasm_op.F32x4_gt | Wasm_op.F32x4_le | Wasm_op.F32x4_lt | Wasm_op.F32x4_max
  | Wasm_op.F32x4_min | Wasm_op.F32x4_mul | Wasm_op.F32x4_ne | Wasm_op.F32x4_neg
  | Wasm_op.F32x4_relaxed_madd | Wasm_op.F32x4_splat | Wasm_op.F32x4_sqrt
  | Wasm_op.F32x4_sub | Wasm_op.F32x4_trunc | Wasm_op.I32_add | Wasm_op.I32_and
  | Wasm_op.I32_clz | Wasm_op.I32_ctz | Wasm_op.I32_div_s | Wasm_op.I32_div_u
  | Wasm_op.I32_eq | Wasm_op.I32_eqz | Wasm_op.I32_extend16_s
  | Wasm_op.I32_extend8_s | Wasm_op.I32_ge_s | Wasm_op.I32_ge_u
  | Wasm_op.I32_gt_s | Wasm_op.I32_gt_u | Wasm_op.I32_le_s | Wasm_op.I32_le_u
  | Wasm_op.I32_lt_s | Wasm_op.I32_lt_u | Wasm_op.I32_mul | Wasm_op.I32_ne
  | Wasm_op.I32_or | Wasm_op.I32_popcnt | Wasm_op.I32_reinterpret_f32
  | Wasm_op.I32_rem_s | Wasm_op.I32_rem_u | Wasm_op.I32_shl | Wasm_op.I32_shr_s
  | Wasm_op.I32_shr_u | Wasm_op.I32_sub | Wasm_op.I32_wrap_i64 | Wasm_op.I32_xor
  | Wasm_op.I32x4_splat | Wasm_op.I64_add | Wasm_op.I64_and | Wasm_op.I64_div_s
  | Wasm_op.I64_div_u | Wasm_op.I64_eq | Wasm_op.I64_eqz
  | Wasm_op.I64_extend_i32_s | Wasm_op.I64_extend_i32_u | Wasm_op.I64_ge_s
  | Wasm_op.I64_ge_u | Wasm_op.I64_gt_s | Wasm_op.I64_gt_u | Wasm_op.I64_le_s
  | Wasm_op.I64_le_u | Wasm_op.I64_lt_s | Wasm_op.I64_lt_u | Wasm_op.I64_mul
  | Wasm_op.I64_ne | Wasm_op.I64_or | Wasm_op.I64_rem_s | Wasm_op.I64_rem_u
  | Wasm_op.I64_shl | Wasm_op.I64_shr_s | Wasm_op.I64_shr_u | Wasm_op.I64_sub
  | Wasm_op.I64_xor | Wasm_op.I64x2_add | Wasm_op.I64x2_eq | Wasm_op.I64x2_ge_s
  | Wasm_op.I64x2_gt_s | Wasm_op.I64x2_le_s | Wasm_op.I64x2_lt_s
  | Wasm_op.I64x2_mul | Wasm_op.I64x2_ne | Wasm_op.I64x2_shl
  | Wasm_op.I64x2_shr_s | Wasm_op.I64x2_shr_u | Wasm_op.I64x2_splat
  | Wasm_op.I64x2_sub | Wasm_op.V128_and | Wasm_op.V128_andnot
  | Wasm_op.V128_any_true | Wasm_op.V128_bitselect | Wasm_op.V128_not
  | Wasm_op.V128_or | Wasm_op.V128_xor ->
      None

module String_map = Map.Make (String)

type t = {
  arithmetic : int;
  conversion : int;
  helper_calls : int String_map.t;
  imports : string list;
  memory_move : int;
}

let empty =
  {
    arithmetic = 0;
    conversion = 0;
    helper_calls = String_map.empty;
    imports = [];
    memory_move = 0;
  }

let touches_f64 (ty : Wasm_type.t) =
  match ty with
  | Wasm_type.F64 -> true
  | Wasm_type.F32 | Wasm_type.I32 | Wasm_type.I64 | Wasm_type.V128 -> false

let func_type_touches (ft : Wasm.Func_type.t) =
  List.exists touches_f64 ft.Wasm.Func_type.params
  || List.exists touches_f64 ft.Wasm.Func_type.results

let import_name (i : Wasm.Import.t) =
  i.Wasm.Import.module_name ^ "." ^ i.Wasm.Import.name

let bump_call name t =
  let n = Option.value ~default:0 (String_map.find_opt name t.helper_calls) in
  { t with helper_calls = String_map.add name (n + 1) t.helper_calls }

let move t = { t with memory_move = t.memory_move + 1 }

(* The callees of the module's function index space: imports, then functions. *)
type env = {
  callees : (string * bool) array;
  locals : Wasm_type.t array;
  globals : Wasm_type.t array;
}

let rec of_instr env t (i : Wasm.Instr.t) =
  match i with
  | Wasm.Instr.Block (_, l) | Wasm.Instr.Loop (_, l) ->
      List.fold_left (of_instr env) t l
  | Wasm.Instr.If (_, a, b) ->
      List.fold_left (of_instr env) (List.fold_left (of_instr env) t a) b
  | Wasm.Instr.Call n -> (
      match env.callees.(n) with
      | name, true -> bump_call name t
      | _, false -> t)
  | Wasm.Instr.F64_const _ -> move t
  | Wasm.Instr.Global_get n | Wasm.Instr.Global_set n ->
      if touches_f64 env.globals.(n) then move t else t
  | Wasm.Instr.Load (l, _) ->
      if touches_f64 (Wasm.Load.value_type l) then move t else t
  | Wasm.Instr.Local_get n | Wasm.Instr.Local_set n | Wasm.Instr.Local_tee n ->
      if touches_f64 env.locals.(n) then move t else t
  | Wasm.Instr.Numeric op -> (
      match role_of_op op with
      | Some Role.Arithmetic -> { t with arithmetic = t.arithmetic + 1 }
      | Some Role.Conversion -> { t with conversion = t.conversion + 1 }
      | Some Role.Memory_move -> move t
      | Some Role.Helper_call | None -> t)
  | Wasm.Instr.Simd_lane (l, _) -> (
      match Wasm.Simd_lane.scalar l with
      | Wasm_type.F64 -> move t
      | Wasm_type.F32 | Wasm_type.I32 | Wasm_type.I64 | Wasm_type.V128 -> t)
  | Wasm.Instr.Store (s, _) ->
      if touches_f64 (Wasm.Store.value_type s) then move t else t
  | Wasm.Instr.Br _ | Wasm.Instr.Br_if _ | Wasm.Instr.Drop
  | Wasm.Instr.F32_const _ | Wasm.Instr.I32_const _ | Wasm.Instr.I64_const _
  | Wasm.Instr.Memory_copy | Wasm.Instr.Memory_fill | Wasm.Instr.Return
  | Wasm.Instr.Select | Wasm.Instr.Simd_load _ | Wasm.Instr.Simd_store _
  | Wasm.Instr.Unreachable | Wasm.Instr.V128_const _ ->
      (* [Select] and [Drop] are untyped here; the producing instruction is
         what the census counts. A vector load or store moves lanes of a type
         only its consumer knows. *)
      t

let of_module (m : Wasm.Module.t) =
  let imports = m.Wasm.Module.imports and funcs = m.Wasm.Module.funcs in
  let callees =
    Array.of_list
      (List.map
         (fun (i : Wasm.Import.t) ->
           (import_name i, func_type_touches i.Wasm.Import.type_))
         imports
      @ List.mapi
          (fun k (f : Wasm.Func.t) ->
            (Printf.sprintf "func%d" k, func_type_touches f.Wasm.Func.type_))
          funcs)
  in
  let globals =
    Array.of_list
      (List.map
         (fun (g : Wasm.Global.t) -> g.Wasm.Global.type_)
         m.Wasm.Module.globals)
  in
  let t =
    List.fold_left
      (fun t (f : Wasm.Func.t) ->
        let locals =
          Array.of_list
            (f.Wasm.Func.type_.Wasm.Func_type.params @ f.Wasm.Func.locals)
        in
        List.fold_left
          (of_instr { callees; locals; globals })
          t f.Wasm.Func.body)
      empty funcs
  in
  { t with imports = List.map import_name imports }

let helpers t = String_map.bindings t.helper_calls
let imports t = t.imports

let count t = function
  | Role.Arithmetic -> t.arithmetic
  | Role.Conversion -> t.conversion
  | Role.Helper_call ->
      String_map.fold (fun _ n acc -> acc + n) t.helper_calls 0
  | Role.Memory_move -> t.memory_move

let total t = List.fold_left (fun acc r -> acc + count t r) 0 Role.all

let pp ppf t =
  Format.fprintf ppf
    "arithmetic %d, conversion %d, helper_call %d, memory_move %d; imports %d"
    t.arithmetic t.conversion (count t Role.Helper_call) t.memory_move
    (List.length t.imports)
