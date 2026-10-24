module Precision = struct
  type t = F32 | F64

  let name = function F32 -> "f32" | F64 -> "f64"
end

type t =
  | Fp32_contracted
  | Fp32_ordered
  | Reference_f64
  | Simd_fp32_ordered
  | Simd_fp32_relaxed

let all =
  [
    Fp32_contracted;
    Fp32_ordered;
    Reference_f64;
    Simd_fp32_ordered;
    Simd_fp32_relaxed;
  ]

let name = function
  | Fp32_contracted -> "fp32_contracted"
  | Fp32_ordered -> "fp32_ordered"
  | Reference_f64 -> "reference_f64"
  | Simd_fp32_ordered -> "simd_fp32_ordered"
  | Simd_fp32_relaxed -> "simd_fp32_relaxed"

let of_name s = List.find_opt (fun p -> String.equal (name p) s) all

module Contraction = struct
  type t = Dots | Everywhere | Forbidden

  let name = function
    | Dots -> "dots"
    | Everywhere -> "permitted"
    | Forbidden -> "none"
end

let contraction = function
  | Fp32_contracted -> Contraction.Dots
  | Fp32_ordered | Reference_f64 | Simd_fp32_ordered -> Contraction.Forbidden
  | Simd_fp32_relaxed -> Contraction.Everywhere

let contraction_permitted p = contraction p = Contraction.Everywhere

let reassociation_permitted = function
  | Fp32_contracted | Fp32_ordered | Reference_f64 | Simd_fp32_ordered -> false
  | Simd_fp32_relaxed -> true

let identity p =
  Printf.sprintf "numerics=%s vectorized=%s other=%s fma=%s reassociation=%s"
    (name p)
    (Precision.name
       (match p with
       | Reference_f64 -> Precision.F64
       | Fp32_contracted | Fp32_ordered | Simd_fp32_ordered | Simd_fp32_relaxed
         ->
           Precision.F32))
    (Precision.name
       (match p with
       | Fp32_contracted | Fp32_ordered -> Precision.F32
       | Reference_f64 | Simd_fp32_ordered | Simd_fp32_relaxed -> Precision.F64))
    (Contraction.name (contraction p))
    (if reassociation_permitted p then "permitted" else "none")

module Backend = struct
  type t = C | Interpreter | Javascript | Machine | Wasm

  let name = function
    | C -> "c"
    | Interpreter -> "interpreter"
    | Javascript -> "javascript"
    | Machine -> "machine"
    | Wasm -> "wasm"
end

type unsupported = [ `Unsupported_policy of Backend.t * t ]

let pp_unsupported ppf : [< unsupported ] -> unit = function
  | `Unsupported_policy (b, p) ->
      Format.fprintf ppf "the %s backend does not run the %s policy"
        (Backend.name b) (name p)

let check ~backend p =
  match (backend, p) with
  | _, Reference_f64 | Backend.Machine, _ -> Err.return p
  | (Backend.C | Backend.Wasm), Fp32_contracted ->
      Err.fail (`Unsupported_policy (backend, p))
  | (Backend.C | Backend.Wasm), _ -> Err.return p
  | (Backend.Interpreter | Backend.Javascript), _ ->
      Err.fail (`Unsupported_policy (backend, p))

module Refusal = struct
  type t = Wide_load of Tensor_id.t * string

  let pp ppf = function
    | Wide_load (id, f) ->
        Format.fprintf ppf "t%d: float read of a %s payload"
          (Tensor_id.to_int id) f
end

(* The first refusal in program order. A refusal found stops the walk.

   A float read of an [f64], [i32] or [i64] payload, an int64 converted to a
   float and an index converted to a float are all admitted: each is one rounding
   to binary32 of an exactly known value, in the oracle, in C and in Wasm alike
   (a [float] cast; [f32.convert_i64_s] and [f32.demote_f64] of an exact widening
   in Wasm, never an int64 through binary64 first). A quantized [i8] or [i16]
   read is the explicit operations over binary32 parameters ([q] to float, minus
   the zero point, times the scale, each rounded once). What stays refused is a
   payload with no binary32 decode: none today. *)
exception Refused of Refusal.t

let fmt_name (b : Loop_buffer.t) =
  let (Payload.Fmt f) = b.Loop_buffer.sg.Tensor_sig.fmt in
  Payload.fmt_name f

let check_load (b : Loop_buffer.t) =
  match fmt_name b with
  | "bf16" | "bool" | "f16" | "f32" | "f64" | "i16" | "i32" | "i64" | "i8" -> ()
  | f -> raise (Refused (Refusal.Wide_load (b.Loop_buffer.id, f)))

let rec expr : type a. a Loop_expr.t -> unit = function
  | Loop_expr.Array_get _ | Loop_expr.Const _ | Loop_expr.I64_const _
  | Loop_expr.I64_of_index _ | Loop_expr.Load_i64 _ | Loop_expr.Load_i64_flat _
  | Loop_expr.Temp _ ->
      ()
  | Loop_expr.Binary (_, a, b) | Loop_expr.Float_max (a, b) ->
      expr a;
      expr b
  | Loop_expr.Fma (a, b, c) ->
      expr a;
      expr b;
      expr c
  | Loop_expr.Float_to_i64 a | Loop_expr.Round_f32 a | Loop_expr.Unary (_, a) ->
      expr a
  | Loop_expr.I64_binary (_, a, b) ->
      expr a;
      expr b
  | Loop_expr.I64_to_float a -> expr a
  | Loop_expr.Load (b, _) | Loop_expr.Load_flat (b, _) -> check_load b
  | Loop_expr.Select (p, a, b) ->
      pred p;
      expr a;
      expr b
  | Loop_expr.Value_of_index _ -> ()

and pred : Loop_expr.pred -> unit = function
  | Loop_bool.I64_eq (a, b) | Loop_bool.I64_lt (a, b) ->
      expr a;
      expr b
  | Loop_bool.Pool_better (a, b)
  | Loop_bool.Value_eq (a, b)
  | Loop_bool.Value_lt (a, b) ->
      expr a;
      expr b
  | Loop_bool.Not p -> pred p
  | Loop_bool.Or (p, q) ->
      pred p;
      pred q
  | Loop_bool.Index_eq _ | Loop_bool.Index_lt _ | Loop_bool.Index_overflows _
  | Loop_bool.Out_of_range _ ->
      ()

let failure : Loop_failure.t -> unit = function
  | Loop_failure.Gather_out_of_range { raw; _ } -> expr raw
  | Loop_failure.I64_from_float { value } -> expr value
  | Loop_failure.I64_division_by_zero | Loop_failure.I64_division_overflow
  | Loop_failure.Index_overflow _ | Loop_failure.Load_out_of_range _
  | Loop_failure.Local_out_of_range _ | Loop_failure.Scan_lane_out_of_range _
  | Loop_failure.Scan_row_out_of_range _ ->
      ()

let stored : Loop_stored.t -> unit = function
  | Loop_stored.Bool e | Loop_stored.F32 e -> expr e
  | Loop_stored.I64 e -> expr e

let rec stmt : Loop_stmt.t -> unit = function
  | Loop_stmt.Array_set (_, _, e) -> expr e
  | Loop_stmt.Assign (_, _, e) -> expr e
  | Loop_stmt.Assign_index_of_i64 (_, e) -> expr e
  | Loop_stmt.Fail_if (p, f) ->
      pred p;
      failure f
  | Loop_stmt.For { body; _ } -> List.iter stmt body
  | Loop_stmt.Reduce_sum { body; term; _ } ->
      List.iter stmt body;
      expr term
  | Loop_stmt.If (p, yes, no) ->
      pred p;
      List.iter stmt yes;
      List.iter stmt no
  | Loop_stmt.Store { value; _ } | Loop_stmt.Store_flat { value; _ } ->
      stored value
  | Loop_stmt.Alloc _ | Loop_stmt.Assign_index _ | Loop_stmt.Charge_scan_update
  | Loop_stmt.Mark _ | Loop_stmt.Release_scan_state _
  | Loop_stmt.Reserve_scan_state _ | Loop_stmt.Reset_meter ->
      ()

let admit (p : Loop_program.t) =
  match List.iter stmt p.Loop_program.body with
  | () -> Ok ()
  | exception Refused r -> Error r

let coverage ~numerics ~f32_kernels ~kernels ~f32_invocations ~invocations
    ~refusals =
  Format.asprintf
    "numerics %s; binary32 kernels %d of %d (invocations %d of %d)%s"
    (name numerics) f32_kernels kernels f32_invocations invocations
    (String.concat ""
       (List.map
          (fun (r, n) -> Format.asprintf "; %d binary64: %a" n Refusal.pp r)
          refusals))

let round32 x = Int32.float_of_bits (Int32.bits_of_float x)

(* Above 2^53 a binary64 conversion would already round once. The magnitude's
   low 11 bits fold into a sticky bit instead, which leaves the 53-bit value
   exact and keeps every rounding decision binary32 needs. *)
let round32_of_i64 n =
  let neg = Int64.compare n 0L < 0 in
  let mag = if neg then Int64.neg n else n in
  let v =
    if Int64.compare (Int64.shift_right_logical mag 53) 0L = 0 then
      Int64.to_float mag
    else
      let sticky =
        if Int64.equal (Int64.logand mag 0x7FFL) 0L then 0L else 1L
      in
      let high = Int64.logor (Int64.shift_right_logical mag 11) sticky in
      Int64.to_float high *. 2048.
  in
  round32 (if neg then -.v else v)

let erf32 = F32_math.Value.erf

(* A C [float] constant reading back to exactly [round32 x]: the host rounds the
   binary64 value, never the decimal text a second time. *)
let f32_literal x =
  let x = round32 x in
  if Float.is_nan x then "NAN"
  else if x = Float.infinity then "INFINITY"
  else if x = Float.neg_infinity then "(-INFINITY)"
  else "(" ^ Printf.sprintf "%h" x ^ "f)"

(* Binary32 [fmaf], one rounding: the shared implementation. *)
let fma32 = F32_math.fma32
