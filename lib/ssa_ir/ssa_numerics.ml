module Precision = struct
  type t = F32 | F64

  let name = function F32 -> "f32" | F64 -> "f64"
  let scalar = function F32 -> Ssa_type.F32 | F64 -> Ssa_type.F64
end

type t = Fp32_ordered | Reference_f64 | Simd_fp32_ordered | Simd_fp32_relaxed

let all = [ Fp32_ordered; Reference_f64; Simd_fp32_ordered; Simd_fp32_relaxed ]

let name = function
  | Fp32_ordered -> "fp32_ordered"
  | Reference_f64 -> "reference_f64"
  | Simd_fp32_ordered -> "simd_fp32_ordered"
  | Simd_fp32_relaxed -> "simd_fp32_relaxed"

let of_name s = List.find_opt (fun p -> String.equal (name p) s) all

let contraction_permitted = function
  | Fp32_ordered | Reference_f64 | Simd_fp32_ordered -> false
  | Simd_fp32_relaxed -> true

let reassociation_permitted = contraction_permitted

let identity p =
  Printf.sprintf "numerics=%s vectorized=%s other=%s fma=%s reassociation=%s"
    (name p)
    (Precision.name
       (match p with
       | Reference_f64 -> Precision.F64
       | Fp32_ordered | Simd_fp32_ordered | Simd_fp32_relaxed -> Precision.F32))
    (Precision.name
       (match p with
       | Fp32_ordered -> Precision.F32
       | Reference_f64 | Simd_fp32_ordered | Simd_fp32_relaxed -> Precision.F64))
    (if contraction_permitted p then "permitted" else "none")
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
  | _, Reference_f64 | (Backend.C | Backend.Machine | Backend.Wasm), _ ->
      Err.return p
  | (Backend.Interpreter | Backend.Javascript), _ ->
      Err.fail (`Unsupported_policy (backend, p))

module Refusal = struct
  type t = Wide_load of Ssa_id.Buffer.t * Ssa_format.Family.t

  let pp ppf = function
    | Wide_load (id, f) ->
        Format.fprintf ppf "%a: float read of a %s payload" Ssa_id.Buffer.pp id
          (Ssa_format.Family.name f)
end

(* Nothing is refused today: every float read is one rounding of an exactly
   known value ([f16], [bf16], [bool], [f32], [f64], [i32], [i64]) or the
   explicit operations of a dequantization over binary32 parameters ([i8],
   [i16]: [q] to float, minus the zero point, times the scale, each rounded
   once). The walk and the refusal type stay for a payload that has no binary32
   decode: it would be refused here, and every consumer already reports it. *)
let admit (_ : Ssa_program.t) : (unit, Refusal.t) result = Ok ()

let wide_values (p : Ssa_program.t) =
  let out = ref [] in
  let wide (v : Ssa_value.t) =
    match v.Ssa_value.ty with
    | Ssa_type.Scalar Ssa_type.F64 | Ssa_type.Vec (Ssa_type.F64, _) -> true
    | Ssa_type.Effect | Ssa_type.Local | Ssa_type.Mask _ | Ssa_type.Scalar _
    | Ssa_type.Vec _ ->
        false
  in
  let rec region (r : Ssa_region.t) = List.iter stmt r.Ssa_region.body
  and stmt : Ssa_region.t Ssa_stmt.t -> unit = function
    | Ssa_stmt.Instr i ->
        List.iter
          (fun v -> if wide v then out := (v, i.Ssa_instr.op) :: !out)
          i.Ssa_instr.results
    | Ssa_stmt.For { body; _ } | Ssa_stmt.Ordered_sum { body; _ } -> region body
    | Ssa_stmt.If { then_; else_; _ } ->
        region then_;
        region else_
  in
  region p.Ssa_program.entry;
  List.rev !out

let pp_wide ppf l =
  Fmt.(list ~sep:(any ", "))
    (fun ppf ((v : Ssa_value.t), op) ->
      Fmt.pf ppf "%a : %a = %s" Ssa_id.Value.pp v.Ssa_value.id Ssa_type.pp
        v.Ssa_value.ty (Ssa_op.name op))
    ppf l

let coverage ~numerics ~f32_kernels ~kernels ~f32_invocations ~invocations
    ~refusals =
  Format.asprintf
    "numerics %s; binary32 kernels %d of %d (invocations %d of %d)%s"
    (name numerics) f32_kernels kernels f32_invocations invocations
    (String.concat ""
       (List.map
          (fun (r, n) -> Format.asprintf "; %d binary64: %a" n Refusal.pp r)
          refusals))

let round32 = Ssa_const.round_f32
let erf32 = F32_math.Value.erf

(* The product of two binary32 values is exact in binary64 (48 bits). Its sum
   with [c] is split exactly by TwoSum into [s + e]; rounding [s + e] straight
   to binary32 is correct only if [s] carries a sticky bit, so when [e] is
   nonzero [s] is first replaced by the neighbouring binary64 whose last
   mantissa bit is odd (round to odd): binary64 has more than [2 * 24 + 2] bits,
   so that second rounding never changes the answer. Nonfinite operands fall
   back to ordinary arithmetic, which is already IEEE for them. *)
let fma32 a b c =
  let finite x = Float.is_finite x in
  if not (finite a && finite b && finite c) then round32 ((a *. b) +. c)
  else
    let p = a *. b in
    let s = p +. c in
    let bb = s -. p in
    let e = p -. (s -. bb) +. (c -. bb) in
    if e = 0. then round32 s
    else
      let bits = Int64.bits_of_float s in
      let odd =
        if Int64.logand bits 1L = 1L then s
        else
          let toward_larger_magnitude = e > 0. = (s > 0.) in
          Int64.float_of_bits
            (if toward_larger_magnitude then Int64.succ bits
             else Int64.pred bits)
      in
      round32 odd
