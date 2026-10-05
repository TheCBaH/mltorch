module Precision = struct
  type t = F32 | F64

  let name = function F32 -> "f32" | F64 -> "f64"
end

type t = Reference_f64 | Simd_fp32_ordered | Simd_fp32_relaxed

let all = [ Reference_f64; Simd_fp32_ordered; Simd_fp32_relaxed ]

let name = function
  | Reference_f64 -> "reference_f64"
  | Simd_fp32_ordered -> "simd_fp32_ordered"
  | Simd_fp32_relaxed -> "simd_fp32_relaxed"

let of_name s = List.find_opt (fun p -> String.equal (name p) s) all

let contraction_permitted = function
  | Reference_f64 | Simd_fp32_ordered -> false
  | Simd_fp32_relaxed -> true

let reassociation_permitted = contraction_permitted

let identity p =
  Printf.sprintf "numerics=%s vectorized=%s other=f64 fma=%s reassociation=%s"
    (name p)
    (Precision.name
       (match p with
       | Reference_f64 -> Precision.F64
       | Simd_fp32_ordered | Simd_fp32_relaxed -> Precision.F32))
    (if contraction_permitted p then "permitted" else "none")
    (if reassociation_permitted p then "permitted" else "none")

module Backend = struct
  type t = C | Interpreter | Javascript | Wasm

  let name = function
    | C -> "c"
    | Interpreter -> "interpreter"
    | Javascript -> "javascript"
    | Wasm -> "wasm"
end

type unsupported = [ `Unsupported_policy of Backend.t * t ]

let pp_unsupported ppf : [< unsupported ] -> unit = function
  | `Unsupported_policy (b, p) ->
      Format.fprintf ppf "the %s backend does not run the %s policy"
        (Backend.name b) (name p)

let check ~backend p =
  match (backend, p) with
  | _, Reference_f64 | (Backend.C | Backend.Wasm), _ -> Err.return p
  | (Backend.Interpreter | Backend.Javascript), _ ->
      Err.fail (`Unsupported_policy (backend, p))

module Refusal = struct
  type t = Wide_load of Ssa_id.Buffer.t * Ssa_format.Family.t

  let pp ppf = function
    | Wide_load (id, f) ->
        Format.fprintf ppf "%a: float read of a %s payload" Ssa_id.Buffer.pp id
          (Ssa_format.Family.name f)
end

(* The first refusal in program order: a dequantizing read has no binary32
   decode. Every other float read is one rounding of an exactly known value. *)
let admit (p : Ssa_program.t) =
  let exception Refused of Refusal.t in
  let refuse buffer decode =
    match decode with
    | Ssa_op.Decode.I16_dequant | Ssa_op.Decode.I8_dequant ->
        raise
          (Refused (Refusal.Wide_load (buffer, Ssa_op.Decode.family decode)))
    | Ssa_op.Decode.Bf16_to_f64 | Ssa_op.Decode.Bool_to_f64
    | Ssa_op.Decode.F16_to_f64 | Ssa_op.Decode.F32_to_f64
    | Ssa_op.Decode.F64_to_f64 | Ssa_op.Decode.I32_to_f64 | Ssa_op.Decode.I64
    | Ssa_op.Decode.I64_to_f64 ->
        ()
  in
  let rec region (r : Ssa_region.t) = List.iter stmt r.Ssa_region.body
  and stmt : Ssa_region.t Ssa_stmt.t -> unit = function
    | Ssa_stmt.Instr i -> (
        match i.Ssa_instr.op with
        | Ssa_op.Load { buffer; decode; _ }
        | Ssa_op.Load_in_bounds { buffer; decode; _ }
        | Ssa_op.Vec_load { buffer; decode; _ } ->
            refuse buffer decode
        | _ -> ())
    | Ssa_stmt.For { body; _ } | Ssa_stmt.Ordered_sum { body; _ } -> region body
    | Ssa_stmt.If { then_; else_; _ } ->
        region then_;
        region else_
  in
  match region p.Ssa_program.entry with
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

let round32 = Ssa_const.round_f32

let erf32 x =
  let r = round32 in
  let p = r 0.3275911 and a1 = r 0.254829592 and a2 = r (-0.284496736) in
  let a3 = r 1.421413741 and a4 = r (-1.453152027) and a5 = r 1.061405429 in
  let sign = if x < 0. then -1. else 1. in
  let ax = Float.abs x in
  let t = r (1. /. r (1. +. r (p *. ax))) in
  let up a acc = r (a +. acc) and mul_t x = r (t *. x) in
  let poly =
    mul_t (up a1 (mul_t (up a2 (mul_t (up a3 (mul_t (up a4 (mul_t a5))))))))
  in
  let e = r (exp (-.r (ax *. ax))) in
  r (sign *. r (1. -. r (poly *. e)))

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
