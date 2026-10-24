module Precision = struct
  type t = F32 | F64

  let name = function F32 -> "f32" | F64 -> "f64"
  let scalar = function F32 -> Ssa_type.F32 | F64 -> Ssa_type.F64
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

let erf32 = F32_math.Value.erf

(* Binary32 [fmaf], one rounding: the shared implementation. *)
let fma32 = F32_math.fma32
