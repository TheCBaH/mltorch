(* A typed constant as raw bits. A float keeps its exact IEEE bits (binary32 in
   the low 32), an integer its canonical zero-extended bits, a predicate 0 or
   1. Nothing here rounds: a constructor from a host float rounds explicitly. *)
type t = { ty : Mir_type.t; bits : int64 }

let f64 x = { ty = Mir_type.F64; bits = Int64.bits_of_float x }

let f32_bits b =
  { ty = Mir_type.F32; bits = Int64.logand (Int64.of_int32 b) 0xFFFF_FFFFL }

(* [x] must already be a binary32 value; a value that is not rounds once. *)
let f32 x = f32_bits (Int32.bits_of_float x)
let int w x = { ty = Mir_type.Int w; bits = Mir_width.normalize w x }
let i64 x = int Mir_width.W64 x
let i32 x = int Mir_width.W32 x
let pred b = { ty = Mir_type.Pred; bits = (if b then 1L else 0L) }

(* The canonical-bits invariant the verifier checks. *)
let well_formed t =
  match t.ty with
  | Mir_type.F64 -> true
  | Mir_type.F32 -> Mir_width.canonical Mir_width.W32 t.bits
  | Mir_type.Int w -> Mir_width.canonical w t.bits
  | Mir_type.Pred -> Int64.equal t.bits 0L || Int64.equal t.bits 1L
  | Mir_type.Mask _ | Mir_type.Order | Mir_type.Ptr | Mir_type.Vec _ -> false

let equal a b = Mir_type.equal a.ty b.ty && Int64.equal a.bits b.bits

let pp fmt t =
  match t.ty with
  | Mir_type.F64 -> Fmt.pf fmt "%h:f64" (Int64.float_of_bits t.bits)
  | Mir_type.F32 ->
      Fmt.pf fmt "%h:f32" (Int32.float_of_bits (Int64.to_int32 t.bits))
  | Mir_type.Int w ->
      Fmt.pf fmt "%Ld:%a" (Mir_width.signed w t.bits) Mir_width.pp w
  | Mir_type.Pred -> Fmt.pf fmt "%b" (Int64.equal t.bits 1L)
  | Mir_type.Mask _ | Mir_type.Order | Mir_type.Ptr | Mir_type.Vec _ ->
      Fmt.pf fmt "0x%Lx:%a" t.bits Mir_type.pp t.ty
