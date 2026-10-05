(* The scalar domains. [Index] is the language's valid 32-bit position,
   [I64] an exact two's-complement integer, and [Offset] a native byte offset
   that exists only for a layout boundary. [F32] and [F64] are different types
   even where an interpreter holds both in an OCaml [float]. *)
type scalar = F32 | F64 | I64 | Index | Offset | Pred

(* The lane count of a vector or mask. A domain of its own: no extent, position
   or id can be passed for it. *)
module Lanes =
  Core.Tagged_int.Make
    (struct
      let prefix = "x"
    end)
    ()

(* [Effect] is a compile-time sequencing token with no machine representation. *)
type t =
  | Effect
  | Local
  | Mask of Lanes.t
  | Scalar of scalar
  | Vec of scalar * Lanes.t

let scalar_name = function
  | F32 -> "f32"
  | F64 -> "f64"
  | I64 -> "i64"
  | Index -> "index"
  | Offset -> "offset"
  | Pred -> "pred"

let equal a b =
  match (a, b) with
  | Effect, Effect | Local, Local -> true
  | Mask a, Mask b -> Lanes.equal a b
  | Scalar a, Scalar b -> a = b
  | Vec (s, a), Vec (t, b) -> s = t && Lanes.equal a b
  | (Effect | Local | Mask _ | Scalar _ | Vec _), _ -> false

let pp fmt = function
  | Effect -> Fmt.string fmt "effect"
  | Local -> Fmt.string fmt "local"
  | Mask l -> Fmt.pf fmt "mask<%a>" Lanes.pp l
  | Scalar s -> Fmt.string fmt (scalar_name s)
  | Vec (s, l) -> Fmt.pf fmt "vec<%a,%s>" Lanes.pp l (scalar_name s)

(* The phantom markers of the builder's typed handles. *)
type chain = Chain_marker
type f32 = F32_marker
type f64 = F64_marker
type i64 = I64_marker
type index = Index_marker
type local = Local_marker
type offset = Offset_marker
type pred = Pred_marker
