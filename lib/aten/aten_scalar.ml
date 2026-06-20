(* OCaml encoding of c10::Scalar — PyTorch's type-erased single number (a 0-dim
   tensor whose dtype is only known at runtime). c10::Scalar is a tagged union;
   the tag's category (integral / floating) participates in result-dtype
   promotion, so the distinction is semantically load-bearing, not cosmetic —
   [int_tensor + Int 1] stays integral, [int_tensor + Float 1.] promotes.

   The value crosses the C ABI as [struct atc_scalar] (atg_shim.h): a [Tag.t]
   discriminant plus a union of an int64, a double, and a bool payload. The
   [scalar] / [scalar_opt] ctypes views in aten_operation_description.ml marshal it.

   Complex and symbolic scalars are out of scope (no CPU op in the bound set
   produces them). *)

type t = Int of int64 | Float of float | Bool of bool

(* OCaml image of [enum atc_scalar_tag] (atg_shim.h): which union arm is live.
   [None_] is only produced by the optional-Scalar encoding (an absent
   [Scalar?]); a present scalar is never tagged [None_]. *)
module Tag = struct
  type t = Int | Float | Bool | None_

  let to_int = function Int -> 0 | Float -> 1 | Bool -> 2 | None_ -> 3

  let of_int = function
    | 0 -> Int
    | 1 -> Float
    | 2 -> Bool
    | 3 -> None_
    | n -> Printf.ksprintf failwith "unknown atc_scalar_tag %d" n
end

(* The live tag of a present scalar (never [None_]). *)
let tag : t -> Tag.t = function
  | Int _ -> Tag.Int
  | Float _ -> Tag.Float
  | Bool _ -> Tag.Bool
