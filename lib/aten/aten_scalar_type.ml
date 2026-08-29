(* OCaml encoding of c10::ScalarType / atc_scalar_type from shim.h.

   [to_string] is alphabetical. The numeric conversion tables retain c10's
   stable enum values. *)
type t = Bool | Byte | Char | Double | Float | Half | Int | Long | Short

(** This conversion table deliberately follows c10's stable numeric enum values;
    do not alphabetize its branches. *)
let to_int = function
  | Byte -> 0
  | Char -> 1
  | Short -> 2
  | Int -> 3
  | Long -> 4
  | Half -> 5
  | Float -> 6
  | Double -> 7
  | Bool -> 11

let of_int = function
  | 0 -> Some Byte
  | 1 -> Some Char
  | 2 -> Some Short
  | 3 -> Some Int
  | 4 -> Some Long
  | 5 -> Some Half
  | 6 -> Some Float
  | 7 -> Some Double
  | 11 -> Some Bool
  | _ -> None

(* For diagnostics. The names are ATen's own spelling, so an error naming a
   dtype reads the same as the schema and the Python side do. *)
let to_string = function
  | Bool -> "Bool"
  | Byte -> "Byte"
  | Char -> "Char"
  | Double -> "Double"
  | Float -> "Float"
  | Half -> "Half"
  | Int -> "Int"
  | Long -> "Long"
  | Short -> "Short"
