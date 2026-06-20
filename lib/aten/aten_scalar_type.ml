(* OCaml encoding of c10::ScalarType / atc_scalar_type from shim.h. *)
type t = Byte | Char | Short | Int | Long | Half | Float | Double | Bool

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
