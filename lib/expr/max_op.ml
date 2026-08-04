(* See max_op.mli. *)

type t = Float_max | Pool_max

let pool_better ~best ~value = value > best || Float.is_nan value

let apply op a b =
  match op with
  | Float_max -> Float.max a b
  | Pool_max -> if pool_better ~best:a ~value:b then b else a
