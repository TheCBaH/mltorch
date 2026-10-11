module W = Wire

type t = Left | Right

let all = [ Left; Right ]

let compare a b =
  match (a, b) with
  | Left, Left | Right, Right -> 0
  | Left, Right -> -1
  | Right, Left -> 1

let equal a b = compare a b = 0
let other = function Left -> Right | Right -> Left
let to_string = function Left -> "left" | Right -> "right"
let pp ppf s = Fmt.string ppf (to_string s)

let decode : t W.decoder =
 fun j ->
  let ( let* ) = W.( let* ) in
  let* s = W.string j in
  match s with
  | "left" -> Ok Left
  | "right" -> Ok Right
  | tag -> W.unknown_tag ~kind:"side" tag

let encode s = W.enc_string (to_string s)
