(* See axis.mli. *)

type t = N | T | D | H | W | C

let all = [ N; T; D; H; W; C ]
let to_int = function N -> 0 | T -> 1 | D -> 2 | H -> 3 | W -> 4 | C -> 5
let compare a b = Int.compare (to_int a) (to_int b)
let equal a b = to_int a = to_int b

let to_string = function
  | N -> "N"
  | T -> "T"
  | D -> "D"
  | H -> "H"
  | W -> "W"
  | C -> "C"

let pp fmt a = Fmt.string fmt (to_string a)
