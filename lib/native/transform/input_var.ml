(* See input_var.mli. *)

type t = int

let compare = Int.compare
let equal = Int.equal
let of_int i = i
let pp fmt v = Fmt.pf fmt "v%d" v
let to_int v = v

module Map = Map.Make (Int)
