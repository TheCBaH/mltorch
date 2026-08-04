(* See source.mli. *)

type t = int

let create n = n
let to_int n = n
let compare = Int.compare
let equal = Int.equal
let hash n = n
let pp fmt n = Fmt.pf fmt "t%d" n

module Ord = struct
  type nonrec t = t

  let compare = compare
end

module Map = Map.Make (Ord)
module Set = Set.Make (Ord)
