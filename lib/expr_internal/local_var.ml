(* Scalar-local identities share Builder's ordinal supply with reducer
   identities, but have a distinct type and namespace. *)

type t = int

let compare = Int.compare
let equal = Int.equal
let hash n = n
let pp fmt n = Fmt.pf fmt "#%d" n

module Ord = struct
  type nonrec t = t

  let compare = compare
end

module Map = Map.Make (Ord)
module Set = Set.Make (Ord)
