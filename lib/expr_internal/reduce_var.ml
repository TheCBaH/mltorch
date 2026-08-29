(* Reducer identities are library-private integers.  The public [Expr]
   signature keeps the representation abstract. *)

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
