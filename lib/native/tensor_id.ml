type t = int

let of_int x = x
let to_int x = x
let equal = Int.equal
let compare = Int.compare
let pp fmt x = Format.fprintf fmt "t%d" x
let jsont = Jsont.map ~dec:of_int ~enc:to_int Jsont.int

let check_room ~next ~count =
  if count < 0 || count > max_int - next then
    invalid_arg
      (Printf.sprintf
         "Tensor_id.check_room: allocating %d ids from %d exhausts the id space"
         count next)

module Ord = struct
  type nonrec t = t

  let compare = compare
end

module Map = Map.Make (Ord)
module Set = Set.Make (Ord)
