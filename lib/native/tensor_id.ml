type t = int

let of_int x = x
let to_int x = x
let equal = Int.equal
let compare = Int.compare
let pp fmt x = Format.fprintf fmt "t%d" x
let jsont = Jsont.map ~dec:of_int ~enc:to_int Jsont.int

module Map = Map.Make (struct
  type nonrec t = t

  let compare = compare
end)
