module W = Wire

type t = int

let of_int i = i
let to_int p = p
let compare = Int.compare
let equal = Int.equal
let decode : t W.decoder = W.int
let encode = W.enc_int
