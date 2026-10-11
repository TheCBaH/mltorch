module W = Wire

type t = int

let of_index i = if i >= 0 && i <= 255 then Some i else None
let to_index a = a
let compare = Int.compare
let equal = Int.equal
let pp = Fmt.int

let decode : t W.decoder =
 fun j ->
  let ( let* ) = W.( let* ) in
  let* i = W.int j in
  match of_index i with
  | Some a -> Ok a
  | None -> W.fail (W.Constraint Out_of_range)

let encode = W.enc_int
