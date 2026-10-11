module W = Wire

module type S = sig
  type t

  val of_index : int -> t option
  val to_index : t -> int
  val compare : t -> t -> int
  val equal : t -> t -> bool
  val pp : Format.formatter -> t -> unit
  val decode : t W.decoder
  val encode : t W.encoder
end

let max_index = 65535

module Make () : S = struct
  type t = int

  let of_index i = if i >= 0 && i <= max_index then Some i else None
  let to_index p = p
  let compare = Int.compare
  let equal = Int.equal
  let pp = Fmt.int

  let decode : t W.decoder =
   fun j ->
    let ( let* ) = W.( let* ) in
    let* i = W.int j in
    match of_index i with
    | Some p -> Ok p
    | None -> W.fail (W.Constraint Out_of_range)

  let encode p = W.enc_int p
end
