module type S = sig
  type t = private int64

  val of_int64 : int64 -> t
  val to_int64 : t -> int64
  val zero : t
  val add : t -> t -> t
  val compare : t -> t -> int
  val equal : t -> t -> bool
  val pp : t Fmt.t
end

module Make () : S = struct
  type t = int64

  let of_int64 n =
    if n < 0L then invalid_arg "Coral_capacity.Sized.of_int64: negative size"
    else n

  let to_int64 n = n
  let zero = 0L

  let add a b =
    let s = Int64.add a b in
    if s < a then invalid_arg "Coral_capacity.Sized.add: overflow" else s

  let compare = Int64.compare
  let equal = Int64.equal
  let pp = Fmt.int64
end
