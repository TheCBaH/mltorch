(* See tagged_int.mli. *)

module type S = sig
  type t = private int

  val of_int : int -> t
  val to_int : t -> int
  val equal : t -> t -> bool
  val compare : t -> t -> int
  val succ : t -> t
  val pp : Format.formatter -> t -> unit

  module Map : Map.S with type key = t
  module Set : Set.S with type elt = t

  module Next : sig
    type id = t
    type t = private int

    val first : t
    val of_int : int -> t
    val after : id -> t -> t
    val alloc : t -> id * t
    val check_room : t -> count:int -> unit
    val alloc_n : t -> int -> id list * t
    val reaches : t -> id -> bool
    val equal : t -> t -> bool
    val compare : t -> t -> int
    val pp : Format.formatter -> t -> unit
  end
end

module Make
    (P : sig
      val prefix : string
    end)
    () : S = struct
  type t = int

  let of_int x = x
  let to_int x = x
  let equal = Int.equal
  let compare = Int.compare
  let succ x = x + 1
  let pp ppf x = Format.fprintf ppf "%s%d" P.prefix x

  module Ord = struct
    type nonrec t = t

    let compare = compare
  end

  module Map = Map.Make (Ord)
  module Set = Set.Make (Ord)

  module Next = struct
    type id = t
    type nonrec t = int

    let first = 0
    let of_int x = x
    let after id n = Stdlib.max n (id + 1)
    let alloc n = (n, n + 1)

    let check_room n ~count =
      if count < 0 || count > max_int - n then
        invalid_arg
          (Printf.sprintf
             "Tagged_int.Next.alloc_n: allocating %d %s ids from %d exhausts \
              the id space"
             count P.prefix n)

    let alloc_n n count =
      check_room n ~count;
      (List.init count (fun i -> n + i), n + count)

    let reaches n id = id >= n
    let equal = Int.equal
    let compare = Int.compare
    let pp ppf n = Format.fprintf ppf "%s%d" P.prefix n
  end
end
