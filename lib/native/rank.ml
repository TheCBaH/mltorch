type t = int

let of_int n = if n < 0 then invalid_arg "Rank.of_int: negative" else n
let of_list l = List.length l
let of_array a = Array.length a
let to_int n = n
let equal = Int.equal
let compare = Int.compare
let pp = Format.pp_print_int

let jsont =
  Jsont.map ~kind:"rank"
    ~dec:(fun n ->
      if n < 0 then Jsont.Error.msgf Jsont.Meta.none "rank: negative, got %d" n
      else n)
    ~enc:to_int Jsont.int
