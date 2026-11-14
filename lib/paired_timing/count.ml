type t = int

let of_int n = if n < 0 then invalid_arg "Count.of_int: negative" else n
let to_int n = n
let pp = Fmt.int
