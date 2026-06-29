type t = float
type 'role index = int
type input = Tensor.packed

let const x = x
let add = ( +. )
let sub = ( -. )
let mul = ( *. )
let div = ( /. )
let exp = Stdlib.exp
let sqrt = Stdlib.sqrt

type b = bool

let lt a b = a < b
let select c a b = if c then a else b
let index_zero = 0
let index_extent (e : Dim.extent Dim.t) = (e :> int)
let index_const n = n
let of_index i = i
let index_add = ( + )
let index_scale k i = k * i
let index_min = Stdlib.min
let clamp_low x = Stdlib.max 0 x
let assume_index x = x
let load inp idx = Tensor.read_at inp idx

let sum ~lo ~hi f =
  let rec loop i acc = if i >= hi then acc else loop (i + 1) (acc +. f i) in
  loop lo 0.

let max_reduce ~lo ~hi f =
  let rec loop i acc =
    if i >= hi then acc else loop (i + 1) (Float.max acc (f i))
  in
  loop lo neg_infinity
