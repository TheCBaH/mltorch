type t = float
type 'role ix = int
type input = Tensor.packed

let const x = x
let add = ( +. )
let sub = ( -. )
let mul = ( *. )
let div = ( /. )
let exp = Stdlib.exp
let sqrt = Stdlib.sqrt

type b = bool

let lt (a : float) (b : float) = a < b
let select c a b = if c then a else b
let izero = 0
let iext (e : Dim.extent Dim.t) = (e :> int)
let iconst n = n
let of_index i = i
let iadd = ( + )
let iscale k i = k * i
let imin = Stdlib.min
let clamp_low x = Stdlib.max 0 x
let assume_index x = x
let load inp (idx : Axis.t -> int) : float = Tensor.read_guarded inp idx

let sum ~lo ~hi f =
  let rec loop i acc = if i >= hi then acc else loop (i + 1) (acc +. f i) in
  loop lo 0.

let maxr ~lo ~hi f =
  let rec loop i acc =
    if i >= hi then acc else loop (i + 1) (Float.max acc (f i))
  in
  loop lo neg_infinity
