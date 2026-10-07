(* An integer width. An operation says how it interprets the bits (signed or
   unsigned); the width alone names no signedness. *)
type t = W16 | W32 | W64 | W8

let all = [ W16; W32; W64; W8 ]
let bits = function W8 -> 8 | W16 -> 16 | W32 -> 32 | W64 -> 64
let bytes = function W8 -> 1L | W16 -> 2L | W32 -> 4L | W64 -> 8L
let equal (a : t) b = a = b
let name t = "i" ^ string_of_int (bits t)

(* The bits a value of this width may hold: canonical values are
   zero-extended into an [int64]. *)
let mask = function
  | W8 -> 0xFFL
  | W16 -> 0xFFFFL
  | W32 -> 0xFFFF_FFFFL
  | W64 -> -1L

let canonical t x = Int64.equal (Int64.logand x (mask t)) x
let normalize t x = Int64.logand x (mask t)

(* The signed reading of canonical bits. *)
let signed t x =
  match t with
  | W64 -> x
  | W8 | W16 | W32 ->
      let shift = 64 - bits t in
      Int64.shift_right (Int64.shift_left x shift) shift

let min_signed t =
  match t with
  | W64 -> Int64.min_int
  | _ -> Int64.neg (Int64.shift_left 1L (bits t - 1))

let max_signed t =
  match t with
  | W64 -> Int64.max_int
  | _ -> Int64.pred (Int64.shift_left 1L (bits t - 1))

let pp fmt t = Fmt.string fmt (name t)
