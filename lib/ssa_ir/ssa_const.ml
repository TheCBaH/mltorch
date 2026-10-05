(* A typed constant with its exact bits: a float keeps its IEEE bits (a NaN is
   canonical only under comparison), an integer its full width. An [F32] holds
   a value already representable in binary32. *)
type t =
  | F32 of float
  | F64 of float
  | I64 of int64
  | Index of int64
  | Pred of bool

let ty = function
  | F32 _ -> Ssa_type.Scalar Ssa_type.F32
  | F64 _ -> Ssa_type.Scalar Ssa_type.F64
  | I64 _ -> Ssa_type.Scalar Ssa_type.I64
  | Index _ -> Ssa_type.Scalar Ssa_type.Index
  | Pred _ -> Ssa_type.Scalar Ssa_type.Pred

(* Bits, so the signed zeros and distinct NaNs of a float stay distinct for
   identity (CSE, the printer). All NaNs share one class: a JavaScript engine
   does not preserve a payload. *)
let equal a b =
  match (a, b) with
  | F32 x, F32 y | F64 x, F64 y -> Core.Float_bits.equal_portable x y
  | I64 x, I64 y | Index x, Index y -> Int64.equal x y
  | Pred x, Pred y -> Bool.equal x y
  | (F32 _ | F64 _ | I64 _ | Index _ | Pred _), _ -> false

(* The inclusive index domain: a valid language position. *)
let index_min = -0x8000_0000L
let index_max = 0x7FFF_FFFFL

let in_index_domain x =
  Int64.compare x index_min >= 0 && Int64.compare x index_max <= 0

let round_f32 x = Int32.float_of_bits (Int32.bits_of_float x)

(* Binary32 of an int64 with a single rounding, ties to even, from the integer
   alone: the magnitude's top 24 bits, then the discarded bits against a half.
   Above 2^53 a binary64 conversion would already round once; nothing here
   converts to a float until the value fits in 24 bits. *)
let round32_of_i64 n =
  if Int64.equal n 0L then 0.
  else
    let neg = Int64.compare n 0L < 0 in
    (* min_int's magnitude is 2^63, a power of two, exact as unsigned *)
    let mag = if neg then Int64.neg n else n in
    let bits = ref 0 in
    (let m = ref mag in
     while not (Int64.equal !m 0L) do
       incr bits;
       m := Int64.shift_right_logical !m 1
     done);
    let value =
      if !bits <= 24 then Int64.to_float mag
      else
        let drop = !bits - 24 in
        let q = Int64.shift_right_logical mag drop in
        let rem = Int64.logand mag (Int64.pred (Int64.shift_left 1L drop)) in
        let half = Int64.shift_left 1L (drop - 1) in
        let c = Int64.unsigned_compare rem half in
        let q =
          if c > 0 || (c = 0 && Int64.equal (Int64.logand q 1L) 1L) then
            Int64.succ q
          else q
        in
        Int64.to_float q *. Float.pow 2. (float_of_int drop)
    in
    if neg then -.value else value

let pp fmt = function
  | F32 x -> Fmt.pf fmt "%h:f32" x
  | F64 x -> Fmt.pf fmt "%h:f64" x
  | I64 x -> Fmt.pf fmt "%Ld:i64" x
  | Index x -> Fmt.pf fmt "%Ld:index" x
  | Pred b -> Fmt.pf fmt "%b:pred" b
