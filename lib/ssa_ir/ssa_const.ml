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

(* Binary32 of an int64 with a single rounding. Above 2^53 a binary64 conversion
   would already round once: the magnitude's low 11 bits fold into a sticky bit
   instead, which leaves the 53-bit value exact and keeps every rounding decision
   binary32 needs. *)
let round32_of_i64 n =
  let neg = Int64.compare n 0L < 0 in
  let mag = if neg then Int64.neg n else n in
  let v =
    if Int64.equal (Int64.shift_right_logical mag 53) 0L then Int64.to_float mag
    else
      let sticky =
        if Int64.equal (Int64.logand mag 0x7FFL) 0L then 0L else 1L
      in
      let high = Int64.logor (Int64.shift_right_logical mag 11) sticky in
      Int64.to_float high *. 2048.
  in
  round_f32 (if neg then -.v else v)

let pp fmt = function
  | F32 x -> Fmt.pf fmt "%h:f32" x
  | F64 x -> Fmt.pf fmt "%h:f64" x
  | I64 x -> Fmt.pf fmt "%Ld:i64" x
  | Index x -> Fmt.pf fmt "%Ld:index" x
  | Pred b -> Fmt.pf fmt "%b:pred" b
