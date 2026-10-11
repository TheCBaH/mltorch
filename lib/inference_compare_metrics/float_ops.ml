open Inference_compare_schema

(* Round a double to binary32 and back. *)
let r32 x = Int32.float_of_bits (Int32.bits_of_float x)

let round (a : Float_policy.arithmetic) x =
  match a with Binary32 -> r32 x | Binary64 -> x

let scale_of (s : Float_policy.scale) a b =
  match s with
  | Left_scale -> Float.abs a
  | Right_scale -> Float.abs b
  | Symmetric_max_scale -> Float.max (Float.abs a) (Float.abs b)

type klass =
  | Finite
  | Inf_left
  | Inf_opposite
  | Inf_right
  | Inf_same
  | Nan_both_different
  | Nan_both_same
  | Nan_left
  | Nan_right

(* Exceptional operands are classified before any subtraction or division. NaN
   outranks infinity: a NaN/infinity pair is a NaN pair. [Inf_left] is a left
   infinity against a finite right operand. *)
let classify a b ~abits ~bbits =
  let an = Float.is_nan a and bn = Float.is_nan b in
  if an || bn then
    if an && bn then
      if Int64.equal abits bbits then Nan_both_same else Nan_both_different
    else if an then Nan_left
    else Nan_right
  else
    let ai = Float.abs a = Float.infinity
    and bi = Float.abs b = Float.infinity in
    if ai || bi then
      if ai && bi then if a = b then Inf_same else Inf_opposite
      else if ai then Inf_left
      else Inf_right
    else Finite

let absolute a b : Difference_measurement.magnitude =
  let d = Float.abs (a -. b) in
  if Float.is_finite d then Finite d else Unavailable Arithmetic_overflow

(* Relative difference d/s in binary64. A zero denominator is 0/0 when the
   numerator is zero and unbounded otherwise. If d or d/s overflows, the ratio
   is recomputed from operands divided by the scale before it is declared an
   overflow; no epsilon replaces a tiny divisor. *)
let relative scale a b : Difference_measurement.magnitude =
  let s = scale_of scale a b in
  let d = Float.abs (a -. b) in
  if s = 0.0 then if d = 0.0 then Unavailable Zero_over_zero else Unbounded
  else
    let r = d /. s in
    if Float.is_finite r then Finite r
    else
      let r' = Float.abs ((a /. s) -. (b /. s)) in
      if Float.is_finite r' then Finite r' else Unavailable Arithmetic_overflow

(* Monotone integer ranks of the ordered float encodings, for ULP distance.
   With signed zeros equal both zeros share rank 0; distinct, -0 sits one rank
   below +0. [bits] is zero-extended to 64 bits for binary32. *)
let ulp_key ~width ~distinct_zero bits =
  let sign = if width = 4 then 0x8000_0000L else Int64.min_int in
  let magnitude = Int64.logand bits (Int64.lognot sign) in
  if Int64.equal (Int64.logand bits sign) 0L then magnitude
  else if distinct_zero then Int64.sub (Int64.neg magnitude) 1L
  else Int64.neg magnitude

(* Unsigned distance: the true difference is below 2^64, so the wrapped
   subtraction of the larger key from the smaller is exact as an unsigned. *)
let ulp_distance ~width ~distinct_zero abits bbits =
  let ka = ulp_key ~width ~distinct_zero abits
  and kb = ulp_key ~width ~distinct_zero bbits in
  if Int64.compare ka kb >= 0 then Int64.sub ka kb else Int64.sub kb ka

let unravel shape i =
  let dims = Array.of_list shape in
  let rank = Array.length dims in
  let idx = Array.make rank 0L in
  let rest = ref (Int64.of_int i) in
  for d = rank - 1 downto 0 do
    let n = dims.(d) in
    if Int64.compare n 0L > 0 then begin
      idx.(d) <- Int64.rem !rest n;
      rest := Int64.div !rest n
    end
  done;
  Array.to_list idx
