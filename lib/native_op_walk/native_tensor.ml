(* Synthesize a native input tensor of a given shape from the walk's PCG — the
   native analog of the ATen [Walk.tensor_spec]. Draws one f32-canonical
   uniform(-1,1) per element; the per-coord assignment is immaterial (Direct and
   Symbolic see the same tensor), so a flat counter over [materialize] suffices. *)

module Pcg = Walk_core.Pcg

let synth pcg (shape : Vec6.shape) : Tensor.packed * Pcg.t =
  let numel = (Vec6.numel shape :> int) in
  let rec draw k pcg acc =
    if k = 0 then (acc, pcg)
    else
      let v, pcg = Pcg.uniform ~low:(-1.) ~high:1. pcg in
      draw (k - 1) pcg (v :: acc)
  in
  let vals, pcg = draw numel pcg [] in
  let arr = Array.of_list vals in
  let i = ref 0 in
  let t =
    Tensor.materialize shape (fun _ ->
        let v = arr.(!i) in
        incr i;
        v)
  in
  (t, pcg)

(* [synth] via a caller-supplied per-element draw, rather than the fixed
   uniform(-1,1) -- for an operand whose domain rules out part of that range
   (see [synth_nonzero]/[synth_positive]). *)
let synth_with pcg (shape : Vec6.shape) draw_elt : Tensor.packed * Pcg.t =
  let numel = (Vec6.numel shape :> int) in
  let rec draw k pcg acc =
    if k = 0 then (acc, pcg)
    else
      let v, pcg = draw_elt pcg in
      draw (k - 1) pcg (v :: acc)
  in
  let vals, pcg = draw numel pcg [] in
  let arr = Array.of_list vals in
  let i = ref 0 in
  let t =
    Tensor.materialize shape (fun _ ->
        let v = arr.(!i) in
        incr i;
        v)
  in
  (t, pcg)

(* Magnitude in [0.25, 1.25], random sign -- bounded away from zero. For a
   divisor: a plain uniform(-1,1) draw is very likely to land close to zero
   across a few hundred elements, and a near-zero divisor amplifies an
   ordinary last-bit rounding difference between the Direct and Symbolic
   evaluators into a reported mismatch that is a property of neither. *)
let draw_nonzero pcg =
  let mag, pcg = Pcg.uniform ~low:0.25 ~high:1.25 pcg in
  let s, pcg = Pcg.uniform ~low:0. ~high:1. pcg in
  ((if s < 0.5 then -.mag else mag), pcg)

let synth_nonzero pcg shape = synth_with pcg shape draw_nonzero

(* Uniform in [0, 1] -- strictly non-negative, for an operand like [sqrt]'s
   input whose domain excludes negative values (a negative input is a NaN on
   both backends, which a tolerance-based comparator can't call "matched"). *)
let draw_nonneg pcg = Pcg.uniform ~low:0. ~high:1. pcg
let synth_nonneg pcg shape = synth_with pcg shape draw_nonneg

(* Uniform in [0.25, 1.25] -- strictly POSITIVE and bounded away from zero,
   unlike [draw_nonneg]. For an operand like [Pow]'s base under a negative or
   fractional exponent: a negative base is a NaN for a non-integer exponent
   (same hazard [draw_nonneg] exists for), and a base near zero blows up under
   a negative exponent, amplifying an ordinary last-bit rounding difference
   into a reported mismatch the same way [draw_nonzero] guards a divisor
   against. Mirrors [Aten_walk_recipes.Walk.gen_values_positive]. *)
let draw_positive pcg = Pcg.uniform ~low:0.25 ~high:1.25 pcg
let synth_positive pcg shape = synth_with pcg shape draw_positive
