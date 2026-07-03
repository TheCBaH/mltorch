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
