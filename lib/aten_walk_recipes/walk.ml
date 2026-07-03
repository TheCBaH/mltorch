(* The walk machinery now lives in [walk_core] (backend-neutral, shared with the
   native engine). This module re-exports it and adds the ATen-specific tensor
   synthesis (an [Aten_spec.Tensor_spec.t] with [Values] drawn from the PCG).

   NOTE: the per-op recipes here are the ATen walks' own config-space
   descriptions; they are not shared with native (whose ops own their walk
   properties in lib/native/ops). The ATen self-containment refactor will fold
   these into the generator's emit helpers. *)

include Walk_core.Walk
module Pcg = Walk_core.Pcg
module Sv = Aten_spec.Scalar_value
module Tspec = Aten_spec.Tensor_spec

(* Draw [n] f32-canonical uniform(-1,1) values, threading the PCG. *)
let gen_values pcg n =
  let rec go i pcg acc =
    if i = 0 then (List.rev acc, pcg)
    else
      let v, pcg = Pcg.uniform ~low:(-1.) ~high:1. pcg in
      go (i - 1) pcg (Sv.Float v :: acc)
  in
  go n pcg []

(* A dtype=F32, Values-source tensor spec of the given shape. The PCG advances
   one step per element; [Values] means the runner's own PCG is never touched. *)
let tensor_spec pcg shape =
  let n = List.fold_left ( * ) 1 shape in
  let vals, pcg = gen_values pcg n in
  ({ Tspec.dtype = Aten_spec.Dtype.F32; shape; source = Tspec.Values vals }, pcg)
