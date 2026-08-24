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

(* Draw [n] f32-canonical values with magnitude in [0.25, 1.25] and a random
   sign -- bounded away from zero, unlike [gen_values]. A divisor drawn from
   [gen_values] can land arbitrarily close to zero (uniform(-1,1) over
   hundreds of elements makes this likely, not merely possible), and a
   near-zero divisor amplifies the last-bit rounding difference between ATen's
   kernel and native's into a mismatch that has nothing to do with either
   being wrong -- the same hazard [native_op_walk]'s [Div] (unlike
   [Div_scalar]) is kept out of the walk list for. *)
let gen_values_nonzero pcg n =
  let rec go i pcg acc =
    if i = 0 then (List.rev acc, pcg)
    else
      let mag, pcg = Pcg.uniform ~low:0.25 ~high:1.25 pcg in
      let s, pcg = Pcg.uniform ~low:0. ~high:1. pcg in
      let v = if s < 0.5 then -.mag else mag in
      go (i - 1) pcg (Sv.Float v :: acc)
  in
  go n pcg []

(* [tensor_spec] with [gen_values_nonzero] instead -- for a divisor operand. *)
let tensor_spec_nonzero pcg shape =
  let n = List.fold_left ( * ) 1 shape in
  let vals, pcg = gen_values_nonzero pcg n in
  ({ Tspec.dtype = Aten_spec.Dtype.F32; shape; source = Tspec.Values vals }, pcg)

(* Draw [n] f32-canonical values uniform in [0.25, 1.25] -- strictly positive,
   unlike [gen_values_nonzero] (which allows either sign). For a quantity that
   must stay positive to avoid a NaN neither backend is "wrong" about, e.g. a
   batch-norm [running_var]: a symmetric draw would sometimes make
   [running_var + eps] negative, and sqrt of that is a NaN on both sides that
   a tolerance-based comparator has no good way to call "matched". *)
let gen_values_positive pcg n =
  let rec go i pcg acc =
    if i = 0 then (List.rev acc, pcg)
    else
      let v, pcg = Pcg.uniform ~low:0.25 ~high:1.25 pcg in
      go (i - 1) pcg (Sv.Float v :: acc)
  in
  go n pcg []

(* [tensor_spec] with [gen_values_positive] instead -- for e.g. a variance
   operand. *)
let tensor_spec_positive pcg shape =
  let n = List.fold_left ( * ) 1 shape in
  let vals, pcg = gen_values_positive pcg n in
  ({ Tspec.dtype = Aten_spec.Dtype.F32; shape; source = Tspec.Values vals }, pcg)
