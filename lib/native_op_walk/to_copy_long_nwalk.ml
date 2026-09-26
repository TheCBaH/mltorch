(* [To_copy]'s [Long] target on an F32 operand (plan S6/T6.3/W): the real
   ATen `.long()` cast, [Compute_to_long]'s checked Float-to-I64 arm
   ([Value.i64_of_float]: truncate toward zero, reject NaN/infinities/
   out-of-range magnitudes) -- an I64-declared [stages_i64] producer, unlike
   [To_copy_bool_nwalk]'s Bool-storage-on-a-float-stage shape. Reuses
   [Pointwise.To_copy.Walk] like the other two [To_copy] walks. [synth]'s
   uniform(-1,1) draw is always in [Compute_to_long]'s accepted range, so
   every step succeeds without needing a bespoke draw. *)

module M = struct
  module W = Pointwise.To_copy.Walk (Walk_limits.L)
  include W

  type subject = Native_subject.t

  let target = "to_copy_long"

  let build pcg c =
    let shape = W.shape c in
    let x, pcg = Native_tensor.synth pcg shape in
    let g =
      Err.or_raise ~pp_error:Graph_builder.pp_error
        Graph_builder.(
          build ~name:"to_copy_long" ~outputs:(fun r -> [ r ])
          @@
          let* xi = input ~shape ~name:"x" () in
          to_copy ~name:"out" Pointwise.To_copy.Long xi)
    in
    let inputs = List.combine g.Graph_ir.Graph.inputs [ x ] in
    ({ Native_subject.target; graph = g; inputs }, pcg)
end
