(* [To_copy]'s [Bool] target on an F32 operand (plan S6/T6.4/W): the real
   ATen `.bool()` cast, [Compute]'s [Bool] arm ([S.eq]-based nonzero test),
   landing through [Kernel.Result_conversion.Nonzero_bool] on an ordinary
   float stage -- reuses [Pointwise.To_copy.Walk] like
   [To_copy_float_i64_nwalk]. *)

module M = struct
  module W = Pointwise.To_copy.Walk (Walk_limits.L)
  include W

  type subject = Native_subject.t

  let target = "to_copy_bool"

  let build pcg c =
    let shape = W.shape c in
    let x, pcg = Native_tensor.synth pcg shape in
    let g =
      Err.or_raise ~pp_error:Graph_builder.pp_error
        Graph_builder.(
          build ~name:"to_copy_bool" ~outputs:(fun r -> [ r ])
          @@
          let* xi = input ~shape ~name:"x" () in
          to_copy ~name:"out" Pointwise.To_copy.Bool xi)
    in
    let inputs = List.combine g.Graph_ir.Graph.inputs [ x ] in
    ({ Native_subject.target; graph = g; inputs }, pcg)
end
