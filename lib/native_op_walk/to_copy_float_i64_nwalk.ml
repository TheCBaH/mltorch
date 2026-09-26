(* [To_copy]'s [Float] target on an I64 operand (plan S6/T6.2/W): the
   EdgeNeXt/mvitv2 "I64 Arange -> Float cast" acceptance pattern's own
   promoted-consumer step, exercised for the first time here -- [To_copy]
   itself has no other walk yet (design §2's own unwalked-op list). Output
   stays F32, same promotion shape as [Mul_scalar_i64_nwalk]. *)

module M = struct
  module W = Pointwise.To_copy.Walk (Walk_limits.L)
  include W

  type subject = Native_subject.t

  let target = "to_copy_float_i64"

  let build pcg c =
    let shape = W.shape c in
    let x, pcg = Native_tensor.synth_i64 pcg shape in
    let g =
      Err.or_raise ~pp_error:Graph_builder.pp_error
        Graph_builder.(
          build ~name:"to_copy_float_i64" ~outputs:(fun r -> [ r ])
          @@
          let* xi = input ~shape ~fmt:(Payload.Fmt Payload.I64) ~name:"x" () in
          to_copy ~name:"out" Pointwise.To_copy.Float xi)
    in
    let inputs = List.combine g.Graph_ir.Graph.inputs [ x ] in
    ({ Native_subject.target; graph = g; inputs }, pcg)
end
