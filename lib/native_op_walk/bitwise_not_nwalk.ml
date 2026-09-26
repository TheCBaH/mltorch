(* [Bitwise_not]'s first walk (plan S6/T6.4/W; design §2's own unwalked-op
   list). Config space reused from [Pointwise.Clone.Walk] (shape-only,
   matching [Bitwise_not]'s own single shape-only operand); its output is
   unconditionally Bool ([Graph_builder.bitwise_not]), landing through
   [Kernel.Result_conversion.Nonzero_bool] on an ordinary float stage --
   already-working machinery, not gated on T6.0's I64-output question. *)

module M = struct
  module W = Pointwise.Clone.Walk (Walk_limits.L)
  include W

  type subject = Native_subject.t

  let target = "bitwise_not"

  let build pcg c =
    let shape = W.shape c in
    let x, pcg = Native_tensor.synth pcg shape in
    let g =
      Err.or_raise ~pp_error:Graph_builder.pp_error
        Graph_builder.(
          build ~name:"bitwise_not" ~outputs:(fun r -> [ r ])
          @@
          let* xi = input ~shape ~name:"x" () in
          bitwise_not ~name:"out" xi)
    in
    let inputs = List.combine g.Graph_ir.Graph.inputs [ x ] in
    ({ Native_subject.target; graph = g; inputs }, pcg)
end
