(* [Eq_tensor]'s first walk (plan S6/T6.4/W; design §2's own unwalked-op
   list). Config space from [Pointwise.Bin.Walk] (shared with
   [Add]/[Sub]/[Mul]); output is unconditionally Bool
   ([Graph_builder.eq_tensor]), landing through
   [Kernel.Result_conversion.Nonzero_bool] on an ordinary float stage. Two
   INDEPENDENTLY drawn operands, so an exact element-wise match is measure
   zero -- Direct/Symbolic agreement on "false" everywhere is exactly as
   informative a check as agreement on "true" would be. *)

module M = struct
  module W = Pointwise.Bin.Walk (Walk_limits.L)
  include W

  type subject = Native_subject.t

  let target = "eq_tensor"

  let build pcg c =
    let shape = W.shape c in
    let a, pcg = Native_tensor.synth pcg shape in
    let b, pcg = Native_tensor.synth pcg shape in
    let g =
      Err.or_raise ~pp_error:Graph_builder.pp_error
        Graph_builder.(
          build ~name:"eq_tensor" ~outputs:(fun r -> [ r ])
          @@
          let* ai = input ~shape ~name:"a" () in
          let* bi = input ~shape ~name:"b" () in
          eq_tensor ~name:"out" ai bi)
    in
    let inputs = List.combine g.Graph_ir.Graph.inputs [ a; b ] in
    ({ Native_subject.target; graph = g; inputs }, pcg)
end
