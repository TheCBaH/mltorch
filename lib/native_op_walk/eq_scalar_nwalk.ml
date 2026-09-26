(* [Eq_scalar]'s first walk (plan S6/T6.4/W; design §2's own unwalked-op
   list). Config space from [Pointwise.Scalar_bin.Walk] (shared with
   [Add_scalar]/[Mul_scalar] etc.); output is unconditionally Bool
   ([Graph_builder.eq_scalar]), landing through
   [Kernel.Result_conversion.Nonzero_bool] on an ordinary float stage. *)

module M = struct
  module W = Pointwise.Scalar_bin.Walk (Walk_limits.L)
  include W

  type subject = Native_subject.t

  let target = "eq_scalar"

  let build pcg c =
    let shape = W.shape c in
    let x, pcg = Native_tensor.synth pcg shape in
    let g =
      Err.or_raise ~pp_error:Graph_builder.pp_error
        Graph_builder.(
          build ~name:"eq_scalar" ~outputs:(fun r -> [ r ])
          @@
          let* xi = input ~shape ~name:"x" () in
          eq_scalar ~name:"out" c.W.scalar xi)
    in
    let inputs = List.combine g.Graph_ir.Graph.inputs [ x ] in
    ({ Native_subject.target; graph = g; inputs }, pcg)
end
