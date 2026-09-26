(* [Ne_scalar]'s first walk (plan S6/T6.4/W); see [Eq_scalar_nwalk]'s own
   comment for the rationale, identical here. *)

module M = struct
  module W = Pointwise.Scalar_bin.Walk (Walk_limits.L)
  include W

  type subject = Native_subject.t

  let target = "ne_scalar"

  let build pcg c =
    let shape = W.shape c in
    let x, pcg = Native_tensor.synth pcg shape in
    let g =
      Err.or_raise ~pp_error:Graph_builder.pp_error
        Graph_builder.(
          build ~name:"ne_scalar" ~outputs:(fun r -> [ r ])
          @@
          let* xi = input ~shape ~name:"x" () in
          ne_scalar ~name:"out" c.W.scalar xi)
    in
    let inputs = List.combine g.Graph_ir.Graph.inputs [ x ] in
    ({ Native_subject.target; graph = g; inputs }, pcg)
end
