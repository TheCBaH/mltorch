(* Assembles Mul_scalar's walk: its config space (Pointwise.Scalar_bin.Walk,
   applied to the shared limits) plus the native-graph [build]. *)

module M = struct
  module W = Pointwise.Scalar_bin.Walk (Walk_limits.L)
  include W

  type subject = Native_subject.t

  let target = "mul_scalar"

  let build pcg c =
    let shape = W.shape c in
    let x, pcg = Native_tensor.synth pcg shape in
    let g =
      Err.or_raise ~pp_error:Graph_builder.pp_error
        Graph_builder.(
          build ~name:"mul_scalar" ~outputs:(fun r -> [ r ])
          @@
          let* xi = input ~shape ~name:"x" () in
          mul_scalar ~name:"out" c.W.scalar xi)
    in
    let inputs = List.combine g.Graph_ir.Graph.inputs [ x ] in
    ({ Native_subject.target; graph = g; inputs }, pcg)
end
