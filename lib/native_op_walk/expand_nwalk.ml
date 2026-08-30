(* Assembles Expand's walk: shape config space from [Pointwise.Clone.Walk]
   (Expand has no [Walk] submodule of its own, the same reason [Sqrt_nwalk]
   reuses [Relu]'s). The drawn shape is the TARGET [size]; the source is the
   same shape with [H] forced to 1, so every step exercises a real broadcast
   read through [Pointwise_binary.broadcast_coord] rather than the degenerate
   self-equals-target case. *)

module M = struct
  module W = Pointwise.Clone.Walk (Walk_limits.L)
  include W

  type subject = Native_subject.t

  let target = "expand"

  let build pcg c =
    let size = W.shape c in
    let x_shape = Vec6.set size Axis.H Dim.one in
    let x, pcg = Native_tensor.synth pcg x_shape in
    let g =
      Err.or_raise ~pp_error:Graph_builder.pp_error
        Graph_builder.(
          build ~name:"expand" ~outputs:(fun r -> [ r ])
          @@
          let* xi = input ~shape:x_shape ~name:"x" () in
          expand ~name:"out" { Pointwise.Expand.size } xi)
    in
    let inputs = List.combine g.Graph_ir.Graph.inputs [ x ] in
    ({ Native_subject.target; graph = g; inputs }, pcg)
end
