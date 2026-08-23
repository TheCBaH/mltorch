(* Assembles Sigmoid's walk: its config space (Pointwise.Sigmoid.Walk, applied
   to the shared limits) plus the native-graph [build]. *)

module M = struct
  module W = Pointwise.Sigmoid.Walk (Walk_limits.L)
  include W

  type subject = Native_subject.t

  let target = "sigmoid"

  let build pcg c =
    let shape = W.shape c in
    let x, pcg = Native_tensor.synth pcg shape in
    let g =
      Err.or_raise ~pp_error:Graph_builder.pp_error
        Graph_builder.(
          build ~name:"sigmoid" ~outputs:(fun r -> [ r ])
          @@
          let* xi = input ~shape ~name:"x" () in
          sigmoid ~name:"out" xi)
    in
    let inputs = List.combine g.Graph_ir.Graph.inputs [ x ] in
    ({ Native_subject.target; graph = g; inputs }, pcg)
end
