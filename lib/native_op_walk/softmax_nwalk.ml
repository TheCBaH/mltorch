(* Assembles Softmax's walk: config space from Reduce.Softmax.Walk, one-input
   graph softmaxed over the config's axis. *)

module M = struct
  module W = Reduce.Softmax.Walk (Walk_limits.L)
  include W

  type subject = Native_subject.t

  let target = "softmax"

  let build pcg c =
    let shape = W.shape c in
    let x, pcg = Native_tensor.synth pcg shape in
    let g =
      Err.or_raise ~pp_error:Graph_builder.pp_error
        Graph_builder.(
          build ~name:"softmax" ~outputs:(fun r -> [ r ])
          @@
          let* xi = input ~shape ~name:"x" () in
          softmax ~name:"out" (W.params c) xi)
    in
    let inputs = List.combine g.Graph_ir.Graph.inputs [ x ] in
    ({ Native_subject.target; graph = g; inputs }, pcg)
end
