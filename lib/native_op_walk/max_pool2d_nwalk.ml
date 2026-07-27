(* Assembles MaxPool2d's walk: config space from Pool.MaxPool2d.Walk (the shared
   pool window cfg, applied to the shared limits), one-input graph. *)

module M = struct
  module Wk = Pool.MaxPool2d.Walk (Walk_limits.L)
  include Wk

  type subject = Native_subject.t

  let target = "max_pool2d"

  let build pcg c =
    let shape = Wk.shape c in
    let x, pcg = Native_tensor.synth pcg shape in
    let g =
      Core.or_raise Graph_builder.pp_error
        Graph_builder.(
          build ~name:"max_pool2d" ~outputs:(fun r -> [ r ])
          @@
          let* xi = input ~shape ~name:"x" () in
          max_pool2d ~name:"out" (Wk.params c) xi)
    in
    let inputs = List.combine g.Graph_ir.Graph.inputs [ x ] in
    ({ Native_subject.target; graph = g; inputs }, pcg)
end
