(* Assembles AvgPool2d's walk: config space from Pool.AvgPool2d.Walk, one-input
   graph. *)

module M = struct
  module Wk = Pool.AvgPool2d.Walk (Walk_limits.L)
  include Wk

  type subject = Native_subject.t

  let target = "avg_pool2d"

  let build pcg c =
    let shape = Wk.shape c in
    let x, pcg = Native_tensor.synth pcg shape in
    let g =
      match
        Graph_builder.(
          build ~name:"avg_pool2d" ~outputs:(fun r -> [ r ])
          @@
          let* xi = input ~shape ~name:"x" () in
          avg_pool2d ~name:"out" (Wk.params c) xi)
      with
      | Ok g -> g
      | Error e ->
          failwith
            (Format.asprintf "%a" Graph_builder.pp_error e.Core.Error.kind)
    in
    let inputs = List.combine g.Graph_ir.Graph.inputs [ x ] in
    ({ Native_subject.target; graph = g; inputs }, pcg)
end
