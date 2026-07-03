(* Assembles Conv2d's walk: config space + native-layout shape derivations from
   Conv.Conv2d.Walk (applied to the shared limits); the graph wires x/weight/bias
   (in that order, matching [Graph.inputs]) into one conv2d node. *)

module M = struct
  module W = Conv.Conv2d.Walk (Walk_limits.L)
  include W

  type subject = Native_subject.t

  let target = "conv2d"

  let build pcg c =
    let x, pcg = Native_tensor.synth pcg (W.x_shape c) in
    let weight, pcg = Native_tensor.synth pcg (W.weight_shape c) in
    let bias, pcg = Native_tensor.synth pcg (W.bias_shape c) in
    let g =
      match
        Graph_builder.(
          build ~name:"conv2d" ~outputs:(fun r -> [ r ])
          @@
          let* xi = input ~shape:(W.x_shape c) ~name:"x" () in
          let* wi = input ~shape:(W.weight_shape c) ~name:"weight" () in
          let* bi = input ~shape:(W.bias_shape c) ~name:"bias" () in
          conv2d ~name:"out" (W.params c) ~x:xi ~weight:wi ~bias:bi ())
      with
      | Ok g -> g
      | Error e ->
          failwith
            (Format.asprintf "%a" Graph_builder.pp_error e.Core.Error.kind)
    in
    let inputs = List.combine g.Graph_ir.Graph.inputs [ x; weight; bias ] in
    ({ Native_subject.target; graph = g; inputs }, pcg)
end
