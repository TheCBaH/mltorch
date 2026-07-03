(* Assembles Conv2d_padding's walk: config space from Conv.Conv2d_padding.Walk
   (string padding mode), x/weight/bias graph. *)

module M = struct
  module W = Conv.Conv2d_padding.Walk (Walk_limits.L)
  include W

  type subject = Native_subject.t

  let target = "conv2d_padding"

  let build pcg c =
    let x, pcg = Native_tensor.synth pcg (W.x_shape c) in
    let weight, pcg = Native_tensor.synth pcg (W.weight_shape c) in
    let bias, pcg = Native_tensor.synth pcg (W.bias_shape c) in
    let g =
      match
        Graph_builder.(
          build ~name:"conv2d_padding" ~outputs:(fun r -> [ r ])
          @@
          let* xi = input ~shape:(W.x_shape c) ~name:"x" () in
          let* wi = input ~shape:(W.weight_shape c) ~name:"weight" () in
          let* bi = input ~shape:(W.bias_shape c) ~name:"bias" () in
          conv2d_padding ~name:"out" (W.params c) ~x:xi ~weight:wi ~bias:bi ())
      with
      | Ok g -> g
      | Error e ->
          failwith
            (Format.asprintf "%a" Graph_builder.pp_error e.Core.Error.kind)
    in
    let inputs = List.combine g.Graph_ir.Graph.inputs [ x; weight; bias ] in
    ({ Native_subject.target; graph = g; inputs }, pcg)
end
