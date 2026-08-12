(* Assembles Convolution's walk (non-transposed): config space from
   Conv.Convolution.Walk (reuses Conv2d's space), x/weight/bias graph. *)

module M = struct
  module Wk = Conv.Convolution.Walk (Walk_limits.L)
  include Wk

  type subject = Native_subject.t

  let target = "convolution"

  let build pcg c =
    let x, pcg = Native_tensor.synth pcg (Wk.x_shape c) in
    let weight, pcg = Native_tensor.synth pcg (Wk.weight_shape c) in
    let bias, pcg = Native_tensor.synth pcg (Wk.bias_shape c) in
    let g =
      Err.or_raise ~pp_error:Graph_builder.pp_error
        Graph_builder.(
          build ~name:"convolution" ~outputs:(fun r -> [ r ])
          @@
          let* xi = input ~shape:(Wk.x_shape c) ~name:"x" () in
          let* wi = input ~shape:(Wk.weight_shape c) ~name:"weight" () in
          let* bi = input ~shape:(Wk.bias_shape c) ~name:"bias" () in
          convolution ~name:"out" (Wk.params c) ~x:xi ~weight:wi ~bias:bi ())
    in
    let inputs = List.combine g.Graph_ir.Graph.inputs [ x; weight; bias ] in
    ({ Native_subject.target; graph = g; inputs }, pcg)
end
