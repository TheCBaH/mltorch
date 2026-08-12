(* Assembles Reshape's walk: config from Reshape.Reshape.Walk (vary the input
   shape; the target flattens all elements onto C), one-input graph. Exercises
   the flat-offset + delinearize index math under Direct==Symbolic. *)

module M = struct
  module W = Reshape.Reshape.Walk (Walk_limits.L)
  include W

  type subject = Native_subject.t

  let target = "reshape"

  let build pcg c =
    let shape = W.shape c in
    let x, pcg = Native_tensor.synth pcg shape in
    let g =
      Err.or_raise ~pp_error:Graph_builder.pp_error
        Graph_builder.(
          build ~name:"reshape" ~outputs:(fun r -> [ r ])
          @@
          let* xi = input ~shape ~name:"x" () in
          reshape ~name:"out" (W.params c) xi)
    in
    let inputs = List.combine g.Graph_ir.Graph.inputs [ x ] in
    ({ Native_subject.target; graph = g; inputs }, pcg)
end
