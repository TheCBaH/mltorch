(* Assembles Div's walk: config space from Pointwise.Bin.Walk (shared binary
   pointwise space), two-input native graph -- same shape as [Sub_nwalk]/
   [Mul_nwalk], except [b] (the divisor) is drawn via [Native_tensor.
   synth_nonzero] rather than [Native_tensor.synth]: see its doc comment.
   This is the reason [Div] wasn't registered in [Native_op_walk.all_walks]
   alongside [Div_scalar] until now -- a plain uniform(-1,1) divisor made the
   walk flaky, not wrong. *)

module M = struct
  module W = Pointwise.Bin.Walk (Walk_limits.L)
  include W

  type subject = Native_subject.t

  let target = "div"

  let build pcg c =
    let shape = W.shape c in
    let a, pcg = Native_tensor.synth pcg shape in
    let b, pcg = Native_tensor.synth_nonzero pcg shape in
    let g =
      Err.or_raise ~pp_error:Graph_builder.pp_error
        Graph_builder.(
          build ~name:"div" ~outputs:(fun r -> [ r ])
          @@
          let* ai = input ~shape ~name:"a" () in
          let* bi = input ~shape ~name:"b" () in
          div ~name:"out" ai bi)
    in
    let inputs = List.combine g.Graph_ir.Graph.inputs [ a; b ] in
    ({ Native_subject.target; graph = g; inputs }, pcg)
end
