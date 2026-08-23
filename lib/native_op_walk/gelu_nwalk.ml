(* Assembles Gelu's walk: its config space (Pointwise.Gelu.Walk, applied to
   the shared limits) plus the native-graph [build]. Because [Direct.erf] and
   grounded [Symbolic]'s [erf] share one implementation (see the [erf]
   primitive in expr.ml), this walk proves staging/scheduling agreement, not
   the erf approximation's accuracy against ATen -- that proof is
   [native_bridge_test.ml]'s [verify_print]. *)

module M = struct
  module W = Pointwise.Gelu.Walk (Walk_limits.L)
  include W

  type subject = Native_subject.t

  let target = "gelu"

  let build pcg c =
    let shape = W.shape c in
    let x, pcg = Native_tensor.synth pcg shape in
    let g =
      Err.or_raise ~pp_error:Graph_builder.pp_error
        Graph_builder.(
          build ~name:"gelu" ~outputs:(fun r -> [ r ])
          @@
          let* xi = input ~shape ~name:"x" () in
          gelu ~name:"out" xi)
    in
    let inputs = List.combine g.Graph_ir.Graph.inputs [ x ] in
    ({ Native_subject.target; graph = g; inputs }, pcg)
end
