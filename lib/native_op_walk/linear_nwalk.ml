(* Assembles Linear's walk: config space + native-layout shape derivations from
   Linear.Linear.Walk (applied to the shared limits); the graph wires
   x/weight/bias (in that order, matching [Graph.inputs]) into one linear node.

   The bias is an AXIS, not a fixture: [Graph_ir]'s [Linear] carries it as an
   option and [Eval_op] synthesizes a zero one when it is absent, so the two
   states are different graphs through different code and both have to be
   walked. *)

module M = struct
  module W = Linear.Linear.Walk (Walk_limits.L)
  include W

  type subject = Native_subject.t

  let target = "linear"

  let build pcg c =
    let x, pcg = Native_tensor.synth pcg (W.x_shape c) in
    let weight, pcg = Native_tensor.synth pcg (W.weight_shape c) in
    let bias, pcg =
      if W.bias_present c then
        let b, pcg = Native_tensor.synth pcg (W.bias_shape c) in
        (Some b, pcg)
      else (None, pcg)
    in
    let g =
      Err.or_raise ~pp_error:Graph_builder.pp_error
        Graph_builder.(
          build ~name:"linear" ~outputs:(fun r -> [ r ])
          @@
          let* xi = input ~shape:(W.x_shape c) ~name:"x" () in
          let* wi = input ~shape:(W.weight_shape c) ~name:"weight" () in
          let* bi =
            if W.bias_present c then
              let* b = input ~shape:(W.bias_shape c) ~name:"bias" () in
              return (Some b)
            else return None
          in
          linear ~name:"out" (W.params c) ~x:xi ~weight:wi ?bias:bi ())
    in
    let inputs =
      List.combine g.Graph_ir.Graph.inputs ([ x; weight ] @ Option.to_list bias)
    in
    ({ Native_subject.target; graph = g; inputs }, pcg)
end
