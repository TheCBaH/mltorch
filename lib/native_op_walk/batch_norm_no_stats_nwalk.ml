(* Walk assembly for training batch normalisation without running statistics.
   All three results are graph outputs: Direct/Symbolic agreement must exercise
   the saved mean and inverse standard deviation, not merely the activations. *)

module M = struct
  module W = Norm.BatchNormNoStats.Walk (Walk_limits.L)
  include W

  type subject = Native_subject.t

  let target = "batch_norm_no_stats"

  let build pcg c =
    let shape = W.shape c in
    let x, pcg = Native_tensor.synth pcg shape in
    let cshape =
      Norm.BatchNormNoStats.stats_shape ~x_shape:shape (W.params c)
    in
    let weight, pcg = Native_tensor.synth pcg cshape in
    let bias, pcg = Native_tensor.synth pcg cshape in
    let g =
      Err.or_raise ~pp_error:Graph_builder.pp_error
        Graph_builder.(
          build ~name:"batch_norm_no_stats" ~outputs:(fun r -> r)
          @@
          let* xi = input ~shape ~name:"x" () in
          let* wi =
            if c.weight then
              let+ wi = input ~shape:cshape ~name:"weight" () in
              Some wi
            else return None
          in
          let* bi =
            if c.bias then
              let+ bi = input ~shape:cshape ~name:"bias" () in
              Some bi
            else return None
          in
          batch_norm_no_stats ~name:"out" (W.params c) ~x:xi ?weight:wi ?bias:bi
            ())
    in
    let inputs =
      List.combine g.Graph_ir.Graph.inputs
        ([ x ]
        @ (if c.weight then [ weight ] else [])
        @ if c.bias then [ bias ] else [])
    in
    ({ Native_subject.target; graph = g; inputs }, pcg)
end
