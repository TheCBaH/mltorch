(* Assembles Vector_norm's walk: config space from Reduce.Vector_norm.Walk,
   one-input graph reducing over the config's axis subset. The squared leaf
   keeps the summed value nonnegative regardless of the drawn input's sign, so
   no [synth_nonneg] is needed the way [Sqrt]'s own walk needs one. *)

module M = struct
  module W = Reduce.Vector_norm.Walk (Walk_limits.L)
  include W

  type subject = Native_subject.t

  let target = "vector_norm"

  let build pcg c =
    let shape = W.shape c in
    let x, pcg = Native_tensor.synth pcg shape in
    let g =
      Err.or_raise ~pp_error:Graph_builder.pp_error
        Graph_builder.(
          build ~name:"vector_norm" ~outputs:(fun r -> [ r ])
          @@
          let* xi = input ~shape ~name:"x" () in
          vector_norm ~name:"out" (W.params c) xi)
    in
    let inputs = List.combine g.Graph_ir.Graph.inputs [ x ] in
    ({ Native_subject.target; graph = g; inputs }, pcg)
end
