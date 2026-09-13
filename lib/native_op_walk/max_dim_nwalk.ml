(* Assembles MaxDim's walk: config space from Reduce.MaxDim.Walk, a one-input
   graph with BOTH outputs (values, indices) so the Direct==Symbolic check
   covers the paired [max_dim]/[max_dim_index] fold too, the same reason
   [Max_pool2d_with_indices_nwalk] builds both of its own outputs. *)

module M = struct
  module W = Reduce.MaxDim.Walk (Walk_limits.L)
  include W

  type subject = Native_subject.t

  let target = "max_dim"

  let build pcg c =
    let shape = W.shape c in
    let x, pcg = Native_tensor.synth pcg shape in
    let g =
      Err.or_raise ~pp_error:Graph_builder.pp_error
        Graph_builder.(
          build ~name:"max_dim" ~outputs:(fun (v, i) -> [ v; i ])
          @@
          let* xi = input ~shape ~name:"x" () in
          max_dim (W.params c) xi)
    in
    let inputs = List.combine g.Graph_ir.Graph.inputs [ x ] in
    ({ Native_subject.target; graph = g; inputs }, pcg)
end
