(* Assembles Pow's walk: reuses [Pointwise.Scalar_bin.Walk] for the
   shape/scalar config space (shared with Add_scalar/Div_scalar/Mul_scalar).
   [x] is drawn via [Native_tensor.synth_positive] rather than [synth]: two of
   the shared candidates (0.5, -2.) are undefined or singular at a
   non-positive base, the same domain hazard [Sqrt]'s own walk exists to
   avoid. *)

module M = struct
  module W = Pointwise.Scalar_bin.Walk (Walk_limits.L)
  include W

  type subject = Native_subject.t

  let target = "pow"

  let build pcg c =
    let shape = W.shape c in
    let x, pcg = Native_tensor.synth_positive pcg shape in
    let g =
      Err.or_raise ~pp_error:Graph_builder.pp_error
        Graph_builder.(
          build ~name:"pow" ~outputs:(fun r -> [ r ])
          @@
          let* xi = input ~shape ~name:"x" () in
          pow ~name:"out" c.W.scalar xi)
    in
    let inputs = List.combine g.Graph_ir.Graph.inputs [ x ] in
    ({ Native_subject.target; graph = g; inputs }, pcg)
end
