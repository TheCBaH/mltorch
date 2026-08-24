(* Assembles Sqrt's walk: reuses [Pointwise.Relu.Walk] for the shape space
   (a single tensor of any shape -- generic, not relu-specific; see its own
   comment), since [Sqrt] has no [Walk] submodule of its own. [x] is drawn via
   [Native_tensor.synth_nonneg] rather than [Native_tensor.synth]: sqrt's
   domain excludes negative inputs (see [Pointwise.Sqrt]'s own comment), and a
   negative draw would produce a NaN on both backends that a tolerance-based
   comparator can't call "matched". *)

module M = struct
  module W = Pointwise.Relu.Walk (Walk_limits.L)
  include W

  type subject = Native_subject.t

  let target = "sqrt"

  let build pcg c =
    let shape = W.shape c in
    let x, pcg = Native_tensor.synth_nonneg pcg shape in
    let g =
      Err.or_raise ~pp_error:Graph_builder.pp_error
        Graph_builder.(
          build ~name:"sqrt" ~outputs:(fun r -> [ r ])
          @@
          let* xi = input ~shape ~name:"x" () in
          sqrt ~name:"out" xi)
    in
    let inputs = List.combine g.Graph_ir.Graph.inputs [ x ] in
    ({ Native_subject.target; graph = g; inputs }, pcg)
end
