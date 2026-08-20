(* Assembles Slice's walk. The config space lives with the op (split.ml's
   [Walk]), where the bounds are DERIVED from the drawn extent rather than
   mutated beside it -- so no step can produce an empty or out-of-range
   selection, and every drawn config is a valid graph.

   What this covers is staging, not arithmetic: Direct and Symbolic instantiate
   the SAME [Compute] functor, so a slice that read the wrong axis passes here.
   Its value is that the affine index map -- [start + step * i] narrowed with
   [clamp_low] -- goes through [Expr]'s staging, folding and grounding rather
   than only through [Direct]'s integer arithmetic. The independent oracle is
   the ATen walk. *)

module M = struct
  module Wk = Split.Slice.Walk (Walk_limits.L)
  include Wk

  type subject = Native_subject.t

  let target = "slice"

  let build pcg c =
    let shape = Wk.shape c in
    let x, pcg = Native_tensor.synth pcg shape in
    let g =
      Err.or_raise ~pp_error:Graph_builder.pp_error
        Graph_builder.(
          build ~name:"slice" ~outputs:(fun r -> [ r ])
          @@
          let* xi = input ~shape ~name:"x" () in
          slice (Wk.params c) xi)
    in
    let inputs = List.combine g.Graph_ir.Graph.inputs [ x ] in
    ({ Native_subject.target; graph = g; inputs }, pcg)
end
