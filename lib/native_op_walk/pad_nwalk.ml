(* Assembles Pad's walk. The config space lives with the op (pad.ml's [Walk]),
   where the pad entries are DERIVED from the drawn shape rather than mutated
   beside it -- so no step can produce a reflect pad wider than its axis or a
   crop that empties one, and every drawn config is a valid graph.

   What this covers is staging, not arithmetic: Direct and Symbolic instantiate
   the SAME [Compute] functor, so a wrong mirror passes here. Its value is that
   [Pad] is the only op in the engine whose compute uses [index_max] and an
   [index_eq]-driven [select], and this is what exercises both through
   [Expr]'s staging, folding and grounding rather than only through [Direct]. The
   independent oracle is the ATen walk. *)

module M = struct
  module Wk = Pad.Pad.Walk (Walk_limits.L)
  include Wk

  type subject = Native_subject.t

  let target = "pad"

  let build pcg c =
    let shape = Wk.shape c in
    let x, pcg = Native_tensor.synth pcg shape in
    let g =
      Err.or_raise ~pp_error:Graph_builder.pp_error
        Graph_builder.(
          build ~name:"pad" ~outputs:(fun r -> [ r ])
          @@
          let* xi = input ~shape ~name:"x" () in
          pad (Wk.params c) xi)
    in
    let inputs = List.combine g.Graph_ir.Graph.inputs [ x ] in
    ({ Native_subject.target; graph = g; inputs }, pcg)
end
