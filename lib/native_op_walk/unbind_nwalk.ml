(* Assembles Unbind's walk: a one-input graph exposing EVERY slice as a graph
   output, so the Direct==Symbolic check covers each ordinal rather than just
   the first. This is also the only walk whose output count varies with the
   config, which is what exercises the variable-arity builder/eval path itself
   — the axis mutation changes how many stages the symbolic program has. *)

module M = struct
  module Wk = Split.Unbind.Walk (Walk_limits.L)
  include Wk

  type subject = Native_subject.t

  let target = "unbind"

  let build pcg c =
    let shape = Wk.shape c in
    let x, pcg = Native_tensor.synth pcg shape in
    let g =
      Err.or_raise ~pp_error:Graph_builder.pp_error
        Graph_builder.(
          build ~name:"unbind" ~outputs:Fun.id
          @@
          let* xi = input ~shape ~name:"x" () in
          unbind (Wk.params c) xi)
    in
    let inputs = List.combine g.Graph_ir.Graph.inputs [ x ] in
    ({ Native_subject.target; graph = g; inputs }, pcg)
end
