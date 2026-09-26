(* [Split_with_sizes]'s own first walk (plan S7/T7.1; design's own
   unwalked-op list): a one-input graph exposing every window as a graph
   output, the same "every ordinal, not just the first" shape
   [Unbind_nwalk] uses -- the two are siblings ([Split_with_sizes] keeps the
   split axis, [Unbind] drops it), so their walk files mirror each other. *)

module M = struct
  module Wk = Split.Split_with_sizes.Walk (Walk_limits.L)
  include Wk

  type subject = Native_subject.t

  let target = "split_with_sizes"

  let build pcg c =
    let shape = Wk.shape c in
    let x, pcg = Native_tensor.synth pcg shape in
    let g =
      Err.or_raise ~pp_error:Graph_builder.pp_error
        Graph_builder.(
          build ~name:"split_with_sizes" ~outputs:Fun.id
          @@
          let* xi = input ~shape ~name:"x" () in
          split_with_sizes (Wk.params c) xi)
    in
    let inputs = List.combine g.Graph_ir.Graph.inputs [ x ] in
    ({ Native_subject.target; graph = g; inputs }, pcg)
end
