(* [Split_with_sizes]'s exact int64 arm (plan S6/S7/T7.1): the same subject
   as [Split_with_sizes_nwalk], but over an I64 input -- exercises
   [Split.Split_with_sizes.Compute_i64], not the default float-pixel one. *)

module M = struct
  module Wk = Split.Split_with_sizes.Walk (Walk_limits.L)
  include Wk

  type subject = Native_subject.t

  let target = "split_with_sizes_i64"

  let build pcg c =
    let shape = Wk.shape c in
    let x, pcg = Native_tensor.synth_i64 pcg shape in
    let g =
      Err.or_raise ~pp_error:Graph_builder.pp_error
        Graph_builder.(
          build ~name:"split_with_sizes_i64" ~outputs:Fun.id
          @@
          let* xi = input ~shape ~fmt:(Payload.Fmt Payload.I64) ~name:"x" () in
          split_with_sizes (Wk.params c) xi)
    in
    let inputs = List.combine g.Graph_ir.Graph.inputs [ x ] in
    ({ Native_subject.target; graph = g; inputs }, pcg)
end
