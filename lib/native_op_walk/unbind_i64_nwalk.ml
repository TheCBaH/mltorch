(* [Unbind]'s exact int64 arm (plan S6/S7/T7.1): the same variable-arity
   subject as [Unbind_nwalk], but over an I64 input -- exercises
   [Split.Unbind.Compute_i64], not [Unbind_nwalk]'s own default float-pixel
   one, so a slice of an I64 tensor stays I64 storage rather than
   round-tripping through the engine's f32 compute domain. *)

module M = struct
  module Wk = Split.Unbind.Walk (Walk_limits.L)
  include Wk

  type subject = Native_subject.t

  let target = "unbind_i64"

  let build pcg c =
    let shape = Wk.shape c in
    let x, pcg = Native_tensor.synth_i64 pcg shape in
    let g =
      Err.or_raise ~pp_error:Graph_builder.pp_error
        Graph_builder.(
          build ~name:"unbind_i64" ~outputs:Fun.id
          @@
          let* xi = input ~shape ~fmt:(Payload.Fmt Payload.I64) ~name:"x" () in
          unbind (Wk.params c) xi)
    in
    let inputs = List.combine g.Graph_ir.Graph.inputs [ x ] in
    ({ Native_subject.target; graph = g; inputs }, pcg)
end
