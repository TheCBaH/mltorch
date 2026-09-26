(* The I64 twin of [Permute_nwalk] (plan S6/T6.1/W); see
   [Reshape_i64_nwalk]'s own comment for the rationale, identical here. *)

module M = struct
  module W = Permute.Permute.Walk (Walk_limits.L)
  include W

  type subject = Native_subject.t

  let target = "permute_i64"

  let build pcg c =
    let shape = W.shape c in
    let x, pcg = Native_tensor.synth_i64 pcg shape in
    let g =
      Err.or_raise ~pp_error:Graph_builder.pp_error
        Graph_builder.(
          build ~name:"permute_i64" ~outputs:(fun r -> [ r ])
          @@
          let* xi = input ~shape ~fmt:(Payload.Fmt Payload.I64) ~name:"x" () in
          permute ~name:"out" (W.perm c) xi)
    in
    let inputs = List.combine g.Graph_ir.Graph.inputs [ x ] in
    ({ Native_subject.target; graph = g; inputs }, pcg)
end
