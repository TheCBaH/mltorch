(* The I64 twin of [Mul_nwalk] (plan S6/T6.1/W); see [Add_i64_nwalk]'s own
   comment for the rationale, identical here. *)

module M = struct
  module W = Pointwise.Bin.Walk (Walk_limits.L)
  include W

  type subject = Native_subject.t

  let target = "mul_i64"

  let build pcg c =
    let shape = W.shape c in
    let a, pcg = Native_tensor.synth_i64 pcg shape in
    let b, pcg = Native_tensor.synth_i64 pcg shape in
    let g =
      Err.or_raise ~pp_error:Graph_builder.pp_error
        Graph_builder.(
          build ~name:"mul_i64" ~outputs:(fun r -> [ r ])
          @@
          let* ai = input ~shape ~fmt:(Payload.Fmt Payload.I64) ~name:"a" () in
          let* bi = input ~shape ~fmt:(Payload.Fmt Payload.I64) ~name:"b" () in
          mul ~name:"out" ai bi)
    in
    let inputs = List.combine g.Graph_ir.Graph.inputs [ a; b ] in
    ({ Native_subject.target; graph = g; inputs }, pcg)
end
