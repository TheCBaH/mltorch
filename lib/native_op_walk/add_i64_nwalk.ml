(* The I64 twin of [Add_nwalk] (plan S6/T6.1/W): both operands declared I64,
   so [Graph_builder.add] threads the exact-carrier arm ([Compute_i64]) both
   backends dispatch to, rather than the default float-pixel arm
   [Add_nwalk]'s own F32 operands reach. Same config space (shared limits),
   different synthesis (small integers, [Native_tensor.synth_i64], bounded
   away from int64 overflow). *)

module M = struct
  module W = Pointwise.Bin.Walk (Walk_limits.L)
  include W

  type subject = Native_subject.t

  let target = "add_i64"

  let build pcg c =
    let shape = W.shape c in
    let a, pcg = Native_tensor.synth_i64 pcg shape in
    let b, pcg = Native_tensor.synth_i64 pcg shape in
    let g =
      Err.or_raise ~pp_error:Graph_builder.pp_error
        Graph_builder.(
          build ~name:"add_i64" ~outputs:(fun r -> [ r ])
          @@
          let* ai = input ~shape ~fmt:(Payload.Fmt Payload.I64) ~name:"a" () in
          let* bi = input ~shape ~fmt:(Payload.Fmt Payload.I64) ~name:"b" () in
          add ~name:"out" ai bi)
    in
    let inputs = List.combine g.Graph_ir.Graph.inputs [ a; b ] in
    ({ Native_subject.target; graph = g; inputs }, pcg)
end
