(* The I64 twin of [Reshape_nwalk] (plan S6/T6.1/W): an I64-declared input, so
   [Graph_builder.reshape] threads the exact-carrier arm ([Compute_i64],
   [Tensor.i64_load]) both backends dispatch to, rather than the default
   float-pixel arm [Reshape_nwalk]'s own F32 input reaches. *)

module M = struct
  module W = Reshape.Reshape.Walk (Walk_limits.L)
  include W

  type subject = Native_subject.t

  let target = "reshape_i64"

  let build pcg c =
    let shape = W.shape c in
    let x, pcg = Native_tensor.synth_i64 pcg shape in
    let g =
      Err.or_raise ~pp_error:Graph_builder.pp_error
        Graph_builder.(
          build ~name:"reshape_i64" ~outputs:(fun r -> [ r ])
          @@
          let* xi = input ~shape ~fmt:(Payload.Fmt Payload.I64) ~name:"x" () in
          reshape ~name:"out" (W.params c) xi)
    in
    let inputs = List.combine g.Graph_ir.Graph.inputs [ x ] in
    ({ Native_subject.target; graph = g; inputs }, pcg)
end
