(* The I64-operand twin of [Mul_scalar_nwalk] (plan S6/T6.2/W): an
   I64-declared input, so both backends dispatch through the explicit
   [i64_to_float]/[i64_load] promotion arm rather than [Mul_scalar_nwalk]'s
   own default float-pixel one. Unlike [Add_i64_nwalk] et al., the OUTPUT
   stays F32 (design §3's "promotion" case): only the READ changes, so the
   graph itself looks identical to the F32 walk's, no [~fmt] override on the
   output side. *)

module M = struct
  module W = Pointwise.Scalar_bin.Walk (Walk_limits.L)
  include W

  type subject = Native_subject.t

  let target = "mul_scalar_i64"

  let build pcg c =
    let shape = W.shape c in
    let x, pcg = Native_tensor.synth_i64 pcg shape in
    let g =
      Err.or_raise ~pp_error:Graph_builder.pp_error
        Graph_builder.(
          build ~name:"mul_scalar_i64" ~outputs:(fun r -> [ r ])
          @@
          let* xi = input ~shape ~fmt:(Payload.Fmt Payload.I64) ~name:"x" () in
          mul_scalar ~name:"out" c.W.scalar xi)
    in
    let inputs = List.combine g.Graph_ir.Graph.inputs [ x ] in
    ({ Native_subject.target; graph = g; inputs }, pcg)
end
