(* [Ne_tensor]'s first walk (plan S6/T6.4/W); see [Eq_tensor_nwalk]'s own
   comment for the rationale, identical here. *)

module M = struct
  module W = Pointwise.Bin.Walk (Walk_limits.L)
  include W

  type subject = Native_subject.t

  let target = "ne_tensor"

  let build pcg c =
    let shape = W.shape c in
    let a, pcg = Native_tensor.synth pcg shape in
    let b, pcg = Native_tensor.synth pcg shape in
    let g =
      Err.or_raise ~pp_error:Graph_builder.pp_error
        Graph_builder.(
          build ~name:"ne_tensor" ~outputs:(fun r -> [ r ])
          @@
          let* ai = input ~shape ~name:"a" () in
          let* bi = input ~shape ~name:"b" () in
          ne_tensor ~name:"out" ai bi)
    in
    let inputs = List.combine g.Graph_ir.Graph.inputs [ a; b ] in
    ({ Native_subject.target; graph = g; inputs }, pcg)
end
