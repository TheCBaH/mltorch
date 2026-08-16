(* Assembles Sub's walk: config space from Pointwise.Bin.Walk (shared binary
   pointwise space, applied to the shared limits), two-input native graph.

   [Bin.Walk] uses ONE shape for both operands (equal-shape; broadcast not
   exercised -- its own header says so), so this walk does not cover native
   broadcast for [sub], any more than [Add_nwalk]/[Mul_nwalk] do for [add]/
   [mul]. Broadcast evidence for [sub.Tensor] comes from the ATen side
   (Recipe_binary, an independent oracle) plus the focused fixtures in
   test/native_bridge_test.ml -- a deliberate scope decision (op3-impl.md
   Part IV #4), not an oversight. *)

module M = struct
  module W = Pointwise.Bin.Walk (Walk_limits.L)
  include W

  type subject = Native_subject.t

  let target = "sub"

  let build pcg c =
    let shape = W.shape c in
    let a, pcg = Native_tensor.synth pcg shape in
    let b, pcg = Native_tensor.synth pcg shape in
    let g =
      Err.or_raise ~pp_error:Graph_builder.pp_error
        Graph_builder.(
          build ~name:"sub" ~outputs:(fun r -> [ r ])
          @@
          let* ai = input ~shape ~name:"a" () in
          let* bi = input ~shape ~name:"b" () in
          sub ~name:"out" ai bi)
    in
    let inputs = List.combine g.Graph_ir.Graph.inputs [ a; b ] in
    ({ Native_subject.target; graph = g; inputs }, pcg)
end
