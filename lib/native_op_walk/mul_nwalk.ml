(* Assembles Mul's walk: config space from Pointwise.Bin.Walk, two-input graph. *)

module M = struct
  module W = Pointwise.Bin.Walk (Walk_limits.L)
  include W

  type subject = Native_subject.t

  let target = "mul"

  let build pcg c =
    let shape = W.shape c in
    let a, pcg = Native_tensor.synth pcg shape in
    let b, pcg = Native_tensor.synth pcg shape in
    let g =
      match
        Graph_builder.(
          build ~name:"mul" ~outputs:(fun r -> [ r ])
          @@
          let* ai = input ~shape ~name:"a" () in
          let* bi = input ~shape ~name:"b" () in
          mul ~name:"out" ai bi)
      with
      | Ok g -> g
      | Error e ->
          failwith
            (Format.asprintf "%a" Graph_builder.pp_error e.Core.Error.kind)
    in
    let inputs = List.combine g.Graph_ir.Graph.inputs [ a; b ] in
    ({ Native_subject.target; graph = g; inputs }, pcg)
end
