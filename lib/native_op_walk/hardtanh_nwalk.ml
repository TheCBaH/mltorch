(* Assembles Hardtanh's walk: its config space (Pointwise.Hardtanh.Walk, applied to the
   shared limits) plus the native-graph [build]. *)

module M = struct
  module W = Pointwise.Hardtanh.Walk (Walk_limits.L)
  include W

  type subject = Native_subject.t

  let target = "hardtanh"

  let build pcg c =
    let shape = W.shape c in
    let x, pcg = Native_tensor.synth pcg shape in
    let g =
      match
        Graph_builder.(
          build ~name:"hardtanh" ~outputs:(fun r -> [ r ])
          @@
          let* xi = input ~shape ~name:"x" () in
          hardtanh ~name:"out" c.W.bounds xi)
      with
      | Ok g -> g
      | Error e ->
          failwith
            (Format.asprintf "%a" Graph_builder.pp_error e.Core.Error.kind)
    in
    let inputs = List.combine g.Graph_ir.Graph.inputs [ x ] in
    ({ Native_subject.target; graph = g; inputs }, pcg)
end
