(* Assembles MaxPool2dWithIndices's walk: the shared pool window cfg, a one-input
   graph with BOTH outputs (values, indices) so the Direct==Symbolic check
   covers the argmax/value_of_index path too. *)

module M = struct
  module Wk = Pool.MaxPool2dWithIndices.Walk (Walk_limits.L)
  include Wk

  type subject = Native_subject.t

  let target = "max_pool2d_with_indices"

  let build pcg c =
    let shape = Wk.shape c in
    let x, pcg = Native_tensor.synth pcg shape in
    let g =
      match
        Graph_builder.(
          build ~name:"max_pool2d_with_indices" ~outputs:(fun (v, i) ->
              [ v; i ])
          @@
          let* xi = input ~shape ~name:"x" () in
          max_pool2d_with_indices (Wk.params c) xi)
      with
      | Ok g -> g
      | Error e ->
          failwith
            (Format.asprintf "%a" Graph_builder.pp_error e.Core.Error.kind)
    in
    let inputs = List.combine g.Graph_ir.Graph.inputs [ x ] in
    ({ Native_subject.target; graph = g; inputs }, pcg)
end
