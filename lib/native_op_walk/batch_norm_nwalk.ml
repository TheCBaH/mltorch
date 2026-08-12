(* Assembles BatchNorm's walk: config space from Norm.BatchNorm.Walk, a
   five-input graph (x + the per-channel weight/bias/running_mean/running_var
   [C] vectors). running_var is forced non-negative so sqrt(var+eps) is real —
   otherwise both backends produce nan, which compares unequal. *)

module M = struct
  module W = Norm.BatchNorm.Walk (Walk_limits.L)
  include W

  type subject = Native_subject.t

  let target = "batch_norm"

  let build pcg c =
    let shape = W.shape c in
    let x, pcg = Native_tensor.synth pcg shape in
    let cn = Dim.to_int (Vec6.get shape Axis.C) in
    let cshape = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:1 ~w:1 ~c:cn in
    let weight, pcg = Native_tensor.synth pcg cshape in
    let bias, pcg = Native_tensor.synth pcg cshape in
    let running_mean, pcg = Native_tensor.synth pcg cshape in
    let running_var0, pcg = Native_tensor.synth pcg cshape in
    let running_var =
      Tensor.materialize cshape (fun coord ->
          Float.abs (Tensor.read running_var0 coord) +. 1.)
    in
    let g =
      Err.or_raise ~pp_error:Graph_builder.pp_error
        Graph_builder.(
          build ~name:"batch_norm" ~outputs:(fun r -> [ r ])
          @@
          let* xi = input ~shape ~name:"x" () in
          let* wi = input ~shape:cshape ~name:"weight" () in
          let* bi = input ~shape:cshape ~name:"bias" () in
          let* rmi = input ~shape:cshape ~name:"running_mean" () in
          let* rvi = input ~shape:cshape ~name:"running_var" () in
          batch_norm ~name:"out" (W.params c) ~x:xi ~weight:wi ~bias:bi
            ~running_mean:rmi ~running_var:rvi ())
    in
    let inputs =
      List.combine g.Graph_ir.Graph.inputs
        [ x; weight; bias; running_mean; running_var ]
    in
    ({ Native_subject.target; graph = g; inputs }, pcg)
end
