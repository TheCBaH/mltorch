(* Assembles RmsNorm's walk: config space from Norm.RmsNorm.Walk, a graph with
   one or two inputs depending on whether the config has a weight. The absent
   case is a genuinely different graph -- [Graph_ir]'s [Rms_norm] carries
   [weight : Tensor_ref.t option] -- so both must be walked. *)

module M = struct
  module W = Norm.RmsNorm.Walk (Walk_limits.L)
  include W

  type subject = Native_subject.t

  let target = "rms_norm"

  let build pcg c =
    let shape = W.shape c in
    let x, pcg = Native_tensor.synth pcg shape in
    let build_graph inputs body =
      let g = Err.or_raise ~pp_error:Graph_builder.pp_error body in
      ( {
          Native_subject.target;
          graph = g;
          inputs = List.combine g.Graph_ir.Graph.inputs inputs;
        },
        pcg )
    in
    if W.weight_present c then
      let wshape = W.weight_shape c in
      let weight, pcg = Native_tensor.synth pcg wshape in
      let g =
        Err.or_raise ~pp_error:Graph_builder.pp_error
          Graph_builder.(
            build ~name:"rms_norm" ~outputs:(fun r -> [ r ])
            @@
            let* xi = input ~shape ~name:"x" () in
            let* wi = input ~shape:wshape ~name:"weight" () in
            rms_norm ~name:"out" (W.params c) ~x:xi ~weight:wi ())
      in
      ( {
          Native_subject.target;
          graph = g;
          inputs = List.combine g.Graph_ir.Graph.inputs [ x; weight ];
        },
        pcg )
    else
      build_graph [ x ]
        Graph_builder.(
          build ~name:"rms_norm" ~outputs:(fun r -> [ r ])
          @@
          let* xi = input ~shape ~name:"x" () in
          rms_norm ~name:"out" (W.params c) ~x:xi ())
end
