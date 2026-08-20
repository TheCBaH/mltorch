(* Assembles LayerNorm's walk: config space from Norm.LayerNorm.Walk, and a
   graph with one, two or three inputs depending on which affine operands the
   config has. All four combinations are genuinely different graphs --
   [Graph_ir]'s [Layer_norm] carries [weight] and [bias] as INDEPENDENT options
   and [Eval_op] fills each absent one separately -- so all four must be walked;
   the absent case is a different code path, not a ones/zeros tensor.

   What this covers is staging, scheduling and shape machinery, NOT the pixel
   arithmetic: [Direct] and [Symbolic] instantiate the same [Compute] functor,
   so a layer_norm that forgot to subtract the mean passes here. The
   independent oracles are the ATen walk and the hand-computed values in
   test/native/compute_test.ml. *)

module M = struct
  module W = Norm.LayerNorm.Walk (Walk_limits.L)
  include W

  type subject = Native_subject.t

  let target = "layer_norm"

  let build pcg c =
    let shape = W.shape c in
    let affine_shape = W.affine_shape c in
    let x, pcg = Native_tensor.synth pcg shape in
    (* Synthesised in the order the builder declares the inputs, which is the
       order [Graph.inputs] comes back in -- the two lists are zipped below. *)
    let synth_opt present pcg =
      if present then
        let t, pcg = Native_tensor.synth pcg affine_shape in
        (Some t, pcg)
      else (None, pcg)
    in
    let weight, pcg = synth_opt (W.weight_present c) pcg in
    let bias, pcg = synth_opt (W.bias_present c) pcg in
    let g =
      Err.or_raise ~pp_error:Graph_builder.pp_error
        Graph_builder.(
          build ~name:"layer_norm" ~outputs:(fun r -> [ r ])
          @@
          let* xi = input ~shape ~name:"x" () in
          let opt_input present name =
            if present then
              let* i = input ~shape:affine_shape ~name () in
              return (Some i)
            else return None
          in
          let* wi = opt_input (W.weight_present c) "weight" in
          let* bi = opt_input (W.bias_present c) "bias" in
          layer_norm ~name:"out" (W.params c) ~x:xi ?weight:wi ?bias:bi ())
    in
    let tensors = (x :: Option.to_list weight) @ Option.to_list bias in
    ( {
        Native_subject.target;
        graph = g;
        inputs = List.combine g.Graph_ir.Graph.inputs tensors;
      },
      pcg )
end
