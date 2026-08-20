(* Assembles Sdpa's walk: config space from Attention.Sdpa.Walk, a graph with
   three or four inputs depending on whether the config has a mask.
   [Graph_ir]'s [Sdpa] carries [mask : Tensor_ref.t option] and [Eval_op]
   fills the absent case at the all-ones shape (op8-impl.md F11), a
   genuinely different code path, so both must be walked.

   This covers staging, scheduling and shape machinery through the shared
   [Direct]/[Symbolic] [Compute] functor -- NOT the pixel arithmetic, which a
   formula error would pass here just as silently as it would for any other
   op (see op7-impl.md's caveat on [Rms_norm_nwalk]). The independent oracles
   are the hand-computed values and mutation battery in
   test/native/compute_test.ml and (against real ATen) commit 4's
   [Recipe_sdpa]. *)

module M = struct
  module W = Attention.Sdpa.Walk (Walk_limits.L)
  include W

  type subject = Native_subject.t

  let target = "sdpa"

  let build pcg c =
    let query_shape = W.query_shape c in
    let key_shape = W.key_shape c in
    let query, pcg = Native_tensor.synth pcg query_shape in
    let key, pcg = Native_tensor.synth pcg key_shape in
    let value, pcg = Native_tensor.synth pcg key_shape in
    let mask, pcg =
      if W.mask_present c then
        let m, pcg = Native_tensor.synth pcg (W.mask_shape c) in
        (Some m, pcg)
      else (None, pcg)
    in
    let g =
      Err.or_raise ~pp_error:Graph_builder.pp_error
        Graph_builder.(
          build ~name:"sdpa" ~outputs:(fun r -> [ r ])
          @@
          let* qi = input ~shape:query_shape ~name:"query" () in
          let* ki = input ~shape:key_shape ~name:"key" () in
          let* vi = input ~shape:key_shape ~name:"value" () in
          let* mi =
            if W.mask_present c then
              let* m = input ~shape:(W.mask_shape c) ~name:"mask" () in
              return (Some m)
            else return None
          in
          sdpa ~name:"out" (W.params c) ~query:qi ~key:ki ~value:vi ?mask:mi ())
    in
    let tensors = [ query; key; value ] @ Option.to_list mask in
    ( {
        Native_subject.target;
        graph = g;
        inputs = List.combine g.Graph_ir.Graph.inputs tensors;
      },
      pcg )
end
