(* Pooling-family operator dispatch for [Native_interp_lower].  Each family
   returns [None] for targets it does not own; the lowerer owns the one
   unsupported-operator fallback. Split out of native_interp_lower_compute.ml
   (which crossed the tracked 1000-line ceiling, scripts/check-file-size.sh)
   rather than folded into it -- the [Op_bridge] side of this same family has
   its own file, op_bridge_pool.ml, for the same reason. *)

open Pytorch_types
open Native_interp_error
open Native_interp_decode
open Native_interp_decode_conv
open Native_interp_decode_shape

let targets =
  [
    "torch.ops.aten.adaptive_avg_pool2d.default";
    "torch.ops.aten.adaptive_max_pool2d.default";
    "torch.ops.aten.avg_pool2d.default";
    "torch.ops.aten.max_pool2d.default";
    "torch.ops.aten.max_pool2d_with_indices.default";
  ]

let dispatch ~ctx ~env (node : Node.t) =
  if not (List.mem node.target targets) then None
  else
    Some
      (let open Graph_builder in
       let esc = ctx.Native_interp_lower_context.esc in
       let graph = ctx.Native_interp_lower_context.graph in
       let get = Native_interp_lower_context.get ctx env node in
       match node.target with
       (* NCHW -> channel-last around the shared pooling compute, mirroring
          [Op_bridge_pool]'s own relayout; the pooled *value* is
          branch-selecting but continuous at the boundary
          ([Output_transfer]'s own note), so no separate correctness concern
          from moving it through a permute either side.
          [materialized_output_names] must not gain it, since that list is for
          nodes whose trailing outputs are dropped. *)
       | "torch.ops.aten.max_pool2d.default" ->
           let* x = permute perm_nchw_to_nhwc (get "self") in
           let* y = max_pool2d (pool_params esc node) x in
           let* y = permute perm_nhwc_to_nchw y in
           return [ y ]
       | "torch.ops.aten.adaptive_avg_pool2d.default" ->
           let x_name = tensor_name esc node "self" in
           let got =
             meta_rank
               (tensor_meta esc graph ~ssa:x_name
                  ~role:`Adaptive_avg_pool2d_input)
           in
           if (got :> int) <> 3 && (got :> int) <> 4 then
             malformed esc (`Adaptive_pool_rank { tensor = x_name; got });
           let out_h, out_w =
             match ints_arg esc node "output_size" with
             | [ h; w ] -> (h, w)
             | xs ->
                 malformed esc
                   (`Bad_arity
                      { Bad_arity.param = `Output_size; got = List.length xs })
           in
           let params =
             {
               Pool.AdaptiveAvgPool2d.output_size =
                 {
                   h = pos esc ~op:node.target ~param:`Output_size out_h;
                   w = pos esc ~op:node.target ~param:`Output_size out_w;
                 };
             }
           in
           let* x = permute perm_nchw_to_nhwc (get "self") in
           let* y = adaptive_avg_pool2d params x in
           let* y = permute perm_nhwc_to_nchw y in
           return [ y ]
       (* ATen has no value-only overload here (unlike [max_pool2d.default]'s
          genuine two-op split), so this always returns (values, indices);
          the indices edge is routed to the same [Discard] sink
          [max_pool2d_with_indices.default] uses just below, and
          [materialized_output_names] drops its serialized name from tracking
          the same way it does for that op (see native_interp_decode.ml). *)
       | "torch.ops.aten.adaptive_max_pool2d.default" ->
           let x_name = tensor_name esc node "self" in
           let got =
             meta_rank
               (tensor_meta esc graph ~ssa:x_name
                  ~role:`Adaptive_max_pool2d_input)
           in
           if (got :> int) <> 3 && (got :> int) <> 4 then
             malformed esc (`Adaptive_pool_rank { tensor = x_name; got });
           let out_h, out_w =
             match ints_arg esc node "output_size" with
             | [ h; w ] -> (h, w)
             | xs ->
                 malformed esc
                   (`Bad_arity
                      { Bad_arity.param = `Output_size; got = List.length xs })
           in
           let params =
             {
               Pool.AdaptiveMaxPool2d.output_size =
                 {
                   h = pos esc ~op:node.target ~param:`Output_size out_h;
                   w = pos esc ~op:node.target ~param:`Output_size out_w;
                 };
             }
           in
           let* x = permute perm_nchw_to_nhwc (get "self") in
           let* values, indices = adaptive_max_pool2d_with_indices params x in
           let* () = discard indices in
           let* values = permute perm_nhwc_to_nchw values in
           return [ values ]
       | "torch.ops.aten.avg_pool2d.default" ->
           let* x = permute perm_nchw_to_nhwc (get "self") in
           let* y = avg_pool2d (avg_pool_params esc node) x in
           let* y = permute perm_nhwc_to_nchw y in
           return [ y ]
       | "torch.ops.aten.max_pool2d_with_indices.default" ->
           let* x = permute perm_nchw_to_nhwc (get "self") in
           let* values, indices =
             max_pool2d_with_indices (pool_params esc node) x
           in
           let* () = discard indices in
           let* values = permute perm_nhwc_to_nchw values in
           return [ values ]
       | _ -> assert false)
