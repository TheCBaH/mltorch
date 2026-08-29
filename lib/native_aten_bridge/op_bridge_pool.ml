(* Pooling family op-dispatch arms for [Op_bridge], split from op_bridge.ml. See op_bridge.ml for the [dispatch] facade that
   folds every family module (including this one) into one lookup. *)

open Pytorch_types
open Op_bridge_error
open Op_bridge_decode

let dispatch ~(aten_env : aten_env) (node : Node.t) :
    (Graph_ir.graph * (Graph_ir.Tensor_id.t * Tensor.packed) list, error) Err.t
    option =
  match node.target with
  | "torch.ops.aten.max_pool2d.default" ->
      Some
        (let* aten_x = tensor_arg aten_env node "self" in
         let* ceil_mode = pool_dilation_and_ceil_mode node in
         let* kernel_size = ints_arg node "kernel_size" in
         let* stride = pool_stride kernel_size node in
         let* padding = ints_arg ~default:[ 0; 0 ] node "padding" in
         let* kh, kw = hw2 "kernel_size" kernel_size in
         let* sh, sw = hw2 "stride" stride in
         let* ph, pw = hw2 "padding" padding in
         let* x = native_of_aten "self" aten_x in
         let* params =
           make_pool_params ~op:node.Node.target ~ceil_mode kh kw sh sw ph pw
         in
         try
           build_g ~name:"max_pool2d_relayout" [ x ] (function
             | [ x_id ] ->
                 let open Graph_builder in
                 let* x' = permute perm_nchw_to_nhwc x_id in
                 let* y' = max_pool2d params x' in
                 let+ y = permute perm_nhwc_to_nchw y' in
                 [ y ]
             | _ -> assert false)
         with Invalid_argument msg -> fail (`Validation_failure msg))
  | "torch.ops.aten.adaptive_avg_pool2d.default" ->
      Some
        (let* aten_x = tensor_arg aten_env node "self" in
         let got = aten_rank aten_x in
         let* () =
           if got = 3 || got = 4 then return ()
           else fail (`Adaptive_pool_rank { Adaptive_pool_rank.got })
         in
         let* output_size = ints_arg node "output_size" in
         let* out_h, out_w =
           match output_size with
           | [ h; w ] -> return (h, w)
           | values -> fail (`Invalid_hw_arg { name = "output_size"; values })
         in
         let* h = pos ~op:node.Node.target ~param:`Output_size out_h in
         let* w = pos ~op:node.Node.target ~param:`Output_size out_w in
         let* x = native_of_aten "self" aten_x in
         let params = { Pool.AdaptiveAvgPool2d.output_size = { h; w } } in
         build_g ~name:"adaptive_avg_pool2d_relayout" [ x ] (function
           | [ x_id ] ->
               let open Graph_builder in
               let* x' = permute perm_nchw_to_nhwc x_id in
               let* y' = adaptive_avg_pool2d params x' in
               let+ y = permute perm_nhwc_to_nchw y' in
               [ y ]
           | _ -> assert false))
  | "torch.ops.aten.max_pool2d_with_indices.default" ->
      Some
        ((* Two ATen outputs (values, indices). We materialise both, relayout the
            value back to NCHW as the graph output, and route the dead indices
            edge into a Discard sink (see .ai/native_multi_output_design.md). *)
         let* aten_x = tensor_arg aten_env node "self" in
         let* ceil_mode = pool_dilation_and_ceil_mode node in
         let* kernel_size = ints_arg node "kernel_size" in
         let* stride = pool_stride kernel_size node in
         let* padding = ints_arg ~default:[ 0; 0 ] node "padding" in
         let* kh, kw = hw2 "kernel_size" kernel_size in
         let* sh, sw = hw2 "stride" stride in
         let* ph, pw = hw2 "padding" padding in
         let* x = native_of_aten "self" aten_x in
         let* params =
           make_pool_params ~op:node.Node.target ~ceil_mode kh kw sh sw ph pw
         in
         try
           build_g ~name:"max_pool2d_with_indices_relayout" [ x ] (function
             | [ x_id ] ->
                 let open Graph_builder in
                 let* x' = permute perm_nchw_to_nhwc x_id in
                 let* values, indices = max_pool2d_with_indices params x' in
                 let* () = discard indices in
                 let+ y = permute perm_nhwc_to_nchw values in
                 [ y ]
             | _ -> assert false)
         with Invalid_argument msg -> fail (`Validation_failure msg))
  | _ -> None
