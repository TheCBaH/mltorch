(* Patch-extraction and resampling family (col2im/im2col/upsample) op-dispatch arms for [Op_bridge], split from op_bridge_shape.ml. See
   op_bridge.ml for the [dispatch] facade that folds every family module
   (including this one) into one lookup. *)

open Pytorch_types
open Op_bridge_error
open Op_bridge_decode
module D = Interp_decode

let dispatch ~(aten_env : aten_env) (node : Node.t) :
    (Graph_ir.graph * (Graph_ir.Tensor_id.t * Tensor.packed) list, error) Err.t
    option =
  match node.target with
  | "torch.ops.aten.col2im.default" ->
      Some
        (let op = "col2im.default" in
         let* x = native_tensor_arg aten_env node "self" in
         let* output_size = ints_arg node "output_size" in
         let* kernel_size = ints_arg node "kernel_size" in
         let* dilation = ints_arg node "dilation" in
         let* padding = ints_arg node "padding" in
         let* stride = ints_arg node "stride" in
         let* out_h, out_w = hw2 "output_size" output_size in
         let* kh, kw = hw2 "kernel_size" kernel_size in
         let* dh, dw = hw2 "dilation" dilation in
         let* ph, pw = hw2 "padding" padding in
         let* sh, sw = hw2 "stride" stride in
         let* out_h = extent ~op ~param:`Output_size out_h in
         let* out_w = extent ~op ~param:`Output_size out_w in
         let* kh = extent ~op ~param:`Kernel_size kh in
         let* kw = extent ~op ~param:`Kernel_size kw in
         let* dh = pos ~op ~param:`Dilation dh in
         let* dw = pos ~op ~param:`Dilation dw in
         let* ph = nonneg ~op ~param:`Padding ph in
         let* pw = nonneg ~op ~param:`Padding pw in
         let* sh = pos ~op ~param:`Stride sh in
         let* sw = pos ~op ~param:`Stride sw in
         let window =
           Im2col.Params.
             {
               h =
                 {
                   Im2col.Window.kernel = kh;
                   dilation = dh;
                   pad = ph;
                   stride = sh;
                 };
               w =
                 {
                   Im2col.Window.kernel = kw;
                   dilation = dw;
                   pad = pw;
                   stride = sw;
                 };
             }
         in
         let params =
           Im2col.Col2im.{ window; output_h = out_h; output_w = out_w }
         in
         build_g ~name:"col2im" [ x ] (function
           | [ x_id ] ->
               let open Graph_builder in
               let* y' = col2im params x_id in
               let+ y = permute perm_nhwc_to_nchw y' in
               [ y ]
           | operands -> arity_mismatch ~op:node.Node.target operands))
  | "torch.ops.aten.im2col.default" ->
      Some
        (let op = "im2col.default" in
         let* x = native_tensor_arg aten_env node "self" in
         let* kernel_size = ints_arg node "kernel_size" in
         let* dilation = ints_arg node "dilation" in
         let* padding = ints_arg node "padding" in
         let* stride = ints_arg node "stride" in
         let* kh, kw = hw2 "kernel_size" kernel_size in
         let* dh, dw = hw2 "dilation" dilation in
         let* ph, pw = hw2 "padding" padding in
         let* sh, sw = hw2 "stride" stride in
         let* kh = extent ~op ~param:`Kernel_size kh in
         let* kw = extent ~op ~param:`Kernel_size kw in
         let* dh = pos ~op ~param:`Dilation dh in
         let* dw = pos ~op ~param:`Dilation dw in
         let* ph = nonneg ~op ~param:`Padding ph in
         let* pw = nonneg ~op ~param:`Padding pw in
         let* sh = pos ~op ~param:`Stride sh in
         let* sw = pos ~op ~param:`Stride sw in
         let params =
           Im2col.Params.
             {
               h =
                 {
                   Im2col.Window.kernel = kh;
                   dilation = dh;
                   pad = ph;
                   stride = sh;
                 };
               w =
                 {
                   Im2col.Window.kernel = kw;
                   dilation = dw;
                   pad = pw;
                   stride = sw;
                 };
             }
         in
         build_g ~name:"im2col" [ x ] (function
           | [ x_id ] ->
               let open Graph_builder in
               let* x' = permute perm_nchw_to_nhwc x_id in
               let+ y = im2col params x' in
               [ y ]
           | operands -> arity_mismatch ~op:node.Node.target operands))
  | "torch.ops.aten.upsample_bicubic2d.vec" ->
      Some
        (let* aten_x = tensor_arg aten_env node "input" in
         let* () = require_rank "input" ~expected:4 aten_x in
         let* output_size = ints_arg node "output_size" in
         let* scale_factors = floats_arg node "scale_factors" in
         let* align_corners = bool_arg node "align_corners" in
         let w_shape = Aten_tensor.shape aten_x in
         let in_h = w_shape.(2) and in_w = w_shape.(3) in
         let* out_h, out_w =
           resolve_upsample_size ~op:"upsample_bicubic2d.vec" ~in_h ~in_w
             output_size scale_factors
         in
         let* h = pos ~op:node.Node.target ~param:`Output_size out_h in
         let* w = pos ~op:node.Node.target ~param:`Output_size out_w in
         let params =
           { Resize.Bicubic2d.output_size = { h; w }; align_corners }
         in
         let* x = native_of_aten "input" aten_x in
         build_g ~name:"upsample_bicubic2d_relayout" [ x ] (function
           | [ x_id ] ->
               let open Graph_builder in
               let* x' = permute perm_nchw_to_nhwc x_id in
               let* y' = upsample_bicubic2d params x' in
               let+ y = permute perm_nhwc_to_nchw y' in
               [ y ]
           | operands -> arity_mismatch ~op:node.Node.target operands))
  | "torch.ops.aten.upsample_bilinear2d.vec" ->
      Some
        (let* aten_x = tensor_arg aten_env node "input" in
         let* () = require_rank "input" ~expected:4 aten_x in
         let* output_size = ints_arg node "output_size" in
         let* scale_factors = floats_arg node "scale_factors" in
         let* align_corners = bool_arg node "align_corners" in
         let w_shape = Aten_tensor.shape aten_x in
         let in_h = w_shape.(2) and in_w = w_shape.(3) in
         let* out_h, out_w =
           resolve_upsample_size ~op:"upsample_bilinear2d.vec" ~in_h ~in_w
             output_size scale_factors
         in
         let* h = pos ~op:node.Node.target ~param:`Output_size out_h in
         let* w = pos ~op:node.Node.target ~param:`Output_size out_w in
         let params =
           { Resize.Bilinear2d.output_size = { h; w }; align_corners }
         in
         let* x = native_of_aten "input" aten_x in
         build_g ~name:"upsample_bilinear2d_relayout" [ x ] (function
           | [ x_id ] ->
               let open Graph_builder in
               let* x' = permute perm_nchw_to_nhwc x_id in
               let* y' = upsample_bilinear2d params x' in
               let+ y = permute perm_nhwc_to_nchw y' in
               [ y ]
           | operands -> arity_mismatch ~op:node.Node.target operands))
  (* Schema: `upsample_nearest2d.vec(Tensor input, SymInt[]? output_size,
     float[]? scale_factors)` -- no [align_corners] at all, unlike
     [upsample_bilinear2d.vec] just above (nothing to interpolate between;
     see [Resize.Nearest_axis]'s module doc). Otherwise the identical
     output_size/scale_factors resolution. *)
  | "torch.ops.aten.upsample_nearest2d.vec" ->
      Some
        (let* aten_x = tensor_arg aten_env node "input" in
         let* () = require_rank "input" ~expected:4 aten_x in
         let* output_size = ints_arg node "output_size" in
         let* scale_factors = floats_arg node "scale_factors" in
         let w_shape = Aten_tensor.shape aten_x in
         let in_h = w_shape.(2) and in_w = w_shape.(3) in
         let* out_h, out_w =
           resolve_upsample_size ~op:"upsample_nearest2d.vec" ~in_h ~in_w
             output_size scale_factors
         in
         let* h = pos ~op:node.Node.target ~param:`Output_size out_h in
         let* w = pos ~op:node.Node.target ~param:`Output_size out_w in
         let params = { Resize.Nearest2d.output_size = { h; w } } in
         let* x = native_of_aten "input" aten_x in
         build_g ~name:"upsample_nearest2d_relayout" [ x ] (function
           | [ x_id ] ->
               let open Graph_builder in
               let* x' = permute perm_nchw_to_nhwc x_id in
               let* y' = upsample_nearest2d params x' in
               let+ y = permute perm_nhwc_to_nchw y' in
               [ y ]
           | operands -> arity_mismatch ~op:node.Node.target operands))
  | _ -> None
