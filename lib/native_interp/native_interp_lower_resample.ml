(* Resampling-family operator dispatch for [Native_interp_lower]: the
   patch-extraction ops ([im2col]/[col2im]) and the three [upsample_*] ops,
   which resize the spatial axes. Like every family it returns [None] for
   targets it does not own; the lowerer owns the one unsupported-operator
   fallback. Split out of the shape family under the tracked file-size
   ceiling (scripts/check-file-size.sh). *)

open Pytorch_types
open Native_interp_decode
open Native_interp_decode_shape

let targets =
  [
    "torch.ops.aten.col2im.default";
    "torch.ops.aten.im2col.default";
    "torch.ops.aten.upsample_bicubic2d.vec";
    "torch.ops.aten.upsample_bilinear2d.vec";
    "torch.ops.aten.upsample_nearest2d.vec";
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
       | "torch.ops.aten.col2im.default" ->
           let op = "col2im.default" in
           let out_h, out_w =
             hw2 esc `Output_size (ints_arg esc node "output_size")
           in
           let kh, kw =
             hw2 esc `Kernel_size (ints_arg esc node "kernel_size")
           in
           let dh, dw = hw2 esc `Dilation (ints_arg esc node "dilation") in
           let ph, pw = hw2 esc `Padding (ints_arg esc node "padding") in
           let sh, sw = hw2 esc `Stride (ints_arg esc node "stride") in
           let window =
             Im2col.Params.
               {
                 h =
                   {
                     Im2col.Window.kernel =
                       extent esc ~op ~param:`Kernel_size kh;
                     dilation = pos esc ~op ~param:`Dilation dh;
                     pad = nonneg esc ~op ~param:`Padding ph;
                     stride = pos esc ~op ~param:`Stride sh;
                   };
                 w =
                   {
                     Im2col.Window.kernel =
                       extent esc ~op ~param:`Kernel_size kw;
                     dilation = pos esc ~op ~param:`Dilation dw;
                     pad = nonneg esc ~op ~param:`Padding pw;
                     stride = pos esc ~op ~param:`Stride sw;
                   };
               }
           in
           let params =
             Im2col.Col2im.
               {
                 window;
                 output_h = extent esc ~op ~param:`Output_size out_h;
                 output_w = extent esc ~op ~param:`Output_size out_w;
               }
           in
           let* y' = col2im params (get "self") in
           let* y = permute perm_nhwc_to_nchw y' in
           return [ y ]
       | "torch.ops.aten.im2col.default" ->
           let op = "im2col.default" in
           let kh, kw =
             hw2 esc `Kernel_size (ints_arg esc node "kernel_size")
           in
           let dh, dw = hw2 esc `Dilation (ints_arg esc node "dilation") in
           let ph, pw = hw2 esc `Padding (ints_arg esc node "padding") in
           let sh, sw = hw2 esc `Stride (ints_arg esc node "stride") in
           let params =
             Im2col.Params.
               {
                 h =
                   {
                     Im2col.Window.kernel =
                       extent esc ~op ~param:`Kernel_size kh;
                     dilation = pos esc ~op ~param:`Dilation dh;
                     pad = nonneg esc ~op ~param:`Padding ph;
                     stride = pos esc ~op ~param:`Stride sh;
                   };
                 w =
                   {
                     Im2col.Window.kernel =
                       extent esc ~op ~param:`Kernel_size kw;
                     dilation = pos esc ~op ~param:`Dilation dw;
                     pad = nonneg esc ~op ~param:`Padding pw;
                     stride = pos esc ~op ~param:`Stride sw;
                   };
               }
           in
           let* x = permute perm_nchw_to_nhwc (get "self") in
           let* y = im2col params x in
           return [ y ]
       (* Schema: `upsample_bilinear2d.vec(Tensor input, SymInt[]? output_size,
         bool align_corners, float[]? scale_factors)`. Exactly one of
         [output_size]/[scale_factors] is ever given (ATen's own
         `compute_output_size` rejects both-or-neither); a [scale_factors]
         node is resolved to an explicit size here, at import time, using
         ATen's own `floor(input_size * scale_factor)` -- the Native op only
         ever sees a concrete size, since Native shapes are static. Same
         resolution [Op_bridge]'s arm performs, restated here only because
         this importer reads serialized metadata where that one reads a live
         tensor. *)
       (* Schema: `upsample_bicubic2d.vec(Tensor input, SymInt[]? output_size,
         bool align_corners, float[]? scale_factors)` -- same shape as
         [upsample_bilinear2d.vec] just below, including the output_size/
         scale_factors resolution; only the Native op differs. *)
       | "torch.ops.aten.upsample_bicubic2d.vec" ->
           let x_name = tensor_name esc node "input" in
           let _n, _c, in_h, in_w =
             sizes_rank_4 esc ~tensor:x_name
               (static_sizes esc ~tensor:x_name
                  (tensor_meta esc graph ~ssa:x_name
                     ~role:`Upsample_bicubic2d_input))
           in
           let output_size = ints_arg esc node "output_size" in
           let scale_factors = floats_arg esc node "scale_factors" in
           let align_corners = bool_arg esc node "align_corners" in
           let out_h, out_w =
             resolve_upsample_size esc ~op:node.target ~in_h ~in_w output_size
               scale_factors
           in
           let params =
             {
               Resize.Bicubic2d.output_size =
                 {
                   h = pos esc ~op:node.target ~param:`Output_size out_h;
                   w = pos esc ~op:node.target ~param:`Output_size out_w;
                 };
               align_corners;
             }
           in
           let* x = permute perm_nchw_to_nhwc (get "input") in
           let* y = upsample_bicubic2d params x in
           let* y = permute perm_nhwc_to_nchw y in
           return [ y ]
       | "torch.ops.aten.upsample_bilinear2d.vec" ->
           let x_name = tensor_name esc node "input" in
           let _n, _c, in_h, in_w =
             sizes_rank_4 esc ~tensor:x_name
               (static_sizes esc ~tensor:x_name
                  (tensor_meta esc graph ~ssa:x_name
                     ~role:`Upsample_bilinear2d_input))
           in
           let output_size = ints_arg esc node "output_size" in
           let scale_factors = floats_arg esc node "scale_factors" in
           let align_corners = bool_arg esc node "align_corners" in
           let out_h, out_w =
             resolve_upsample_size esc ~op:node.target ~in_h ~in_w output_size
               scale_factors
           in
           let params =
             {
               Resize.Bilinear2d.output_size =
                 {
                   h = pos esc ~op:node.target ~param:`Output_size out_h;
                   w = pos esc ~op:node.target ~param:`Output_size out_w;
                 };
               align_corners;
             }
           in
           let* x = permute perm_nchw_to_nhwc (get "input") in
           let* y = upsample_bilinear2d params x in
           let* y = permute perm_nhwc_to_nchw y in
           return [ y ]
       (* No [align_corners] at all, unlike [upsample_bilinear2d.vec] just
         above -- see [Resize.Nearest_axis]'s module doc. Otherwise the
         identical output_size/scale_factors resolution. *)
       | "torch.ops.aten.upsample_nearest2d.vec" ->
           let x_name = tensor_name esc node "input" in
           let _n, _c, in_h, in_w =
             sizes_rank_4 esc ~tensor:x_name
               (static_sizes esc ~tensor:x_name
                  (tensor_meta esc graph ~ssa:x_name
                     ~role:`Upsample_nearest2d_input))
           in
           let output_size = ints_arg esc node "output_size" in
           let scale_factors = floats_arg esc node "scale_factors" in
           let out_h, out_w =
             resolve_upsample_size esc ~op:node.target ~in_h ~in_w output_size
               scale_factors
           in
           let params =
             {
               Resize.Nearest2d.output_size =
                 {
                   h = pos esc ~op:node.target ~param:`Output_size out_h;
                   w = pos esc ~op:node.target ~param:`Output_size out_w;
                 };
             }
           in
           let* x = permute perm_nchw_to_nhwc (get "input") in
           let* y = upsample_nearest2d params x in
           let* y = permute perm_nhwc_to_nchw y in
           return [ y ]
       | _ -> assert false)
