(* Shape/view-family operator dispatch for [Native_interp_lower].  Each family
   returns [None] for targets it does not own; the lowerer owns the one
   unsupported-operator fallback. *)

open Pytorch_types
open Native_interp_error
open Native_interp_decode

let targets =
  [
    "torch.ops.aten._unsafe_view.default";
    "torch.ops.aten.alias.default";
    "torch.ops.aten.cat.default";
    "torch.ops.aten.pad.default";
    "torch.ops.aten.permute.default";
    "torch.ops.aten.select.int";
    "torch.ops.aten.slice.Tensor";
    "torch.ops.aten.split.Tensor";
    "torch.ops.aten.split_with_sizes.default";
    "torch.ops.aten.stack.default";
    "torch.ops.aten.transpose.int";
    "torch.ops.aten.unbind.int";
    "torch.ops.aten.unsqueeze.default";
    "torch.ops.aten.upsample_bilinear2d.vec";
    "torch.ops.aten.upsample_nearest2d.vec";
    "torch.ops.aten.view.default";
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
       | "torch.ops.aten.permute.default" ->
           let x_name = tensor_name esc node "self" in
           let rank =
             meta_rank (tensor_meta esc graph ~ssa:x_name ~role:`Permute_input)
           in
           let* y =
             permute
               (native_perm esc ~tensor:x_name ~rank (ints_arg esc node "dims"))
               (get "self")
           in
           return [ y ]
       (* Reuses [native_perm], the same permute machinery [permute.default]
         builds on, rather than a new builder -- outer padding axes stay
         identity because [native_perm] already does that. Equal dims are a
         real identity transpose, not special-cased away: [List.init] produces
         the identity list, and it lowers like any other permutation. *)
       | "torch.ops.aten.transpose.int" ->
           let x_name = tensor_name esc node "self" in
           let rank =
             meta_rank
               (tensor_meta esc graph ~ssa:x_name ~role:`Transpose_input)
           in
           let norm_dim d =
             let d = if d < 0 then d + rank else d in
             if d < 0 || d >= rank then
               malformed esc (`Axis_out_of_range { axis = d; rank });
             d
           in
           let d0 = norm_dim (int_arg esc node "dim0") in
           let d1 = norm_dim (int_arg esc node "dim1") in
           let dims =
             List.init rank (fun i ->
                 if i = d0 then d1 else if i = d1 then d0 else i)
           in
           let* y =
             permute (native_perm esc ~tensor:x_name ~rank dims) (get "self")
           in
           return [ y ]
       (* The only arm whose output count is not fixed by the op. By the time it
         runs, [output_names] has already bounded and flattened the serialized
         names, so what is left is to normalize the axis, derive the count from
         already-validated metadata, and CHECK the two against each other before
         allocating anything.
         The check has to precede the builder, not merely precede binding: both
         the name list and the extent are model data, and [add_env]'s
         [Invalid_argument] is an invariant about this module rather than a
         report about the graph. *)
       (* The pad list is indexed FROM the innermost dimension, so the rank is
         load-bearing: with the wrong one the same list pads different axes and
         still produces a plausible graph. It comes from the serialized
         metadata here and from the live tensor on the bridge, but the
         un-reversal itself is [Pad.Pad.params_of_aten] on both sides. *)
       | "torch.ops.aten.pad.default" ->
           let x_name = tensor_name esc node "self" in
           let rank =
             meta_rank (tensor_meta esc graph ~ssa:x_name ~role:`Pad_input)
           in
           let params =
             Err.Escape.or_throw esc
               (Pad.Pad.params_of_aten ~rank ~pad:(ints_arg esc node "pad")
                  ~mode:(string_arg esc ~default:"constant" node "mode")
                  ~value:(float_opt_arg_opt esc node "value"))
           in
           let* y = pad params (get "self") in
           return [ y ]
       (* The variadic Native [Concat] op, direct -- see [Op_bridge]'s
         "torch.ops.aten.cat.default" arm for why the rank check runs BEFORE
         [Concat]'s own axis-agreement shape rule rather than being left to
         it. *)
       | "torch.ops.aten.cat.default" -> (
           match tensor_names_arg esc node "tensors" with
           | [] -> malformed esc (`Concat_no_tensors node.target)
           | name0 :: _ as names ->
               let rank =
                 meta_rank
                   (tensor_meta esc graph ~ssa:name0 ~role:`Concat_input)
               in
               List.iter
                 (fun name ->
                   let got =
                     meta_rank
                       (tensor_meta esc graph ~ssa:name ~role:`Concat_input)
                   in
                   if got <> rank then
                     malformed esc
                       (`Concat_rank_mismatch
                          {
                            Concat_rank_mismatch.op = node.target;
                            first = rank;
                            other = got;
                          }))
                 names;
               let d =
                 let dim = int_arg esc ~default:0 node "dim" in
                 let dim = if dim < 0 then dim + rank else dim in
                 if dim < 0 || dim >= rank then
                   malformed esc (`Axis_out_of_range { axis = dim; rank });
                 dim
               in
               let axis = List.nth (used_axes_for esc ~tensor:name0 rank) d in
               let* y =
                 concat { Concat.Concat.axis }
                   (List.map (env_find esc env) names)
               in
               return [ y ])
       (* One [Stack] node: inserts a size-1 axis per operand at [axis], then
         joins them -- see [Op_bridge]'s arm for why (ATen's own definition),
         and [unsqueeze.default] below for the rank+1 [dim] convention this
         reuses via the [d > rank] bound (not [>=]). *)
       | "torch.ops.aten.stack.default" -> (
           match tensor_names_arg esc node "tensors" with
           | [] -> malformed esc (`Concat_no_tensors node.target)
           | name0 :: _ as names ->
               let rank =
                 meta_rank (tensor_meta esc graph ~ssa:name0 ~role:`Stack_input)
               in
               List.iter
                 (fun name ->
                   let got =
                     meta_rank
                       (tensor_meta esc graph ~ssa:name ~role:`Stack_input)
                   in
                   if got <> rank then
                     malformed esc
                       (`Concat_rank_mismatch
                          {
                            Concat_rank_mismatch.op = node.target;
                            first = rank;
                            other = got;
                          }))
                 names;
               let d =
                 let dim = int_arg esc ~default:0 node "dim" in
                 let dim = if dim < 0 then dim + rank + 1 else dim in
                 if dim < 0 || dim > rank then
                   malformed esc (`Axis_out_of_range { axis = dim; rank });
                 dim
               in
               let axis =
                 List.nth (used_axes_for esc ~tensor:name0 (rank + 1)) d
               in
               let* y =
                 stack { Concat.Stack.axis } (List.map (env_find esc env) names)
               in
               return [ y ])
       (* One [Select] node: picks index [idx] along the normalized axis and
         drops it. The normalized ATen-level position [d] is needed once, as
         the frame axis via [used_axes_for], the same shape [transpose.int]'s
         [norm_dim] takes below. *)
       | "torch.ops.aten.select.int" ->
           let x_name = tensor_name esc node "self" in
           let rank =
             meta_rank (tensor_meta esc graph ~ssa:x_name ~role:`Select_input)
           in
           let d =
             let dim = int_arg esc node "dim" in
             let dim = if dim < 0 then dim + rank else dim in
             if dim < 0 || dim >= rank then
               malformed esc (`Axis_out_of_range { axis = dim; rank });
             dim
           in
           let axis = List.nth (used_axes_for esc ~tensor:x_name rank) d in
           let shape = tensor_shape esc graph x_name in
           let extent = Vec6.get shape axis in
           let idx =
             resolve_select_index esc ~extent ~index:(int_arg esc node "index")
           in
           let* y = select { Split.Select.axis; index = idx } (get "self") in
           return [ y ]
       (* Legalized to [Reshape] alone: inserting a size-1 axis never changes
         the linearized data order, so no [Slice] is needed, unlike
         [select.int]'s axis removal. [dim] is judged against rank+1 valid
         positions (the OUTPUT rank), so this does not reuse the local
         [norm_dim] pattern above verbatim -- the upper bound is relaxed by
         one, the same adjustment [Op_bridge.norm_unsqueeze_dim] makes. *)
       | "torch.ops.aten.unsqueeze.default" ->
           let x_name = tensor_name esc node "self" in
           let rank =
             meta_rank
               (tensor_meta esc graph ~ssa:x_name ~role:`Unsqueeze_input)
           in
           let d =
             let dim = int_arg esc node "dim" in
             let dim = if dim < 0 then dim + rank + 1 else dim in
             if dim < 0 || dim > rank then
               malformed esc (`Axis_out_of_range { axis = dim; rank });
             dim
           in
           let shape = tensor_shape esc graph x_name in
           let aten_list = Array.to_list (Aten_shape.to_aten ~rank shape) in
           let front = List.filteri (fun i _ -> i < d) aten_list in
           let back = List.filteri (fun i _ -> i >= d) aten_list in
           let out_sizes =
             List.map (fun x -> SymInt.Int x) (front @ [ 1 ] @ back)
           in
           let* y =
             reshape
               { Reshape.Reshape.shape = shape_of_sizes esc x_name out_sizes }
               (get "self")
           in
           return [ y ]
       (* Everything after [Aten_shape.resolve_slice] is shared with the bridge
         arm; what differs is where the extent comes from, and that is the point
         of a shared resolver rather than two normalizations. Here it is the
         SERIALIZED shape, which is also where an unresolved symbolic dimension
         is already refused; on the bridge it is the live tensor. *)
       | "torch.ops.aten.slice.Tensor" ->
           let x_name = tensor_name esc node "self" in
           let rank =
             meta_rank (tensor_meta esc graph ~ssa:x_name ~role:`Slice_input)
           in
           let axis =
             match
               axes_for_rank esc ~tensor:x_name rank
                 [ int_arg esc ~default:0 node "dim" ]
             with
             | [ a ] -> a
             | _ ->
                 invalid_arg "Native_interp: axes_for_rank lost its singleton"
           in
           let extent = Vec6.get (tensor_shape esc graph x_name) axis in
           let start = int_opt_arg_opt esc node "start" in
           let stop = int_opt_arg_opt esc node "end" in
           let step = int_arg esc ~default:1 node "step" in
           let bounds = resolve_slice_arg esc ~extent ~start ~stop ~step in
           let* y =
             slice
               {
                 Split.Slice.axis;
                 start = bounds.Aten_shape.Slice_bounds.start;
                 stop = bounds.Aten_shape.Slice_bounds.stop;
                 step = bounds.Aten_shape.Slice_bounds.step;
               }
               (get "self")
           in
           return [ y ]
       | "torch.ops.aten.unbind.int" ->
           let x_name = tensor_name esc node "self" in
           let rank =
             meta_rank (tensor_meta esc graph ~ssa:x_name ~role:`Unbind_input)
           in
           let axis =
             match
               axes_for_rank esc ~tensor:x_name rank
                 [ int_arg esc ~default:0 node "dim" ]
             with
             | [ a ] -> a
             | _ ->
                 invalid_arg "Native_interp: axes_for_rank lost its singleton"
           in
           (* No arity check here: [bind] does it against the ids actually
             produced, which is both stronger and total over ops. *)
           unbind { Split.Unbind.axis } (get "self")
       (* Divides [axis] into contiguous windows of [split_sizes], KEEPING the
         axis in every output, unlike [unbind.int]. Same shape as that arm --
         rank from the input's own metadata, [dim] resolved through
         [axes_for_rank] -- with [split_sizes] read as an int list the way
         [view.default]'s [size] is; [Split.Split_with_sizes.output_shapes] is
         what checks the sizes are positive and sum to the axis's extent, not
         this arm. No arity check here either, for the reason [unbind.int]'s
         comment gives. *)
       | "torch.ops.aten.split_with_sizes.default" ->
           let x_name = tensor_name esc node "self" in
           let rank =
             meta_rank
               (tensor_meta esc graph ~ssa:x_name ~role:`Split_with_sizes_input)
           in
           let axis =
             match
               axes_for_rank esc ~tensor:x_name rank
                 [ int_arg esc ~default:0 node "dim" ]
             with
             | [ a ] -> a
             | _ ->
                 invalid_arg "Native_interp: axes_for_rank lost its singleton"
           in
           let sizes = ints_arg esc node "split_sizes" in
           split_with_sizes { Split.Split_with_sizes.axis; sizes } (get "self")
       (* `split.Tensor(self, split_size, dim)` -- the equal-chunk-size
         sibling of [split_with_sizes.default] just above, binding to the
         *same* [Split_with_sizes] node via [split_tensor_sizes] (which also
         bounds the derived chunk count before building the list -- see its
         own comment in native_interp_decode.ml). *)
       | "torch.ops.aten.split.Tensor" ->
           let x_name = tensor_name esc node "self" in
           let rank =
             meta_rank
               (tensor_meta esc graph ~ssa:x_name ~role:`Split_tensor_input)
           in
           let axis =
             match
               axes_for_rank esc ~tensor:x_name rank
                 [ int_arg esc ~default:0 node "dim" ]
             with
             | [ a ] -> a
             | _ ->
                 invalid_arg "Native_interp: axes_for_rank lost its singleton"
           in
           let extent =
             (Vec6.get (tensor_shape esc graph x_name) axis :> int)
           in
           let split_size =
             pos esc ~op:node.target ~param:`Split_size
               (int_arg esc node "split_size")
           in
           let sizes =
             split_tensor_sizes esc ~extent ~split_size:(split_size :> int)
           in
           split_with_sizes { Split.Split_with_sizes.axis; sizes } (get "self")
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
       (* Both overloads share this body -- same argument names, same shared
         resolver, same reason op_bridge.ml's arm does (drift risk, exact
         target still visible in every diagnostic via [tensor]/[node.target]).
         [Identical] AFTER materialization; see .ai/native_aten_bridge_layout.md. *)
       | "torch.ops.aten._unsafe_view.default" | "torch.ops.aten.view.default"
         ->
           let tensor = tensor_name esc node "self" in
           let shape = tensor_shape esc graph tensor in
           let* y =
             reshape
               {
                 Reshape.Reshape.shape =
                   resolve_view esc ~tensor shape (ints_arg esc node "size");
               }
               (get "self")
           in
           return [ y ]
       (* [alias(self) -> Tensor] is a pure identity view (same storage, same
         size/strides) -- ATen's own schema takes no argument beyond [self].
         Binds to the *existing* [Clone] node: paramless, shape-preserving,
         and already the node this repo uses for "same value, no change" (see
         [clone.default]'s own arm in native_interp_lower_compute.ml) --
         Native4D already elides it entirely (`.ai/native4d_design.md` §7.1),
         so this legalization costs nothing there either. *)
       | "torch.ops.aten.alias.default" ->
           let* y = clone (get "self") in
           return [ y ]
       (* Not mergeable with [addmm.default] below, though both build a [Linear]:
         there [self] IS the bias and is required, here the bias is optional,
         and the two weights are transposes of each other. *)
       | _ -> assert false)
