(* Shape/view family (permute/slice/cat/...) op-dispatch arms for [Op_bridge], split from op_bridge.ml. See op_bridge.ml for the [dispatch] facade that
   folds every family module (including this one) into one lookup. *)

open Pytorch_types
open Op_bridge_error
open Op_bridge_decode
module D = Interp_decode

let dispatch ~(aten_env : aten_env) (node : Node.t) :
    (Graph_ir.graph * (Graph_ir.Tensor_id.t * Tensor.packed) list, error) Err.t
    option =
  match node.target with
  | "torch.ops.aten.clone.default" ->
      Some
        (let* x = native_tensor_arg aten_env node "self" in
         (* Native's engine has exactly one physical layout per shape --
            there is no channels-last stride concept anywhere in the six-axis
            frame -- so a request to make the result CONTIGUOUS, or to
            PRESERVE whatever format the input already has, is always already
            true: every native tensor is already in that one dense layout.
            (Confirmed against the corpus: every `clone.default` occurrence
            across the 100-model sweep requests either no format or exactly
            `ContiguousFormat` -- see `.ai/pt2_model_support.md`.) A request
            for an actual different physical arrangement (channels-last) asks
            for something this engine cannot represent, so it is refused
            rather than silently treated as a no-op. *)
         let* mf =
           decode_result (D.memory_format_opt_arg_result node "memory_format")
         in
         match mf with
         | None | Some (Aten_memory_format.Contiguous | Preserve) ->
             build_g ~name:"clone" [ x ] (function
               | [ x_id ] ->
                   let open Graph_builder in
                   let+ y = clone x_id in
                   [ y ]
               | _ -> assert false)
         | Some Aten_memory_format.ChannelsLast ->
             fail (`Unsupported_memory_format `Channels_last)
         | Some Aten_memory_format.ChannelsLast3d ->
             fail (`Unsupported_memory_format `Channels_last_3d))
  (* `_to_copy(Tensor self, *, ScalarType? dtype=None, Layout? layout=None,
     Device? device=None, bool? pin_memory=None, bool non_blocking=False,
     MemoryFormat? memory_format=None) -> Tensor`. Restricted to the
     three-way [Pointwise.To_copy.target] domain the corpus actually uses
     (bool/float/long) -- see that module's own comment for why there is no
     Native value representation to target outside it. [dtype=None] means
     "keep self's own dtype", which this engine cannot distinguish from the
     float target (there is no per-tensor dtype tag on an ordinary compute
     edge -- see graph_builder.ml's "op-output edges are F32" note), so it
     is treated as [Float], the identity. [non_blocking] is a
     device-transfer hint, read-and-discarded the same way it is in
     [copy.default]'s arm above; [layout]/[device]/[pin_memory] are rejected
     unless omitted/default, the same [zeros.default] precedent;
     [memory_format] is rejected the same way [clone.default]'s own arm
     rejects a real relayout request. *)
  | "torch.ops.aten._to_copy.default" ->
      Some
        (let* x = native_tensor_arg aten_env node "self" in
         let* target =
           match D.find_arg node "dtype" with
           | None | Some (Argument.None _) -> return Pointwise.To_copy.Float
           | Some (Argument.Scalar_type Pytorch_types.ScalarType.BOOL) ->
               return Pointwise.To_copy.Bool
           | Some (Argument.Scalar_type Pytorch_types.ScalarType.FLOAT) ->
               return Pointwise.To_copy.Float
           | Some (Argument.Scalar_type Pytorch_types.ScalarType.LONG) ->
               return Pointwise.To_copy.Long
           | Some _ ->
               fail
                 (`Validation_failure
                    "_to_copy: only default/BOOL/FLOAT/LONG dtype are supported")
         in
         let reject_absent name =
           match D.find_arg node name with
           | None | Some (Argument.None _) -> return ()
           | Some _ ->
               fail
                 (`Validation_failure
                    ("_to_copy: " ^ name ^ " must be omitted or None"))
         in
         let* () = reject_absent "layout" in
         let* () =
           match D.find_arg node "device" with
           | None | Some (Argument.None _) -> return ()
           | Some (Argument.Device { Device.type_ = "cpu"; index = None }) ->
               return ()
           | Some _ ->
               fail
                 (`Validation_failure
                    "_to_copy: device must be omitted/None or CPU without an \
                     index")
         in
         let* () =
           match D.find_arg node "pin_memory" with
           | None | Some (Argument.None _) | Some (Argument.Bool false) ->
               return ()
           | Some _ ->
               fail
                 (`Validation_failure
                    "_to_copy: pin_memory must be omitted/None or false")
         in
         let* () = reject_absent "memory_format" in
         build_g ~name:"to_copy" [ x ] (function
           | [ x_id ] ->
               let open Graph_builder in
               let+ y = to_copy target x_id in
               [ y ]
           | _ -> assert false))
  (* `copy(Tensor self, Tensor src, bool non_blocking=False) -> Tensor`.
     [self]'s VALUE never matters -- real ATen's [copy_]/[copy] fully
     OVERWRITES [self] with [src] broadcast to [self]'s shape, so the result
     depends only on [src]'s values and [self]'s declared shape/dtype, never
     on what [self] held before. [non_blocking] is a device-transfer hint,
     irrelevant to this single-device engine, read-and-discarded the same
     way [implicit] is in [expand.default]'s arm below. Binds directly to
     the EXISTING [Pointwise.Expand] node -- the corpus's own occurrences
     are exclusively the functionalized form of `tensor[idx] = value`
     (`self` is a [select.int] view, immediately followed by a
     [select_scatter.default] writing the result back), where [self] and
     [src] always share one shape (verified: zero broadcasts, zero dtype
     casts across every occurrence in the 100-model corpus), but the
     translation is the fully general one regardless: `Expand{size=self's
     shape}` applied to [src] computes exactly ATen's own broadcast, not an
     overfit to the observed equal-shape case. The same "map onto an
     existing op after translating parameters" route [sub.Tensor]'s scalar
     form takes to [Add_scalar] -- one node, [copy.default]'s own identity
     not preserved in the graph, same as that arm's. *)
  | "torch.ops.aten.copy.default" ->
      Some
        (let* self_t = tensor_arg aten_env node "self" in
         let* (_ : bool) = bool_arg node "non_blocking" in
         let* target =
           Err.map_error
             (fun e -> `Aten_shape e)
             (Aten_shape.of_aten (Aten_tensor.shape self_t))
         in
         let* src = native_tensor_arg aten_env node "src" in
         build_g ~name:"copy" [ src ] (function
           | [ src_id ] ->
               let open Graph_builder in
               let+ y = expand { Pointwise.Expand.size = target } src_id in
               [ y ]
           | _ -> assert false))
  | "torch.ops.aten.permute.default" ->
      Some
        (let* t = tensor_arg aten_env node "self" in
         let rank = aten_rank t in
         let* dims = ints_arg node "dims" in
         let* perm = native_perm_of_aten ~op:"permute.default" ~rank dims in
         let* x = native_of_aten "self" t in
         build_g ~name:"permute" [ x ] (function
           | [ x_id ] ->
               let open Graph_builder in
               let+ y = permute perm x_id in
               [ y ]
           | _ -> assert false))
  (* Reuses [native_perm_of_aten], the same permute machinery
     [permute.default] builds on, rather than a new builder -- outer padding
     axes stay identity because that helper already does that. Equal dims are
     a real identity transpose, not special-cased away: [List.init] produces
     the identity list, and it lowers like any other permutation. *)
  | "torch.ops.aten.transpose.int" ->
      Some
        (let* t = tensor_arg aten_env node "self" in
         let rank = aten_rank t in
         let* dim0 = int_arg node "dim0" in
         let* dim1 = int_arg node "dim1" in
         let* d0 = norm_dim ~op:"transpose.int" ~rank dim0 in
         let* d1 = norm_dim ~op:"transpose.int" ~rank dim1 in
         let dims =
           List.init rank (fun i ->
               if i = d0 then d1 else if i = d1 then d0 else i)
         in
         let* perm = native_perm_of_aten ~op:"transpose.int" ~rank dims in
         let* x = native_of_aten "self" t in
         build_g ~name:"transpose" [ x ] (function
           | [ x_id ] ->
               let open Graph_builder in
               let+ y = permute perm x_id in
               [ y ]
           | _ -> assert false))
  (* The only arm returning a variable number of outputs, and the only one whose
     count is fixed by the OPERAND rather than the op. Every slice is exposed:
     unlike a fixed tuple's dead output, there is nothing here to drop, and
     [Verify.verify_node] requires exact cardinality for a dynamic list for
     precisely that reason.

     This must NOT call ATen's own unbind. ATen is the oracle
     [Interp_verify]/[Verify.verify_node] runs; the native side has to execute
     [Graph_ir.Unbind] through [Eval_direct] or the comparison is vacuous. *)
  | "torch.ops.aten.pad.default" ->
      Some
        (let* aten_x = tensor_arg aten_env node "self" in
         let* () = require_f32 "self" aten_x in
         (* Rank from the ORIGINAL ATen tensor, before [of_aten] right-aligns it
            into the frame and the rank stops being recoverable -- the same
            ordering [unbind.int] uses, and here it is load-bearing twice over:
            the pad list is indexed FROM the innermost dimension, so a wrong rank
            silently pads the wrong axes. *)
         let rank = aten_rank aten_x in
         let* pad = ints_arg node "pad" in
         let* mode = string_arg ~default:"constant" node "mode" in
         let* value = float_opt_arg node "value" in
         (* No [Err.map_error]: [params_of_aten] already fails with this
            module's own [`Bad_pad_list] row, so the detection origin is the
            check that found the fault rather than this call site. *)
         let* params = Pad.Pad.params_of_aten ~rank ~pad ~mode ~value in
         let* x = native_of_aten "self" aten_x in
         build_g ~name:"pad" [ x ] (function
           | [ x_id ] ->
               let open Graph_builder in
               let+ y = pad params x_id in
               [ y ]
           | _ -> assert false))
  (* The bounds come from the LIVE tensor's extent, which is the whole reason
     this arm and the importer's differ at all: everything after
     [Aten_shape.resolve_slice] is shared, and the extent is the one thing the
     two paths learn differently. Rank before [of_aten], as [pad.default] and
     [unbind.int] do -- the frame does not carry it. *)
  | "torch.ops.aten.slice.Tensor" ->
      Some
        (let* aten_x = tensor_arg aten_env node "self" in
         let* () = require_f32 "self" aten_x in
         let rank = aten_rank aten_x in
         let* dim = int_arg ~default:0 node "dim" in
         let* start = int_opt_arg node "start" in
         let* stop = int_opt_arg node "end" in
         let* step = int_arg ~default:1 node "step" in
         let* x = native_of_aten "self" aten_x in
         let* axis = dim_axis ~op:"slice.Tensor" ~rank dim in
         let extent = Vec6.get (packed_shape x) axis in
         let* bounds =
           Err.map_error
             (fun e -> `Aten_shape e)
             (Aten_shape.resolve_slice ~extent ~start ~stop ~step)
         in
         build_g ~name:"slice" [ x ] (function
           | [ x_id ] ->
               let open Graph_builder in
               let+ y =
                 slice
                   {
                     Split.Slice.axis;
                     start = bounds.Aten_shape.Slice_bounds.start;
                     stop = bounds.Aten_shape.Slice_bounds.stop;
                     step = bounds.Aten_shape.Slice_bounds.step;
                   }
                   x_id
               in
               [ y ]
           | _ -> assert false))
  (* The variadic Native [Concat] op, direct: every operand keeps its rank,
     [dim] names an existing axis. ATen requires every tensor to share one
     rank (no broadcasting across the list), checked explicitly rather than
     left to [Concat.output_shape]'s own axis-agreement rule — see
     [Concat_rank_mismatch]'s doc comment for why that rule would report the
     wrong fault. *)
  | "torch.ops.aten.cat.default" ->
      Some
        (let* aten_xs = tensors_arg aten_env node "tensors" in
         let* () = Err.List.iter (require_f32 "tensors") aten_xs in
         let* dim = int_arg ~default:0 node "dim" in
         match aten_xs with
         | [] -> fail (`Concat_no_tensors "cat.default")
         | aten_x0 :: _ ->
             let rank = aten_rank aten_x0 in
             let* () =
               Err.List.iter
                 (fun t ->
                   let got = aten_rank t in
                   if got = rank then return ()
                   else
                     fail
                       (`Concat_rank_mismatch
                          {
                            Concat_rank_mismatch.op = "cat.default";
                            first = rank;
                            other = got;
                          }))
                 aten_xs
             in
             let* d = norm_dim ~op:"cat.default" ~rank dim in
             let axis = Aten_shape.axis_of_dim ~rank d in
             let* xs = Err.List.map (native_of_aten "tensors") aten_xs in
             build_g ~name:"cat" xs (fun ids ->
                 let open Graph_builder in
                 let+ y = concat { Concat.Concat.axis } ids in
                 [ y ]))
  (* One [Stack] node: inserts a size-1 axis per operand at [axis], then joins
     them — ATen's own definition is exactly
     `cat([t.unsqueeze(dim) for t in tensors], dim)`, which [Stack]'s shape
     rule/[Compute] reuse from [Concat]/[Split.Select]'s implementations
     rather than materializing as separate [Reshape] nodes. [dim] is judged
     against rank+1 valid positions, the OUTPUT rank, same reasoning as
     [norm_unsqueeze_dim]. *)
  | "torch.ops.aten.stack.default" ->
      Some
        (let* aten_xs = tensors_arg aten_env node "tensors" in
         let* () = Err.List.iter (require_f32 "tensors") aten_xs in
         let* dim = int_arg ~default:0 node "dim" in
         match aten_xs with
         | [] -> fail (`Concat_no_tensors "stack.default")
         | aten_x0 :: _ ->
             let rank = aten_rank aten_x0 in
             let* () =
               Err.List.iter
                 (fun t ->
                   let got = aten_rank t in
                   if got = rank then return ()
                   else
                     fail
                       (`Concat_rank_mismatch
                          {
                            Concat_rank_mismatch.op = "stack.default";
                            first = rank;
                            other = got;
                          }))
                 aten_xs
             in
             let* d = norm_unsqueeze_dim ~op:"stack.default" ~rank dim in
             let* xs = Err.List.map (native_of_aten "tensors") aten_xs in
             let axis = Aten_shape.axis_of_dim ~rank:(rank + 1) d in
             build_g ~name:"stack" xs (fun ids ->
                 let open Graph_builder in
                 let+ y = stack { Concat.Stack.axis } ids in
                 [ y ]))
  (* One [Select] node: picks index [idx] along [axis] and drops it. Unlike
     [slice.Tensor], ATen REJECTS an out-of-range [index] rather than clamping
     it -- [Aten_shape.resolve_index], not [resolve_slice]. *)
  | "torch.ops.aten.select.int" ->
      Some
        (let* aten_x = tensor_arg aten_env node "self" in
         let* () = require_f32 "self" aten_x in
         let rank = aten_rank aten_x in
         let* dim = int_arg node "dim" in
         let* index = int_arg node "index" in
         let* x = native_of_aten "self" aten_x in
         let* d = norm_dim ~op:"select.int" ~rank dim in
         let axis = Aten_shape.axis_of_dim ~rank d in
         let extent = Vec6.get (packed_shape x) axis in
         let* idx =
           Err.map_error
             (fun e -> `Aten_shape e)
             (Aten_shape.resolve_index ~extent ~index)
         in
         build_g ~name:"select" [ x ] (function
           | [ x_id ] ->
               let open Graph_builder in
               let+ y = select { Split.Select.axis; index = idx } x_id in
               [ y ]
           | _ -> assert false))
  (* One [Select_scatter] node: [self] with [src] written at [idx] along
     [axis], and every other position carried through from [self] unchanged
     -- the write-back counterpart of [select.int] just above, sharing its
     [Aten_shape.resolve_index] canonicalisation (ATen REJECTS an
     out-of-range index rather than clamping it). [index] is declared
     [SymInt] in ATen's own schema, same as [select.int]'s plain [int] once
     traced: the corpus's own occurrences all arrive as [as_int], so
     [int_arg] reads it unchanged. *)
  | "torch.ops.aten.select_scatter.default" ->
      Some
        (let* aten_self = tensor_arg aten_env node "self" in
         let* () = require_f32 "self" aten_self in
         let* aten_src = tensor_arg aten_env node "src" in
         let* () = require_f32 "src" aten_src in
         let rank = aten_rank aten_self in
         let* dim = int_arg node "dim" in
         let* index = int_arg node "index" in
         let* self = native_of_aten "self" aten_self in
         let* src = native_of_aten "src" aten_src in
         let* d = norm_dim ~op:"select_scatter.default" ~rank dim in
         let axis = Aten_shape.axis_of_dim ~rank d in
         let extent = Vec6.get (packed_shape self) axis in
         let* idx =
           Err.map_error
             (fun e -> `Aten_shape e)
             (Aten_shape.resolve_index ~extent ~index)
         in
         build_g ~name:"select_scatter" [ self; src ] (function
           | [ self_id; src_id ] ->
               let open Graph_builder in
               let+ y =
                 select_scatter
                   { Split.Select_scatter.axis; index = idx }
                   ~self:self_id ~src:src_id
               in
               [ y ]
           | _ -> assert false))
  (* Legalized to [Reshape] alone: inserting a size-1 axis never changes the
     linearized data order, so no [Slice] is needed, unlike [select.int]'s
     axis removal. The target is [Aten_shape.to_aten]'s list with a bare [1]
     spliced in at the normalized position -- the same round-trip
     [select.int]/[view.default] use. *)
  | "torch.ops.aten.unsqueeze.default" ->
      Some
        (let* aten_x = tensor_arg aten_env node "self" in
         let* () = require_f32 "self" aten_x in
         let rank = aten_rank aten_x in
         let* dim = int_arg node "dim" in
         let* x = native_of_aten "self" aten_x in
         let* d = norm_unsqueeze_dim ~op:"unsqueeze.default" ~rank dim in
         let aten_list =
           Array.to_list (Aten_shape.to_aten ~rank (packed_shape x))
         in
         let front = List.filteri (fun i _ -> i < d) aten_list in
         let back = List.filteri (fun i _ -> i >= d) aten_list in
         let out_shape = Array.of_list (front @ [ 1 ] @ back) in
         let* target =
           Err.map_error (fun e -> `Aten_shape e) (Aten_shape.of_aten out_shape)
         in
         build_g ~name:"unsqueeze" [ x ] (function
           | [ x_id ] ->
               let open Graph_builder in
               let+ y = reshape { Reshape.Reshape.shape = target } x_id in
               [ y ]
           | _ -> assert false))
  | "torch.ops.aten.unbind.int" ->
      Some
        (let* aten_x = tensor_arg aten_env node "self" in
         (* Rank comes from the ORIGINAL ATen tensor: [of_aten] right-aligns
            into the six-axis frame, after which the rank is not recoverable. *)
         let rank = aten_rank aten_x in
         let* dim = int_arg ~default:0 node "dim" in
         (* Convert BEFORE judging the dim, so a rank Native cannot represent is
            reported as the rank fault it is rather than as a bad dimension. *)
         let* x = native_of_aten "self" aten_x in
         let* axis = dim_axis ~op:"unbind.int" ~rank dim in
         build_g ~name:"unbind" [ x ] (function
           | [ x_id ] ->
               let open Graph_builder in
               unbind { Split.Unbind.axis } x_id
           | _ -> assert false))
  (* One [Split_with_sizes] node: divides [axis] into contiguous windows of
     [split_sizes], KEEPING the axis in every output, unlike [unbind.int]
     which drops it. [split_sizes] is a required arg (no schema default), the
     same as [view.default]'s [size]; [Split.Split_with_sizes.output_shapes]
     is what checks the sizes are positive and sum to the axis's extent, not
     this arm -- the same division of labor [cat.default]'s rank check and
     [Concat.output_shape]'s axis-agreement check follow. Preserves the
     input's own dtype like [unbind.int], not [require_f32]'d like
     [cat.default]/[stack.default]. *)
  | "torch.ops.aten.split_with_sizes.default" ->
      Some
        (let* aten_x = tensor_arg aten_env node "self" in
         let rank = aten_rank aten_x in
         let* dim = int_arg ~default:0 node "dim" in
         let* sizes = ints_arg node "split_sizes" in
         let* x = native_of_aten "self" aten_x in
         let* axis = dim_axis ~op:"split_with_sizes.default" ~rank dim in
         build_g ~name:"split_with_sizes" [ x ] (function
           | [ x_id ] ->
               let open Graph_builder in
               split_with_sizes { Split.Split_with_sizes.axis; sizes } x_id
           | _ -> assert false))
  (* `split.Tensor(self, split_size, dim)` -- the equal-chunk-size sibling of
     [split_with_sizes.default] just above. Binds to the *existing*
     [Split_with_sizes] node: [chunk_sizes] derives the same sizes list ATen
     itself would split into, so this is "translate params, still one node",
     not a decomposition. *)
  | "torch.ops.aten.split.Tensor" ->
      Some
        (let* aten_x = tensor_arg aten_env node "self" in
         let rank = aten_rank aten_x in
         let* dim = int_arg ~default:0 node "dim" in
         let* split_size = int_arg node "split_size" in
         let* x = native_of_aten "self" aten_x in
         let* axis = dim_axis ~op:"split.Tensor" ~rank dim in
         let* split_size =
           pos ~op:"split.Tensor" ~param:`Split_size split_size
         in
         let extent = (Vec6.get (packed_shape x) axis :> int) in
         let sizes = chunk_sizes ~extent ~split_size:(split_size :> int) in
         build_g ~name:"split" [ x ] (function
           | [ x_id ] ->
               let open Graph_builder in
               split_with_sizes { Split.Split_with_sizes.axis; sizes } x_id
           | _ -> assert false))
  (* Both overloads share this body: after commit 1's checked resolver the body
     is three calls, and two copies of it is exactly the drift risk this repo
     keeps warning about. Still exact-target dispatch in op3.md's sense -- both
     targets are named, there is no fallthrough, and [native_of_aten]/every
     diagnostic below reads [node.target], so a failure still says which
     overload it was (op3-impl.md Part IV #2). [Identical] AFTER
     materialization, not alias-identical: ATen may return a view over the
     same storage, native graph edges are values. See
     .ai/native_aten_bridge_layout.md. *)
  (* Schema: `upsample_bilinear2d.vec(Tensor input, SymInt[]? output_size,
     bool align_corners, float[]? scale_factors)`. Exactly one of
     [output_size]/[scale_factors] is ever given (ATen's own
     `compute_output_size` rejects both-or-neither); a [scale_factors] graph
     is resolved to an explicit size here, at import time, using ATen's own
     `floor(input_size * scale_factor)` -- the Native op only ever sees a
     concrete size, since Native shapes are static. *)
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
           | _ -> assert false))
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
           | _ -> assert false))
  | "torch.ops.aten.view.default" | "torch.ops.aten._unsafe_view.default" ->
      Some
        ((* Contiguous reshape. [of_aten] inputs are already ATen-row-major, so
            the native reshape needs no surrounding permutes; the target native
            shape is [size] right-aligned (a single -1 resolved against numel). *)
         let* aten_x = tensor_arg aten_env node "self" in
         let* size = ints_arg node "size" in
         let* x = native_of_aten "self" aten_x in
         let (Tensor.Tensor r) = x in
         (* [x]'s shape already cleared [Tensor_bridge.of_aten]'s numel
            preflight, so this count is < [Kernel.Limits.Hard.numel] and the
            [int64] conversion below cannot itself be the overflow this design
            guards against. *)
         let numel = Int64.of_int (Dim.to_int (Vec6.numel r.shape)) in
         let* resolved =
           Err.map_error
             (fun e -> `Aten_shape e)
             (Aten_shape.resolve_view_size ~numel size)
         in
         let* target =
           Err.map_error
             (fun e -> `Aten_shape e)
             (Aten_shape.of_aten (Array.of_list resolved))
         in
         let params = { Reshape.Reshape.shape = target } in
         build_g ~name:"view" [ x ] (function
           | [ x_id ] ->
               let open Graph_builder in
               let+ y = reshape params x_id in
               [ y ]
           | _ -> assert false))
  (* [alias(self) -> Tensor] is a pure identity view (same storage, same
     size/strides) -- ATen's own schema takes no argument beyond [self]. Binds
     to the *existing* [Clone] node: paramless, [output_shape] the identity,
     and already the node this repo uses for "same value, no change" (see
     [clone.default]'s own arm) -- Native4D already elides it entirely
     (`.ai/native4d_design.md` §7.1), so this legalization costs nothing
     there either. *)
  | "torch.ops.aten.alias.default" ->
      Some
        (let* aten_x = tensor_arg aten_env node "self" in
         let* x = native_of_aten "self" aten_x in
         build_g ~name:"alias" [ x ] (function
           | [ x_id ] ->
               let open Graph_builder in
               let+ y = clone x_id in
               [ y ]
           | _ -> assert false))
  (* `expand(Tensor(a) self, SymInt[] size, *, bool implicit=False) ->
     Tensor(a)`. [implicit] is read-and-discarded: ATen's own
     [at::native::expand] (TensorShape.cpp) takes it as a literally UNUSED
     parameter, so decoding it proves the schema shape without pretending it
     affects the result -- the same treatment [cudnn_enabled] gets in the
     batch_norm arm. Binds to a genuinely new op, [Pointwise.Expand]:
     broadcasts [self] to the resolved [size] by reading through
     [Pointwise_binary.broadcast_coord], the same one-sided reduction a
     binary op's own operand already uses. *)
  | "torch.ops.aten.expand.default" ->
      Some
        (let* t = tensor_arg aten_env node "self" in
         let* size = ints_arg node "size" in
         let* (_ : bool) = bool_arg node "implicit" in
         let* resolved =
           Err.map_error
             (fun e -> `Aten_shape e)
             (Aten_shape.resolve_expand_size ~self_dims:(Aten_tensor.shape t)
                ~size)
         in
         let* target =
           Err.map_error
             (fun e -> `Aten_shape e)
             (Aten_shape.of_aten (Array.of_list resolved))
         in
         let* x = native_of_aten "self" t in
         build_g ~name:"expand" [ x ] (function
           | [ x_id ] ->
               let open Graph_builder in
               let+ y = expand { Pointwise.Expand.size = target } x_id in
               [ y ]
           | _ -> assert false))
  (* `repeat(Tensor self, SymInt[] repeats) -> Tensor`. [repeats] is checked
     against [self]'s own ATen rank ([Aten_shape.resolve_repeat_size], the
     same split [expand.default]'s arm above makes for [size]) then
     right-aligned into the frame with [of_aten] -- unlike [expand]'s [size],
     [repeats] has no [-1] convention to resolve first, so there is no
     translation step between the two. Binds to the genuinely new
     [Repeat.Repeat]: each output element is the source element at the
     output coordinate MODULO the source extent, independently per axis
     ([Repeat.Repeat.Compute.pixel]), not a broadcast. *)
  | "torch.ops.aten.repeat.default" ->
      Some
        (let* t = tensor_arg aten_env node "self" in
         let* repeats = ints_arg node "repeats" in
         let* resolved =
           Err.map_error
             (fun e -> `Aten_shape e)
             (Aten_shape.resolve_repeat_size ~self_dims:(Aten_tensor.shape t)
                ~repeats)
         in
         let* target =
           Err.map_error
             (fun e -> `Aten_shape e)
             (Aten_shape.of_aten (Array.of_list resolved))
         in
         let* x = native_of_aten "self" t in
         build_g ~name:"repeat" [ x ] (function
           | [ x_id ] ->
               let open Graph_builder in
               let+ y = repeat { Repeat.Repeat.repeats = target } x_id in
               [ y ]
           | _ -> assert false))
  (* `repeat_interleave.self_int(Tensor self, SymInt repeats, int? dim=None,
     *, SymInt? output_size=None) -> Tensor`, restricted to an explicit
     [dim] -- the [dim=None] flatten-first form is a genuinely different
     reshape+repeat composition, deferred until a model demonstrates it (see
     [Repeat.RepeatInterleave]'s own comment). [output_size] is
     read-and-discarded, the same treatment [implicit] gets in the
     [expand.default] arm above: it exists only so a Tensor-valued [repeats]
     can skip a device sync, and this is the SymInt-scalar overload. *)
  | "torch.ops.aten.repeat_interleave.self_int" ->
      Some
        (let op = "repeat_interleave.self_int" in
         let* t = tensor_arg aten_env node "self" in
         let rank = aten_rank t in
         let* dim = int_opt_arg node "dim" in
         let* dim =
           match dim with
           | Some d -> return d
           | None ->
               fail
                 (`Validation_failure
                    (op ^ ": dim=None (flatten-first) is not yet supported"))
         in
         let* axis = dim_axis ~op ~rank dim in
         let* repeats_int = int_arg node "repeats" in
         let* repeats = pos ~op ~param:`Repeats repeats_int in
         let* x = native_of_aten "self" t in
         build_g ~name:"repeat_interleave" [ x ] (function
           | [ x_id ] ->
               let open Graph_builder in
               let+ y =
                 repeat_interleave
                   { Repeat.RepeatInterleave.axis; repeats }
                   x_id
               in
               [ y ]
           | _ -> assert false))
  (* [zeros(SymInt[] size, *, ScalarType? dtype=None, Layout? layout=None,
     Device? device=None, bool? pin_memory=None) -> Tensor].  This is a real
     factory, not a fill disguised as an input-free pointwise op: shape and
     dtype are retained in the Native payload. *)
  | "torch.ops.aten.zeros.default" ->
      Some
        (let* size = ints_arg node "size" in
         let* shape =
           Aten_shape.of_aten (Array.of_list size)
           |> Err.map_error (fun e -> `Aten_shape e)
         in
         let* fmt =
           match D.find_arg node "dtype" with
           | None | Some (Argument.None _) -> return (Payload.Fmt Payload.F32)
           | Some (Argument.Scalar_type Pytorch_types.ScalarType.FLOAT) ->
               return (Payload.Fmt Payload.F32)
           | Some (Argument.Scalar_type Pytorch_types.ScalarType.DOUBLE) ->
               return (Payload.Fmt Payload.F64)
           | Some _ ->
               fail
                 (`Validation_failure
                    "zeros: only default/FLOAT and DOUBLE dtype are supported")
         in
         let reject_absent name =
           match D.find_arg node name with
           | None | Some (Argument.None _) -> return ()
           | Some _ ->
               fail
                 (`Validation_failure
                    ("zeros: " ^ name ^ " must be omitted or None"))
         in
         let* () = reject_absent "layout" in
         let* () =
           match D.find_arg node "device" with
           | None | Some (Argument.None _) -> return ()
           | Some (Argument.Device { Device.type_ = "cpu"; index = None }) ->
               return ()
           | Some _ ->
               fail
                 (`Validation_failure
                    "zeros: device must be omitted/None or CPU without an index")
         in
         let* () =
           match D.find_arg node "pin_memory" with
           | None | Some (Argument.None _) | Some (Argument.Bool false) ->
               return ()
           | Some _ ->
               fail
                 (`Validation_failure
                    "zeros: pin_memory must be omitted/None or false")
         in
         let* g =
           Graph_builder.build ~name:"zeros"
             ~outputs:(fun y -> [ y ])
             (Graph_builder.zeros { Factory.Zeros.shape; fmt })
           |> Err.map_error (fun e -> `Build e)
         in
         return (g, []))
  | ("torch.ops.aten.arange.default" | "torch.ops.aten.arange.start") as target
    ->
      Some
        (let scalar_input_is_float =
           List.exists
             (fun name ->
               match D.find_arg node name with
               | Some (Argument.Float _) -> true
               | _ -> false)
             [ "start"; "end"; "step" ]
         in
         let* fmt =
           match D.find_arg node "dtype" with
           | None | Some (Argument.None _) ->
               return
                 (if scalar_input_is_float then Payload.Fmt Payload.F32
                  else Payload.Fmt Payload.I64)
           | Some (Argument.Scalar_type Pytorch_types.ScalarType.LONG) ->
               return (Payload.Fmt Payload.I64)
           | Some (Argument.Scalar_type Pytorch_types.ScalarType.FLOAT) ->
               return (Payload.Fmt Payload.F32)
           | Some _ ->
               fail
                 (`Validation_failure
                    "arange: only default/LONG and FLOAT dtype are supported")
         in
         let reject_absent name =
           match D.find_arg node name with
           | None | Some (Argument.None _) -> return ()
           | Some _ ->
               fail
                 (`Validation_failure
                    ("arange: " ^ name ^ " must be omitted or None"))
         in
         let* () = reject_absent "layout" in
         let* () =
           match D.find_arg node "device" with
           | None | Some (Argument.None _) -> return ()
           | Some (Argument.Device { Device.type_ = "cpu"; index = None }) ->
               return ()
           | Some _ ->
               fail
                 (`Validation_failure
                    "arange: device must be omitted/None or CPU without an \
                     index")
         in
         let* () =
           match D.find_arg node "pin_memory" with
           | None | Some (Argument.None _) | Some (Argument.Bool false) ->
               return ()
           | Some _ ->
               fail
                 (`Validation_failure
                    "arange: pin_memory must be omitted/None or false")
         in
         let* start, stop =
           match target with
           | "torch.ops.aten.arange.default" ->
               let* stop =
                 scalar_arg ~default:(Aten_scalar.Int 0L) node "end"
               in
               return (0., stop)
           | "torch.ops.aten.arange.start" ->
               let* start =
                 scalar_arg ~default:(Aten_scalar.Int 0L) node "start"
               in
               let* stop =
                 scalar_arg ~default:(Aten_scalar.Int 0L) node "end"
               in
               return (start, stop)
           | _ -> assert false
         in
         let* step = scalar_arg ~default:(Aten_scalar.Int 1L) node "step" in
         let* g =
           Graph_builder.build ~name:"arange"
             ~outputs:(fun y -> [ y ])
             (Graph_builder.arange { Factory.Arange.start; stop; step; fmt })
           |> Err.map_error (fun e -> `Build e)
         in
         return (g, []))
  (* `_assert_tensor_metadata(Tensor a, SymInt[]? size=None, SymInt[]?
     stride=None, ScalarType? dtype=None, *, Device? device=None, Layout?
     layout=None) -> ()`: a pure debug assertion the exporter inserts to pin
     a tensor's metadata -- it returns no [Tensor] at all (this repo's own
     PT2 producer always serializes it with an empty node [outputs] list),
     so [build_g]'s own body returns [[]] rather than [[y]]. Given an
     already-successfully-exported graph with shapes resolved at parse time
     the assertion is unconditionally true, so [a] is routed to the same
     [Discard] sink [max_pool2d_with_indices.default]'s dead indices edge
     uses rather than left dangling. [size]/[stride]/[dtype]/[device]/
     [layout] restate facts already true of [a] and have no existing typed
     decoder (unlike [implicit]'s or [cudnn_enabled]'s plain [bool_arg]) --
     decoding them would add new accessor machinery for values Native never
     acts on, so they are left unread, matching [Native_interp]'s arm. No
     real ATen C call is made here: unlike every other bridge arm, this op's
     "computation" is discarding a tensor [aten_env] already holds from an
     earlier node's real dispatch, so there is nothing to link against and
     no [bin/aten_ops_gen.ml] entry is needed. *)
  | "torch.ops.aten._assert_tensor_metadata.default" ->
      Some
        (let* x = native_tensor_arg aten_env node "a" in
         build_g ~name:"assert_tensor_metadata" [ x ] (function
           | [ x_id ] ->
               let open Graph_builder in
               let+ () = discard x_id in
               []
           | _ -> assert false))
  (* `index.Tensor(Tensor self, Tensor?[] indices) -> Tensor`, restricted to
     the one evidenced shape family (`.ai/index_tensor_design.md`): [indices]
     has exactly one live (non-None) entry, of ATen rank 1, at every other
     position [None]. Unlike [Native_interp] (metadata-only import, below),
     this bridge is reached with [indices]' live entry ALREADY RESOLVED to a
     real ATen tensor VALUE by [aten_env] -- whatever node produced it,
     `clone.default` included, since ATen's own [clone] preserves both value
     and dtype exactly. So there is no "trace past Clone" step to take here:
     [Op_bridge.dispatch] sees only this one node (no surrounding graph to
     trace through, unlike [Native_interp]'s importer, which retains the
     whole node list), and reading the live entry's real, already-correct
     value via [native_of_aten] is enough on its own -- the round-6 problem
     that trace-back exists to solve is specifically about [Native_interp]'s
     metadata-only import stamping a wrong dtype on a *newly built* [Clone]
     node's OWN output signature, which never happens here. *)
  | "torch.ops.aten.index.Tensor" ->
      Some
        (let* self_t = tensor_arg aten_env node "self" in
         let rank = aten_rank self_t in
         let* entries =
           decode_result (D.optional_tensors_arg_result aten_env node "indices")
         in
         let got = List.length entries in
         let index_list_fail (fault : Op_bridge_error.Index_list.fault) =
           fail (`Index_list { Op_bridge_error.Index_list.fault })
         in
         if got <> rank then
           index_list_fail
             (Op_bridge_error.Index_list.Length_mismatch
                { expected = rank; got })
         else
           let live_positions =
             List.mapi (fun i e -> (i, e)) entries
             |> List.filter_map (fun (i, e) ->
                 if Option.is_some e then Some i else None)
           in
           match live_positions with
           | [] -> index_list_fail Op_bridge_error.Index_list.No_live_entry
           | _ :: _ :: _ ->
               index_list_fail
                 (Op_bridge_error.Index_list.Multiple_live_entries
                    live_positions)
           | [ p ] ->
               let index_t = Option.get (List.nth entries p) in
               let dtype = Aten_tensor.scalar_type index_t in
               if not (dtype = Aten_scalar_type.Long) then
                 index_list_fail
                   (Op_bridge_error.Index_list.Wrong_dtype
                      { position = p; dtype })
               else
                 let index_rank = aten_rank index_t in
                 if index_rank <> 1 then
                   index_list_fail
                     (Op_bridge_error.Index_list.Wrong_rank
                        { position = p; rank = index_rank })
                 else
                   let* axis = dim_axis ~op:"index.Tensor" ~rank p in
                   let* self_n = native_of_aten "self" self_t in
                   let* index_n = native_of_aten "index" index_t in
                   build_g ~name:"index_tensor" [ self_n; index_n ] (function
                     | [ self_id; index_id ] ->
                         let open Graph_builder in
                         let+ y =
                           index_tensor
                             { Index_tensor.Index_tensor.axis }
                             ~self:self_id ~index:index_id
                         in
                         [ y ]
                     | _ -> assert false))
  | _ -> None
