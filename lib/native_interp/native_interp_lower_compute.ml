(* Compute-family operator dispatch for [Native_interp_lower].  Each family
   returns [None] for targets it does not own; the lowerer owns the one
   unsupported-operator fallback. *)

open Pytorch_types
open Schema_runtime
open Native_interp_error
open Native_interp_decode
open Native_interp_decode_conv

let targets =
  [
    "torch.ops.aten._native_batch_norm_legit_no_training.default";
    "torch.ops.aten._native_batch_norm_legit.no_stats";
    "torch.ops.aten.add.Tensor";
    "torch.ops.aten.adaptive_avg_pool2d.default";
    "torch.ops.aten.addcmul.default";
    "torch.ops.aten.addmm.default";
    "torch.ops.aten.amax.default";
    "torch.ops.aten.avg_pool2d.default";
    "torch.ops.aten.clamp.default";
    "torch.ops.aten.clamp_min.default";
    "torch.ops.aten.clone.default";
    "torch.ops.aten.conv2d.default";
    "torch.ops.aten.conv2d.padding";
    "torch.ops.aten.convolution.default";
    "torch.ops.aten.div.Tensor";
    "torch.ops.aten.gelu.default";
    "torch.ops.aten.group_norm.default";
    "torch.ops.aten.hardsigmoid.default";
    "torch.ops.aten.hardsigmoid_.default";
    "torch.ops.aten.hardswish.default";
    "torch.ops.aten.hardswish_.default";
    "torch.ops.aten.hardtanh.default";
    "torch.ops.aten.index.Tensor";
    "torch.ops.aten.layer_norm.default";
    "torch.ops.aten.leaky_relu.default";
    "torch.ops.aten.linalg_vector_norm.default";
    "torch.ops.aten.linear.default";
    "torch.ops.aten.matmul.default";
    "torch.ops.aten.max_pool2d.default";
    "torch.ops.aten.max_pool2d_with_indices.default";
    "torch.ops.aten.mean.dim";
    "torch.ops.aten.mul.Scalar";
    "torch.ops.aten.mul.Tensor";
    "torch.ops.aten.native_layer_norm.default";
    "torch.ops.aten.pow.Tensor_Scalar";
    "torch.ops.aten.relu.default";
    "torch.ops.aten.rms_norm.default";
    "torch.ops.aten.rsqrt.default";
    "torch.ops.aten.scaled_dot_product_attention.default";
    "torch.ops.aten.sigmoid.default";
    "torch.ops.aten.silu.default";
    "torch.ops.aten.silu_.default";
    "torch.ops.aten.softmax.int";
    "torch.ops.aten.sub.Tensor";
    "torch.ops.aten.sum.dim_IntList";
  ]

let dispatch ~ctx ~env (node : Node.t) =
  if not (List.mem node.target targets) then None
  else
    Some
      (let open Graph_builder in
       let esc = ctx.Native_interp_lower_context.esc in
       let graph = ctx.Native_interp_lower_context.graph in
       let reads = ctx.Native_interp_lower_context.reads in
       let get = Native_interp_lower_context.get ctx env node in
       let tensor_or_scalar =
         Native_interp_lower_context.tensor_or_scalar ctx env node
       in
       match node.target with
       | "torch.ops.aten.conv2d.default" ->
           let params = conv2d_params esc graph node in
           let* x = permute perm_nchw_to_nhwc (get "input") in
           let* w = permute perm_oihw_to_conv_weight (get "weight") in
           let bias_name =
             optional_tensor_name ~absent_ok:true esc node "bias"
           in
           Option.iter
             (fun ssa ->
               require_rank esc graph ~ssa ~role:`Conv2d_bias ~expected:1)
             bias_name;
           let bias = Option.map (env_find esc env) bias_name in
           let* y = conv2d params ~x ~weight:w ?bias () in
           let* y = permute perm_nhwc_to_nchw y in
           return [ y ]
       (* Same decode and same relayouts as the arm above; only the padding
         contract differs, and the mode is carried into the IR unresolved. *)
       | "torch.ops.aten.conv2d.padding" ->
           let params = conv2d_padding_params esc graph node in
           let* x = permute perm_nchw_to_nhwc (get "input") in
           let* w = permute perm_oihw_to_conv_weight (get "weight") in
           let bias_name =
             optional_tensor_name ~absent_ok:true esc node "bias"
           in
           Option.iter
             (fun ssa ->
               require_rank esc graph ~ssa ~role:`Conv2d_padding_bias
                 ~expected:1)
             bias_name;
           let bias = Option.map (env_find esc env) bias_name in
           let* y = conv2d_padding params ~x ~weight:w ?bias () in
           let* y = permute perm_nhwc_to_nchw y in
           return [ y ]
       | "torch.ops.aten.convolution.default" ->
           let params, _, _, _ = conv_params esc graph node in
           let* x = permute perm_nchw_to_nhwc (get "input") in
           let* w = permute perm_oihw_to_conv_weight (get "weight") in
           let bias_name = optional_tensor_name esc node "bias" in
           Option.iter
             (fun ssa ->
               require_rank esc graph ~ssa ~role:`Convolution_bias ~expected:1)
             bias_name;
           let bias = Option.map (env_find esc env) bias_name in
           let* y = convolution params ~x ~weight:w ?bias () in
           let* y = permute perm_nhwc_to_nchw y in
           return [ y ]
       | "torch.ops.aten._native_batch_norm_legit_no_training.default" ->
           let* x = permute perm_nchw_to_nhwc (get "input") in
           let params =
             { Norm.BatchNorm.channel = Axis.C; eps = float_arg esc node "eps" }
           in
           let* y =
             batch_norm params ~x
               ?weight:
                 (Option.map (env_find esc env)
                    (optional_tensor_name esc node "weight"))
               ?bias:
                 (Option.map (env_find esc env)
                    (optional_tensor_name esc node "bias"))
               ~running_mean:(get "running_mean")
               ~running_var:(get "running_var") ()
           in
           let* y = permute perm_nhwc_to_nchw y in
           return [ y ]
       | "torch.ops.aten._native_batch_norm_legit.no_stats" -> (
           (* Training batch norm without running statistics is not the
              inference target above: its three real outputs retain the batch
              reductions, so no running-stat fold or output dropping is valid. *)
           let training = bool_arg esc node "training" in
           let momentum = float_arg esc node "momentum" in
           let () =
             if not training then
               malformed esc
                 (`Unsupported_option
                    { op = node.target; option = `Training training })
             else if not (Float.equal momentum 0.) then
               malformed esc
                 (`Unsupported_option
                    { op = node.target; option = `Momentum momentum })
           in
           let params =
             {
               Norm.BatchNormNoStats.channel = Axis.C;
               eps = float_arg esc node "eps";
             }
           in
           let x_name = tensor_name esc node "input" in
           let rank =
             meta_rank
               (tensor_meta esc graph ~ssa:x_name
                  ~role:`Batch_norm_no_stats_input)
           in
           let to_channel_last, from_channel_last =
             batch_norm_channel_perms ~rank
           in
           let* x = permute to_channel_last (get "input") in
           let weight =
             Option.map (env_find esc env)
               (optional_tensor_name esc node "weight")
           in
           let bias =
             Option.map (env_find esc env)
               (optional_tensor_name esc node "bias")
           in
           let* ys = batch_norm_no_stats params ~x ?weight ?bias () in
           match ys with
           | [ y'; mean; invstd ] ->
               let+ y = permute from_channel_last y' in
               [ y; mean; invstd ]
           | _ -> assert false)
       (* [input] is right-aligned NCHW like [batch_norm]'s, so the same
         permute pair relays it around the op -- the serialized counterpart of
         [Op_bridge]'s [group_norm.default] arm: same checks, same node, same
         treatment of the optional operands (NO ones/zeros tensors when
         absent, for the reason the [layer_norm] arm below states). *)
       | "torch.ops.aten.group_norm.default" ->
           let* x = permute perm_nchw_to_nhwc (get "input") in
           let num_groups = int_arg esc node "num_groups" in
           let eps = float_arg esc ~default:1e-05 node "eps" in
           (* Decoded, not ignored, then discarded -- matching the bridge and
             matching ATen: a non-boolean here is a malformed node. *)
           let (_ : bool) = bool_arg esc ~default:true node "cudnn_enabled" in
           let groups =
             pos esc ~op:"group_norm.default" ~param:`Groups num_groups
           in
           let params = { Norm.GroupNorm.channel = Axis.C; groups; eps } in
           let affine name role =
             let ssa = optional_tensor_name ~absent_ok:true esc node name in
             Option.iter
               (fun ssa -> require_rank esc graph ~ssa ~role ~expected:1)
               ssa;
             Option.map (env_find esc env) ssa
           in
           let* y =
             group_norm params ~x
               ?weight:(affine "weight" `Group_norm_weight)
               ?bias:(affine "bias" `Group_norm_bias)
               ()
           in
           let* y = permute perm_nhwc_to_nchw y in
           return [ y ]
       (* [normalized_shape] is validated against the input, which is the check
         the bridge is missing: it reads only the LENGTH (op_bridge.ml:899) and
         never compares the extents, so a shape that names the wrong axes
         normalizes over the wrong ones and produces a plausible wrong answer.
         [trailing_axes] there does not check [k <= rank] either, and silently
         returns the whole axis list when it is exceeded. *)
       (* The FUNCTIONAL layer norm and its DECOMPOSED twin, in one body, and
         the serialized counterparts of [Op_bridge]'s arms: same checks, same
         node, same treatment of the optional operands. The only differences
         from the bridge are where the extents come from (static metadata here,
         a live tensor there) and that this one can meet an unresolved symbolic
         extent, which [static_sizes] already refuses.

         The two TARGETS differ in three things and in nothing else:
         [native_layer_norm]'s [eps] is required with no schema default, it has
         no [cudnn_enable], and it returns a 3-tuple whose trailing two elements
         this graph does not have. The arithmetic, the axis derivation, the
         normalized_shape validation and the affine handling are identical, so
         they are written once -- CLAUDE.md's rule against a second copy of a
         shape rule free to drift, applied to dispatch, as [view]/[_unsafe_view]
         already does. Both targets are named explicitly, there is no
         fallthrough, and every diagnostic below carries [op] or the resolved
         tensor name, so a failure still says which overload produced it. *)
       | ( "torch.ops.aten.layer_norm.default"
         | "torch.ops.aten.native_layer_norm.default" ) as target ->
           let op, eps =
             match target with
             | "torch.ops.aten.layer_norm.default" ->
                 (* Decoded and then discarded, matching the bridge and matching
                   ATen: [layer_norm_symint] takes it as [bool /* cudnn_enable,
                   deprecated */] and drops it. Accepting only [false] would
                   reject the schema's own default. Decoding it is still what
                   rejects a non-boolean.

                   A required float WITH a schema default, so [float_arg
                   ~default] -- not rms_norm's [float_opt_arg], whose absent
                   case means "ATen picks the machine epsilon". *)
                 let (_ : bool) =
                   bool_arg esc ~default:true node "cudnn_enable"
                 in
                 ( Norm.Target.Layer_norm,
                   float_arg esc ~default:1e-05 node "eps" )
             | _ ->
                 (* The decomposed form's [eps] has NO default in the schema, so
                   its absence is a malformed node rather than 1e-5. Every
                   corpus occurrence spells it 1e-06 explicitly. *)
                 (Norm.Target.Native_layer_norm, float_arg esc node "eps")
           in
           (* The two dropped outputs, refused if anything reads one. Checked
             before the node is built, so the diagnostic names the op that
             cannot provide the statistic rather than the consumer that wanted
             it. The arity is checked here too: [materialized_output_names]
             keeps only the head, so a 2-tuple would pass the generic bind check
             and silently drop one fewer output than it should. *)
           let* () =
             match op with
             | Norm.Target.Layer_norm | Norm.Target.Rms_norm -> return ()
             | Norm.Target.Native_layer_norm ->
                 let names = output_names esc node in
                 (match names with
                 | [ _; mean; rstd ] ->
                     List.iter
                       (fun (stat, ssa) ->
                         if String_map.mem ssa (Lazy.force reads) then
                           malformed esc
                             (`Live_layer_norm_stats
                                { Live_layer_norm_stats.op = target; stat; ssa }))
                       [ (`Mean, mean); (`Rstd, rstd) ]
                 | _ ->
                     malformed esc
                       (`Output_arity
                          {
                            op = target;
                            serialized = List.length names;
                            derived = 3;
                          }));
                 return ()
           in
           let x_name = tensor_name esc node "input" in
           let sizes =
             static_sizes esc ~tensor:x_name
               (tensor_meta esc graph ~ssa:x_name ~role:`Layer_norm_input)
           in
           let rank = List.length sizes in
           let normalized = ints_arg esc node "normalized_shape" in
           let k = List.length normalized in
           if k < 1 || k > rank then
             malformed esc (`Normalized_rank { op; rank; got = k });
           let trailing l = List.filteri (fun i _ -> i >= rank - k) l in
           let expected = trailing sizes in
           if expected <> normalized then
             malformed esc
               (`Normalized_shape { op; expected; got = normalized });
           let params =
             {
               Norm.LayerNorm.dims =
                 trailing (used_axes_for esc ~tensor:x_name rank);
               eps;
             }
           in
           (* NO ones/zeros tensors when an affine operand is absent -- see the
             rms_norm arm below. Both importers must make the same choice or one
             of [Graph_ir]'s option arms becomes unreachable from one of them. *)
           let affine name role =
             let ssa = optional_tensor_name ~absent_ok:true esc node name in
             (* Rank [k], not rank 1: ATen indexes both affine operands by the
               whole normalized shape. *)
             Option.iter
               (fun ssa -> require_rank esc graph ~ssa ~role ~expected:k)
               ssa;
             Option.map (env_find esc env) ssa
           in
           let* y =
             layer_norm params ~x:(get "input")
               ?weight:(affine "weight" `Layer_norm_weight)
               ?bias:(affine "bias" `Layer_norm_bias)
               ()
           in
           return [ y ]
       | "torch.ops.aten.rms_norm.default" ->
           (* "input", not "self": the schema is
             rms_norm(Tensor input, SymInt[] normalized_shape, Tensor? weight,
             float? eps), and [Op_bridge] reads the same name. *)
           let x_name = tensor_name esc node "input" in
           let sizes =
             static_sizes esc ~tensor:x_name
               (tensor_meta esc graph ~ssa:x_name ~role:`Rms_norm_input)
           in
           let rank = List.length sizes in
           let normalized = ints_arg esc node "normalized_shape" in
           let k = List.length normalized in
           if k < 1 || k > rank then
             malformed esc
               (`Normalized_rank { op = Norm.Target.Rms_norm; rank; got = k });
           let trailing l = List.filteri (fun i _ -> i >= rank - k) l in
           let expected = trailing sizes in
           if expected <> normalized then
             malformed esc
               (`Normalized_shape
                  { op = Norm.Target.Rms_norm; expected; got = normalized });
           let params =
             {
               Norm.RmsNorm.dims =
                 trailing (used_axes_for esc ~tensor:x_name rank);
               eps =
                 float_opt_arg esc ~default:Norm.RmsNorm.default_eps node "eps";
             }
           in
           (* NO ones tensor when the weight is absent. [Graph_ir]'s [Rms_norm]
             carries [weight : Tensor_ref.t option] and Native4D reads the
             option (lower.ml:293-299); synthesizing a constant would make the
             two importers build structurally different graphs for the same
             node and leave that arm unreachable from one of them. *)
           let weight_name =
             optional_tensor_name ~absent_ok:true esc node "weight"
           in
           (* Rank [k], not rank 1: ATen indexes the weight by the whole
             normalized shape, so a multi-axis normalization takes a
             multi-axis weight. *)
           Option.iter
             (fun ssa ->
               require_rank esc graph ~ssa ~role:`Rms_norm_weight ~expected:k)
             weight_name;
           let* y =
             rms_norm params ~x:(get "input")
               ?weight:(Option.map (env_find esc env) weight_name)
               ()
           in
           return [ y ]
       (* op8-impl.md commit 3: the serialized twin of [Op_bridge]'s arm --
         same checks, same typed rejections ([Attention.Sdpa.Reject], shared
         so the two importers cannot drift), same treatment of the optional
         mask. Q/K/V/mask rank is checked on the DECLARED metadata rank
         (F13), the only place it survives: [Tensor_sig.t] keeps none.
         [value.C <> query.C] and an inadmissible mask broadcast shape are
         flash-oracle boundaries (F4) [Attention.Sdpa.output_shape] already
         rejects via the eventual [Graph_builder.build] call, not re-checked
         here. *)
       | "torch.ops.aten.scaled_dot_product_attention.default" ->
           let query_name = tensor_name esc node "query" in
           let key_name = tensor_name esc node "key" in
           let value_name = tensor_name esc node "value" in
           require_rank esc graph ~ssa:query_name ~role:`Sdpa_query ~expected:4;
           require_rank esc graph ~ssa:key_name ~role:`Sdpa_key ~expected:4;
           require_rank esc graph ~ssa:value_name ~role:`Sdpa_value ~expected:4;
           let dropout_p = float_arg esc ~default:0.0 node "dropout_p" in
           if not (Float.equal dropout_p 0.0) then
             malformed esc
               (`Sdpa_reject (Attention.Sdpa.Reject.Dropout dropout_p));
           let is_causal = bool_arg esc ~default:false node "is_causal" in
           if is_causal then
             malformed esc (`Sdpa_reject Attention.Sdpa.Reject.Causal);
           let enable_gqa = bool_arg esc ~default:false node "enable_gqa" in
           if enable_gqa then
             malformed esc (`Sdpa_reject Attention.Sdpa.Reject.Gqa);
           let scale =
             match float_opt_arg_opt esc node "scale" with
             | None -> Attention.Sdpa.Scale.Default
             | Some s ->
                 if not (Float.is_finite s) then
                   malformed esc
                     (`Sdpa_reject (Attention.Sdpa.Reject.Non_finite_scale s));
                 if Float.compare s 0.0 < 0 then
                   malformed esc
                     (`Sdpa_reject (Attention.Sdpa.Reject.Negative_scale s));
                 Attention.Sdpa.Scale.Explicit s
           in
           (* NO ones/zeros tensor for an absent mask -- [Graph_ir]'s [Sdpa]
             carries [mask : Tensor_ref.t option] and [Eval_op] fills it;
             materialising here would build a structurally different graph
             from [Op_bridge]'s for the same node. *)
           let mask_name =
             optional_tensor_name ~absent_ok:true esc node "attn_mask"
           in
           Option.iter
             (fun ssa ->
               let meta = tensor_meta esc graph ~ssa ~role:`Sdpa_mask in
               (match meta.TensorMeta.dtype with
               | ScalarType.BOOL ->
                   malformed esc
                     (`Sdpa_reject Attention.Sdpa.Reject.Boolean_mask)
               | _ -> ());
               let got = meta_rank meta in
               if got <> 2 && got <> 4 then
                 malformed esc
                   (`Sdpa_reject
                      (Attention.Sdpa.Reject.Rank
                         {
                           arg_name = "sdpa attn_mask";
                           expected = [ 2; 4 ];
                           got;
                         })))
             mask_name;
           let params = { Attention.Sdpa.scale } in
           let* y =
             sdpa params ~query:(get "query") ~key:(get "key")
               ~value:(get "value")
               ?mask:(Option.map (env_find esc env) mask_name)
               ()
           in
           return [ y ]
       | "torch.ops.aten.pow.Tensor_Scalar" ->
           let exponent = required_scalar_arg esc node "exponent" in
           let* y = pow exponent (get "self") in
           return [ y ]
       | "torch.ops.aten.relu.default" ->
           let* y = relu (get "self") in
           return [ y ]
       (* Serialized twin of [Op_bridge_pointwise]'s [rsqrt.default] arm:
          legalizes onto the existing [Pow] node at exponent -0.5, the exact
          reciprocal-of-sqrt special case its [Compute] already implements. *)
       | "torch.ops.aten.rsqrt.default" ->
           let* y = pow (-0.5) (get "self") in
           return [ y ]
       | "torch.ops.aten.add.Tensor" -> (
           reject_alpha esc node;
           match tensor_or_scalar "other" with
           | `Tensor other ->
               let* y = add (get "self") other in
               return [ y ]
           | `Scalar s ->
               let* y = add_scalar s (get "self") in
               return [ y ])
       (* Serialized twin of [Op_bridge_pointwise]'s [addcmul.default] arm:
          self + value * tensor1 * tensor2, decomposed to [Mul]/[Add]/
          [Mul_scalar] rather than a new [Graph_ir] op -- see the bridge arm's
          comment. [Mul_scalar] is skipped for the
          verified default [value=1]. *)
       | "torch.ops.aten.addcmul.default" ->
           let value = scalar_arg esc ~default:1. node "value" in
           let* prod = mul (get "tensor1") (get "tensor2") in
           let* scaled =
             if Float.equal value 1. then return prod else mul_scalar value prod
           in
           let* y = add (get "self") scaled in
           return [ y ]
       (* [x - s] legalizes to [x + (-s)]: IEEE negation is exact and the
         builder narrows to f32 on both spellings either way, so the two are
         bit-identical (op3-impl.md F7). No [sub_scalar] builder exists and
         none should: this negation is the whole legalization. *)
       | "torch.ops.aten.sub.Tensor" -> (
           reject_alpha esc node;
           match tensor_or_scalar "other" with
           | `Tensor other ->
               let* y = sub (get "self") other in
               return [ y ]
           | `Scalar s ->
               let* y = add_scalar (-.s) (get "self") in
               return [ y ])
       (* `dtype` is decoded and rejected rather than silently dropped when
         present, the same treatment `linalg_vector_norm.default`'s gets. *)
       | "torch.ops.aten.sum.dim_IntList" ->
           reject_dtype esc node;
           let x_name = tensor_name esc node "self" in
           let rank =
             meta_rank (tensor_meta esc graph ~ssa:x_name ~role:`Sum_input)
           in
           let params =
             {
               Reduce.Sum.dims =
                 axes_for_rank esc ~tensor:x_name rank (ints_arg esc node "dim");
               keepdim = bool_arg esc node "keepdim";
             }
           in
           let* y = sum params (get "self") in
           return [ y ]
       | "torch.ops.aten.clamp.default" ->
           let params : Pointwise.Clamp.params =
             {
               min = scalar_opt_arg esc node "min";
               max = scalar_opt_arg esc node "max";
             }
           in
           let* y = clamp params (get "self") in
           return [ y ]
       (* clamp_min(self, min) = clamp(self, min=min, max=None): the same
          [Pointwise.Clamp] node [clamp.default] builds, one bound only. *)
       | "torch.ops.aten.clamp_min.default" ->
           let min = required_scalar_arg esc node "min" in
           let* y =
             clamp { Pointwise.Clamp.min = Some min; max = None } (get "self")
           in
           return [ y ]
       | "torch.ops.aten.clone.default" ->
           check_clone_memory_format esc node;
           let* y = clone (get "self") in
           return [ y ]
       | "torch.ops.aten.div.Tensor" -> (
           match tensor_or_scalar "other" with
           | `Tensor other ->
               let* y = div (get "self") other in
               return [ y ]
           | `Scalar s ->
               let* y = div_scalar s (get "self") in
               return [ y ])
       | "torch.ops.aten.hardsigmoid.default"
       | "torch.ops.aten.hardsigmoid_.default" ->
           let* y = hardsigmoid (get "self") in
           return [ y ]
       | "torch.ops.aten.hardswish.default"
       | "torch.ops.aten.hardswish_.default" ->
           let* y = hardswish (get "self") in
           return [ y ]
       | "torch.ops.aten.sigmoid.default" ->
           let* y = sigmoid (get "self") in
           return [ y ]
       | "torch.ops.aten.silu.default" | "torch.ops.aten.silu_.default" ->
           let* y = silu (get "self") in
           return [ y ]
       | "torch.ops.aten.gelu.default" ->
           let approximate =
             string_arg esc ~default:"none" node "approximate"
           in
           let approximate =
             match approximate with
             | "none" -> Pointwise.Gelu.Exact
             | "tanh" -> Pointwise.Gelu.Tanh
             | _ ->
                 malformed esc
                   (`Unsupported_option
                      { op = node.target; option = `Approximate approximate })
           in
           let* y = gelu approximate (get "self") in
           return [ y ]
       | "torch.ops.aten.hardtanh.default" ->
           (* Schema defaults are -1/1; MobileNet-v2 always serialises 0/6. *)
           let params : Pointwise.Hardtanh.params =
             {
               min_val = scalar_arg esc ~default:(-1.) node "min_val";
               max_val = scalar_arg esc ~default:1. node "max_val";
             }
           in
           let* y = hardtanh params (get "self") in
           return [ y ]
       | "torch.ops.aten.leaky_relu.default" ->
           let params : Pointwise.Leaky_relu.params =
             {
               negative_slope =
                 scalar_arg esc ~default:0.01 node "negative_slope";
             }
           in
           let* y = leaky_relu params (get "self") in
           return [ y ]
       | "torch.ops.aten.mul.Scalar" ->
           let s = required_scalar_arg esc node "other" in
           let* y = mul_scalar s (get "self") in
           return [ y ]
       | "torch.ops.aten.mul.Tensor" -> (
           match tensor_or_scalar "other" with
           | `Tensor other ->
               let* y = mul (get "self") other in
               return [ y ]
           | `Scalar s ->
               let* y = mul_scalar s (get "self") in
               return [ y ])
       (* The functional overload has ONE output and takes the generic path:
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
           if got <> 3 && got <> 4 then
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
       | "torch.ops.aten.mean.dim" ->
           let x_name = tensor_name esc node "self" in
           let rank =
             meta_rank (tensor_meta esc graph ~ssa:x_name ~role:`Mean_input)
           in
           let params =
             {
               Reduce.Mean.dims =
                 axes_for_rank esc ~tensor:x_name rank (ints_arg esc node "dim");
               keepdim = bool_arg esc node "keepdim";
             }
           in
           let* y = mean params (get "self") in
           return [ y ]
       | "torch.ops.aten.amax.default" ->
           let x_name = tensor_name esc node "self" in
           let rank =
             meta_rank (tensor_meta esc graph ~ssa:x_name ~role:`Amax_input)
           in
           let params =
             {
               Reduce.Amax.dims =
                 axes_for_rank esc ~tensor:x_name rank (ints_arg esc node "dim");
               keepdim = bool_arg esc node "keepdim";
             }
           in
           let* y = amax params (get "self") in
           return [ y ]
       | "torch.ops.aten.linalg_vector_norm.default" ->
           let ord = scalar_arg esc ~default:2. node "ord" in
           if not (Float.equal ord 2.) then
             malformed esc
               (`Unsupported_option
                  { op = node.target; option = `Vector_norm_ord ord });
           reject_dtype esc node;
           let x_name = tensor_name esc node "self" in
           let rank =
             meta_rank
               (tensor_meta esc graph ~ssa:x_name ~role:`Vector_norm_input)
           in
           let params =
             {
               Reduce.Vector_norm.dims =
                 axes_for_rank esc ~tensor:x_name rank (ints_arg esc node "dim");
               keepdim = bool_arg esc node "keepdim";
             }
           in
           let* y = vector_norm params (get "self") in
           return [ y ]
       (* [dim] is required by the schema (no default), unlike [mean.dim]'s /
          [amax.default]'s optional dim LIST -- but [int_arg]'s own default of
          0 is the established treatment every other single-[dim] importer arm
          here already gets ([unbind.int], [select.int], [slice.Tensor]), not
          a shortcut introduced for this op. See .ai/matmul_softmax_design.md
          §3. *)
       | "torch.ops.aten.softmax.int" ->
           reject_dtype esc node;
           let x_name = tensor_name esc node "self" in
           let rank =
             meta_rank (tensor_meta esc graph ~ssa:x_name ~role:`Softmax_input)
           in
           let axis =
             match
               axes_for_rank esc ~tensor:x_name rank [ int_arg esc node "dim" ]
             with
             | [ a ] -> a
             | _ ->
                 invalid_arg "Native_interp: axes_for_rank lost its singleton"
           in
           let* y = softmax { Reduce.Softmax.axis } (get "self") in
           return [ y ]
       (* Two shape families (`.ai/matmul_softmax_design.md` §4-5), same
         reasoning and same split as [Op_bridge]'s arm -- see there for the
         full accounting. Batch-less (rank-2, or rank>=3 with every axis but
         the last two at extent 1) binds to the EXISTING [Bmm] node;
         batched/multi-head (both operands the same rank>=3 but not
         batch-less) binds to [Batched_matmul], which validates full
         [N]/[T]/[D]/[H] agreement itself. Anything else (rank<2 on either
         side, or unequal ranks) is the ORIGINAL typed rejection, unchanged. *)
       | "torch.ops.aten.matmul.default" ->
           let a_name = tensor_name esc node "self" in
           let b_name = tensor_name esc node "other" in
           let a_sizes =
             static_sizes esc ~tensor:a_name
               (tensor_meta esc graph ~ssa:a_name ~role:`Matmul_self)
           in
           let b_sizes =
             static_sizes esc ~tensor:b_name
               (tensor_meta esc graph ~ssa:b_name ~role:`Matmul_other)
           in
           let batchless sizes =
             let rank = List.length sizes in
             rank >= 2
             && List.for_all (( = ) 1)
                  (List.filteri (fun i _ -> i < rank - 2) sizes)
           in
           let rank_a = List.length a_sizes and rank_b = List.length b_sizes in
           if batchless a_sizes && batchless b_sizes then
             let* y = bmm (get "self") (get "other") in
             return [ y ]
           else if rank_a = rank_b && rank_a >= 3 then
             let* y = batched_matmul (get "self") (get "other") in
             return [ y ]
           else
             malformed esc
               (`Matmul_unsupported_shape
                  { Matmul_unsupported_shape.self = a_sizes; other = b_sizes })
       | "torch.ops.aten.linear.default" ->
           let w_name = tensor_name esc node "weight" in
           let _out_features, in_features =
             sizes_rank_2 esc ~tensor:w_name
               (static_sizes esc ~tensor:w_name
                  (tensor_meta esc graph ~ssa:w_name ~role:`Linear_weight))
           in
           let* w = permute perm_linear_weight (get "weight") in
           let bias_name =
             optional_tensor_name ~absent_ok:true esc node "bias"
           in
           Option.iter
             (fun ssa ->
               require_rank esc graph ~ssa ~role:`Linear_bias ~expected:1)
             bias_name;
           let bias = Option.map (env_find esc env) bias_name in
           (* No check that the input's trailing extent matches [in_features]:
             [Linear.output_shape] already compares both the activation's and
             the weight's C against it (linear.ml:69-84), and a second copy of
             that rule here is one that could drift from it. *)
           let* y =
             linear
               {
                 Linear.Linear.in_features =
                   dim_extent esc ~tensor:w_name in_features;
               }
               ~x:(get "input") ~weight:w ?bias ()
           in
           return [ y ]
       | "torch.ops.aten.addmm.default" ->
           let w_name = tensor_name esc node "mat2" in
           let in_features =
             match
               static_sizes esc ~tensor:w_name
                 (tensor_meta esc graph ~ssa:w_name ~role:`Addmm_weight)
             with
             | n :: _ -> n
             | [] ->
                 (* A rank-zero mat2, which this arm cannot read a feature count
                   from. It used to report [`Symbolic], which was simply not
                   true; [`Expected_rank] arrived with linear.default's own rank
                   check and is the accurate row. Only the RANK-ZERO case is
                   rejected here -- addmm reads dim 0 and has never required
                   rank two, so demanding it would be a behaviour change rather
                   than a corrected diagnostic. *)
                 malformed esc
                   (`Bad_dimension
                      {
                        tensor = w_name;
                        fault = `Expected_rank { expected = 2; got = 0 };
                      })
           in
           let* w = permute perm_addmm_weight (get "mat2") in
           let* y =
             linear
               {
                 Linear.Linear.in_features =
                   dim_extent esc ~tensor:w_name in_features;
               }
               ~x:(get "mat1") ~weight:w ~bias:(get "self") ()
           in
           return [ y ]
       (* `index.Tensor(Tensor self, Tensor?[] indices) -> Tensor`, restricted
          to the one evidenced shape family (`.ai/index_tensor_design.md`):
          [indices] has exactly one live entry, of ATen rank 1, every other
          position [None]. Metadata-only, unlike [Op_bridge]'s arm: the live
          entry's dtype/rank come from [tensor_meta], never a materialized
          value, and its NAME may need tracing past a wrapping
          [clone.default] before it can be bound (round 6) -- a step
          [Op_bridge] does not need, since it already resolves the live
          entry's real ATen VALUE directly, clone or not. *)
       | "torch.ops.aten.index.Tensor" ->
           let self_name = tensor_name esc node "self" in
           let self_rank =
             meta_rank
               (tensor_meta esc graph ~ssa:self_name ~role:`Index_tensor_self)
           in
           let entries = optional_tensor_names_arg esc node "indices" in
           let got = List.length entries in
           let index_list_fail (fault : Index_list.fault) =
             malformed esc (`Index_list { Index_list.fault })
           in
           if got <> self_rank then
             index_list_fail
               (Index_list.Length_mismatch { expected = self_rank; got })
           else
             let live_positions =
               List.mapi (fun i e -> (i, e)) entries
               |> List.filter_map (fun (i, e) -> Option.map (fun _ -> i) e)
             in
             let p =
               match live_positions with
               | [] -> index_list_fail Index_list.No_live_entry
               | _ :: _ :: _ ->
                   index_list_fail
                     (Index_list.Multiple_live_entries live_positions)
               | [ p ] -> p
             in
             let live_name = Option.get (List.nth entries p) in
             let meta =
               tensor_meta esc graph ~ssa:live_name ~role:`Index_tensor_index
             in
             (match meta.TensorMeta.dtype with
             | ScalarType.LONG -> ()
             | dtype ->
                 index_list_fail
                   (Index_list.Wrong_dtype
                      { position = p; dtype = Pt2_dtype.scalar_type_name dtype }));
             let index_rank = meta_rank meta in
             if index_rank <> 1 then
               index_list_fail
                 (Index_list.Wrong_rank { position = p; rank = index_rank })
             else
               let axis =
                 match axes_for_rank esc ~tensor:self_name self_rank [ p ] with
                 | [ a ] -> a
                 | _ ->
                     invalid_arg
                       "Native_interp: axes_for_rank lost its singleton"
               in
               let index_name =
                 Native_interp_decode.resolve_index_source graph
                   ~constant_names:
                     ctx.Native_interp_lower_context.constant_names live_name
               in
               let* y =
                 index_tensor
                   { Index_tensor.Index_tensor.axis }
                   ~self:(get "self")
                   ~index:(env_find esc env index_name)
               in
               return [ y ]
       | _ -> assert false)
