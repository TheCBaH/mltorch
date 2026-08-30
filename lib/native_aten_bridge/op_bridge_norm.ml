(* Normalization family (batch/group/layer/rms) op-dispatch arms for [Op_bridge], split from op_bridge.ml. See op_bridge.ml for the [dispatch] facade that
   folds every family module (including this one) into one lookup. *)

open Pytorch_types
open Op_bridge_error
open Op_bridge_decode

let dispatch ~(aten_env : aten_env) (node : Node.t) :
    (Graph_ir.graph * (Graph_ir.Tensor_id.t * Tensor.packed) list, error) Err.t
    option =
  match node.target with
  | "torch.ops.aten._native_batch_norm_legit_no_training.default" ->
      Some
        ((* Inference batch norm: only out0 (the normalised activations) is
            represented. ATen also returns save_mean/save_invstd, but in eval mode
            those are recorded size-[0] in the exported graph and are dropped
            (the engine has no empty tensors). See
            .ai/native_multi_output_design.md. *)
         let* aten_x = tensor_arg aten_env node "input" in
         let* x = native_of_aten "input" aten_x in
         let* aten_rm = tensor_arg aten_env node "running_mean" in
         let* rm = native_of_aten "running_mean" aten_rm in
         let* aten_rv = tensor_arg aten_env node "running_var" in
         let* rv = native_of_aten "running_var" aten_rv in
         let* eps = float_arg node "eps" in
         let (Tensor.Tensor rm_r) = rm in
         (* weight/bias are optional (ATen `Tensor?`); materialise the identity
            (ones / zeros) [C] vector when absent, as [rms_norm] does. *)
         let* weight =
           if optional_tensor_present node "weight" then
             let* w = tensor_arg aten_env node "weight" in
             native_of_aten "weight" w
           else return (Tensor.materialize rm_r.shape (fun _ -> 1.))
         in
         let* bias =
           if optional_tensor_present node "bias" then
             let* b = tensor_arg aten_env node "bias" in
             native_of_aten "bias" b
           else return (Tensor.materialize rm_r.shape (fun _ -> 0.))
         in
         let params = { Norm.BatchNorm.channel = Axis.C; eps } in
         build_g ~name:"batch_norm_relayout" [ x; weight; bias; rm; rv ]
           (function
           | [ x_id; w_id; b_id; rm_id; rv_id ] ->
               let open Graph_builder in
               let* x' = permute perm_nchw_to_nhwc x_id in
               let* y' =
                 batch_norm params ~x:x' ~weight:w_id ~bias:b_id
                   ~running_mean:rm_id ~running_var:rv_id ()
               in
               let+ y = permute perm_nhwc_to_nchw y' in
               [ y ]
           | _ -> assert false))
  | "torch.ops.aten._native_batch_norm_legit.no_stats" ->
      Some
        (let* training = bool_arg node "training" in
         if not training then
           fail
             (`Validation_failure
                "_native_batch_norm_legit.no_stats: training=false is not \
                 supported (only true)")
         else
           let* momentum = float_arg node "momentum" in
           if not (Float.equal momentum 0.) then
             fail
               (`Validation_failure
                  (Format.sprintf
                     "_native_batch_norm_legit.no_stats: momentum=%g is not \
                      supported (only 0)"
                     momentum))
           else
             let* eps = float_arg node "eps" in
             let* aten_x = tensor_arg aten_env node "input" in
             let to_channel_last, from_channel_last =
               batch_norm_channel_perms ~rank:(aten_rank aten_x)
             in
             let* x = native_of_aten "input" aten_x in
             let* affine =
               Err.List.map
                 (fun name ->
                   if optional_tensor_present node name then
                     let* t = tensor_arg aten_env node name in
                     let* t = native_of_aten name t in
                     return (Some t)
                   else return None)
                 [ "weight"; "bias" ]
             in
             let weight, bias =
               match affine with
               | [ weight; bias ] -> (weight, bias)
               | _ -> assert false
             in
             let params = { Norm.BatchNormNoStats.channel = Axis.C; eps } in
             build_g ~name:"batch_norm_no_stats_relayout"
               (([ x ] @ Option.to_list weight) @ Option.to_list bias)
               (fun ids ->
                 let open Graph_builder in
                 let x_id, weight, bias =
                   match (ids, weight, bias) with
                   | [ x_id; weight; bias ], Some _, Some _ ->
                       (x_id, Some weight, Some bias)
                   | [ x_id; weight ], Some _, None -> (x_id, Some weight, None)
                   | [ x_id; bias ], None, Some _ -> (x_id, None, Some bias)
                   | [ x_id ], None, None -> (x_id, None, None)
                   | _ -> assert false
                 in
                 let* x' = permute to_channel_last x_id in
                 let* ys = batch_norm_no_stats params ~x:x' ?weight ?bias () in
                 match ys with
                 | [ y'; mean; invstd ] ->
                     let+ y = permute from_channel_last y' in
                     [ y; mean; invstd ]
                 | _ -> assert false))
  (* [input] is right-aligned NCHW like [batch_norm]'s, so the same
     [perm_nchw_to_nhwc]/[perm_nhwc_to_nchw] pair relays it around the op --
     [Group_norm.Compute] reads its window on native C, which is where the
     permute puts ATen's channel axis. Unlike [batch_norm]'s arm, absent
     [weight]/[bias] stay [None] rather than being materialised as ones/zeros
     tensors here, for the reason the [layer_norm]/[native_layer_norm] arm
     below states at length: [Graph_ir] carries them as options and [Eval_op]
     fills them, so materialising upstream would build a structurally
     different graph from [Native_interp]'s.

     NOT bound in [bin/aten_ops_gen.ml]'s [selection] (deliberately, unlike
     every other op this bridge lowers): `at::native::group_norm`'s CPU
     kernel is outside the hand-curated source closure [lib/aten/
     build_archive.sh] compiles (see its own header comment on why that
     closure is small on purpose) -- pulling it in undefined-symbol'd every
     binary linking [aten] at [RegisterCompositeImplicitAutograd_0.cpp].
     This arm needs no ATen C binding to build a Native graph, only
     [Interp_decode]'s node-argument helpers, so real-model import and the
     payload-free sweep both work; only real-ATen verification
     ([Interp_verify]/[Interp_dispatch]) is unavailable for this op until
     that closure grows. *)
  | "torch.ops.aten.group_norm.default" ->
      Some
        (let* aten_x = tensor_arg aten_env node "input" in
         let* num_groups = int_arg node "num_groups" in
         let* eps = float_arg ~default:1e-05 node "eps" in
         (* Decoded, not ignored, then discarded -- the cuDNN implementation
            hint, the same treatment [layer_norm.default]'s [cudnn_enable]
            gets below and for the same reason: a non-boolean here is a
            malformed node, and this is what says so. *)
         let* (_ : bool) = bool_arg ~default:true node "cudnn_enabled" in
         let* groups = pos ~op:"group_norm.default" ~param:`Groups num_groups in
         let* x = native_of_aten "input" aten_x in
         let* affine =
           Err.List.map
             (fun name ->
               if optional_tensor_present node name then
                 let* t = tensor_arg aten_env node name in
                 let* () =
                   require_rank
                     (Printf.sprintf "group_norm %s" name)
                     ~expected:1 t
                 in
                 let* t = native_of_aten name t in
                 return (Some t)
               else return None)
             [ "weight"; "bias" ]
         in
         let weight_opt, bias_opt =
           match affine with [ w; b ] -> (w, b) | _ -> assert false
         in
         let params = { Norm.GroupNorm.channel = Axis.C; groups; eps } in
         build_g ~name:"group_norm_relayout"
           (([ x ] @ Option.to_list weight_opt) @ Option.to_list bias_opt)
           (fun ids ->
             let open Graph_builder in
             (* All FOUR states spelled out, matched against the options the
                operand list was built from -- the same discipline
                [layer_norm]'s arm below uses and for the same reason: "bias
                but no weight" is a state a paired encoding would get wrong. *)
             let x_id, weight, bias =
               match (ids, weight_opt, bias_opt) with
               | [ x_id; w_id; b_id ], Some _, Some _ ->
                   (x_id, Some w_id, Some b_id)
               | [ x_id; w_id ], Some _, None -> (x_id, Some w_id, None)
               | [ x_id; b_id ], None, Some _ -> (x_id, None, Some b_id)
               | [ x_id ], None, None -> (x_id, None, None)
               | _ -> assert false
             in
             let* x' = permute perm_nchw_to_nhwc x_id in
             let* y' = group_norm params ~x:x' ?weight ?bias () in
             let+ y = permute perm_nhwc_to_nchw y' in
             [ y ]))
  (* The FUNCTIONAL layer norm and its DECOMPOSED twin, in one body. They differ
     in three things and in nothing else: [native_layer_norm]'s [eps] is
     required with no schema default, it has no [cudnn_enable], and it returns
     a 3-tuple. The arithmetic, the axis derivation, the normalized_shape
     validation and the affine handling are identical, so they are written once.

     THE 3-TUPLE NEEDS NOTHING HERE. [Verify.requires_exact_outputs] is true
     only for a dynamic [Argument.Tensors] return; a fixed tuple falls under the
     leading-outputs rule, so exposing one output is legitimate and is verified
     against the first ATen result alone. The liveness question -- whether a
     graph READS [mean] or [rstd] -- is a whole-graph property this single-node
     bridge cannot see, and [Native_interp] is where it is answered
     ([`Live_layer_norm_stats]). *)
  | ( "torch.ops.aten.layer_norm.default"
    | "torch.ops.aten.native_layer_norm.default" ) as target ->
      Some
        (let functional = target = "torch.ops.aten.layer_norm.default" in
         let op =
           if functional then Norm.Target.Layer_norm
           else Norm.Target.Native_layer_norm
         in
         let* t = tensor_arg aten_env node "input" in
         let* normalized_shape = ints_arg node "normalized_shape" in
         let* dims =
           normalized_dims ~op ~x_shape:(Aten_tensor.shape t) ~normalized_shape
         in
         (* [layer_norm]'s eps is a REQUIRED float with a schema default of
            1e-5, so [float_arg ~default], not [eps_arg]: rms_norm's eps is a
            [float?] whose absence means "ATen picks", and reading one with the
            other's decoder is how a null epsilon comes to be read as zero.
            [native_layer_norm]'s has no default at all -- a third spelling of
            the same argument -- so its absence is a malformed node. *)
         let* eps =
           if functional then float_arg ~default:1e-05 node "eps"
           else float_arg node "eps"
         in
         (* Decoded, not ignored, and then deliberately DISCARDED -- which is
            the one argument in this repository where that is the faithful
            reading rather than the [alpha]-shaped bug. ATen's own composite is
            [layer_norm_symint (..., bool /* cudnn_enable, deprecated */)]: it
            names the parameter in a comment and drops it, computing
            native_layer_norm either way. Accepting only the value some corpus
            happens to show would reject the schema's own default of true and
            with it almost every real node. Decoding it still matters: a
            non-boolean there is a malformed node, and this is what says so.
            The decomposed form does not carry the argument at all -- none
            survives export -- so it is read only for the functional one. *)
         let* () =
           if functional then
             let* (_ : bool) = bool_arg ~default:true node "cudnn_enable" in
             return ()
           else return ()
         in
         let params = { Norm.LayerNorm.dims; eps } in
         let* x = native_of_aten "input" t in
         (* NO ones/zeros tensors for absent affine operands, for the reason the
            rms_norm arm below states at length: [Graph_ir] carries them as
            options, [Eval_op] fills them, and materialising here would build a
            structurally different graph from [Native_interp]'s. *)
         let k = List.length normalized_shape in
         let* affine =
           Err.List.map
             (fun name ->
               if optional_tensor_present node name then
                 let* t = tensor_arg aten_env node name in
                 (* Rank [k], the length of normalized_shape: ATen indexes both
                    affine operands by the whole normalized shape. *)
                 let* () =
                   require_rank
                     (Fmt.str "%a %s" Norm.Target.pp op name)
                     ~expected:k t
                 in
                 let* t = native_of_aten name t in
                 return (Some t)
               else return None)
             [ "weight"; "bias" ]
         in
         let weight_opt, bias_opt =
           match affine with [ w; b ] -> (w, b) | _ -> assert false
         in
         build_g ~name:"layer_norm"
           (([ x ] @ Option.to_list weight_opt) @ Option.to_list bias_opt)
           (fun ids ->
             let open Graph_builder in
             (* All FOUR states spelled out, matched against the options the
                operand list was built from -- so the id positions and the
                options cannot disagree. "bias but no weight" is the state no
                model produces and the one a paired encoding would get wrong,
                which is why it is written rather than folded away. *)
             let x_id, weight, bias =
               match (ids, weight_opt, bias_opt) with
               | [ x ], None, None -> (x, None, None)
               | [ x; w ], Some _, None -> (x, Some w, None)
               | [ x; b ], None, Some _ -> (x, None, Some b)
               | [ x; w; b ], Some _, Some _ -> (x, Some w, Some b)
               | _ -> assert false
             in
             let+ y = layer_norm params ~x:x_id ?weight ?bias () in
             [ y ]))
  | "torch.ops.aten.rms_norm.default" ->
      Some
        (let* t = tensor_arg aten_env node "input" in
         let* normalized_shape = ints_arg node "normalized_shape" in
         let* dims =
           normalized_dims ~op:Norm.Target.Rms_norm
             ~x_shape:(Aten_tensor.shape t) ~normalized_shape
         in
         let* eps = eps_arg node "eps" in
         let params = { Norm.RmsNorm.dims; eps } in
         let* x = native_of_aten "input" t in
         (* NO ones tensor for an absent weight. [Graph_ir]'s [Rms_norm] carries
            [weight : Tensor_ref.t option] and Native4D reads the option
            (lower.ml:293-299); materializing a constant made this path build a
            structurally different graph from [Native_interp]'s for the same
            node, and left that arm unreachable from here. The numeric result is
            unchanged -- multiplying by ones is what the option's absence
            means. *)
         let* weight_opt =
           if optional_tensor_present node "weight" then
             let* weight = tensor_arg aten_env node "weight" in
             (* Rank [k], the length of normalized_shape: ATen indexes the
                weight by the whole normalized shape. *)
             let* () =
               require_rank "rms_norm weight"
                 ~expected:(List.length normalized_shape)
                 weight
             in
             let* weight = native_of_aten "weight" weight in
             return (Some weight)
           else return None
         in
         build_g ~name:"rms_norm"
           ([ x ] @ Option.to_list weight_opt)
           (function
             | [ x_id ] ->
                 let open Graph_builder in
                 let+ y = rms_norm params ~x:x_id () in
                 [ y ]
             | [ x_id; w_id ] ->
                 let open Graph_builder in
                 let+ y = rms_norm params ~x:x_id ~weight:w_id () in
                 [ y ]
             | _ -> assert false))
  | _ -> None
