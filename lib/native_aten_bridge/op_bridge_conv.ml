(* Conv2d family op-dispatch arms for [Op_bridge], split from op_bridge.ml. See op_bridge.ml for the [dispatch] facade that
   folds every family module (including this one) into one lookup. *)

open Pytorch_types
open Op_bridge_error
open Op_bridge_decode

let dispatch ~(aten_env : aten_env) (node : Node.t) :
    (Graph_ir.graph * (Graph_ir.Tensor_id.t * Tensor.packed) list, error) Err.t
    option =
  match node.target with
  | "torch.ops.aten.conv1d.default" ->
      Some
        (let* groups = int_arg ~default:1 node "groups" in
         let* dilation = ints_arg ~default:[ 1 ] node "dilation" in
         let* d = w1 "dilation" dilation in
         let* aten_x = tensor_arg aten_env node "input" in
         let* aten_w = tensor_arg aten_env node "weight" in
         let w_shape = Aten_tensor.shape aten_w in
         if Array.length w_shape <> 3 then
           fail (`Conv1d_invalid_weight_rank w_shape)
         else
           let* stride = ints_arg ~default:[ 1 ] node "stride" in
           let* padding = ints_arg ~default:[ 0 ] node "padding" in
           let* s = w1 "stride" stride in
           let* p = w1 "padding" padding in
           let* bias_opt =
             if optional_tensor_present node "bias" then
               let* bias = tensor_arg aten_env node "bias" in
               let* () = require_rank "bias" ~expected:1 bias in
               let* bias = native_of_aten "bias" bias in
               return (Some bias)
             else return None
           in
           let* x = native_of_aten "input" aten_x in
           let* w = native_of_aten "weight" aten_w in
           try
             let* params =
               make_conv1d_params ~op:node.Node.target w_shape s p d groups
             in
             let tensors = [ x; w ] @ Option.to_list bias_opt in
             build_g ~name:"conv1d_relayout" tensors (function
               | [ x_id; w_id ] ->
                   let open Graph_builder in
                   let* x' = permute perm_conv1d x_id in
                   let* w' = permute perm_conv1d w_id in
                   let* y' = conv1d params ~x:x' ~weight:w' () in
                   let+ y = permute perm_conv1d y' in
                   [ y ]
               | [ x_id; w_id; b_id ] ->
                   let open Graph_builder in
                   let* x' = permute perm_conv1d x_id in
                   let* w' = permute perm_conv1d w_id in
                   let* y' = conv1d params ~x:x' ~weight:w' ~bias:b_id () in
                   let+ y = permute perm_conv1d y' in
                   [ y ]
               | _ -> assert false)
           with Invalid_argument msg -> fail (`Validation_failure msg))
  | "torch.ops.aten.conv2d.default" ->
      Some
        (let* groups = int_arg ~default:1 node "groups" in
         let* dilation = ints_arg ~default:[ 1; 1 ] node "dilation" in
         let* dh, dw = hw2 "dilation" dilation in
         let* aten_x = tensor_arg aten_env node "input" in
         let* aten_w = tensor_arg aten_env node "weight" in
         let w_shape = Aten_tensor.shape aten_w in
         if Array.length w_shape <> 4 then
           fail (`Conv2d_invalid_weight_rank w_shape)
         else
           let* stride = ints_arg ~default:[ 1; 1 ] node "stride" in
           let* padding = ints_arg ~default:[ 0; 0 ] node "padding" in
           let* sh, sw = hw2 "stride" stride in
           let* ph, pw = hw2 "padding" padding in
           let* bias_opt =
             if optional_tensor_present node "bias" then
               let* bias = tensor_arg aten_env node "bias" in
               let* () = require_rank "bias" ~expected:1 bias in
               let* bias = native_of_aten "bias" bias in
               return (Some bias)
             else return None
           in
           let* x = native_of_aten "input" aten_x in
           let* w = native_of_aten "weight" aten_w in
           (* INSIDE the [try], with the rest of the parameter construction.
              Moving it out was justified by a mistake: a [try] does not
              interfere with the typed [Error] this returns, because that is a
              VALUE and [let*] short-circuits on it without raising. What the
              [try] catches is the other half -- [Dim.extent],
              [Op_config.Pos.of_int] and [Nonneg.of_int] assert their
              preconditions, and [Window_axis.factor] bounds magnitude only, so
              an ordinary [groups=0] or [stride=[0,1]] still reaches an
              assertion. Outside the boundary those escaped [dispatch] as an
              uncaught [Invalid_argument]. *)
           try
             let* params =
               make_conv2d_params ~op:node.Node.target w_shape sh sw ph pw dh dw
                 groups
             in
             let tensors = [ x; w ] @ Option.to_list bias_opt in
             build_g ~name:"conv2d_relayout" tensors (function
               | [ x_id; w_id ] ->
                   let open Graph_builder in
                   let* x' = permute perm_nchw_to_nhwc x_id in
                   let* w' = permute perm_oihw_to_conv_weight w_id in
                   let* y' = conv2d params ~x:x' ~weight:w' () in
                   let+ y = permute perm_nhwc_to_nchw y' in
                   [ y ]
               | [ x_id; w_id; b_id ] ->
                   let open Graph_builder in
                   let* x' = permute perm_nchw_to_nhwc x_id in
                   let* w' = permute perm_oihw_to_conv_weight w_id in
                   let* y' = conv2d params ~x:x' ~weight:w' ~bias:b_id () in
                   let+ y = permute perm_nhwc_to_nchw y' in
                   [ y ]
               | _ -> assert false)
           with Invalid_argument msg -> fail (`Validation_failure msg))
  | "torch.ops.aten.conv3d.default" ->
      Some
        (let* groups = int_arg ~default:1 node "groups" in
         let* dilation = ints_arg ~default:[ 1; 1; 1 ] node "dilation" in
         let* dd, dh, dw = dhw3 "dilation" dilation in
         let* aten_x = tensor_arg aten_env node "input" in
         let* aten_w = tensor_arg aten_env node "weight" in
         let w_shape = Aten_tensor.shape aten_w in
         if Array.length w_shape <> 5 then
           fail (`Conv3d_invalid_weight_rank w_shape)
         else
           let* stride = ints_arg ~default:[ 1; 1; 1 ] node "stride" in
           let* padding = ints_arg ~default:[ 0; 0; 0 ] node "padding" in
           let* sd, sh, sw = dhw3 "stride" stride in
           let* pd, ph, pw = dhw3 "padding" padding in
           let* bias_opt =
             if optional_tensor_present node "bias" then
               let* bias = tensor_arg aten_env node "bias" in
               let* () = require_rank "bias" ~expected:1 bias in
               let* bias = native_of_aten "bias" bias in
               return (Some bias)
             else return None
           in
           let* x = native_of_aten "input" aten_x in
           let* w = native_of_aten "weight" aten_w in
           try
             let* params =
               make_conv3d_params ~op:node.Node.target w_shape sd sh sw pd ph pw
                 dd dh dw groups
             in
             let tensors = [ x; w ] @ Option.to_list bias_opt in
             build_g ~name:"conv3d_relayout" tensors (function
               | [ x_id; w_id ] ->
                   let open Graph_builder in
                   let* x' = permute perm_conv3d x_id in
                   let* w' = permute perm_conv3d w_id in
                   let* y' = conv3d params ~x:x' ~weight:w' () in
                   let+ y = permute perm_conv3d_inv y' in
                   [ y ]
               | [ x_id; w_id; b_id ] ->
                   let open Graph_builder in
                   let* x' = permute perm_conv3d x_id in
                   let* w' = permute perm_conv3d w_id in
                   let* y' = conv3d params ~x:x' ~weight:w' ~bias:b_id () in
                   let+ y = permute perm_conv3d_inv y' in
                   [ y ]
               | _ -> assert false)
           with Invalid_argument msg -> fail (`Validation_failure msg))
  | "torch.ops.aten.conv2d.padding" ->
      Some
        (let* groups = int_arg ~default:1 node "groups" in
         let* dilation = ints_arg ~default:[ 1; 1 ] node "dilation" in
         let* dh, dw = hw2 "dilation" dilation in
         let* aten_x = tensor_arg aten_env node "input" in
         let* aten_w = tensor_arg aten_env node "weight" in
         let w_shape = Aten_tensor.shape aten_w in
         if Array.length w_shape <> 4 then
           fail (`Conv2d_padding_invalid_weight_rank w_shape)
         else
           let* stride = ints_arg ~default:[ 1; 1 ] node "stride" in
           let* padding = string_arg ~default:"valid" node "padding" in
           let* sh, sw = hw2 "stride" stride in
           let* bias_opt =
             if optional_tensor_present node "bias" then
               let* bias = tensor_arg aten_env node "bias" in
               let* () = require_rank "bias" ~expected:1 bias in
               let* bias = native_of_aten "bias" bias in
               return (Some bias)
             else return None
           in
           let* x = native_of_aten "input" aten_x in
           let* w = native_of_aten "weight" aten_w in
           try
             let* params =
               make_conv2d_padding_params ~op:node.Node.target sh sw padding dh
                 dw groups
             in
             let tensors = [ x; w ] @ Option.to_list bias_opt in
             build_g ~name:"conv2d_padding_relayout" tensors (function
               | [ x_id; w_id ] ->
                   let open Graph_builder in
                   let* x' = permute perm_nchw_to_nhwc x_id in
                   let* w' = permute perm_oihw_to_conv_weight w_id in
                   let* y' = conv2d_padding params ~x:x' ~weight:w' () in
                   let+ y = permute perm_nhwc_to_nchw y' in
                   [ y ]
               | [ x_id; w_id; b_id ] ->
                   let open Graph_builder in
                   let* x' = permute perm_nchw_to_nhwc x_id in
                   let* w' = permute perm_oihw_to_conv_weight w_id in
                   let* y' =
                     conv2d_padding params ~x:x' ~weight:w' ~bias:b_id ()
                   in
                   let+ y = permute perm_nhwc_to_nchw y' in
                   [ y ]
               | _ -> assert false)
           with Invalid_argument msg -> fail (`Validation_failure msg))
  | "torch.ops.aten.convolution.default" ->
      Some
        (let* transposed = bool_arg node "transposed" in
         let* groups = int_arg ~default:1 node "groups" in
         let* dilation = ints_arg ~default:[ 1; 1 ] node "dilation" in
         let* dh, dw = hw2 "dilation" dilation in
         let* aten_x = tensor_arg aten_env node "input" in
         let* aten_w = tensor_arg aten_env node "weight" in
         let w_shape = Aten_tensor.shape aten_w in
         if Array.length w_shape <> 4 then
           fail (`Convolution_invalid_weight_rank w_shape)
         else
           let* stride = ints_arg ~default:[ 1; 1 ] node "stride" in
           let* padding = ints_arg ~default:[ 0; 0 ] node "padding" in
           let* output_padding =
             ints_arg ~default:[ 0; 0 ] node "output_padding"
           in
           let* sh, sw = hw2 "stride" stride in
           let* ph, pw = hw2 "padding" padding in
           let* oph, opw = hw2 "output_padding" output_padding in
           let* bias_opt =
             if optional_tensor_present node "bias" then
               let* bias = tensor_arg aten_env node "bias" in
               let* () = require_rank "bias" ~expected:1 bias in
               let* bias = native_of_aten "bias" bias in
               return (Some bias)
             else return None
           in
           let* x = native_of_aten "input" aten_x in
           let* w = native_of_aten "weight" aten_w in
           try
             let* params =
               make_convolution_params ~op:node.Node.target sh sw ph pw dh dw
                 transposed oph opw groups
             in
             let tensors = [ x; w ] @ Option.to_list bias_opt in
             build_g ~name:"convolution_relayout" tensors (function
               | [ x_id; w_id ] ->
                   let open Graph_builder in
                   let* x' = permute perm_nchw_to_nhwc x_id in
                   let* w' = permute perm_oihw_to_conv_weight w_id in
                   let* y' = convolution params ~x:x' ~weight:w' () in
                   let+ y = permute perm_nhwc_to_nchw y' in
                   [ y ]
               | [ x_id; w_id; b_id ] ->
                   let open Graph_builder in
                   let* x' = permute perm_nchw_to_nhwc x_id in
                   let* w' = permute perm_oihw_to_conv_weight w_id in
                   let* y' =
                     convolution params ~x:x' ~weight:w' ~bias:b_id ()
                   in
                   let+ y = permute perm_nhwc_to_nchw y' in
                   [ y ]
               | _ -> assert false)
           with Invalid_argument msg -> fail (`Validation_failure msg))
  | _ -> None
