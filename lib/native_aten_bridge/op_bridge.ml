(* Native-side dispatch: given a graph node and the ATen environment (inputs),
   build a native Graph_ir.graph encoding the equivalent computation.

   Returns [None] if no native implementation exists for this op.  Returns
   [Some (Error e)] if the op is mapped but argument conversion or param
   validation fails.  Returns [Some (Ok (g, bindings))] where [g] is the native
   graph and [bindings] maps each graph input id to its converted native tensor.

   Ops requiring NCHW<->NHWC relayout (conv2d, max_pool2d, linear/addmm) produce
   a graph named "<op>_relayout" that wraps the core op in permute nodes.  All
   other ops produce a flat single-op graph.  Unmapped ops return None. *)

open Pytorch_types
module D = Interp_decode

type aten_env = Interp_decode.env
type invalid_hw_arg = { name : string; values : int list }
type tensor_bridge_error = { arg_name : string; message : string }

type error =
  [ `Decode of Interp_decode.error
  | `Tensor_bridge of tensor_bridge_error
  | `Build of Graph_builder.error
  | `Invalid_hw_arg of invalid_hw_arg
  | `Validation_failure of string
  | `Addmm_invalid_weight_rank of int array
  | `Conv2d_invalid_weight_rank of int array
  | `Conv2d_padding_invalid_weight_rank of int array
  | `Convolution_invalid_weight_rank of int array
  | `Linear_invalid_weight_rank of int array ]

let pp_int_list ppf xs =
  Format.fprintf ppf "[%s]" (String.concat ", " (List.map string_of_int xs))

let pp_int_array ppf xs = pp_int_list ppf (Array.to_list xs)

let pp_error ppf : [< error ] -> unit = function
  | `Decode e -> Interp_decode.pp_error ppf e
  | `Tensor_bridge { arg_name; message } ->
      Format.fprintf ppf "%s: %s" arg_name message
  | `Build e -> Graph_builder.pp_error ppf e
  | `Invalid_hw_arg { name; values } ->
      Format.fprintf ppf "%s: expected [h; w] or [v], got %a" name pp_int_list
        values
  | `Validation_failure msg -> Format.pp_print_string ppf msg
  | `Addmm_invalid_weight_rank shape ->
      Format.fprintf ppf "addmm: mat2 must be rank-2, got shape %a" pp_int_array
        shape
  | `Conv2d_invalid_weight_rank shape ->
      Format.fprintf ppf "conv2d: weight must be rank-4, got shape %a"
        pp_int_array shape
  | `Conv2d_padding_invalid_weight_rank shape ->
      Format.fprintf ppf "conv2d.padding: weight must be rank-4, got shape %a"
        pp_int_array shape
  | `Convolution_invalid_weight_rank shape ->
      Format.fprintf ppf "convolution: weight must be rank-4, got shape %a"
        pp_int_array shape
  | `Linear_invalid_weight_rank shape ->
      Format.fprintf ppf "linear: weight must be rank-2, got shape %a"
        pp_int_array shape

let ( let* ) = Core.Syntax.( let* )
let return = Core.return
let fail = Core.fail
let decode_result r = Core.map_error (fun e -> `Decode e) r
let tensor_arg env node name = decode_result (D.tensor_arg_result env node name)

let int_arg ?default node name =
  decode_result (D.int_arg_result ?default node name)

let ints_arg ?(default = []) node name =
  decode_result (D.ints_arg_result ~default node name)

let bool_arg ?(default = false) node name =
  decode_result (D.bool_arg_result ~default node name)

let string_arg ~default node name =
  decode_result (D.string_arg_result ~default node name)

let packed_shape (Tensor.Tensor r) = r.shape

(* Monadically allocate one input edge per packed tensor, left to right.
   The resulting ids match graph.inputs in insertion order. *)
let rec alloc_inputs = function
  | [] -> Graph_builder.return []
  | t :: rest ->
      let open Graph_builder in
      let* id = input ~shape:(packed_shape t) () in
      let+ ids = alloc_inputs rest in
      id :: ids

(* Build a graph from [tensors] (native packed args) and [body] mapping input
   ids to output ids.  Returns the graph and its input bindings [(id, packed)]. *)
let build_g ~name tensors body =
  let* g =
    Graph_builder.build ~name ~outputs:Fun.id
      (let open Graph_builder in
       let* ids = alloc_inputs tensors in
       body ids)
    |> Core.map_error (fun e -> `Build e)
  in
  return (g, List.combine g.Graph_ir.Graph.inputs tensors)

let some_graph = function Ok x -> Some (Ok x) | Error e -> Some (Error e)

(* Convert an ATen tensor to native, prefixing errors with [arg_name]. *)
let native_of_aten arg_name t =
  match Tensor_bridge.of_aten t with
  | Ok x -> Core.return x
  | Error message -> Core.fail (`Tensor_bridge { arg_name; message })

let native_tensor_arg aten_env node name =
  let* tensor = tensor_arg aten_env node name in
  native_of_aten name tensor

let optional_tensor_present node name =
  match D.find_arg node name with
  | Some (Argument.Tensor _)
  | Some (Argument.Optional_tensor (OptionalTensorArgument.Tensor _)) ->
      true
  | _ -> false

let aten_rank t = Array.length (Aten_tensor.shape t)

(* [D.scalar_arg_result]/[scalar_opt_arg_result] hand back an [Aten_scalar.t],
   which also admits a Bool the native scalar domain has no meaning for — so the
   narrowing to float is explicit and checked rather than assumed. *)
let float_of_aten_scalar name = function
  | Aten_scalar.Int i -> Core.return (Int64.to_float i)
  | Aten_scalar.Float f -> Core.return f
  | Aten_scalar.Bool _ ->
      Core.fail (`Validation_failure (name ^ ": expected a numeric scalar"))

let scalar_arg ~default node name =
  let* s = decode_result (D.scalar_arg_result ~default node name) in
  float_of_aten_scalar name s

let scalar_opt_arg node name =
  let* s = decode_result (D.scalar_opt_arg_result node name) in
  match s with
  | None -> return None
  | Some s ->
      let* f = float_of_aten_scalar name s in
      return (Some f)

(* [add.Tensor]/[sub.Tensor] compute [self + alpha * other]. No model in this zoo
   serialises a non-default alpha, so it is unimplemented — but silently dropping
   it would make the bridge claim semantics it does not deliver, and this arm is
   the one an ATen-vs-native comparison runs through. Reject instead. *)
let reject_alpha node =
  let* alpha = scalar_opt_arg node "alpha" in
  match alpha with
  | None -> return ()
  | Some a when Float.equal a 1. -> return ()
  | Some a ->
      fail
        (`Validation_failure
           (Printf.sprintf "alpha=%g is not supported (only 1)" a))

(* A Tensor-typed argument the exporter serialised as a bare scalar — the
   native twin of [Interp_decode.tensor_or_scalar_arg], except that the native
   side routes the scalar into an op parameter instead of materialising it. *)
let tensor_or_scalar aten_env node name =
  match D.find_arg node name with
  | Some (Argument.Int i) -> return (`Scalar (float_of_int i))
  | Some (Argument.Float f) -> return (`Scalar f)
  | _ ->
      let* t = native_tensor_arg aten_env node name in
      return (`Tensor t)

(* --- Permutations for NCHW/NHWC relayout ---
   All are full 6-axis bijections; see .ai/native_aten_bridge_layout.md. *)

(* Right-aligned rank-4 NCHW (D=Nbatch, H=Cch, W=Hsp, C=Wsp) ->
   channel-last in the same outer frame (D=Nbatch, H=Hsp, W=Wsp, C=Cch). *)
let perm_nchw_to_nhwc : Permute.Permute.perm =
  let open Axis in
  [ (N, N); (T, T); (D, D); (H, W); (W, C); (C, H) ]

(* Channel-last in the same outer frame -> right-aligned rank-4 NCHW.
   Inverse of [perm_nchw_to_nhwc]. *)
let perm_nhwc_to_nchw : Permute.Permute.perm =
  let open Axis in
  [ (N, N); (T, T); (D, D); (H, C); (W, H); (C, W) ]

(* OIHW conv weight (D=Cout, H=Cin, W=Kh, C=Kw) ->
   native [N=Cout, H=Kh, W=Kw, C=Cin].  Unlike activations, weights must move
   Cout onto N because Conv2d.output_shape reads output channels from weight N. *)
let perm_oihw_to_conv_weight : Permute.Permute.perm =
  let open Axis in
  [ (N, D); (T, T); (D, N); (H, W); (W, C); (C, H) ]

(* Rank-2 addmm weight [In,Out] (W=In, C=Out) -> native [N=Out, C=In]. *)
let perm_addmm_weight : Permute.Permute.perm =
  let open Axis in
  [ (N, C); (T, T); (D, D); (H, H); (W, N); (C, W) ]

(* Rank-2 linear weight [Out,In] (W=Out, C=In) -> native [N=Out, C=In]. *)
let perm_linear_weight : Permute.Permute.perm =
  let open Axis in
  [ (N, W); (T, T); (D, D); (H, H); (W, N); (C, C) ]

(* --- Arg helpers shared by mean.dim, rms_norm, permute --- *)

(* [mean.dim] and [rms_norm] reference frame axes through [Aten_shape.axis_of_dim],
   so the dims must be derived from the ATen input's RANK (not the right-aligned
   6D shape), keeping reduced axes consistent with where [of_aten] places the
   data. [aten.mean.dim] treats a missing/None/empty dim list as "all dims", so
   the bridge normalizes all three spellings the same way. *)
let dims_arg node ~rank name =
  match D.find_arg node name with
  | Some (Argument.Ints []) | Some (Argument.None _) | None ->
      Aten_shape.used_axes ~rank
  | Some (Argument.Ints xs) -> List.map (Aten_shape.axis_of_dim ~rank) xs
  | _ -> Aten_shape.used_axes ~rank

let trailing_axes ~rank ~k =
  let all = Aten_shape.used_axes ~rank in
  List.filteri (fun i _ -> i >= rank - k) all

let float32_eps = 1.1920929e-07

let eps_arg node name =
  match D.find_arg node name with
  | Some (Argument.Float f) -> f
  | _ -> float32_eps

let ones_weight (x_shape : Vec6.shape) dims =
  let ones = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:1 ~w:1 ~c:1 in
  let wshape =
    List.fold_left (fun s a -> Vec6.set s a (Vec6.get x_shape a)) ones dims
  in
  Tensor.materialize wshape (fun _ -> 1.0)

(* Build a full 6D native permutation from an ATen [dims] list and the tensor
   rank.  For the [rank] used axes, [dims.(i)] is the ATen input dim for output
   position [i]; outer padding axes are the identity. *)
let native_perm_of_aten ~rank dims =
  let used = Aten_shape.used_axes ~rank in
  let outer = List.filter (fun a -> not (List.mem a used)) Axis.all in
  let outer_perm = List.map (fun a -> (a, a)) outer in
  let inner_perm =
    List.mapi
      (fun i d ->
        (Aten_shape.axis_of_dim ~rank i, Aten_shape.axis_of_dim ~rank d))
      dims
  in
  outer_perm @ inner_perm

(* --- Param helpers for conv2d / pool2d --- *)

(* Validate a 2-element int list as [h; w].  A single-element list is accepted
   as [v; v] (symmetric). *)
let hw2 name = function
  | [ h; w ] -> Core.return (h, w)
  | [ v ] -> Core.return (v, v)
  | values -> Core.fail (`Invalid_hw_arg { name; values })

(* Construct Conv2d.params from the ATen weight shape array
   (rank-4: [Cout,Cin/groups,Kh,Kw]) and validated config ints.
   Raises [Invalid_argument] on bad dims. *)
let conv_axis_window ~kernel ~stride ~pad ~dilation : Conv.Conv2d.axis_window =
  {
    kernel = Dim.extent kernel;
    stride = Op_config.Pos.of_int stride;
    pad_before = Op_config.Nonneg.of_int pad;
    pad_after = Op_config.Nonneg.of_int pad;
    dilation = Op_config.Pos.of_int dilation;
  }

let make_conv2d_params w_shape sh sw ph pw dh dw groups =
  {
    Conv.Conv2d.h =
      conv_axis_window ~kernel:w_shape.(2) ~stride:sh ~pad:ph ~dilation:dh;
    w = conv_axis_window ~kernel:w_shape.(3) ~stride:sw ~pad:pw ~dilation:dw;
    in_channels = Dim.extent (w_shape.(1) * groups);
    groups = Op_config.Pos.of_int groups;
  }

let make_conv2d_padding_params sh sw padding dh dw groups :
    Conv.Conv2d_padding.params =
  {
    stride = { h = Op_config.Pos.of_int sh; w = Op_config.Pos.of_int sw };
    padding = Conv.Conv2d_padding.padding_of_string padding;
    dilation = { h = Op_config.Pos.of_int dh; w = Op_config.Pos.of_int dw };
    groups = Op_config.Pos.of_int groups;
  }

let make_convolution_params sh sw ph pw dh dw transposed oph opw groups :
    Conv.Convolution.params =
  {
    stride = { h = Op_config.Pos.of_int sh; w = Op_config.Pos.of_int sw };
    padding = { h = Op_config.Nonneg.of_int ph; w = Op_config.Nonneg.of_int pw };
    dilation = { h = Op_config.Pos.of_int dh; w = Op_config.Pos.of_int dw };
    transposed;
    output_padding =
      { h = Op_config.Nonneg.of_int oph; w = Op_config.Nonneg.of_int opw };
    groups = Op_config.Pos.of_int groups;
  }

(* Resolve a single inferred [-1] in a view/reshape size against the total
   numel (PyTorch convention). *)
let resolve_view_size size numel =
  if List.mem (-1) size then
    let known =
      List.fold_left (fun p d -> if d = -1 then p else p * d) 1 size
    in
    List.map (fun d -> if d = -1 then numel / known else d) size
  else size

(* Pool stride defaults to kernel_size when absent (PyTorch convention). *)
let pool_stride kernel_size node =
  let* stride = ints_arg ~default:[] node "stride" in
  return (match stride with [] -> kernel_size | s -> s)

(* --- Op dispatch --- *)

let dispatch ~(aten_env : aten_env) (node : Node.t) :
    ( Graph_ir.graph * (Graph_ir.Tensor_id.t * Tensor.packed) list,
      error )
    Core.result
    option =
  (* Arms in global alphabetical order by the dispatched op name. *)
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
         let eps = eps_arg node "eps" in
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
  | "torch.ops.aten.add.Tensor" | "torch.ops.aten.add_.Tensor" ->
      Some
        (let* () = reject_alpha node in
         let* a = native_tensor_arg aten_env node "self" in
         let* other = tensor_or_scalar aten_env node "other" in
         match other with
         | `Tensor b ->
             build_g ~name:"add" [ a; b ] (function
               | [ a_id; b_id ] ->
                   let open Graph_builder in
                   let+ y = add a_id b_id in
                   [ y ]
               | _ -> assert false)
         | `Scalar scalar ->
             build_g ~name:"add_scalar" [ a ] (function
               | [ a_id ] ->
                   let open Graph_builder in
                   let+ y = add_scalar scalar a_id in
                   [ y ]
               | _ -> assert false))
  | "torch.ops.aten.addmm.default" ->
      Some
        ((* addmm(bias, mat1, mat2) = bias + mat1 @ mat2; alpha=beta=1 assumed. *)
         let* aten_bias = tensor_arg aten_env node "self" in
         let* aten_x = tensor_arg aten_env node "mat1" in
         let* aten_w = tensor_arg aten_env node "mat2" in
         let w_shape = Aten_tensor.shape aten_w in
         if Array.length w_shape <> 2 then
           fail (`Addmm_invalid_weight_rank w_shape)
         else
           let* bias = native_of_aten "self" aten_bias in
           let* x = native_of_aten "mat1" aten_x in
           let* w = native_of_aten "mat2" aten_w in
           try
             let params =
               { Linear.Linear.in_features = Dim.extent w_shape.(0) }
             in
             build_g ~name:"addmm_relayout" [ bias; x; w ] (function
               | [ bias_id; x_id; w_id ] ->
                   let open Graph_builder in
                   let* w' = permute perm_addmm_weight w_id in
                   let+ y = linear params ~x:x_id ~weight:w' ~bias:bias_id () in
                   [ y ]
               | _ -> assert false)
           with Invalid_argument msg -> fail (`Validation_failure msg))
  | "torch.ops.aten.bmm.default" -> (
      match
        ( native_tensor_arg aten_env node "self",
          native_tensor_arg aten_env node "mat2" )
      with
      | Error e, _ | _, Error e -> Some (Error e)
      | Ok a, Ok b ->
          build_g ~name:"bmm" [ a; b ] (function
            | [ a_id; b_id ] ->
                let open Graph_builder in
                let+ y = bmm a_id b_id in
                [ y ]
            | _ -> assert false)
          |> some_graph)
  | "torch.ops.aten.clamp.default" | "torch.ops.aten.clamp_.default" ->
      Some
        (let* x = native_tensor_arg aten_env node "self" in
         let* min = scalar_opt_arg node "min" in
         let* max = scalar_opt_arg node "max" in
         (* The both-absent pair is rejected by [Graph_builder] via
            [Clamp.output_shape], the same place ATen's meta function rejects
            it, so no check is needed here. *)
         build_g ~name:"clamp" [ x ] (function
           | [ x_id ] ->
               let open Graph_builder in
               let+ y = clamp { Pointwise.Clamp.min; max } x_id in
               [ y ]
           | _ -> assert false))
  | "torch.ops.aten.clone.default" ->
      Some
        (let* x = native_tensor_arg aten_env node "self" in
         (* A requested memory_format is a layout change this op does not
            perform; rejecting beats silently ignoring it. *)
         match D.find_arg node "memory_format" with
         | Some (Argument.None _) | None ->
             build_g ~name:"clone" [ x ] (function
               | [ x_id ] ->
                   let open Graph_builder in
                   let+ y = clone x_id in
                   [ y ]
               | _ -> assert false)
         | Some _ ->
             fail (`Validation_failure "clone: memory_format is not supported"))
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
               let* bias = native_of_aten "bias" bias in
               return (Some bias)
             else return None
           in
           let* x = native_of_aten "input" aten_x in
           let* w = native_of_aten "weight" aten_w in
           try
             let params = make_conv2d_params w_shape sh sw ph pw dh dw groups in
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
               let* bias = native_of_aten "bias" bias in
               return (Some bias)
             else return None
           in
           let* x = native_of_aten "input" aten_x in
           let* w = native_of_aten "weight" aten_w in
           try
             let params =
               make_conv2d_padding_params sh sw padding dh dw groups
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
               let* bias = native_of_aten "bias" bias in
               return (Some bias)
             else return None
           in
           let* x = native_of_aten "input" aten_x in
           let* w = native_of_aten "weight" aten_w in
           try
             let params =
               make_convolution_params sh sw ph pw dh dw transposed oph opw
                 groups
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
  | "torch.ops.aten.div.Tensor" | "torch.ops.aten.div_.Tensor" ->
      Some
        (let* a = native_tensor_arg aten_env node "self" in
         let* other = tensor_or_scalar aten_env node "other" in
         match other with
         | `Tensor b ->
             build_g ~name:"div" [ a; b ] (function
               | [ a_id; b_id ] ->
                   let open Graph_builder in
                   let+ y = div a_id b_id in
                   [ y ]
               | _ -> assert false)
         | `Scalar scalar ->
             build_g ~name:"div_scalar" [ a ] (function
               | [ a_id ] ->
                   let open Graph_builder in
                   let+ y = div_scalar scalar a_id in
                   [ y ]
               | _ -> assert false))
  | "torch.ops.aten.hardtanh.default" | "torch.ops.aten.hardtanh_.default" ->
      Some
        (let* x = native_tensor_arg aten_env node "self" in
         (* Schema `Scalar min_val=-1, Scalar max_val=1`: both bounds may arrive
            as Int or Float, and either may be absent. *)
         let* min_val =
           scalar_arg ~default:(Aten_scalar.Float (-1.)) node "min_val"
         in
         let* max_val =
           scalar_arg ~default:(Aten_scalar.Float 1.) node "max_val"
         in
         build_g ~name:"hardtanh" [ x ] (function
           | [ x_id ] ->
               let open Graph_builder in
               let+ y = hardtanh { Pointwise.Hardtanh.min_val; max_val } x_id in
               [ y ]
           | _ -> assert false))
  | "torch.ops.aten.linear.default" ->
      Some
        (let* aten_x = tensor_arg aten_env node "input" in
         let* aten_w = tensor_arg aten_env node "weight" in
         let w_shape = Aten_tensor.shape aten_w in
         if Array.length w_shape <> 2 then
           fail (`Linear_invalid_weight_rank w_shape)
         else
           let* bias_opt =
             if optional_tensor_present node "bias" then
               let* bias = tensor_arg aten_env node "bias" in
               let* bias = native_of_aten "bias" bias in
               return (Some bias)
             else return None
           in
           let* x = native_of_aten "input" aten_x in
           let* w = native_of_aten "weight" aten_w in
           try
             let params =
               { Linear.Linear.in_features = Dim.extent w_shape.(1) }
             in
             let tensors = [ x; w ] @ Option.to_list bias_opt in
             build_g ~name:"linear_relayout" tensors (function
               | [ x_id; w_id ] ->
                   let open Graph_builder in
                   let* w' = permute perm_linear_weight w_id in
                   let+ y = linear params ~x:x_id ~weight:w' () in
                   [ y ]
               | [ x_id; w_id; b_id ] ->
                   let open Graph_builder in
                   let* w' = permute perm_linear_weight w_id in
                   let+ y = linear params ~x:x_id ~weight:w' ~bias:b_id () in
                   [ y ]
               | _ -> assert false)
           with Invalid_argument msg -> fail (`Validation_failure msg))
  | "torch.ops.aten.max_pool2d.default" ->
      Some
        (let* aten_x = tensor_arg aten_env node "self" in
         let* kernel_size = ints_arg node "kernel_size" in
         let* stride = pool_stride kernel_size node in
         let* padding = ints_arg ~default:[ 0; 0 ] node "padding" in
         let* kh, kw = hw2 "kernel_size" kernel_size in
         let* sh, sw = hw2 "stride" stride in
         let* ph, pw = hw2 "padding" padding in
         let* x = native_of_aten "self" aten_x in
         try
           let params =
             {
               Pool.MaxPool2d.kernel = { h = Dim.extent kh; w = Dim.extent kw };
               stride =
                 { h = Op_config.Pos.of_int sh; w = Op_config.Pos.of_int sw };
               pad =
                 {
                   h = Op_config.Nonneg.of_int ph;
                   w = Op_config.Nonneg.of_int pw;
                 };
             }
           in
           build_g ~name:"max_pool2d_relayout" [ x ] (function
             | [ x_id ] ->
                 let open Graph_builder in
                 let* x' = permute perm_nchw_to_nhwc x_id in
                 let* y' = max_pool2d params x' in
                 let+ y = permute perm_nhwc_to_nchw y' in
                 [ y ]
             | _ -> assert false)
         with Invalid_argument msg -> fail (`Validation_failure msg))
  | "torch.ops.aten.max_pool2d_with_indices.default" ->
      Some
        ((* Two ATen outputs (values, indices). We materialise both, relayout the
            value back to NCHW as the graph output, and route the dead indices
            edge into a Discard sink (see .ai/native_multi_output_design.md). *)
         let* aten_x = tensor_arg aten_env node "self" in
         let* kernel_size = ints_arg node "kernel_size" in
         let* stride = pool_stride kernel_size node in
         let* padding = ints_arg ~default:[ 0; 0 ] node "padding" in
         let* kh, kw = hw2 "kernel_size" kernel_size in
         let* sh, sw = hw2 "stride" stride in
         let* ph, pw = hw2 "padding" padding in
         let* x = native_of_aten "self" aten_x in
         try
           let params =
             {
               Pool.MaxPool2d.kernel = { h = Dim.extent kh; w = Dim.extent kw };
               stride =
                 { h = Op_config.Pos.of_int sh; w = Op_config.Pos.of_int sw };
               pad =
                 {
                   h = Op_config.Nonneg.of_int ph;
                   w = Op_config.Nonneg.of_int pw;
                 };
             }
           in
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
  | "torch.ops.aten.mean.dim" ->
      Some
        (let* t = tensor_arg aten_env node "self" in
         let rank = aten_rank t in
         let dims = dims_arg node ~rank "dim" in
         let* keepdim = bool_arg node "keepdim" in
         let* x = native_of_aten "self" t in
         let params = { Reduce.Mean.dims; keepdim } in
         build_g ~name:"mean" [ x ] (function
           | [ x_id ] ->
               let open Graph_builder in
               let+ y = mean params x_id in
               [ y ]
           | _ -> assert false))
  | "torch.ops.aten.mul.Tensor" | "torch.ops.aten.mul_.Tensor" -> (
      match
        ( native_tensor_arg aten_env node "self",
          native_tensor_arg aten_env node "other" )
      with
      | Error e, _ | _, Error e -> Some (Error e)
      | Ok a, Ok b ->
          build_g ~name:"mul" [ a; b ] (function
            | [ a_id; b_id ] ->
                let open Graph_builder in
                let+ y = mul a_id b_id in
                [ y ]
            | _ -> assert false)
          |> some_graph)
  | "torch.ops.aten.permute.default" ->
      Some
        (let* t = tensor_arg aten_env node "self" in
         let rank = aten_rank t in
         let* dims = ints_arg node "dims" in
         let perm = native_perm_of_aten ~rank dims in
         let* x = native_of_aten "self" t in
         build_g ~name:"permute" [ x ] (function
           | [ x_id ] ->
               let open Graph_builder in
               let+ y = permute perm x_id in
               [ y ]
           | _ -> assert false))
  | "torch.ops.aten.relu.default" | "torch.ops.aten.relu_.default" -> (
      match native_tensor_arg aten_env node "self" with
      | Error e -> Some (Error e)
      | Ok x ->
          build_g ~name:"relu" [ x ] (function
            | [ x_id ] ->
                let open Graph_builder in
                let+ y = relu x_id in
                [ y ]
            | _ -> assert false)
          |> some_graph)
  | "torch.ops.aten.rms_norm.default" ->
      Some
        (let* t = tensor_arg aten_env node "input" in
         let rank = aten_rank t in
         let* normalized_shape = ints_arg node "normalized_shape" in
         let k = List.length normalized_shape in
         let dims = trailing_axes ~rank ~k in
         let eps = eps_arg node "eps" in
         let params = { Norm.RmsNorm.dims; eps } in
         let* x = native_of_aten "input" t in
         let* weight =
           if optional_tensor_present node "weight" then
             let* weight = tensor_arg aten_env node "weight" in
             native_of_aten "weight" weight
           else
             let (Tensor.Tensor r) = x in
             return (ones_weight r.shape dims)
         in
         build_g ~name:"rms_norm" [ x; weight ] (function
           | [ x_id; w_id ] ->
               let open Graph_builder in
               let+ y = rms_norm params ~x:x_id ~weight:w_id () in
               [ y ]
           | _ -> assert false))
  | "torch.ops.aten.sqrt.default" | "torch.ops.aten.sqrt_.default" -> (
      match native_tensor_arg aten_env node "self" with
      | Error e -> Some (Error e)
      | Ok x ->
          build_g ~name:"sqrt" [ x ] (function
            | [ x_id ] ->
                let open Graph_builder in
                let+ y = sqrt x_id in
                [ y ]
            | _ -> assert false)
          |> some_graph)
  | "torch.ops.aten.sub.Tensor" | "torch.ops.aten.sub_.Tensor" -> (
      match
        ( (let* () = reject_alpha node in
           native_tensor_arg aten_env node "self"),
          native_tensor_arg aten_env node "other" )
      with
      | Error e, _ | _, Error e -> Some (Error e)
      | Ok a, Ok b ->
          build_g ~name:"sub" [ a; b ] (function
            | [ a_id; b_id ] ->
                let open Graph_builder in
                let+ y = sub a_id b_id in
                [ y ]
            | _ -> assert false)
          |> some_graph)
  | "torch.ops.aten.view.default" ->
      Some
        ((* Contiguous reshape. [of_aten] inputs are already ATen-row-major, so
            the native reshape needs no surrounding permutes; the target native
            shape is [size] right-aligned (a single -1 resolved against numel). *)
         let* aten_x = tensor_arg aten_env node "self" in
         let* size = ints_arg node "size" in
         let* x = native_of_aten "self" aten_x in
         let (Tensor.Tensor r) = x in
         let numel = (Vec6.numel r.shape :> int) in
         let resolved = resolve_view_size size numel in
         match Aten_shape.of_aten (Array.of_list resolved) with
         | Error e ->
             fail
               (`Validation_failure
                  (Format.asprintf "view: %a" Aten_shape.pp_error
                     e.Core.Error.kind))
         | Ok target ->
             let params = { Reshape.Reshape.shape = target } in
             build_g ~name:"view" [ x ] (function
               | [ x_id ] ->
                   let open Graph_builder in
                   let+ y = reshape params x_id in
                   [ y ]
               | _ -> assert false))
  | _ -> None
