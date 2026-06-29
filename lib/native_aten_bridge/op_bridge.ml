(* Native-side dispatch: given a graph node and the ATen environment (inputs),
   build a native Graph_ir.graph encoding the equivalent computation.

   Returns [None] if no native implementation exists for this op.  Returns
   [Some (Error msg)] if the op is mapped but argument conversion or param
   validation fails.  Returns [Some (Ok (g, bindings))] where [g] is the native
   graph and [bindings] maps each graph input id to its converted native tensor.

   Ops requiring NCHW<->NHWC relayout (conv2d, max_pool2d, linear/addmm) produce
   a graph named "<op>_relayout" that wraps the core op in permute nodes.  All
   other ops produce a flat single-op graph.  Unmapped ops return None. *)

open Pytorch_types
module D = Interp_decode

type aten_env = Interp_decode.env

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
  let g =
    Graph_builder.build ~name ~outputs:Fun.id
      (let open Graph_builder in
       let* ids = alloc_inputs tensors in
       body ids)
  in
  (g, List.combine g.Graph_ir.Graph.inputs tensors)

(* Convert an ATen tensor to native, prefixing errors with [arg_name]. *)
let native_of_aten arg_name t =
  match Tensor_bridge.of_aten t with
  | Ok x -> Ok x
  | Error e -> Error (Printf.sprintf "%s: %s" arg_name e)

let native_tensor_arg aten_env node name =
  native_of_aten name (D.tensor_arg aten_env node name)

let optional_tensor_present node name =
  match D.find_arg node name with
  | Some (Argument.Tensor _)
  | Some (Argument.Optional_tensor (OptionalTensorArgument.Tensor _)) ->
      true
  | _ -> false

let aten_rank t = Array.length (Aten_tensor.shape t)

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
   6D shape), keeping reduced axes consistent with where [of_aten] places the data. *)
let dims_arg node ~rank name =
  match D.find_arg node name with
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
  | [ h; w ] -> Ok (h, w)
  | [ v ] -> Ok (v, v)
  | _ -> Error (Printf.sprintf "%s: expected [h; w] or [v]" name)

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

(* Pool stride defaults to kernel_size when absent (PyTorch convention). *)
let pool_stride kernel_size node =
  match D.ints_arg ~default:[] node "stride" with [] -> kernel_size | s -> s

(* --- Op dispatch --- *)

let dispatch ~(aten_env : aten_env) (node : Node.t) :
    ( Graph_ir.graph * (Graph_ir.Tensor_id.t * Tensor.packed) list,
      string )
    result
    option =
  (* Arms in global alphabetical order by the dispatched op name. *)
  match node.target with
  | "torch.ops.aten.add.Tensor" | "torch.ops.aten.add_.Tensor" -> (
      match
        ( native_tensor_arg aten_env node "self",
          native_tensor_arg aten_env node "other" )
      with
      | Error e, _ | _, Error e -> Some (Error e)
      | Ok a, Ok b ->
          let g, bindings =
            build_g ~name:"add" [ a; b ] (function
              | [ a_id; b_id ] ->
                  let open Graph_builder in
                  let+ y = add a_id b_id in
                  [ y ]
              | _ -> assert false)
          in
          Some (Ok (g, bindings)))
  | "torch.ops.aten.addmm.default" -> (
      (* addmm(bias, mat1, mat2) = bias + mat1 @ mat2; alpha=beta=1 assumed. *)
      let aten_bias = D.tensor_arg aten_env node "self" in
      let aten_x = D.tensor_arg aten_env node "mat1" in
      let aten_w = D.tensor_arg aten_env node "mat2" in
      let w_shape = Aten_tensor.shape aten_w in
      if Array.length w_shape <> 2 then
        Some (Error "addmm: mat2 must be rank-2")
      else
        match
          ( native_of_aten "self" aten_bias,
            native_of_aten "mat1" aten_x,
            native_of_aten "mat2" aten_w )
        with
        | Error e, _, _ | _, Error e, _ | _, _, Error e -> Some (Error e)
        | Ok bias, Ok x, Ok w -> (
            try
              let params =
                { Linear.Linear.in_features = Dim.extent w_shape.(0) }
              in
              let g, bindings =
                build_g ~name:"addmm_relayout" [ bias; x; w ] (function
                  | [ bias_id; x_id; w_id ] ->
                      let open Graph_builder in
                      let* w' = permute perm_addmm_weight w_id in
                      let+ y =
                        linear params ~x:x_id ~weight:w' ~bias:bias_id ()
                      in
                      [ y ]
                  | _ -> assert false)
              in
              Some (Ok (g, bindings))
            with Invalid_argument msg -> Some (Error msg)))
  | "torch.ops.aten.bmm.default" -> (
      match
        ( native_tensor_arg aten_env node "self",
          native_tensor_arg aten_env node "mat2" )
      with
      | Error e, _ | _, Error e -> Some (Error e)
      | Ok a, Ok b ->
          let g, bindings =
            build_g ~name:"bmm" [ a; b ] (function
              | [ a_id; b_id ] ->
                  let open Graph_builder in
                  let+ y = bmm a_id b_id in
                  [ y ]
              | _ -> assert false)
          in
          Some (Ok (g, bindings)))
  | "torch.ops.aten.conv2d.default" | "torch.ops.aten.convolution.default" -> (
      (* Skip transposed convolutions (convolution.default only); native Conv2d
         models dense, grouped, and depthwise forward conv. *)
      let is_conv =
        String.equal node.target "torch.ops.aten.convolution.default"
      in
      if is_conv && D.bool_arg node "transposed" then None
      else
        let groups = D.int_arg ~default:1 node "groups" in
        let dilation = D.ints_arg ~default:[ 1; 1 ] node "dilation" in
        match hw2 "dilation" dilation with
        | Error e -> Some (Error e)
        | Ok (dh, dw) -> (
            let aten_x = D.tensor_arg aten_env node "input" in
            let aten_w = D.tensor_arg aten_env node "weight" in
            let w_shape = Aten_tensor.shape aten_w in
            if Array.length w_shape <> 4 then
              Some (Error "conv2d: weight must be rank-4")
            else
              let stride = D.ints_arg ~default:[ 1; 1 ] node "stride" in
              let padding = D.ints_arg ~default:[ 0; 0 ] node "padding" in
              match (hw2 "stride" stride, hw2 "padding" padding) with
              | Error e, _ | _, Error e -> Some (Error e)
              | Ok (sh, sw), Ok (ph, pw) -> (
                  let has_bias = optional_tensor_present node "bias" in
                  let bias_res =
                    if has_bias then
                      match
                        native_of_aten "bias"
                          (D.tensor_arg aten_env node "bias")
                      with
                      | Ok b -> Ok (Some b)
                      | Error e -> Error e
                    else Ok None
                  in
                  match
                    ( native_of_aten "input" aten_x,
                      native_of_aten "weight" aten_w,
                      bias_res )
                  with
                  | Error e, _, _ | _, Error e, _ | _, _, Error e ->
                      Some (Error e)
                  | Ok x, Ok w, Ok bias_opt -> (
                      try
                        let params =
                          make_conv2d_params w_shape sh sw ph pw dh dw groups
                        in
                        let tensors = [ x; w ] @ Option.to_list bias_opt in
                        let g, bindings =
                          build_g ~name:"conv2d_relayout" tensors (function
                            | [ x_id; w_id ] ->
                                let open Graph_builder in
                                let* x' = permute perm_nchw_to_nhwc x_id in
                                let* w' =
                                  permute perm_oihw_to_conv_weight w_id
                                in
                                let* y' = conv2d params ~x:x' ~weight:w' () in
                                let+ y = permute perm_nhwc_to_nchw y' in
                                [ y ]
                            | [ x_id; w_id; b_id ] ->
                                let open Graph_builder in
                                let* x' = permute perm_nchw_to_nhwc x_id in
                                let* w' =
                                  permute perm_oihw_to_conv_weight w_id
                                in
                                let* y' =
                                  conv2d params ~x:x' ~weight:w' ~bias:b_id ()
                                in
                                let+ y = permute perm_nhwc_to_nchw y' in
                                [ y ]
                            | _ -> assert false)
                        in
                        Some (Ok (g, bindings))
                      with Invalid_argument msg -> Some (Error msg)))))
  | "torch.ops.aten.linear.default" -> (
      let aten_x = D.tensor_arg aten_env node "input" in
      let aten_w = D.tensor_arg aten_env node "weight" in
      let w_shape = Aten_tensor.shape aten_w in
      if Array.length w_shape <> 2 then
        Some (Error "linear: weight must be rank-2")
      else
        let has_bias = optional_tensor_present node "bias" in
        let bias_res =
          if has_bias then
            match native_of_aten "bias" (D.tensor_arg aten_env node "bias") with
            | Ok b -> Ok (Some b)
            | Error e -> Error e
          else Ok None
        in
        match
          ( native_of_aten "input" aten_x,
            native_of_aten "weight" aten_w,
            bias_res )
        with
        | Error e, _, _ | _, Error e, _ | _, _, Error e -> Some (Error e)
        | Ok x, Ok w, Ok bias_opt -> (
            try
              let params =
                { Linear.Linear.in_features = Dim.extent w_shape.(1) }
              in
              let tensors = [ x; w ] @ Option.to_list bias_opt in
              let g, bindings =
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
              in
              Some (Ok (g, bindings))
            with Invalid_argument msg -> Some (Error msg)))
  | "torch.ops.aten.max_pool2d.default" -> (
      let aten_x = D.tensor_arg aten_env node "self" in
      let kernel_size = D.ints_arg node "kernel_size" in
      let stride = pool_stride kernel_size node in
      let padding = D.ints_arg ~default:[ 0; 0 ] node "padding" in
      match
        ( hw2 "kernel_size" kernel_size,
          hw2 "stride" stride,
          hw2 "padding" padding )
      with
      | Error e, _, _ | _, Error e, _ | _, _, Error e -> Some (Error e)
      | Ok (kh, kw), Ok (sh, sw), Ok (ph, pw) -> (
          match native_of_aten "self" aten_x with
          | Error e -> Some (Error e)
          | Ok x -> (
              try
                let params =
                  {
                    Pool.MaxPool2d.kernel =
                      { h = Dim.extent kh; w = Dim.extent kw };
                    stride =
                      {
                        h = Op_config.Pos.of_int sh;
                        w = Op_config.Pos.of_int sw;
                      };
                    pad =
                      {
                        h = Op_config.Nonneg.of_int ph;
                        w = Op_config.Nonneg.of_int pw;
                      };
                  }
                in
                let g, bindings =
                  build_g ~name:"max_pool2d_relayout" [ x ] (function
                    | [ x_id ] ->
                        let open Graph_builder in
                        let* x' = permute perm_nchw_to_nhwc x_id in
                        let* y' = max_pool2d params x' in
                        let+ y = permute perm_nhwc_to_nchw y' in
                        [ y ]
                    | _ -> assert false)
                in
                Some (Ok (g, bindings))
              with Invalid_argument msg -> Some (Error msg))))
  | "torch.ops.aten.mean.dim" -> (
      let t = D.tensor_arg aten_env node "self" in
      let rank = aten_rank t in
      let dims = dims_arg node ~rank "dim" in
      let keepdim = D.bool_arg node "keepdim" in
      match native_of_aten "self" t with
      | Error e -> Some (Error e)
      | Ok x ->
          let params = { Reduce.Mean.dims; keepdim } in
          let g, bindings =
            build_g ~name:"mean" [ x ] (function
              | [ x_id ] ->
                  let open Graph_builder in
                  let+ y = mean params x_id in
                  [ y ]
              | _ -> assert false)
          in
          Some (Ok (g, bindings)))
  | "torch.ops.aten.mul.Tensor" | "torch.ops.aten.mul_.Tensor" -> (
      match
        ( native_tensor_arg aten_env node "self",
          native_tensor_arg aten_env node "other" )
      with
      | Error e, _ | _, Error e -> Some (Error e)
      | Ok a, Ok b ->
          let g, bindings =
            build_g ~name:"mul" [ a; b ] (function
              | [ a_id; b_id ] ->
                  let open Graph_builder in
                  let+ y = mul a_id b_id in
                  [ y ]
              | _ -> assert false)
          in
          Some (Ok (g, bindings)))
  | "torch.ops.aten.permute.default" -> (
      let t = D.tensor_arg aten_env node "self" in
      let rank = aten_rank t in
      let dims = D.ints_arg node "dims" in
      let perm = native_perm_of_aten ~rank dims in
      match native_of_aten "self" t with
      | Error e -> Some (Error e)
      | Ok x ->
          let g, bindings =
            build_g ~name:"permute" [ x ] (function
              | [ x_id ] ->
                  let open Graph_builder in
                  let+ y = permute perm x_id in
                  [ y ]
              | _ -> assert false)
          in
          Some (Ok (g, bindings)))
  | "torch.ops.aten.relu.default" | "torch.ops.aten.relu_.default" -> (
      match native_tensor_arg aten_env node "self" with
      | Error e -> Some (Error e)
      | Ok x ->
          let g, bindings =
            build_g ~name:"relu" [ x ] (function
              | [ x_id ] ->
                  let open Graph_builder in
                  let+ y = relu x_id in
                  [ y ]
              | _ -> assert false)
          in
          Some (Ok (g, bindings)))
  | "torch.ops.aten.rms_norm.default" -> (
      let t = D.tensor_arg aten_env node "input" in
      let rank = aten_rank t in
      let k = List.length (D.ints_arg node "normalized_shape") in
      let dims = trailing_axes ~rank ~k in
      let eps = eps_arg node "eps" in
      let params = { Norm.RmsNorm.dims; eps } in
      match native_of_aten "input" t with
      | Error e -> Some (Error e)
      | Ok x -> (
          let weight_res =
            if optional_tensor_present node "weight" then
              native_of_aten "weight" (D.tensor_arg aten_env node "weight")
            else
              let (Tensor.Tensor r) = x in
              Ok (ones_weight r.shape dims)
          in
          match weight_res with
          | Error e -> Some (Error e)
          | Ok weight ->
              let g, bindings =
                build_g ~name:"rms_norm" [ x; weight ] (function
                  | [ x_id; w_id ] ->
                      let open Graph_builder in
                      let+ y = rms_norm params ~x:x_id ~weight:w_id () in
                      [ y ]
                  | _ -> assert false)
              in
              Some (Ok (g, bindings))))
  | _ -> None
