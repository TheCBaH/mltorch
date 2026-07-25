open Pytorch_types
open Schema_runtime
module Tensor_id = Graph_ir.Tensor_id
module Node_id = Graph_ir.Node_id

type error =
  [ `Unsupported_input of string
  | `Unsupported_operator of string
  | `Malformed_graph of string
  | `Tensor_bridge of string
  | `Eval of Eval_direct.error
  | `Build of Graph_builder.error
  | `Provenance of Pt2_native_graph.error ]

type hooks =
  | Hooks : {
      on_start : Pt2_native_graph.t -> Graph_ir.node -> 'a;
      on_end : Pt2_native_graph.t -> Graph_ir.node -> 'a -> unit;
    }
      -> hooks

let pp_error ppf : [< error ] -> unit = function
  | `Unsupported_input s -> Fmt.pf ppf "unsupported PT2 input: %s" s
  | `Unsupported_operator s -> Fmt.pf ppf "unsupported PT2 operator: %s" s
  | `Malformed_graph s -> Fmt.pf ppf "malformed PT2 graph: %s" s
  | `Tensor_bridge s -> Fmt.pf ppf "PT2 tensor bridge: %s" s
  | `Eval e -> Eval_direct.pp_error ppf e
  | `Build e -> Graph_builder.pp_error ppf e
  | `Provenance e -> Pt2_native_graph.pp_error ppf e

exception Lower_error of error

let malformed fmt =
  Printf.ksprintf (fun s -> raise (Lower_error (`Malformed_graph s))) fmt

let shape_of_sizes name sizes =
  let dims =
    List.map
      (function
        | SymInt.Int i when i >= 0 -> i
        | SymInt.Int i -> malformed "%s has negative dimension %d" name i
        | SymInt.Expr _ -> malformed "%s has a symbolic dimension" name)
      sizes
  in
  match List.rev dims with
  | [ c; w; h; d; t; n ] -> Vec6.shape ~n ~t ~d ~h ~w ~c
  | [ c; w; h; d; t ] -> Vec6.shape ~n:1 ~t ~d ~h ~w ~c
  | [ c; w; h; d ] -> Vec6.shape ~n:1 ~t:1 ~d ~h ~w ~c
  | [ c; w; h ] -> Vec6.shape ~n:1 ~t:1 ~d:1 ~h ~w ~c
  | [ c; w ] -> Vec6.shape ~n:1 ~t:1 ~d:1 ~h:1 ~w ~c
  | c :: [] -> Vec6.shape ~n:1 ~t:1 ~d:1 ~h:1 ~w:1 ~c
  | [] -> Vec6.shape ~n:1 ~t:1 ~d:1 ~h:1 ~w:1 ~c:1
  | _ -> malformed "%s has rank greater than six" name

let tensor_shape (graph : Pytorch_types.Graph.t) name =
  match String_map.find_opt name graph.tensor_values with
  | Some meta -> shape_of_sizes name meta.TensorMeta.sizes
  | None -> malformed "no tensor metadata for %S" name

let find_arg (node : Pytorch_types.Node.t) name =
  match
    List.find_opt (fun (a : NamedArgument.t) -> a.name = name) node.Node.inputs
  with
  | Some a -> a.arg
  | None -> malformed "%s: missing argument %S" node.target name

let tensor_name (node : Pytorch_types.Node.t) name =
  match find_arg node name with
  | Argument.Tensor t -> t.TensorArgument.name
  | _ -> malformed "%s.%s is not a tensor" node.target name

let optional_tensor_name (node : Pytorch_types.Node.t) name =
  match find_arg node name with
  | Argument.Tensor t -> Some t.TensorArgument.name
  | Argument.None _ -> None
  | Argument.Optional_tensor (OptionalTensorArgument.Tensor t) ->
      Some t.TensorArgument.name
  | Argument.Optional_tensor (OptionalTensorArgument.None _) -> None
  | _ -> malformed "%s.%s is not an optional tensor" node.target name

let ints_arg ?(default = []) (node : Pytorch_types.Node.t) name =
  match
    List.find_opt (fun (a : NamedArgument.t) -> a.name = name) node.Node.inputs
  with
  | None -> default
  | Some { arg = Argument.Ints xs; _ } -> xs
  | Some { arg = Argument.None _; _ } -> default
  | Some _ -> malformed "%s.%s is not an int list" node.target name

let int_arg ?(default = 0) (node : Pytorch_types.Node.t) name =
  match
    List.find_opt (fun (a : NamedArgument.t) -> a.name = name) node.Node.inputs
  with
  | None -> default
  | Some { arg = Argument.Int i; _ } -> i
  | Some _ -> malformed "%s.%s is not an int" node.target name

let bool_arg ?(default = false) (node : Pytorch_types.Node.t) name =
  match
    List.find_opt (fun (a : NamedArgument.t) -> a.name = name) node.Node.inputs
  with
  | None -> default
  | Some { arg = Argument.Bool b; _ } -> b
  | Some _ -> malformed "%s.%s is not a bool" node.target name

let float_arg ?(default = 0.) (node : Pytorch_types.Node.t) name =
  match
    List.find_opt (fun (a : NamedArgument.t) -> a.name = name) node.Node.inputs
  with
  | None -> default
  | Some { arg = Argument.Float f; _ } -> f
  | Some _ -> malformed "%s.%s is not a float" node.target name

let output_names (node : Pytorch_types.Node.t) =
  List.map
    (function
      | Argument.Tensor t -> t.TensorArgument.name
      | _ -> malformed "%s has a non-tensor output" node.target)
    node.outputs

let is_nontrivial_node (node : Pytorch_types.Node.t) =
  match node.target with
  | "torch.ops.aten.convolution.default"
  | "torch.ops.aten._native_batch_norm_legit_no_training.default"
  | "torch.ops.aten.max_pool2d_with_indices.default"
  | "torch.ops.aten.addmm.default" ->
      true
  | _ -> false

let materialized_output_names (node : Pytorch_types.Node.t) =
  match node.target with
  | "torch.ops.aten._native_batch_norm_legit_no_training.default"
  | "torch.ops.aten.max_pool2d_with_indices.default" ->
      [ List.hd (output_names node) ]
  | _ -> output_names node

let hw2 name = function
  | [ h; w ] -> (h, w)
  | [ x ] -> (x, x)
  | xs ->
      malformed "%s must have one or two values, got %d" name (List.length xs)

let env_find env name =
  match String_map.find_opt name env with
  | Some x -> x
  | None -> malformed "SSA tensor %S is not defined" name

let add_env env names ids =
  if List.compare_lengths names ids <> 0 then
    malformed "internal output arity mismatch";
  List.fold_left2 (fun e name id -> String_map.add name id e) env names ids

let perm_nchw_to_nhwc =
  let open Axis in
  [ (N, N); (T, T); (D, D); (H, W); (W, C); (C, H) ]

let perm_nhwc_to_nchw =
  let open Axis in
  [ (N, N); (T, T); (D, D); (H, C); (W, H); (C, W) ]

let perm_oihw_to_conv_weight =
  let open Axis in
  [ (N, D); (T, T); (D, N); (H, W); (W, C); (C, H) ]

let perm_addmm_weight =
  let open Axis in
  [ (N, C); (T, T); (D, D); (H, H); (W, N); (C, W) ]

let conv_params (graph : Pytorch_types.Graph.t) (node : Pytorch_types.Node.t) =
  let weight_name = tensor_name node "weight" in
  let weight_meta =
    match String_map.find_opt weight_name graph.tensor_values with
    | Some x -> x
    | None -> malformed "no metadata for convolution weight %S" weight_name
  in
  let sizes =
    List.map
      (function
        | SymInt.Int i -> i
        | SymInt.Expr _ -> malformed "symbolic convolution weight")
      weight_meta.TensorMeta.sizes
  in
  let cout, cin, kh, kw =
    match sizes with
    | [ a; b; c; d ] -> (a, b, c, d)
    | _ -> malformed "convolution weight is not rank four"
  in
  let _ = cout in
  let sh, sw = hw2 "stride" (ints_arg ~default:[ 1; 1 ] node "stride") in
  let ph, pw = hw2 "padding" (ints_arg ~default:[ 0; 0 ] node "padding") in
  let dh, dw = hw2 "dilation" (ints_arg ~default:[ 1; 1 ] node "dilation") in
  let groups = int_arg ~default:1 node "groups" in
  ( {
      Conv.Convolution.stride =
        { h = Op_config.Pos.of_int sh; w = Op_config.Pos.of_int sw };
      padding =
        { h = Op_config.Nonneg.of_int ph; w = Op_config.Nonneg.of_int pw };
      dilation = { h = Op_config.Pos.of_int dh; w = Op_config.Pos.of_int dw };
      transposed = bool_arg node "transposed";
      output_padding =
        { h = Op_config.Nonneg.of_int 0; w = Op_config.Nonneg.of_int 0 };
      groups = Op_config.Pos.of_int groups;
    },
    cin,
    kh,
    kw )

let pool_params (node : Pytorch_types.Node.t) =
  let kh, kw = hw2 "kernel_size" (ints_arg node "kernel_size") in
  let stride = ints_arg ~default:[ kh; kw ] node "stride" in
  let sh, sw = hw2 "stride" stride in
  let ph, pw = hw2 "padding" (ints_arg ~default:[ 0; 0 ] node "padding") in
  {
    Pool.MaxPool2d.kernel = { h = Dim.extent kh; w = Dim.extent kw };
    stride = { h = Op_config.Pos.of_int sh; w = Op_config.Pos.of_int sw };
    pad = { h = Op_config.Nonneg.of_int ph; w = Op_config.Nonneg.of_int pw };
  }

let axes_for_rank rank dims =
  let used = List.filteri (fun i _ -> i >= 6 - rank) Axis.all in
  List.map
    (fun d ->
      let d = if d < 0 then d + rank else d in
      if d < 0 || d >= rank then malformed "invalid dimension %d" d
      else List.nth used d)
    dims

let native_perm ~rank dims =
  let used = List.filteri (fun i _ -> i >= 6 - rank) Axis.all in
  let outer = List.filter (fun a -> not (List.mem a used)) Axis.all in
  List.map (fun a -> (a, a)) outer
  @ List.mapi
      (fun i d ->
        let d = if d < 0 then d + rank else d in
        if d < 0 || d >= rank then malformed "invalid dimension %d" d;
        (List.nth used i, List.nth used d))
      dims

let shape_numel shape = (Vec6.numel shape :> int)

let resolve_view shape size =
  let numel = shape_numel shape in
  let known = List.fold_left (fun n x -> if x = -1 then n else n * x) 1 size in
  let size = List.map (fun x -> if x = -1 then numel / known else x) size in
  shape_of_sizes "view" (List.map (fun x -> SymInt.Int x) size)

let lower program =
  try
    let graph : Pytorch_types.Graph.t =
      program.ExportedProgram.graph_module.graph
    in
    let sign = program.ExportedProgram.graph_module.GraphModule.signature in
    let source_specs =
      List.map
        (function
          | InputSpec.User_input { UserInputSpec.arg = Argument.Tensor a } ->
              (`Input, a.TensorArgument.name, None)
          | InputSpec.Parameter p ->
              (`Constant, p.arg.TensorArgument.name, Some p.parameter_name)
          | InputSpec.Buffer p ->
              (`Constant, p.arg.TensorArgument.name, Some p.buffer_name)
          | InputSpec.Tensor_constant p ->
              (`Constant, p.arg.TensorArgument.name, Some p.tensor_constant_name)
          | _ -> raise (Lower_error (`Unsupported_input "non-tensor input")))
        sign.GraphSignature.input_specs
    in
    let tensor_origins = ref Tensor_id.Map.empty in
    let captured_targets = ref Tensor_id.Map.empty in
    let node_outputs = ref [] in
    let body =
      let open Graph_builder in
      let* env =
        List.fold_left
          (fun acc (kind, name, target) ->
            let* env = acc in
            let shape = tensor_shape graph name in
            let* id =
              match kind with
              | `Input -> input ~shape ()
              | `Constant -> constant ~shape ()
            in
            tensor_origins :=
              Tensor_id.Map.add id
                (Pt2_native_graph.Source
                   {
                     graph_path = [];
                     ssa_name = name;
                     meta = String_map.find_opt name graph.tensor_values;
                   })
                !tensor_origins;
            Option.iter
              (fun x ->
                captured_targets := Tensor_id.Map.add id x !captured_targets)
              target;
            return (String_map.add name id env))
          (return String_map.empty) source_specs
      in
      let lower_op env node =
        let get name = env_find env (tensor_name node name) in
        match node.target with
        | "torch.ops.aten.convolution.default" ->
            let params, _, _, _ = conv_params graph node in
            let* x = permute perm_nchw_to_nhwc (get "input") in
            let* w = permute perm_oihw_to_conv_weight (get "weight") in
            let bias =
              Option.map (env_find env) (optional_tensor_name node "bias")
            in
            let* y = convolution params ~x ~weight:w ?bias () in
            let* y = permute perm_nhwc_to_nchw y in
            return [ y ]
        | "torch.ops.aten._native_batch_norm_legit_no_training.default" ->
            let* x = permute perm_nchw_to_nhwc (get "input") in
            let params =
              { Norm.BatchNorm.channel = Axis.C; eps = float_arg node "eps" }
            in
            let* y =
              batch_norm params ~x
                ?weight:
                  (Option.map (env_find env)
                     (optional_tensor_name node "weight"))
                ?bias:
                  (Option.map (env_find env) (optional_tensor_name node "bias"))
                ~running_mean:(get "running_mean")
                ~running_var:(get "running_var") ()
            in
            let* y = permute perm_nhwc_to_nchw y in
            return [ y ]
        | "torch.ops.aten.relu.default" ->
            let* y = relu (get "self") in
            return [ y ]
        | "torch.ops.aten.add.Tensor" ->
            let* y = add (get "self") (get "other") in
            return [ y ]
        | "torch.ops.aten.max_pool2d_with_indices.default" ->
            let* x = permute perm_nchw_to_nhwc (get "self") in
            let* values, indices =
              max_pool2d_with_indices (pool_params node) x
            in
            let* () = discard indices in
            let* values = permute perm_nhwc_to_nchw values in
            return [ values ]
        | "torch.ops.aten.mean.dim" ->
            let x_name = tensor_name node "self" in
            let rank =
              match String_map.find_opt x_name graph.tensor_values with
              | Some m -> List.length m.TensorMeta.sizes
              | None -> malformed "no metadata for mean input"
            in
            let params =
              {
                Reduce.Mean.dims = axes_for_rank rank (ints_arg node "dim");
                keepdim = bool_arg node "keepdim";
              }
            in
            let* y = mean params (get "self") in
            return [ y ]
        | "torch.ops.aten.permute.default" ->
            let x_name = tensor_name node "self" in
            let rank =
              match String_map.find_opt x_name graph.tensor_values with
              | Some m -> List.length m.TensorMeta.sizes
              | None -> malformed "no metadata for permute input"
            in
            let* y =
              permute (native_perm ~rank (ints_arg node "dims")) (get "self")
            in
            return [ y ]
        | "torch.ops.aten.view.default" ->
            let shape = tensor_shape graph (tensor_name node "self") in
            let* y =
              reshape
                {
                  Reshape.Reshape.shape =
                    resolve_view shape (ints_arg node "size");
                }
                (get "self")
            in
            return [ y ]
        | "torch.ops.aten.addmm.default" ->
            let w_name = tensor_name node "mat2" in
            let in_features =
              match String_map.find_opt w_name graph.tensor_values with
              | Some m -> (
                  match m.TensorMeta.sizes with
                  | SymInt.Int n :: _ -> n
                  | _ -> malformed "invalid addmm weight")
              | None -> malformed "no metadata for addmm weight"
            in
            let* w = permute perm_addmm_weight (get "mat2") in
            let* y =
              linear
                { Linear.Linear.in_features = Dim.extent in_features }
                ~x:(get "mat1") ~weight:w ~bias:(get "self") ()
            in
            return [ y ]
        | target -> raise (Lower_error (`Unsupported_operator target))
      in
      let lower_node index env node =
        let names = materialized_output_names node in
        let bind ids =
          List.iter2
            (fun name id ->
              tensor_origins :=
                Tensor_id.Map.add id
                  (Pt2_native_graph.Source
                     {
                       graph_path = [];
                       ssa_name = name;
                       meta = String_map.find_opt name graph.tensor_values;
                     })
                  !tensor_origins)
            names ids;
          node_outputs := (index, ids) :: !node_outputs;
          add_env env names ids
        in
        if is_nontrivial_node node then
          let* ids = group ~label:node.target (lower_op env node) in
          return (bind ids)
        else
          let* ids = lower_op env node in
          return (bind ids)
      in
      let* env =
        List.fold_left
          (fun acc (index, node) ->
            let* env = acc in
            lower_node index env node)
          (return env)
          (List.mapi (fun i n -> (i, n)) graph.nodes)
      in
      let outputs =
        List.map
          (function
            | Argument.Tensor a -> env_find env a.TensorArgument.name
            | _ -> malformed "non-tensor graph output")
          graph.outputs
      in
      return outputs
    in
    match Graph_builder.build ~name:"pt2" ~outputs:Fun.id body with
    | Error e -> Core.fail (`Build e.Core.Error.kind)
    | Ok native_graph ->
        let node_origins =
          List.fold_left
            (fun acc (source_index, ids) ->
              let origin_node = List.nth graph.nodes source_index in
              List.fold_left
                (fun acc (node : Graph_ir.Node.t) ->
                  if
                    List.exists
                      (fun id -> List.mem id node.Graph_ir.Node.outputs)
                      ids
                  then
                    Node_id.Map.add node.Graph_ir.Node.id
                      [
                        {
                          Pt2_native_graph.Node_origin.graph_path = [];
                          index = source_index;
                          target = origin_node.target;
                          name = origin_node.name;
                          metadata = origin_node.metadata;
                        };
                      ]
                      acc
                  else acc)
                acc native_graph.Graph_ir.Graph.nodes)
            Node_id.Map.empty !node_outputs
        in
        Pt2_native_graph.make ~graph:native_graph
          ~tensor_origins:!tensor_origins ~node_origins
          ~captured_targets:!captured_targets
        |> Core.map_error (fun e -> `Provenance e)
  with Lower_error e -> Core.fail e

let lower_archive archive = lower (Pt2_archive.program archive)

let tensor_of_pt2 (tensor : Pt2_tensor.t) =
  match tensor.dtype with
  | Pt2_dtype.Float32 -> (
      let shape =
        try
          shape_of_sizes "tensor"
            (List.map (fun x -> SymInt.Int x) tensor.sizes)
        with Lower_error (`Malformed_graph s) ->
          raise (Lower_error (`Tensor_bridge s))
      in
      if List.compare_lengths tensor.sizes tensor.strides <> 0 then
        Core.fail (`Tensor_bridge "sizes and strides have different ranks")
      else
        let storage_index coord =
          List.fold_left ( + ) tensor.storage_offset
            (List.mapi
               (fun i stride ->
                 let axis =
                   List.nth Axis.all (6 - List.length tensor.sizes + i)
                 in
                 Dim.to_int (Vec6.get coord axis) * stride)
               tensor.strides)
        in
        let data_len = Bytes.length tensor.data in
        try
          Core.return
            (Tensor.materialize shape (fun coord ->
                 let offset = storage_index coord * 4 in
                 if offset < 0 || offset + 4 > data_len then
                   raise (Invalid_argument "storage offset is outside data");
                 Int32.float_of_bits (Bytes.get_int32_le tensor.data offset)))
        with Invalid_argument s -> Core.fail (`Tensor_bridge s))
  | dtype ->
      Core.fail
        (`Tensor_bridge
           (Format.asprintf "only float32 is supported, got %s"
              (Pt2_dtype.to_string dtype)))

let run ?hooks archive ~input =
  let open Core.Syntax in
  let* lowered = lower_archive archive in
  let graph = lowered.Pt2_native_graph.graph in
  let eval_hooks =
    match hooks with
    | None -> None
    | Some (Hooks h) ->
        Some
          (Eval_direct.Hooks
             {
               on_start = (fun node -> h.on_start lowered node);
               on_end = (fun node state -> h.on_end lowered node state);
             })
  in
  let* input = tensor_of_pt2 input in
  let user_ids =
    List.filter
      (fun id -> Graph_ir.input_kind graph id = Graph_ir.Input.Input)
      graph.Graph_ir.Graph.inputs
  in
  let* inputs =
    match user_ids with
    | [ id ] -> Core.return [ (id, input) ]
    | ids ->
        Core.fail
          (`Unsupported_input
             (Format.asprintf "expected one user input, got %d"
                (List.length ids)))
  in
  let used_constants =
    List.concat_map
      (fun node -> Graph_ir.operands node.Graph_ir.Node.op)
      graph.Graph_ir.Graph.nodes
  in
  let* constants =
    Core.List.map
      (fun (id, target) ->
        let* raw =
          Pt2_archive.load_captured_tensor archive target
          |> Core.map_error (fun e ->
              `Tensor_bridge (Format.asprintf "%a" Pt2_archive.pp_error e))
        in
        let+ tensor = tensor_of_pt2 raw in
        (id, tensor))
      (List.filter
         (fun (id, _) -> List.mem id used_constants)
         (Tensor_id.Map.bindings lowered.captured_targets))
  in
  let* env =
    Eval_direct.run ?hooks:eval_hooks ~constants graph ~inputs
    |> Core.map_error (fun e -> `Eval e)
  in
  Core.List.map
    (fun id ->
      match Tensor_id.Map.find_opt id env with
      | Some tensor -> Core.return tensor
      | None -> Core.fail (`Malformed_graph "native output was not evaluated"))
    graph.Graph_ir.Graph.outputs
