open Graph_ir

module Graph_path = struct
  type t = int list

  let root = []
  let child path node_index = path @ [ node_index ]

  let pp fmt path =
    match path with
    | [] -> Format.pp_print_string fmt "root"
    | _ ->
        Fmt.pf fmt "root/%a"
          (Fmt.list ~sep:(Fmt.any "/") Format.pp_print_int)
          path
end

module Tensor_origin = struct
  type t = {
    graph_path : Graph_path.t;
    ssa_name : string;
    meta : Pytorch_types.TensorMeta.t option;
  }
end

module Node_origin = struct
  type t = {
    graph_path : Graph_path.t;
    index : int;
    target : string;
    name : string option;
    metadata : string Schema_runtime.String_map.t;
  }
end

type tensor_origin = Source of Tensor_origin.t | Derived

type t = {
  graph : graph;
  tensor_origins : tensor_origin Tensor_id.Map.t;
  node_origins : Node_origin.t list Node_id.Map.t;
  captured_targets : string Tensor_id.Map.t;
}

type error =
  [ `Unknown_tensor_id of Tensor_id.t
  | `Unknown_node_id of Node_id.t
  | `Captured_target_for_non_constant of Tensor_id.t ]

let pp_error ppf : [< error ] -> unit = function
  | `Unknown_tensor_id id ->
      Format.fprintf ppf "PT2 provenance refers to unknown tensor %a"
        Tensor_id.pp id
  | `Unknown_node_id id ->
      Format.fprintf ppf "PT2 provenance refers to unknown node %a" Node_id.pp
        id
  | `Captured_target_for_non_constant id ->
      Format.fprintf ppf "captured target attached to non-constant tensor %a"
        Tensor_id.pp id

let graph_ids (g : graph) =
  let rec walk tensors nodes gr =
    let tensors =
      Tensor_id.Map.fold
        (fun id _ ids -> Tensor_id.Map.add id () ids)
        gr.Graph.tensors tensors
    in
    List.fold_left
      (fun (tensors, nodes) (node : node) ->
        let nodes = Node_id.Map.add node.Node.id () nodes in
        match node.Node.op with
        | Subgraph { graph; _ } -> walk tensors nodes graph
        | _ -> (tensors, nodes))
      (tensors, nodes) gr.Graph.nodes
  in
  walk Tensor_id.Map.empty Node_id.Map.empty g

let make ~graph ~tensor_origins ~node_origins ~captured_targets =
  let tensors, nodes = graph_ids graph in
  let open Core.Syntax in
  let* () =
    Core.List.iter
      (fun (id, _) ->
        if Tensor_id.Map.mem id tensors then Core.return ()
        else Core.fail (`Unknown_tensor_id id))
      (Tensor_id.Map.bindings tensor_origins)
  in
  let* () =
    Core.List.iter
      (fun (id, _) ->
        if Node_id.Map.mem id nodes then Core.return ()
        else Core.fail (`Unknown_node_id id))
      (Node_id.Map.bindings node_origins)
  in
  let* () =
    Core.List.iter
      (fun (id, _) ->
        if Graph_ir.input_kind graph id = Input.Constant then Core.return ()
        else Core.fail (`Captured_target_for_non_constant id))
      (Tensor_id.Map.bindings captured_targets)
  in
  Core.return { graph; tensor_origins; node_origins; captured_targets }
