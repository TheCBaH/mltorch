(* A Native-family graph (any dialect over the shared graph records) as an
   ordinary graph of tensor dataflow: tensors are numbered edges, nodes are
   numbered, and an operation's parameters travel as attribute text that the
   extraction digest folds in. The dialect-specific part is only how to read an
   operation's operands, its name and its attribute text. *)

open Inference_compare_schema
module Tid = Graph_ir.Tensor_id
module Nid = Graph_ir.Node_id

type 'op ops = {
  dialect : string;
  operands : 'op -> Tid.t list;
  name : 'op -> string;
  attributes : 'op -> string;
}

type 'op t = {
  graph : 'op Graph_common.Graph.t;
  ops : 'op ops;
  key : Graph_key.t;
  desc : Graph_desc.t;
}

let ( let* ) = Result.bind
let text r = Result.map_error (fun _ -> "invalid identifier") r

let rec all f = function
  | [] -> Ok []
  | x :: rest ->
      let* y = f x in
      let* ys = all f rest in
      Ok (y :: ys)

let tensor_local id = Printf.sprintf "t%d" (Tid.to_int id)
let node_local id = Printf.sprintf "n%d" (Nid.to_int id)

let tensor_ref ~key ~scope id : (Tensor_ref.t, string) result =
  let* local_id = text (Id.Tensor_local.of_string (tensor_local id)) in
  Ok { Tensor_ref.graph = key; scope; local_id }

let node_ref ~key ~scope id : (Node_ref.t, string) result =
  let* local_id = text (Id.Node_local.of_string (node_local id)) in
  Ok { Node_ref.graph = key; scope; local_id }

let spec_of_sig (s : Tensor_sig.t) : (Tensor_spec.t, string) result =
  let (Payload.Fmt f) = s.fmt in
  let shape =
    List.map (fun a -> Int64.of_int (Vec6.get s.shape a :> int)) Axis.all
  in
  match Id.Codec.of_string (Payload.fmt_name f) with
  | Error _ -> Error "unsupported element type"
  | Ok codec ->
      Ok
        {
          Tensor_spec.dtype = { codec; parameters_digest = None };
          logical_shape = shape;
          layout_id = Dtypes.layout;
          value_semantics = true;
        }

let sig_of (g : _ Graph_common.Graph.t) id =
  match Tid.Map.find_opt id g.Graph_common.Graph.tensors with
  | Some s -> Ok s
  | None -> Error "tensor without signature"

let spec_of g id =
  let* s = sig_of g id in
  spec_of_sig s

(* The description of any set of nodes of [g]; [inputs] become sources. In the
   full graph the roles follow the graph's own classification; in an extracted
   body every source is a coordinator-supplied input named by its id. *)
let describe ~(ops : 'op ops) ~key ~scope ~(g : 'op Graph_common.Graph.t)
    ~(nodes : 'op Graph_common.Node.t list) ~(inputs : Tid.t list)
    ~(outputs : Tid.t list) ~(roles : [ `Graph | `Body ]) :
    (Graph_desc.t, string) result =
  let* dialect = text (Id.Dialect.of_string ops.dialect) in
  let* node_descs =
    all
      (fun (n : 'op Graph_common.Node.t) ->
        let* node = node_ref ~key ~scope n.id in
        let* operation = text (Id.Operator.of_string (ops.name n.op)) in
        let* operands = all (tensor_ref ~key ~scope) (ops.operands n.op) in
        let* results = all (tensor_ref ~key ~scope) n.outputs in
        Ok
          {
            Node_desc.node;
            operator = { dialect; operation };
            operands;
            results;
            control_predecessors = [];
            side_effect = Pure;
          })
      nodes
  in
  let* sources =
    all
      (fun id ->
        let* tensor = tensor_ref ~key ~scope id in
        let* spec = spec_of g id in
        let name = tensor_local id in
        let* role =
          match (roles, Graph_common.input_kind g id) with
          | `Graph, Graph_common.Input.Constant ->
              let* n = text (Name.Constant.of_string name) in
              Ok (Input_role.Constant n)
          | _ ->
              let* n = text (Name.User_input.of_string name) in
              Ok (Input_role.User_input n)
        in
        Ok { Graph_source.tensor; role; spec })
      inputs
  in
  let* final_outputs = all (tensor_ref ~key ~scope) outputs in
  let* tensor_specs =
    all
      (fun (n : 'op Graph_common.Node.t) ->
        all
          (fun id ->
            let* r = tensor_ref ~key ~scope id in
            let* spec = spec_of g id in
            Ok (r, spec))
          n.outputs)
      nodes
    |> Result.map List.concat
  in
  Ok
    {
      Graph_desc.key;
      nodes = node_descs;
      sources;
      final_outputs;
      output_tree =
        Tuple
          (List.map (fun t -> Graph_desc.Output_tree.Tensor t) final_outputs);
      tensor_specs;
    }

(* A whole graph under a digest of [identity_text]. *)
let make ~(ops : 'op ops) ~hashers ~algorithm ~version ~identity_text
    (g : 'op Graph_common.Graph.t) : ('op t, string) result =
  let* hex =
    match
      Inference_compare_codecs.Payload_hasher.digest_string hashers algorithm
        identity_text
    with
    | Ok d -> Ok (Digest.Payload.hex d)
    | Error _ -> Error "digest backend unavailable"
  in
  let* graph_digest =
    Digest.Graph.make algorithm ~hex |> Result.map_error (fun _ -> "digest")
  in
  let* dialect = text (Id.Dialect.of_string ops.dialect) in
  let* version = text (Id.Graph_version.of_string version) in
  let key = { Graph_key.dialect; version; graph_digest } in
  let* desc =
    describe ~ops ~key ~scope:[] ~g ~nodes:g.nodes ~inputs:g.inputs
      ~outputs:g.outputs ~roles:`Graph
  in
  Ok { graph = g; ops; key; desc }

let json_text codec v =
  match Jsont_bytesrw.encode_string codec v with Ok s -> s | Error m -> m

let native_ops : Graph_ir.op ops =
  {
    dialect = "native";
    operands = Graph_ir.operands;
    name = Graph_ir.op_name;
    attributes = json_text Graph_ir.op_jsont;
  }

let native4d_ops : Native4d.Op.t ops =
  {
    dialect = "native4d";
    operands = Native4d.Op.operands;
    name = Native4d.Op.name;
    attributes = json_text Native4d.Op.jsont;
  }

(* Node attribute text by reference, for the extraction digest. *)
let attributes_of (m : 'op t) (r : Node_ref.t) =
  let s = Id.Node_local.to_string r.local_id in
  match int_of_string_opt (String.sub s 1 (String.length s - 1)) with
  | None -> ""
  | Some i -> (
      match
        List.find_opt
          (fun (n : 'op Graph_common.Node.t) -> Nid.to_int n.id = i)
          m.graph.nodes
      with
      | Some n -> m.ops.attributes n.op
      | None -> "")
