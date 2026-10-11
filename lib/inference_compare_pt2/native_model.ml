(* The lowered Native graph as the Native side sees it: an ordinary graph of
   tensor dataflow (tensors are numbered edges, nodes are numbered), with the
   complete record of which Native nodes each exporter node produced. The
   description carries operators and dataflow; each op's parameters are
   folded into the extraction digest through {!node_attributes}. *)

open Inference_compare_schema
module Tid = Graph_ir.Tensor_id
module Nid = Graph_ir.Node_id

type t = {
  lowered : Pt2_native_graph.t;
  capture : Native_interp.Capture.t;
  graph : Graph_ir.graph;
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
  let shape = List.map (fun a -> Int64.of_int (Vec6.get s.shape a :> int)) Axis.all in
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

let sig_of (g : Graph_ir.graph) id =
  match Tid.Map.find_opt id g.Graph_ir.Graph.tensors with
  | Some s -> Ok s
  | None -> Error "tensor without signature"

(* The description of any set of nodes of [g]; [inputs] become sources. In the
   full graph the roles follow the graph's own classification; in an extracted
   body every source is a coordinator-supplied input named by its id. *)
let describe ~key ~scope ~(g : Graph_ir.graph) ~(nodes : Graph_ir.node list)
    ~(inputs : Tid.t list) ~(outputs : Tid.t list) ~(roles : [ `Graph | `Body ]) :
    (Graph_desc.t, string) result =
  let* dialect = text (Id.Dialect.of_string "native") in
  let* node_descs =
    all
      (fun (n : Graph_ir.node) ->
        let* node = node_ref ~key ~scope n.id in
        let* operation = text (Id.Operator.of_string (Graph_ir.op_name n.op)) in
        let* operands = all (tensor_ref ~key ~scope) (Graph_ir.operands n.op) in
        let* results = all (tensor_ref ~key ~scope) n.outputs in
        Ok
          {
            Node_desc.node;
            operator = { dialect; operation };
            operands;
            results;
            control_predecessors = [];
            effect = Pure;
          })
      nodes
  in
  let* sources =
    all
      (fun id ->
        let* tensor = tensor_ref ~key ~scope id in
        let* s = sig_of g id in
        let* spec = spec_of_sig s in
        let name = tensor_local id in
        let* role =
          match (roles, Graph_ir.input_kind g id) with
          | `Graph, Graph_ir.Input.Constant ->
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
      (fun (n : Graph_ir.node) ->
        all
          (fun id ->
            let* r = tensor_ref ~key ~scope id in
            let* s = sig_of g id in
            let* spec = spec_of_sig s in
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
      output_tree = Tuple (List.map (fun t -> Graph_desc.Output_tree.Tensor t) final_outputs);
      tensor_specs;
    }

let of_lowered ~hashers ~algorithm (lowered : Pt2_native_graph.t) capture : (t, string) result =
  let g = lowered.Pt2_native_graph.graph in
  let* graph_text =
    Jsont_bytesrw.encode_string Graph_ir.graph_jsont g
  in
  let* hex =
    match
      Inference_compare_codecs.Payload_hasher.digest_string hashers algorithm graph_text
    with
    | Ok d -> Ok (Digest.Payload.hex d)
    | Error _ -> Error "digest backend unavailable"
  in
  let* graph_digest =
    Digest.Graph.make algorithm ~hex |> Result.map_error (fun _ -> "digest")
  in
  let* dialect = text (Id.Dialect.of_string "native") in
  let* version = text (Id.Graph_version.of_string "pt2-lowered") in
  let key = { Graph_key.dialect; version; graph_digest } in
  let* desc =
    describe ~key ~scope:[] ~g ~nodes:g.Graph_ir.Graph.nodes ~inputs:g.inputs
      ~outputs:g.outputs ~roles:`Graph
  in
  Ok { lowered; capture; graph = g; key; desc }

(* The text standing for one node in a digest: its operation and every
   parameter, which the dataflow description does not carry. *)
let node_attributes (n : Graph_ir.node) =
  match Jsont_bytesrw.encode_string Graph_ir.op_jsont n.op with Ok s -> s | Error m -> m
