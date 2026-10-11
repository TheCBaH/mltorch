(* Native4D dialect tooling: realize a planned extraction as a sub-graph of the
   four-axis graph (the same op values, the same edge ids), with every external
   operand a coordinator-supplied input. The twin of {!Native_dialect} over the
   four-axis op. *)

open Inference_compare_schema
module Ex = Inference_compare_extract
module Tid = Graph_ir.Tensor_id
module Nid = Graph_ir.Node_id
module G = Graph_model

type graph = Native4d.Op.t G.t
type body = { graph : Native4d.Graph.graph; key : Graph_key.t; scope : Scope.t }

let dialect =
  match Id.Dialect.of_string "native4d" with
  | Ok d -> d
  | Error _ -> invalid_arg "Native4d_dialect"

let describe (m : graph) = m.G.desc

let id_of prefix local =
  let s = Id.Tensor_local.to_string local in
  if String.length s > 1 && s.[0] = prefix then
    int_of_string_opt (String.sub s 1 (String.length s - 1))
  else None

let node_id_of (r : Node_ref.t) =
  let s = Id.Node_local.to_string r.local_id in
  if String.length s > 1 && s.[0] = 'n' then
    Option.map Nid.of_int
      (int_of_string_opt (String.sub s 1 (String.length s - 1)))
  else None

let tensor_id_of (r : Tensor_ref.t) =
  Option.map Tid.of_int (id_of 't' r.local_id)

let extract (m : graph) (plan : Ex.Extraction.t) :
    (body, Ex.Extract_error.t) Err.t =
  let fail code =
    Ex.Extract_error.fail
      (Ex.Extract_error.make ~side:plan.side ~unit:plan.unit code)
  in
  let full = m.G.graph in
  let find_node id =
    List.find_opt
      (fun (n : Native4d.Graph.node) -> Nid.equal n.id id)
      full.nodes
  in
  let originals =
    List.map
      (fun (n : Node_desc.t) ->
        List.find_map
          (fun ((o : Node_ref.t), (e : Node_ref.t)) ->
            if Node_ref.equal e n.node then node_id_of o else None)
          plan.descriptor.original_to_extracted_nodes)
      plan.body.nodes
  in
  if List.exists Option.is_none originals then fail Failure.Unknown_member
  else
    let nodes =
      List.filter_map (fun id -> Option.bind id find_node) originals
    in
    let inputs = List.filter_map tensor_id_of plan.inputs in
    let outputs = List.filter_map tensor_id_of plan.outputs in
    if
      List.length nodes <> List.length originals
      || List.length inputs <> List.length plan.inputs
      || List.length outputs <> List.length plan.outputs
    then fail Failure.Unknown_member
    else
      let used =
        Tid.Set.of_list
          (inputs @ outputs
          @ List.concat_map
              (fun (n : Native4d.Graph.node) ->
                n.outputs @ Native4d.Op.operands n.op)
              nodes)
      in
      let tensors =
        Tid.Map.filter (fun id _ -> Tid.Set.mem id used) full.tensors
      in
      let graph : Native4d.Graph.graph =
        {
          nodes;
          root =
            {
              Graph_common.Group.id = Graph_common.Group_id.of_int 0;
              label = None;
              items =
                List.map
                  (fun (n : Native4d.Graph.node) ->
                    Graph_common.Group.Node n.id)
                  nodes;
            };
          tensors;
          inputs;
          input_kinds = Tid.Map.empty;
          outputs;
        }
      in
      let scope =
        match plan.body.nodes with
        | n :: _ -> n.node.scope
        | [] -> ( match plan.inputs with t :: _ -> t.scope | [] -> [])
      in
      Ok { graph; key = plan.body.key; scope }

let describe_body (b : body) : Graph_desc.t =
  match
    G.describe ~ops:G.native4d_ops ~key:b.key ~scope:b.scope ~g:b.graph
      ~nodes:b.graph.nodes ~inputs:b.graph.inputs ~outputs:b.graph.outputs
      ~roles:`Body
  with
  | Ok d -> d
  | Error m -> invalid_arg ("Native4d_dialect.describe_body: " ^ m)

let validate (_ : body) = Ok ()

module Dialect : Ex.Dialect.S with type graph = graph and type body = body =
struct
  type nonrec graph = graph
  type nonrec body = body

  let dialect = dialect
  let describe = describe
  let extract = extract
  let describe_body = describe_body
  let validate = validate
end
