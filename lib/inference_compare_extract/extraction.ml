open Inference_compare_schema
module Plan = Inference_compare_plan
module GI = Plan.Graph_index
module H = Inference_compare_codecs.Payload_hasher

type t = {
  unit : Id.Unit.t;
  side : Side.t;
  body : Graph_desc.t;
  descriptor : Extracted_graph.t;
  inputs : Tensor_ref.t list;
  outputs : Tensor_ref.t list;
  attributes : (Node_ref.t * string) list;
}

let ( let* ) = Err.bind
let fail = Extract_error.fail

let text_error ~side ~unit (e : Text_error.t Err.Error.t) =
  ignore e;
  Extract_error.make ~side ~unit Boundary_mismatch

let identifier make s ~side ~unit =
  match make s with Ok v -> Ok v | Error e -> fail (text_error ~side ~unit e)

let body_preimage (t : t) =
  let w = Wire.canonical_string in
  Digest_preimage.of_json Extracted_graph_payload
    (Wire.obj
       [
         Wire.field "body"
           (fun g -> Jsont.Json.string (Graph_desc.preimage g))
           t.body;
         Wire.field "nodes"
           (Wire.enc_list (Wire.enc_pair Node_ref.encode Node_ref.encode))
           t.descriptor.original_to_extracted_nodes;
         Wire.field "tensors"
           (Wire.enc_list (Wire.enc_pair Tensor_ref.encode Tensor_ref.encode))
           t.descriptor.original_to_extracted_tensors;
         Wire.field "attributes"
           (Wire.enc_list (Wire.enc_pair Node_ref.encode Wire.enc_string))
           t.attributes;
       ])
  |> fun s ->
  ignore w;
  s

let digest_hex ~hashers ~algorithm ~side ~unit preimage =
  match H.digest_string hashers algorithm preimage with
  | Ok d -> Ok (Digest.Payload.hex d)
  | Error e ->
      fail
        (Extract_error.make ~side ~unit
           ~detail:(Digest (Err.Error.kind e))
           Evidence_failed)

let plan ?(node_attributes = fun (_ : Node_ref.t) -> "") ~hashers ~algorithm idx
    (b : Plan.Partition_check.Boundary.t) =
  let side = b.side and unit = b.unit in
  let src = (GI.graph idx).key in
  let* component =
    identifier Name.Scope_component.of_string (Id.Unit.to_string unit) ~side
      ~unit
  in
  let* version =
    identifier Id.Graph_version.of_string
      (Printf.sprintf "%s#%s:%s"
         (Id.Graph_version.to_string src.version)
         (Id.Unit.to_string unit) (Side.to_string side))
      ~side ~unit
  in
  let ext_key : Graph_key.t =
    { dialect = src.dialect; version; graph_digest = src.graph_digest }
  in
  let tref (r : Tensor_ref.t) : Tensor_ref.t =
    { graph = ext_key; scope = component :: r.scope; local_id = r.local_id }
  in
  let nref (r : Node_ref.t) : Node_ref.t =
    { graph = ext_key; scope = component :: r.scope; local_id = r.local_id }
  in
  let members = Node_ref.Set.of_list b.nodes in
  let rec nodes acc = function
    | [] -> Ok (List.rev acc)
    | n :: rest -> (
        match GI.node idx n with
        | None -> fail (Extract_error.make ~side ~unit ~node:n Unknown_member)
        | Some (d : Node_desc.t) ->
            nodes
              ({
                 d with
                 Node_desc.node = nref d.node;
                 operands = List.map tref d.operands;
                 results = List.map tref d.results;
                 control_predecessors =
                   List.filter_map
                     (fun p ->
                       if Node_ref.Set.mem p members then Some (nref p)
                       else None)
                     d.control_predecessors;
               }
              :: acc)
              rest)
  in
  let* body_nodes = nodes [] b.nodes in
  let spec_of t =
    match GI.spec idx t with
    | Some s -> Ok s
    | None -> fail (Extract_error.make ~side ~unit ~tensor:t Unknown_member)
  in
  let rec sources acc = function
    | [] -> Ok (List.rev acc)
    | t :: rest ->
        let* spec = spec_of t in
        let* name =
          identifier Name.User_input.of_string
            (Id.Tensor_local.to_string t.Tensor_ref.local_id)
            ~side ~unit
        in
        sources
          ({ Graph_source.tensor = tref t; role = User_input name; spec } :: acc)
          rest
  in
  let* body_sources = sources [] b.inputs in
  let produced =
    List.concat_map (fun (n : Node_desc.t) -> n.results) body_nodes
  in
  let* result_specs =
    let originals =
      List.concat_map
        (fun n -> match GI.node idx n with Some d -> d.results | None -> [])
        b.nodes
    in
    let rec go acc = function
      | [] -> Ok (List.rev acc)
      | (orig, ext) :: rest ->
          let* spec = spec_of orig in
          go ((ext, spec) :: acc) rest
    in
    go [] (List.combine originals produced)
  in
  let outputs = List.map tref b.outputs in
  let unsealed : Graph_desc.t =
    {
      key = ext_key;
      nodes = body_nodes;
      sources = body_sources;
      final_outputs = outputs;
      output_tree =
        Tuple (List.map (fun t -> Graph_desc.Output_tree.Tensor t) outputs);
      tensor_specs = result_specs;
    }
  in
  let* graph_hex =
    digest_hex ~hashers ~algorithm ~side ~unit (Graph_desc.preimage unsealed)
  in
  let* graph_digest =
    match Digest.Graph.make algorithm ~hex:graph_hex with
    | Ok d -> Ok d
    | Error _ -> fail (Extract_error.make ~side ~unit Evidence_failed)
  in
  let extracted_key = { ext_key with graph_digest } in
  let retag_tensor (r : Tensor_ref.t) =
    { r with Tensor_ref.graph = extracted_key }
  in
  let retag_node (r : Node_ref.t) = { r with Node_ref.graph = extracted_key } in
  let body : Graph_desc.t =
    {
      key = extracted_key;
      nodes =
        List.map
          (fun (n : Node_desc.t) ->
            {
              n with
              Node_desc.node = retag_node n.node;
              operands = List.map retag_tensor n.operands;
              results = List.map retag_tensor n.results;
              control_predecessors = List.map retag_node n.control_predecessors;
            })
          unsealed.nodes;
      sources =
        List.map
          (fun (s : Graph_source.t) ->
            { s with Graph_source.tensor = retag_tensor s.tensor })
          unsealed.sources;
      final_outputs = List.map retag_tensor unsealed.final_outputs;
      output_tree =
        Tuple
          (List.map
             (fun t -> Graph_desc.Output_tree.Tensor (retag_tensor t))
             unsealed.final_outputs);
      tensor_specs =
        List.map (fun (t, s) -> (retag_tensor t, s)) unsealed.tensor_specs;
    }
  in
  let node_map =
    List.map2 (fun orig (n : Node_desc.t) -> (orig, n.node)) b.nodes body.nodes
  in
  let all_originals =
    b.inputs
    @ List.concat_map
        (fun n -> match GI.node idx n with Some d -> d.results | None -> [])
        b.nodes
  in
  let tensor_map =
    List.map2
      (fun orig ext -> (orig, ext))
      all_originals
      (List.map (fun (s : Graph_source.t) -> s.tensor) body.sources
      @ List.concat_map (fun (n : Node_desc.t) -> n.results) body.nodes)
  in
  let inputs = List.map (fun (s : Graph_source.t) -> s.tensor) body.sources in
  let draft : t =
    {
      unit;
      side;
      body;
      descriptor =
        {
          source_graph = src;
          extracted_key;
          unit;
          inputs;
          outputs = body.final_outputs;
          original_to_extracted_nodes = node_map;
          original_to_extracted_tensors = tensor_map;
          graph_payload_digest =
            (match
               Digest.Extracted_graph_payload.make algorithm
                 ~hex:(String.make 64 '0')
             with
            | Ok d -> d
            | Error _ -> invalid_arg "Extraction.plan");
        };
      inputs;
      outputs = body.final_outputs;
      attributes =
        List.map2
          (fun orig (n : Node_desc.t) -> (n.node, node_attributes orig))
          b.nodes body.nodes;
    }
  in
  let* payload_hex =
    digest_hex ~hashers ~algorithm ~side ~unit (body_preimage draft)
  in
  let* payload =
    match Digest.Extracted_graph_payload.make algorithm ~hex:payload_hex with
    | Ok d -> Ok d
    | Error _ -> fail (Extract_error.make ~side ~unit Evidence_failed)
  in
  let result =
    {
      draft with
      descriptor = { draft.descriptor with graph_payload_digest = payload };
    }
  in
  (* The extracted body is itself an ordinary graph: the planner's validator
     must accept it. *)
  let limits = Body_limits.v in
  match GI.build ~limits ~side body with
  | Ok _ -> Ok result
  | Error e ->
      fail
        (Extract_error.make ~side ~unit
           ~detail:(Invalid_body (Err.Error.kind e))
           Boundary_mismatch)

let extract_all ?(node_attributes = fun (_ : Side.t) (_ : Node_ref.t) -> "")
    ~hashers ~algorithm structural =
  let order = Plan.Unit_order.order (Plan.Structural.order structural) in
  let rec go acc = function
    | [] -> Ok (List.rev acc)
    | u :: rest ->
        let rec sides acc = function
          | [] -> Ok acc
          | side :: more -> (
              let idx =
                match (side : Side.t) with
                | Left -> Plan.Structural.left structural
                | Right -> Plan.Structural.right structural
              in
              match
                Plan.Partition_check.boundary
                  (Plan.Structural.partition structural)
                  u side
              with
              | None -> fail (Extract_error.make ~side ~unit:u Unknown_member)
              | Some b ->
                  let* e =
                    plan ~node_attributes:(node_attributes side) ~hashers
                      ~algorithm idx b
                  in
                  sides (e :: acc) more)
        in
        let* acc = sides acc [ Side.Left; Side.Right ] in
        go acc rest
  in
  go [] order
