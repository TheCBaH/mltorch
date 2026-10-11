module W = Wire

type t = {
  schema_version : Schema_version.t;
  left_graph : Graph_key.t;
  right_graph : Graph_key.t;
  units : Matched_unit.t list;
  tensor_relations : Tensor_relation.t list;
  dispositions : Member_disposition.t list;
  frontier : Frontier_mapping.t;
  derivation_digest : Digest.Derivation.t list;
}

let decode : t W.decoder =
  W.record (fun o ->
      let ( let* ) = W.( let* ) in
      let* schema_version =
        W.Obj.required o "schema_version" Schema_version.decode
      in
      let* left_graph = W.Obj.required o "left_graph" Graph_key.decode in
      let* right_graph = W.Obj.required o "right_graph" Graph_key.decode in
      let* units =
        W.Obj.required o "units"
          (W.unique
             ~key:(fun (u : Matched_unit.t) -> Id.Unit.to_string u.unit_id)
             (W.list Matched_unit.decode))
      in
      let* tensor_relations =
        W.Obj.required o "tensor_relations"
          (W.unique
             ~key:(fun (r : Tensor_relation.t) ->
               Id.Tensor_relation.to_string r.relation_id)
             (W.list Tensor_relation.decode))
      in
      let* dispositions =
        W.Obj.required o "dispositions" (W.list Member_disposition.decode)
      in
      let* frontier = W.Obj.required o "frontier" Frontier_mapping.decode in
      let* derivation_digest =
        W.Obj.required o "derivation_digest" (W.list Wire_digest.derivation)
      in
      Ok
        {
          schema_version;
          left_graph;
          right_graph;
          units;
          tensor_relations;
          dispositions;
          frontier;
          derivation_digest;
        })

let encode m =
  W.obj
    [
      W.field "schema_version" Schema_version.encode m.schema_version;
      W.field "left_graph" Graph_key.encode m.left_graph;
      W.field "right_graph" Graph_key.encode m.right_graph;
      W.field "units" (W.enc_list Matched_unit.encode) m.units;
      W.field "tensor_relations"
        (W.enc_list Tensor_relation.encode)
        m.tensor_relations;
      W.field "dispositions"
        (W.enc_list Member_disposition.encode)
        m.dispositions;
      W.field "frontier" Frontier_mapping.encode m.frontier;
      W.field "derivation_digest"
        (W.enc_list Wire_digest.encode_derivation)
        m.derivation_digest;
    ]

(* Canonical form. Units, relations, dispositions and the frontier's tables
   are sets keyed by their identifiers, so their order carries no meaning and
   is fixed here; ordered lists (a cluster's inputs and outputs, recipe
   arguments, route sources) keep the order the artifact declared, because
   that order is the operand and output occurrence order. *)

let sort_by key items =
  List.stable_sort (fun a b -> String.compare (key a) (key b)) items

let by_encoding encode = sort_by (fun v -> W.canonical_string (encode v))

let canonical_cluster (c : Cluster.t) =
  { c with Cluster.nodes = by_encoding Node_ref.encode c.nodes }

let canonical_recipe (r : Recipe.t) =
  {
    r with
    Recipe.nodes =
      sort_by
        (fun (n : Recipe.node) -> Id.Conversion_node.to_string n.id)
        r.nodes;
  }

let canonical_relation (r : Tensor_relation.t) =
  {
    r with
    Tensor_relation.left_members = by_encoding Tensor_ref.encode r.left_members;
    right_members = by_encoding Tensor_ref.encode r.right_members;
    member_pairs =
      sort_by
        (fun (p : Member_pair.t) -> Id.Member_pair.to_string p.pair_id)
        r.member_pairs;
    routes =
      sort_by
        (fun (v : Value_route.t) -> Id.Value_route.to_string v.route_id)
        r.routes;
  }

let canonical_frontier (f : Frontier_mapping.t) =
  {
    Frontier_mapping.canonical_values =
      sort_by
        (fun (c : Canonical_spec.t) -> Id.Canonical_value.to_string c.key)
        f.canonical_values;
    (* Occurrence order is the declared final output order: kept. *)
    final_outputs = f.final_outputs;
    initial_values =
      sort_by
        (fun (b : Initial_binding.t) -> Id.Canonical_value.to_string b.key)
        f.initial_values;
    input_bindings =
      by_encoding Input_binding.encode
        (List.map
           (fun (b : Input_binding.t) ->
             {
               b with
               Input_binding.from_common = canonical_recipe b.from_common;
             })
           f.input_bindings);
    input_checks =
      by_encoding Input_check.encode
        (List.map
           (fun (c : Input_check.t) ->
             {
               c with
               Input_check.from_imported_inputs =
                 canonical_recipe c.from_imported_inputs;
             })
           f.input_checks);
    output_bindings =
      by_encoding Output_binding.encode
        (List.map
           (fun (b : Output_binding.t) ->
             {
               b with
               Output_binding.from_left_outputs =
                 canonical_recipe b.from_left_outputs;
               from_right_outputs = canonical_recipe b.from_right_outputs;
             })
           f.output_bindings);
  }

let canonicalize m =
  {
    m with
    units =
      sort_by
        (fun (u : Matched_unit.t) -> Id.Unit.to_string u.unit_id)
        (List.map
           (fun (u : Matched_unit.t) ->
             {
               u with
               Matched_unit.left = canonical_cluster u.left;
               right = canonical_cluster u.right;
               output_pairs = sort_by Id.Member_pair.to_string u.output_pairs;
             })
           m.units);
    tensor_relations =
      sort_by
        (fun (r : Tensor_relation.t) ->
          Id.Tensor_relation.to_string r.relation_id)
        (List.map canonical_relation m.tensor_relations);
    dispositions = by_encoding Member_disposition.encode m.dispositions;
    frontier = canonical_frontier m.frontier;
    derivation_digest = sort_by Digest.Derivation.to_string m.derivation_digest;
  }

let preimage m = Digest_preimage.of_json Mapping (encode (canonicalize m))

(* The frontier alone, for its own digest. *)
let frontier_preimage m =
  Digest_preimage.of_json Frontier_mapping
    (Frontier_mapping.encode (canonicalize m).frontier)
