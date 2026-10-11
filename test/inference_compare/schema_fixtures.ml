(* Small, hand-built records for the schema tests. Constructors fail loudly:
   a fixture that does not satisfy its own invariants is a test bug. *)

open Inference_compare_schema

let ok pp r = Err.or_raise ~pp_error:pp r
let text r = ok Text_error.pp r
let digest r = ok Digest_error.pp r

let unit_id s = text (Id.Unit.of_string s)
let case_id s = text (Id.Case.of_string s)
let canonical s = text (Id.Canonical_value.of_string s)
let pair_id s = text (Id.Member_pair.of_string s)
let comparison s = text (Id.Comparison.of_string s)
let sha ch = String.make 64 ch

let algorithm = text (Id.Digest_algorithm.of_string "sha256")
let graph_digest ch = digest (Digest.Graph.make algorithm ~hex:(sha ch))
let payload_digest ch = digest (Digest.Payload.make algorithm ~hex:(sha ch))

let graph_key ~dialect ~version ch : Graph_key.t =
  {
    dialect = text (Id.Dialect.of_string dialect);
    version = text (Id.Graph_version.of_string version);
    graph_digest = graph_digest ch;
  }

let left_graph = graph_key ~dialect:"aten" ~version:"v1" 'a'
let right_graph = graph_key ~dialect:"native" ~version:"v1" 'b'

let scope names =
  List.map (fun n -> text (Name.Scope_component.of_string n)) names

let tensor graph ?(path = []) name : Tensor_ref.t =
  {
    graph;
    scope = scope path;
    local_id = text (Id.Tensor_local.of_string name);
  }

let node graph ?(path = []) name : Node_ref.t =
  { graph; scope = scope path; local_id = text (Id.Node_local.of_string name) }

let f32 shape : Tensor_spec.t =
  {
    dtype =
      { codec = text (Id.Codec.of_string "f32"); parameters_digest = None };
    logical_shape = shape;
    layout_id = text (Id.Layout.of_string "row_major");
    value_semantics = true;
  }

let operator dialect name : Operator_ref.t =
  {
    dialect = text (Id.Dialect.of_string dialect);
    operation = text (Id.Operator.of_string name);
  }

let node_desc graph name op ~operands ~results : Node_desc.t =
  {
    node = node graph name;
    operator = operator "aten" op;
    operands = List.map (tensor graph) operands;
    results = List.map (tensor graph) results;
    control_predecessors = [];
    effect = Pure;
  }

(* x -> add -> h -> mul -> y, returning [y] twice and [h] once so output
   occurrences repeat. *)
let graph_desc : Graph_desc.t =
  let g = left_graph in
  {
    key = g;
    nodes =
      [
        node_desc g "n_add" "add" ~operands:[ "x"; "x" ] ~results:[ "h" ];
        node_desc g "n_mul" "mul" ~operands:[ "h"; "w" ] ~results:[ "y" ];
      ];
    sources =
      [
        {
          tensor = tensor g "x";
          role = User_input (text (Name.User_input.of_string "x"));
          spec = f32 [ 2L; 3L ];
        };
        {
          tensor = tensor g "w";
          role = Constant (text (Name.Constant.of_string "w"));
          spec = f32 [ 2L; 3L ];
        };
      ];
    final_outputs = [ tensor g "y"; tensor g "y"; tensor g "h" ];
    output_tree =
      Tuple
        [
          Tensor (tensor g "y");
          Tensor_list [ Tensor (tensor g "y"); Tensor (tensor g "h") ];
        ];
    tensor_specs =
      [
        (tensor g "x", f32 [ 2L; 3L ]);
        (tensor g "h", f32 [ 2L; 3L ]);
        (tensor g "y", f32 [ 2L; 3L ]);
      ];
  }

let recipe id ~result nodes spec : Recipe.t =
  {
    id = text (Id.Recipe.of_string id);
    nodes;
    result;
    output_spec = spec;
    lossless = true;
  }

let reshape_recipe : Recipe.t =
  let n = text (Id.Conversion_node.of_string "r0") in
  recipe "from_x"
    ~result:(Conversion_result n)
    [
      {
        id = n;
        operation = Reshape [ 6L ];
        arguments = [ Canonical (canonical "x") ];
      };
    ]
    (f32 [ 6L ])

let identity_recipe id key spec : Recipe.t =
  recipe id ~result:(Canonical (canonical key)) [] spec

let member_pair : Member_pair.t =
  {
    pair_id = pair_id "p_y";
    comparison_id = comparison "cmp_y";
    left = tensor left_graph "y";
    right = tensor right_graph "y2";
    claim = Approximate [ text (Id.Precision.of_string "f32") ];
    left_to_comparison = [];
    right_to_comparison = [ Reshape [ 2L; 3L ] ];
    comparison_spec = f32 [ 2L; 3L ];
  }

let mapping : Mapping_artifact.t =
  let cluster g names ins outs : Cluster.t =
    {
      nodes = List.map (node g) names;
      inputs = List.map (tensor g) ins;
      outputs = List.map (tensor g) outs;
    }
  in
  {
    schema_version = V1;
    left_graph;
    right_graph;
    units =
      [
        {
          unit_id = unit_id "u2";
          left = cluster left_graph [ "n_mul" ] [ "h"; "w" ] [ "y" ];
          right = cluster right_graph [ "m_mul" ] [ "h2"; "w2" ] [ "y2" ];
          output_pairs = [ pair_id "p_y" ];
        };
        {
          unit_id = unit_id "u1";
          left = cluster left_graph [ "n_add" ] [ "x" ] [ "h" ];
          right = cluster right_graph [ "m_add" ] [ "x2" ] [ "h2" ];
          output_pairs = [];
        };
      ];
    tensor_relations =
      [
        {
          relation_id = text (Id.Tensor_relation.of_string "rel_y");
          left_members = [ tensor left_graph "y" ];
          right_members = [ tensor right_graph "y2" ];
          member_pairs = [ member_pair ];
          routes = [];
        };
      ];
    dispositions =
      [
        Eliminated
          {
            node = node left_graph "n_dead";
            reason = text (Prose.Description.of_string "folded");
          };
      ];
    frontier =
      {
        canonical_values =
          [ { key = canonical "x"; spec = f32 [ 2L; 3L ] } ];
        initial_values =
          [
            {
              key = canonical "x";
              payload_id = text (Id.Payload.of_string "case0/x");
              payload_digest = payload_digest 'c';
            };
          ];
        input_bindings =
          [
            {
              unit = unit_id "u1";
              side = Left;
              tensor = tensor left_graph "x";
              from_common = identity_recipe "r_x_left" "x" (f32 [ 2L; 3L ]);
            };
            {
              unit = unit_id "u1";
              side = Right;
              tensor = tensor right_graph "x2";
              from_common = reshape_recipe;
            };
          ];
        input_checks = [];
        output_bindings = [];
      };
    derivation_digest = [ digest (Digest.Derivation.make algorithm ~hex:(sha 'd')) ];
  }
