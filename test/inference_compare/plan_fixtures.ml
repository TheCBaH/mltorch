(* The worked U1/U2 example as concrete endpoint graphs and a supplied
   mapping, plus a second fixture whose union dependency is cyclic. *)

open Inference_compare_schema
module Plan = Inference_compare_plan

let text r = Err.or_raise ~pp_error:Text_error.pp r
let digest_error r = Err.or_raise ~pp_error:Digest_error.pp r
let sha256 = text (Id.Digest_algorithm.of_string "sha256")

let graph_key dialect ch : Graph_key.t =
  {
    dialect = text (Id.Dialect.of_string dialect);
    version = text (Id.Graph_version.of_string "v1");
    graph_digest =
      digest_error (Digest.Graph.make sha256 ~hex:(String.make 64 ch));
  }

let left_key = graph_key "aten" 'a'
let right_key = graph_key "native" 'b'

let tensor g name : Tensor_ref.t =
  { graph = g; scope = []; local_id = text (Id.Tensor_local.of_string name) }

let node g name : Node_ref.t =
  { graph = g; scope = []; local_id = text (Id.Node_local.of_string name) }

let f32 ?(layout = "row_major") shape : Tensor_spec.t =
  {
    dtype =
      {
        codec = Inference_compare_codecs.Element_codec.f32;
        parameters_digest = None;
      };
    logical_shape = shape;
    layout_id = text (Id.Layout.of_string layout);
    value_semantics = true;
  }

let operator d o : Operator_ref.t =
  { dialect = text (Id.Dialect.of_string d); operation = text (Id.Operator.of_string o) }

let node_desc ?(effect = Node_desc.Pure) ?(control = []) g name op operands
    results : Node_desc.t =
  {
    node = node g name;
    operator = operator "x" op;
    operands = List.map (tensor g) operands;
    results = List.map (tensor g) results;
    control_predecessors = List.map (node g) control;
    effect;
  }

let source g name role spec : Graph_source.t =
  { tensor = tensor g name; role; spec }

let user n = Input_role.User_input (text (Name.User_input.of_string n))
let constant n = Input_role.Constant (text (Name.Constant.of_string n))

let graph_desc key nodes sources finals specs : Graph_desc.t =
  {
    key;
    nodes;
    sources;
    final_outputs = List.map (tensor key) finals;
    output_tree =
      Tuple (List.map (fun n -> Graph_desc.Output_tree.Tensor (tensor key n)) finals);
    tensor_specs = List.map (fun (n, s) -> (tensor key n, s)) specs;
  }

let m23 = f32 [ 2L; 3L ]
let m32 = f32 ~layout:"nhwc" [ 3L; 2L ]

(* Left: n1 = add(x, c1) -> h; n2 = mul(h, c2) -> y.
   Right keeps x transposed: m1a relayouts it (a helper), m1b adds, m2 muls. *)
let left_graph =
  let g = left_key in
  graph_desc g
    [
      node_desc g "n1" "add" [ "x"; "c1" ] [ "h" ];
      node_desc g "n2" "mul" [ "h"; "c2" ] [ "y" ];
    ]
    [
      source g "x" (user "x") m23;
      source g "c1" (constant "c1") m23;
      source g "c2" (constant "c2") m23;
    ]
    [ "y" ]
    [ ("h", m23); ("y", m23) ]

let right_graph =
  let g = right_key in
  graph_desc g
    [
      node_desc g "m1a" "permute" [ "xt" ] [ "xr" ];
      node_desc g "m1b" "add" [ "xr"; "k1" ] [ "hr" ];
      node_desc g "m2" "mul" [ "hr"; "k2" ] [ "yr" ];
    ]
    [
      source g "xt" (user "x") m32;
      source g "k1" (constant "c1") m23;
      source g "k2" (constant "c2") m23;
    ]
    [ "yr" ]
    [ ("xr", m23); ("hr", m23); ("yr", m23) ]

let ckey s = text (Id.Canonical_value.of_string s)
let uid s = text (Id.Unit.of_string s)
let node_id s = text (Id.Conversion_node.of_string s)
let recipe_id s = text (Id.Recipe.of_string s)
let comparison s = text (Id.Comparison.of_string s)

let recipe ?(lossless = true) id nodes result output_spec : Recipe.t =
  { id = recipe_id id; nodes; result; output_spec; lossless }

let identity_from_canonical id key spec =
  recipe id [] (Recipe.Canonical (ckey key)) spec

let identity_from_boundary id t spec =
  recipe id [] (Recipe.Boundary t) spec

let transpose id arg spec : Recipe.t =
  recipe id
    [ { id = node_id "t"; operation = Permute [ 1; 0 ]; arguments = [ arg ] } ]
    (Recipe.Conversion_result (node_id "t"))
    spec

let binding unit side tensor from_common : Input_binding.t =
  { unit = uid unit; side; tensor; from_common }

let check unit side canonical from_imported_inputs : Input_check.t =
  { unit = uid unit; side; canonical = ckey canonical; from_imported_inputs }

let cluster g names ins outs : Cluster.t =
  {
    nodes = List.map (node g) names;
    inputs = List.map (tensor g) ins;
    outputs = List.map (tensor g) outs;
  }

let payload name ch : Initial_binding.t =
  {
    key = ckey name;
    payload_id = text (Id.Payload.of_string ("case/" ^ name));
    payload_digest =
      digest_error (Digest.Payload.make sha256 ~hex:(String.make 64 ch));
  }

let mapping : Mapping_artifact.t =
  let l = left_key and r = right_key in
  {
    schema_version = V1;
    left_graph = l;
    right_graph = r;
    units =
      [
        {
          unit_id = uid "u1";
          left = cluster l [ "n1" ] [ "x"; "c1" ] [ "h" ];
          right = cluster r [ "m1a"; "m1b" ] [ "xt"; "k1" ] [ "hr" ];
          output_pairs = [];
        };
        {
          unit_id = uid "u2";
          left = cluster l [ "n2" ] [ "h"; "c2" ] [ "y" ];
          right = cluster r [ "m2" ] [ "hr"; "k2" ] [ "yr" ];
          output_pairs = [ text (Id.Member_pair.of_string "p_y") ];
        };
      ];
    tensor_relations =
      [
        {
          relation_id = text (Id.Tensor_relation.of_string "rel_y");
          left_members = [ tensor l "y" ];
          right_members = [ tensor r "yr" ];
          member_pairs =
            [
              {
                pair_id = text (Id.Member_pair.of_string "p_y");
                comparison_id = comparison "cmp_pair_y";
                left = tensor l "y";
                right = tensor r "yr";
                claim = Identical;
                left_to_comparison = [];
                right_to_comparison = [];
                comparison_spec = m23;
              };
            ];
          routes = [];
        };
      ];
    dispositions =
      [
        Internal_helper { node = node r "m1a"; owner_unit = uid "u1" };
      ];
    frontier =
      {
        canonical_values =
          [
            { key = ckey "x"; spec = m23 };
            { key = ckey "c1"; spec = m23 };
            { key = ckey "c2"; spec = m23 };
            { key = ckey "h"; spec = m23 };
            { key = ckey "y"; spec = m23 };
          ];
        final_outputs =
          [
            {
              occurrence = text (Id.Output_occurrence.of_string "out0");
              canonical = ckey "y";
              left = tensor l "y";
              right = tensor r "yr";
            };
          ];
        initial_values = [ payload "x" '1'; payload "c1" '2'; payload "c2" '3' ];
        input_bindings =
          [
            binding "u1" Left (tensor l "x") (identity_from_canonical "b1" "x" m23);
            binding "u1" Left (tensor l "c1") (identity_from_canonical "b2" "c1" m23);
            binding "u1" Right (tensor r "xt")
              (transpose "b3" (Recipe.Canonical (ckey "x")) m32);
            binding "u1" Right (tensor r "k1") (identity_from_canonical "b4" "c1" m23);
            binding "u2" Left (tensor l "h") (identity_from_canonical "b5" "h" m23);
            binding "u2" Left (tensor l "c2") (identity_from_canonical "b6" "c2" m23);
            binding "u2" Right (tensor r "hr") (identity_from_canonical "b7" "h" m23);
            binding "u2" Right (tensor r "k2") (identity_from_canonical "b8" "c2" m23);
          ];
        input_checks =
          [
            check "u1" Left "x" (identity_from_boundary "k1" (tensor l "x") m23);
            check "u1" Left "c1" (identity_from_boundary "k2" (tensor l "c1") m23);
            check "u1" Right "x"
              (transpose "k3" (Recipe.Boundary (tensor r "xt")) m23);
            check "u1" Right "c1" (identity_from_boundary "k4" (tensor r "k1") m23);
            check "u2" Left "h" (identity_from_boundary "k5" (tensor l "h") m23);
            check "u2" Left "c2" (identity_from_boundary "k6" (tensor l "c2") m23);
            check "u2" Right "h" (identity_from_boundary "k7" (tensor r "hr") m23);
            check "u2" Right "c2" (identity_from_boundary "k8" (tensor r "k2") m23);
          ];
        output_bindings =
          [
            {
              unit = uid "u1";
              key = ckey "h";
              from_left_outputs = identity_from_boundary "o1" (tensor l "h") m23;
              from_right_outputs = identity_from_boundary "o2" (tensor r "hr") m23;
              comparison_id = comparison "cmp_h";
            };
            {
              unit = uid "u2";
              key = ckey "y";
              from_left_outputs = identity_from_boundary "o3" (tensor l "y") m23;
              from_right_outputs = identity_from_boundary "o4" (tensor r "yr") m23;
              comparison_id = comparison "cmp_y";
            };
          ];
      };
    derivation_digest = [];
  }

let limits : Resource_limits.t =
  {
    max_live_bytes = 1_000_000L;
    max_conversion_scratch_bytes = 1_000_000L;
    max_snapshot_bytes = 1_000_000L;
    max_retained_disk_bytes = 0L;
    max_mismatch_samples = 8L;
    retention = No_retention;
    retention_requirement = Best_effort;
    verified_spill = false;
    max_nodes = 1000L;
    max_edges = 1000L;
    max_units = 100L;
    max_recipe_nodes = 1000L;
    max_selector_nodes = 100L;
    max_selector_depth = 16L;
    max_path_length = 8L;
    max_retained_witnesses = 100L;
  }

let options : Plan.Correspondence_check.options =
  {
    codecs = Inference_compare_codecs.Element_codec.Registry.builtin;
    operations = Inference_compare_codecs.Conversion_registry.empty;
  }

let plan ?(limits = limits) ?(left = left_graph) ?(right = right_graph) m =
  Plan.Structural.check ~limits ~options ~left ~right m

let pp_error = Err.Error.pp_kind Plan.Plan_error.pp

(* Run [m] and print a one-line outcome. *)
let outcome ?limits ?left ?right name m =
  match plan ?limits ?left ?right m with
  | Ok p ->
      Format.printf "%s: ok units=[%a]@." name
        Fmt.(list ~sep:(any " ") Id.Unit.pp)
        (List.map
           (fun (u : Plan.Correspondence_check.Unit_plan.t) -> u.unit)
           (Plan.Structural.units p))
  | Error e -> Format.printf "%s: %a@." name pp_error e

(* Cycle fixture. Left: a1 -> b1 (tensor p). Right: b2 -> a2 (tensor r).
   Units A = {a1, a2} and B = {b1, b2}; each endpoint is acyclic, the union is
   not. Canonical p is carried by left p / right s (a diagnostic of a2), and
   canonical q by left q / right r. *)
let cyc_left_key = graph_key "aten" 'c'
let cyc_right_key = graph_key "native" 'd'

let cyc_left =
  let g = cyc_left_key in
  graph_desc g
    [
      node_desc g "a1" "id" [ "x" ] [ "p" ];
      node_desc g "b1" "id" [ "p" ] [ "q" ];
    ]
    [ source g "x" (user "x") m23 ]
    [ "q" ]
    [ ("p", m23); ("q", m23) ]

let cyc_right =
  let g = cyc_right_key in
  graph_desc g
    [
      node_desc g "b2" "id" [ "x" ] [ "r" ];
      node_desc g "a2" "id" [ "r" ] [ "s" ];
    ]
    [ source g "x" (user "x") m23 ]
    [ "r" ]
    [ ("r", m23); ("s", m23) ]

let cyc_mapping : Mapping_artifact.t =
  let l = cyc_left_key and r = cyc_right_key in
  {
    schema_version = V1;
    left_graph = l;
    right_graph = r;
    units =
      [
        {
          unit_id = uid "A";
          left = cluster l [ "a1" ] [ "x" ] [ "p" ];
          right = cluster r [ "a2" ] [ "r" ] [ "s" ];
          output_pairs = [];
        };
        {
          unit_id = uid "B";
          left = cluster l [ "b1" ] [ "p" ] [ "q" ];
          right = cluster r [ "b2" ] [ "x" ] [ "r" ];
          output_pairs = [];
        };
      ];
    tensor_relations = [];
    dispositions = [];
    frontier =
      {
        canonical_values =
          [
            { key = ckey "x"; spec = m23 };
            { key = ckey "p"; spec = m23 };
            { key = ckey "q"; spec = m23 };
          ];
        final_outputs =
          [
            {
              occurrence = text (Id.Output_occurrence.of_string "out0");
              canonical = ckey "q";
              left = tensor l "q";
              right = tensor r "r";
            };
          ];
        initial_values = [ payload "x" '4' ];
        input_bindings =
          [
            binding "A" Left (tensor l "x") (identity_from_canonical "b1" "x" m23);
            binding "A" Right (tensor r "r") (identity_from_canonical "b2" "q" m23);
            binding "B" Left (tensor l "p") (identity_from_canonical "b3" "p" m23);
            binding "B" Right (tensor r "x") (identity_from_canonical "b4" "x" m23);
          ];
        input_checks =
          [
            check "A" Left "x" (identity_from_boundary "k1" (tensor l "x") m23);
            check "A" Right "q" (identity_from_boundary "k2" (tensor r "r") m23);
            check "B" Left "p" (identity_from_boundary "k3" (tensor l "p") m23);
            check "B" Right "x" (identity_from_boundary "k4" (tensor r "x") m23);
          ];
        output_bindings =
          [
            {
              unit = uid "A";
              key = ckey "p";
              from_left_outputs = identity_from_boundary "o1" (tensor l "p") m23;
              from_right_outputs = identity_from_boundary "o2" (tensor r "s") m23;
              comparison_id = comparison "cmp_p";
            };
            {
              unit = uid "B";
              key = ckey "q";
              from_left_outputs = identity_from_boundary "o3" (tensor l "q") m23;
              from_right_outputs = identity_from_boundary "o4" (tensor r "r") m23;
              comparison_id = comparison "cmp_q";
            };
          ];
      };
    derivation_digest = [];
  }
