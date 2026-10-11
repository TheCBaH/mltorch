(* Three small synthetic graphs and the maps between them, for composition
   tests. A map is described by its units, its tensor correspondences and the
   nodes one side deletes or creates; boundaries, identity recipes, routes,
   canonical keys (the left tensor's name) and pinned constants are derived. *)

open Inference_compare_schema
module P = Plan_fixtures
module C = Chain_fixture
module Plan = Inference_compare_plan
module Compose = Inference_compare_compose.Compose

type graph = {
  key : Graph_key.t;
  shape : int64 list;
  nodes : (string * string * string list * string list) list;
      (** name, operation, operands, results *)
  sources : string list;  (** The first is the user input. *)
  finals : string list;
}

let spec_of (g : graph) = { C.i64_spec with logical_shape = g.shape }

let desc (g : graph) : Graph_desc.t =
  let spec = spec_of g in
  let produced = List.concat_map (fun (_, _, _, r) -> r) g.nodes in
  P.graph_desc g.key
    (List.map
       (fun (n, op, operands, results) ->
         {
           (P.node_desc g.key n op operands results) with
           operator = P.operator "synthetic" op;
         })
       g.nodes)
    (List.mapi
       (fun i s ->
         P.source g.key s (if i = 0 then P.user "x" else P.constant s) spec)
       g.sources)
    g.finals
    (List.map (fun p -> (p, spec)) produced)

type map = {
  left : graph;
  right : graph;
  units : (string * string list * string list) list;
      (** unit, left nodes, right nodes *)
  tensors : (string * string) list;  (** left tensor, right tensor *)
  eliminated : string list;  (** left nodes *)
  created : string list;  (** right nodes *)
  constants : string list;  (** right sources with no left counterpart *)
}

let dedupe l =
  List.fold_left (fun acc x -> if List.mem x acc then acc else acc @ [ x ]) [] l

let boundary (g : graph) names =
  let members = List.filter (fun (n, _, _, _) -> List.mem n names) g.nodes in
  let inside = List.concat_map (fun (_, _, _, r) -> r) members in
  let outside =
    List.filter (fun (n, _, _, _) -> not (List.mem n names)) g.nodes
  in
  let ins =
    dedupe
      (List.concat_map
         (fun (_, _, o, _) -> List.filter (fun x -> not (List.mem x inside)) o)
         members)
  and outs =
    List.filter
      (fun r ->
        List.mem r g.finals
        || List.exists (fun (_, _, o, _) -> List.mem r o) outside)
      inside
  in
  (ins, outs)

(* [arg] holds a value of spec [from]; read it as [into]. *)
let convert ~id arg ~(from : Tensor_spec.t) ~(into : Tensor_spec.t) =
  if Tensor_spec.equal from into then P.recipe id [] arg into
  else
    P.recipe id
      [
        {
          Recipe.id = P.node_id "rs";
          operation = Reshape into.logical_shape;
          arguments = [ arg ];
        };
      ]
      (Recipe.Conversion_result (P.node_id "rs"))
      into

let reason = P.text (Prose.Description.of_string "by construction")

let mapping (m : map) : Mapping_artifact.t =
  let l = m.left.key and r = m.right.key in
  let lspec = spec_of m.left and rspec = spec_of m.right in
  let spec = lspec in
  let key_of_right rt =
    match List.find_opt (fun (_, y) -> y = rt) m.tensors with
    | Some (x, _) -> x
    | None -> "k_" ^ rt
  in
  let right_of_left lt = List.assoc lt m.tensors in
  let cluster g key names =
    let ins, outs = boundary g names in
    ( ins,
      outs,
      {
        Cluster.nodes = List.map (P.node key) names;
        inputs = List.map (P.tensor key) ins;
        outputs = List.map (P.tensor key) outs;
      } )
  in
  let infos =
    List.map
      (fun (u, ln, rn) ->
        let lin, lout, lc = cluster m.left l ln
        and rin, rout, rc = cluster m.right r rn in
        (u, lin, lout, rin, rout, lc, rc))
      m.units
  in
  let units =
    List.map
      (fun (u, _, _, _, _, lc, rc) ->
        {
          Matched_unit.unit_id = P.uid u;
          left = lc;
          right = rc;
          output_pairs = [];
        })
      infos
  in
  let id fmt = Printf.ksprintf Fun.id fmt in
  let input_bindings =
    List.concat_map
      (fun (u, lin, _, rin, _, _, _) ->
        List.map
          (fun t ->
            P.binding u Left (P.tensor l t)
              (P.identity_from_canonical (id "bl-%s-%s" u t) t spec))
          lin
        @ List.map
            (fun t ->
              P.binding u Right (P.tensor r t)
                (convert ~id:(id "br-%s-%s" u t)
                   (Recipe.Canonical (P.ckey (key_of_right t)))
                   ~from:spec ~into:rspec))
            rin)
      infos
  in
  let input_checks =
    List.concat_map
      (fun (u, lin, _, rin, _, _, _) ->
        List.map
          (fun t ->
            P.check u Left t
              (P.identity_from_boundary (id "kl-%s-%s" u t) (P.tensor l t) spec))
          lin
        @ List.map
            (fun t ->
              P.check u Right (key_of_right t)
                (convert ~id:(id "kr-%s-%s" u t)
                   (Recipe.Boundary (P.tensor r t))
                   ~from:rspec ~into:spec))
            rin)
      infos
  in
  let output_bindings =
    List.concat_map
      (fun (u, _, lout, _, _, _, _) ->
        List.map
          (fun t ->
            {
              Output_binding.unit = P.uid u;
              key = P.ckey t;
              from_left_outputs =
                P.identity_from_boundary (id "ol-%s-%s" u t) (P.tensor l t) spec;
              from_right_outputs =
                convert ~id:(id "or-%s-%s" u t)
                  (Recipe.Boundary (P.tensor r (right_of_left t)))
                  ~from:rspec ~into:spec;
              comparison_id = P.comparison ("cmp_" ^ t);
            })
          lout)
      infos
  in
  let left_keys =
    dedupe
      (m.left.sources
      @ List.concat_map (fun (_, _, lout, _, _, _, _) -> lout) infos)
  in
  let keys = left_keys @ List.map (fun c -> "k_" ^ c) m.constants in
  let relation (lt, rt) =
    let route dir src dst =
      {
        Value_route.route_id =
          P.text (Id.Value_route.of_string (dir ^ "_" ^ lt));
        sources = [ src ];
        destination = dst;
        steps =
          (if Tensor_spec.equal lspec rspec then []
           else
             [
               Conversion_step.Reshape
                 (if dir = "f" then rspec else lspec).logical_shape;
             ]);
        claim = Identical;
        lossy = false;
      }
    in
    let lt_ref = P.tensor l lt and rt_ref = P.tensor r rt in
    {
      Tensor_relation.relation_id =
        P.text (Id.Tensor_relation.of_string ("rel_" ^ lt));
      left_members = [ lt_ref ];
      right_members = [ rt_ref ];
      member_pairs = [];
      routes = [ route "f" lt_ref rt_ref; route "r" rt_ref lt_ref ];
    }
  in
  let chain_for_values =
    C.synthetic [] ~sources:m.left.sources ~finals:[] ~units:[]
  in
  {
    schema_version = V1;
    left_graph = l;
    right_graph = r;
    units;
    tensor_relations = List.map relation m.tensors;
    dispositions =
      List.map
        (fun n -> Member_disposition.Eliminated { node = P.node l n; reason })
        m.eliminated
      @ List.map
          (fun n -> Member_disposition.Created { node = P.node r n; reason })
          m.created;
    frontier =
      {
        canonical_values =
          List.map (fun k -> { Canonical_spec.key = P.ckey k; spec }) keys;
        final_outputs =
          List.mapi
            (fun i (lf, rf) ->
              {
                Final_output.occurrence =
                  P.text
                    (Id.Output_occurrence.of_string (Printf.sprintf "out%d" i));
                canonical = P.ckey lf;
                left = P.tensor l lf;
                right = P.tensor r rf;
              })
            (List.combine m.left.finals m.right.finals);
        initial_values =
          List.map
            (fun s -> C.initial_binding chain_for_values s)
            m.left.sources
          @ List.map (fun c -> P.payload ("k_" ^ c) '5') m.constants;
        input_bindings;
        input_checks;
        output_bindings;
      };
    derivation_digest = [];
  }

let check ?(limits = P.limits) (m : map) mapping =
  Plan.Structural.check ~limits ~options:P.options ~left:(desc m.left)
    ~right:(desc m.right) mapping

let hashers = Inference_compare_sha256.registry
let algorithm = Inference_compare_sha256.algorithm
let compose a b = Compose.compose ~hashers ~algorithm a b

(* Meaning, not provenance: the derivation list records how a map was reached. *)
let meaning (m : Mapping_artifact.t) =
  Mapping_artifact.canonicalize { m with derivation_digest = [] }

let graph ?(shape = [ 3L ]) key nodes ~sources ~finals =
  { key; shape; nodes; sources; finals }

let key_x = P.graph_key "synthetic" '1'
let key_y = P.graph_key "synthetic" '2'
let key_z = P.graph_key "synthetic" '3'
let key_w = P.graph_key "synthetic" '4'
