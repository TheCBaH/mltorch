(* The correspondence the importer knows while it lowers: which Native nodes
   each exporter node produced (relayout helpers included), which Native edge
   each exporter tensor denotes, and the conversions between the two
   representations. Exported as an ordinary mapping artifact; nothing here is
   discovered by comparing values or names. *)

open Inference_compare_schema
module Nid = Graph_ir.Node_id
module Tid = Graph_ir.Tensor_id

let ( let* ) = Result.bind
let text r = Result.map_error (fun _ -> "invalid identifier") r

let rec all f = function
  | [] -> Ok []
  | x :: rest ->
      let* y = f x in
      let* ys = all f rest in
      Ok (y :: ys)

let dedupe l =
  List.fold_left (fun acc x -> if List.mem x acc then acc else acc @ [ x ]) [] l

(* A cluster's boundary, recomputed from the dataflow the way the planner will:
   what it reads from outside, and what it produces that something outside (or
   the graph's own result list) uses. *)
let boundary ~(operands : 'n -> string list) ~(results : 'n -> string list)
    ~(nodes : 'n list) ~(all_nodes : 'n list) ~(finals : string list) =
  let inside = List.concat_map results nodes in
  let ins =
    dedupe
      (List.concat_map
         (fun n -> List.filter (fun o -> not (List.mem o inside)) (operands n))
         nodes)
  in
  let outside = List.filter (fun n -> not (List.memq n nodes)) all_nodes in
  let read_outside = List.concat_map operands outside in
  let outs =
    List.filter (fun r -> List.mem r finals || List.mem r read_outside) inside
  in
  (ins, outs)

type t = {
  mapping : Mapping_artifact.t;
  keys : string list;  (** Canonical values: exporter tensor names. *)
}

let unit_name i = Printf.sprintf "u%05d" i

let make ~(aten : Aten_model.t) ~(native : Native_model.t)
    ~(payload_digest : string -> (Digest.Payload.t, string) result) :
    (t, string) result =
  let capture = native.capture in
  let emitted i = Option.value ~default:[] (List.assoc_opt i capture.emitted) in
  (* exporter name -> Native edge *)
  let native_of_name = capture.sources @ List.concat_map snd capture.bound in
  let native_edge name = List.assoc_opt name native_of_name in
  let g = native.graph in
  let native_nodes = g.Graph_ir.Graph.nodes in
  let native_node id =
    List.find_opt (fun (n : Graph_ir.node) -> Nid.equal n.id id) native_nodes
  in
  let native_final = List.map (fun id -> id) g.outputs in
  let tl = Native_model.tensor_local in
  (* ATen side *)
  let aten_nodes = Array.to_list aten.nodes |> List.mapi (fun i n -> (i, n)) in
  let aten_outputs =
    List.concat_map Aten_model.tensor_names_of_arg aten.graph.outputs
  in
  let aten_boundary i =
    boundary
      ~operands:(fun (_, n) -> Aten_model.operands n)
      ~results:(fun (_, n) -> Aten_model.results n)
      ~nodes:[ List.nth aten_nodes i ]
      ~all_nodes:aten_nodes ~finals:aten_outputs
  in
  let native_boundary i =
    let nodes = List.filter_map native_node (emitted i) in
    boundary
      ~operands:(fun (n : Graph_ir.node) ->
        List.map tl (Graph_ir.operands n.op))
      ~results:(fun (n : Graph_ir.node) -> List.map tl n.outputs)
      ~nodes ~all_nodes:native_nodes ~finals:(List.map tl native_final)
  in
  let akey = aten.key and nkey = native.key in
  let aten_spec name =
    match List.assoc_opt name aten.specs with
    | Some s -> Ok s
    | None -> Error ("no ATen spec for " ^ name)
  in
  let native_spec id =
    let* s = Native_model.sig_of g id in
    Native_model.spec_of_sig s
  in
  let aten_ref name = Aten_model.tensor_ref ~key:akey ~scope:[] name in
  let native_ref id = Native_model.tensor_ref ~key:nkey ~scope:[] id in
  let id_of_local s =
    Tid.of_int (int_of_string (String.sub s 1 (String.length s - 1)))
  in
  let n = Array.length aten.nodes in
  (* Per unit: the ATen and Native tensors crossing its boundary, as exporter
     names (the Native ones are translated back through the binding). *)
  let name_of_edge = List.map (fun (nm, id) -> (tl id, nm)) native_of_name in
  let to_name local = List.assoc_opt local name_of_edge in
  let units_info =
    List.init n (fun i ->
        let a_in, a_out = aten_boundary i in
        let n_in, n_out = native_boundary i in
        (i, a_in, a_out, n_in, n_out))
  in
  (* Consistency: both sides must cross the unit boundary at the same
     exporter tensors. Anything else is an importer gap, reported here with the
     unit and tensor rather than discovered later as a planning error. *)
  let* () =
    let rec go = function
      | [] -> Ok ()
      | (i, a_in, a_out, n_in, n_out) :: rest ->
          let names l = List.filter_map to_name l in
          let missing l = List.filter (fun x -> to_name x = None) l in
          if missing n_in <> [] || missing n_out <> [] then
            Error
              (Printf.sprintf
                 "unit %d: Native edge(s) %s cross the boundary with no \
                  exporter name"
                 i
                 (String.concat "," (missing n_in @ missing n_out)))
          else if
            List.sort compare a_in <> List.sort compare (dedupe (names n_in))
          then
            Error
              (Printf.sprintf "unit %d: inputs differ: ATen [%s] vs Native [%s]"
                 i (String.concat "," a_in)
                 (String.concat "," (names n_in)))
          else if
            List.sort compare a_out <> List.sort compare (dedupe (names n_out))
          then
            Error
              (Printf.sprintf
                 "unit %d: outputs differ: ATen [%s] vs Native [%s]" i
                 (String.concat "," a_out)
                 (String.concat "," (names n_out)))
          else go rest
    in
    go units_info
  in
  let keys =
    dedupe
      (List.map (fun (s : Aten_model.source) -> s.name) aten.sources
      @ List.concat_map (fun (_, _, a_out, _, _) -> a_out) units_info)
  in
  let* canonical_values =
    all
      (fun k ->
        let* spec = aten_spec k in
        let* key = text (Id.Canonical_value.of_string k) in
        Ok { Canonical_spec.key; spec })
      keys
  in
  (* recipes *)
  let recipe_id s = text (Id.Recipe.of_string s) in
  let node_id s = text (Id.Conversion_node.of_string s) in
  let identity ~id arg spec =
    let* id = recipe_id id in
    Ok
      {
        Recipe.id;
        nodes = [];
        result = arg;
        output_spec = spec;
        lossless = true;
      }
  in
  let reshape ~id arg (target : Tensor_spec.t) =
    let* id = recipe_id id in
    let* nid = node_id "reshape" in
    Ok
      {
        Recipe.id;
        nodes =
          [
            {
              Recipe.id = nid;
              operation = Reshape target.logical_shape;
              arguments = [ arg ];
            };
          ];
        result = Conversion_result nid;
        output_spec = target;
        lossless = true;
      }
  in
  (* from the canonical (ATen-shaped) value to a Native-shaped input, or back *)
  let to_native ~id arg ~(canonical : Tensor_spec.t) ~(native : Tensor_spec.t) =
    if Tensor_spec.equal canonical native then identity ~id arg native
    else reshape ~id arg native
  in
  let to_canonical ~id arg ~(canonical : Tensor_spec.t)
      ~(native : Tensor_spec.t) =
    if Tensor_spec.equal canonical native then identity ~id arg canonical
    else reshape ~id arg canonical
  in
  let side_name s = Side.to_string s in
  (* units *)
  let* units =
    all
      (fun (i, a_in, a_out, n_in, n_out) ->
        let* unit_id = text (Id.Unit.of_string (unit_name i)) in
        let* anode = Aten_model.node_ref ~key:akey ~scope:[] i in
        let* nnodes =
          all (Native_model.node_ref ~key:nkey ~scope:[]) (emitted i)
        in
        let* a_in = all aten_ref a_in in
        let* a_out = all aten_ref a_out in
        let* n_in = all (fun l -> native_ref (id_of_local l)) n_in in
        let* n_out = all (fun l -> native_ref (id_of_local l)) n_out in
        Ok
          {
            Matched_unit.unit_id;
            left = { Cluster.nodes = [ anode ]; inputs = a_in; outputs = a_out };
            right = { Cluster.nodes = nnodes; inputs = n_in; outputs = n_out };
            output_pairs = [];
          })
      units_info
  in
  (* bindings, checks, output bindings *)
  let* input_bindings, input_checks, output_bindings =
    let acc_b = ref [] and acc_c = ref [] and acc_o = ref [] in
    let* () =
      let rec go = function
        | [] -> Ok ()
        | (i, a_in, a_out, _, _) :: rest ->
            let uid = unit_name i in
            let* unit = text (Id.Unit.of_string uid) in
            let* () =
              let rec ins = function
                | [] -> Ok ()
                | name :: more ->
                    let* canonical = aten_spec name in
                    let* key = text (Id.Canonical_value.of_string name) in
                    let* ar = aten_ref name in
                    let edge =
                      match native_edge name with
                      | Some e -> e
                      | None -> invalid_arg "checked above"
                    in
                    let* nr = native_ref edge in
                    let* nspec = native_spec edge in
                    let* left_b =
                      identity
                        ~id:
                          (Printf.sprintf "b-%s-%s-%s" uid (side_name Left) name)
                        (Recipe.Canonical key) canonical
                    in
                    let* right_b =
                      to_native
                        ~id:
                          (Printf.sprintf "b-%s-%s-%s" uid (side_name Right)
                             name)
                        (Recipe.Canonical key) ~canonical ~native:nspec
                    in
                    let* left_c =
                      identity
                        ~id:
                          (Printf.sprintf "k-%s-%s-%s" uid (side_name Left) name)
                        (Recipe.Boundary ar) canonical
                    in
                    let* right_c =
                      to_canonical
                        ~id:
                          (Printf.sprintf "k-%s-%s-%s" uid (side_name Right)
                             name)
                        (Recipe.Boundary nr) ~canonical ~native:nspec
                    in
                    acc_b :=
                      !acc_b
                      @ [
                          {
                            Input_binding.unit;
                            side = Left;
                            tensor = ar;
                            from_common = left_b;
                          };
                          {
                            Input_binding.unit;
                            side = Right;
                            tensor = nr;
                            from_common = right_b;
                          };
                        ];
                    acc_c :=
                      !acc_c
                      @ [
                          {
                            Input_check.unit;
                            side = Left;
                            canonical = key;
                            from_imported_inputs = left_c;
                          };
                          {
                            Input_check.unit;
                            side = Right;
                            canonical = key;
                            from_imported_inputs = right_c;
                          };
                        ];
                    ins more
              in
              ins a_in
            in
            let* () =
              let rec outs = function
                | [] -> Ok ()
                | name :: more ->
                    let* canonical = aten_spec name in
                    let* key = text (Id.Canonical_value.of_string name) in
                    let* ar = aten_ref name in
                    let edge = Option.get (native_edge name) in
                    let* nr = native_ref edge in
                    let* nspec = native_spec edge in
                    let* l =
                      identity
                        ~id:(Printf.sprintf "o-%s-left-%s" uid name)
                        (Recipe.Boundary ar) canonical
                    in
                    let* r =
                      to_canonical
                        ~id:(Printf.sprintf "o-%s-right-%s" uid name)
                        (Recipe.Boundary nr) ~canonical ~native:nspec
                    in
                    let* comparison_id =
                      text (Id.Comparison.of_string ("cmp:" ^ name))
                    in
                    acc_o :=
                      !acc_o
                      @ [
                          {
                            Output_binding.unit;
                            key;
                            from_left_outputs = l;
                            from_right_outputs = r;
                            comparison_id;
                          };
                        ];
                    outs more
              in
              outs a_out
            in
            go rest
      in
      go units_info
    in
    Ok (!acc_b, !acc_c, !acc_o)
  in
  (* initial payloads *)
  let* initial_values =
    all
      (fun (s : Aten_model.source) ->
        let* key = text (Id.Canonical_value.of_string s.name) in
        let* payload_id = text (Id.Payload.of_string ("case/" ^ s.name)) in
        let* payload_digest = payload_digest s.name in
        Ok { Initial_binding.key; payload_id; payload_digest })
      aten.sources
  in
  (* ordered final outputs *)
  let* final_outputs =
    let native_out_names = List.map (fun id -> tl id) native_final in
    if List.length aten_outputs <> List.length native_out_names then
      Error "the two graphs return a different number of outputs"
    else
      all
        (fun (i, (a, nloc)) ->
          let* occurrence =
            text (Id.Output_occurrence.of_string (Printf.sprintf "out%d" i))
          in
          let* canonical = text (Id.Canonical_value.of_string a) in
          let* left = aten_ref a in
          let* right = native_ref (id_of_local nloc) in
          Ok { Final_output.occurrence; canonical; left; right })
        (List.mapi
           (fun i x -> (i, x))
           (List.combine aten_outputs native_out_names))
  in
  (* helpers: emitted nodes that produce nothing the rest of the model sees *)
  let* dispositions =
    all Fun.id
      (List.concat_map
         (fun (i, _, _, _, n_out) ->
           List.filter_map
             (fun id ->
               match native_node id with
               | None -> None
               | Some (nd : Graph_ir.node) ->
                   if List.exists (fun o -> List.mem (tl o) n_out) nd.outputs
                   then None
                   else
                     Some
                       (let* node =
                          Native_model.node_ref ~key:nkey ~scope:[] id
                        in
                        let* owner_unit =
                          text (Id.Unit.of_string (unit_name i))
                        in
                        Ok
                          (Member_disposition.Internal_helper
                             { node; owner_unit })))
             (emitted i))
         units_info)
  in
  let* tensor_relations =
    all
      (fun k ->
        let* relation_id = text (Id.Tensor_relation.of_string ("rel:" ^ k)) in
        let* a = aten_ref k in
        let* e =
          match native_edge k with
          | Some e -> Ok e
          | None -> Error ("no Native edge for " ^ k)
        in
        let* nr = native_ref e in
        let* canonical = aten_spec k in
        let* nspec = native_spec e in
        let route dir source destination steps =
          let* route_id =
            text (Id.Value_route.of_string (Printf.sprintf "%s:%s" dir k))
          in
          Ok
            {
              Value_route.route_id;
              sources = [ source ];
              destination;
              steps;
              claim = Claim.Identical;
              lossy = false;
            }
        in
        let steps (target : Tensor_spec.t) =
          if Tensor_spec.equal canonical nspec then []
          else [ Conversion_step.Reshape target.logical_shape ]
        in
        let* forward = route "to-native" a nr (steps nspec) in
        let* backward = route "from-native" nr a (steps canonical) in
        Ok
          {
            Tensor_relation.relation_id;
            left_members = [ a ];
            right_members = [ nr ];
            member_pairs = [];
            routes = [ forward; backward ];
          })
      keys
  in
  Ok
    {
      mapping =
        {
          Mapping_artifact.schema_version = V1;
          left_graph = akey;
          right_graph = nkey;
          units;
          tensor_relations;
          dispositions;
          frontier =
            {
              Frontier_mapping.canonical_values;
              final_outputs;
              initial_values;
              input_bindings;
              input_checks;
              output_bindings;
            };
          derivation_digest = [];
        };
      keys;
    }
