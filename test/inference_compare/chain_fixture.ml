(* A generic builder: from a description of the left graph and its unit
   grouping it derives the right graph (same structure, other names and
   operator vocabulary) and a complete mapping with identity conversions. Every
   case below is then a one-line change to the description. *)

open Inference_compare_schema
module P = Plan_fixtures
module Plan = Inference_compare_plan

type node = {
  name : string;
  op : string;
  right_op : string;
  operands : string list;
  results : string list;
  assertion : bool;  (** A metadata assertion: no results, no arithmetic. *)
}

type t = {
  spec : Tensor_spec.t;
  left_dialect : string;
  right_dialect : string;
  nodes : node list;
  sources : string list;  (** The first is the user input, the rest constants. *)
  finals : string list;
  units : (string * string list) list;
  pairs : (string * string) list;  (** (unit, output tensor) declared pairs. *)
}

let node ?right_op ?(assertion = false) name op operands results =
  { name; op; right_op = Option.value right_op ~default:op; operands; results; assertion }

let left_key = P.graph_key "aten" 'e'
let right_key = P.graph_key "native" 'f'
let rn s = s ^ "_r"

let op_ref dialect op : Operator_ref.t = P.operator dialect op

let graph key dialect ~rename ~op_of (t : t) : Graph_desc.t =
  let spec = t.spec in
  let nm n = if rename then rn n else n in
  let produced = List.concat_map (fun n -> n.results) t.nodes in
  let nodes =
    List.map
      (fun n ->
        {
          Node_desc.node = P.node key (nm n.name);
          operator = op_ref dialect (op_of n);
          operands = List.map (fun o -> P.tensor key (nm o)) n.operands;
          results = List.map (fun r -> P.tensor key (nm r)) n.results;
          control_predecessors = [];
          effect = (if n.assertion then Metadata_assertion else Pure);
        })
      t.nodes
  in
  let sources =
    List.mapi
      (fun i s ->
        P.source key (nm s)
          (if i = 0 then P.user s else P.constant s)
          spec)
      t.sources
  in
  {
    key;
    nodes;
    sources;
    final_outputs = List.map (fun f -> P.tensor key (nm f)) t.finals;
    output_tree = Tuple (List.map (fun f -> Graph_desc.Output_tree.Tensor (P.tensor key (nm f))) t.finals);
    tensor_specs = List.map (fun p -> (P.tensor key (nm p), spec)) produced;
  }

let graphs t =
  ( graph left_key t.left_dialect ~rename:false ~op_of:(fun n -> n.op) t,
    graph right_key t.right_dialect ~rename:true ~op_of:(fun n -> n.right_op) t )

(* Deterministic pinned inputs of a synthetic chain: the first source is the
   user input [1; 2; 3]; the k-th constant is [10k; 20k; 30k]. *)
let initial_ints (t : t) name =
  let rec index i = function
    | [] -> 0
    | s :: rest -> if s = name then i else index (i + 1) rest
  in
  match index 0 t.sources with
  | 0 -> [ 1L; 2L; 3L ]
  | k ->
      let k = Int64.of_int k in
      [ Int64.mul 10L k; Int64.mul 20L k; Int64.mul 30L k ]

let le_bytes ints =
  let b = Bytes.create (8 * List.length ints) in
  List.iteri (fun i v -> Bytes.set_int64_le b (i * 8) v) ints;
  Bytes.to_string b

let is_i64 (t : t) = Id.Codec.to_string t.spec.dtype.codec = "i64"

let initial_binding (t : t) name =
  if not (is_i64 t) then P.payload name '7'
  else
    let digest =
      match
        Inference_compare_codecs.Payload_hasher.digest_string
          Inference_compare_sha256.registry Inference_compare_sha256.algorithm
          (le_bytes (initial_ints t name))
      with
      | Ok d -> d
      | Error _ -> failwith "initial digest"
    in
    {
      Initial_binding.key = P.ckey name;
      payload_id = P.text (Id.Payload.of_string ("case/" ^ name));
      payload_digest = digest;
    }

let mapping (t : t) : Mapping_artifact.t =
  let spec = t.spec in
  let l = left_key and r = right_key in
  let produced_in names =
    List.concat_map (fun n -> if List.mem n.name names then n.results else []) t.nodes
  in
  let consumed_outside names tensor =
    List.exists (fun n -> (not (List.mem n.name names)) && List.mem tensor n.operands) t.nodes
  in
  let dedupe l =
    List.fold_left (fun acc x -> if List.mem x acc then acc else acc @ [ x ]) [] l
  in
  let unit_info (u, names) =
    let inside = produced_in names in
    let members = List.filter (fun n -> List.mem n.name names) t.nodes in
    let inputs =
      dedupe (List.concat_map (fun n -> List.filter (fun o -> not (List.mem o inside)) n.operands) members)
    in
    let outputs =
      List.filter (fun o -> List.mem o t.finals || consumed_outside names o) inside
    in
    (u, names, inputs, outputs)
  in
  let infos = List.map unit_info t.units in
  let side_tensor side name = match (side : Side.t) with Left -> P.tensor l name | Right -> P.tensor r (rn name) in
  let side_node side name = match (side : Side.t) with Left -> P.node l name | Right -> P.node r (rn name) in
  let cluster side names ins outs : Cluster.t =
    {
      nodes = List.map (side_node side) names;
      inputs = List.map (side_tensor side) ins;
      outputs = List.map (side_tensor side) outs;
    }
  in
  let pair_for (u, tensor) =
    ignore u;
    {
      Member_pair.pair_id = P.text (Id.Member_pair.of_string ("pair_" ^ tensor));
      comparison_id = P.comparison ("cmp_pair_" ^ tensor);
      left = P.tensor l tensor;
      right = P.tensor r (rn tensor);
      claim = Identical;
      left_to_comparison = [];
      right_to_comparison = [];
      comparison_spec = spec;
    }
  in
  let units =
    List.map
      (fun (u, names, ins, outs) ->
        {
          Matched_unit.unit_id = P.uid u;
          left = cluster Left names ins outs;
          right = cluster Right names ins outs;
          output_pairs =
            List.filter_map
              (fun (pu, tensor) ->
                if pu = u then Some (pair_for (pu, tensor)).pair_id else None)
              t.pairs;
        })
      infos
  in
  let all_outputs = List.concat_map (fun (u, _, _, outs) -> List.map (fun o -> (u, o)) outs) infos in
  let id prefix u side name = Printf.sprintf "%s_%s_%s_%s" prefix u (Side.to_string side) name in
  let sides = [ Side.Left; Side.Right ] in
  let input_bindings =
    List.concat_map
      (fun (u, _, ins, _) ->
        List.concat_map
          (fun side ->
            List.map
              (fun i ->
                P.binding u side (side_tensor side i)
                  (P.identity_from_canonical (id "b" u side i) i spec))
              ins)
          sides)
      infos
  in
  let input_checks =
    List.concat_map
      (fun (u, _, ins, _) ->
        List.concat_map
          (fun side ->
            List.map
              (fun i ->
                P.check u side i
                  (P.identity_from_boundary (id "k" u side i) (side_tensor side i) spec))
              ins)
          sides)
      infos
  in
  let output_bindings =
    List.map
      (fun (u, o) ->
        {
          Output_binding.unit = P.uid u;
          key = P.ckey o;
          from_left_outputs = P.identity_from_boundary (id "ol" u Left o) (side_tensor Left o) spec;
          from_right_outputs = P.identity_from_boundary (id "or" u Right o) (side_tensor Right o) spec;
          comparison_id = P.comparison ("cmp_" ^ o);
        })
      all_outputs
  in
  let keys = dedupe (t.sources @ List.map snd all_outputs) in
  {
    schema_version = V1;
    left_graph = l;
    right_graph = r;
    units;
    tensor_relations =
      List.map
        (fun p ->
          let m = pair_for p in
          {
            Tensor_relation.relation_id = P.text (Id.Tensor_relation.of_string ("rel_" ^ snd p));
            left_members = [ m.left ];
            right_members = [ m.right ];
            member_pairs = [ m ];
            routes = [];
          })
        t.pairs;
    dispositions = [];
    frontier =
      {
        canonical_values = List.map (fun k -> { Canonical_spec.key = P.ckey k; spec }) keys;
        final_outputs =
          List.mapi
            (fun i f ->
              {
                Final_output.occurrence = P.text (Id.Output_occurrence.of_string (Printf.sprintf "out%d" i));
                canonical = P.ckey f;
                left = P.tensor l f;
                right = P.tensor r (rn f);
              })
            t.finals;
        initial_values = List.map (fun s -> initial_binding t s) t.sources;
        input_bindings;
        input_checks;
        output_bindings;
      };
    derivation_digest = [];
  }

let structural ?(limits = P.limits) (t : t) =
  let left, right = graphs t in
  match Plan.Structural.check ~limits ~options:P.options ~left ~right (mapping t) with
  | Ok s -> s
  | Error e ->
      Format.kasprintf failwith "chain fixture invalid: %a" P.pp_error e

(* x -matmul(w)-> m;  m,x -add-> a;  a -softmax-> s;  s,w -matmul-> o;
   o -reshape-> r -sum-> t;  m,m -mul-> mm.  Finals: t and mm. *)
let attention =
  {
    spec = P.m23;
    left_dialect = "aten";
    right_dialect = "native";
    nodes =
      [
        node "n_mm" "matmul" [ "x"; "w" ] [ "m" ] ~right_op:"gemm";
        node "n_add" "add" [ "m"; "x" ] [ "a" ];
        node "n_sm" "softmax" [ "a" ] [ "s" ] ~right_op:"softmax_part";
        node "n_att" "matmul" [ "s"; "w" ] [ "o" ] ~right_op:"gemm";
        node "n_rs" "reshape" [ "o" ] [ "r" ];
        node "n_sum" "sum" [ "r" ] [ "t" ];
        node "n_mul" "mul" [ "m"; "m" ] [ "mm" ];
      ];
    sources = [ "x"; "w" ];
    finals = [ "t"; "mm" ];
    units =
      [
        ("u_mm", [ "n_mm" ]);
        ("u_add", [ "n_add" ]);
        ("u_sm", [ "n_sm" ]);
        ("u_att", [ "n_att" ]);
        ("u_tail", [ "n_rs"; "n_sum" ]);
        ("u_mul", [ "n_mul" ]);
      ];
    pairs = [];
  }

(* Synthetic-dialect graphs over int64 vectors, executable by the synthetic
   runners. *)
let i64_spec =
  {
    P.m23 with
    dtype = { codec = Inference_compare_codecs.Element_codec.i64; parameters_digest = None };
    logical_shape = [ 3L ];
  }

let synthetic nodes ~sources ~finals ~units =
  {
    spec = i64_spec;
    left_dialect = "synthetic";
    right_dialect = "synthetic";
    nodes;
    sources;
    finals;
    units;
    pairs = [];
  }

(* The worked U1/U2 example in exact integers: U1 adds 1 on the left and 2 on
   the right (the right side's deviation is injected by the runner), U2
   multiplies by 100. *)
let u1u2 =
  synthetic
    [ node "n1" "addc:1" [ "x" ] [ "h" ]; node "n2" "mulc:100" [ "h" ] [ "y" ] ]
    ~sources:[ "x" ] ~finals:[ "y" ]
    ~units:[ ("U1", [ "n1" ]); ("U2", [ "n2" ]) ]
