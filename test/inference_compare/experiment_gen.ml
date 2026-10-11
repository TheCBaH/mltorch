(* Writes a complete experiment directory (graphs, mapping, selection, case
   payloads, experiment manifest) from the synthetic chains the unit tests use. *)

open Inference_compare_schema

module T = struct
  module Chain_fixture = Chain_fixture
  module Coord_fixture = Coord_fixture
  module Plan_fixtures = Plan_fixtures
end

module C = T.Chain_fixture
module H = Inference_compare_host
module R = Inference_compare_engines.Synthetic_runner

let residual =
  C.synthetic
    [
      C.node "n1" "add" [ "x"; "w" ] [ "h" ];
      C.node "n2" "mul" [ "h"; "w" ] [ "m" ];
      C.node "n3" "add" [ "m"; "x" ] [ "y" ];
    ]
    ~sources:[ "x"; "w" ] ~finals:[ "y" ]
    ~units:[ ("A", [ "n1" ]); ("B", [ "n2" ]); ("D", [ "n3" ]) ]

let write path text =
  let oc = open_out_bin path in
  output_string oc text;
  close_out oc

let json j = Wire_text.to_string j ^ "\n"

type options = {
  retention : Resource_limits.retention;
  required : bool;
  disk_bytes : int64;
  retain_unit : string option;
  donor_right : bool;
}

let default_options =
  {
    retention = No_retention;
    required = false;
    disk_bytes = 1_000_000L;
    retain_unit = None;
    donor_right = false;
  }

let generate ~dir ~scenario (o : options) =
  (try Unix.mkdir dir 0o755 with _ -> ());
  let chain =
    match scenario with
    | "unmapped" ->
        { residual with units = [ ("A", [ "n1" ]); ("B", [ "n2" ]) ] }
    | _ -> residual
  in
  let left, right = C.graphs chain in
  let mapping = C.mapping chain in
  let default_cfg name = R.default_config name in
  let right_cfg =
    match scenario with
    | "deviate" -> { (default_cfg "right") with offsets = [ ("n1_r", 1L) ] }
    | "corrupt" -> { (default_cfg "right") with corrupt_imports = [ (0, 5L) ] }
    | "refuse" -> { (default_cfg "right") with refuse_ops = [ "mul" ] }
    | "fail" -> { (default_cfg "right") with fail_nodes = [ "n2_r" ] }
    | _ -> default_cfg "right"
  in
  let selection : Policy_selection.t =
    { default = Some T.Coord_fixture.exact_policy; explicit = []; rules = [] }
  in
  let case =
    List.map
      (fun s ->
        let file = s ^ ".bin" in
        write (Filename.concat dir file) (C.le_bytes (C.initial_ints chain s));
        {
          H.Experiment_file.key = T.Plan_fixtures.ckey s;
          payload_id = T.Plan_fixtures.text (Id.Payload.of_string ("case/" ^ s));
          file;
        })
      chain.sources
  in
  let limits =
    {
      T.Plan_fixtures.limits with
      retention = o.retention;
      retention_requirement = (if o.required then Required else Best_effort);
      max_retained_disk_bytes = o.disk_bytes;
    }
  in
  let experiment : H.Experiment_file.t =
    {
      experiment_id = T.Plan_fixtures.text (Id.Experiment.of_string "exp");
      case_id = T.Plan_fixtures.text (Id.Case.of_string "case0");
      digest_algorithm = Inference_compare_sha256.algorithm;
      left =
        { graph = "left.graph.json"; engine = Synthetic (default_cfg "left") };
      right = { graph = "right.graph.json"; engine = Synthetic right_cfg };
      mapping = "mapping.json";
      selection = "selection.json";
      case;
      default_donor = (if o.donor_right then Right else Left);
      overrides = [];
      limits;
      retain_units =
        (match o.retain_unit with
        | Some u -> [ T.Plan_fixtures.uid u ]
        | None -> []);
    }
  in
  write (Filename.concat dir "left.graph.json") (json (Graph_desc.encode left));
  write
    (Filename.concat dir "right.graph.json")
    (json (Graph_desc.encode right));
  write
    (Filename.concat dir "mapping.json")
    (json (Mapping_artifact.encode mapping));
  write
    (Filename.concat dir "selection.json")
    (json (Policy_selection.encode selection));
  write
    (Filename.concat dir "experiment.json")
    (json (H.Experiment_file.encode experiment))
