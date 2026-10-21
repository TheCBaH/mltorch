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

type certification = {
  local_none : bool;
      (** No required element rule on the local comparisons: they cannot fail.
      *)
  reference_right : bool;
  declared_fidelity : bool;
  end_to_end : bool;
  required_acceptance : bool;
  peers : bool;
  unfrozen : bool;
}

let default_certification =
  {
    local_none = false;
    reference_right = false;
    declared_fidelity = false;
    end_to_end = false;
    required_acceptance = false;
    peers = false;
    unfrozen = false;
  }

type options = {
  retention : Resource_limits.retention;
  required : bool;
  disk_bytes : int64;
  retain_unit : string option;
  donor_right : bool;
  certify : certification option;
}

let default_options =
  {
    retention = No_retention;
    required = false;
    disk_bytes = 1_000_000L;
    retain_unit = None;
    donor_right = false;
    certify = None;
  }

let generate ~dir ~scenario (o : options) =
  (try Unix.mkdir dir 0o755 with Unix.Unix_error (Unix.EEXIST, _, _) -> ());
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
  let local_policy : Comparison_policy.t =
    match o.certify with
    | Some { local_none = true; _ } ->
        { T.Coord_fixture.exact_policy with element = None }
    | _ -> T.Coord_fixture.exact_policy
  in
  let selection : Policy_selection.t =
    { default = Some local_policy; explicit = []; rules = [] }
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
    (json (H.Experiment_file.encode experiment));
  match o.certify with
  | None -> ()
  | Some c ->
      let hashers = Inference_compare_sha256.registry
      and algorithm = Inference_compare_sha256.algorithm in
      let digest mk text =
        match
          Inference_compare_codecs.Payload_hasher.digest_string hashers
            algorithm text
        with
        | Ok d -> Result.get_ok (mk algorithm ~hex:(Digest.Payload.hex d))
        | Error _ -> failwith "digest"
      in
      let occurrence =
        T.Plan_fixtures.text (Id.Output_occurrence.of_string "out0")
      in
      let contract : Computation_contract.t =
        {
          reference =
            (if c.peers then Peers
             else if c.reference_right then Right_reference
             else Left_reference);
          input_domain =
            {
              description =
                T.Plan_fixtures.text (Prose.Description.of_string "the case");
              specification_digest = digest Digest.Input_domain.make "case";
            };
          local_policy_selection = selection;
          final_policies = [ (occurrence, T.Coord_fixture.exact_policy) ];
          allowed_conversion_digest = [];
          required_full_graph_acceptance = c.required_acceptance;
        }
      in
      write
        (Filename.concat dir "contract.json")
        (json (Computation_contract.encode contract));
      let workflow : H.Workflow_file.t =
        {
          experiment = "experiment.json";
          contract = "contract.json";
          workflow =
            {
              reference_side = (if c.reference_right then Right else Left);
              tested_side = (if c.reference_right then Left else Right);
              contract_digest =
                digest Digest.Contract.make
                  (Computation_contract.preimage contract);
              cases = [ T.Plan_fixtures.text (Id.Case.of_string "case0") ];
              partition_fidelity =
                (if c.declared_fidelity then Declared_contract
                 else Bitwise_required);
              end_to_end_reference_check = c.end_to_end;
            };
          frozen_outputs =
            (if c.required_acceptance && not c.unfrozen then
               [ { H.Workflow_file.occurrence; file = "frozen.bin" } ]
             else []);
        }
      in
      write
        (Filename.concat dir "validation.json")
        (json (H.Workflow_file.encode workflow))
