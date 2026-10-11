open Inference_compare_schema
module Codecs = Inference_compare_codecs
module H = Inference_compare_host
module P = Inference_compare_pt2
module L = Codecs.Logical_value

let ( let* ) = Result.bind
let hashers = Inference_compare_sha256.registry
let algorithm = Inference_compare_sha256.algorithm
let text r = Result.map_error (fun _ -> "invalid identifier") r

let io r =
  Result.map_error (fun e -> Format.asprintf "%a" H.File_io.pp_error e) r

(* The exporter's samples are single images; the model takes a batch. *)
let with_batch (t : Pt2_tensor.t) =
  let stride = Pt2_tensor.numel t in
  { t with Pt2_tensor.sizes = 1 :: t.sizes; strides = stride :: t.strides }

let limits : Resource_limits.t =
  {
    max_live_bytes = 2_000_000_000L;
    max_conversion_scratch_bytes = 1_000_000_000L;
    max_snapshot_bytes = 1_000_000_000L;
    max_retained_disk_bytes = 0L;
    max_mismatch_samples = 8L;
    retention = No_retention;
    retention_requirement = Best_effort;
    verified_spill = false;
    max_nodes = 1_000_000L;
    max_edges = 10_000_000L;
    max_units = 1_000_000L;
    max_recipe_nodes = 10_000_000L;
    max_selector_nodes = 1_000L;
    max_selector_depth = 16L;
    max_path_length = 8L;
    max_retained_witnesses = 100_000L;
  }

(* Every floating site is compared in binary32 within [tolerance]; the policy
   is a default, so a model with an integer tensor needs its own selection. *)
let selection ~tolerance : Policy_selection.t =
  let policy : Comparison_policy.t =
    {
      element =
        Some
          (Floating
             {
               atol = tolerance;
               rtol = tolerance;
               scale = Symmetric_max_scale;
               arithmetic = Binary32;
               nan = Reject;
               infinity = Same_sign;
               signed_zero = Equal;
               required = Tolerance;
             });
      statistics = { direction = Right_minus_left; variance = Population };
      relative_scale = Symmetric_max_scale;
      tensor_metrics = [];
    }
  in
  { default = Some policy; explicit = []; rules = [] }

let write path contents = io (H.File_io.write_atomic path contents)
let json j = Wire_text.to_string j ^ "\n"

let absolute path =
  if Filename.is_relative path then Filename.concat (Sys.getcwd ()) path
  else path

let sample_of ~inputs ~index =
  match Pt2_archive.load_pt_tensor_map inputs with
  | Error e ->
      Error
        (Format.asprintf "%s: %a" inputs Pt2_archive.pp_error (Err.Error.kind e))
  | Ok samples -> (
      match List.nth_opt samples index with
      | Some (_, t) -> Ok (with_batch t)
      | None -> Error (Printf.sprintf "%s has no sample %d" inputs index))

type right = Native | Native4d
type certify = { tested : Side.t; fidelity : Partition_fidelity.t }

let write_experiment ?(right = Native) ?certify ~dir ~model ~inputs ~index
    ~tolerance () =
  let model = absolute model in
  let* sample = sample_of ~inputs ~index in
  let* aten = P.Aten_model.of_file ~hashers ~algorithm model in
  let* input_name =
    match
      List.find_opt
        (fun (s : P.Aten_model.source) -> s.captured = None)
        aten.sources
    with
    | Some s -> Ok s.name
    | None -> Error "the model has no user input"
  in
  let named_inputs = [ (input_name, sample) ] in
  let* initial, left_graph, right_graph, mapping =
    match right with
    | Native ->
        let* pair =
          P.Pair.load ~hashers ~algorithm ~path:model ~inputs:named_inputs
        in
        Ok
          ( P.Pair.initial pair,
            P.Pair.left_graph pair,
            P.Pair.right_graph pair,
            P.Pair.mapping pair )
    | Native4d ->
        let* pair =
          P.Pair4d.load ~hashers ~algorithm ~path:model ~inputs:named_inputs
        in
        Ok
          ( P.Pair4d.initial pair,
            P.Pair4d.left_graph pair,
            P.Pair4d.right_graph pair,
            P.Pair4d.mapping pair )
  in
  let* contents =
    Result.map_error
      (fun e -> Format.asprintf "%a" H.File_io.pp_error e)
      (H.File_io.read model)
  in
  let* pin = P.Route.file_pin ~hashers contents in
  let* () = io (H.File_io.mkdir_p (Filename.concat dir "case")) in
  let* case =
    let rec go i acc = function
      | [] -> Ok (List.rev acc)
      | (key, v) :: rest ->
          let file = Printf.sprintf "case/%04d.bin" i in
          let* () = write (Filename.concat dir file) (L.contents v) in
          let* payload_id =
            text (Id.Payload.of_string (Printf.sprintf "case/%d" i))
          in
          go (i + 1) ({ H.Experiment_file.key; payload_id; file } :: acc) rest
    in
    go 0 [] initial
  in
  let* experiment_id = text (Id.Experiment.of_string "pt2-pair") in
  let* case_id = text (Id.Case.of_string (Printf.sprintf "sample-%d" index)) in
  let experiment : H.Experiment_file.t =
    {
      experiment_id;
      case_id;
      digest_algorithm = algorithm;
      left =
        { graph = "left.graph.json"; engine = Aten { model; sha256 = pin } };
      right =
        {
          graph = "right.graph.json";
          engine =
            (match right with
            | Native -> Native { model; sha256 = pin }
            | Native4d -> Native4d { model; sha256 = pin });
        };
      mapping = "mapping.json";
      selection = "selection.json";
      case;
      default_donor = Left;
      overrides = [];
      limits;
      retain_units = [];
    }
  in
  let at = Filename.concat dir in
  let* () =
    write (at "left.graph.json") (json (Graph_desc.encode left_graph))
  in
  let* () =
    write (at "right.graph.json") (json (Graph_desc.encode right_graph))
  in
  let* () =
    write (at "mapping.json") (json (Mapping_artifact.encode mapping))
  in
  let* () =
    write (at "selection.json")
      (json (Policy_selection.encode (selection ~tolerance)))
  in
  let* () =
    write (at "experiment.json") (json (H.Experiment_file.encode experiment))
  in
  match certify with
  | None -> Ok ()
  | Some c ->
      let policy =
        match (selection ~tolerance).default with
        | Some p -> p
        | None -> invalid_arg "Pt2_export.selection"
      in
      let* inputs_text =
        Result.map_error
          (fun e -> Format.asprintf "%a" H.File_io.pp_error e)
          (H.File_io.read inputs)
      in
      let* domain_digest =
        match
          Codecs.Payload_hasher.digest_string hashers algorithm inputs_text
        with
        | Ok d ->
            text
              (Digest.Input_domain.make algorithm ~hex:(Digest.Payload.hex d))
        | Error _ -> Error "digest backend unavailable"
      in
      let* description =
        text (Prose.Description.of_string "the exported sample")
      in
      let* finals =
        Ok
          (List.map
             (fun (f : Final_output.t) -> (f.occurrence, policy))
             mapping.frontier.final_outputs)
      in
      let contract : Computation_contract.t =
        {
          reference =
            (match c.tested with
            | Right -> Left_reference
            | Left -> Right_reference);
          input_domain = { description; specification_digest = domain_digest };
          local_policy_selection = selection ~tolerance;
          final_policies = finals;
          allowed_conversion_digest = [];
          required_full_graph_acceptance = false;
        }
      in
      let* contract_digest =
        match
          Codecs.Payload_hasher.digest_string hashers algorithm
            (Computation_contract.preimage contract)
        with
        | Ok d ->
            text (Digest.Contract.make algorithm ~hex:(Digest.Payload.hex d))
        | Error _ -> Error "digest backend unavailable"
      in
      let* () =
        write (at "contract.json") (json (Computation_contract.encode contract))
      in
      let workflow : H.Workflow_file.t =
        {
          experiment = "experiment.json";
          contract = "contract.json";
          workflow =
            {
              reference_side =
                (match c.tested with Right -> Left | Left -> Right);
              tested_side = c.tested;
              contract_digest;
              cases = [ case_id ];
              partition_fidelity = c.fidelity;
              end_to_end_reference_check = true;
            };
          frozen_outputs = [];
        }
      in
      write (at "validation.json") (json (H.Workflow_file.encode workflow))
