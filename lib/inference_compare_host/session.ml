(* Loading an experiment from disk and assembling its sealed plan. Nothing
   here runs inference; every failure is a typed value carrying the path or the
   structured planner/policy error that caused it. *)

open Inference_compare_schema
module Codecs = Inference_compare_codecs
module L = Codecs.Logical_value
module Plan = Inference_compare_plan
module Ex = Inference_compare_extract
module Pol = Inference_compare_policy
module M = Inference_compare_metrics
module Core = Inference_compare_core

type error =
  | Io of File_io.error
  | Decode of { path : string; error : Wire.error }
  | Case of { key : Id.Canonical_value.t; reason : string }
  | Structure of Plan.Plan_error.t
  | Extraction of Ex.Extract_error.t
  | Policy of Pol.Policy_error.t
  | Sealing of string
  | Unknown_algorithm of Id.Digest_algorithm.t

let pp_error ppf = function
  | Io e -> File_io.pp_error ppf e
  | Decode { path; error } -> Fmt.pf ppf "%s: %a" path Wire.pp_error error
  | Case { key; reason } ->
      Fmt.pf ppf "case payload %a: %s" Id.Canonical_value.pp key reason
  | Structure e -> Fmt.pf ppf "mapping: %a" Plan.Plan_error.pp e
  | Extraction e -> Fmt.pf ppf "extraction: %a" Ex.Extract_error.pp e
  | Policy e -> Fmt.pf ppf "policy: %a" Pol.Policy_error.pp e
  | Sealing s -> Fmt.pf ppf "sealing: %s" s
  | Unknown_algorithm a ->
      Fmt.pf ppf "unknown digest algorithm %a" Id.Digest_algorithm.pp a

type loaded = {
  dir : string;
  experiment : Experiment_file.t;
  texts : (string * string) list;
      (** Raw text of each input file, keyed by role, for identity pins. *)
  left_graph : Graph_desc.t;
  right_graph : Graph_desc.t;
  mapping : Mapping_artifact.t;
  selection : Policy_selection.t;
  initial_ctx : L.Context.t;
  initial : (Id.Canonical_value.t * L.t) list;
  case_texts : (Id.Canonical_value.t * string) list;
}

let ( let* ) = Result.bind
let read path = Result.map_error (fun e -> Io e) (File_io.read path)

let decode_file path decoder =
  let* text = read path in
  match Wire_text.decode_string decoder text with
  | Ok v -> Ok (v, text)
  | Error e -> Error (Decode { path; error = Err.Error.kind e })

let hashers = Inference_compare_sha256.registry

let load experiment_path =
  let dir = Filename.dirname experiment_path in
  let* experiment, experiment_text =
    decode_file experiment_path Experiment_file.decode
  in
  let at p = File_io.resolve ~base:dir p in
  let* left_graph, left_text =
    decode_file
      (at experiment.Experiment_file.left.Experiment_file.graph)
      Graph_desc.decode
  in
  let* right_graph, right_text =
    decode_file
      (at experiment.Experiment_file.right.Experiment_file.graph)
      Graph_desc.decode
  in
  let* mapping, mapping_text =
    decode_file (at experiment.Experiment_file.mapping) Mapping_artifact.decode
  in
  let* selection, selection_text =
    decode_file
      (at experiment.Experiment_file.selection)
      Policy_selection.decode
  in
  let* () =
    if
      List.exists
        (Id.Digest_algorithm.equal experiment.Experiment_file.digest_algorithm)
        (Codecs.Payload_hasher.Registry.algorithms hashers)
    then Ok ()
    else Error (Unknown_algorithm experiment.Experiment_file.digest_algorithm)
  in
  let initial_ctx =
    {
      L.Context.codecs = Codecs.Element_codec.Registry.builtin;
      hashers;
      algorithm = experiment.Experiment_file.digest_algorithm;
      ledger = Codecs.Owned_blob.Ledger.create ();
    }
  in
  let spec_of key =
    List.find_map
      (fun (c : Canonical_spec.t) ->
        if Id.Canonical_value.equal c.key key then Some c.spec else None)
      mapping.frontier.canonical_values
  in
  let rec payloads acc texts = function
    | [] -> Ok (List.rev acc, List.rev texts)
    | (c : Experiment_file.case_payload) :: rest -> (
        match spec_of c.Experiment_file.key with
        | None ->
            Error
              (Case
                 {
                   key = c.Experiment_file.key;
                   reason = "not a canonical value of the mapping";
                 })
        | Some spec -> (
            let* bytes = read (at c.Experiment_file.file) in
            match L.make initial_ctx spec bytes with
            | Ok v ->
                payloads
                  ((c.Experiment_file.key, v) :: acc)
                  ((c.Experiment_file.key, bytes) :: texts)
                  rest
            | Error e ->
                Error
                  (Case
                     {
                       key = c.Experiment_file.key;
                       reason =
                         Format.asprintf "%a" Codecs.Codec_error.pp
                           (Err.Error.kind e);
                     })))
  in
  let* initial, case_texts = payloads [] [] experiment.Experiment_file.case in
  Ok
    {
      dir;
      experiment;
      texts =
        [
          ("experiment", experiment_text);
          ("left_graph", left_text);
          ("right_graph", right_text);
          ("mapping", mapping_text);
          ("selection", selection_text);
        ];
      left_graph;
      right_graph;
      mapping;
      selection;
      initial_ctx;
      initial;
      case_texts;
    }

type validated = {
  loaded : loaded;
  structural : Plan.Structural.t;
  sites : Comparison_site.t list;
  resolved : Pol.Rule_resolve.result;
}

let codecs = Codecs.Element_codec.Registry.builtin

let options : Plan.Correspondence_check.options =
  { codecs; operations = Codecs.Conversion_registry.empty }

(* Structure and policy, no extraction and no engine: the [validate] command. *)
let validate (l : loaded) =
  let limits = l.experiment.Experiment_file.limits in
  let* structural =
    Result.map_error
      (fun e -> Structure (Err.Error.kind e))
      (Plan.Structural.check ~limits ~options ~left:l.left_graph
         ~right:l.right_graph l.mapping)
  in
  let* sites =
    Result.map_error
      (fun e -> Policy (Err.Error.kind e))
      (Pol.Comparison_sites.build structural)
  in
  let ctx : Pol.Selector_eval.ctx =
    {
      left = Plan.Structural.left structural;
      right = Plan.Structural.right structural;
      partition = Plan.Structural.partition structural;
      limits;
    }
  in
  let* resolved =
    Result.map_error
      (fun e -> Policy (Err.Error.kind e))
      (Pol.Rule_resolve.resolve ~ctx ~codecs ~env:M.Tensor_metrics.empty_env
         ~placeholder_digest:
           (Pol.Seal.placeholder l.experiment.Experiment_file.digest_algorithm)
         sites l.selection)
  in
  let* () =
    Result.map_error
      (fun e -> Policy (Err.Error.kind e))
      (Pol.Donor_check.validate structural
         {
           experiment_id = l.experiment.Experiment_file.experiment_id;
           default_donor = l.experiment.Experiment_file.default_donor;
           overrides = l.experiment.Experiment_file.overrides;
           compare_policy_digest =
             Pol.Seal.placeholder l.experiment.Experiment_file.digest_algorithm;
         })
  in
  Ok { loaded = l; structural; sites; resolved }

type prepared = {
  validated : validated;
  sealed : Core.Sealed_plan.t;
  left_endpoint : Core.Side_exec.endpoint;
  right_endpoint : Core.Side_exec.endpoint;
  left_identity : Engine_identity.t;
  right_identity : Engine_identity.t;
}

(* Extraction, sealing and endpoint instantiation: the [plan] command stops
   here, with no preparation of any engine graph. *)
let prepare (v : validated) =
  let l = v.loaded in
  let algorithm = l.experiment.Experiment_file.digest_algorithm in
  let* extractions =
    Result.map_error
      (fun e -> Extraction (Err.Error.kind e))
      (Ex.Extraction.extract_all ~hashers ~algorithm v.structural)
  in
  let* bindings =
    Result.map_error
      (fun e ->
        Sealing (Format.asprintf "%a" Pol.Seal.pp_error (Err.Error.kind e)))
      (Pol.Seal.seal ~hashers ~algorithm ~selection:l.selection
         v.resolved.bindings)
  in
  let* sealed =
    Result.map_error
      (fun e ->
        Sealing
          (Format.asprintf "%a" Core.Sealed_plan.pp_error (Err.Error.kind e)))
      (Core.Sealed_plan.seal ~hashers ~algorithm v.structural extractions
         bindings)
  in
  let source side =
    Plan.Graph_index.graph
      (match (side : Side.t) with
      | Left -> Plan.Structural.left v.structural
      | Right -> Plan.Structural.right v.structural)
  in
  let left_endpoint, left_identity =
    Engine_spec.instantiate
      l.experiment.Experiment_file.left.Experiment_file.engine
      ~source:(source Left)
  in
  let right_endpoint, right_identity =
    Engine_spec.instantiate
      l.experiment.Experiment_file.right.Experiment_file.engine
      ~source:(source Right)
  in
  Ok
    {
      validated = v;
      sealed;
      left_endpoint;
      right_endpoint;
      left_identity;
      right_identity;
    }
