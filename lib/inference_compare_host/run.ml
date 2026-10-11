open Inference_compare_schema
module Core = Inference_compare_core
module Codecs = Inference_compare_codecs
module L = Codecs.Logical_value

type outcome = {
  result : Core.Coordinator.result;
  directory : string;
  finalized : (unit, Evidence.error) result;
}

let default_run_id (p : Session.prepared) =
  let e = p.validated.loaded.experiment in
  let plan = Core.Sealed_plan.plan p.sealed in
  let hex = Digest.Plan.hex plan.plan_digest in
  match
    Id.Run.of_string
      (Id.Experiment.to_string e.experiment_id ^ "-" ^ String.sub hex 0 12)
  with
  | Ok r -> r
  | Error _ -> invalid_arg "Run.default_run_id"

let config (p : Session.prepared) ~emit ~retain ~mode ~ledger_limit :
    Core.Coordinator.config =
  let l = p.validated.loaded in
  let e = l.experiment in
  {
    ctx =
      {
        L.Context.codecs = Codecs.Element_codec.Registry.builtin;
        hashers = Session.hashers;
        algorithm = e.digest_algorithm;
        ledger = Codecs.Owned_blob.Ledger.create ~limit:ledger_limit ();
      };
    codecs = Codecs.Element_codec.Registry.builtin;
    operations = Codecs.Conversion_registry.empty;
    metrics = Inference_compare_metrics.Tensor_metrics.empty_env;
    hashers = Session.hashers;
    run_id = Id.Run.of_string "run" |> Result.get_ok;
    case_id = e.case_id;
    experiment =
      {
        experiment_id = e.experiment_id;
        default_donor = e.default_donor;
        overrides = e.overrides;
        compare_policy_digest =
          Inference_compare_policy.Seal.placeholder e.digest_algorithm;
      };
    initial = l.initial;
    expected_identities = (Some p.left_identity, Some p.right_identity);
    emit;
    retain;
    mode;
  }

let run ~root ?run_id ?(observe = fun _ -> ()) ?(keep_finals = false)
    (p : Session.prepared) =
  let run_id = match run_id with Some r -> r | None -> default_run_id p in
  match Evidence.create ~root ~hashers:Session.hashers p ~run_id with
  | Error e -> Error e
  | Ok ev ->
      let limit = p.validated.loaded.experiment.limits.max_live_bytes in
      let cfg =
        {
          (config p ~emit:(Evidence.emit ev)
             ~retain:(fun s ->
               observe s;
               Evidence.retain ev s)
             ~mode:Full ~ledger_limit:limit)
          with
          run_id;
        }
      in
      let result =
        try
          Core.Coordinator.run cfg p.sealed ~left:p.left_endpoint
            ~right:p.right_endpoint
        with exn ->
          Evidence.abandon ev;
          raise exn
      in
      let finalized = Evidence.finalize ev result in
      if not keep_finals then List.iter (fun (_, v) -> L.drop v) result.finals;
      Ok { result; directory = Evidence.directory ev; finalized }

let exit_code (r : Core.Coordinator.result) =
  match (r.completion.execution, r.failure) with
  | Complete, _ -> (
      match r.completion.numerical with
      | Agreement -> 0
      | Disagreement -> 1
      | Not_established -> 4)
  | Interrupted, _ -> 4
  | (Failed | Refused), Some { phase = Planning; _ } -> 2
  | (Failed | Refused), _ -> 3
