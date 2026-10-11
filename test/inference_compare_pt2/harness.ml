(* Plan, extract, resolve policy, seal and run one pair through the shared
   coordinator, printing what it finds. Shared by the pair drivers. *)
open Inference_compare_schema
module Plan = Inference_compare_plan
module Core = Inference_compare_core
module Ex = Inference_compare_extract
module Pol = Inference_compare_policy
module M = Inference_compare_metrics
module Codecs = Inference_compare_codecs
module L = Codecs.Logical_value

let hashers = Inference_compare_sha256.registry
let algorithm = Inference_compare_sha256.algorithm

let run ~donor ~tol ~mapping ~left_graph ~right_graph ~initial ~node_attributes
    ~left_endpoint ~right_endpoint =
  let limits =
    {
      Inference_compare_test.Plan_fixtures.limits with
      max_nodes = 1_000_000L;
      max_edges = 10_000_000L;
      max_units = 1_000_000L;
      max_recipe_nodes = 10_000_000L;
      max_live_bytes = 1_000_000_000L;
      max_selector_nodes = 1000L;
    }
  in
  let structural =
    match
      Plan.Structural.check ~limits
        ~options:Inference_compare_test.Plan_fixtures.options ~left:left_graph
        ~right:right_graph mapping
    with
    | Ok s -> s
    | Error e ->
        Format.eprintf "structural: %a@."
          (Err.Error.pp_kind Plan.Plan_error.pp)
          e;
        exit 1
  in
  let extractions =
    match
      Ex.Extraction.extract_all ~node_attributes ~hashers ~algorithm structural
    with
    | Ok l -> l
    | Error e ->
        Format.eprintf "extract: %a@." (Err.Error.pp_kind Ex.Extract_error.pp) e;
        exit 1
  in
  Printf.printf "extracted %d graphs\n%!" (List.length extractions);
  let policy : Comparison_policy.t =
    {
      element =
        Some
          (Floating
             {
               atol = tol;
               rtol = tol;
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
  let selection : Policy_selection.t =
    { default = Some policy; explicit = []; rules = [] }
  in
  let sites =
    match Pol.Comparison_sites.build structural with
    | Ok s -> s
    | Error _ -> failwith "sites"
  in
  let ctx : Pol.Selector_eval.ctx =
    {
      left = Plan.Structural.left structural;
      right = Plan.Structural.right structural;
      partition = Plan.Structural.partition structural;
      limits;
    }
  in
  let resolved =
    match
      Pol.Rule_resolve.resolve ~ctx
        ~codecs:Codecs.Element_codec.Registry.builtin
        ~env:M.Tensor_metrics.empty_env
        ~placeholder_digest:(Pol.Seal.placeholder algorithm)
        sites selection
    with
    | Ok r -> r
    | Error e ->
        Format.eprintf "policy: %a@." (Err.Error.pp_kind Pol.Policy_error.pp) e;
        exit 1
  in
  let bindings =
    match Pol.Seal.seal ~hashers ~algorithm ~selection resolved.bindings with
    | Ok b -> b
    | Error _ -> failwith "seal"
  in
  let sealed =
    match
      Core.Sealed_plan.seal ~hashers ~algorithm structural extractions bindings
    with
    | Ok s -> s
    | Error e ->
        Format.eprintf "seal: %a@."
          (Err.Error.pp_kind Core.Sealed_plan.pp_error)
          e;
        exit 1
  in
  let t0 = Unix.gettimeofday () in
  let run_ctx =
    {
      L.Context.codecs = Codecs.Element_codec.Registry.builtin;
      hashers;
      algorithm;
      ledger = Codecs.Owned_blob.Ledger.create ~limit:1_000_000_000L ();
    }
  in
  let cfg : Core.Coordinator.config =
    {
      ctx = run_ctx;
      codecs = Codecs.Element_codec.Registry.builtin;
      operations = Codecs.Conversion_registry.empty;
      metrics = M.Tensor_metrics.empty_env;
      hashers;
      run_id = Result.get_ok (Id.Run.of_string "r");
      case_id = Result.get_ok (Id.Case.of_string "c");
      experiment =
        {
          experiment_id = Result.get_ok (Id.Experiment.of_string "e");
          default_donor = donor;
          overrides = [];
          compare_policy_digest = Pol.Seal.placeholder algorithm;
        };
      initial;
      expected_identities = (None, None);
      emit = (fun _ -> Ok ());
      retain = (fun _ -> Ok ());
      mode = Full;
    }
  in
  let r =
    Core.Coordinator.run cfg sealed ~left:left_endpoint ~right:right_endpoint
  in
  if Sys.getenv_opt "TIMING" <> None then
    Printf.eprintf "run took %.1fs\n" (Unix.gettimeofday () -. t0);
  Printf.printf "execution=%s numerical=%s executed=%d units\n"
    (match r.completion.execution with
    | Complete -> "complete"
    | Failed -> "failed"
    | Refused -> "refused"
    | Interrupted -> "interrupted")
    (match r.completion.numerical with
    | Agreement -> "agreement"
    | Disagreement -> "disagreement"
    | Not_established -> "not_established")
    (List.length r.executed);
  (match r.failure with
  | Some f ->
      Format.printf "failure: %s phase=%s unit=%s text=%s@."
        (match Failure.encode_code f.code with
        | Jsont.String (s, _) -> s
        | _ -> "?")
        (match Phase.encode f.phase with Jsont.String (s, _) -> s | _ -> "?")
        (match f.unit with Some u -> Id.Unit.to_string u | None -> "-")
        (match f.actual with
        | Some d -> Prose.Diagnostic_text.to_string d
        | None -> "-")
  | None -> ());
  let verdicts =
    List.map (fun (c : Comparison_result.t) -> c.verdict) r.comparisons
  in
  let count v = List.length (List.filter (( = ) v) verdicts) in
  Printf.printf "comparisons: %d pass, %d mismatch, %d other\n" (count Pass)
    (count Mismatch)
    (List.length verdicts - count Pass - count Mismatch);
  List.iter
    (fun (c : Comparison_result.t) ->
      if c.verdict <> Pass then
        Format.printf "  %s %s max_abs=%s violations=%s@."
          (Id.Comparison.to_string c.comparison_id)
          (match c.verdict with
          | Mismatch -> "mismatch"
          | Pass -> "pass"
          | Metadata_error -> "metadata"
          | Metric_unavailable -> "unavailable"
          | Unsupported -> "unsupported")
          (match c.element with
          | Some e -> (
              match e.absolute.maximum_finite with
              | Some v -> Printf.sprintf "%.3g" v
              | None -> "-")
          | None -> "-")
          (match Option.bind c.element (fun e -> e.tolerance_violations) with
          | Some v -> Int64.to_string v
          | None -> "-"))
    r.comparisons;

  (r, run_ctx)
