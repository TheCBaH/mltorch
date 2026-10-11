open Inference_compare_schema
module Plan = Inference_compare_plan
module Core = Inference_compare_core
module Ex = Inference_compare_extract
module Pol = Inference_compare_policy
module M = Inference_compare_metrics
module Codecs = Inference_compare_codecs
module L = Codecs.Logical_value

let ( let* ) = Result.bind
let hashers = Session.hashers
let algorithm = Inference_compare_sha256.algorithm

let limits : Resource_limits.t =
  {
    max_live_bytes = 1_000_000_000L;
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
    max_selector_nodes = 1000L;
    max_selector_depth = 16L;
    max_path_length = 8L;
    max_retained_witnesses = 100L;
  }

let site_policy ~tolerance (site : Comparison_site.t) : Comparison_policy.t =
  let floating arithmetic : Element_policy.t =
    Floating
      {
        atol = tolerance;
        rtol = tolerance;
        scale = Symmetric_max_scale;
        arithmetic;
        nan = Reject;
        infinity = Same_sign;
        signed_zero = Equal;
        required = Tolerance;
      }
  in
  (* A codec the registry does not know has no element rule; validation then
     refuses the site by name rather than this guessing one. *)
  let element =
    match
      Codecs.Element_codec.Registry.find Session.codecs site.spec.dtype.codec
    with
    | None -> None
    | Some c -> (
        match c.element with
        | Boolean -> Some Element_policy.Exact_boolean
        | Float_binary32 -> Some (floating Binary32)
        | Float_binary64 -> Some (floating Binary64)
        | Signed_integer | Unsigned_integer -> Some Element_policy.Exact_integer
        )
  in
  {
    element;
    statistics = { direction = Right_minus_left; variance = Population };
    relative_scale = Symmetric_max_scale;
    tensor_metrics = [];
  }

let selection ~tolerance sites : Policy_selection.t =
  {
    default = None;
    explicit =
      List.map
        (fun (s : Comparison_site.t) ->
          (s.comparison, site_policy ~tolerance s))
        sites;
    rules = [];
  }

type outcome = {
  result : Core.Coordinator.result;
  ctx : L.Context.t;
  extracted : int;
}

let policy_error e = Session.Policy (Err.Error.kind e)

let run ~on_event ~donor ~tolerance ~mapping ~left_graph ~right_graph ~initial
    ~node_attributes ~left_endpoint ~right_endpoint =
  let* structural =
    Result.map_error
      (fun e -> Session.Structure (Err.Error.kind e))
      (Plan.Structural.check ~limits ~options:Session.options ~left:left_graph
         ~right:right_graph mapping)
  in
  let* extractions =
    Result.map_error
      (fun e -> Session.Extraction (Err.Error.kind e))
      (Ex.Extraction.extract_all ~node_attributes ~hashers ~algorithm structural)
  in
  let* sites =
    Result.map_error policy_error (Pol.Comparison_sites.build structural)
  in
  let selection = selection ~tolerance sites in
  let ctx : Pol.Selector_eval.ctx =
    {
      left = Plan.Structural.left structural;
      right = Plan.Structural.right structural;
      partition = Plan.Structural.partition structural;
      limits;
    }
  in
  let placeholder = Pol.Seal.placeholder algorithm in
  let* resolved =
    Result.map_error policy_error
      (Pol.Rule_resolve.resolve ~ctx ~codecs:Session.codecs
         ~env:M.Tensor_metrics.empty_env ~placeholder_digest:placeholder sites
         selection)
  in
  let* bindings =
    Result.map_error
      (fun e ->
        Session.Sealing
          (Format.asprintf "%a" Pol.Seal.pp_error (Err.Error.kind e)))
      (Pol.Seal.seal ~hashers ~algorithm ~selection resolved.bindings)
  in
  let* sealed =
    Result.map_error
      (fun e ->
        Session.Sealing
          (Format.asprintf "%a" Core.Sealed_plan.pp_error (Err.Error.kind e)))
      (Core.Sealed_plan.seal ~hashers ~algorithm structural extractions bindings)
  in
  let run_ctx =
    {
      L.Context.codecs = Session.codecs;
      hashers;
      algorithm;
      ledger = Codecs.Owned_blob.Ledger.create ~limit:limits.max_live_bytes ();
    }
  in
  let cfg : Core.Coordinator.config =
    {
      ctx = run_ctx;
      codecs = Session.codecs;
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
          compare_policy_digest = placeholder;
        };
      initial;
      expected_identities = (None, None);
      emit =
        (fun e ->
          on_event e;
          Ok ());
      retain = (fun _ -> Ok ());
      mode = Full;
    }
  in
  let result =
    Core.Coordinator.run cfg sealed ~left:left_endpoint ~right:right_endpoint
  in
  Ok { result; ctx = run_ctx; extracted = List.length extractions }

let error_exit_code = 2
let exit_code = Run.exit_code
