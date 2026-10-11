(* A harness running the coordinator over a synthetic chain with two
   configurable runners. *)

open Inference_compare_schema
module Plan = Inference_compare_plan
module Ex = Inference_compare_extract
module E = Inference_compare_engines
module R = E.Synthetic_runner
module Core = Inference_compare_core
module Pol = Inference_compare_policy
module M = Inference_compare_metrics
module L = Inference_compare_codecs.Logical_value
module Codecs = Inference_compare_codecs
module P = Plan_fixtures
module C = Chain_fixture

let hashers = Inference_compare_sha256.registry
let algorithm = Inference_compare_sha256.algorithm

let exact_policy : Comparison_policy.t =
  {
    element = Some Exact_integer;
    statistics = { direction = Right_minus_left; variance = Population };
    relative_scale = Right_scale;
    tensor_metrics = [];
  }

let sealed structural ?(policy = exact_policy) () =
  let sites = Result.get_ok (Pol.Comparison_sites.build structural) in
  let ctx : Pol.Selector_eval.ctx =
    {
      left = Plan.Structural.left structural;
      right = Plan.Structural.right structural;
      partition = Plan.Structural.partition structural;
      limits = P.limits;
    }
  in
  let selection : Policy_selection.t =
    { default = Some policy; explicit = []; rules = [] }
  in
  let resolved =
    Result.get_ok
      (Pol.Rule_resolve.resolve ~ctx
         ~codecs:Codecs.Element_codec.Registry.builtin
         ~env:M.Tensor_metrics.empty_env
         ~placeholder_digest:(Pol.Seal.placeholder algorithm)
         sites selection)
  in
  Result.get_ok (Pol.Seal.seal ~hashers ~algorithm ~selection resolved.bindings)

type t = {
  structural : Plan.Structural.t;
  extractions : Ex.Extraction.t list;
  bindings : Policy_bindings.t;
  chain : C.t;
}

let make ?policy chain =
  let structural = C.structural chain in
  let extractions =
    Result.get_ok (Ex.Extraction.extract_all ~hashers ~algorithm structural)
  in
  { structural; extractions; bindings = sealed structural ?policy (); chain }

let endpoint runner source : Core.Side_exec.endpoint =
  let module D =
    (val E.Synthetic_dialect.make ()
        : Ex.Dialect.S with type graph = Graph_desc.t
         and type body = Graph_desc.t)
  in
  Endpoint
    { runner; build = (fun ex -> Ex.Dialect.realize (module D) source ex) }

let initial_values t =
  let ictx = Codec_fixtures.context () in
  List.map
    (fun s ->
      ( P.ckey s,
        Codec_fixtures.integers ictx C.i64_spec (C.initial_ints t.chain s) ))
    t.chain.sources

let result_ints ctx (r : Core.Coordinator.result) =
  List.map
    (fun (occ, v) ->
      (Id.Output_occurrence.to_string occ, Codec_fixtures.to_integers ctx v))
    r.finals

type setup = {
  left_cfg : R.config;
  right_cfg : R.config;
  donor : Side.t;
  overrides : Donor_override.t list;
  emit : Event.t -> (unit, string) result;
  limit : int64 option;
  expected_identities : Engine_identity.t option * Engine_identity.t option;
  initial_override : (Id.Canonical_value.t * L.t) list option;
}

let setup ?(left = R.default_config "left") ?(right = R.default_config "right")
    ?(donor = Side.Left) ?(overrides = []) ?(emit = fun _ -> Ok ()) ?limit
    ?(expected_identities = (None, None)) ?initial_override () =
  {
    left_cfg = left;
    right_cfg = right;
    donor;
    overrides;
    emit;
    limit;
    expected_identities;
    initial_override;
  }

(* The live blob count of the run's ledger, readable from an event sink. *)
let probe : (unit -> int) ref = ref (fun () -> 0)

type run = {
  result : Core.Coordinator.result;
  ctx : L.Context.t;
  left : R.counters;
  right : R.counters;
}

let execute t (s : setup) =
  let ctx = Codec_fixtures.context ?limit:s.limit () in
  (probe := fun () -> Codecs.Owned_blob.Ledger.live_blobs ctx.ledger);
  let lrunner, lc = R.create s.left_cfg
  and rrunner, rc = R.create s.right_cfg in
  let ls = Plan.Graph_index.graph (Plan.Structural.left t.structural)
  and rs = Plan.Graph_index.graph (Plan.Structural.right t.structural) in
  let cfg : Core.Coordinator.config =
    {
      ctx;
      codecs = Codecs.Element_codec.Registry.builtin;
      operations = Codecs.Conversion_registry.empty;
      metrics = M.Tensor_metrics.empty_env;
      hashers;
      run_id = P.text (Id.Run.of_string "run0");
      case_id = P.text (Id.Case.of_string "case0");
      experiment =
        {
          experiment_id = P.text (Id.Experiment.of_string "exp0");
          default_donor = s.donor;
          overrides = s.overrides;
          compare_policy_digest = Pol.Seal.placeholder algorithm;
        };
      bindings = t.bindings;
      initial = Option.value s.initial_override ~default:(initial_values t);
      expected_identities = s.expected_identities;
      emit = s.emit;
    }
  in
  let result =
    Core.Coordinator.run cfg t.structural t.extractions
      ~left:(endpoint lrunner ls) ~right:(endpoint rrunner rs)
  in
  { result; ctx; left = lc; right = rc }

let event_name (e : Event.t) =
  let u =
    match e.unit with Some u -> Id.Unit.to_string u ^ ":" | None -> ""
  in
  u
  ^
  match e.payload with
  | Inputs_verified _ -> "inputs_verified"
  | Graph_completed { side; _ } ->
      "graph_completed(" ^ Side.to_string side ^ ")"
  | Outputs_compared _ -> "outputs_compared"
  | Frontier_committed _ -> "frontier_committed"
  | Failed f -> (
      "failed:"
      ^
      match Failure.encode_code f.code with
      | Jsont.String (s, _) -> s
      | _ -> "?")

let verdicts (r : Core.Coordinator.result) =
  List.map
    (fun (c : Comparison_result.t) ->
      Id.Comparison.to_string c.comparison_id
      ^ "="
      ^
      match c.verdict with
      | Pass -> "pass"
      | Mismatch -> "mismatch"
      | Metadata_error -> "metadata_error"
      | Metric_unavailable -> "metric_unavailable"
      | Unsupported -> "unsupported")
    r.comparisons

let completion (r : Core.Coordinator.result) =
  Printf.sprintf "execution=%s numerical=%s coverage=%s"
    (match r.completion.execution with
    | Complete -> "complete"
    | Failed -> "failed"
    | Refused -> "refused"
    | Interrupted -> "interrupted")
    (match r.completion.numerical with
    | Agreement -> "agreement"
    | Disagreement -> "disagreement"
    | Not_established -> "not_established")
    (match r.completion.coverage with
    | Complete -> "complete"
    | Partial _ -> "partial")

let observed (c : R.counters) =
  List.rev_map
    (fun run ->
      String.concat " "
        (List.map
           (fun (n, xs) ->
             Printf.sprintf "%s=%s" n
               (Format.asprintf "%a" Codec_fixtures.pp_ints xs))
           run))
    c.observed_inputs
