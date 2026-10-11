(* Run the real ATen / Native pair on one model and sample through the shared
   frontier coordinator and print what it finds. *)
open Inference_compare_schema
module P = Inference_compare_pt2
module Plan = Inference_compare_plan
module Core = Inference_compare_core
module Ex = Inference_compare_extract
module Pol = Inference_compare_policy
module M = Inference_compare_metrics
module Codecs = Inference_compare_codecs
module L = Codecs.Logical_value

let hashers = Inference_compare_sha256.registry
let algorithm = Inference_compare_sha256.algorithm

let with_batch (tensor : Pt2_tensor.t) =
  let stride = Pt2_tensor.numel tensor in
  {
    tensor with
    Pt2_tensor.sizes = 1 :: tensor.sizes;
    strides = stride :: tensor.strides;
  }

let donor : Side.t =
  match Sys.getenv_opt "DONOR" with Some "right" -> Right | _ -> Left

let calibrate = Sys.getenv_opt "CALIBRATE" <> None

let tol =
  match Sys.getenv_opt "TOL" with Some s -> float_of_string s | None -> 1e-4

let () =
  let model = Sys.argv.(1) and inputs_pt = Sys.argv.(2) in
  let sample =
    match Pt2_archive.load_pt_tensor_map inputs_pt with
    | Ok ((_, t) :: _) -> with_batch t
    | _ -> failwith "inputs"
  in
  let aten0 =
    match P.Aten_model.of_file ~hashers ~algorithm model with
    | Ok a -> a
    | Error m -> failwith m
  in
  let input_name =
    match
      List.find_opt
        (fun (s : P.Aten_model.source) -> s.captured = None)
        aten0.sources
    with
    | Some s -> s.name
    | None -> failwith "no user input"
  in
  let pair =
    match
      P.Pair.load ~hashers ~algorithm ~path:model
        ~inputs:[ (input_name, sample) ]
    with
    | Ok p -> p
    | Error m -> failwith ("pair: " ^ m)
  in
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
        ~options:Inference_compare_test.Plan_fixtures.options
        ~left:(P.Pair.left_graph pair) ~right:(P.Pair.right_graph pair)
        (P.Pair.mapping pair)
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
      Ex.Extraction.extract_all
        ~node_attributes:(P.Pair.node_attributes pair)
        ~hashers ~algorithm structural
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
      initial = P.Pair.initial pair;
      expected_identities = (None, None);
      emit = (fun _ -> Ok ());
      retain = (fun _ -> Ok ());
      mode = Full;
    }
  in
  let r =
    Core.Coordinator.run cfg sealed
      ~left:(P.Pair.aten_endpoint pair)
      ~right:(P.Pair.native_endpoint pair)
  in
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

  if calibrate then begin
    (* The donor's segmented run (each unit fed the donor's own committed
       values) against that engine's whole-graph run, bitwise. *)
    let inputs = [ (input_name, sample) ] in
    let archive = aten0.archive in
    let whole : (string list, string) result =
      match donor with
      | Left -> (
          match Interp.run_named archive ~inputs with
          | Error _ -> Error "whole ATen run failed"
          | Ok ts ->
              Ok
                (List.map2
                   (fun t (_, v) ->
                     match P.Conv.logical_of_aten run_ctx (L.spec v) t with
                     | Ok w -> L.contents w
                     | Error m -> m)
                   ts r.finals))
      | Right -> (
          match Native_interp.run_named archive ~inputs with
          | Error _ -> Error "whole Native run failed"
          | Ok ts ->
              Ok
                (List.map2
                   (fun (Tensor.Tensor t as p) (_, v) ->
                     let shape =
                       List.map
                         (fun a -> Int64.of_int (Vec6.get t.shape a :> int))
                         Axis.all
                     in
                     let spec =
                       { (L.spec v) with Tensor_spec.logical_shape = shape }
                     in
                     match P.Conv.logical_of_native run_ctx spec p with
                     | Ok w -> L.contents w
                     | Error m -> m)
                   ts r.finals))
    in
    match whole with
    | Error m -> Printf.printf "calibration: %s\n" m
    | Ok whole ->
        let same =
          List.for_all2
            (fun w (_, v) -> String.equal w (L.contents v))
            whole r.finals
        in
        if not same then
          List.iter2
            (fun w (_, v) ->
              let n = String.length w / 4 in
              let worst = ref 0.0 and diff = ref 0 in
              for i = 0 to n - 1 do
                let a = Int32.float_of_bits (String.get_int32_le w (4 * i))
                and b =
                  Int32.float_of_bits
                    (String.get_int32_le (L.contents v) (4 * i))
                in
                if a <> b then incr diff;
                worst := Float.max !worst (Float.abs (a -. b))
              done;
              Printf.printf "  %d of %d elements differ, max abs %.3g\n" !diff n
                !worst)
            whole r.finals;
        Printf.printf "calibration: %d final output(s) %s the whole-graph run\n"
          (List.length whole)
          (if same then "bitwise equal" else "DIFFER from")
  end
