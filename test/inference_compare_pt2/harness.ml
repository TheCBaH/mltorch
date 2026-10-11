(* Run one pair through {!Inference_compare_host.Pair_run} and print what it
   finds. Shared by the pair drivers. *)
open Inference_compare_schema
module Host = Inference_compare_host

let run_with ~(on_event : Event.t -> unit) ~donor ~tol ~mapping ~left_graph
    ~right_graph ~initial ~node_attributes ~left_endpoint ~right_endpoint =
  let t0 = Unix.gettimeofday () in
  let r, run_ctx =
    match
      Host.Pair_run.run ~on_event ~donor ~tolerance:tol ~mapping ~left_graph
        ~right_graph ~initial ~node_attributes ~left_endpoint ~right_endpoint
    with
    | Ok o ->
        Printf.printf "extracted %d graphs\n%!" o.extracted;
        (o.result, o.ctx)
    | Error e ->
        Format.eprintf "%a@." Host.Session.pp_error e;
        exit Host.Pair_run.error_exit_code
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

let run = run_with ~on_event:(fun _ -> ())
