(* Scheduling evaluation over the tracked model.json corpus: for each model,
   the canonical Native graph is scheduled under each requested configuration
   and every candidate order is placed by the production planner, so the
   figures that matter are allocated pool bytes before and after. No weights,
   no inputs, no tensor operation.

   Output is deterministic JSONL (no timings): one coverage row per
   configuration and one row per model and configuration, in corpus order.
   Timings are measurements and go to [--timings] only. [--summary] renders the
   same rows as markdown. Exits nonzero on a corpus-count mismatch or a failed
   scheduling; a classified refusal is a result. See .ai/ on the arena
   scheduling design. *)

module X = Arena_eval_extract
module J = Arena_eval_json
module Plan = Arena_schedule_plan
module Search = Arena_schedule_search
open Core.Storage_units

let b64 = Byte_size.to_int64
let version = 1

(* --- configurations ------------------------------------------------------- *)

module Setting = struct
  type t = { name : string; limits : Arena_schedule.Limits.t }
end

let default_settings =
  "constructive=1:0,beam1k=8:1000,beam10k=8:10000,beam100k=8:100000"

let parse_settings s =
  let one item =
    match String.split_on_char '=' item with
    | [ name; spec ] -> (
        match String.split_on_char ':' spec with
        | [ w; e ] -> (
            match (int_of_string_opt w, int_of_string_opt e) with
            | Some width, Some expansions -> (
                match
                  Arena_schedule.Limits.make ~width ~expansions
                    ~state_bytes:Arena_schedule.Limits.default_beam.state_bytes
                with
                | Ok limits -> Ok { Setting.name; limits }
                | Error _ -> Error (Fmt.str "invalid limits in %s" item))
            | _ -> Error (Fmt.str "not numbers in %s" item))
        | _ -> Error (Fmt.str "expected NAME=WIDTH:EXPANSIONS, got %s" item))
    | _ -> Error (Fmt.str "expected NAME=WIDTH:EXPANSIONS, got %s" item)
  in
  List.fold_left
    (fun acc item ->
      match (acc, one item) with
      | Error e, _ | _, Error e -> Error e
      | Ok l, Ok s -> Ok (l @ [ s ]))
    (Ok [])
    (String.split_on_char ',' s |> List.filter (( <> ) ""))

(* --- the corpus ----------------------------------------------------------- *)

let models_of dir wanted =
  let all =
    Sys.readdir dir |> Array.to_list
    |> List.filter (fun m ->
        Sys.file_exists (Filename.concat dir (m ^ "/models/model.json")))
    |> List.sort String.compare
  in
  match wanted with [] -> all | w -> List.filter (fun m -> List.mem m w) all

let read path = In_channel.with_open_bin path In_channel.input_all

(* --- rows ----------------------------------------------------------------- *)

let stop_name (s : Arena_schedule.Stop.t) =
  match s with
  | Budget_exhausted -> "budget_exhausted"
  | Completed -> "completed"
  | Lower_bound_reached -> "lower_bound_reached"
  | State_limit -> "state_limit"

let strategy_name s = Fmt.str "%a" Arena_schedule.Strategy.pp s

let order_digest (g : Graph_ir.graph) =
  Digest.to_hex
    (Digest.string
       (String.concat ","
          (List.map
             (fun (n : Graph_ir.node) ->
               string_of_int (Graph_ir.Node_id.to_int n.Graph_ir.Node.id))
             g.Graph_common.Graph.nodes)))

let metrics_json (m : Arena_schedule.Metrics.t) pool_bytes =
  J.obj
    [
      ("target_peak", J.i64 (b64 m.target_peak));
      ("pool_peak_sum", J.i64 (b64 m.pool_peak_sum));
      ("outside_peak", J.i64 (b64 m.outside_peak));
      ("all_peak", J.i64 (b64 m.all_peak));
      ("pool_bytes", J.opt (fun b -> J.i64 (b64 b)) pool_bytes);
    ]

let verdict_json (v : Plan.Verdict.t) =
  match v with
  | Planned _ -> J.str "planned"
  | Refused e -> J.str (Fmt.str "refused: %a" Plan.pp_error e)
  | Skipped -> J.str "skipped"

module Result_row = struct
  type t = {
    model : string;
    setting : string;
    nodes : int;
    selection : Plan.Selection.t;
    digest_before : string;
    digest_after : string;
  }
end

let result_json (r : Result_row.t) =
  let s = r.selection in
  J.obj
    [
      ("type", J.str "schedule");
      ("version", J.int version);
      ("model", J.str r.model);
      ("setting", J.str r.setting);
      ("status", J.str "evaluated");
      ("nodes", J.int r.nodes);
      ("graph_digest_before", J.str r.digest_before);
      ("graph_digest_after", J.str r.digest_after);
      ("strategy", J.str (strategy_name s.strategy));
      ("payload_winner", J.str (strategy_name s.payload_winner));
      ("stop", J.str (stop_name s.stop));
      ("expansions", J.int s.stats.expansions);
      ("state_bytes", J.i64 (b64 s.stats.state_bytes));
      ("baseline", metrics_json s.baseline s.baseline_pool_bytes);
      ("chosen", metrics_json s.metrics s.pool_bytes);
      ( "candidates",
        J.list
          (fun (c : Plan.Report.t) ->
            J.obj
              [
                ("strategy", J.str (strategy_name c.strategy));
                ("metrics", metrics_json c.metrics c.pool_bytes);
                ("verdict", verdict_json c.verdict);
              ])
          s.reports );
    ]

let status_json ~model ~setting ~status ~stage ~diagnostic =
  J.obj
    [
      ("type", J.str "schedule");
      ("version", J.int version);
      ("model", J.str model);
      ("setting", J.str setting);
      ("status", J.str status);
      ("stage", J.str stage);
      ("diagnostic", J.str diagnostic);
    ]

(* --- the run -------------------------------------------------------------- *)

module Tally = struct
  type t = {
    mutable evaluated : int;
    mutable refused : int;
    mutable failed : int;
  }
end

let outcome_fields = function
  | X.Outcome.Failed { stage; diagnostic }
  | X.Outcome.Prerequisite { stage; diagnostic } ->
      ("failed", X.Stage.name stage, diagnostic)
  | X.Outcome.Refused { stage; diagnostic; _ } ->
      ("refused", X.Stage.name stage, diagnostic)

let schedule_config limits : Arena_schedule.Config.t =
  {
    mode = Intermediate;
    retain = Only Graph_ir.Tensor_id.Set.empty;
    alignment = Alignment_policy.standard;
    limits;
  }

let run ~models_dir ~expected ~models ~settings ~output ~timings ~summary =
  let now = Unix.gettimeofday in
  let entries = models_of models_dir models in
  if models = [] && List.length entries <> expected then (
    Fmt.epr "expected %d models, found %d@." expected (List.length entries);
    1)
  else
    let tallies =
      List.map
        (fun (s : Setting.t) ->
          (s.name, { Tally.evaluated = 0; refused = 0; failed = 0 }))
        settings
    in
    let rows = ref [] in
    let out = open_out_bin output in
    let tout = Option.map open_out_bin timings in
    let emit json =
      output_string out (J.encode json);
      output_char out '\n';
      flush out
    in
    let time_row ~model ~setting ~stage seconds =
      Option.iter
        (fun oc ->
          output_string oc
            (J.encode
               (J.obj
                  [
                    ("model", J.str model);
                    ("setting", J.str setting);
                    ("stage", J.str stage);
                    ("seconds", J.num seconds);
                  ]));
          output_char oc '\n';
          flush oc)
        tout
    in
    List.iter
      (fun model ->
        let bytes =
          read (Filename.concat models_dir (model ^ "/models/model.json"))
        in
        match X.canonical ~now bytes with
        | Error (outcome, seconds) ->
            time_row ~model ~setting:"-" ~stage:"import" seconds;
            let status, stage, diagnostic = outcome_fields outcome in
            List.iter
              (fun (s : Setting.t) ->
                let t = List.assoc s.name tallies in
                if status = "refused" then t.refused <- t.refused + 1
                else t.failed <- t.failed + 1;
                emit
                  (status_json ~model ~setting:s.name ~status ~stage ~diagnostic))
              settings
        | Ok (Native_interp.Transformed t, seconds) ->
            time_row ~model ~setting:"-" ~stage:"import" seconds;
            let graph = t.graph in
            List.iter
              (fun (s : Setting.t) ->
                let tally = List.assoc s.name tallies in
                let t0 = now () in
                let result = Plan.choose (schedule_config s.limits) graph in
                time_row ~model ~setting:s.name ~stage:"schedule_and_place"
                  (now () -. t0);
                match result with
                | Ok selection ->
                    tally.evaluated <- tally.evaluated + 1;
                    let row =
                      {
                        Result_row.model;
                        setting = s.name;
                        nodes = List.length graph.Graph_common.Graph.nodes;
                        selection;
                        digest_before = order_digest graph;
                        digest_after = order_digest selection.graph;
                      }
                    in
                    rows := row :: !rows;
                    emit (result_json row)
                | Error e ->
                    let row = Err.Error.kind e in
                    let refused = Plan.is_refusal row in
                    if refused then tally.refused <- tally.refused + 1
                    else tally.failed <- tally.failed + 1;
                    emit
                      (status_json ~model ~setting:s.name
                         ~status:(if refused then "refused" else "failed")
                         ~stage:"schedule"
                         ~diagnostic:(Fmt.str "%a" Plan.pp_error row)))
              settings)
      entries;
    List.iter
      (fun (name, (t : Tally.t)) ->
        emit
          (J.obj
             [
               ("type", J.str "coverage");
               ("setting", J.str name);
               ("expected", J.int expected);
               ("attempted", J.int (List.length entries));
               ("evaluated", J.int t.evaluated);
               ("refused", J.int t.refused);
               ("failed", J.int t.failed);
             ]))
      tallies;
    close_out out;
    Option.iter close_out tout;
    Option.iter
      (fun path ->
        Out_channel.with_open_bin path (fun oc ->
            let ppf = Format.formatter_of_out_channel oc in
            let rows =
              List.rev_map
                (fun (r : Result_row.t) ->
                  {
                    Arena_schedule_report.model = r.model;
                    setting = r.setting;
                    nodes = r.nodes;
                    selection = r.selection;
                  })
                !rows
            in
            Arena_schedule_report.render ppf rows;
            Arena_schedule_report.render_all ppf rows;
            Format.pp_print_flush ppf ()))
      summary;
    if List.exists (fun (_, (t : Tally.t)) -> t.failed > 0) tallies then 1
    else 0

(* --- command line --------------------------------------------------------- *)

open Cmdliner

let models_dir =
  Arg.(
    required
    & opt (some dir) None
    & info [ "models-dir" ] ~docv:"DIR" ~doc:"The model.json corpus.")

let expected =
  Arg.(
    value & opt int 100
    & info [ "expected-models" ] ~docv:"N"
        ~doc:"The corpus size; a different count is an error.")

let models =
  Arg.(
    value
    & opt (list string) []
    & info [ "models" ] ~docv:"M,.."
        ~doc:"Only these models (skips the count check).")

let settings =
  Arg.(
    value
    & opt string default_settings
    & info [ "settings" ] ~docv:"NAME=WIDTH:EXPANSIONS,.."
        ~doc:"The scheduler configurations to evaluate.")

let output =
  Arg.(
    required
    & opt (some string) None
    & info [ "output" ] ~docv:"FILE" ~doc:"Deterministic JSONL rows.")

let timings =
  Arg.(
    value
    & opt (some string) None
    & info [ "timings" ] ~docv:"FILE" ~doc:"Host timings (never deterministic).")

let summary =
  Arg.(
    value
    & opt (some string) None
    & info [ "summary" ] ~docv:"FILE" ~doc:"Markdown summary of the rows.")

let main models_dir expected models settings output timings summary =
  match parse_settings settings with
  | Error e ->
      Fmt.epr "%s@." e;
      2
  | Ok settings ->
      run ~models_dir ~expected ~models ~settings ~output ~timings ~summary

let () =
  let cmd =
    Cmd.v
      (Cmd.info "arena_schedule_eval"
         ~doc:"Evaluate memory-aware scheduling over the model.json corpus.")
      Term.(
        const main $ models_dir $ expected $ models $ settings $ output
        $ timings $ summary)
  in
  exit (Cmd.eval' cmd)
