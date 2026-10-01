(* See arena_schedule_plan.mli. *)

open Core.Storage_units
module Problem = Arena_schedule_problem
module Search = Arena_schedule_search
module G = Arena_schedule_greedy
module Metrics = Arena_schedule.Metrics
module Strategy = Arena_schedule.Strategy

type error =
  [ Problem.error | Arena_run.error | `Predicted_metrics_differ of Strategy.t ]

let pp_error ppf : [< error ] -> unit = function
  | `Predicted_metrics_differ s ->
      Fmt.pf ppf "the %a order's replayed metrics differ from a fresh script"
        Strategy.pp s
  | #Problem.error as e -> Problem.pp_error ppf e
  | #Arena_run.error as e -> Arena_run.pp_error ppf e

let is_refusal : error -> bool = function
  | `Arena_over_limit _ | `Over_budget _ | `Physical_alignment_unsupported _ ->
      true
  | _ -> false

module Plan = struct
  type t = Intermediate of Arena_plan.t | Roles of Storage_plan.t
end

module Verdict = struct
  type t = Planned of Byte_size.t | Refused of error | Skipped
end

module Report = struct
  type t = {
    strategy : Strategy.t;
    metrics : Metrics.t;
    pool_bytes : Byte_size.t option;
    verdict : Verdict.t;
  }
end

module Selection = struct
  type t = {
    graph : Graph_ir.graph;
    plan : Plan.t option;
    strategy : Strategy.t;
    metrics : Metrics.t;
    baseline : Metrics.t;
    payload_winner : Strategy.t;
    pool_bytes : Byte_size.t option;
    baseline_pool_bytes : Byte_size.t option;
    reports : Report.t list;
    declined : error option;
    stop : Arena_schedule.Stop.t;
    stats : Arena_schedule.Stats.t;
  }
end

(* The row of a refusal, as data for the report: the wrapper is dropped on
   purpose, the same way [Arena_run] reports a declined arena. *)
let to_row (e : error Err.Error.t) : error =
  match Err.export ~pos:__POS__ (Error e) with
  | Error row -> row
  | Ok _ -> assert false

(* One candidate, placed. [Ok (plan, pool_bytes, admission)]: a plan exists;
   [admission] is its metadata-only admission check. [Error e] with a refusal
   row means no plan could be built (a pool over a ceiling). *)
type placed = {
  plan : Plan.t;
  pool : Byte_size.t;
  admission : (unit, error) Err.t;
}

let place ?limits ?budget ?physical ~admission (c : Arena_schedule.Config.t)
    graph =
  let open Err.Syntax in
  match c.mode with
  | Intermediate ->
      let* script =
        Eval_direct.dry_run ~alignment:c.alignment ~retain:c.retain graph
        |> Err.map_error ~pos:__POS__ (fun e -> (e :> error))
      in
      let* plan =
        Arena_plan.create ?limits ?budget ~alignment:c.alignment script
        |> Err.map_error ~pos:__POS__ (fun e -> (e :> error))
      in
      Err.return
        {
          plan = Plan.Intermediate plan;
          pool = (Arena_plan.stats plan).Arena_plan.Stats.pool_bytes;
          admission =
            Arena_run.check_plan ?physical ~admission plan
            |> Err.map_error (fun e -> (e :> error));
        }
  | Roles config ->
      let* script =
        Eval_direct.storage_script ~alignment:c.alignment ~retain:c.retain
          config graph
        |> Err.map_error ~pos:__POS__ (fun e -> (e :> error))
      in
      let* plan =
        Storage_plan.create ?limits ?budget script
        |> Err.map_error ~pos:__POS__ (fun e -> (e :> error))
      in
      let* footprint =
        Storage_plan.footprint plan
        |> Err.map_error ~pos:__POS__ (fun e -> (e :> error))
      in
      Err.return
        {
          plan = Plan.Roles plan;
          pool = footprint.Storage_plan.Footprint.execution;
          admission = Err.return ();
        }

(* What one candidate came to: a plan to choose from, or the refusal that
   keeps it out. A defect is never a verdict: it propagates. *)
type outcome =
  | Placed of placed
  | Refused_at_plan of error Err.Error.t
  | Skipped_worse

let try_place ?limits ?budget ?physical ~admission c graph =
  match place ?limits ?budget ?physical ~admission c graph with
  | Ok p -> Err.return (Placed p)
  | Error e when is_refusal (Err.Error.kind e) -> Err.return (Refused_at_plan e)
  | Error _ as e -> e

let ( <?> ) a b = match a with 0 -> b () | c -> c

let choose ?limits ?budget ?(admission = Arena.Admission.Best_effort) ?physical
    (c : Arena_schedule.Config.t) g =
  let open Err.Syntax in
  let* problem = Problem.of_graph c g in
  let* cands, stats, beam_stop =
    match (c.mode, c.retain) with
    | Intermediate, All ->
        let order =
          Array.init (Problem.node_count problem) Problem.Position.of_int
        in
        let* metrics = Problem.metrics problem order in
        Err.return
          ( [ { G.Candidate.strategy = Strategy.Identity; order; metrics } ],
            Arena_schedule.Stats.zero,
            Arena_schedule.Stop.Completed )
    | _ -> Search.portfolio c problem
  in
  let* baseline =
    match cands with
    | first :: _ -> Err.return first
    | [] -> Err.fail ~pos:__POS__ `Not_a_permutation
  in
  let payload_winner =
    match G.select problem cands stats with
    | Ok r -> r.Arena_schedule.Result.strategy
    | Error _ -> Strategy.Identity
  in
  (* Place every candidate whose payload is no worse than the original's. *)
  let* placed =
    List.fold_left
      (fun acc (cand : G.Candidate.t) ->
        let* acc = acc in
        if
          cand != baseline
          && Byte_size.compare cand.metrics.target_peak
               baseline.metrics.target_peak
             > 0
        then Err.return (acc @ [ (cand, None, Skipped_worse) ])
        else
          let* graph = Problem.reorder problem cand.order in
          let* fresh = Problem.fresh_metrics c graph in
          let* () =
            if Metrics.equal cand.metrics fresh then Err.return ()
            else Err.fail ~pos:__POS__ (`Predicted_metrics_differ cand.strategy)
          in
          let* o = try_place ?limits ?budget ?physical ~admission c graph in
          Err.return (acc @ [ (cand, Some graph, o) ]))
      (Err.return []) cands
  in
  (* Admission verdicts; a plan that fails admission still reports its pool. *)
  let judged =
    List.map
      (fun ((cand : G.Candidate.t), graph, o) ->
        match o with
        | Skipped_worse -> (cand, graph, None, `Skipped)
        | Refused_at_plan e -> (cand, graph, None, `Refused e)
        | Placed p -> (
            match p.admission with
            | Ok () -> (cand, graph, Some p, `Admitted)
            | Error e when is_refusal (Err.Error.kind e) ->
                (cand, graph, Some p, `Refused e)
            | Error e -> (cand, graph, Some p, `Defect e)))
      placed
  in
  let* () =
    List.fold_left
      (fun acc (_, _, _, v) ->
        let* () = acc in
        match v with `Defect e -> Error e | _ -> Err.return ())
      (Err.return ()) judged
  in
  let reports =
    List.map
      (fun ((cand : G.Candidate.t), _, p, v) ->
        {
          Report.strategy = cand.strategy;
          metrics = cand.metrics;
          pool_bytes = Option.map (fun p -> p.pool) p;
          verdict =
            (match v with
            | `Skipped -> Verdict.Skipped
            | `Refused e -> Verdict.Refused (to_row e)
            | `Admitted -> Verdict.Planned (Option.get p).pool
            | `Defect _ -> assert false);
        })
      judged
  in
  let baseline_pool_bytes =
    match reports with r :: _ -> r.Report.pool_bytes | [] -> None
  in
  let better (a : G.Candidate.t) pa (b : G.Candidate.t) pb =
    ( Byte_size.compare pa pb <?> fun () ->
      Byte_size.compare a.metrics.target_peak b.metrics.target_peak )
    <?> fun () ->
    Byte_size.compare a.metrics.outside_peak b.metrics.outside_peak
  in
  let best =
    List.fold_left
      (fun best ((cand : G.Candidate.t), graph, p, v) ->
        match (v, p, best) with
        | `Admitted, Some p, None -> Some (cand, graph, p)
        | `Admitted, Some p, Some (bc, _, bp) ->
            if better cand p.pool bc bp.pool < 0 then Some (cand, graph, p)
            else best
        | _ -> best)
      None judged
  in
  let make ~graph ~plan ~strategy ~metrics ~pool_bytes ~declined =
    Err.return
      {
        Selection.graph;
        plan;
        strategy;
        metrics;
        baseline = baseline.metrics;
        payload_winner;
        pool_bytes;
        baseline_pool_bytes;
        reports;
        declined;
        stop = Search.stop_of ~selected:Arena_schedule.Stop.Completed ~beam_stop;
        stats;
      }
  in
  match best with
  | Some (cand, Some graph, p) ->
      make ~graph ~plan:(Some p.plan) ~strategy:cand.strategy
        ~metrics:cand.metrics ~pool_bytes:(Some p.pool) ~declined:None
  | Some (_, None, _) -> assert false
  | None -> (
      (* Every candidate was refused, the original included. *)
      let original_error =
        match judged with (_, _, _, `Refused e) :: _ -> e | _ -> assert false
      in
      match (c.mode, admission) with
      | Intermediate, Arena.Admission.Best_effort ->
          make ~graph:g ~plan:None ~strategy:Strategy.Identity
            ~metrics:baseline.metrics ~pool_bytes:None
            ~declined:(Some (to_row original_error))
      | _ -> Error original_error)
