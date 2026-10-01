(* See arena_schedule_search.mli. *)

open Core.Storage_units
module Problem = Arena_schedule_problem
module Position = Problem.Position
module G = Arena_schedule_greedy
module State = G.State
module Stats = Arena_schedule.Stats
module Stop = Arena_schedule.Stop

module Outcome = struct
  type t = { order : Position.t array option; stop : Stop.t; stats : Stats.t }
end

(* A rough size of one retained state: its persistent maps and order list scale
   with the node count. Checked against the limit before any state exists. *)
let state_bytes_estimate n = Int64.(mul 8L (add 64L (mul 4L (of_int n))))

let lex_compare a b =
  let n = min (Array.length a) (Array.length b) in
  let rec go i =
    if i = n then compare (Array.length a) (Array.length b)
    else match Position.compare a.(i) b.(i) with 0 -> go (i + 1) | c -> c
  in
  go 0

let beam problem (limits : Arena_schedule.Limits.t) =
  let open Err.Syntax in
  let n = Problem.node_count problem in
  let est = state_bytes_estimate n in
  let need = Int64.(mul 2L (mul (of_int limits.width) est)) in
  let none stop stats = Err.return { Outcome.order = None; stop; stats } in
  if limits.expansions = 0 then none Stop.Completed Stats.zero
  else if Int64.compare need (Byte_size.to_int64 limits.state_bytes) > 0 then
    none Stop.State_limit Stats.zero
  else
    let* bound = Problem.lower_bound problem in
    let floor = bound.target_peak in
    let rank s = (Byte_size.max (State.peak s) floor, State.peak s) in
    let compare_states a b =
      let (a1, a2), (b1, b2) = (rank a, rank b) in
      match Byte_size.compare a1 b1 with
      | 0 -> (
          match Byte_size.compare a2 b2 with
          | 0 -> (
              match Byte_size.compare (State.live a) (State.live b) with
              | 0 -> (
                  match
                    Byte_size.compare (State.all_peak a) (State.all_peak b)
                  with
                  | 0 -> lex_compare (State.order a) (State.order b)
                  | c -> c)
              | c -> c)
          | c -> c)
      | c -> c
    in
    (* A sorted list of at most [width] survivors. *)
    let rec insert s = function
      | [] -> [ s ]
      | x :: rest as l ->
          if compare_states s x < 0 then s :: l else x :: insert s rest
    in
    let truncate l = List.filteri (fun i _ -> i < limits.width) l in
    let finish ~order ~stop ~used ~depth ~widest ~retained =
      let retained_bytes =
        Result.get_ok
          (Err.payload
             (Byte_size.of_int64 (Int64.mul est (Int64.of_int retained))))
      in
      Err.return
        {
          Outcome.order;
          stop;
          stats =
            {
              Stats.zero with
              expansions = used;
              depth;
              max_ready_width = widest;
              retained_states = retained;
              state_bytes = retained_bytes;
            };
        }
    in
    let* start = State.start problem in
    let rec layers depth frontier used widest retained =
      match frontier with
      | best :: _ when depth = n ->
          finish
            ~order:(Some (State.order best))
            ~stop:Stop.Completed ~used ~depth ~widest ~retained
      | [] ->
          finish ~order:None ~stop:Stop.Completed ~used ~depth ~widest ~retained
      | _ -> (
          let rec expand next used widest = function
            | [] -> Err.return (Some (next, used, widest))
            | s :: rest -> (
                let ready = State.ready s in
                let widest = max widest (List.length ready) in
                let rec children next used = function
                  | [] -> Err.return (Some (next, used))
                  | p :: ps ->
                      if used >= limits.expansions then Err.return None
                      else
                        let* child = State.step s p in
                        children (truncate (insert child next)) (used + 1) ps
                in
                let* r = children next used ready in
                match r with
                | None -> Err.return None
                | Some (next, used) -> expand next used widest rest)
          in
          let* r = expand [] used widest frontier in
          match r with
          | None ->
              finish ~order:None ~stop:Stop.Budget_exhausted ~used ~depth
                ~widest ~retained
          | Some (next, used, widest) ->
              layers (depth + 1) next used widest
                (max retained (List.length next)))
    in
    layers 0 [ start ] 0 0 1

let portfolio (c : Arena_schedule.Config.t) problem =
  let open Err.Syntax in
  let* cands, stats = G.candidates problem in
  let* outcome = beam problem c.limits in
  let* cands =
    match outcome.order with
    | Some order
      when not (List.exists (fun (k : G.Candidate.t) -> k.order = order) cands)
      ->
        let* metrics = Problem.metrics problem order in
        Err.return
          (cands
          @ [
              {
                G.Candidate.strategy = Arena_schedule.Strategy.Beam;
                order;
                metrics;
              };
            ])
    | _ -> Err.return cands
  in
  let stats =
    {
      stats with
      expansions = outcome.stats.expansions;
      retained_states = outcome.stats.retained_states;
      state_bytes = outcome.stats.state_bytes;
      max_ready_width = max stats.max_ready_width outcome.stats.max_ready_width;
    }
  in
  Err.return (cands, stats, outcome.stop)

(* A search that ended on a limit says so, unless it proved the bound. *)
let stop_of ~selected ~beam_stop =
  match (selected, beam_stop) with
  | Stop.Lower_bound_reached, _ -> Stop.Lower_bound_reached
  | _, ((Stop.Budget_exhausted | State_limit) as s) -> s
  | _ -> Stop.Completed

let run (c : Arena_schedule.Config.t) g =
  let open Err.Syntax in
  let* problem = Problem.of_graph c g in
  match (c.mode, c.retain) with
  | Intermediate, All -> Arena_schedule.identity g
  | _ ->
      let* cands, stats, beam_stop = portfolio c problem in
      let* r = G.select problem cands stats in
      Err.return { r with stop = stop_of ~selected:r.stop ~beam_stop }
