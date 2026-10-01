(* See arena_schedule_greedy.mli. *)

open Core.Storage_units
module Problem = Arena_schedule_problem
module Position = Problem.Position
module Metrics = Arena_schedule.Metrics
module Strategy = Arena_schedule.Strategy

module Score = struct
  type t = { peak : Byte_size.t; live : Byte_size.t; all_peak : Byte_size.t }
end

module State = struct
  type t = {
    problem : Problem.t;
    succs : Position.t list array;
    remaining : int Tensor_id.Map.t;  (** readers not yet run, by tensor *)
    unmet : int Position.Map.t;  (** predecessors not yet run, by node *)
    ready : Position.Set.t;
    order : Position.t list;  (** newest first *)
    live : Byte_size.t;
    live_all : Byte_size.t;
    peak : Byte_size.t;
    peak_all : Byte_size.t;
  }

  let base problem =
    let n = Problem.node_count problem in
    let succs = Array.make n [] in
    let positions = List.init n Position.of_int in
    List.iter
      (fun p ->
        List.iter
          (fun q ->
            let k = (q : Position.t :> int) in
            succs.(k) <- p :: succs.(k))
          (Problem.preds problem p))
      positions;
    let unmet, ready =
      List.fold_left
        (fun (unmet, ready) p ->
          match List.length (Problem.preds problem p) with
          | 0 -> (unmet, Position.Set.add p ready)
          | k -> (Position.Map.add p k unmet, ready))
        (Position.Map.empty, Position.Set.empty)
        positions
    in
    {
      problem;
      succs = Array.map List.rev succs;
      remaining = Tensor_id.Map.empty;
      unmet;
      ready;
      order = [];
      live = Byte_size.zero;
      live_all = Byte_size.zero;
      peak = Byte_size.zero;
      peak_all = Byte_size.zero;
    }

  let ready t = Position.Set.elements t.ready
  let is_done t = Position.Set.is_empty t.ready

  let left t (b : Problem.Block.t) =
    match Tensor_id.Map.find_opt b.id t.remaining with
    | Some n -> n
    | None -> Problem.readers t.problem b.id

  (* The blocks the node releases once it has run: each operand it is the last
     remaining reader of, and each output nothing reads. *)
  let released t p =
    let last =
      List.filter
        (fun (b : Problem.Block.t) -> b.releasable && left t b = 1)
        (Problem.reads t.problem p)
    in
    let unread =
      List.filter
        (fun (b : Problem.Block.t) ->
          b.releasable && Problem.readers t.problem b.id = 0)
        (Problem.blocks t.problem p)
    in
    last @ unread

  let sum ~eligible_only bs =
    List.fold_left
      (fun acc (b : Problem.Block.t) ->
        let open Err.Syntax in
        let* s = acc in
        if eligible_only && not b.eligible then Err.return s
        else Problem.add ~id:b.id s b.bytes)
      (Err.return Byte_size.zero)
      bs

  (* The fixed prefix (role mode: constants and inputs) is allocated, and what
     nothing reads and the script frees is released, before the first node. *)
  let start problem =
    let open Err.Syntax in
    let t = base problem in
    let prefix = Problem.prefix problem in
    let unread =
      List.filter
        (fun (b : Problem.Block.t) ->
          b.releasable && Problem.readers problem b.id = 0)
        prefix
    in
    let* up_t = sum ~eligible_only:true prefix in
    let* up_all = sum ~eligible_only:false prefix in
    let* f_t = sum ~eligible_only:true unread in
    let* f_all = sum ~eligible_only:false unread in
    let id = Tensor_id.of_int 0 in
    let* live = Problem.sub ~id up_t f_t in
    let* live_all = Problem.sub ~id up_all f_all in
    Err.return { t with live; live_all; peak = up_t; peak_all = up_all }

  let transition t p =
    let open Err.Syntax in
    let outs = Problem.blocks t.problem p and freed = released t p in
    let* o_t = sum ~eligible_only:true outs in
    let* o_all = sum ~eligible_only:false outs in
    let* f_t = sum ~eligible_only:true freed in
    let* f_all = sum ~eligible_only:false freed in
    let id = match outs with b :: _ -> b.id | [] -> Tensor_id.of_int 0 in
    let* up_t = Problem.add ~id t.live o_t in
    let* up_all = Problem.add ~id t.live_all o_all in
    let* live = Problem.sub ~id up_t f_t in
    let* live_all = Problem.sub ~id up_all f_all in
    Err.return
      ( Byte_size.max t.peak up_t,
        live,
        Byte_size.max t.peak_all up_all,
        live_all )

  let score t p =
    let open Err.Syntax in
    let* peak, live, all_peak, _ = transition t p in
    Err.return { Score.peak; live; all_peak }

  let step t p =
    let open Err.Syntax in
    let* peak, live, peak_all, live_all = transition t p in
    let remaining =
      List.fold_left
        (fun m (b : Problem.Block.t) -> Tensor_id.Map.add b.id (left t b - 1) m)
        t.remaining
        (Problem.reads t.problem p)
    in
    let unmet, ready =
      List.fold_left
        (fun (unmet, ready) s ->
          match Position.Map.find s unmet - 1 with
          | 0 -> (Position.Map.remove s unmet, Position.Set.add s ready)
          | k -> (Position.Map.add s k unmet, ready))
        (t.unmet, Position.Set.remove p t.ready)
        t.succs.((p : Position.t :> int))
    in
    Err.return
      {
        t with
        remaining;
        unmet;
        ready;
        order = p :: t.order;
        live;
        live_all;
        peak;
        peak_all;
      }

  let order t = Array.of_list (List.rev t.order)
  let peak t = t.peak
  let live t = t.live
end

module Policy = struct
  type t = Live_first | Peak_first
end

let key policy (s : Score.t) =
  match (policy : Policy.t) with
  | Peak_first -> (s.peak, s.live, s.all_peak)
  | Live_first -> (s.live, s.peak, s.all_peak)

let compare_key (a1, a2, a3) (b1, b2, b3) =
  match Byte_size.compare a1 b1 with
  | 0 -> (
      match Byte_size.compare a2 b2 with 0 -> Byte_size.compare a3 b3 | c -> c)
  | c -> c

let schedule problem policy =
  let open Err.Syntax in
  let rec go state evals widest =
    if State.is_done state then
      let order = State.order state in
      Err.return
        ( order,
          {
            Arena_schedule.Stats.zero with
            score_evaluations = evals;
            depth = Array.length order;
            max_ready_width = widest;
          } )
    else
      let ready = State.ready state in
      (* Ascending position, so a strict [<] keeps the earliest on a tie. *)
      let* best =
        List.fold_left
          (fun acc p ->
            let* acc = acc in
            let* s = State.score state p in
            let k = key policy s in
            match acc with
            | Some (_, bk) when compare_key k bk >= 0 -> Err.return acc
            | _ -> Err.return (Some (p, k)))
          (Err.return None) ready
      in
      match best with
      | None -> assert false
      | Some (p, _) ->
          let* state = State.step state p in
          go state (evals + List.length ready) (max widest (List.length ready))
  in
  let* start = State.start problem in
  go start 0 0

module Candidate = struct
  type t = {
    strategy : Strategy.t;
    order : Position.t array;
    metrics : Metrics.t;
  }
end

let candidates problem =
  let open Err.Syntax in
  let identity = Array.init (Problem.node_count problem) Position.of_int in
  let* peak_order, s1 = schedule problem Policy.Peak_first in
  let* live_order, s2 = schedule problem Policy.Live_first in
  let distinct =
    List.fold_left
      (fun acc (strategy, order) ->
        if List.exists (fun (_, o) -> o = order) acc then acc
        else acc @ [ (strategy, order) ])
      []
      [
        (Strategy.Identity, identity);
        (Peak_first, peak_order);
        (Live_first, live_order);
      ]
  in
  let* cands =
    List.fold_left
      (fun acc (strategy, order) ->
        let* acc = acc in
        let* metrics = Problem.metrics problem order in
        Err.return (acc @ [ { Candidate.strategy; order; metrics } ]))
      (Err.return []) distinct
  in
  let stats =
    {
      Arena_schedule.Stats.zero with
      score_evaluations = s1.score_evaluations + s2.score_evaluations;
      depth = s1.depth;
      max_ready_width = max s1.max_ready_width s2.max_ready_width;
    }
  in
  Err.return (cands, stats)

let rank (c : Candidate.t) = (c.metrics.target_peak, c.metrics.all_peak)

let select problem cands stats =
  let open Err.Syntax in
  let better (c : Candidate.t) (best : Candidate.t) =
    let a1, a2 = rank c and b1, b2 = rank best in
    compare_key (a1, a2, Byte_size.zero) (b1, b2, Byte_size.zero) < 0
  in
  match cands with
  | [] -> Err.fail ~pos:__POS__ `Not_a_permutation
  | first :: rest ->
      let best =
        List.fold_left (fun b c -> if better c b then c else b) first rest
      in
      let* graph = Problem.reorder problem best.order in
      let* bound = Problem.lower_bound problem in
      let stop =
        if Byte_size.equal best.metrics.target_peak bound.target_peak then
          Arena_schedule.Stop.Lower_bound_reached
        else Completed
      in
      Err.return
        { Arena_schedule.Result.graph; strategy = best.strategy; stop; stats }

let run (c : Arena_schedule.Config.t) g =
  let open Err.Syntax in
  let* problem = Problem.of_graph c g in
  match (c.mode, c.retain) with
  | Intermediate, All ->
      (* Nothing is ever released, so no order changes the eligible peak. In
         role mode inputs and outputs still move the peak: no shortcut. *)
      Arena_schedule.identity g
  | _ ->
      let* cands, stats = candidates problem in
      select problem cands stats
