module Event = Ia_script.Event
module Block = Ia_check.Block
module Negative_size = Ia_script.Negative_size
module Out_of_pool = Ia_check.Out_of_pool
module Overlap = Ia_check.Overlap

module Script = struct
  type 'k t = 'k Ia_script.t

  let validate = Ia_script.validate
  let events = Ia_script.events
end

module Strategy = Ia_strategy

module Budget = struct
  type t = Ia_search.Budget.t

  let create = Ia_search.Budget.create
end

module Stop = Ia_search.Stop
module Effort = Ia_search.Effort
module Solution = Ia_solution

module Witness = struct
  type 'k t = 'k Ia_solution.t
end

let lower_bound = Ia_script.lower_bound
let placements = Ia_solution.placements
let pool = Ia_solution.pool

let solve strategy (script : _ Script.t) =
  let mode, order = Ia_decode.plan strategy script in
  Result.map (Ia_decode.solution script) (Ia_decode.offsets mode script order)

let check = Ia_check.check

module Stats = struct
  type t = {
    lower_bound : int64;
    constructive : Strategy.t * int64;
    pool : int64;
    iterations : int64;
    stop : Stop.t;
  }

  let pp ppf t =
    let strategy, cpool = t.constructive in
    Format.fprintf ppf
      "lower bound %Ld, constructive %a %Ld, pool %Ld, %Ld iterations (%s)"
      t.lower_bound Strategy.pp strategy cpool t.pool t.iterations
      (match t.stop with
      | Stop.Lower_bound -> "at the bound"
      | Budget_exhausted -> "budget exhausted")
end

let default_budget = Budget.create ~iterations:50L ~seed:0L
let ( let* ) = Result.bind

(* The portfolio: the smallest pool, the first strategy on a tie. *)
let portfolio (script : _ Script.t) =
  let* best =
    Err.List.fold_left
      (fun best strategy ->
        let mode, order = Ia_decode.plan strategy script in
        let* ((_, pool) as result) = Ia_decode.offsets mode script order in
        match best with
        | Some (_, _, _, (_, best_pool)) when Int64.compare best_pool pool <= 0
          ->
            Ok best
        | _ -> Ok (Some (strategy, mode, order, result)))
      None Strategy.all
  in
  match best with Some best -> Ok best | None -> assert false

let solve_best_at ?at ~iterations ~seed (script : _ Script.t) =
  let* lb = lower_bound script in
  let* strategy, mode, order, ((_, cpool) as start) = portfolio script in
  let* results =
    Ia_search.improve_at ?at mode ~checkpoints:iterations ~seed script
      ~lower_bound:lb order start
  in
  Ok
    (List.map
       (fun (c, result, (effort : Effort.t)) ->
         ( c,
           Ia_decode.solution script result,
           {
             Stats.lower_bound = lb;
             constructive = (strategy, cpool);
             pool = snd result;
             iterations = effort.iterations;
             stop = effort.stop;
           } ))
       results)

let solve_best ?(budget = default_budget) (script : _ Script.t) =
  let* results =
    solve_best_at
      ~iterations:[ budget.Ia_search.Budget.iterations ]
      ~seed:budget.Ia_search.Budget.seed script
  in
  match results with
  | [ (_, solution, stats) ] -> Ok (solution, stats)
  | _ -> assert false

let improve_at ?at ~iterations ~seed (script : _ Script.t) solution =
  let* witness = check script solution in
  let* lb = lower_bound script in
  let n = Ia_script.blocks script in
  let offsets = Array.of_list (List.map snd (Ia_solution.placements witness)) in
  let order = Array.init n Fun.id in
  Array.stable_sort (fun a b -> Int64.compare offsets.(a) offsets.(b)) order;
  let* results =
    Ia_search.improve_at ?at Ia_decode.Mode.First_fit ~checkpoints:iterations
      ~seed script ~lower_bound:lb order
      (offsets, Ia_solution.pool witness)
  in
  Ok
    (List.map
       (fun (c, result, effort) ->
         (c, Ia_decode.solution script result, effort))
       results)

let improve budget (script : _ Script.t) solution =
  let* results =
    improve_at
      ~iterations:[ budget.Ia_search.Budget.iterations ]
      ~seed:budget.Ia_search.Budget.seed script solution
  in
  match results with
  | [ (_, solution, effort) ] -> Ok (solution, effort)
  | _ -> assert false

module Reference = struct
  module Limits = Ia_reference.Limits

  module Work = struct
    type t = Ia_reference.Work.t

    let create = Ia_reference.Work.create
    let states = Ia_reference.Work.states
    let max_depth = Ia_reference.Work.max_depth
  end

  module Cut = Ia_reference.Cut
  module Answer = Ia_reference.Answer
  module Invalid_candidate = Ia_reference.Invalid_candidate
  module Status = Ia_reference.Status
  module Stop = Ia_reference.Stop
  module Bounds = Ia_reference.Bounds

  let feasible = Ia_reference.feasible
  let bisect = Ia_reference.bisect
  let minimum = Ia_reference.minimum
end
