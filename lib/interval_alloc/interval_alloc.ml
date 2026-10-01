open Core.Storage_units
module Event = Ia_script.Event
module Block = Ia_check.Block
module Misaligned = Ia_check.Misaligned
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
let fit = Ia_reference.normalize

module Stats = struct
  type t = {
    lower_bound : Byte_size.t;
    constructive : Strategy.t * Byte_size.t;
    pool : Byte_size.t;
    iterations : int64;
    stop : Stop.t;
  }

  let pp ppf t =
    let strategy, cpool = t.constructive in
    Format.fprintf ppf
      "lower bound %a, constructive %a %a, pool %a, %Ld iterations (%s)"
      Byte_size.pp t.lower_bound Strategy.pp strategy Byte_size.pp cpool
      Byte_size.pp t.pool t.iterations
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
        let* result = Ia_decode.offsets mode script order in
        match best with
        | Some (_, _, _, (best_result : Ia_decode.Placed.t))
          when Byte_size.compare best_result.pool result.Ia_decode.Placed.pool
               <= 0 ->
            Ok best
        | _ -> Ok (Some (strategy, mode, order, result)))
      None Strategy.all
  in
  match best with Some best -> Ok best | None -> assert false

let solve_best_at ?at ~iterations ~seed (script : _ Script.t) =
  let* lb = lower_bound script in
  let* strategy, mode, order, start = portfolio script in
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
             constructive = (strategy, start.Ia_decode.Placed.pool);
             pool = result.Ia_decode.Placed.pool;
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
  (* Checked: every block ends inside the pool. *)
  let ends =
    Array.mapi
      (fun i o ->
        Ia_script.invariant (Byte_offset.advance o script.Ia_script.sizes.(i)))
      offsets
  in
  let order = Array.init n Fun.id in
  Array.stable_sort
    (fun a b -> Byte_offset.compare offsets.(a) offsets.(b))
    order;
  let* results =
    Ia_search.improve_at ?at Ia_decode.Mode.First_fit ~checkpoints:iterations
      ~seed script ~lower_bound:lb order
      { Ia_decode.Placed.offsets; ends; pool = Ia_solution.pool witness }
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
