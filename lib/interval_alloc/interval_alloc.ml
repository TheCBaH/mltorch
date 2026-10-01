module Event = Ia_script.Event

module Block = struct
  type 'k t = { key : 'k; offset : int64; size : int64 }
end

module Negative_size = Ia_script.Negative_size

module Out_of_pool = struct
  type 'k t = { block : 'k Block.t; pool : int64 }
end

module Overlap = struct
  type 'k t = { first : 'k Block.t; second : 'k Block.t }
end

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

let check (script : _ Script.t) solution =
  let n = Ia_script.blocks script in
  let keys = script.Ia_script.keys and sizes = script.Ia_script.sizes in
  let offsets = Array.make n None in
  let ( let* ) = Result.bind in
  let* () =
    Err.List.iter
      (fun (k, offset) ->
        match Ia_script.find_index script.Ia_script.equal keys n k with
        | None -> Err.fail ~pos:__POS__ (`Unknown_key k)
        | Some i -> (
            match offsets.(i) with
            | Some _ -> Err.fail ~pos:__POS__ (`Duplicate_placement k)
            | None ->
                offsets.(i) <- Some offset;
                Ok ()))
      (Ia_solution.placements solution)
  in
  let pool = Ia_solution.pool solution in
  let offset_of i =
    match offsets.(i) with Some o -> o | None -> assert false
  in
  let block i =
    { Block.key = keys.(i); offset = offset_of i; size = sizes.(i) }
  in
  let* () =
    let rec unplaced i =
      if i >= n then Ok ()
      else if offsets.(i) = None then Err.fail ~pos:__POS__ (`Unplaced keys.(i))
      else unplaced (i + 1)
    in
    unplaced 0
  in
  let* () =
    let rec bounds i =
      if i >= n then Ok ()
      else
        let offset = offset_of i in
        if Int64.compare offset 0L < 0 then
          Err.fail ~pos:__POS__ (`Negative_offset keys.(i))
        else
          match Ia_script.add_checked offset sizes.(i) with
          | None -> Err.fail ~pos:__POS__ (`Offset_overflow keys.(i))
          | Some hi ->
              if Int64.compare hi pool > 0 then
                Err.fail ~pos:__POS__
                  (`Out_of_pool { Out_of_pool.block = block i; pool })
              else bounds (i + 1)
    in
    bounds 0
  in
  let disjoint a b =
    let lo_a = offset_of a and lo_b = offset_of b in
    Int64.compare (Int64.add lo_a sizes.(a)) lo_b <= 0
    || Int64.compare (Int64.add lo_b sizes.(b)) lo_a <= 0
  in
  let rec overlaps a b =
    if a >= n then Ok ()
    else if b >= n then overlaps (a + 1) (a + 2)
    else if
      sizes.(a) <> 0L
      && sizes.(b) <> 0L
      && Ia_script.conflicts script a b
      && not (disjoint a b)
    then
      Err.fail ~pos:__POS__
        (`Overlap { Overlap.first = block a; second = block b })
    else overlaps a (b + 1)
  in
  let* () = overlaps 0 1 in
  Ok solution

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

let solve_best ?(budget = default_budget) (script : _ Script.t) =
  let* lb = lower_bound script in
  (* The portfolio: the smallest pool, the first strategy on a tie. *)
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
  match best with
  | None -> assert false
  | Some (strategy, mode, order, ((_, cpool) as start)) ->
      let* result, (effort : Effort.t) =
        Ia_search.improve mode budget script ~lower_bound:lb order start
      in
      Ok
        ( Ia_decode.solution script result,
          {
            Stats.lower_bound = lb;
            constructive = (strategy, cpool);
            pool = snd result;
            iterations = effort.iterations;
            stop = effort.stop;
          } )

let improve budget (script : _ Script.t) solution =
  let* witness = check script solution in
  let* lb = lower_bound script in
  let n = Ia_script.blocks script in
  let offsets = Array.of_list (List.map snd (Ia_solution.placements witness)) in
  let order = Array.init n Fun.id in
  Array.stable_sort (fun a b -> Int64.compare offsets.(a) offsets.(b)) order;
  let* result, effort =
    Ia_search.improve Ia_decode.Mode.First_fit budget script ~lower_bound:lb
      order
      (offsets, Ia_solution.pool witness)
  in
  Ok (Ia_decode.solution script result, effort)
