(* See arena_eval.mli. *)

module Config = struct
  type t = {
    iterations : int64 list;
    seeds : int64 list;
    repeats : int;
    reference : Interval_alloc.Reference.Limits.t;
  }
end

module Method = struct
  type t =
    | Constructive of Interval_alloc.Strategy.t
    | Improved of Interval_alloc.Strategy.t
    | Portfolio

  let pp ppf = function
    | Constructive s -> Interval_alloc.Strategy.pp ppf s
    | Improved s -> Fmt.pf ppf "%a+improve" Interval_alloc.Strategy.pp s
    | Portfolio -> Fmt.string ppf "portfolio"
end

module Timing = struct
  type t = { construct : float; search : float; check : float }
end

module Row = struct
  type t = {
    method_ : Method.t;
    budget : (int64 * int64) option;
    constructive_pool : Core.Storage_units.Byte_size.t;
    pool : Core.Storage_units.Byte_size.t;
    effort : Interval_alloc.Effort.t option;
    placements : (Tensor_id.t * Core.Storage_units.Byte_offset.t) list;
    digest : string;
    timing : Timing.t;
  }
end

module Reference_row = struct
  type t = {
    bounds :
      (Tensor_id.t * Core.Storage_units.Byte_offset.t) list
      Interval_alloc.Reference.Bounds.t;
    states : int64;
    max_depth : int64;
    seconds : float;
    digest : string;
  }
end

type error =
  [ `Arena_placement of Arena_plan.Placement_error.t
  | `Invalid_candidate of Interval_alloc.Reference.Invalid_candidate.t ]

let digest placements =
  Digest.to_hex
    (Digest.string
       (String.concat ";"
          (List.map
             (fun (id, offset) ->
               Printf.sprintf "%d:%Ld" (Tensor_id.to_int id)
                 (Core.Storage_units.Byte_offset.to_int64 offset))
             placements)))

let placement e = `Arena_placement (e : Arena_plan.Placement_error.t)

(* The first run's result and the fastest of [repeats] runs' time. *)
let timed ~now ~repeats f =
  let open Err.Syntax in
  let once () =
    let start = now () in
    let+ r = f () in
    (r, now () -. start)
  in
  let* first, t = once () in
  let rec more best n =
    if n <= 1 then Err.return best
    else
      let* _, t = once () in
      more (Float.min best t) (n - 1)
  in
  let+ best = more t repeats in
  (first, best)

let checked ~now ~repeats script solution =
  timed ~now ~repeats (fun () ->
      Interval_alloc.check script solution
      |> Err.map_error ~pos:__POS__ (fun e -> placement e))

let budgets (config : Config.t) =
  List.concat_map
    (fun iterations -> List.map (fun seed -> (iterations, seed)) config.seeds)
    config.iterations

(* One checkpointed walk per seed, [repeats] times: its results, and at each
   checkpoint the fastest time since the walk started. [walk ~at] runs it,
   calling [at] at each checkpoint. *)
let walks ~now ~repeats (config : Config.t) walk =
  let open Err.Syntax in
  let once seed =
    let start = now () and times = ref [] in
    let+ results =
      walk ~seed ~at:(fun b -> times := (b, now () -. start) :: !times)
    in
    (results, !times)
  in
  Err.List.map
    (fun seed ->
      let* results, times = once seed in
      let rec more times n =
        if n <= 1 then Err.return times
        else
          let* _, t = once seed in
          more
            (List.map (fun (b, x) -> (b, Float.min x (List.assoc b t))) times)
            (n - 1)
      in
      let+ times = more times repeats in
      (seed, results, times))
    config.seeds

(* Rows in [budgets config] order, from per-seed walks. *)
let in_budget_order config per_seed row =
  Err.List.map
    (fun (iterations, seed) ->
      let _, results, times =
        List.find (fun (s, _, _) -> Int64.equal s seed) per_seed
      in
      let _, result, extra =
        List.find (fun (b, _, _) -> Int64.equal b iterations) results
      in
      row ~iterations ~seed ~time:(List.assoc iterations times) result extra)
    (budgets config)

let strategies ~now (config : Config.t) script =
  let open Err.Syntax in
  let repeats = config.repeats in
  let row ~method_ ~budget ~constructive_pool ~effort ~construct ~search
      solution =
    let+ witness, check = checked ~now ~repeats script solution in
    let placements = Interval_alloc.placements witness in
    {
      Row.method_;
      budget;
      constructive_pool;
      pool = Interval_alloc.pool witness;
      effort;
      placements;
      digest = digest placements;
      timing = { Timing.construct; search; check };
    }
  in
  let placement_error r = Err.map_error ~pos:__POS__ (fun e -> placement e) r in
  let per_strategy strategy =
    let* solution, construct =
      timed ~now ~repeats (fun () ->
          Interval_alloc.solve strategy script |> placement_error)
    in
    let constructive_pool = Interval_alloc.Solution.pool solution in
    let* base =
      row ~method_:(Method.Constructive strategy) ~budget:None
        ~constructive_pool ~effort:None ~construct ~search:0. solution
    in
    let* per_seed =
      walks ~now ~repeats config (fun ~seed ~at ->
          Interval_alloc.improve_at ~at ~iterations:config.iterations ~seed
            script solution
          |> placement_error)
    in
    let+ improved =
      in_budget_order config per_seed
        (fun ~iterations ~seed ~time solution effort ->
          row ~method_:(Method.Improved strategy)
            ~budget:(Some (iterations, seed))
            ~constructive_pool ~effort:(Some effort) ~construct ~search:time
            solution)
    in
    base :: improved
  in
  let* per = Err.List.map per_strategy Interval_alloc.Strategy.all in
  (* Checkpoint 0 is always taken, so the portfolio's constructive time is
     read off the same walk. *)
  let* per_seed =
    walks ~now ~repeats config (fun ~seed ~at ->
        Interval_alloc.solve_best_at ~at ~iterations:(0L :: config.iterations)
          ~seed script
        |> placement_error)
  in
  let+ portfolio =
    in_budget_order config per_seed
      (fun ~iterations ~seed ~time solution (stats : Interval_alloc.Stats.t) ->
        let _, _, times =
          List.find (fun (s, _, _) -> Int64.equal s seed) per_seed
        in
        let construct = List.assoc 0L times in
        row ~method_:Method.Portfolio
          ~budget:(Some (iterations, seed))
          ~constructive_pool:(snd stats.constructive)
          ~effort:
            (Some
               {
                 Interval_alloc.Effort.iterations = stats.iterations;
                 stop = stats.stop;
               })
          ~construct
          ~search:(Float.max 0. (time -. construct))
          solution)
  in
  List.concat per @ portfolio

let reference ~now (config : Config.t) script =
  let open Err.Syntax in
  let* incumbent, _ =
    Interval_alloc.solve_best
      ~budget:(Interval_alloc.Budget.create ~iterations:0L ~seed:0L)
      script
    |> Err.map_error ~pos:__POS__ (fun e -> placement e)
  in
  let start = now () in
  let* bounds, work =
    Interval_alloc.Reference.minimum config.reference script ~incumbent
    |> Err.map_error ~pos:__POS__ (function
      | `Invalid_candidate c -> `Invalid_candidate c
      | ( `Duplicate_placement _ | `Live_overflow _ | `Misaligned _
        | `Offset_overflow _ | `Out_of_pool _ | `Overlap _ | `Unknown_key _
        | `Unplaced _ ) as e ->
          placement e)
  in
  let seconds = now () -. start in
  let placements = Interval_alloc.placements bounds.incumbent in
  Err.return
    {
      Reference_row.bounds = { bounds with incumbent = placements };
      states = Interval_alloc.Reference.Work.states work;
      max_depth = Interval_alloc.Reference.Work.max_depth work;
      seconds;
      digest = digest placements;
    }

let placed (config : Config.t) problem =
  let budget =
    Interval_alloc.Budget.create
      ~iterations:(List.fold_left Int64.max 0L config.iterations)
      ~seed:(match config.seeds with s :: _ -> s | [] -> 0L)
  in
  Err.map
    (fun (witness, _) -> Interval_alloc.pool witness)
    (Arena_plan.place ~budget problem)
