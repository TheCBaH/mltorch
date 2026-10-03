(* Choosing by witnessed pool bytes: never larger than the original, the plan
   always belongs to the returned graph, admission rescues and fallbacks. *)

open Graph_ir
module P = Arena_schedule_problem
module Plan = Arena_schedule_plan
module M = Arena_schedule.Metrics
module B = Core.Storage_units.Byte_size

let pp_error ppf e = Plan.pp_error ppf (Err.Error.kind e)
let ok = function Ok v -> v | Error e -> Fmt.failwith "%a" pp_error e

let config ?(mode = Arena_schedule.Mode.Intermediate)
    ?(alignment = Alignment_policy.standard)
    ?(limits = Arena_schedule.Limits.default_beam) () =
  {
    Arena_schedule.Config.mode;
    retain = Only Tensor_id.Set.empty;
    alignment;
    limits;
  }

let random_graph seed ~nodes =
  let rng = Random.State.make [| seed |] in
  let mp = { Reduce.Mean.dims = [ Axis.C ]; keepdim = true } in
  Graph_fixtures.build (Fmt.str "random%d" seed)
    Graph_builder.(
      let* x = input ~shape:(Graph_fixtures.s1c 8) () in
      let rec go i wide narrow last =
        if i = nodes then return last
        else
          let pick l = List.nth l (Random.State.int rng (List.length l)) in
          match Random.State.int rng 3 with
          | 0 ->
              let* t = relu (pick wide) in
              go (i + 1) (t :: wide) narrow t
          | 1 ->
              let* t = Graph_builder.mean mp (pick wide) in
              go (i + 1) wide (t :: narrow) t
          | _ ->
              let src = if narrow = [] then wide else narrow in
              let* t = add (pick src) (pick src) in
              if src == narrow then go (i + 1) wide (t :: narrow) t
              else go (i + 1) (t :: wide) narrow t
      in
      go 0 [ x ] [] x)

let i64 = function Some b -> B.to_int64 b | None -> -1L
let strategy = Fmt.to_to_string Arena_schedule.Strategy.pp

let%expect_test "payload and pool disagree: the smaller witnessed pool wins" =
  (* Seed 5: the beam has the lowest payload peak but a larger pool than the
     original, which stays. *)
  let g = random_graph 5 ~nodes:9 in
  let s = ok (Plan.choose (config ()) g) in
  let payload =
    List.find
      (fun (r : Plan.Report.t) -> r.strategy = s.payload_winner)
      s.reports
  in
  Fmt.pr "chosen %s pool %Ld; payload winner %s pool %Ld@."
    (strategy s.strategy) (i64 s.pool_bytes)
    (strategy s.payload_winner)
    (i64 payload.pool_bytes);
  Fmt.pr "payload peaks: chosen %Ld, winner %Ld@."
    (B.to_int64 s.metrics.target_peak)
    (B.to_int64 payload.metrics.target_peak);
  [%expect
    {|
    chosen identity pool 160; payload winner beam pool 196
    payload peaks: chosen 96, winner 68 |}]

let script_of_plan = function
  | Some (Plan.Plan.Intermediate p) -> `Alloc (Arena_plan.script p)
  | Some (Roles p) -> `Storage (Storage_plan.script p)
  | None -> `None

let plan_matches_graph (c : Arena_schedule.Config.t) (s : Plan.Selection.t) =
  match (script_of_plan s.plan, c.mode) with
  | `Alloc script, Intermediate ->
      let fresh =
        Result.get_ok
          (Err.payload
             (Eval_direct.dry_run ~alignment:c.alignment ~retain:c.retain
                s.graph))
      in
      Alloc_script.first_difference script fresh = None
  | `Storage script, Roles cfg ->
      let fresh =
        Result.get_ok
          (Err.payload
             (Eval_direct.storage_script ~alignment:c.alignment ~retain:c.retain
                cfg s.graph))
      in
      Storage_script.first_difference script fresh = None
  | _ -> false

let%expect_test "never larger than the original; the plan belongs to the graph"
    =
  let roles =
    List.concat_map
      (fun layout ->
        List.map
          (fun inputs ->
            Arena_schedule.Mode.Roles
              { Storage_script.Config.layout; constants = Borrowed; inputs })
          Storage_script.Ownership.[ Borrowed; Copied ])
      Storage_script.Layout.[ Separate; Shared_execution ]
  in
  let graphs =
    Graph_fixtures.all |> List.map (fun (n, b) -> (n, b ())) |> fun l ->
    l @ List.init 12 (fun i -> ("random", random_graph (i + 1) ~nodes:8))
  in
  let bad = ref 0 and smaller = ref 0 and total = ref 0 in
  List.iter
    (fun (name, g) ->
      List.iter
        (fun mode ->
          List.iter
            (fun alignment ->
              let c = config ~mode ~alignment () in
              let s = ok (Plan.choose c g) in
              incr total;
              ok (Arena_schedule.check_permutation ~original:g s.graph);
              let le a b = B.compare a b <= 0 in
              let pool_ok =
                match (s.pool_bytes, s.baseline_pool_bytes) with
                | Some a, Some b -> le a b
                | _ -> false
              in
              if
                not
                  (pool_ok
                  && le s.metrics.target_peak s.baseline.target_peak
                  && plan_matches_graph c s)
              then (
                incr bad;
                Fmt.pr "%s: violated@." name);
              if i64 s.pool_bytes < i64 s.baseline_pool_bytes then incr smaller)
            Alignment_policy.[ standard; with_host page_alignment ])
        (Arena_schedule.Mode.Intermediate :: roles))
    graphs;
  Fmt.pr "bad=%d total>100=%b smaller>0=%b@." !bad (!total > 100) (!smaller > 0);
  [%expect {| bad=0 total>100=true smaller>0=true |}]

let footprint = function
  | Some (Plan.Plan.Intermediate p) ->
      Result.get_ok (Err.payload (Arena.footprint p))
  | _ -> assert false

let%expect_test "admission: a candidate rescues a refused original; fallbacks" =
  let g = random_graph 4 ~nodes:9 in
  let free = ok (Plan.choose (config ()) g) in
  let fits = footprint free.plan in
  Fmt.pr "chosen %s, pool %Ld vs baseline %Ld@." (strategy free.strategy)
    (i64 free.pool_bytes)
    (i64 free.baseline_pool_bytes);
  (* Required with exactly the chosen footprint: the original does not fit. *)
  let s = ok (Plan.choose ~admission:(Required fits) (config ()) g) in
  Fmt.pr "rescue: %s, original %s@." (strategy s.strategy)
    (match (List.hd s.reports).verdict with
    | Planned _ -> "planned"
    | Refused e -> Fmt.str "refused (%a)" Plan.pp_error e
    | Skipped -> "skipped");
  (* One byte short for every candidate. *)
  let short = B.of_int64 0L |> Err.payload |> Result.get_ok in
  (match Plan.choose ~admission:(Required short) (config ()) g with
  | Ok _ -> Fmt.pr "required zero: unexpectedly ok@."
  | Error e -> Fmt.pr "required zero: %a@." pp_error e);
  (* A pool ceiling no order can meet: Best_effort declines to the original. *)
  let l = Kernel.Limits.default in
  let tiny =
    Kernel.Limits.create ~max_size:l.max_size ~max_depth:l.max_depth
      ~max_values:l.max_values ~max_dep_depth:l.max_dep_depth
      ~max_inputs:l.max_inputs ~max_outputs:l.max_outputs
      ~max_extent:l.max_extent ~max_numel:l.max_numel ~max_bytes:8L
      ~max_local_slots:l.max_local_slots ~max_scan_state:l.max_scan_state
      ~max_scan_updates_per_key:l.max_scan_updates_per_key
      ~max_scan_updates_total:l.max_scan_updates_total
    |> Err.payload |> Result.get_ok
  in
  let d = ok (Plan.choose ~limits:tiny (config ()) g) in
  Fmt.pr "best-effort: plan=%b original=%b declined=%b@." (d.plan <> None)
    (d.graph == g) (d.declined <> None);
  (match Plan.choose ~limits:tiny ~admission:(Required fits) (config ()) g with
  | Ok _ -> Fmt.pr "required+ceiling: unexpectedly ok@."
  | Error e -> Fmt.pr "required+ceiling: %a@." pp_error e);
  [%expect
    {|
    chosen peak-first, pool 96 vs baseline 260
    rescue: peak-first, original refused (arena: the run needs 264 bytes but the budget is 100)
    required zero: arena: the run needs 264 bytes but the budget is 0
    best-effort: plan=false original=true declined=true
    required+ceiling: arena: the float32 pool needs 65 cells (260 bytes), over 8 bytes |}]
