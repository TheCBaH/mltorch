(* Bounded beam search: a graph both greedy policies miss and the beam solves,
   budget and state limits, determinism, and fixed-width monotonicity. *)

open Graph_ir
module P = Arena_schedule_problem
module G = Arena_schedule_greedy
module S = Arena_schedule_search
module M = Arena_schedule.Metrics
module B = Core.Storage_units.Byte_size

let pp_error ppf e = P.pp_error ppf (Err.Error.kind e)
let ok = function Ok v -> v | Error e -> Fmt.failwith "%a" pp_error e
let ok_l = function Ok v -> v | Error _ -> failwith "limits"

let config ?(limits = Arena_schedule.Limits.constructive_only) () =
  {
    Arena_schedule.Config.mode = Intermediate;
    retain = Only Tensor_id.Set.empty;
    alignment = Alignment_policy.standard;
    limits;
  }

let peak g = B.to_int64 (ok (P.fresh_metrics (config ()) g)).M.target_peak

let limits ?(state = 1_000_000_000L) ~width ~expansions () =
  Arena_schedule.Limits.make ~width ~expansions
    ~state_bytes:(Result.get_ok (Err.payload (B.of_int64 state)))
  |> Err.payload |> ok_l

let pos = P.Position.of_int

let all_orders ?(cap = 50_000) p =
  let n = P.node_count p in
  let found = ref [] and count = ref 0 in
  let placed = Array.make n false in
  let cur = Array.make n (pos 0) in
  let rec go i =
    if !count < cap then
      if i = n then (
        found := Array.copy cur :: !found;
        incr count)
      else
        for k = 0 to n - 1 do
          if
            (not placed.(k))
            && List.for_all
                 (fun q -> placed.((q : P.Position.t :> int)))
                 (P.preds p (pos k))
          then (
            placed.(k) <- true;
            cur.(i) <- pos k;
            go (i + 1);
            placed.(k) <- false)
        done
  in
  go 0;
  (!found, !count < cap)

let oracle p =
  let orders, complete = all_orders p in
  ( List.fold_left
      (fun acc o -> min acc (peak (ok (P.reorder p o))))
      Int64.max_int orders,
    complete )

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

let stop_name = function
  | Arena_schedule.Stop.Budget_exhausted -> "budget"
  | Completed -> "completed"
  | Lower_bound_reached -> "bound"
  | State_limit -> "state-limit"

let%expect_test "the beam solves graphs both greedy policies miss" =
  List.iter
    (fun seed ->
      let g = random_graph seed ~nodes:9 in
      let p = ok (P.of_graph (config ()) g) in
      let greedy = ok (G.run (config ()) g) in
      let best, _ = oracle p in
      let c = config ~limits:(limits ~width:8 ~expansions:100_000 ()) () in
      let r = ok (S.run c g) in
      Fmt.pr "seed %d: greedy %Ld beam %Ld optimum %Ld via %a (%s)@." seed
        (peak greedy.graph) (peak r.graph) best Arena_schedule.Strategy.pp
        r.strategy (stop_name r.stop))
    [ 5; 11; 124; 167; 8 ];
  [%expect
    {|
    seed 5: greedy 96 beam 68 optimum 68 via beam (completed)
    seed 11: greedy 96 beam 64 optimum 64 via beam (bound)
    seed 124: greedy 96 beam 64 optimum 64 via beam (bound)
    seed 167: greedy 128 beam 96 optimum 96 via beam (bound)
    seed 8: greedy 68 beam 68 optimum 64 via peak-first (completed) |}]

let%expect_test "budgets: monotone at fixed width, deterministic" =
  let g = random_graph 124 ~nodes:9 in
  let p = ok (P.of_graph (config ()) g) in
  let width = 4 in
  let run expansions =
    let c = config ~limits:(limits ~width ~expansions ()) () in
    ok (S.run c g)
  in
  let prev = ref Int64.max_int in
  List.iter
    (fun expansions ->
      let r = run expansions in
      let again = run expansions in
      let pk = peak r.graph in
      assert (pk <= !prev);
      prev := pk;
      let same =
        List.map (fun n -> n.Node.id) r.graph.Graph.nodes
        = List.map (fun n -> n.Node.id) again.graph.Graph.nodes
      in
      Fmt.pr "budget %d: peak %Ld used %d depth %d %s repeat=%b@." expansions pk
        r.stats.expansions r.stats.depth (stop_name r.stop) same)
    [ 0; 1; 10; 30; 60; 100; 1000 ];
  ignore p;
  [%expect
    {|
    budget 0: peak 96 used 0 depth 9 completed repeat=true
    budget 1: peak 96 used 1 depth 9 budget repeat=true
    budget 10: peak 96 used 5 depth 9 budget repeat=true
    budget 30: peak 96 used 20 depth 9 budget repeat=true
    budget 60: peak 96 used 53 depth 9 budget repeat=true
    budget 100: peak 64 used 69 depth 9 bound repeat=true
    budget 1000: peak 64 used 69 depth 9 bound repeat=true |}]

let%expect_test "state limit and invalid configuration" =
  let g = random_graph 124 ~nodes:9 in
  let tiny = limits ~state:1L ~width:8 ~expansions:1000 () in
  let r = ok (S.run (config ~limits:tiny ()) g) in
  Fmt.pr "%s expansions=%d@." (stop_name r.stop) r.stats.expansions;
  let p = ok (P.of_graph (config ()) g) in
  let o = ok (S.beam p (limits ~width:2 ~expansions:0 ())) in
  Fmt.pr "no budget: %b %s@." (o.order = None) (stop_name o.stop);
  [%expect {|
    state-limit expansions=0
    no budget: true completed |}]

let%expect_test "a budget ending exactly at the last expansion still completes"
    =
  let g = random_graph 124 ~nodes:9 in
  let p = ok (P.of_graph (config ()) g) in
  List.iter
    (fun expansions ->
      let o = ok (S.beam p (limits ~width:4 ~expansions ())) in
      Fmt.pr "budget %d: %s order=%b@." expansions (stop_name o.stop)
        (o.order <> None))
    [ 68; 69; 70 ];
  [%expect
    {|
    budget 68: budget order=false
    budget 69: completed order=true
    budget 70: completed order=true |}]
