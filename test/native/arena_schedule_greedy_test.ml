(* The constructive schedules, against an independent exhaustive oracle: every
   valid order of a tiny graph, scored by a fresh dry run of the reordered
   graph rather than by the incremental engine. *)

open Graph_ir
module P = Arena_schedule_problem
module G = Arena_schedule_greedy
module M = Arena_schedule.Metrics
module B = Core.Storage_units.Byte_size

let only_empty = Release_schedule.Retain.Only Tensor_id.Set.empty
let pp_error ppf e = P.pp_error ppf (Err.Error.kind e)
let ok = function Ok v -> v | Error e -> Fmt.failwith "%a" pp_error e

let config ?(retain = only_empty) () =
  {
    Arena_schedule.Config.mode = Intermediate;
    retain;
    alignment = Alignment_policy.standard;
    limits = Arena_schedule.Limits.constructive_only;
  }

let peak g = B.to_int64 (ok (P.fresh_metrics (config ()) g)).M.target_peak

(* All topological orders, as position arrays, up to [cap]. *)
let all_orders ?(cap = 20_000) p =
  let n = P.node_count p in
  let found = ref [] and count = ref 0 in
  let placed = Array.make n false in
  let cur = Array.make n (P.Position.of_int 0) in
  let rec go i =
    if !count < cap then
      if i = n then (
        found := Array.copy cur :: !found;
        incr count)
      else
        for k = 0 to n - 1 do
          let pk = P.Position.of_int k in
          if
            (not placed.(k))
            && List.for_all
                 (fun q -> placed.((q : P.Position.t :> int)))
                 (P.preds p pk)
          then (
            placed.(k) <- true;
            cur.(i) <- pk;
            go (i + 1);
            placed.(k) <- false)
        done
  in
  go 0;
  (List.rev !found, !count < cap)

let oracle p =
  let orders, complete = all_orders p in
  let best =
    List.fold_left
      (fun acc o -> min acc (peak (ok (P.reorder p o))))
      Int64.max_int orders
  in
  (best, complete)

let branches () =
  Graph_fixtures.build "branches"
    Graph_builder.(
      let* x = input ~shape:(Graph_fixtures.s1c 8) () in
      let mp = { Reduce.Mean.dims = [ Axis.C ]; keepdim = true } in
      let* big_a = relu x in
      let* a = Graph_builder.mean mp big_a in
      let* big_b = sigmoid x in
      let* b = Graph_builder.mean mp big_b in
      add a b)

(* A seeded random DAG over 8-wide and 1-wide tensors. *)
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

let%expect_test "fixtures: valid, never worse, at least the bound" =
  let c = config () in
  List.iter
    (fun (name, build) ->
      let g = build () in
      let r = ok (G.run c g) in
      ok (Arena_schedule.check_permutation ~original:g r.graph);
      let before = peak g and after = peak r.graph in
      let bound =
        B.to_int64 (ok (P.lower_bound (ok (P.of_graph c g)))).M.target_peak
      in
      if after > before || after < bound then
        Fmt.pr "%s: before %Ld after %Ld bound %Ld@." name before after bound)
    (("branches", branches) :: Graph_fixtures.all);
  [%expect {||}]

let%expect_test "the two-branch example improves 68 to 40; bound 36 is strict" =
  let c = config () in
  (* A, B, a, b, join: both big tensors live at once. *)
  let g =
    let p = ok (P.of_graph c (branches ())) in
    ok (P.reorder p (Array.map P.Position.of_int [| 0; 2; 1; 3; 4 |]))
  in
  let r = ok (G.run c g) in
  Fmt.pr "%Ld -> %Ld via %a, %s@." (peak g) (peak r.graph)
    Arena_schedule.Strategy.pp r.strategy
    (match r.stop with
    | Lower_bound_reached -> "bound reached"
    | Completed -> "completed"
    | Budget_exhausted -> "budget"
    | State_limit -> "state limit");
  let best, complete = oracle (ok (P.of_graph c g)) in
  Fmt.pr "oracle %Ld complete=%b@." best complete;
  [%expect
    {|
    68 -> 40 via peak-first, completed
    oracle 40 complete=true |}]

let%expect_test "deterministic, and Retain.All is the identity" =
  let g = branches () in
  let order r =
    List.map (fun n -> n.Node.id) r.Arena_schedule.Result.graph.Graph.nodes
  in
  let a = ok (G.run (config ()) g) and b = ok (G.run (config ()) g) in
  Fmt.pr "%b@." (order a = order b);
  let all = ok (G.run (config ~retain:All ()) g) in
  Fmt.pr "%b %a@." (all.graph == g) Arena_schedule.Strategy.pp all.strategy;
  [%expect {|
    true
    true identity |}]

let%expect_test "seeded tiny graphs: greedy against the exhaustive oracle" =
  let c = config () in
  let worse = ref 0 and equal = ref 0 and better = ref 0 and strict = ref 0 in
  let unresolved = ref 0 and optimal = ref 0 in
  for seed = 1 to 40 do
    let g = random_graph seed ~nodes:7 in
    let p = ok (P.of_graph c g) in
    let r = ok (G.run c g) in
    let sel = peak r.graph and orig = peak g in
    let best, complete = oracle p in
    let bound = B.to_int64 (ok (P.lower_bound p)).M.target_peak in
    if not complete then incr unresolved
    else (
      if best < bound then
        Fmt.pr "seed %d: bound %Ld above optimum %Ld@." seed bound best;
      if sel < best then
        Fmt.pr "seed %d: selected %Ld below optimum %Ld@." seed sel best;
      if best > bound then incr strict;
      if sel = best then incr optimal);
    if sel > orig then incr worse
    else if sel = orig then incr equal
    else incr better
  done;
  Fmt.pr
    "worse=%d equal=%d better=%d optimal=%d strict-bound=%d unresolved=%d@."
    !worse !equal !better !optimal !strict !unresolved;
  [%expect
    {| worse=0 equal=5 better=35 optimal=38 strict-bound=2 unresolved=0 |}]
