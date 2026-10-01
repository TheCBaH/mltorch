open Interval_alloc

let script_of_events events =
  match Script.validate ~equal:Int.equal events with
  | Ok s -> s
  | Error _ -> assert false

let ok = function Ok x -> x | Error _ -> assert false
let budget ?(seed = 1L) n = Budget.create ~iterations:n ~seed

let pp_stop ppf = function
  | Stop.Budget_exhausted -> Fmt.string ppf "budget exhausted"
  | Stop.Lower_bound -> Fmt.string ppf "at the bound"

(* A script where every constructive strategy is above the bound and the search
   finds a smaller pool than all four. *)
let beaten =
  let a k size = Event.Alloc { key = k; size } and f k = Event.Free k in
  [
    a 0 7L;
    a 1 1L;
    a 2 0L;
    f 2;
    a 3 4L;
    a 4 1L;
    a 5 2L;
    a 6 8L;
    a 7 3L;
    f 3;
    a 8 2L;
    f 0;
    a 9 9L;
    f 8;
    a 10 0L;
    f 7;
    f 10;
    f 9;
    f 1;
    f 5;
    f 6;
  ]

let%expect_test "search strictly beats every constructive strategy" =
  let s = script_of_events beaten in
  List.iter
    (fun strategy ->
      Fmt.pr "%a: %Ld@." Strategy.pp strategy
        (Solution.pool (ok (solve strategy s))))
    Strategy.all;
  let sol, stats = ok (solve_best ~budget:(budget 100L) s) in
  Fmt.pr "%a@." Stats.pp stats;
  Fmt.pr "checked: %b@." (Result.is_ok (check s sol));
  [%expect
    {|
    greedy_by_area: 28
    greedy_by_lifetime: 33
    greedy_by_size: 28
    greedy_by_size_best_fit: 28
    lower bound 26, constructive greedy_by_area 28, pool 26, 5 iterations (at the bound)
    checked: true |}]

let%expect_test "budget and bound" =
  let run name events b =
    let s = script_of_events events in
    let _, st = ok (solve_best ~budget:b s) in
    Fmt.pr "%s: %Ld iterations, %a, pool %Ld, bound %Ld@." name st.iterations
      pp_stop st.stop st.pool st.lower_bound
  in
  let a k size = Event.Alloc { key = k; size } and f k = Event.Free k in
  run "budget 0, above the bound" beaten (budget 0L);
  run "budget 0, at the bound" [ a 0 8L ] (budget 0L);
  run "budget 100, constructive at the bound"
    [ a 0 4L; f 0; a 1 6L; f 1 ]
    (budget 100L);
  run "budget 100, bound reached partway" beaten (budget 100L);
  run "negative budget" beaten (budget (-5L));
  [%expect
    {|
    budget 0, above the bound: 0 iterations, budget exhausted, pool 28, bound 26
    budget 0, at the bound: 0 iterations, at the bound, pool 8, bound 8
    budget 100, constructive at the bound: 0 iterations, at the bound, pool 6, bound 6
    budget 100, bound reached partway: 5 iterations, at the bound, pool 26, bound 26
    negative budget: 0 iterations, budget exhausted, pool 28, bound 26 |}]

let random_scripts ~seed ~count ~size ~max_size =
  let g = Gen.make seed in
  List.init count (fun _ ->
      script_of_events (Gen.script g ~n:(3 + Gen.below g size) ~max_size))

let%expect_test "random scripts: search is sound and never grows a pool" =
  let scripts = random_scripts ~seed:11 ~count:200 ~size:80 ~max_size:50 in
  let before = ref 0. and after = ref 0. and worst = ref 0. in
  let optimal = ref 0 and total = ref 0 in
  List.iteri
    (fun i s ->
      let lb = ok (lower_bound s) in
      let best_constructive =
        List.fold_left
          (fun m st -> Int64.min m (Solution.pool (ok (solve st s))))
          Int64.max_int Strategy.all
      in
      let sol, st = ok (solve_best ~budget:(budget 30L) s) in
      if Result.is_error (check s sol) then Fmt.pr "script %d: fails check@." i;
      if Solution.pool sol < lb then Fmt.pr "script %d: below bound@." i;
      if Solution.pool sol > best_constructive then
        Fmt.pr "script %d: worse than constructive@." i;
      if st.iterations > 30L then Fmt.pr "script %d: over budget@." i;
      if st.stop = Stop.Lower_bound <> (st.pool = lb) then
        Fmt.pr "script %d: stop disagrees with bound@." i;
      (* [improve] from each strategy's solution never grows it. *)
      List.iter
        (fun strategy ->
          let start = ok (solve strategy s) in
          let better, _ = ok (improve (budget 10L) s start) in
          if Solution.pool better > Solution.pool start then
            Fmt.pr "script %d: improve grew a pool@." i;
          if Result.is_error (check s better) then
            Fmt.pr "script %d: improve fails check@." i)
        Strategy.all;
      if lb > 0L then begin
        let r x = Int64.to_float x /. Int64.to_float lb in
        before := !before +. r best_constructive;
        after := !after +. r (Solution.pool sol);
        worst := Float.max !worst (r (Solution.pool sol));
        incr total;
        if st.stop = Stop.Lower_bound then incr optimal
      end)
    scripts;
  let mean x = x /. float_of_int !total in
  Fmt.pr
    "scripts %d: mean pool/bound %.3f -> %.3f (max %.3f), %d at the bound@."
    !total (mean !before) (mean !after) !worst !optimal;
  [%expect
    {| scripts 200: mean pool/bound 1.008 -> 1.005 (max 1.060), 137 at the bound |}]

let%expect_test
    "same budget and seed, same offsets; a bigger budget never loses" =
  List.iteri
    (fun i s ->
      let run b = Solution.placements (fst (ok (solve_best ~budget:b s))) in
      if run (budget 25L) <> run (budget 25L) then
        Fmt.pr "script %d: not deterministic@." i;
      let pools =
        List.map
          (fun n -> Solution.pool (fst (ok (solve_best ~budget:(budget n) s))))
          [ 0L; 5L; 20L; 80L ]
      in
      if pools <> List.sort (fun a b -> Int64.compare b a) pools then
        Fmt.pr "script %d: budget not monotone@." i)
    (random_scripts ~seed:5 ~count:60 ~size:60 ~max_size:40);
  Fmt.pr "done@.";
  [%expect {| done |}]

(* Test-only exact oracle. Some optimal placement puts each block, taken in
   offset order, at the lowest offset that clears the blocks placed before it,
   so trying every order is complete. Independent of the library's decoder. *)
let optimum events =
  let blocks =
    let alloc = Hashtbl.create 8 and free = Hashtbl.create 8 in
    List.iteri
      (fun pos -> function
        | Event.Alloc { key; size } -> Hashtbl.replace alloc key (pos, size)
        | Event.Free key -> Hashtbl.replace free key pos)
      events;
    let length = List.length events in
    Hashtbl.fold
      (fun key (a, size) acc ->
        let f = Option.value (Hashtbl.find_opt free key) ~default:length in
        (a, f, Int64.to_int size) :: acc)
      alloc []
    |> List.filter (fun (_, _, size) -> size > 0)
    |> Array.of_list
  in
  let n = Array.length blocks in
  let best = ref max_int in
  let offsets = Array.make n (-1) in
  let clash i j =
    let a, f, _ = blocks.(i) and a', f', _ = blocks.(j) in
    a < f' && a' < f
  in
  let rec go placed pool =
    if pool < !best then
      if placed = n then best := pool
      else
        for b = 0 to n - 1 do
          if offsets.(b) < 0 then begin
            let _, _, size = blocks.(b) in
            let busy =
              List.filter_map
                (fun j ->
                  if offsets.(j) >= 0 && clash b j then
                    let _, _, sj = blocks.(j) in
                    Some (offsets.(j), offsets.(j) + sj)
                  else None)
                (List.init n Fun.id)
              |> List.sort compare
            in
            let off =
              List.fold_left
                (fun cursor (lo, hi) ->
                  if lo - cursor >= size then cursor else max cursor hi)
                0 busy
            in
            offsets.(b) <- off;
            go (placed + 1) (max pool (off + size));
            offsets.(b) <- -1
          end
        done
  in
  go 0 0;
  Int64.of_int !best

let%expect_test "against the exact optimum on small scripts" =
  let g = Gen.make 23 in
  let gaps = ref [] and bad = ref 0 and proven = ref 0 in
  for i = 1 to 300 do
    let events = Gen.script g ~n:(2 + Gen.below g 6) ~max_size:12 in
    let s = script_of_events events in
    let opt = optimum events in
    let sol, st = ok (solve_best ~budget:(budget 50L) s) in
    if Solution.pool sol < opt then begin
      incr bad;
      Fmt.pr "script %d: below the optimum@." i
    end;
    if st.stop = Stop.Lower_bound then begin
      incr proven;
      if st.pool <> opt then Fmt.pr "script %d: bound claimed, not optimal@." i
    end;
    if opt > 0L then
      gaps :=
        (Int64.to_float (Solution.pool sol) /. Int64.to_float opt) :: !gaps
  done;
  let n = List.length !gaps in
  Fmt.pr
    "scripts %d: mean pool/optimum %.3f, max %.3f, %d proven at the bound@." n
    (List.fold_left ( +. ) 0. !gaps /. float_of_int n)
    (List.fold_left Float.max 0. !gaps)
    !proven;
  [%expect
    {| scripts 299: mean pool/optimum 1.000, max 1.029, 299 proven at the bound |}]

(* One checkpointed walk equals a separate run at each budget: same placement,
   same effort. The grid is unsorted and repeats a budget on purpose. *)
let%expect_test "checkpoints equal separate runs" =
  let grid = [ 40L; 0L; 5L; 5L; 200L; 17L ] in
  let bad = ref 0 and checked = ref 0 in
  List.iteri
    (fun i s ->
      List.iter
        (fun seed ->
          let times = ref [] in
          let at_best =
            ok
              (solve_best_at
                 ~at:(fun b -> times := b :: !times)
                 ~iterations:grid ~seed s)
          in
          if List.rev !times <> List.map (fun (b, _, _) -> b) at_best then
            incr bad;
          List.iter
            (fun (b, sol, (st : Stats.t)) ->
              incr checked;
              let sol', st' =
                ok (solve_best ~budget:(Budget.create ~iterations:b ~seed) s)
              in
              if
                Solution.placements sol <> Solution.placements sol' || st <> st'
              then begin
                incr bad;
                Fmt.pr "script %d seed %Ld budget %Ld: solve_best differs@." i
                  seed b
              end)
            at_best;
          let start = fst (ok (solve_best ~budget:(budget 0L) s)) in
          List.iter
            (fun (b, sol, eff) ->
              incr checked;
              let sol', eff' =
                ok (improve (Budget.create ~iterations:b ~seed) s start)
              in
              if
                Solution.placements sol <> Solution.placements sol'
                || eff <> eff'
              then begin
                incr bad;
                Fmt.pr "script %d seed %Ld budget %Ld: improve differs@." i seed
                  b
              end)
            (ok (improve_at ~iterations:grid ~seed s start)))
        [ 1L; 7L ])
    (random_scripts ~seed:9 ~count:40 ~size:50 ~max_size:40);
  Fmt.pr "checked %d, differing %d@." !checked !bad;
  [%expect {| checked 800, differing 0 |}]
