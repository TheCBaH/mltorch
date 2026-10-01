(* [Arena_eval]: every strategy row is present and checked, a search never
   loses to its start, and the same budget and seed replay the same placement. *)

module IA = Interval_alloc

let t = Tensor_id.of_int

(* A fake clock: each reading is one second after the last. *)
let clock () =
  let now = ref 0. in
  fun () ->
    now := !now +. 1.;
    !now

let config ?(iterations = [ 0L; 5L; 50L ]) ?(seeds = [ 1L; 2L ]) () =
  {
    Arena_eval.Config.iterations;
    seeds;
    repeats = 2;
    reference =
      { IA.Reference.Limits.max_states = 100_000L; max_depth = 1_000L };
  }

let ok = function Ok x -> x | Error _ -> Fmt.failwith "unexpected error"
let script events = ok (IA.Script.validate ~equal:Tensor_id.equal events)

(* Test literals, meant valid: a refusal is a broken test. *)
let units r = Err.or_raise ~pp_error:Core.Storage_units.pp_error r
let bytes v = units (Core.Storage_units.Byte_size.of_int64 v)
let one = units (Core.Storage_units.Byte_alignment.of_int64 1L)
let sz = Core.Storage_units.Byte_size.to_int64

(* A seeded random script over tensor ids, identical on every run. *)
let random ~seed ~n =
  let s = ref (Int64.of_int seed) in
  let below bound =
    s := Int64.add (Int64.mul !s 6364136223846793005L) 1442695040888963407L;
    Int64.to_int (Int64.shift_right_logical !s 33) mod bound
  in
  let live = ref [] and events = ref [] and next = ref 0 in
  while !next < n || !live <> [] do
    if !next < n && (!live = [] || below 3 <> 0) then begin
      events :=
        IA.Event.Alloc
          {
            key = t !next;
            size = bytes (Int64.of_int (1 + below 40));
            alignment = one;
          }
        :: !events;
      live := !next :: !live;
      incr next
    end
    else begin
      let k = List.nth !live (below (List.length !live)) in
      live := List.filter (( <> ) k) !live;
      events := IA.Event.Free (t k) :: !events
    end
  done;
  script (List.rev !events)

let pp_budget ppf = function
  | None -> Fmt.string ppf "-"
  | Some (i, s) -> Fmt.pf ppf "%Ld/%Ld" i s

let%expect_test "the matrix: every strategy, improved, and the portfolio" =
  let rows =
    ok
      (Arena_eval.strategies ~now:(clock ()) (config ()) (random ~seed:3 ~n:40))
  in
  List.iter
    (fun (r : Arena_eval.Row.t) ->
      Fmt.pr "%-32s %-5s %4Ld -> %4Ld  construct %.0f search %.0f check %.0f@."
        (Fmt.str "%a" Arena_eval.Method.pp r.method_)
        (Fmt.str "%a" pp_budget r.budget)
        (sz r.constructive_pool) (sz r.pool) r.timing.construct r.timing.search
        r.timing.check)
    rows;
  [%expect
    {|
    greedy_by_area                   -      413 ->  413  construct 1 search 0 check 1
    greedy_by_area+improve           0/1    413 ->  413  construct 1 search 1 check 1
    greedy_by_area+improve           0/2    413 ->  413  construct 1 search 1 check 1
    greedy_by_area+improve           5/1    413 ->  403  construct 1 search 2 check 1
    greedy_by_area+improve           5/2    413 ->  409  construct 1 search 2 check 1
    greedy_by_area+improve           50/1   413 ->  403  construct 1 search 3 check 1
    greedy_by_area+improve           50/2   413 ->  403  construct 1 search 3 check 1
    greedy_by_lifetime               -      420 ->  420  construct 1 search 0 check 1
    greedy_by_lifetime+improve       0/1    420 ->  420  construct 1 search 1 check 1
    greedy_by_lifetime+improve       0/2    420 ->  420  construct 1 search 1 check 1
    greedy_by_lifetime+improve       5/1    420 ->  403  construct 1 search 2 check 1
    greedy_by_lifetime+improve       5/2    420 ->  417  construct 1 search 2 check 1
    greedy_by_lifetime+improve       50/1   420 ->  403  construct 1 search 3 check 1
    greedy_by_lifetime+improve       50/2   420 ->  403  construct 1 search 3 check 1
    greedy_by_size                   -      404 ->  404  construct 1 search 0 check 1
    greedy_by_size+improve           0/1    404 ->  404  construct 1 search 1 check 1
    greedy_by_size+improve           0/2    404 ->  404  construct 1 search 1 check 1
    greedy_by_size+improve           5/1    404 ->  404  construct 1 search 2 check 1
    greedy_by_size+improve           5/2    404 ->  404  construct 1 search 2 check 1
    greedy_by_size+improve           50/1   404 ->  403  construct 1 search 3 check 1
    greedy_by_size+improve           50/2   404 ->  403  construct 1 search 3 check 1
    greedy_by_size_best_fit          -      404 ->  404  construct 1 search 0 check 1
    greedy_by_size_best_fit+improve  0/1    404 ->  404  construct 1 search 1 check 1
    greedy_by_size_best_fit+improve  0/2    404 ->  404  construct 1 search 1 check 1
    greedy_by_size_best_fit+improve  5/1    404 ->  404  construct 1 search 2 check 1
    greedy_by_size_best_fit+improve  5/2    404 ->  403  construct 1 search 2 check 1
    greedy_by_size_best_fit+improve  50/1   404 ->  403  construct 1 search 3 check 1
    greedy_by_size_best_fit+improve  50/2   404 ->  403  construct 1 search 3 check 1
    portfolio                        0/1    404 ->  404  construct 1 search 0 check 1
    portfolio                        0/2    404 ->  404  construct 1 search 0 check 1
    portfolio                        5/1    404 ->  404  construct 1 search 1 check 1
    portfolio                        5/2    404 ->  404  construct 1 search 1 check 1
    portfolio                        50/1   404 ->  403  construct 1 search 2 check 1
    portfolio                        50/2   404 ->  403  construct 1 search 2 check 1 |}]

let%expect_test "random scripts: never worse than the start, and replayable" =
  let bad = ref 0 in
  for seed = 1 to 40 do
    let s = random ~seed ~n:30 in
    let lb = ok (IA.lower_bound s) in
    let run () = ok (Arena_eval.strategies ~now:(clock ()) (config ()) s) in
    let rows = run () and again = run () in
    List.iter2
      (fun (a : Arena_eval.Row.t) (b : Arena_eval.Row.t) ->
        if not (String.equal a.digest b.digest) then begin
          incr bad;
          Fmt.pr "seed %d %a: not replayable@." seed Arena_eval.Method.pp
            a.method_
        end;
        if a.pool > a.constructive_pool || a.pool < lb then begin
          incr bad;
          Fmt.pr "seed %d %a: pool %Ld from %Ld, bound %Ld@." seed
            Arena_eval.Method.pp a.method_ (sz a.pool) (sz a.constructive_pool)
            (sz lb)
        end;
        match (a.method_, a.budget) with
        | Arena_eval.Method.Improved _, Some (0L, _) ->
            if a.pool <> a.constructive_pool then begin
              incr bad;
              Fmt.pr "seed %d: zero budget moved the pool@." seed
            end
        | _ -> ())
      rows again;
    (* A bigger budget extends the same walk. *)
    let pools method_ seed =
      List.filter_map
        (fun (r : Arena_eval.Row.t) ->
          match r.budget with
          | Some (_, s) when r.method_ = method_ && s = seed -> Some r.pool
          | _ -> None)
        rows
    in
    List.iter
      (fun m ->
        let p = pools m 1L in
        if
          p <> List.sort (fun a b -> Core.Storage_units.Byte_size.compare b a) p
        then begin
          incr bad;
          Fmt.pr "seed %d %a: budget not monotone@." seed Arena_eval.Method.pp m
        end)
      (Arena_eval.Method.Portfolio
      :: List.map (fun s -> Arena_eval.Method.Improved s) IA.Strategy.all)
  done;
  Fmt.pr "defects %d@." !bad;
  [%expect {| defects 0 |}]

(* The reference minimum of a script whose optimum is above the live bound. *)
let%expect_test "the reference row" =
  let a k s = IA.Event.Alloc { key = t k; size = bytes s; alignment = one }
  and f k = IA.Event.Free (t k) in
  let s =
    script
      [
        a 2 4L;
        a 5 3L;
        f 5;
        a 3 1L;
        a 4 2L;
        f 2;
        a 0 2L;
        f 4;
        a 1 3L;
        f 0;
        f 3;
        a 6 4L;
        f 1;
        f 6;
      ]
  in
  let r = ok (Arena_eval.reference ~now:(clock ()) (config ()) s) in
  let b = r.bounds in
  Fmt.pr "live %Ld, incumbent %Ld, [%Ld, %Ld], %Ld states, %.0fs, %d placed@."
    (sz b.live_bound) (sz b.initial_upper) (sz b.lower) (sz b.upper) r.states
    r.seconds (List.length b.incumbent);
  [%expect {| live 7, incumbent 8, [8, 8], 50 states, 1s, 7 placed |}]
