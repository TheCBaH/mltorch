(* [Interval_alloc.Reference]: the bounded exact search, graded against an
   independent brute-force enumeration of aligned integer offsets. *)

open Core.Storage_units
open Interval_alloc
module R = Reference

let ok = function
  | Ok x -> x
  | Error e -> (
      match Err.Error.kind e with
      | `Invalid_candidate _ -> Fmt.failwith "invalid candidate"
      | _ -> Fmt.failwith "unexpected error")

let script_of_events events = ok (Script.validate ~equal:Int.equal events)

let alloc key size =
  Event.Alloc { key; size = Units.size size; alignment = Units.one }

let free key = Event.Free key

let limits ?(max_states = 1_000_000L) ?(max_depth = 1_000L) () =
  { R.Limits.max_states; max_depth }

(* Test-only oracle, sharing nothing with the library: live ranges from the
   events, then for each length from zero up, a backtracking enumeration of
   every aligned integer offset of every positive block. *)
let brute_force events =
  let length = List.length events in
  let blocks = Hashtbl.create 8 in
  List.iteri
    (fun pos -> function
      | Event.Alloc { key; size; alignment } ->
          Hashtbl.replace blocks key
            ( pos,
              length,
              Int64.to_int (Byte_size.to_int64 size),
              Int64.to_int (Byte_alignment.to_int64 alignment) )
      | Event.Free key ->
          let a, _, size, align = Hashtbl.find blocks key in
          Hashtbl.replace blocks key (a, pos, size, align))
    events;
  let blocks =
    Hashtbl.fold (fun _ b acc -> b :: acc) blocks []
    |> List.filter (fun (_, _, size, _) -> size > 0)
    |> List.sort compare |> Array.of_list
  in
  let n = Array.length blocks in
  let clash i j =
    let a, f, _, _ = blocks.(i) and a', f', _, _ = blocks.(j) in
    a < f' && a' < f
  in
  let offsets = Array.make n 0 in
  let rec fits pool i =
    i >= n
    ||
    let _, _, size, align = blocks.(i) in
    let clear o j =
      (not (clash i j))
      ||
      let _, _, sj, _ = blocks.(j) in
      o + size <= offsets.(j) || offsets.(j) + sj <= o
    in
    let rec try_at o =
      o + size <= pool
      && (List.for_all (clear o) (List.init i Fun.id)
          &&
          (offsets.(i) <- o;
           fits pool (i + 1))
         || try_at (o + align))
    in
    try_at 0
  in
  let rec from pool = if fits pool 0 then pool else from (pool + 1) in
  Int64.of_int (from 0)

let incumbent s =
  fst (ok (solve_best ~budget:(Budget.create ~iterations:0L ~seed:0L) s))

let minimum ?(limits = limits ()) s =
  ok (R.minimum limits s ~incumbent:(incumbent s))

let pp_status ppf = function
  | R.Status.Incomplete -> Fmt.string ppf "incomplete"
  | Optimal_above_live_bound -> Fmt.string ppf "optimal above the live bound"
  | Optimal_live_bound -> Fmt.string ppf "optimal at the live bound"

let pp_stop ppf = function
  | R.Stop.Closed -> Fmt.string ppf "closed"
  | Depth_limit -> Fmt.string ppf "depth limit"
  | State_limit -> Fmt.string ppf "state limit"

let show ((b : _ R.Bounds.t), work) =
  Fmt.pr "%a (%a): live %a, incumbent %a, [%a, %a], %Ld queries@." pp_status
    b.status pp_stop b.stop Byte_size.pp b.live_bound Byte_size.pp
    b.initial_upper Byte_size.pp b.lower Byte_size.pp b.upper b.queries;
  Fmt.pr "  witness pool %a, %Ld states, depth %Ld@." Byte_size.pp
    (pool b.incumbent) (R.Work.states work) (R.Work.max_depth work)

let against_brute_force ?max_log_alignment ~seed () =
  let g = Gen.make seed in
  let bad = ref 0 and above = ref 0 and total = ref 0 in
  for i = 1 to 400 do
    let events =
      Gen.script ?max_log_alignment g ~n:(2 + Gen.below g 5) ~max_size:4
    in
    let b, _ = minimum (script_of_events events) in
    let upper = Byte_size.to_int64 b.R.Bounds.upper in
    incr total;
    if b.R.Bounds.status = R.Status.Optimal_above_live_bound then incr above;
    if b.R.Bounds.status = R.Status.Incomplete then
      Fmt.pr "script %d: incomplete@." i
    else if upper <> brute_force events then begin
      incr bad;
      Fmt.pr "script %d: %Ld, brute force %Ld@." i upper (brute_force events)
    end
  done;
  Fmt.pr "scripts %d, disagreements %d, above the live bound %d@." !total !bad
    !above

let%expect_test "against brute force on small scripts" =
  against_brute_force ~seed:31 ();
  [%expect {| scripts 400, disagreements 0, above the live bound 0 |}]

(* Alignments up to 4 on sizes up to 4: an optimum above the live bound is
   common here, since padding is not in that bound. *)
let%expect_test "against brute force on small aligned scripts" =
  against_brute_force ~max_log_alignment:2 ~seed:37 ();
  [%expect {| scripts 400, disagreements 0, above the live bound 79 |}]

(* Seven blocks whose optimum is one above the live bound: the only way to
   close the interval is an exhaustive [Infeasible] at the live bound. *)
let above_live =
  [
    alloc 2 4L;
    alloc 5 3L;
    free 5;
    alloc 3 1L;
    alloc 4 2L;
    free 2;
    alloc 0 2L;
    free 4;
    alloc 1 3L;
    free 0;
    free 3;
    alloc 6 4L;
    free 1;
    free 6;
  ]

let%expect_test "an optimum above the live bound" =
  Fmt.pr "brute force %Ld@." (brute_force above_live);
  show (minimum (script_of_events above_live));
  [%expect
    {|
    brute force 8
    optimal above the live bound (closed): live 7, incumbent 8, [8, 8], 1 queries
      witness pool 8, 50 states, depth 8 |}]

let%expect_test "one ceiling" =
  let s = script_of_events above_live in
  let answer ?(limits = limits ()) c =
    match
      ok (R.feasible limits (R.Work.create ()) s ~ceiling:(Units.size c))
    with
    | R.Answer.Feasible w ->
        Fmt.pr "%Ld: feasible, pool %a@." c Byte_size.pp (pool w)
    | Infeasible -> Fmt.pr "%Ld: infeasible@." c
    | Unknown R.Cut.Depth -> Fmt.pr "%Ld: unknown (depth)@." c
    | Unknown R.Cut.States -> Fmt.pr "%Ld: unknown (states)@." c
  in
  List.iter (fun c -> answer c) [ 3L; 7L; 8L; 20L ];
  (* A cut is never a proof. *)
  answer ~limits:(limits ~max_states:3L ()) 7L;
  answer ~limits:(limits ~max_depth:1L ()) 7L;
  [%expect
    {|
    3: infeasible
    7: infeasible
    8: feasible, pool 8
    20: feasible, pool 8
    7: unknown (states)
    7: unknown (depth) |}]

let%expect_test "limits stop the search, keeping the incumbent" =
  let s = script_of_events above_live in
  show (minimum ~limits:(limits ~max_states:0L ()) s);
  show (minimum ~limits:(limits ~max_depth:0L ()) s);
  [%expect
    {|
    incomplete (state limit): live 7, incumbent 8, [7, 8], 1 queries
      witness pool 8, 0 states, depth 0
    incomplete (depth limit): live 7, incumbent 8, [7, 8], 1 queries
      witness pool 8, 0 states, depth 0 |}]

let%expect_test "an incumbent at the live bound needs no query" =
  show
    (minimum
       ~limits:(limits ~max_states:0L ~max_depth:0L ())
       (script_of_events [ alloc 0 3L; alloc 1 2L; free 0; alloc 2 3L ]));
  [%expect
    {|
    optimal at the live bound (closed): live 5, incumbent 5, [5, 5], 0 queries
      witness pool 5, 0 states, depth 0 |}]

let%expect_test "zero-size blocks and an empty pool" =
  show (minimum (script_of_events [ alloc 0 0L; alloc 1 0L; free 0 ]));
  show (minimum (script_of_events [ alloc 0 0L; alloc 1 5L; alloc 2 0L ]));
  [%expect
    {|
    optimal at the live bound (closed): live 0, incumbent 0, [0, 0], 0 queries
      witness pool 0, 0 states, depth 0
    optimal at the live bound (closed): live 5, incumbent 5, [5, 5], 0 queries
      witness pool 5, 0 states, depth 0 |}]

(* Two live blocks whose sizes sum past [Int64.max_int]: pruned as not
   fitting, not wrapped. *)
let%expect_test "no overflow near the top of int64" =
  let big = Int64.shift_left 1L 62 in
  let s = script_of_events [ alloc 0 big; alloc 1 big ] in
  (match
     ok
       (R.feasible (limits ()) (R.Work.create ()) s
          ~ceiling:(Units.size Int64.max_int))
   with
  | R.Answer.Infeasible -> Fmt.pr "infeasible@."
  | _ -> Fmt.pr "not infeasible@.");
  [%expect {| infeasible |}]

(* The bisection alone, against a scripted oracle that logs every ceiling it
   is asked: a placement fits from [opt] up, and a feasible answer reports the
   length [report c]. *)
let bisect ?(report = Fun.id) ?(unknown = fun _ -> false) ~live ~upper opt =
  let asked = ref [] in
  let query c =
    let c = Byte_size.to_int64 c in
    asked := c :: !asked;
    if unknown c then Ok (R.Answer.Unknown R.Cut.States)
    else if Int64.compare c opt >= 0 then
      Ok (R.Answer.Feasible (Units.size (report c)))
    else Ok R.Answer.Infeasible
  in
  let upper = Units.size upper in
  let result =
    R.bisect ~live_bound:(Units.size live) ~upper ~incumbent:upper ~pool:Fun.id
      ~query
  in
  Fmt.pr "asked %a: " Fmt.(list ~sep:(any ", ") int64) (List.rev !asked);
  match Err.payload result with
  | Ok b ->
      Fmt.pr "%a (%a) [%a, %a], %Ld queries@." pp_status b.R.Bounds.status
        pp_stop b.stop Byte_size.pp b.lower Byte_size.pp b.upper b.queries
  | Error (`Invalid_candidate { R.Invalid_candidate.pool; lower; ceiling }) ->
      Fmt.pr "invalid candidate %a (lower %a, ceiling %a)@." Byte_size.pp pool
        Byte_size.pp lower Byte_size.pp ceiling

let%expect_test "bisection" =
  (* Live bound first, then midpoints. *)
  bisect ~live:10L ~upper:20L 14L;
  (* A feasible answer tightens to its actual length, below the ceiling. *)
  bisect
    ~report:(fun c -> Int64.max 14L (Int64.sub c 3L))
    ~live:10L ~upper:20L 14L;
  (* Feasible at the live bound: done after one query. *)
  bisect ~live:10L ~upper:20L 10L;
  (* The optimum is the last ceiling below the incumbent. *)
  bisect ~live:10L ~upper:12L 12L;
  (* Unknown stops, keeping what is proven. *)
  bisect ~unknown:(fun c -> c = 15L) ~live:10L ~upper:20L 14L;
  (* Nothing to ask. *)
  bisect ~live:10L ~upper:10L 10L;
  (* A placement below the proven lower bound is a defect, not a result. *)
  bisect ~report:(fun _ -> 9L) ~live:10L ~upper:20L 10L;
  [%expect
    {|
    asked 10, 15, 12, 13, 14: optimal above the live bound (closed) [14, 14], 5 queries
    asked 10, 15, 12, 13: optimal above the live bound (closed) [14, 14], 4 queries
    asked 10: optimal at the live bound (closed) [10, 10], 1 queries
    asked 10, 11: optimal above the live bound (closed) [12, 12], 2 queries
    asked 10, 15: incomplete (state limit) [11, 20], 2 queries
    asked : optimal at the live bound (closed) [10, 10], 0 queries
    asked 10: invalid candidate 9 (lower 10, ceiling 10) |}]
