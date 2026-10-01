open Core.Storage_units
open Interval_alloc

(* "+a:4" allocates [a] with size 4, "+a:4@8" with alignment 8 as well; "-a"
   frees it. *)
let event s =
  match (s.[0], String.index_opt s ':') with
  | '+', Some i ->
      let rest = String.sub s (i + 1) (String.length s - i - 1) in
      let size, alignment =
        match String.index_opt rest '@' with
        | None -> (rest, Units.one)
        | Some j ->
            ( String.sub rest 0 j,
              Units.alignment
                (Int64.of_string
                   (String.sub rest (j + 1) (String.length rest - j - 1))) )
      in
      Event.Alloc
        {
          key = String.sub s 1 (i - 1);
          size = Units.size (Int64.of_string size);
          alignment;
        }
  | '-', None -> Event.Free (String.sub s 1 (String.length s - 1))
  | _ -> invalid_arg s

let script events =
  match Script.validate ~equal:String.equal (List.map event events) with
  | Ok s -> s
  | Error e ->
      Fmt.failwith "invalid script: %s"
        (Fmt.str "%a" (Err.Error.pp_kind (fun ppf _ -> Fmt.string ppf "?")) e)

let placements s =
  String.concat " "
    (List.map
       (fun (k, o) -> Fmt.str "%s@%a" k Byte_offset.pp o)
       (Solution.placements s))

let show name events =
  let s = script events in
  Fmt.pr "%s: lower bound %a@." name
    (Fmt.result ~ok:Byte_size.pp ~error:(fun ppf _ -> Fmt.string ppf "error"))
    (Err.payload (lower_bound s));
  List.iter
    (fun strategy ->
      match solve strategy s with
      | Error _ -> Fmt.pr "  %a: error@." Strategy.pp strategy
      | Ok sol -> (
          match check s sol with
          | Ok _ ->
              Fmt.pr "  %a: pool %a  %s@." Strategy.pp strategy Byte_size.pp
                (Solution.pool sol) (placements sol)
          | Error _ -> Fmt.pr "  %a: CHECK FAILED@." Strategy.pp strategy))
    Strategy.all

let%expect_test "hand-built scripts" =
  show "empty" [];
  show "one alloc" [ "+a:8" ];
  show "alloc/free/alloc" [ "+a:4"; "-a"; "+b:6"; "-b" ];
  show "all live together" [ "+a:4"; "+b:2"; "+c:3"; "-a"; "-b"; "-c" ];
  show "operand and output conflict" [ "+x:4"; "+y:4"; "-x"; "-y" ];
  show "never freed" [ "+a:4"; "-a"; "+b:2"; "+c:3" ];
  show "zero size" [ "+a:0"; "+b:5"; "-a"; "-b" ];
  show "size beats lifetime"
    [ "+a:0"; "-a"; "+b:1"; "+c:2"; "-c"; "+d:4"; "-b"; "+e:5"; "-d"; "-e" ];
  show "lifetime beats size"
    [ "+a:6"; "+b:8"; "-b"; "+c:7"; "-c"; "+d:6"; "+e:7"; "-d"; "-a"; "-e" ];
  [%expect
    {|
    empty: lower bound 0
      greedy_by_area: pool 0
      greedy_by_lifetime: pool 0
      greedy_by_size: pool 0
      greedy_by_size_best_fit: pool 0
    one alloc: lower bound 8
      greedy_by_area: pool 8  a@0
      greedy_by_lifetime: pool 8  a@0
      greedy_by_size: pool 8  a@0
      greedy_by_size_best_fit: pool 8  a@0
    alloc/free/alloc: lower bound 6
      greedy_by_area: pool 6  a@0 b@0
      greedy_by_lifetime: pool 6  a@0 b@0
      greedy_by_size: pool 6  a@0 b@0
      greedy_by_size_best_fit: pool 6  a@0 b@0
    all live together: lower bound 9
      greedy_by_area: pool 9  a@0 b@7 c@4
      greedy_by_lifetime: pool 9  a@0 b@4 c@6
      greedy_by_size: pool 9  a@0 b@7 c@4
      greedy_by_size_best_fit: pool 9  a@0 b@7 c@4
    operand and output conflict: lower bound 8
      greedy_by_area: pool 8  x@0 y@4
      greedy_by_lifetime: pool 8  x@0 y@4
      greedy_by_size: pool 8  x@0 y@4
      greedy_by_size_best_fit: pool 8  x@0 y@4
    never freed: lower bound 5
      greedy_by_area: pool 5  a@0 b@0 c@2
      greedy_by_lifetime: pool 5  a@0 b@0 c@2
      greedy_by_size: pool 5  a@0 b@3 c@0
      greedy_by_size_best_fit: pool 5  a@0 b@3 c@0
    zero size: lower bound 5
      greedy_by_area: pool 5  a@0 b@0
      greedy_by_lifetime: pool 5  a@0 b@0
      greedy_by_size: pool 5  a@0 b@0
      greedy_by_size_best_fit: pool 5  a@0 b@0
    size beats lifetime: lower bound 9
      greedy_by_area: pool 9  a@0 b@4 c@0 d@0 e@4
      greedy_by_lifetime: pool 10  a@0 b@0 c@1 d@1 e@5
      greedy_by_size: pool 9  a@0 b@2 c@0 d@5 e@0
      greedy_by_size_best_fit: pool 9  a@0 b@2 c@0 d@5 e@0
    lifetime beats size: lower bound 19
      greedy_by_area: pool 19  a@0 b@6 c@6 d@13 e@6
      greedy_by_lifetime: pool 19  a@0 b@6 c@6 d@13 e@6
      greedy_by_size: pool 20  a@8 b@0 c@0 d@14 e@0
      greedy_by_size_best_fit: pool 20  a@8 b@0 c@0 d@14 e@0 |}]

let pp_key ppf k = Fmt.string ppf k

let pp_block ppf { Block.key; offset; size } =
  Fmt.pf ppf "%s[%a,+%a]" key Byte_offset.pp offset Byte_size.pp size

let pp_error ppf = function
  | `Double_alloc k -> Fmt.pf ppf "double alloc %a" pp_key k
  | `Double_free k -> Fmt.pf ppf "double free %a" pp_key k
  | `Duplicate_placement k -> Fmt.pf ppf "duplicate placement %a" pp_key k
  | `Free_unknown k -> Fmt.pf ppf "free of unknown %a" pp_key k
  | `Live_overflow k -> Fmt.pf ppf "live sum overflows at %a" pp_key k
  | `Misaligned { Misaligned.block; alignment } ->
      Fmt.pf ppf "%a not aligned to %a" pp_block block Byte_alignment.pp
        alignment
  | `Offset_overflow k -> Fmt.pf ppf "offset overflow %a" pp_key k
  | `Out_of_pool { Out_of_pool.block; pool } ->
      Fmt.pf ppf "%a outside pool %a" pp_block block Byte_size.pp pool
  | `Overlap { Overlap.first; second } ->
      Fmt.pf ppf "%a overlaps %a" pp_block first pp_block second
  | `Pool_overflow k -> Fmt.pf ppf "pool overflows at %a" pp_key k
  | `Unknown_key k -> Fmt.pf ppf "unknown key %a" pp_key k
  | `Unplaced k -> Fmt.pf ppf "unplaced %a" pp_key k

let%expect_test "validate rejects malformed scripts" =
  let reject events =
    match Script.validate ~equal:String.equal (List.map event events) with
    | Ok _ -> Fmt.pr "accepted@."
    | Error e -> Fmt.pr "%a@." pp_error (Err.Error.kind e)
  in
  reject [ "+a:1"; "+a:2" ];
  reject [ "+a:1"; "-a"; "+a:2" ];
  reject [ "-a" ];
  reject [ "+a:1"; "-a"; "-a" ];
  [%expect
    {|
    double alloc a
    double alloc a
    free of unknown a
    double free a |}]

(* The checker must be able to fail: each mutation of a good solution below is
   a different way for a placement to be wrong. *)
let%expect_test "check rejects bad placements" =
  let s = script [ "+a:4"; "+b:2"; "+c:3"; "-a"; "-b"; "-c" ] in
  let good =
    match solve Strategy.Greedy_by_size s with
    | Ok x -> x
    | Error _ -> assert false
  in
  let pool = Solution.pool good and ps = Solution.placements good in
  let verdict ?(s = s) sol =
    match check s sol with
    | Ok w -> Fmt.pr "ok pool %a@." Byte_size.pp (Interval_alloc.pool w)
    | Error e -> Fmt.pr "%a@." pp_error (Err.Error.kind e)
  in
  let zero = Byte_offset.zero in
  verdict good;
  verdict
    (Solution.Unsafe.make ~pool
       (List.map (fun (k, o) -> if k = "b" then (k, zero) else (k, o)) ps));
  verdict (Solution.Unsafe.make ~pool (List.filter (fun (k, _) -> k <> "c") ps));
  verdict
    (Solution.Unsafe.make
       ~pool:(Units.size (Int64.pred (Byte_size.to_int64 pool)))
       ps);
  verdict (Solution.Unsafe.make ~pool (("z", zero) :: ps));
  verdict (Solution.Unsafe.make ~pool (("a", zero) :: ps));
  verdict
    (Solution.Unsafe.make ~pool:(Units.size Int64.max_int)
       (List.map
          (fun (k, o) ->
            if k = "a" then (k, Units.offset Int64.max_int) else (k, o))
          ps));
  (* Inside the pool and overlapping nothing, but not at a multiple of its
     alignment. *)
  let aligned = script [ "+a:4@4"; "+b:2@2"; "-a"; "-b" ] in
  let pool = Units.size 12L in
  verdict ~s:aligned
    (Solution.Unsafe.make ~pool
       [ ("a", Units.offset 4L); ("b", Units.offset 0L) ]);
  verdict ~s:aligned
    (Solution.Unsafe.make ~pool
       [ ("a", Units.offset 6L); ("b", Units.offset 0L) ]);
  verdict ~s:aligned
    (Solution.Unsafe.make ~pool
       [ ("a", Units.offset 4L); ("b", Units.offset 9L) ]);
  [%expect
    {|
    ok pool 9
    a[0,+4] overlaps b[0,+2]
    unplaced c
    b[7,+2] outside pool 8
    unknown key z
    duplicate placement a
    offset overflow a
    ok pool 12
    a[6,+4] not aligned to 4
    b[9,+2] not aligned to 2 |}]

let%expect_test "lower bound overflow" =
  let s = script [ "+a:9223372036854775807"; "+b:1" ] in
  (match lower_bound s with
  | Ok n -> Fmt.pr "%a@." Byte_size.pp n
  | Error e -> Fmt.pr "%a@." pp_error (Err.Error.kind e));
  (match solve Strategy.Greedy_by_size s with
  | Ok sol -> Fmt.pr "pool %a@." Byte_size.pp (Solution.pool sol)
  | Error e -> Fmt.pr "%a@." pp_error (Err.Error.kind e));
  [%expect {|
    live sum overflows at b
    pool overflows at b |}]

(* Every strategy's solution passes [check] and is no smaller than the bound;
   the same script solved twice gives the same offsets, and renaming the keys
   (event order fixed) does too. *)
let random_scripts ?max_log_alignment ~seed ~max_size () =
  let g = Gen.make seed in
  let worst = ref 0. and total = ref 0. and count = ref 0 in
  for i = 1 to 200 do
    let events =
      Gen.script ?max_log_alignment g
        ~n:(5 + Gen.below g (if i mod 20 = 0 then 250 else 60))
        ~max_size
    in
    let s =
      match Script.validate ~equal:Int.equal events with
      | Ok s -> s
      | Error _ -> assert false
    in
    let renamed =
      List.map
        (function
          | Event.Alloc { key; size; alignment } ->
              Event.Alloc { key = key + 1000; size; alignment }
          | Event.Free k -> Event.Free (k + 1000))
        events
    in
    let s' =
      match Script.validate ~equal:Int.equal renamed with
      | Ok s -> s
      | Error _ -> assert false
    in
    let lb = match lower_bound s with Ok n -> n | Error _ -> assert false in
    List.iter
      (fun strategy ->
        match (solve strategy s, solve strategy s, solve strategy s') with
        | Ok a, Ok b, Ok c ->
            (match check s a with
            | Ok _ -> ()
            | Error _ ->
                Fmt.pr "script %d: %a fails check@." i Strategy.pp strategy);
            if Byte_size.compare (Solution.pool a) lb < 0 then
              Fmt.pr "script %d: below the bound@." i;
            if Solution.placements a <> Solution.placements b then
              Fmt.pr "script %d: not deterministic@." i;
            if
              List.map snd (Solution.placements a)
              <> List.map snd (Solution.placements c)
            then Fmt.pr "script %d: depends on key names@." i;
            if Byte_size.compare lb Byte_size.zero > 0 then begin
              let r =
                Int64.to_float (Byte_size.to_int64 (Solution.pool a))
                /. Int64.to_float (Byte_size.to_int64 lb)
              in
              worst := Float.max !worst r;
              total := !total +. r;
              incr count
            end
        | _ -> Fmt.pr "script %d: solve failed@." i)
      Strategy.all
  done;
  Fmt.pr "solutions %d, mean pool/bound %.3f, max %.3f@." !count
    (!total /. float_of_int !count)
    !worst

let%expect_test "random scripts" =
  random_scripts ~seed:7 ~max_size:50 ();
  [%expect {| solutions 800, mean pool/bound 1.015, max 1.163 |}]

(* Alignments up to 64 on sizes that are mostly not multiples of them: every
   placement must still pass the checker, which now also rejects a misaligned
   offset. *)
let%expect_test "random aligned scripts" =
  random_scripts ~max_log_alignment:6 ~seed:11 ~max_size:200 ();
  [%expect {| solutions 800, mean pool/bound 1.076, max 1.275 |}]
