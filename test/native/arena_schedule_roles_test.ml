(* Role-storage lifetimes: the fixed constant/input prefix, per-(arena, kind)
   pools and result-lifetime blocks, replayed incrementally and checked against
   [Storage_script.peak_bytes] on a fresh script of the reordered graph, over
   the whole layout x ownership x retention matrix. *)

open Graph_ir
module P = Arena_schedule_problem
module G = Arena_schedule_greedy
module M = Arena_schedule.Metrics
module B = Core.Storage_units.Byte_size
module Sc = Storage_script

let pp_error ppf e = P.pp_error ppf (Err.Error.kind e)
let ok = function Ok v -> v | Error e -> Fmt.failwith "%a" pp_error e
let pos = P.Position.of_int

let config ~roles ~retain =
  {
    Arena_schedule.Config.mode = Roles roles;
    retain;
    alignment = Alignment_policy.standard;
    limits = Arena_schedule.Limits.constructive_only;
  }

let matrix =
  List.concat_map
    (fun layout ->
      List.concat_map
        (fun constants ->
          List.map
            (fun inputs -> { Sc.Config.layout; constants; inputs })
            Sc.Ownership.[ Borrowed; Copied ])
        Sc.Ownership.[ Borrowed; Copied ])
    Sc.Layout.[ Separate; Shared_execution ]

let retains g =
  let first =
    match g.Graph.nodes with
    | n :: _ -> Tensor_id.Set.of_list n.Node.outputs
    | [] -> Tensor_id.Set.empty
  in
  Release_schedule.Retain.[ Only Tensor_id.Set.empty; All; Only first ]

let random_order p rng =
  let n = P.node_count p in
  let placed = Array.make n false in
  Array.init n (fun _ ->
      let ready =
        List.filter
          (fun k ->
            (not placed.(k))
            && List.for_all
                 (fun q -> placed.((q : P.Position.t :> int)))
                 (P.preds p (pos k)))
          (List.init n Fun.id)
      in
      let k = List.nth ready (Random.State.int rng (List.length ready)) in
      placed.(k) <- true;
      pos k)

let%expect_test "replay agrees with fresh storage scripts over the matrix" =
  let rng = Random.State.make [| 20261002 |] in
  let checked = ref 0 and bad = ref 0 in
  List.iter
    (fun (name, build) ->
      let g = build () in
      List.iter
        (fun roles ->
          List.iter
            (fun retain ->
              let c = config ~roles ~retain in
              match P.of_graph c g with
              | Error e -> Fmt.pr "%s: %a@." name pp_error e
              | Ok p ->
                  for _ = 1 to 4 do
                    let o = random_order p rng in
                    let m = ok (P.metrics p o) in
                    let fresh = ok (P.fresh_metrics c (ok (P.reorder p o))) in
                    incr checked;
                    if not (M.equal m fresh) then (
                      incr bad;
                      Fmt.pr "%s %a: replay@.%a@.fresh@.%a@." name Sc.Config.pp
                        roles M.pp m M.pp fresh)
                  done)
            (retains g))
        matrix)
    Graph_fixtures.all;
  Fmt.pr "bad=%d checked>400=%b@." !bad (!checked > 400);
  [%expect {| bad=0 checked>400=true |}]

let%expect_test "greedy in role mode: valid, never worse, above the bound" =
  let worse = ref 0 and improved = ref 0 and below = ref 0 in
  List.iter
    (fun (_, build) ->
      let g = build () in
      List.iter
        (fun roles ->
          List.iter
            (fun retain ->
              let c = config ~roles ~retain in
              let r = ok (G.run c g) in
              ok (Arena_schedule.check_permutation ~original:g r.graph);
              let target gr =
                B.to_int64 (ok (P.fresh_metrics c gr)).M.target_peak
              in
              let bound =
                B.to_int64
                  (ok (P.lower_bound (ok (P.of_graph c g)))).M.target_peak
              in
              if target r.graph > target g then incr worse;
              if target r.graph < target g then incr improved;
              if target r.graph < bound then incr below)
            (retains g))
        matrix)
    Graph_fixtures.all;
  Fmt.pr "worse=%d below-bound=%d improved>=0=%b@." !worse !below
    (!improved >= 0);
  [%expect {| worse=0 below-bound=0 improved>=0=true |}]

let%expect_test
    "the prefix is part of the peak: copied inputs are live at entry" =
  let g = (List.assoc "residual" Graph_fixtures.all) () in
  let show roles =
    let c = config ~roles ~retain:(Only Tensor_id.Set.empty) in
    let p = ok (P.of_graph c g) in
    let m = ok (P.metrics p (Array.init (P.node_count p) pos)) in
    Fmt.pr "%a: prefix=%d target=%Ld all=%Ld@." Sc.Config.pp roles
      (List.length (P.prefix p))
      (B.to_int64 m.M.target_peak)
      (B.to_int64 m.M.all_peak)
  in
  List.iter show
    [
      { Sc.Config.layout = Separate; constants = Borrowed; inputs = Borrowed };
      { layout = Separate; constants = Borrowed; inputs = Copied };
      { layout = Shared_execution; constants = Borrowed; inputs = Copied };
    ];
  [%expect
    {|
    layout=separate constants=borrowed inputs=borrowed: prefix=1 target=32 all=48
    layout=separate constants=borrowed inputs=copied: prefix=1 target=48 all=48
    layout=shared_execution constants=borrowed inputs=copied: prefix=1 target=48 all=48 |}]
