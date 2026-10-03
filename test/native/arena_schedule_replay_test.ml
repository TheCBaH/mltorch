(* [Arena_schedule_problem]: the replay of an explicit node order against the
   facts of one baseline dry run, checked against a fresh dry run of the same
   reordered graph. *)

open Graph_ir
module P = Arena_schedule_problem
module M = Arena_schedule.Metrics

let only_empty = Release_schedule.Retain.Only Tensor_id.Set.empty
let pp_error ppf e = P.pp_error ppf (Err.Error.kind e)

let config ?(retain = only_empty) ?(alignment = Alignment_policy.standard) () =
  {
    Arena_schedule.Config.mode = Intermediate;
    retain;
    alignment;
    limits = Arena_schedule.Limits.constructive_only;
  }

let ok = function Ok v -> v | Error e -> Fmt.failwith "%a" pp_error e
let fixture name = (List.assoc name Graph_fixtures.all) ()

let branches () =
  Graph_fixtures.build "branches"
    Graph_builder.(
      let* x = input ~shape:(Graph_fixtures.s1c 8) () in
      let mean = { Reduce.Mean.dims = [ Axis.C ]; keepdim = true } in
      let* big_a = relu x in
      let* a = Graph_builder.mean mean big_a in
      let* big_b = sigmoid x in
      let* b = Graph_builder.mean mean big_b in
      add a b)

(* A duplicate operand ([add t t]), a fan-out of [t] to two readers, and an
   ordinary output nothing reads ([dead]): released right after its producer. *)
let dup_fanout_dead () =
  Graph_fixtures.build "dup_fanout_dead"
    Graph_builder.(
      let* x = input ~shape:(Graph_fixtures.s1c 8) () in
      let* t = relu x in
      let* dead = sqrt x in
      ignore dead;
      let* u = add t t in
      let* v = sigmoid t in
      add u v)

let pos = P.Position.of_int
let order l = Array.of_list (List.map pos l)
let target m = Core.Storage_units.Byte_size.to_int64 m.M.target_peak

let check_against_fresh c p o =
  let m = ok (P.metrics p o) in
  let fresh = ok (P.fresh_metrics c (ok (P.reorder p o))) in
  if not (M.equal m fresh) then
    Fmt.failwith "replay and fresh dry run disagree@.%a@.vs@.%a" M.pp m M.pp
      fresh;
  m

let%expect_test "the design's two-branch example: peaks 17 and 10 (x4 bytes)" =
  let c = config () in
  let p = ok (P.of_graph c (branches ())) in
  let show o =
    let m = check_against_fresh c p (order o) in
    Fmt.pr "%Ld@." (target m)
  in
  show [ 0; 2; 1; 3; 4 ];
  show [ 0; 1; 2; 3; 4 ];
  Fmt.pr "bound %Ld@." (target (ok (P.lower_bound p)));
  [%expect {|
    68
    40
    bound 36 |}]

(* Every valid order of a tiny graph, and seeded random ones of the rest: a
   random topological order is a repeated draw from the ready set. *)
let random_order p rng =
  let n = P.node_count p in
  let placed = Array.make n false in
  let out = Array.make n (pos 0) in
  for i = 0 to n - 1 do
    let ready = ref [] in
    for k = n - 1 downto 0 do
      if
        (not placed.(k))
        && List.for_all
             (fun q -> placed.((q : P.Position.t :> int)))
             (P.preds p (pos k))
      then ready := k :: !ready
    done;
    let k = List.nth !ready (Random.State.int rng (List.length !ready)) in
    placed.(k) <- true;
    out.(i) <- pos k
  done;
  out

let first_ids g =
  match g.Graph.nodes with
  | n :: _ -> Tensor_id.Set.of_list n.Node.outputs
  | [] -> Tensor_id.Set.empty

let%expect_test "replay agrees with a fresh dry run on every fixture" =
  let checked = ref 0 in
  let rng = Random.State.make [| 20261001 |] in
  List.iter
    (fun (name, build) ->
      let g = build () in
      List.iter
        (fun retain ->
          let c = config ~retain () in
          match P.of_graph c g with
          | Error e -> Fmt.pr "%s: %a@." name pp_error e
          | Ok p ->
              for _ = 1 to 12 do
                ignore (check_against_fresh c p (random_order p rng));
                incr checked
              done)
        [ only_empty; Release_schedule.Retain.All; Only (first_ids g) ])
    (("branches", branches)
    :: ("dup_fanout_dead", dup_fanout_dead)
    :: Graph_fixtures.all);
  Fmt.pr "%b@." (!checked > 100);
  [%expect {| true |}]

let%expect_test "invalid orders are rejected" =
  let p = ok (P.of_graph (config ()) (branches ())) in
  let show o =
    match P.metrics p (order o) with
    | Ok _ -> Fmt.pr "ok@."
    | Error e -> Fmt.pr "%a@." pp_error e
  in
  show [ 1; 0; 2; 3; 4 ];
  show [ 0; 1; 2; 3 ];
  show [ 0; 1; 1; 3; 4 ];
  show [ 0; 1; 2; 3; 9 ];
  [%expect
    {|
    node n1 runs before one of its producers
    the node lists are not permutations of one another
    the node lists are not permutations of one another
    the node lists are not permutations of one another |}]

let%expect_test "an unread output and a duplicate operand are counted once" =
  let c = config () in
  let p = ok (P.of_graph c (dup_fanout_dead ())) in
  (* t, dead, u, v, join: t(32) dead(32) live together, dead freed at once. *)
  let m = check_against_fresh c p (order [ 0; 1; 2; 3; 4 ]) in
  Fmt.pr "%Ld readers(t)=%d@." (target m)
    (P.readers p (List.hd (P.node p (pos 0)).Node.outputs));
  [%expect {| 96 readers(t)=2 |}]
