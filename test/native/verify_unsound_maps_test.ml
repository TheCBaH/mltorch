(* Split out of what was verify_test.ml see verify_fixtures.ml. *)

open Graph_ir
open Verify_fixtures

(* ---- what the verifier must NOT prove -------------------------------------

   A test that cannot fail proves nothing, so each of these is a map that is
   wrong in a specific way, checked to come back unproved. Stage 1 has no
   probe, so a failed comparison is [Unproved Exhausted] rather than
   [Refuted] — the prover ran out of moves, it did not exhibit a
   counterexample. *)

(* Two graphs with identical shapes and ids but a different operator. An
   identity map claims every edge is bit-identical; the output edge is not. *)
let%expect_test "verify: an identity map over a changed operator is unproved" =
  let binop op =
    build "binop"
      Graph_builder.(
        let* a = input ~shape:s ~name:"a" () in
        let* b = input ~shape:s ~name:"b" () in
        op a b)
  in
  (* The EMPTY map, not [Graph_map.identity]: identity is [('v, 'v)] and so
     cannot relate two versions at all now. An empty map between two versions is
     the same claim — everything implicitly identical — and it is what a
     consumer could actually be handed. *)
  let module A = (val Version_fixture.of_graph (binop Graph_builder.add)) in
  let module B = (val Version_fixture.of_graph (binop Graph_builder.sub)) in
  verify_map
    (hand_map ~src:A.snapshot ~dst:B.snapshot [] [])
    ~src:A.snapshot ~dst:B.snapshot;
  [%expect
    {|
    {t0} -> {t0} identical: proved (structural) [exhaustive]
    {t1} -> {t1} identical: proved (structural) [exhaustive]
    {t2} -> {t2} identical: refuted: value at (0): src.t2 vs dst.t2 under {v0(0)=0x1p+0, v1(0)=0x1p+1} [exhaustive] |}]

(* The cluster {t0, t1} <-> {t0} is the shape trimming an identity permute
   produces: the input and the trimmed output both correspond to the surviving
   edge. Here the permute is a REAL one, so t1 is not t0 and the claim is
   false — but only for the non-canonical member. The canonical member pairs
   with t0 on both sides and agrees, so a checker that compared one
   representative per side would call this proved.

   t1 is dead in the source (the relu reads t0 directly), which keeps the lie
   confined to this one cluster: the output cluster is honest and must still
   prove. Shape-preserving (H = W = 3) so the failure is a value difference,
   not a shape mismatch. *)
let%expect_test "verify: a false claim about a non-canonical cluster member" =
  let src =
    build "src"
      Graph_builder.(
        let* a = input ~shape:s ~name:"a" () in
        let* _dead = permute Graph_fixtures.swap_hw a in
        relu a)
  in
  let dst =
    build "dst"
      Graph_builder.(
        let* a = input ~shape:s ~name:"a" () in
        relu a)
  in
  let module A = (val Version_fixture.of_graph src) in
  let module B = (val Version_fixture.of_graph dst) in
  let se i = Option.get (Snapshot.edge A.snapshot (Tensor_id.of_int i)) in
  let de i = Option.get (Snapshot.edge B.snapshot (Tensor_id.of_int i)) in
  let sn i = Option.get (Snapshot.node A.snapshot (Node_id.of_int i)) in
  let dn i = Option.get (Snapshot.node B.snapshot (Node_id.of_int i)) in
  let cluster src dst =
    {
      Correspondence.Cluster.src = Correspondence.Set.of_list (List.map se src);
      dst = Correspondence.Set.of_list (List.map de dst);
      label = Correspondence.Identical;
    }
  in
  let map =
    hand_map ~src:A.snapshot ~dst:B.snapshot
      [ cluster [ 0; 1 ] [ 0 ]; cluster [ 2 ] [ 1 ] ]
      (* the source's dead permute node goes away; the two relus pair up *)
      [ Node_map.delete (sn 0); Node_map.pair (sn 1) (dn 0) ]
  in
  verify_map map ~src:A.snapshot ~dst:B.snapshot;
  [%expect
    {|
    {t0, t1} -> {t0} identical: refuted: value at (1,0): src.t0 vs src.t1 under {v0(1,0)=0x1p+0, v0(1,0,0)=0x1p+1} [exhaustive]
    {t2} -> {t1} identical: proved (structural) [exhaustive] |}]

(* Two graphs that compute the same thing, related by a map that swaps their
   inputs: the destination then computes b - a where the source computes a - b.

   Representatives for corresponding inputs have to be allocated fresh. Taking
   the minimum id of the cluster looks natural and is unsound, because the two
   graphs share one numeric namespace: the crossed pair {t0 <-> t1} and
   {t1 <-> t0} both minimise to t0, every input on both sides collapses onto one
   symbolic variable, and sub(a,b) grounds to the same term as sub(b,a). This
   test reported all three clusters proved before that was fixed. *)
let%expect_test "verify: crossed input clusters do not collapse to one variable"
    =
  let g () =
    build "sub"
      Graph_builder.(
        let* a = input ~shape:s ~name:"a" () in
        let* b = input ~shape:s ~name:"b" () in
        sub a b)
  in
  let module A = (val Version_fixture.of_graph (g ())) in
  let module B = (val Version_fixture.of_graph (g ())) in
  let cluster src dst =
    {
      Correspondence.Cluster.src =
        Correspondence.Set.of_list
          (List.map
             (fun i ->
               Option.get (Snapshot.edge A.snapshot (Tensor_id.of_int i)))
             src);
      dst =
        Correspondence.Set.of_list
          (List.map
             (fun i ->
               Option.get (Snapshot.edge B.snapshot (Tensor_id.of_int i)))
             dst);
      label = Correspondence.Identical;
    }
  in
  verify_map
    (hand_map ~src:A.snapshot ~dst:B.snapshot
       [ cluster [ 0 ] [ 1 ]; cluster [ 1 ] [ 0 ] ]
       [])
    ~src:A.snapshot ~dst:B.snapshot;
  [%expect
    {|
    {t0} -> {t1} identical: proved (structural) [exhaustive]
    {t1} -> {t0} identical: proved (structural) [exhaustive]
    {t2} -> {t2} identical: refuted: value at (0): src.t2 vs dst.t2 under {v0(0)=0x1p+0, v1(0)=0x1p+1} [exhaustive] |}]
