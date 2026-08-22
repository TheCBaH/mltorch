(* The symbolic transformation verifier: run a pass, then check the map it
   produced actually holds — without payloads for the graph inputs, so a
   [proved] verdict is a statement about EVERY input rather than one sample.
   See .ai/native_transform_verify.md.

   Goldens carry verdicts and ids only, never magnitudes: a printed float would
   make them depend on the platform's floating point, a verdict does not (the
   same reason fold_batch_norm_test.ml prints one). *)

open Graph_ir

type error = [ Map_verify.error | Pass.error | `Origin of Rewrite.error ]

let pp_error ppf : [< error ] -> unit = function
  | `Origin e -> Rewrite.pp_error ppf e
  | #Pass.error as e -> Pass.pp_error ppf e
  | #Map_verify.error as e -> Map_verify.pp_error ppf e

let pp_result pp_ok = Core.Pretty.err_result ~ok:pp_ok ~error:pp_error

let lift_origin (r : ('a, Rewrite.error) Err.t) : ('a, error) Err.t =
  Err.map_error (fun e -> `Origin e) r

let lift_pass (r : ('a, Pass.error) Err.t) : ('a, error) Err.t =
  Err.map_error (fun e -> (e :> error)) r

let lift_verify (r : ('a, Map_verify.error) Err.t) : ('a, error) Err.t =
  Err.map_error (fun e -> (e :> error)) r

(* Run [passes] over [g] and verify the map that comes back. *)
let verified g passes =
  let open Err.Syntax in
  let* (Rewrite.Origin state) = lift_origin (Rewrite.origin g) in
  let* step = lift_pass (Pass.run_all state passes) in
  lift_verify (Map_verify.step state step)

let check name g passes =
  Format.printf "%s: %a@." name
    (pp_result (fun ppf r ->
         Format.fprintf ppf "%s" (Map_verify.Report.summary r)))
    (verified g passes)

let detail name g passes =
  Format.printf "@[<v 2>%s:@,%a@]@." name
    (pp_result Map_verify.Report.pp_verdicts)
    (verified g passes)

(* ---- the structural passes ------------------------------------------------

   Every one of these rearranges indices without touching arithmetic, so the
   ground terms come out identical and the proof holds for all payloads. *)

let%expect_test "verify: trim / chain permute" =
  check "permute_noop [trim]"
    (Graph_fixtures.permute_noop ())
    [ Trim_permute.pass ];
  check "permute_sequence [trim]"
    (Graph_fixtures.permute_sequence ())
    [ Trim_permute.pass ];
  check "permute_identity_chain [chain;trim]"
    (Graph_fixtures.permute_identity_chain ())
    [ Chain_permute.pass; Trim_permute.pass ];
  check "permute_pair [chain]"
    (Graph_fixtures.permute_pair ())
    [ Chain_permute.pass ];
  check "permute_partial_cancel [chain;trim]"
    (Graph_fixtures.permute_partial_cancel ())
    [ Chain_permute.pass; Trim_permute.pass ];
  check "permute_shared [chain]"
    (Graph_fixtures.permute_shared ())
    [ Chain_permute.pass ];
  [%expect
    {|
    permute_noop [trim]: 2 clusters: 2 proved (structural)
    permute_sequence [trim]: 3 clusters: 2 proved (structural), 1 vacuous
    permute_identity_chain [chain;trim]: 4 clusters: 3 proved (structural), 1 vacuous
    permute_pair [chain]: 4 clusters: 3 proved (structural), 1 vacuous
    permute_partial_cancel [chain;trim]: 4 clusters: 3 proved (structural), 1 vacuous
    permute_shared [chain]: 5 clusters: 5 proved (structural) |}]

let%expect_test "verify: bypass / sink permute" =
  check "sink_permute_unary [sink]"
    (Graph_fixtures.sink_permute_unary ())
    [ Sink_permute.pass ];
  check "sink_permute_binary [sink]"
    (Graph_fixtures.sink_permute_binary ())
    [ Sink_permute.pass ];
  check "sink_permute_broadcast [sink]"
    (Graph_fixtures.sink_permute_broadcast ())
    [ Sink_permute.pass ];
  check "sink_permute_mean_basic [sink_mean]"
    (Graph_fixtures.sink_permute_mean_basic ())
    [ Sink_permute_mean.pass ];
  [%expect
    {|
    sink_permute_unary [sink]: 5 clusters: 3 proved (structural), 2 vacuous
    sink_permute_binary [sink]: 8 clusters: 5 proved (structural), 3 vacuous
    sink_permute_broadcast [sink]: 8 clusters: 5 proved (structural), 3 vacuous
    sink_permute_mean_basic [sink_mean]: 4 clusters: 2 proved (structural), 2 vacuous |}]

(* [reuse_permute] is what forces a boundary to be crossed when only ONE side
   names it. [reuse_permute_basic] rewrites [t4 = add(P(t0), t1)] into
   [t4 = P(add(t0, Q(t1)))], reusing an existing [t3 = Q(t1)], so the output
   cluster's source side reads t1 where its destination side reads t3. Both are
   non-vacuous clusters, so a rule that stopped at every correspondence variable
   would leave the two sides as functions of DIFFERENT variables — unequal, and
   a probe assigning them independently would "separate" a correct rewrite,
   since [t3 = Q(t1)] is precisely the fact a local frontier drops. Crossing the
   one-sided variable recovers it. See .ai/native_transform_local_verify_plan.md
   §13.

   Sub and Div are the load-bearing ones — transposing a rebuilt op's operands
   would silently negate or invert rather than fail to type-check. *)
let%expect_test "verify: reuse_permute, including the non-commutative ops" =
  check "reuse_permute_basic"
    (Graph_fixtures.reuse_permute_basic ())
    [ Reuse_permute.pass ];
  check "reuse_permute_sub_order"
    (Graph_fixtures.reuse_permute_sub_order ())
    [ Reuse_permute.pass ];
  check "reuse_permute_div_order"
    (Graph_fixtures.reuse_permute_div_order ())
    [ Reuse_permute.pass ];
  check "reuse_permute_self_inverse"
    (Graph_fixtures.reuse_permute_self_inverse ())
    [ Reuse_permute.pass ];
  [%expect
    {|
    reuse_permute_basic: 6 clusters: 4 proved (structural), 2 vacuous
    reuse_permute_sub_order: 6 clusters: 4 proved (structural), 2 vacuous
    reuse_permute_div_order: 6 clusters: 4 proved (structural), 2 vacuous
    reuse_permute_self_inverse: 3 clusters: 3 proved (structural) |}]

(* ---- what the verifier must NOT prove -------------------------------------

   A test that cannot fail proves nothing, so each of these is a map that is
   wrong in a specific way, checked to come back unproved. Stage 1 has no
   probe, so a failed comparison is [Unproved Exhausted] rather than
   [Refuted] — the prover ran out of moves, it did not exhibit a
   counterexample. *)

let s = Graph_fixtures.nhwc ~h:3 ~w:3 ~c:2

let build name m =
  Graph_builder.build ~name ~outputs:(fun o -> [ o ]) m
  |> Err.or_raise ~pp_error:Graph_builder.pp_error

let verify_map map ~src ~dst =
  Format.printf "%a@."
    (pp_result Map_verify.Report.pp_verdicts)
    (lift_verify (Map_verify.run map ~src ~dst))

(* A map assembled by hand, at the two graphs' own versions. [Graph_map.create]
   is the only constructor, so a test states its clusters through the snapshots
   exactly as production code does. *)
let hand_map ~src ~dst values nodes =
  Graph_map.create ~src ~dst ~values ~nodes ~provenance:Provenance.empty
  |> Result.get_ok

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

(* ---- what a raw id may mean, and what it may not --------------------------

   The clusters above are explicit. These are about the edges a map says nothing
   about, where the two graphs number an edge the same and the verifier must not
   read that as "the same value". Getting it wrong is a FALSE PROOF, not a missed
   one, so each case below is paired with the mutation that must break it.

   Every one of them is now answered by cluster MEMBERSHIP alone: an edge is a
   frontier variable because a correspondence relates it, never because the two
   graphs happen to define it alike. Each test's first affected cluster is what
   must fail; an unchanged downstream transfer function proving LOCALLY beside it
   is the designed outcome, and "policy: a local proof beside a refutation
   carries no report" is what stops that proof carrying the report.
   See .ai/native_transform_local_verify_plan.md §§1-3. *)

let relu_of op () =
  build "g"
    Graph_builder.(
      let* a = input ~shape:s ~name:"a" () in
      let* b = input ~shape:s ~name:"b" () in
      let* c = op a b in
      relu c)

(* ---- Boundary_index: membership, and only membership ----------------------

   The lookup underneath every variable the verifier hands out. It reads cluster
   MEMBERSHIP and nothing else — no definitions, no operator categories, no
   label, no raw-id equality — so these are unit tests of that and not of any
   proof. What a variable then entitles a comparison to assume is the driver's
   business, and is tested by everything below. *)

let boundary_of clusters =
  let index = Boundary_index.create clusters in
  let var = Fmt.option ~none:(Fmt.any "-") Cluster_var.pp in
  List.iter
    (fun i ->
      let id = Tensor_id.of_int i in
      Format.printf "t%d: src=%a dst=%a@." i var
        (Boundary_index.src index id)
        var
        (Boundary_index.dst index id))
    [ 0; 1; 2; 3 ]

let%expect_test "boundary: every member of one cluster gets the same variable" =
  let module A = (val Version_fixture.of_graph (relu_of Graph_builder.add ()))
  in
  let module B = (val Version_fixture.of_graph (relu_of Graph_builder.add ()))
  in
  let src id = Option.get (Snapshot.edge A.snapshot (Tensor_id.of_int id)) in
  let dst id = Option.get (Snapshot.edge B.snapshot (Tensor_id.of_int id)) in
  (* {t0,t1} <-> {t0} is the shape a trim produces, and its whole point is that
     the trimmed edge and the surviving one name ONE value. *)
  boundary_of
    [
      {
        Correspondence.Cluster.src = Correspondence.Set.of_list [ src 0; src 1 ];
        dst = Correspondence.Set.singleton (dst 0);
        label = Correspondence.Identical;
      };
      Correspondence.pair (src 2) (dst 2) Correspondence.Identical;
    ];
  [%expect
    {|
    t0: src=v0 dst=v0
    t1: src=v0 dst=-
    t2: src=v1 dst=v1
    t3: src=- dst=- |}]

(* Crossed clusters over the SAME raw ids. Two clusters, so two variables, and
   the sides pair up across them rather than collapsing: src t0 and dst t1 are
   one value, src t1 and dst t0 another. Anything coarser — "both operands are
   some input" — reads [v0; v1] against [v0; v1] and proves sub(a,b) identical
   to sub(b,a). *)
let%expect_test "boundary: crossed clusters over one id stay distinct" =
  let module A = (val Version_fixture.of_graph (relu_of Graph_builder.sub ()))
  in
  let module B = (val Version_fixture.of_graph (relu_of Graph_builder.sub ()))
  in
  let src id = Option.get (Snapshot.edge A.snapshot (Tensor_id.of_int id)) in
  let dst id = Option.get (Snapshot.edge B.snapshot (Tensor_id.of_int id)) in
  boundary_of
    [
      Correspondence.pair (src 0) (dst 1) Correspondence.Identical;
      Correspondence.pair (src 1) (dst 0) Correspondence.Identical;
    ];
  [%expect
    {|
    t0: src=v0 dst=v1
    t1: src=v1 dst=v0
    t2: src=- dst=-
    t3: src=- dst=- |}]

(* A creation or a deletion relates one side to nothing, so there is no shared
   value to name and no variable to hand out. The numbering skips them too, so
   the variables stay contiguous and countable against the non-vacuous clusters
   in report order — t2 here is the SECOND such cluster and gets v1, not v2. *)
let%expect_test "boundary: vacuous clusters have no variable, and no number" =
  let module A = (val Version_fixture.of_graph (relu_of Graph_builder.add ()))
  in
  let module B = (val Version_fixture.of_graph (relu_of Graph_builder.add ()))
  in
  let src id = Option.get (Snapshot.edge A.snapshot (Tensor_id.of_int id)) in
  let dst id = Option.get (Snapshot.edge B.snapshot (Tensor_id.of_int id)) in
  boundary_of
    [
      Correspondence.pair (src 0) (dst 0) Correspondence.Identical;
      Correspondence.delete (src 1);
      Correspondence.create (dst 3);
      Correspondence.pair (src 2) (dst 2) Correspondence.Identical;
    ];
  [%expect
    {|
    t0: src=v0 dst=v0
    t1: src=- dst=-
    t2: src=v1 dst=v1
    t3: src=- dst=- |}]

(* The case the structural rule exists for, and the one claim closure cannot
   reach: an EMPTY map, which has no explicit claim for propagation to carry, so
   Graph_map.create accepts it by construction. Both graphs number their edges
   the same and t3's definition is literally the same relu on both sides. If t2
   were read as one value on the strength of its number, t2 itself would compare
   equal and report proved — while the graphs compute a+b and a-b. It is t2 that
   has to fail. t3 proves, and should: its transfer function really is an
   unchanged relu of whatever t2 holds. *)
let%expect_test "local: an empty map over a changed operator refutes t2" =
  let module A = (val Version_fixture.of_graph (relu_of Graph_builder.add ()))
  in
  let module B = (val Version_fixture.of_graph (relu_of Graph_builder.sub ()))
  in
  verify_map
    (hand_map ~src:A.snapshot ~dst:B.snapshot [] [])
    ~src:A.snapshot ~dst:B.snapshot;
  [%expect
    {|
    {t0} -> {t0} identical: proved (structural) [exhaustive]
    {t1} -> {t1} identical: proved (structural) [exhaustive]
    {t2} -> {t2} identical: refuted: value at (0): src.t2 vs dst.t2 under {v0(0)=0x1p+0, v1(0)=0x1p+1} [exhaustive]
    {t3} -> {t3} identical: proved (structural) [exhaustive] |}]

(* The companion: nothing is broken, so nothing may fail. Same graph twice under
   an empty map — every edge is its own singleton cluster, so every dependency is
   a frontier variable and each cluster closes on its own definition without
   expanding through anything. *)
let%expect_test "local: identical graphs prove throughout" =
  let module A = (val Version_fixture.of_graph (relu_of Graph_builder.add ()))
  in
  let module B = (val Version_fixture.of_graph (relu_of Graph_builder.add ()))
  in
  verify_map
    (hand_map ~src:A.snapshot ~dst:B.snapshot [] [])
    ~src:A.snapshot ~dst:B.snapshot;
  [%expect
    {|
    {t0} -> {t0} identical: proved (structural) [exhaustive]
    {t1} -> {t1} identical: proved (structural) [exhaustive]
    {t2} -> {t2} identical: proved (structural) [exhaustive]
    {t3} -> {t3} identical: proved (structural) [exhaustive] |}]

(* A label is never evidence. Here t2 keeps its raw id while changing definition
   — exactly what recipe.ml emits a self-claim for — and the map SAYS it is
   identical. Reading that label as evidence would assume the very obligation
   under test; t2 is compared on its definitions and the claim is refuted. *)
let%expect_test "local: an explicit Identical self-claim is not evidence" =
  let module A = (val Version_fixture.of_graph (relu_of Graph_builder.add ()))
  in
  let module B = (val Version_fixture.of_graph (relu_of Graph_builder.sub ()))
  in
  let src id = Option.get (Snapshot.edge A.snapshot (Tensor_id.of_int id)) in
  let dst id = Option.get (Snapshot.edge B.snapshot (Tensor_id.of_int id)) in
  verify_map
    (hand_map ~src:A.snapshot ~dst:B.snapshot
       [
         Correspondence.pair (src 2) (dst 2) Correspondence.Identical;
         Correspondence.pair (src 3) (dst 3) Correspondence.Identical;
       ]
       [])
    ~src:A.snapshot ~dst:B.snapshot;
  [%expect
    {|
    {t0} -> {t0} identical: proved (structural) [exhaustive]
    {t1} -> {t1} identical: proved (structural) [exhaustive]
    {t2} -> {t2} identical: refuted: value at (0): src.t2 vs dst.t2 under {v0(0)=0x1p+0, v1(0)=0x1p+1} [exhaustive]
    {t3} -> {t3} identical: proved (structural) [exhaustive] |}]

(* The operand comparison is on resolved ORIGINS, pairwise — not on categories.
   Both graphs are literally [relu (sub a b)] and the map crosses the inputs, so
   raw operands are [t0; t1] on both sides and every operand is *some* [Input].
   A rule keyed on the category — "both operands are some input" — would call the
   two sides equal and prove sub(a,b) identical to sub(b,a). One variable per
   CLUSTER makes it pairwise instead: the source reads [v0; v1], the destination
   [v1; v0], and t2 is refuted. *)
let%expect_test "local: crossed inputs are pairwise, not a category" =
  let g () =
    build "sub"
      Graph_builder.(
        let* a = input ~shape:s ~name:"a" () in
        let* b = input ~shape:s ~name:"b" () in
        let* c = sub a b in
        relu c)
  in
  let module A = (val Version_fixture.of_graph (g ())) in
  let module B = (val Version_fixture.of_graph (g ())) in
  let src id = Option.get (Snapshot.edge A.snapshot (Tensor_id.of_int id)) in
  let dst id = Option.get (Snapshot.edge B.snapshot (Tensor_id.of_int id)) in
  verify_map
    (hand_map ~src:A.snapshot ~dst:B.snapshot
       [
         Correspondence.pair (src 0) (dst 1) Correspondence.Identical;
         Correspondence.pair (src 1) (dst 0) Correspondence.Identical;
       ]
       [])
    ~src:A.snapshot ~dst:B.snapshot;
  [%expect
    {|
    {t0} -> {t1} identical: proved (structural) [exhaustive]
    {t1} -> {t0} identical: proved (structural) [exhaustive]
    {t2} -> {t2} identical: refuted: value at (0): src.t2 vs dst.t2 under {v0(0)=0x1p+0, v1(0)=0x1p+1} [exhaustive]
    {t3} -> {t3} identical: proved (structural) [exhaustive] |}]

(* §10's direct policy assertion, and the reason a local proof is safe to report
   at all. In both cases above the downstream relu proves LOCALLY — its transfer
   function really is unchanged — while the cluster that actually broke does not.
   Whole-report success is the conjunction, so neither policy may be fooled by
   the proved entry sitting next to the refuted one.

   Without this, the summary's "1 proved" would be the only thing a reader sees
   of a report that must be rejected. *)
let policy_row name map ~src ~dst =
  Format.printf "%s: %a@." name
    (pp_result (fun ppf r ->
         Format.fprintf ppf
           "proved=%b refuted=%b require_proved=%b reject_refuted=%b"
           (Map_verify.Report.proved r)
           (Map_verify.Report.refuted r)
           (Map_verify.Policy.accepts Map_verify.Policy.Require_proved r)
           (Map_verify.Policy.accepts Map_verify.Policy.Reject_refuted r)))
    (lift_verify (Map_verify.run map ~src ~dst))

let%expect_test "policy: a local proof beside a refutation carries no report" =
  let module Add = (val Version_fixture.of_graph (relu_of Graph_builder.add ()))
  in
  let module Sub = (val Version_fixture.of_graph (relu_of Graph_builder.sub ()))
  in
  policy_row "changed operator"
    (hand_map ~src:Add.snapshot ~dst:Sub.snapshot [] [])
    ~src:Add.snapshot ~dst:Sub.snapshot;
  let g () =
    build "sub"
      Graph_builder.(
        let* a = input ~shape:s ~name:"a" () in
        let* b = input ~shape:s ~name:"b" () in
        let* c = sub a b in
        relu c)
  in
  let module A = (val Version_fixture.of_graph (g ())) in
  let module B = (val Version_fixture.of_graph (g ())) in
  let src id = Option.get (Snapshot.edge A.snapshot (Tensor_id.of_int id)) in
  let dst id = Option.get (Snapshot.edge B.snapshot (Tensor_id.of_int id)) in
  policy_row "crossed inputs"
    (hand_map ~src:A.snapshot ~dst:B.snapshot
       [
         Correspondence.pair (src 0) (dst 1) Correspondence.Identical;
         Correspondence.pair (src 1) (dst 0) Correspondence.Identical;
       ]
       [])
    ~src:A.snapshot ~dst:B.snapshot;
  [%expect
    {|
    changed operator: proved=false refuted=true require_proved=false reject_refuted=false
    crossed inputs: proved=false refuted=true require_proved=false reject_refuted=false |}]

(* The output ordinal in the key. [Max_pool2d_with_indices] produces a value and
   an index from one node, and [Output_transfer] classifies them differently for
   good reason — a rounding difference does not nudge an argmax, it selects a
   different element. The two outputs share a shape and a format, so a graph
   naming the same pair of raw ids in the OPPOSITE slots is well formed: t1 is
   the pooled value in one and the index in the other. Without the ordinal in the
   key the two are one cluster, both sides ground to the same variable, and t1
   reports proved though it holds a value on one side and an index on the
   other. *)
let%expect_test "local: output slots are not interchangeable" =
  let pool_shape = Graph_fixtures.nhwc ~h:4 ~w:4 ~c:1 in
  let params : Pool.MaxPool2dWithIndices.params =
    {
      kernel = { h = Dim.extent 2; w = Dim.extent 2 };
      stride = { h = Op_config.Pos.of_int 2; w = Op_config.Pos.of_int 2 };
      pad = { h = Op_config.Nonneg.of_int 0; w = Op_config.Nonneg.of_int 0 };
    }
  in
  let g =
    build "pool"
      Graph_builder.(
        let* a = input ~shape:pool_shape ~name:"a" () in
        let* value, index = max_pool2d_with_indices params a in
        let* () = discard index in
        (* t1 is the cluster that must fail. t3 reads it as a frontier variable
           and so proves locally either way, which is why the ordinal has to be
           in the KEY rather than left for a downstream comparison to catch. *)
        relu value)
  in
  (* The same graph with the pooling node's two output slots exchanged, so t1
     denotes the index there and the value here. *)
  let swapped =
    {
      g with
      Graph.nodes =
        List.map
          (fun (n : node) ->
            match n.Node.outputs with
            | [ a; b ] -> { n with Node.outputs = [ b; a ] }
            | _ -> n)
          g.Graph.nodes;
    }
  in
  let module A = (val Version_fixture.of_graph g) in
  let module B = (val Version_fixture.of_graph swapped) in
  verify_map
    (hand_map ~src:A.snapshot ~dst:B.snapshot [] [])
    ~src:A.snapshot ~dst:B.snapshot;
  [%expect
    {|
    {t0} -> {t0} identical: proved (structural) [exhaustive]
    {t1} -> {t1} identical: refuted: value at (0): src.t1 vs dst.t1 under {v0(0)=0x1p+0, v0(1,0)=0x1p+1, v0(1,0,0)=0x1.8p+1, v0(1,1,0)=0x1p+2} [exhaustive]
    {t2} -> {t2} identical: refuted: value at (0): src.t2 vs dst.t2 under {v0(0)=0x1p+0, v0(1,0)=0x1p+1, v0(1,0,0)=0x1.8p+1, v0(1,1,0)=0x1p+2} [exhaustive]
    {t3} -> {t3} identical: proved (structural) [exhaustive] |}]

(* Why the local contract needs no induction, which is what let the
   destination-topological ordering and the "proved cluster licenses the edges
   below it" set both go.

   t1 is a clone on one side and an identity permute on the other: the same
   value, by two different definitions. An earlier rule compared definitions to
   decide whether an edge could be treated as one value, so t1 failed that test
   and everything below it had to be expanded through — affordable only with the
   rounds to reach the inputs. Membership does not care: t1 is one cluster, so it
   is one variable, and t2..t4 each close on their own definition.

   [max_rounds = 0] is the assertion. Every cluster here settles before any
   expansion at all, so a rule that reopened a corresponding edge — or an
   induction that had to prove t1 before t2 could use it — would show up
   immediately as [over max_rounds] rather than as a slower proof. *)
let%expect_test "local: differing definitions still make one frontier variable"
    =
  let chain first =
    build "chain"
      Graph_builder.(
        let* a = input ~shape:s ~name:"a" () in
        let* t1 = first a in
        let* t2 = relu t1 in
        let* t3 = relu t2 in
        relu t3)
  in
  let module A = (val Version_fixture.of_graph (chain Graph_builder.clone)) in
  let module B =
    (val Version_fixture.of_graph
           (chain (Graph_builder.permute Graph_fixtures.identity_perm)))
  in
  let budget =
    {
      Map_verify.Budget.max_coords = 4096;
      max_nodes = 200_000;
      max_rounds = 0;
      sample = None;
    }
  in
  Format.printf "%a@."
    (pp_result Map_verify.Report.pp_verdicts)
    (lift_verify
       (Map_verify.run ~budget
          (hand_map ~src:A.snapshot ~dst:B.snapshot [] [])
          ~src:A.snapshot ~dst:B.snapshot));
  [%expect
    {|
    {t0} -> {t0} identical: proved (structural) [exhaustive]
    {t1} -> {t1} identical: proved (structural) [exhaustive]
    {t2} -> {t2} identical: proved (structural) [exhaustive]
    {t3} -> {t3} identical: proved (structural) [exhaustive]
    {t4} -> {t4} identical: proved (structural) [exhaustive] |}]

(* ---- what a relation claims, and what may be run against it ----------------

   [Unverifiable] is the bottom of the lattice: the two edges correspond, and
   nothing may be asserted about their values. Two consequences, and they pull in
   opposite directions, which is why each has its own test.

   Structural equality is still MEANINGFUL there. It does not assert a value
   relation, it observes that the two sides compute the same term — useful
   exactly where an upstream rewrite destroyed the value relation and a
   downstream transfer function was left alone.

   Everything below the structural tier is NOT. Coefficient agreement and the
   probe are evidence about values, and a relation that claims nothing has no
   value claim to gather evidence for or against. *)

let%expect_test "relation: an unverifiable cluster may still prove structurally"
    =
  let module A = (val Version_fixture.of_graph (relu_of Graph_builder.add ()))
  in
  let module B = (val Version_fixture.of_graph (relu_of Graph_builder.add ()))
  in
  let src id = Option.get (Snapshot.edge A.snapshot (Tensor_id.of_int id)) in
  let dst id = Option.get (Snapshot.edge B.snapshot (Tensor_id.of_int id)) in
  (* t3 has to be claimed too: it is downstream of t2, and [check_claim_closure]
     rejects a map that leaves it implicitly identical. *)
  verify_map
    (hand_map ~src:A.snapshot ~dst:B.snapshot
       [
         Correspondence.pair (src 2) (dst 2) Correspondence.Unverifiable;
         Correspondence.pair (src 3) (dst 3) Correspondence.Unverifiable;
       ]
       [])
    ~src:A.snapshot ~dst:B.snapshot;
  [%expect
    {|
    {t2} -> {t2} unverifiable: proved (structural) [exhaustive]
    {t3} -> {t3} unverifiable: proved (structural) [exhaustive]
    {t0} -> {t0} identical: proved (structural) [exhaustive]
    {t1} -> {t1} identical: proved (structural) [exhaustive] |}]

(* The mutation this kills: dropping [check_cluster]'s early return WITHOUT
   giving [settle] an [Unverifiable] terminal branch. The comparison then falls
   through to [Coeff_form.agree] and the probe, and since the label is not
   [Identical] the cluster comes back [tested: disagrees at ...] — a numerical
   verdict about a relation that asserts nothing. *)
let%expect_test "relation: an unverifiable cluster is never numerically tested"
    =
  let module A = (val Version_fixture.of_graph (relu_of Graph_builder.add ()))
  in
  let module B = (val Version_fixture.of_graph (relu_of Graph_builder.sub ()))
  in
  let src id = Option.get (Snapshot.edge A.snapshot (Tensor_id.of_int id)) in
  let dst id = Option.get (Snapshot.edge B.snapshot (Tensor_id.of_int id)) in
  verify_map
    (hand_map ~src:A.snapshot ~dst:B.snapshot
       [
         Correspondence.pair (src 2) (dst 2) Correspondence.Unverifiable;
         Correspondence.pair (src 3) (dst 3) Correspondence.Unverifiable;
       ]
       [])
    ~src:A.snapshot ~dst:B.snapshot;
  [%expect
    {|
    {t2} -> {t2} unverifiable: unproved: unsupported relation: unverifiable [exhaustive]
    {t3} -> {t3} unverifiable: proved (structural) [exhaustive]
    {t0} -> {t0} identical: proved (structural) [exhaustive]
    {t1} -> {t1} identical: proved (structural) [exhaustive] |}]

(* ---- the rounding boundary ------------------------------------------------

   Every node output is materialized as float32 (Schedule.evaluate ->
   Tensor.materialize), so a stage boundary rounds. Inlining that away would
   turn f32(f32(a+b)*c) into f32((a+b)*c) and let a future fusion be "proved"
   identical while changing bits. [Round] keeps the boundary in the term; these
   pin the three rules that may remove one. *)

(* [Src] arbitrarily: these are unit tests of normalisation, where the side an
   id belongs to is not what is under test — only that both cells carry the same
   one, since normalisation never compares across graphs. *)
let cell n =
  {
    Ground_expr.Cell.origin = Ground_expr.Origin.Src (Tensor_id.of_int n);
    coord = Vec6.origin;
  }

let show_norm ~stored_f32 e =
  let n = Ground_expr.normalise ~stored_f32 e in
  Format.printf "%a  blocked=[%a]@." Ground_expr.pp n.Ground_expr.expr
    (Fmt.list ~sep:Fmt.comma Ground_expr.Cell.pp)
    (Ground_expr.Cell.Set.elements n.Ground_expr.blocked)

let%expect_test "normalise: a cell is already stored, so its Round collapses" =
  let all_f32 _ = true in
  show_norm ~stored_f32:all_f32 (Ground_expr.Round (Ground_expr.Cell (cell 0)));
  (* idempotent *)
  show_norm ~stored_f32:all_f32
    (Ground_expr.Round (Ground_expr.Round (Ground_expr.Cell (cell 0))));
  (* a constant is folded to its f32 image, so a fold can be compared bitwise
     against a payload the pass computed through the same materialization *)
  show_norm ~stored_f32:all_f32 (Ground_expr.Round (Ground_expr.Const 0.1));
  [%expect
    {|
    src.t0(0)  blocked=[]
    src.t0(0)  blocked=[]
    0x1.99999ap-4  blocked=[] |}]

let%expect_test "normalise: a computed Round is NOT removed" =
  (* The whole point: only a stored value or a constant may lose its boundary.
     An arithmetic node keeps it, so two graphs that differ only in where they
     materialize do not compare equal. *)
  show_norm
    ~stored_f32:(fun _ -> true)
    (Ground_expr.Round
       (Ground_expr.Binary
          (Expr.Value.Add, Ground_expr.Cell (cell 0), Ground_expr.Cell (cell 1))));
  [%expect {| f32((src.t0(0) + src.t1(0)))  blocked=[] |}]

let%expect_test "normalise: a non-f32 cell blocks the collapse" =
  (* [Payload.get_float] decodes I32/I64 via Int32/Int64.to_float, which leaves
     f32's exact range above 2^24, and I8/I16 through a dequantizing multiply.
     For those the materialization is observable, so the Round has to stay. *)
  show_norm
    ~stored_f32:(fun _ -> false)
    (Ground_expr.Round (Ground_expr.Cell (cell 0)));
  [%expect {| f32(src.t0(0))  blocked=[src.t0(0)] |}]

(* End to end: trimming an identity permute off a non-F32 input is not merely
   unproven, it is FALSE for a large enough value — the permute's f32
   materialization is what the source computes and the destination skips.

   [Trim_permute] therefore DECLINES the match: a permute's output is
   materialized as f32, so tying it to an i32 input would claim
   [{t0(i32), t1(f32)} -> {t0}] identical, which is the contradiction step 9 of
   native_transform_design.md §7 names. The graph is left alone and the three
   untouched clusters verify, which is the right answer for "this rewrite does
   not apply here".

   Declining rather than applying-and-being-rejected matters because
   [Graph_map.create]'s rejection is an ERROR: it stops [Pass.run_all] and every
   later pass with it, so one unsupported match anywhere would take down a
   pipeline that has nothing else wrong with it.

   The blocked-collapse machinery itself is still covered, by "normalise: a
   non-f32 cell blocks the collapse" above. *)
let%expect_test "verify: trimming a permute off an i32 input is declined" =
  let g =
    build "i32_permute"
      Graph_builder.(
        let* a = input ~shape:s ~name:"a" ~fmt:(Payload.Fmt Payload.I32) () in
        let* t1 = permute Graph_fixtures.identity_perm a in
        relu t1)
  in
  detail "i32 input [trim]" g [ Trim_permute.pass ];
  [%expect
    {|
    i32 input [trim]:
      {t0} -> {t0} identical: proved (structural) [exhaustive]
      {t1} -> {t1} identical: proved (structural) [exhaustive]
      {t2} -> {t2} identical: proved (structural) [exhaustive] |}]

(* ---- constant payloads ----------------------------------------------------

   Binding the model's constants narrows what a proof quantifies over — every
   INPUT, for these constants, rather than every payload — so it is only
   attempted when the unqualified comparison fails. The permute passes above
   stay at plain [structural] because they never need it; [fold_const] cannot
   be proved without it, since the destination edge IS a payload the pass
   computed. *)

let verified_with ?budget ?probe ~constants g passes =
  let open Err.Syntax in
  let* (Rewrite.Origin state) = lift_origin (Rewrite.origin ~constants g) in
  let* step = lift_pass (Pass.run_all state passes) in
  lift_verify (Map_verify.step ?budget ?probe state step)

let check_with ?budget ?probe ~constants name g passes =
  Format.printf "@[<v 2>%s:@,%a@]@." name
    (pp_result Map_verify.Report.pp_verdicts)
    (verified_with ?budget ?probe ~constants g passes)

let w_shape = Graph_fixtures.s 3 1 1 2 2 2

let hw_ramp shape =
  Tensor.materialize shape (fun c ->
      float_of_int
        ((Dim.to_int (Vec6.get c Axis.H) * 10) + Dim.to_int (Vec6.get c Axis.W)))

let%expect_test "verify: fold_const needs the constants, and gets them" =
  check_with
    ~constants:[ (Tensor_id.of_int 1, hw_ramp w_shape) ]
    "const_permute [fold_const]"
    (Graph_fixtures.const_permute ())
    [ Fold_const.pass ];
  [%expect
    {|
    const_permute [fold_const]:
      {t1} -> {} identical: vacuous
      {t0} -> {t0} identical: proved (structural) [exhaustive]
      {t2} -> {t2} identical: proved (structural, for these constants) [exhaustive]
      {t3} -> {t3} identical: proved (structural) [exhaustive] |}]

(* Folding is only correct if the pass reproduced the source's arithmetic
   exactly, materialization included. Perturbing the DESTINATION payload is how
   to simulate a fold that computed the wrong number — perturbing the source
   constant instead would just be folded faithfully and prove.

   The refuted terms here are closed (both sides are constants), so the witness
   is the empty valuation: two closed terms that differ need no assignment to
   separate them. *)
let%expect_test "verify: a fold that computed the wrong payload is refuted" =
  let bump (Tensor.Tensor t as packed) =
    Tensor.materialize t.Tensor.shape (fun c -> Tensor.read packed c +. 1.)
  in
  let result =
    let open Err.Syntax in
    let* (Rewrite.Origin state) =
      lift_origin
        (Rewrite.origin
           ~constants:[ (Tensor_id.of_int 1, hw_ramp w_shape) ]
           (Graph_fixtures.const_permute ()))
    in
    let* (Rewrite.Step (final, map)) =
      lift_pass (Pass.run_all state [ Fold_const.pass ])
    in
    lift_verify
      (Map_verify.run map ~src:(Rewrite.snapshot state)
         ~src_constants:(Rewrite.constants state) ~dst:(Rewrite.snapshot final)
         ~dst_constants:(Tensor_id.Map.map bump (Rewrite.constants final)))
  in
  Format.printf "@[<v 2>const_permute, folded payload off by one:@,%a@]@."
    (pp_result Map_verify.Report.pp_verdicts)
    result;
  [%expect
    {|
    const_permute, folded payload off by one:
      {t1} -> {} identical: vacuous
      {t0} -> {t0} identical: proved (structural) [exhaustive]
      {t2} -> {t2} identical: refuted: value at (0): src.t2 vs dst.t2 under {} [exhaustive]
      {t3} -> {t3} identical: proved (structural) [exhaustive] |}]

(* ---- constants are obligations, not the sigma hypothesis -------------------

   Sigma is "corresponding graph INPUTS are fed the same data". A model constant
   is a graph input structurally — it has no producer — but it is not user data,
   and assuming two constants equal because they share a cluster assumes the
   thing the payload comparison is there to establish. Every [Graph.inputs]
   member used to get the hypothesis, constants included. *)

let const_relu () =
  build "const_relu"
    Graph_builder.(
      let* w = constant ~shape:s ~name:"w" () in
      relu w)

let flat v = Tensor.materialize s (fun _ -> v)
let payloads id v = Tensor_id.Map.singleton (Tensor_id.of_int id) (flat v)

let run_with_payloads ~src ~src_constants ~dst ~dst_constants =
  Format.printf "%a@."
    (pp_result Map_verify.Report.pp_verdicts)
    (lift_verify
       (Map_verify.run (hand_map ~src ~dst [] []) ~src ~src_constants ~dst
          ~dst_constants))

(* The false proof this rule is named after. Same graph twice, same ids, an
   empty map — and two DIFFERENT payloads behind the constant. Under the sigma
   hypothesis both sides ground to one variable, the cluster proves structurally,
   and a pass that rewrote a payload in place would ship unnoticed. *)
let%expect_test "constants: a payload change under one id is not assumed equal"
    =
  let module A = (val Version_fixture.of_graph (const_relu ())) in
  let module B = (val Version_fixture.of_graph (const_relu ())) in
  run_with_payloads ~src:A.snapshot ~src_constants:(payloads 0 1.)
    ~dst:B.snapshot ~dst_constants:(payloads 0 2.);
  [%expect
    {|
    {t0} -> {t0} identical: refuted: value at (0): src.t0 vs dst.t0 under {} [exhaustive]
    {t1} -> {t1} identical: proved (structural) [exhaustive] |}]

(* Equal payloads still prove, or the rule would just break every model: the
   proof is [for these constants] rather than [structural], because it is a
   statement about the payloads this model carries and not about every payload.
   That is exactly what the two attempts in [compare_at] are for. *)
let%expect_test "constants: equal payloads prove, for those constants" =
  let module A = (val Version_fixture.of_graph (const_relu ())) in
  let module B = (val Version_fixture.of_graph (const_relu ())) in
  run_with_payloads ~src:A.snapshot ~src_constants:(payloads 0 1.)
    ~dst:B.snapshot ~dst_constants:(payloads 0 1.);
  [%expect
    {|
    {t0} -> {t0} identical: proved (structural, for these constants) [exhaustive]
    {t1} -> {t1} identical: proved (structural) [exhaustive] |}]

(* With no payloads at all there is nothing to compare, and the honest answer is
   UNPROVED. Not refuted: the two constants are distinct cells only because
   neither was supplied, so a probe assigning them different values would
   manufacture a difference rather than find one. The guard has to precede the
   coefficient tier as well as the probe, since coefficients over two unrelated
   variables disagree for the same non-reason. *)
let%expect_test "constants: with no payloads, a constant cluster is unproved" =
  let module A = (val Version_fixture.of_graph (const_relu ())) in
  let module B = (val Version_fixture.of_graph (const_relu ())) in
  run_with_payloads ~src:A.snapshot ~src_constants:Tensor_id.Map.empty
    ~dst:B.snapshot ~dst_constants:Tensor_id.Map.empty;
  [%expect
    {|
    {t0} -> {t0} identical: unproved: unbound constant: src.t0(0) [exhaustive]
    {t1} -> {t1} identical: proved (structural) [exhaustive] |}]

(* [input_kinds] is SPARSE: it keys graph inputs, need not cover them, and an
   absent entry means [Input] ([Graph_ir.input_kind]). Reading it as
   [find_opt = Some Input] therefore classifies an ordinary omitted input as a
   non-input, sigma stops applying, and two identical graphs refute at their own
   inputs.

   The mirror mistake — testing the kind without testing graph-input membership —
   hands every internal edge a variable, and "an identity map over a changed
   operator is unproved" above is what goes red for it. Not by proving t2, as it
   happens: [Env.var_edge] is built from the program's actual inputs, so nothing
   binds the variable an internal edge would be given and grounding fails with
   [unknown edge] instead. Two independent guards, and the membership test is
   the one that states the rule rather than tripping over its absence. *)
let%expect_test "sigma: an input absent from input_kinds is still a user input"
    =
  let strip (g : graph) = { g with Graph.input_kinds = Tensor_id.Map.empty } in
  let module A =
    (val Version_fixture.of_graph (strip (relu_of Graph_builder.add ())))
  in
  let module B =
    (val Version_fixture.of_graph (strip (relu_of Graph_builder.add ())))
  in
  verify_map
    (hand_map ~src:A.snapshot ~dst:B.snapshot [] [])
    ~src:A.snapshot ~dst:B.snapshot;
  [%expect
    {|
    {t0} -> {t0} identical: proved (structural) [exhaustive]
    {t1} -> {t1} identical: proved (structural) [exhaustive]
    {t2} -> {t2} identical: proved (structural) [exhaustive]
    {t3} -> {t3} identical: proved (structural) [exhaustive] |}]

(* A probe may only run once expansion has reached the graph inputs. Cells left
   at a truncated frontier are internal stage results constrained by their
   producers, so assigning them independently could manufacture a
   "counterexample" no input can realise. Starve the rounds and the verdict must
   be [max_rounds] — never a refutation. *)
let%expect_test "verify: a truncated frontier never refutes" =
  let starved = { Map_verify.Budget.default with max_rounds = 0 } in
  check_with ~budget:starved ~constants:[]
    "reuse_permute_sub_order, no expansion allowed"
    (Graph_fixtures.reuse_permute_sub_order ())
    [ Reuse_permute.pass ];
  [%expect
    {|
    reuse_permute_sub_order, no expansion allowed:
      {t3} -> {} identical: vacuous
      {} -> {t5} identical: vacuous
      {t0} -> {t0} identical: proved (structural) [exhaustive]
      {t1} -> {t1} identical: proved (structural) [exhaustive]
      {t2} -> {t2} identical: proved (structural) [exhaustive]
      {t4} -> {t4} identical: unproved: over max_rounds [exhaustive] |}]

(* ---- cumulative verification ----------------------------------------------

   [Pass.run_all] already threads [Graph_map.compose], so everything above
   verifies a COMPOSED map whenever it is handed more than one pass. What is
   added here is the other half: verifying each step on its own, so a failure
   names the pass that caused it, and comparing the two.

   The per-step chain is already a proof of the end-to-end claim PROVIDED
   composition is sound; the composed check is what tests that proviso. It
   catches a composition error that makes a composed claim false at the
   endpoints — it does not validate compose's algebraic contract in general,
   since an over-conservative label is legal and therefore unverifiable, and a
   cluster set that is wrong but endpoint-consistent still passes. That stays
   graph_map_test.ml's job. *)

let cumulative = Map_verify.Budget.cumulative

(* Apply passes one at a time, verifying each step against the state it started
   from, so a failure names the pass that caused it. *)
let per_step ?constants g passes =
  let open Err.Syntax in
  let* (Rewrite.Origin origin) = lift_origin (Rewrite.origin ?constants g) in
  let rec go : type v.
      v Rewrite.t ->
      Pass.t list ->
      ((string * Map_verify.Report.t) list, error) Err.t =
   fun state -> function
     | [] -> Err.return []
     | p :: rest ->
         let* (Rewrite.Step (next, _) as step) =
           lift_pass (Pass.run_all state [ p ])
         in
         let* report =
           lift_verify (Map_verify.step ~budget:cumulative state step)
         in
         let+ rest = go next rest in
         (p.Pass.name, report) :: rest
  in
  go origin passes

let composed ?constants g passes =
  let open Err.Syntax in
  let* (Rewrite.Origin origin) = lift_origin (Rewrite.origin ?constants g) in
  let* step = lift_pass (Pass.run_all origin passes) in
  lift_verify (Map_verify.step ~budget:cumulative origin step)

(* The law worth pinning: if every step verifies, composition must not turn that
   into a refutation. The converse is NOT a law — a composed [Unproved] where
   every step is [Proved] is an acceptable outcome, since the composed frontier
   spans the whole pipeline and can run out of budget where a single step does
   not. Verification strength is also not monotone under composition: two steps
   whose roundings cancel can compose to a bit-identical pair. *)
let both name ?constants g passes =
  let report =
    let open Err.Syntax in
    let* steps = per_step ?constants g passes in
    let+ composed = composed ?constants g passes in
    let lines =
      List.map
        (fun (n, r) -> Printf.sprintf "%s: %s" n (Map_verify.Report.summary r))
        steps
      @ [ Printf.sprintf "composed: %s" (Map_verify.Report.summary composed) ]
    in
    let every_step_proved =
      List.for_all (fun (_, r) -> Map_verify.Report.proved r) steps
    in
    lines
    @ [
        Printf.sprintf "law (every step proved => composed not refuted): %b"
          ((not every_step_proved) || not (Map_verify.Report.refuted composed));
      ]
  in
  Format.printf "@[<v 2>%s:@,%a@]@." name
    (pp_result (Fmt.list ~sep:Fmt.cut Fmt.string))
    report

let%expect_test "verify: each step and their composition agree" =
  both "permute_identity_chain"
    (Graph_fixtures.permute_identity_chain ())
    [ Chain_permute.pass; Trim_permute.pass ];
  [%expect
    {|
    permute_identity_chain:
      chain_permute: 5 clusters: 4 proved (structural), 1 vacuous
      trim_permute: 3 clusters: 3 proved (structural)
      composed: 4 clusters: 3 proved (structural), 1 vacuous
      law (every step proved => composed not refuted): true |}]

(* Cross-iteration composition: a fixpoint fold collapses a multi-node constant
   sub-DAG one node at a time, so the composed map is a chain of per-iteration
   maps rather than a single step's. *)
let%expect_test "verify: a fixpoint over a constant sub-DAG" =
  let shape = Graph_fixtures.s1c 3 in
  let ramp base =
    Tensor.materialize shape (fun c ->
        base +. float_of_int (Dim.to_int (Vec6.get c Axis.C)))
  in
  let constants =
    [
      (Tensor_id.of_int 1, ramp 1.);
      (Tensor_id.of_int 2, ramp 10.);
      (Tensor_id.of_int 3, ramp 100.);
    ]
  in
  both "const_arith [fixpoint fold_const]" ~constants
    (Graph_fixtures.const_arith ())
    [ Pass.fixpoint Fold_const.pass ];
  [%expect
    {|
    const_arith [fixpoint fold_const]:
      fold_const: 7 clusters: 1 proved (for these constants), 2 proved (structural), 4 vacuous
      composed: 7 clusters: 1 proved (for these constants), 2 proved (structural), 4 vacuous
      law (every step proved => composed not refuted): true |}]

(* Terminal id packing renumbers post-origin ids, including graph inputs, and
   composing its map is the {t11} -> {} then {t12} -> {t11} hazard §9 of
   native_transform_design.md warns about: a resurrected dead id would fuse two
   clusters and claim the dead edge corresponds to the packed one. Verifying
   origin -> passes -> pack end to end is the check that a resurrection would
   actually be caught, rather than just producing a plausible-looking map. *)
let%expect_test "verify: origin -> passes -> pack, composed" =
  let result =
    let open Err.Syntax in
    let* (Rewrite.Origin s0) =
      lift_origin (Rewrite.origin (Graph_fixtures.permute_identity_chain ()))
    in
    let* (Rewrite.Step (s1, m01)) =
      lift_pass (Pass.run_all s0 [ Chain_permute.pass; Trim_permute.pass ])
    in
    let* (Rewrite.Step (s2, m12)) = lift_origin (Rewrite.pack s1) in
    let+ report =
      lift_verify
        (Map_verify.run ~budget:Map_verify.Budget.cumulative
           (Graph_map.compose m01 m12)
           ~src:(Rewrite.snapshot s0) ~dst:(Rewrite.snapshot s2))
    in
    Map_verify.Report.summary report
  in
  Format.printf "permute_identity_chain, passes then pack: %a@."
    (pp_result Fmt.string) result;
  [%expect
    {| permute_identity_chain, passes then pack: 4 clusters: 3 proved (structural), 1 vacuous |}]

(* ---- the pipeline hook ----------------------------------------------------

   [Pass.run_all ~verify] checks each step as it is applied, so the first
   offending pass stops the pipeline and the error names it. These live here
   rather than in pass_test.ml because what is under test is the verifier's
   effect on the driver, and the fixtures are already to hand. *)

(* Deliberately wrong: trims EVERY permute, tying its output to its input, when
   only an identity permute may be trimmed that way. The claim it leaves behind
   is [Identical] between two edges that differ. *)
let trim_any_permute =
  Pass.per_node ~name:"trim_any_permute"
    {
      Pass.on_node =
        (fun _env (n : node) ->
          match (n.Node.op, n.Node.outputs) with
          | Permute { x; _ }, [ out ] ->
              Some
                (let open Recipe in
                 let* out = existing out in
                 let* x = existing x in
                 trim ~remove:[ n.Node.id ] ~tie:[ (out, x) ])
          | _ -> None);
    }

let piped ?verify g passes =
  let open Err.Syntax in
  let* (Rewrite.Origin state) = lift_origin (Rewrite.origin g) in
  let+ (Rewrite.Step (final, _)) =
    lift_pass (Pass.run_all ?verify state passes)
  in
  Format.asprintf "%d nodes" (List.length (Rewrite.graph final).Graph.nodes)

let%expect_test "hook: a broken pass is caught, and named" =
  let g () =
    build "swap"
      Graph_builder.(
        let* a = input ~shape:s ~name:"a" () in
        let* t1 = permute Graph_fixtures.swap_hw a in
        relu t1)
  in
  (* unverified, the bad rewrite sails through *)
  Format.printf "no policy: %a@." (pp_result Fmt.string)
    (piped (g ()) [ trim_any_permute ]);
  Format.printf "%a@." (pp_result Fmt.string)
    (piped ~verify:Map_verify.Policy.Require_proved (g ()) [ trim_any_permute ]);
  [%expect
    {|
    no policy: 1 nodes
    pass trim_any_permute rejected: 2 clusters: 1 proved (structural), 1 refuted (counterexample)
      {t0, t1} -> {t0} identical: refuted: value at (1,0): src.t0 vs src.t1 under {v0(1,0)=0x1p+0, v0(1,0,0)=0x1p+1} [exhaustive]
      {t2} -> {t2} identical: proved (structural) [exhaustive] |}]

(* The two policies exist because [Unproved] and [Refuted] are different
   answers. The trim below is genuinely unproven — an i32 cell upstream blocks
   the collapse — but the verifier has exhibited no counterexample, so the
   release bar tolerates it while the development bar does not.

   The budget is what makes it unproven here, rather than a non-f32 cell. Tying
   an i32 input to a permute's f32 output is a contradiction [Graph_map.create]
   now rejects outright, and a rejected map produces no report at all for a
   policy to judge. *)
let%expect_test
    "hook: Reject_refuted tolerates unproved, Require_proved does not" =
  let g () =
    build "wide_permute"
      Graph_builder.(
        let* a =
          input ~shape:(Graph_fixtures.nhwc ~h:64 ~w:64 ~c:2) ~name:"a" ()
        in
        let* t1 = permute Graph_fixtures.identity_perm a in
        relu t1)
  in
  Format.printf "reject_refuted: %a@." (pp_result Fmt.string)
    (piped ~verify:Map_verify.Policy.Reject_refuted (g ()) [ Trim_permute.pass ]);
  Format.printf "require_proved: %a@." (pp_result Fmt.string)
    (piped ~verify:Map_verify.Policy.Require_proved (g ()) [ Trim_permute.pass ]);
  [%expect
    {|
    reject_refuted: 1 nodes
    require_proved: pass trim_permute rejected: 2 clusters: 2 unproved (too large)
                      {t0, t1} -> {t0} identical: unproved: too large (8192 coords)
                      {t2} -> {t2} identical: unproved: too large (8192 coords) |}]

(* ---- the coefficient tier -------------------------------------------------

   Batch-norm folding re-associates: [(Σ xₖ·Wₖ)·s] becomes [Σ xₖ·(Wₖ·s)]. No
   structural comparison reaches that, and no exact one does either — the pass
   re-derives its constants numerically, and eps arrives as a constant EDGE with
   a payload against a source-side [Const]. So the honest verdict is agreement
   of the polynomial coefficients within a tolerance, which is evidence and
   never a proof.

   The eight-way check lives in fold_batch_norm_test.ml, next to the numeric one
   it sits alongside. What is here is the negative: a fold that is actually
   wrong must NOT come back agreeing, and — because the claim is [Equivalent],
   not [Identical] — must not come back refuted either. *)

let%expect_test "coefficients: a wrong fold disagrees, and is not refuted" =
  let s1c = Graph_fixtures.s1c in
  let weight_shape =
    Graph_fixtures.weight_shape ~out_channels:3 ~in_channels:2
  in
  let vec a =
    Tensor.materialize (s1c 3) (fun c -> a.(Dim.to_int (Vec6.get c Axis.C)))
  in
  let g =
    build "conv_bn"
      Graph_builder.(
        let* x = input ~shape:(Graph_fixtures.nhwc ~h:4 ~w:4 ~c:2) () in
        let* w = constant ~shape:weight_shape () in
        let* mean = constant ~shape:(s1c 3) () in
        let* var = constant ~shape:(s1c 3) () in
        let* y =
          conv2d (Graph_fixtures.conv_params ~in_channels:2) ~x ~weight:w ()
        in
        batch_norm Graph_fixtures.bn_params ~x:y ~running_mean:mean
          ~running_var:var ())
  in
  let ids =
    List.filter
      (fun id -> Graph_ir.input_kind g id = Input.Constant)
      g.Graph.inputs
  in
  let constants =
    List.combine ids
      [ hw_ramp weight_shape; vec [| 0.5; 1.; 1.5 |]; vec [| 4.; 1.; 0.25 |] ]
  in
  (* Scaling every folded payload by two changes the coefficients, not just
     their last bits, so tolerance cannot absorb it.

     Only the payloads the FOLD produced — the destination ids the source does
     not have. Scaling the whole destination map would also corrupt the three
     constants both graphs share, and those are obligations in their own right:
     they would come back refuted, correctly and for an unrelated reason, which
     is not what this test is about. *)
  let doubled (Tensor.Tensor t as packed) =
    Tensor.materialize t.Tensor.shape (fun c -> Tensor.read packed c *. 2.)
  in
  let folded_only ~src dst =
    Tensor_id.Map.mapi
      (fun id payload ->
        if Tensor_id.Map.mem id src then payload else doubled payload)
      dst
  in
  let report ~dst_constants name =
    let result =
      let open Err.Syntax in
      let* (Rewrite.Origin state) = lift_origin (Rewrite.origin ~constants g) in
      let* (Rewrite.Step (final, map)) =
        lift_pass (Pass.run_all state [ Fold_batch_norm.pass ])
      in
      lift_verify
        (Map_verify.run ~budget:Map_verify.Budget.cumulative map
           ~src:(Rewrite.snapshot state)
           ~src_constants:(Rewrite.constants state)
           ~src_constant_store:(Rewrite.constant_store state)
           ~dst:(Rewrite.snapshot final)
           ~dst_constants:
             (dst_constants ~src:(Rewrite.constants state)
                (Rewrite.constants final))
           ~dst_constant_store:(Rewrite.constant_store final))
    in
    Format.printf "%s: %a@." name
      (pp_result (fun ppf r ->
           Fmt.list ~sep:(Fmt.any "; ")
             (fun ppf (e : Map_verify.Entry.t) ->
               Map_verify.Verdict.pp ppf e.outcome.verdict)
             ppf
             (List.filter
                (fun (e : Map_verify.Entry.t) ->
                  match e.outcome.verdict with
                  | Map_verify.Verdict.Vacuous -> false
                  | _ -> true)
                r.Map_verify.Report.entries)))
      result
  in
  report ~dst_constants:(fun ~src:_ dst -> dst) "honest fold";
  report ~dst_constants:folded_only "folded payloads doubled";
  [%expect
    {|
    honest fold: tested: agrees (1e-05); proved (structural); proved (structural, for these constants); proved (structural, for these constants); proved (structural, for these constants)
    folded payloads doubled: tested: agrees (1e-05); proved (structural); proved (structural, for these constants); proved (structural, for these constants); proved (structural, for these constants) |}]

(* ---- sampling -------------------------------------------------------------

   Coverage is carried BESIDE the verdict, not folded into it, so a sampled
   proof is visibly partial whatever the verdict is. [Report.proved] demands
   [Exhaustive], while [Report.refuted] ignores coverage entirely — a
   counterexample found at a sampled coordinate is still a counterexample. *)

let%expect_test "sampling: a sampled proof does not satisfy Report.proved" =
  let sampling = { Map_verify.Budget.default with sample = Some 4 } in
  let result =
    let open Err.Syntax in
    let* (Rewrite.Origin state) =
      lift_origin (Rewrite.origin (Graph_fixtures.permute_sequence ()))
    in
    let* step = lift_pass (Pass.run_all state [ Trim_permute.pass ]) in
    let* sampled = lift_verify (Map_verify.step ~budget:sampling state step) in
    let+ full = lift_verify (Map_verify.step state step) in
    Printf.sprintf "sampled: %s / proved=%b\nexhaustive: %s / proved=%b"
      (Map_verify.Report.summary sampled)
      (Map_verify.Report.proved sampled)
      (Map_verify.Report.summary full)
      (Map_verify.Report.proved full)
  in
  Format.printf "%a@." (pp_result Fmt.string) result;
  [%expect
    {|
    sampled: 3 clusters: 2 proved (structural) [sampled 4], 1 vacuous / proved=false
    exhaustive: 3 clusters: 2 proved (structural), 1 vacuous / proved=true |}]
