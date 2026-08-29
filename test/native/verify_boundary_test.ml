(* Split out of what was verify_test.ml see verify_fixtures.ml. *)

open Graph_ir
open Verify_fixtures

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
      ceil_mode = false;
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
