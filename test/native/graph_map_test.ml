(* Exercises the transformation mapping: the claim lattice, cluster
   normalisation, the algebraic laws [compose]/[invert] must obey, endpoint
   validation, and the two cases the design turns on — trimming's many-to-one
   cluster and packing's reuse of a dead id's numeric value.
   See .ai/native_transform_design.md §3. *)

open Graph_ir
module C = Correspondence

let t n = Tensor_id.of_int n

(* The algebra below relates a version to ITSELF: normalisation, composition and
   inversion are statements about ids, and minting graphs to state them would
   obscure what is under test. One tag, wide enough for every id used here. *)
module I = (val Version_fixture.ids 32)

let set ids = C.Set.of_list (List.map I.edge ids)
let cluster ~src ~dst label = { C.Cluster.src = set src; dst = set dst; label }

(* The universes are read off the clusters: the smallest pair of versions the
   relation could describe. Endpoint validation is then vacuous by construction,
   which is right — normalisation and composition are what these cases are
   about, and the checks have their own tests below, against real graphs. *)
let of_ l =
  let side get =
    List.fold_left
      (fun acc c -> Tensor_id.Set.union acc (C.raws (get c)))
      Tensor_id.Set.empty l
  in
  Err.or_raise
    ~pp_error:(Cluster_relation.pp_issue Tensor_id.pp)
    (Result.map_error Err.Error.make
       (C.of_clusters
          ~src:(I.edges (side (fun (c : (_, _) C.Cluster.t) -> c.src)))
          ~dst:(I.edges (side (fun (c : (_, _) C.Cluster.t) -> c.dst)))
          l))

let raws s = Tensor_id.Set.elements (C.raws s)
let show pp v = Fmt.pf Fmt.stdout "@[<v>%a@]@." pp v

(* ---- the claim lattice --------------------------------------------------- *)

let f32 =
  Correspondence.Precision.{ fmt = Payload.Fmt Payload.F32; quant = None }

let bf16 =
  Correspondence.Precision.{ fmt = Payload.Fmt Payload.BF16; quant = None }

let f16 =
  Correspondence.Precision.{ fmt = Payload.Fmt Payload.F16; quant = None }

let approx l = C.Approximate (C.Precision.Set.of_list l)

let%expect_test "join: the weaker claim wins, approximations accumulate" =
  let show a b = Fmt.pf Fmt.stdout "%a@." C.pp_relation (C.join a b) in
  show C.Identical C.Identical;
  show C.Identical C.Equivalent;
  show C.Equivalent (approx [ bf16 ]);
  show (approx [ bf16 ]) (approx [ f16 ]);
  show (approx [ bf16 ]) C.Unverifiable;
  [%expect
    {|
    identical
    equivalent
    approximate(bf16)
    approximate(bf16, f16)
    unverifiable |}]

(* F16 and BF16 are incomparable — range versus mantissa — so the label records
   which representations were traversed instead of ordering them. A widening
   round trip must not read as lossless. *)
let%expect_test "join: F32 -> BF16 -> F32 stays approximate at BF16" =
  let there = approx [ bf16 ] and back = C.Identical in
  Fmt.pf Fmt.stdout "%a@." C.pp_relation (C.join there back);
  Fmt.pf Fmt.stdout "%a@." C.pp_relation (C.join there (approx [ f32 ]));
  [%expect {|
    approximate(bf16)
    approximate(bf16, f32) |}]

(* ---- how an op transfers a claim ----------------------------------------- *)

(* [Output_transfer] is what [Rewrite] consults per output when propagating a
   claim across a node. The distinction that matters for [Clone]: reindexing
   carries [Approximate] through, continuity does not — continuity gives no
   error BOUND, so an approximate claim dies at any actual arithmetic. Clone
   performs none, so classifying it as continuous would lose a claim the graph
   really does still guarantee. *)
let%expect_test "output_transfer: clone is reindexing and keeps Approximate" =
  let show op =
    let cls = Output_transfer.classify op ~output:0 in
    Fmt.pf Fmt.stdout "%a: %a -> %a@." Output_transfer.pp cls C.pp_relation
      (approx [ bf16 ]) C.pp_relation
      (Output_transfer.transfer (approx [ bf16 ]) cls)
  in
  show (Clone { Pointwise.Clone.x = t 0 });
  show (Relu { Pointwise.Relu.x = t 0 });
  show (Add_scalar { Pointwise.Scalar_bin.x = t 0; scalar = 3. });
  show
    (Clamp { Pointwise.Clamp.params = { min = Some 0.; max = None }; x = t 0 });
  show
    (Hardtanh
       { Pointwise.Hardtanh.params = { min_val = 0.; max_val = 6. }; x = t 0 });
  [%expect
    {|
    reindexing: approximate(bf16) -> approximate(bf16)
    continuous: approximate(bf16) -> unverifiable
    continuous: approximate(bf16) -> unverifiable
    continuous: approximate(bf16) -> unverifiable
    continuous: approximate(bf16) -> unverifiable |}]

(* [Unbind] is the case that made the class's definition matter. Each output is
   a SLICE, so the outputs' value multisets partition the input's rather than
   each reproducing it — it is not a permutation, and "the value multiset is
   unchanged" would not license it. The rule that does is the one the class
   actually needs: every output element is COPIED from an input element with no
   arithmetic, and an [Approximate] bound is per-element, so a slice carries it
   exactly as a permutation does.

   Every ordinal is asked, since the classifier takes [~output] and a per-output
   answer is exactly what this op could get wrong. *)
let%expect_test "output_transfer: every unbind slice keeps Approximate" =
  let op = Unbind { Split.Unbind.params = { axis = Axis.C }; x = t 0 } in
  List.iter
    (fun output ->
      let cls = Output_transfer.classify op ~output in
      Fmt.pf Fmt.stdout "out%d %a: %a -> %a@." output Output_transfer.pp cls
        C.pp_relation (approx [ bf16 ]) C.pp_relation
        (Output_transfer.transfer (approx [ bf16 ]) cls))
    [ 0; 1; 2 ];
  [%expect
    {|
    out0 reindexing: approximate(bf16) -> approximate(bf16)
    out1 reindexing: approximate(bf16) -> approximate(bf16)
    out2 reindexing: approximate(bf16) -> approximate(bf16) |}]

(* ---- normalisation ------------------------------------------------------- *)

let%expect_test
    "normalisation: clusters sharing an id merge, self-identity drops" =
  (* Two clusters mapping onto t0 are one cluster; {t9} -> {t9} identical is
     implicit and disappears; an approximate self-link must survive, since a
     changed value must never hide in the implicit bulk. *)
  show C.pp
    (of_
       [
         cluster ~src:[ 1 ] ~dst:[ 0 ] C.Identical;
         cluster ~src:[ 2 ] ~dst:[ 0 ] C.Identical;
         cluster ~src:[ 9 ] ~dst:[ 9 ] C.Identical;
         cluster ~src:[ 7 ] ~dst:[ 7 ] (approx [ bf16 ]);
       ]);
  [%expect
    {|
    {t1, t2} -> {t0} identical
    {t7} -> {t7} approximate(bf16) |}]

let%expect_test "normalisation: labels join when clusters merge" =
  show C.pp
    (of_
       [
         cluster ~src:[ 1 ] ~dst:[ 0 ] C.Identical;
         cluster ~src:[ 2 ] ~dst:[ 0 ] C.Equivalent;
       ]);
  [%expect {| {t1, t2} -> {t0} equivalent |}]

(* ---- trimming: the many-to-one case a partial matching cannot express ----- *)

let%expect_test "trimming an identity permute: source input joins the cluster" =
  (* A: t0 -permute-> t1, B: t0. Both the untouched input and the trimmed
     output correspond to the surviving edge. *)
  let m = of_ [ cluster ~src:[ 0; 1 ] ~dst:[ 0 ] C.Identical ] in
  show C.pp m;
  Fmt.pf Fmt.stdout "forward t1 = %a@."
    (Fmt.braces (Fmt.list ~sep:Fmt.comma Tensor_id.pp))
    (raws (C.forward m (I.edge 1)));
  [%expect {|
    {t0, t1} -> {t0} identical
    forward t1 = {t0} |}]

let%expect_test "trimming a chain widens the same cluster" =
  let m = of_ [ cluster ~src:[ 0; 1; 2 ] ~dst:[ 0 ] C.Identical ] in
  show C.pp m;
  [%expect {| {t0, t1, t2} -> {t0} identical |}]

(* ---- algebraic laws ------------------------------------------------------ *)

let%expect_test "compose with identity is the original" =
  let m = of_ [ cluster ~src:[ 0; 1 ] ~dst:[ 0 ] C.Identical ] in
  show C.pp (C.compose m C.identity);
  show C.pp (C.compose C.identity m);
  [%expect {|
    {t0, t1} -> {t0} identical
    {t0, t1} -> {t0} identical |}]

let%expect_test "invert twice is the original; invert swaps sides" =
  let m = of_ [ cluster ~src:[ 0; 1 ] ~dst:[ 0 ] C.Equivalent ] in
  show C.pp (C.invert m);
  show C.pp (C.invert (C.invert m));
  [%expect
    {|
    {t0} -> {t0, t1} equivalent
    {t0, t1} -> {t0} equivalent |}]

let%expect_test "compose joins labels across the middle" =
  (* bn fold (equivalent) followed by a bf16 pass: the strongest honest claim
     between the endpoints is approximate. *)
  let m01 = of_ [ cluster ~src:[ 0 ] ~dst:[ 1 ] C.Equivalent ] in
  let m12 = of_ [ cluster ~src:[ 1 ] ~dst:[ 2 ] (approx [ bf16 ]) ] in
  show C.pp (C.compose m01 m12);
  [%expect {| {t0} -> {t2} approximate(bf16) |}]

let%expect_test "compose is associative over renames" =
  let a = of_ [ cluster ~src:[ 0 ] ~dst:[ 1 ] C.Identical ] in
  let b = of_ [ cluster ~src:[ 1 ] ~dst:[ 2 ] C.Equivalent ] in
  let c = of_ [ cluster ~src:[ 2 ] ~dst:[ 3 ] C.Identical ] in
  show C.pp (C.compose (C.compose a b) c);
  show C.pp (C.compose a (C.compose b c));
  [%expect {|
    {t0} -> {t3} equivalent
    {t0} -> {t3} equivalent |}]

(* ---- creation, deletion, and the packing hazard -------------------------- *)

let%expect_test "created then deleted vanishes" =
  let create = of_ [ C.create (I.edge 11) ] in
  let delete = of_ [ C.delete (I.edge 11) ] in
  show C.pp (C.compose create delete);
  [%expect {| identity |}]

(* Packing may reclaim the numeric value of a DEAD post-origin id. Composition
   must not identity-extend the dead id through the intervening maps and fuse it
   with the cluster that reuses its value — and the guard that prevents it is a
   side condition on extension, so associativity has to be demonstrated, not
   assumed. See .ai/native_transform_design.md §3, §9. *)
let%expect_test "delete, create, repack: the dead id never fuses with its reuse"
    =
  let m12 = of_ [ C.delete (I.edge 11) ] in
  let m23 = of_ [ C.create (I.edge 12) ] in
  let m34 = of_ [ cluster ~src:[ 12 ] ~dst:[ 11 ] C.Identical ] in
  let left = C.compose (C.compose m12 m23) m34 in
  let right = C.compose m12 (C.compose m23 m34) in
  show C.pp left;
  show C.pp right;
  Fmt.pf Fmt.stdout "same: %b@." (C.clusters left = C.clusters right);
  [%expect
    {|
    {t11} -> {} identical
    {} -> {t11} identical
    {t11} -> {} identical
    {} -> {t11} identical
    same: true |}]

let%expect_test "created and deleted report the unpaired ids" =
  let m =
    of_
      [
        C.create (I.edge 5);
        C.delete (I.edge 3);
        cluster ~src:[ 1 ] ~dst:[ 2 ] C.Identical;
      ]
  in
  let show name s =
    Fmt.pf Fmt.stdout "%s = %a@." name
      (Fmt.braces (Fmt.list ~sep:Fmt.comma Tensor_id.pp))
      (raws s)
  in
  show "created" (C.created m);
  show "deleted" (C.deleted m);
  [%expect {|
    created = {t5}
    deleted = {t3} |}]

(* ---- node clusters ------------------------------------------------------- *)

let%expect_test "node map: fusion, creation, deletion and reverse lookup" =
  let m =
    Err.or_raise
      ~pp_error:(Cluster_relation.pp_issue Node_id.pp)
      (Result.map_error Err.Error.make
         (Node_map.of_clusters
            ~src:
              (I.nodes
                 (Node_id.Set.of_list (List.map Node_id.of_int [ 1; 2; 3 ])))
            ~dst:
              (I.nodes (Node_id.Set.of_list (List.map Node_id.of_int [ 7; 8 ])))
            [
              Node_map.fused ~from:[ I.node 1; I.node 2 ] (I.node 7);
              Node_map.fused ~from:[] (I.node 8);
              Node_map.delete (I.node 3);
            ]))
  in
  Fmt.pf Fmt.stdout "@[<v>%a@]@." Node_map.pp m;
  Fmt.pf Fmt.stdout "backward n7 = %a@."
    (Fmt.braces (Fmt.list ~sep:Fmt.comma Node_id.pp))
    (Node_id.Set.elements (Node_map.raws (Node_map.backward m (I.node 7))));
  [%expect
    {|
    {n1, n2} -> {n7}
    {n3} -> {}
    {} -> {n8}
    backward n7 = {n1, n2} |}]

(* ---- construction, which is where a map is checked ----------------------- *)

(* Version-indexed ids stop a source id being written into a destination side,
   but they cannot say the two graphs are the intended pair, nor that the labels
   mean anything. [Graph_map.create] is where that is established, and it is the
   only constructor. *)

let s1c n = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:1 ~w:1 ~c:n

let two_input_graph () =
  Graph_builder.(
    build ~name:"g" ~outputs:(fun o -> [ o ])
    @@
    let* a = input ~shape:(s1c 3) () in
    let* b = input ~shape:(s1c 3) () in
    add a b)
  |> Result.get_ok

let one_input_graph () =
  Graph_builder.(
    build ~name:"g" ~outputs:(fun o -> [ o ])
    @@
    let* a = input ~shape:(s1c 3) () in
    relu a)
  |> Result.get_ok

(* [t2 = add(a,b); t3 = relu(t2)] against [t2 = sub(a,b); t3 = relu(t2)]: same
   ids on both sides, and t3's definition is untouched, so t3 is implicitly
   identical unless the map says otherwise. *)
let relu_of op () =
  Graph_builder.(
    build ~name:"g" ~outputs:(fun o -> [ o ])
    @@
    let* a = input ~shape:(s1c 3) () in
    let* b = input ~shape:(s1c 3) () in
    let* c = op a b in
    relu c)
  |> Result.get_ok

let show_create r =
  match r with
  | Error e -> Fmt.pf Fmt.stdout "%a@." Graph_map.pp_error (Err.Error.kind e)
  | Ok _ -> Fmt.pf Fmt.stdout "ok@."

let%expect_test "create: a map of the graph onto itself passes" =
  let module A = (val Version_fixture.of_graph (two_input_graph ())) in
  let module B = (val Version_fixture.of_graph (two_input_graph ())) in
  show_create
    (Graph_map.create ~src:A.snapshot ~dst:B.snapshot ~values:[] ~nodes:[]
       ~provenance:Provenance.empty);
  [%expect {| ok |}]

let%expect_test "create: identity between differing graphs is rejected" =
  (* Both graphs exist and are well formed; nothing about the map is
     type-incorrect. Only implicit-identity coverage catches it. *)
  let module A = (val Version_fixture.of_graph (two_input_graph ())) in
  let module B = (val Version_fixture.of_graph (one_input_graph ())) in
  show_create
    (Graph_map.create ~src:A.snapshot ~dst:B.snapshot ~values:[] ~nodes:[]
       ~provenance:Provenance.empty);
  [%expect
    {| value map: t2 is implicitly identity but absent from the destination |}]

let%expect_test "create: creating an id the source already has is rejected" =
  let module A = (val Version_fixture.of_graph (two_input_graph ())) in
  let module B = (val Version_fixture.of_graph (two_input_graph ())) in
  let e id = Option.get (Snapshot.edge B.snapshot (t id)) in
  show_create
    (Graph_map.create ~src:A.snapshot ~dst:B.snapshot
       ~values:[ C.create (e 0) ]
       ~nodes:[] ~provenance:Provenance.empty);
  [%expect {| value map: t0 is mapped to but unmentioned as a source |}]

(* The claim-closure rule. t2 is claimed [Unverifiable]; t3 = relu(t2) is left
   unmentioned, hence implicitly [Identical] — and every consumer is then wrong
   about t3, not just the symbolic verifier. [Pt2_native_graph] resolves an
   unmentioned dst id to [Identical] and hands back captured SOURCE bytes for
   it, which is data corruption rather than imprecision. *)
let%expect_test "create: a claim not closed over the destination is rejected" =
  let module A = (val Version_fixture.of_graph (relu_of Graph_builder.add ()))
  in
  let module B = (val Version_fixture.of_graph (relu_of Graph_builder.sub ()))
  in
  let src id = Option.get (Snapshot.edge A.snapshot (t id)) in
  let dst id = Option.get (Snapshot.edge B.snapshot (t id)) in
  show_create
    (Graph_map.create ~src:A.snapshot ~dst:B.snapshot
       ~values:[ C.pair (src 2) (dst 2) C.Unverifiable ]
       ~nodes:[] ~provenance:Provenance.empty);
  [%expect
    {| value map: t3 is implicitly identical but downstream of a weaker claim |}]

let%expect_test "create: the same map with t3 spoken for is accepted" =
  let module A = (val Version_fixture.of_graph (relu_of Graph_builder.add ()))
  in
  let module B = (val Version_fixture.of_graph (relu_of Graph_builder.sub ()))
  in
  let src id = Option.get (Snapshot.edge A.snapshot (t id)) in
  let dst id = Option.get (Snapshot.edge B.snapshot (t id)) in
  show_create
    (Graph_map.create ~src:A.snapshot ~dst:B.snapshot
       ~values:
         [
           C.pair (src 2) (dst 2) C.Unverifiable;
           C.pair (src 3) (dst 3) C.Unverifiable;
         ]
       ~nodes:[] ~provenance:Provenance.empty);
  [%expect {| ok |}]

(* Step 9 of .ai/native_transform_design.md §7. An [Identical] claim across two
   formats is a contradiction — it is [Approximate] — and it is the claim the PT2
   lens reads to decide it may return the source's captured bytes. *)
let bf16_graph () =
  Graph_builder.(
    build ~name:"g" ~outputs:(fun o -> [ o ])
    @@
    let* a = input ~shape:(s1c 3) ~fmt:(Payload.Fmt Payload.BF16) () in
    relu a)
  |> Result.get_ok

let%expect_test "create: Identical across two formats is rejected" =
  let module A = (val Version_fixture.of_graph (one_input_graph ())) in
  let module B = (val Version_fixture.of_graph (bf16_graph ())) in
  let src id = Option.get (Snapshot.edge A.snapshot (t id)) in
  let dst id = Option.get (Snapshot.edge B.snapshot (t id)) in
  show_create
    (Graph_map.create ~src:A.snapshot ~dst:B.snapshot
       ~values:
         [
           C.pair (src 0) (dst 0) C.Identical;
           C.pair (src 1) (dst 1) C.Identical;
         ]
       ~nodes:[] ~provenance:Provenance.empty);
  [%expect
    {| value map: t0 and t0 are claimed identical across different formats |}]

(* The shape half of the same check, which is why [permute_sequence] is not the
   fixture for rewrite_test's chain trim: its intermediate is a real
   rearrangement, and tying it to the input claims two shapes correspond. *)
let wide_graph () =
  Graph_builder.(
    build ~name:"g" ~outputs:(fun o -> [ o ])
    @@
    let* a = input ~shape:(s1c 4) () in
    relu a)
  |> Result.get_ok

let%expect_test "create: a cluster spanning two shapes is rejected" =
  let module A = (val Version_fixture.of_graph (one_input_graph ())) in
  let module B = (val Version_fixture.of_graph (wide_graph ())) in
  let src id = Option.get (Snapshot.edge A.snapshot (t id)) in
  let dst id = Option.get (Snapshot.edge B.snapshot (t id)) in
  show_create
    (Graph_map.create ~src:A.snapshot ~dst:B.snapshot
       ~values:
         [
           C.pair (src 0) (dst 0) C.Identical;
           C.pair (src 1) (dst 1) C.Identical;
         ]
       ~nodes:[] ~provenance:Provenance.empty);
  [%expect {| value map: t0 and t0 correspond but differ in shape |}]

let%expect_test "create: the same pair claimed Approximate is accepted" =
  let module A = (val Version_fixture.of_graph (one_input_graph ())) in
  let module B = (val Version_fixture.of_graph (bf16_graph ())) in
  let src id = Option.get (Snapshot.edge A.snapshot (t id)) in
  let dst id = Option.get (Snapshot.edge B.snapshot (t id)) in
  let a = approx [ bf16 ] in
  show_create
    (Graph_map.create ~src:A.snapshot ~dst:B.snapshot
       ~values:[ C.pair (src 0) (dst 0) a; C.pair (src 1) (dst 1) a ]
       ~nodes:[] ~provenance:Provenance.empty);
  [%expect {| ok |}]

(* [compose] takes no snapshots, so it cannot re-run the checks. Closure has to
   survive it anyway, which is why the property is stated here and why both
   consumers of a composed map re-check it. *)
let%expect_test "compose: two closed maps compose to a closed one" =
  let module A = (val Version_fixture.of_graph (relu_of Graph_builder.add ()))
  in
  let module B = (val Version_fixture.of_graph (relu_of Graph_builder.sub ()))
  in
  let module C_ = (val Version_fixture.of_graph (relu_of Graph_builder.sub ()))
  in
  let edge : type v. v Snapshot.t -> int -> v C.id =
   fun s id -> Option.get (Snapshot.edge s (t id))
  in
  let weakened : type a b. a Snapshot.t -> b Snapshot.t -> (a, b) Graph_map.t =
   fun src dst ->
    Graph_map.create ~src ~dst
      ~values:
        [
          C.pair (edge src 2) (edge dst 2) C.Unverifiable;
          C.pair (edge src 3) (edge dst 3) C.Unverifiable;
        ]
      ~nodes:[] ~provenance:Provenance.empty
    |> Result.get_ok
  in
  let ab = weakened A.snapshot B.snapshot in
  let bc = weakened B.snapshot C_.snapshot in
  show_create
    (Graph_map.check_claim_closure (Graph_map.compose ab bc) ~src:A.snapshot
       ~dst:C_.snapshot);
  [%expect {| ok |}]

let%expect_test "clusters_over adds the untouched identities" =
  let module A = (val Version_fixture.of_graph (two_input_graph ())) in
  let module B = (val Version_fixture.of_graph (two_input_graph ())) in
  let src id = Option.get (Snapshot.edge A.snapshot (t id)) in
  let dst id = Option.get (Snapshot.edge B.snapshot (t id)) in
  let map =
    Graph_map.create ~src:A.snapshot
      ~dst:B.snapshot
        (* t2 = add(t0,t1) is downstream of the weakened t0, so closure requires
         the map to speak about it — [Approximate] through a continuous op is
         [Unverifiable]. t1 is the one edge left implicit. *)
      ~values:
        [
          C.pair (src 0) (dst 0) (approx [ bf16 ]);
          C.pair (src 2) (dst 2) C.Unverifiable;
        ]
      ~nodes:[] ~provenance:Provenance.empty
    |> Result.get_ok
  in
  Fmt.pf Fmt.stdout "@[<v>explicit:@,%a@]@." C.pp (Graph_map.values map);
  Fmt.pf Fmt.stdout "@[<v>over the graphs:@,%a@]@."
    Fmt.(list ~sep:cut C.Cluster.pp)
    (Graph_map.clusters_over map ~src:A.snapshot ~dst:B.snapshot);
  [%expect
    {|
    explicit:
    {t0} -> {t0} approximate(bf16)
    {t2} -> {t2} unverifiable
    over the graphs:
    {t0} -> {t0} approximate(bf16)
    {t2} -> {t2} unverifiable
    {t1} -> {t1} identical |}]

(* ---- id supply ----------------------------------------------------------- *)

let%expect_test "id supply: seeded past every id, monotone, origin frozen" =
  let g = two_input_graph () in
  let ids = Id_supply.of_graph g in
  Fmt.pf Fmt.stdout "%a@." Id_supply.pp ids;
  let a, ids = Id_supply.tensor ids in
  let b, ids = Id_supply.tensor ids in
  let n, ids = Id_supply.node ids in
  Fmt.pf Fmt.stdout "allocated %a %a %a@." Tensor_id.pp a Tensor_id.pp b
    Node_id.pp n;
  Fmt.pf Fmt.stdout "%a@." Id_supply.pp ids;
  (* The origin watermark never moves, which is what lets packing compact
     post-origin ids into values disjoint from every origin id. *)
  Fmt.pf Fmt.stdout "origin: %a@." Id_supply.pp (Id_supply.origin ids);
  Fmt.pf Fmt.stdout "t0 post-origin: %b, %a post-origin: %b@."
    (Id_supply.is_post ids (t 0))
    Tensor_id.pp a (Id_supply.is_post ids a);
  [%expect
    {|
    ids next=(t3 n1 g1) origin=(t3 n1 g1)
    allocated t3 t4 n1
    ids next=(t5 n2 g1) origin=(t3 n1 g1)
    origin: ids next=(t3 n1 g1) origin=(t3 n1 g1)
    t0 post-origin: false, t3 post-origin: true |}]

(* ---- provenance ---------------------------------------------------------- *)

(* Found by typing the endpoints, not by reading: [compose] pulled an already-
   derived middle id's sources back through the A->B correspondence a SECOND
   time. Those are already source-side ids, and [backward] reads its argument as
   a middle id — silent while a source id happens not to occur as a middle
   destination, wrong the moment it does. Here t5 is both: the derivation's
   source, and the edge t7 was renamed onto.

   The guard against a regression is the SIGNATURE, not this case: writing the
   old form again collapses ['a] and ['b] and provenance.mli rejects it. What
   this pins is the answer, which no type can state. *)
let%expect_test "provenance: a derived middle id's sources are not pulled back"
    =
  let a = Provenance.of_list [ (set [ 5 ], I.edge 9) ] in
  let b = Provenance.of_list [ (set [ 9 ], I.edge 12) ] in
  let ab = of_ [ cluster ~src:[ 7 ] ~dst:[ 5 ] C.Identical ] in
  let composed = Provenance.compose a b ~values:(ab, C.identity) in
  Fmt.pf Fmt.stdout "@[<v>%a@]@." Provenance.pp composed;
  [%expect {|
    {t5} -> t9
    {t5} -> t12 |}]

let%expect_test "provenance is directional and survives a later rename" =
  (* w -Permute-> wp folds: w is deleted, wp derives from it. A later pass
     renames wp, and the derivation must follow. *)
  let p01 = Provenance.of_list [ (set [ 0 ], I.edge 5) ] in
  let m01 = of_ [ C.delete (I.edge 0) ] in
  let m12 = of_ [ cluster ~src:[ 5 ] ~dst:[ 6 ] C.Identical ] in
  let composed = Provenance.compose p01 Provenance.empty ~values:(m01, m12) in
  Fmt.pf Fmt.stdout "@[<v>%a@]@." Provenance.pp composed;
  Fmt.pf Fmt.stdout "sources_of t6 = %a@."
    (Fmt.braces (Fmt.list ~sep:Fmt.comma Tensor_id.pp))
    (raws (Provenance.sources_of composed (I.edge 6)));
  [%expect {|
    {t0} -> t6
    sources_of t6 = {t0} |}]
