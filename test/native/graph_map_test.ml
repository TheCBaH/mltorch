(* Exercises the transformation mapping: the claim lattice, cluster
   normalisation, the algebraic laws [compose]/[invert] must obey, endpoint
   validation, and the two cases the design turns on — trimming's many-to-one
   cluster and packing's reuse of a dead id's numeric value.
   See .ai/native_transform_design.md §3. *)

open Graph_ir
module C = Correspondence

let t n = Tensor_id.of_int n
let n_ n = Node_id.of_int n
let set ids = Tensor_id.Set.of_list (List.map t ids)
let cluster ~src ~dst label = { C.Cluster.src = set src; dst = set dst; label }
let of_ l = C.of_clusters l
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
    (Tensor_id.Set.elements (C.forward m (t 1)));
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
  let create = of_ [ C.create (t 11) ] in
  let delete = of_ [ C.delete (t 11) ] in
  show C.pp (C.compose create delete);
  [%expect {| identity |}]

(* Packing may reclaim the numeric value of a DEAD post-origin id. Composition
   must not identity-extend the dead id through the intervening maps and fuse it
   with the cluster that reuses its value — and the guard that prevents it is a
   side condition on extension, so associativity has to be demonstrated, not
   assumed. See .ai/native_transform_design.md §3, §9. *)
let%expect_test "delete, create, repack: the dead id never fuses with its reuse"
    =
  let m12 = of_ [ C.delete (t 11) ] in
  let m23 = of_ [ C.create (t 12) ] in
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
        C.create (t 5);
        C.delete (t 3);
        cluster ~src:[ 1 ] ~dst:[ 2 ] C.Identical;
      ]
  in
  let show name s =
    Fmt.pf Fmt.stdout "%s = %a@." name
      (Fmt.braces (Fmt.list ~sep:Fmt.comma Tensor_id.pp))
      (Tensor_id.Set.elements s)
  in
  show "created" (C.created m);
  show "deleted" (C.deleted m);
  [%expect {|
    created = {t5}
    deleted = {t3} |}]

(* ---- node clusters ------------------------------------------------------- *)

let%expect_test "node map: fusion, creation, deletion and reverse lookup" =
  let m =
    Node_map.of_clusters
      [
        Node_map.fused ~from:[ n_ 1; n_ 2 ] (n_ 7);
        Node_map.fused ~from:[] (n_ 8);
        Node_map.delete (n_ 3);
      ]
  in
  Fmt.pf Fmt.stdout "@[<v>%a@]@." Node_map.pp m;
  Fmt.pf Fmt.stdout "backward n7 = %a@."
    (Fmt.braces (Fmt.list ~sep:Fmt.comma Node_id.pp))
    (Node_id.Set.elements (Node_map.backward m (n_ 7)));
  [%expect
    {|
    {n1, n2} -> {n7}
    {n3} -> {}
    {} -> {n8}
    backward n7 = {n1, n2} |}]

(* ---- validation against a pair of graphs --------------------------------- *)

(* Phantoms cannot tie a map to two PARTICULAR graphs — [of_clusters] is
   polymorphic in them — so a well-typed map can name ids that exist in neither
   endpoint, and [identity] type-checks between unrelated graphs. Anything
   consuming a map from outside the rewrite path must call [validate]. *)

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

let check map ~src ~dst =
  match Graph_map.validate map ~src ~dst with
  | Ok () -> Fmt.pf Fmt.stdout "ok@."
  | Error e -> Fmt.pf Fmt.stdout "%a@." Graph_map.pp_error e.Core.Error.kind

let value_map clusters = { Graph_map.identity with values = of_ clusters }

let%expect_test "validate: a map of the graph onto itself passes" =
  let g = two_input_graph () in
  check Graph_map.identity ~src:g ~dst:g;
  [%expect {| ok |}]

let%expect_test "validate: an endpoint absent from its own side is rejected" =
  let g = two_input_graph () in
  check (value_map [ cluster ~src:[ 0 ] ~dst:[ 99 ] C.Identical ]) ~src:g ~dst:g;
  [%expect {| value map: dst t99 is not in the destination |}]

let%expect_test "validate: identity between differing graphs is rejected" =
  (* Both graphs exist and are well formed; nothing about the map is
     type-incorrect. Only implicit-identity coverage catches it. *)
  let src = two_input_graph () and dst = one_input_graph () in
  check Graph_map.identity ~src ~dst;
  [%expect
    {| value map: t2 is implicitly identity but absent from the destination |}]

let%expect_test "validate: creating an id the source already has is rejected" =
  let g = two_input_graph () in
  check (value_map [ C.create (t 0) ]) ~src:g ~dst:g;
  [%expect {| value map: t0 is mapped to but unmentioned as a source |}]

let%expect_test "clusters_over adds the untouched identities" =
  let g = two_input_graph () in
  let map = value_map [ cluster ~src:[ 0 ] ~dst:[ 0 ] (approx [ bf16 ]) ] in
  Fmt.pf Fmt.stdout "@[<v>explicit:@,%a@]@." C.pp map.Graph_map.values;
  Fmt.pf Fmt.stdout "@[<v>over the graphs:@,%a@]@."
    Fmt.(list ~sep:cut C.Cluster.pp)
    (Graph_map.clusters_over map ~src:g ~dst:g);
  [%expect
    {|
    explicit:
    {t0} -> {t0} approximate(bf16)
    over the graphs:
    {t0} -> {t0} approximate(bf16)
    {t1} -> {t1} identical
    {t2} -> {t2} identical |}]

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

let%expect_test "provenance is directional and survives a later rename" =
  (* w -Permute-> wp folds: w is deleted, wp derives from it. A later pass
     renames wp, and the derivation must follow. *)
  let p01 = Provenance.of_list [ ([ t 0 ], t 5) ] in
  let m01 = of_ [ C.delete (t 0) ] in
  let m12 = of_ [ cluster ~src:[ 5 ] ~dst:[ 6 ] C.Identical ] in
  let composed = Provenance.compose p01 Provenance.empty ~values:(m01, m12) in
  Fmt.pf Fmt.stdout "@[<v>%a@]@." Provenance.pp composed;
  Fmt.pf Fmt.stdout "sources_of t6 = %a@."
    (Fmt.braces (Fmt.list ~sep:Fmt.comma Tensor_id.pp))
    (Tensor_id.Set.elements (Provenance.sources_of composed (t 6)));
  [%expect {|
    {t0} -> t6
    sources_of t6 = {t0} |}]
