(* Recovering PT2 provenance for a transformed graph.
   See .ai/native_transform_design.md §10.

   The sidecar is never rebuilt: it stays anchored to the graph the importer
   produced, and every question about a destination id is answered by walking the
   composed map backwards. So these tests run a real pipeline — batch-norm
   folding, constant folding, then packing — over a hand-built sidecar, and ask
   the lens about the far end. *)

open Graph_ir
module P = Pt2_native_graph

let t_ n = Tensor_id.of_int n
let n_ n = Node_id.of_int n
let s1c = Graph_fixtures.s1c

(* ---- a sidecar for the [chain] fixture ------------------------------------ *)

let vec values =
  Tensor.materialize (s1c 3) (fun c -> values.(Dim.to_int (Vec6.get c Axis.C)))

let ramp shape =
  Tensor.materialize shape (fun c ->
      float_of_int
        ((Dim.to_int (Vec6.get c Axis.N) * 7)
        + (Dim.to_int (Vec6.get c Axis.H) * 3)
        + Dim.to_int (Vec6.get c Axis.W))
      /. 4.)

let chain_constants =
  [
    (t_ 1, ramp (Graph_fixtures.weight_shape ~out_channels:3 ~in_channels:2));
    (t_ 2, vec [| 1.; 2.; 3. |]);
    (t_ 3, vec [| 2.; 0.5; 1.5 |]);
    (t_ 4, vec [| -1.; 0.5; 2. |]);
    (t_ 5, vec [| 0.5; 1.; 1.5 |]);
    (t_ 6, vec [| 4.; 1.; 0.25 |]);
  ]

(* t1 weight, t2 conv bias, t3 gamma, t4 beta, t5 mean, t6 var — the constants
   [Graph_fixtures.chain] declares, in that order. Only these carry an archive
   path: a captured target may only sit on a constant. *)
let captured =
  [
    (1, "p_conv_weight", "conv.weight");
    (2, "p_conv_bias", "conv.bias");
    (3, "p_bn_weight", "bn.weight");
    (4, "p_bn_bias", "bn.bias");
    (5, "b_bn_running_mean", "bn.running_mean");
    (6, "b_bn_running_var", "bn.running_var");
  ]

(* The graph input and the model output: PT2 names with no archive data behind
   them, so they exercise origin inheritance on its own. *)
let plain = [ (0, "x"); (9, "relu_default") ]

let map_of l =
  List.fold_left (fun m (k, v) -> Tensor_id.Map.add k v m) Tensor_id.Map.empty l

let node_origin index target : P.Node_origin.t =
  {
    graph_path = P.Graph_path.root;
    index;
    target;
    name = None;
    metadata = Schema_runtime.String_map.empty;
  }

let origin_of ssa =
  P.Source
    {
      P.Tensor_origin.graph_path = P.Graph_path.root;
      ssa_name = ssa;
      meta = None;
    }

let sidecar_for graph =
  P.make ~graph
    ~tensor_origins:
      (map_of
         (List.map (fun (id, ssa, _) -> (t_ id, origin_of ssa)) captured
         @ List.map (fun (id, ssa) -> (t_ id, origin_of ssa)) plain))
    ~node_origins:
      (List.fold_left
         (fun m (id, origin) -> Node_id.Map.add id origin m)
         Node_id.Map.empty
         [
           (n_ 0, [ node_origin 0 "torch.ops.aten.conv2d.default" ]);
           ( n_ 1,
             [
               node_origin 1
                 "torch.ops.aten._native_batch_norm_legit_no_training.default";
             ] );
           (n_ 2, [ node_origin 2 "torch.ops.aten.relu.default" ]);
         ])
    ~captured_targets:
      (map_of (List.map (fun (id, _, target) -> (t_ id, target)) captured))
  |> Err.or_raise ~pp_error:P.pp_error

(* A sidecar carrying nothing but its graph, for the case where the point is
   which graph it describes rather than what it says about it. *)
let bare_sidecar graph =
  P.make ~graph ~tensor_origins:Tensor_id.Map.empty
    ~node_origins:Node_id.Map.empty ~captured_targets:Tensor_id.Map.empty
  |> Err.or_raise ~pp_error:P.pp_error

(* ---- driving a pipeline and lensing the far end --------------------------- *)

let fail e = Format.printf "%a@." P.pp_lens_error (Err.Error.kind e)

(* The destination version is existential, so the probe has to be explicitly
   polymorphic to receive a lens at all. *)
type probe = { probe : 'b. 'b P.lens -> graph -> unit }

let over_chain ?sidecar passes { probe } =
  let g = Graph_fixtures.chain () in
  let sidecar = Option.value sidecar ~default:(sidecar_for g) in
  match Rewrite.origin ~constants:chain_constants g with
  | Error e -> Format.printf "origin: %a@." Rewrite.pp_error (Err.Error.kind e)
  | Ok (Rewrite.Origin s0) -> (
      match Pass.run_all s0 passes with
      | Error e -> Format.printf "%a@." Pass.pp_error (Err.Error.kind e)
      | Ok (Rewrite.Step (s1, m01)) -> (
          match Rewrite.pack s1 with
          | Error e -> Format.printf "%a@." Rewrite.pp_error (Err.Error.kind e)
          | Ok (Rewrite.Step (s2, m12)) -> (
              let map = Graph_map.compose m01 m12 in
              match P.lens sidecar ~src:s0 map ~dst:s2 with
              | Error e -> fail e
              | Ok lens -> probe lens (Rewrite.graph s2))))

let folded = [ Fold_batch_norm.pass; Pass.fixpoint Fold_const.pass ]

let report lens id =
  let ids fmt l = Fmt.brackets (Fmt.list ~sep:Fmt.comma Tensor_id.pp) fmt l in
  Format.printf "@[<v 2>%a:@," Tensor_id.pp id;
  (match P.tensor_origins lens id with
  | Error e -> fail e
  | Ok origins ->
      Format.printf "origins: %a@,"
        (Fmt.brackets
           (Fmt.list ~sep:Fmt.comma (fun fmt (o : P.Tensor_origin.t) ->
                Fmt.string fmt o.ssa_name)))
        origins);
  (match P.captured_target lens id with
  | Error e -> fail e
  | Ok target ->
      Format.printf "captured: %a@,"
        (Fmt.option ~none:(Fmt.any "none") Fmt.string)
        target);
  Format.printf "provenance: %a@]@." ids (P.provenance_sources lens id)

(* ---- the whole point ------------------------------------------------------ *)

let%expect_test
    "lens: a folded constant has no PT2 identity and no archive path" =
  (* t10 is the weight the batch-norm fold computed and constant folding
     materialised. It is a CREATION, so it has no PT2 name — and crucially no
     captured target, since the archive holds the pre-fold weight and those bytes
     are not a valid payload for this edge. The derivation is still available,
     separately, through provenance. *)
  over_chain folded { probe = (fun lens _ -> report lens (t_ 10)) };
  [%expect
    {|
    t10:
      origins: []
      captured: none
      provenance: [t1, t3, t6] |}]

let%expect_test "lens: an untouched edge keeps its origin and its archive path"
    =
  (* t0 survives the whole pipeline unmentioned by any cluster, so it resolves
     to itself. t9 survives too but only as [Equivalent], the fold having changed
     what feeds it — its PT2 name is still recoverable, since a name is not a
     claim about bytes, while its archive path would not be. *)
  over_chain folded
    {
      probe =
        (fun lens _ ->
          report lens (t_ 0);
          report lens (t_ 9));
    };
  [%expect
    {|
    t0:
      origins: [x]
      captured: none
      provenance: []
    t9:
      origins: [relu_default]
      captured: none
      provenance: [] |}]

let%expect_test "lens: an untransformed graph resolves every constant" =
  (* No passes at all, so the lens is the identity case: every captured constant
     still answers with its own PT2 name and archive path. That is the baseline
     the transformed answers are a deviation from. *)
  over_chain []
    {
      probe =
        (fun lens _ ->
          List.iter (fun (id, _, _) -> report lens (t_ id)) captured);
    };
  [%expect
    {|
    t1:
      origins: [p_conv_weight]
      captured: conv.weight
      provenance: []
    t2:
      origins: [p_conv_bias]
      captured: conv.bias
      provenance: []
    t3:
      origins: [p_bn_weight]
      captured: bn.weight
      provenance: []
    t4:
      origins: [p_bn_bias]
      captured: bn.bias
      provenance: []
    t5:
      origins: [b_bn_running_mean]
      captured: bn.running_mean
      provenance: []
    t6:
      origins: [b_bn_running_var]
      captured: bn.running_var
      provenance: [] |}]

let%expect_test "lens: node origins combine through the node clusters" =
  (* The fold turned {n0 conv, n1 batch_norm} into one node, so the surviving
     conv reports BOTH ATen targets — which is exactly the one-to-many node
     mapping the sidecar was designed to allow. *)
  over_chain folded
    {
      probe =
        (fun lens g ->
          List.iter
            (fun (n : node) ->
              match P.node_origins lens n.Node.id with
              | Error e -> fail e
              | Ok origins ->
                  Format.printf "%a: %a@." Node_id.pp n.Node.id
                    (Fmt.brackets
                       (Fmt.list ~sep:Fmt.comma
                          (fun fmt (o : P.Node_origin.t) ->
                            Fmt.string fmt o.target)))
                    origins)
            g.Graph.nodes);
    };
  [%expect
    {|
    n3: [torch.ops.aten.conv2d.default,
         torch.ops.aten._native_batch_norm_legit_no_training.default]
    n2: [torch.ops.aten.relu.default] |}]

(* ---- what it refuses ------------------------------------------------------ *)

let%expect_test "lens: a sidecar for a different graph is rejected" =
  (* The check that ties the sidecar to the map's source. Without it the lens
     would answer confidently with another graph's names.

     The first case is an equal-but-not-identical graph, which is why the
     comparison is over canonical bytes: a structural [=] on a record holding a
     [Tensor_id.Map] can report two identical maps unequal. *)
  over_chain
    ~sidecar:(bare_sidecar (Graph_fixtures.chain ()))
    folded
    { probe = (fun _ _ -> Format.printf "accepted@.") };
  over_chain
    ~sidecar:(bare_sidecar (Graph_fixtures.const_arith ()))
    folded
    { probe = (fun _ _ -> Format.printf "accepted@.") };
  [%expect
    {|
    accepted
    the sidecar does not describe the map's source graph |}]

let%expect_test "lens: an id absent from the destination is rejected" =
  (* Sparse correspondence resolves anything unmentioned to itself, so without
     the destination membership check a bogus id would quietly "resolve" to the
     same-numbered source. t5 is a real ORIGIN id that the fold consumed. *)
  over_chain folded
    {
      probe =
        (fun lens _ ->
          match P.captured_target lens (t_ 5) with
          | Error e -> fail e
          | Ok target ->
              Format.printf "resolved: %a@."
                (Fmt.option ~none:(Fmt.any "none") Fmt.string)
                target);
    };
  [%expect {| t5 is not an edge of the destination graph |}]

let%expect_test "lens: a weaker claim never yields an archive path" =
  (* Hand-built, because nothing in the pipeline relates a CAPTURED constant by
     anything but [Identical] — and that is the point: if a precision pass ever
     does, the bytes in the archive are the old representation and handing them
     back would be data corruption rather than imprecision. *)
  let g = Graph_fixtures.chain () in
  let sidecar = sidecar_for g in
  match Rewrite.origin ~constants:chain_constants g with
  | Error e -> Format.printf "origin: %a@." Rewrite.pp_error (Err.Error.kind e)
  | Ok (Rewrite.Origin s0) ->
      (let snap = Rewrite.snapshot s0 in
       let e id = Option.get (Snapshot.edge snap (t_ id)) in
       (* Every node output carries the claim too. A weaker claim on t1 reaches
          everything computed from it, and [Graph_map.create] will not accept a
          map that says otherwise — which is the same rule that stops the lens
          handing back source bytes for a downstream edge. *)
       let downstream =
         List.concat_map
           (fun (n : Graph_ir.node) -> n.Graph_ir.Node.outputs)
           (Snapshot.graph snap).Graph_ir.Graph.nodes
       in
       let claim rel =
         Graph_map.create ~src:snap ~dst:snap
           ~values:
             (Correspondence.pair (e 1) (e 1) rel
             :: List.map
                  (fun id ->
                    let x = Option.get (Snapshot.edge snap id) in
                    Correspondence.pair x x rel)
                  downstream)
           ~nodes:[] ~provenance:Provenance.empty
         |> Result.get_ok
       in
       let show rel =
         match P.lens sidecar ~src:s0 (claim rel) ~dst:s0 with
         | Error e -> fail e
         | Ok lens -> (
             match P.captured_target lens (t_ 1) with
             | Error e -> fail e
             | Ok target ->
                 Format.printf "%a -> %a@." Correspondence.pp_relation rel
                   (Fmt.option ~none:(Fmt.any "none") Fmt.string)
                   target)
       in
       show Correspondence.Identical;
       show Correspondence.Equivalent;
       show Correspondence.Unverifiable;
       match () with () -> ());
      [%expect
        {|
    identical -> conv.weight
    equivalent -> none
    unverifiable -> none |}]

let%expect_test
    "lens: an id outside the destination graph is a distinct error, not absent \
     provenance" =
  (* t999/n999 aren't in the destination graph at all — a caller bug, not "no
     provenance recorded" (which is t10 above: present, just empty). The lens
     must say so via Unknown_destination_tensor/_node rather than silently
     answering as if the id were merely unrecorded. *)
  over_chain folded
    {
      probe =
        (fun lens _ ->
          report lens (t_ 999);
          match P.node_origins lens (n_ 999) with
          | Error e -> fail e
          | Ok _ -> print_string "unexpected Ok\n");
    };
  [%expect
    {|
    t999:
      t999 is not an edge of the destination graph
    t999 is not an edge of the destination graph
    provenance: []
    n999 is not a node of the destination graph |}]
