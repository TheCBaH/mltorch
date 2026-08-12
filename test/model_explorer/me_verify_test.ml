(* [Me_verify]: placing verification results in the rendered hierarchy
   (.ai/model_explorer_design.md).

   The rule that matters here is invisible in any report over a real model,
   because the two cases never sit side by side in one: a cluster inside one
   group, and a cluster spanning two. And the reason the exporter places
   clusters itself at all — [Map_verify.Group_path] appends a group's LABEL
   alone while the importer labels every group with the PT2 target, so sibling
   groups collapse into one path that cannot key the ID-qualified namespace. *)

module MV = Map_verify
module MS = Me_session

let limits = Me_limits.Limits.untrusted
let tid = Graph_ir.Tensor_id.of_int
let nid = Graph_ir.Node_id.of_int

let pp_err ppf r =
  Core.Pretty.err_result ~ok:(Fmt.any "ok") ~error:Me_verify.pp_error ppf r

(* --- placement --- *)

let cluster dst =
  {
    Correspondence.Cluster.Erased.src = Graph_ir.Tensor_id.Set.empty;
    dst =
      List.fold_left
        (fun s t -> Graph_ir.Tensor_id.Set.add (tid t) s)
        Graph_ir.Tensor_id.Set.empty dst;
    label = Correspondence.Identical;
  }

let entry
    ?(outcome =
      MV.Outcome.
        { coverage = MV.Coverage.exhaustive; verdict = MV.Verdict.Vacuous }) dst
    =
  { MV.Entry.cluster = cluster dst; group = []; outcome }

(* Two sibling groups whose LABELS are equal -- exactly what the importer
   produces, since it labels every group with the PT2 node's target. *)
let namespace_of id =
  match Graph_ir.Node_id.to_int id with
  | 0 | 1 -> "convolution#g1"
  | 2 -> "convolution#g2"
  | _ -> ""

let producer t =
  match Graph_ir.Tensor_id.to_int t with
  | 10 -> Some (nid 0)
  | 11 -> Some (nid 1)
  | 20 -> Some (nid 2)
  (* Everything else is a graph input or a constant: produced by no node, and
     so belonging to no group. *)
  | _ -> None

let place dst =
  Printf.printf "%-20s -> %S\n"
    ("[" ^ String.concat "," (List.map string_of_int dst) ^ "]")
    (Me_verify.placement ~namespace_of ~producer (entry dst))

let%expect_test "where a cluster sits" =
  (* Sibling groups whose labels are equal do NOT collapse: the id qualifier is
     what keeps [g1] and [g2] apart, and it is the whole reason this is
     recomputed rather than read off [Map_verify.Entry.group]. *)
  List.iter place [ [ 10 ]; [ 10; 11 ]; [ 10; 20 ]; [ 10; 99 ]; [ 99 ]; [] ];
  [%expect
    {|
    [10]                 -> "convolution#g1"
    [10,11]              -> "convolution#g1"
    [10,20]              -> ""
    [10,99]              -> "convolution#g1"
    [99]                 -> ""
    []                   -> "" |}]

let%expect_test "a producerless member does not drag a cluster to the root" =
  (* [10,99] above is the case: a fold's cluster names the folded weight, which
     is a graph constant and belongs to no group. Counting it as "root" would
     move every cluster that touches a weight out of its group -- which is most
     of them, on a real model. *)
  Printf.printf "with a constant member: %S\nwithout it:             %S\n"
    (Me_verify.placement ~namespace_of ~producer (entry [ 10; 99 ]))
    (Me_verify.placement ~namespace_of ~producer (entry [ 10 ]));
  [%expect
    {|
    with a constant member: "convolution#g1"
    without it:             "convolution#g1" |}]

(* --- the rollup and the node data, over a real little graph --- *)

let sig_ id =
  Tensor_sig.create ~id:(tid id) ~name:""
    ~shape:(Vec6.shape ~n:1 ~t:1 ~d:1 ~h:4 ~w:4 ~c:3)
    ~fmt:(Payload.Fmt Payload.F32) ()

let relu ~id ~x ~out =
  {
    Graph_ir.Node.id = nid id;
    op = Graph_ir.Relu { Pointwise.Relu.x = tid x };
    outputs = [ tid out ];
  }

(* n0 in group g1 "features", n1 in group g2 "classifier". *)
let graph () =
  {
    Graph_ir.Graph.nodes = [ relu ~id:0 ~x:0 ~out:2; relu ~id:1 ~x:2 ~out:3 ];
    root =
      {
        Graph_ir.Group.id = Graph_ir.Group_id.of_int 0;
        label = None;
        items =
          [
            Graph_ir.Group.Group
              {
                Graph_ir.Group.id = Graph_ir.Group_id.of_int 1;
                label = Some "features";
                items = [ Graph_ir.Group.Node (nid 0) ];
              };
            Graph_ir.Group.Group
              {
                Graph_ir.Group.id = Graph_ir.Group_id.of_int 2;
                label = Some "features";
                items = [ Graph_ir.Group.Node (nid 1) ];
              };
          ];
      };
    tensors =
      List.fold_left
        (fun m i -> Graph_ir.Tensor_id.Map.add (tid i) (sig_ i) m)
        Graph_ir.Tensor_id.Map.empty [ 0; 2; 3 ];
    inputs = [ tid 0 ];
    input_kinds =
      Graph_ir.Tensor_id.Map.add (tid 0) Graph_ir.Input.Input
        Graph_ir.Tensor_id.Map.empty;
    outputs = [ tid 3 ];
  }

let outcome verdict = MV.Outcome.{ coverage = MV.Coverage.exhaustive; verdict }

let report =
  {
    MV.Report.entries =
      [
        entry
          ~outcome:(outcome (MV.Verdict.Proved MV.Strength.Structural))
          [ 2 ];
        entry ~outcome:(outcome MV.Verdict.Vacuous) [ 3 ];
        (* No producer: a graph input, so this one lands at the root. *)
        entry ~outcome:(outcome MV.Verdict.Vacuous) [ 0 ];
      ];
  }

let%expect_test "the rollup keys groupNodeAttributes by namespace" =
  (* Two groups with the SAME label, which is what the importer emits; the
     rollup keeps them apart because it uses the id-qualified namespace the
     projection uses. A label-only key would report one group with two
     clusters. *)
  Format.printf "%a@."
    (Core.Pretty.err_result
       ~ok:
         (Fmt.list ~sep:(Fmt.any "@\n") (fun fmt (ns, attrs) ->
              Fmt.pf fmt "%S %a" ns
                (Fmt.list ~sep:(Fmt.any " ") (fun fmt (k, v) ->
                     Fmt.pf fmt "%s=%s" k v))
                attrs))
       ~error:Me_verify.pp_error)
    (Me_verify.by_namespace ~limits (graph ()) report);
  [%expect
    {|
    "" vacuous=1
    "features#g1" proved (structural)=1
    "features#g2" vacuous=1 |}]

let%expect_test "the node data is keyed by node id, not by namespace" =
  (* The two say different things and both are needed: the node datum is one
     node's weakest outcome, the group attribute is its subtree's tally. Only
     the node-keyed form can colour a node. *)
  Format.printf "%a@."
    (Core.Pretty.err_result
       ~ok:(fun fmt (d : MS.Node_data_set.t) ->
         Fmt.pf fmt "%s over %s@\n%a" d.MS.Node_data_set.name
           d.MS.Node_data_set.graph
           (Fmt.list ~sep:(Fmt.any "@\n")
              (fun fmt (id, (v : MS.Node_data_set.value)) ->
                Fmt.pf fmt "%s %g %a" id v.MS.Node_data_set.value
                  (Fmt.option ~none:(Fmt.any "-") Fmt.string)
                  v.MS.Node_data_set.label))
           d.MS.Node_data_set.results)
       ~error:Me_verify.pp_error)
    (Me_verify.node_data ~limits ~graph:"g/native/001" (graph ()) report);
  [%expect
    {|
    verification over g/native/001
    n0 1 proved (structural)
    n1 0 vacuous |}]

let%expect_test "a claimless output does not mask a real verdict" =
  (* One node, two outputs: t2 is proved and t3 is vacuous. The join keeps the
     PROVED one, because [Vacuous] ranks 0 -- below proved -- precisely so that
     a creation or a deletion, which claims nothing, cannot swallow a verdict
     that does. That is also why the rank is not a quality score. *)
  let two_outputs =
    let g = graph () in
    {
      g with
      Graph_ir.Graph.nodes =
        [
          {
            Graph_ir.Node.id = nid 0;
            op = Graph_ir.Relu { Pointwise.Relu.x = tid 0 };
            outputs = [ tid 2; tid 3 ];
          };
        ];
    }
  in
  Format.printf "%a@."
    (Core.Pretty.err_result
       ~ok:(fun fmt (d : MS.Node_data_set.t) ->
         Fmt.list ~sep:(Fmt.any "@\n")
           (fun fmt (id, (v : MS.Node_data_set.value)) ->
             Fmt.pf fmt "%s %a" id
               (Fmt.option ~none:(Fmt.any "-") Fmt.string)
               v.MS.Node_data_set.label)
           fmt d.MS.Node_data_set.results)
       ~error:Me_verify.pp_error)
    (Me_verify.node_data ~limits ~graph:"g" two_outputs report);
  [%expect {| n0 proved (structural) |}]

(* --- ceilings --- *)

let%expect_test "the two ceilings" =
  let tight_groups =
    Err.or_raise ~pp_error:Me_limits.pp_error
      (Me_limits.Limits.create ~max_groups_per_graph:2 limits)
  in
  Format.printf "groups %a@." pp_err
    (Me_verify.by_namespace ~limits:tight_groups (graph ()) report);
  let tight_data =
    Err.or_raise ~pp_error:Me_limits.pp_error
      (Me_limits.Limits.create ~max_node_data_results_per_graph:1 limits)
  in
  Format.printf "results %a@." pp_err
    (Me_verify.node_data ~limits:tight_data ~graph:"g" (graph ()) report);
  [%expect
    {|
    groups verification groupNodeAttributes = 3 is over the ceiling
    results verification nodeDataResults = 2 is over the ceiling |}]

(* --- the audit status --- *)

let%expect_test "an empty log and an unoverflowed one agree" =
  (* [Outcome_counts.empty] and "nothing overflowed" are the same statement, so
     the absent case needs no separate representation on the wire. *)
  let show (s : MS.Capability.Pass_audit_status.t) =
    Printf.printf "retained=%Ld omitted=%Ld counts=%d\n"
      s.MS.Capability.Pass_audit_status.retained_reports s.omitted_reports
      (List.length (Pass.Outcome_counts.bindings s.omitted_counts))
  in
  show (Me_verify.audit_status Pass.Audit_log.empty);
  [%expect {| retained=0 omitted=0 counts=0 |}]
