(* [Me_session]: the capability table, what [validate] resolves, and the
   determinism claim (.ai/model_explorer_design.md).

   Two properties here are the reason the module exists and neither is visible
   in a rendering: that the capability key/payload table admits exactly the
   combinations it says, and that two loads under different runtime epochs
   produce byte-identical JSON. Both are asserted rather than displayed. *)

module S = Me_session
module ME = Model_explorer

let limits = Me_limits.Limits.untrusted

let pp_ok ppf r =
  Core.Pretty.core_result ~ok:(Fmt.any "ok") ~error:S.Session.pp_error ppf r

let check label session =
  Format.printf "%-30s %a@." label pp_ok (S.Session.validate ~limits session)

(* --- a minimal but complete session --- *)

let node ?(outputs = 1) ?(incoming = []) id =
  ME.GraphNode.create ~id ~label:id ~namespace:""
    ~incomingEdges:
      (List.map
         (fun (src, slot) ->
           ME.IncomingEdge.create ~sourceNodeId:src ~sourceNodeOutputId:slot
             ~targetNodeInputId:"0" ())
         incoming)
    ~outputsMetadata:
      (List.init outputs (fun i ->
           ME.MetadataItem.create ~id:(string_of_int i) ~attrs:[]))
    ()

let graph id nodes = ME.Graph.create ~id ~nodes ()
let collection graphs = ME.GraphCollection.create ~label:"mltorch:m" ~graphs ()

let capabilities ~stage_graph =
  List.map
    (fun k ->
      let status =
        match k with
        | S.Capability.Graph_stage _ ->
            S.Capability.Available (S.Capability.Graph stage_graph)
        | S.Capability.Feature S.Capability.Flow ->
            S.Capability.Available (S.Capability.Graph stage_graph)
        | S.Capability.Feature S.Capability.Verification ->
            S.Capability.Available
              (S.Capability.Verification_summary Pass.Outcome_counts.empty)
        | S.Capability.Feature S.Capability.Pass_audits ->
            S.Capability.Available
              (S.Capability.Pass_audit_status
                 {
                   S.Capability.Pass_audit_status.retained_reports = 3L;
                   omitted_reports = 0L;
                   omitted_counts = Pass.Outcome_counts.empty;
                 })
        | S.Capability.Feature S.Capability.Fold
        | S.Capability.Feature S.Capability.Expression_detail ->
            S.Capability.Available S.Capability.Present
        | S.Capability.Feature S.Capability.Loop_ir
        | S.Capability.Feature S.Capability.Codegen ->
            S.Capability.Unavailable
              { reason = S.Capability.Not_implemented; detail = None }
      in
      { S.Capability.key = k; status })
    S.Capability.all_keys

let base =
  let g =
    graph "g/native/000" [ node "n0"; node ~incoming:[ ("n0", "0") ] "n1" ]
  in
  {
    S.Session.schema_version = 1;
    producer = { S.Producer.tool = "mltorch"; session_schema = 1 };
    model =
      {
        S.Model_summary.name = "resnet18";
        source_kind = S.Model_summary.Pt2;
        source_bytes = 46_000_000L;
        source_sha256 = None;
        pt2_graph_count = 1;
        op_targets = 2;
      };
    graph_collections = [ collection [ g ] ];
    views =
      [
        {
          S.View.id = "v/native";
          label = "Initial Native";
          kind = S.View.Stage S.Capability.Initial_native;
          collection = "mltorch:m";
          graph = "g/native/000";
        };
      ];
    comparisons = [];
    node_data_sets = [];
    flow = None;
    capabilities = capabilities ~stage_graph:"g/native/000";
    diagnostics = [];
    default_view = "v/native";
  }

let%expect_test "a complete session validates" =
  check "base" base;
  [%expect {| base                           ok |}]

(* --- the capability table --- *)

let%expect_test "every key, and the payload each admits" =
  (* The table is the specification, so it is enumerated rather than sampled.
     Each key is offered every payload shape in turn and the admitted ones are
     printed — a table that admitted everything would be visible here as a row
     of all-yes. *)
  let payloads =
    [
      ("graph", S.Capability.Available (S.Capability.Graph "g"));
      ( "verification",
        S.Capability.Available
          (S.Capability.Verification_summary Pass.Outcome_counts.empty) );
      ( "audits",
        S.Capability.Available
          (S.Capability.Pass_audit_status
             {
               S.Capability.Pass_audit_status.retained_reports = 0L;
               omitted_reports = 0L;
               omitted_counts = Pass.Outcome_counts.empty;
             }) );
      ("present", S.Capability.Available S.Capability.Present);
      ( "unavailable",
        S.Capability.Unavailable
          { reason = S.Capability.Over_limit; detail = None } );
      ("not_requested", S.Capability.Not_requested);
    ]
  in
  Printf.printf "%-30s" "";
  List.iter (fun (n, _) -> Printf.printf "%-14s" n) payloads;
  print_newline ();
  List.iter
    (fun k ->
      Printf.printf "%-30s" (S.Capability.key_name k);
      List.iter
        (fun (_, status) ->
          Printf.printf "%-14s"
            (if S.Capability.compatible { S.Capability.key = k; status } then
               "yes"
             else "-"))
        payloads;
      print_newline ())
    S.Capability.all_keys;
  [%expect
    {|
                                  graph         verification  audits        present       unavailable   not_requested
    stage:source                  yes           -             -             -             yes           yes
    stage:initial_native          yes           -             -             -             yes           yes
    stage:canonical               yes           -             -             -             yes           yes
    stage:native4d                yes           -             -             -             yes           yes
    stage:stage_program           yes           -             -             -             yes           yes
    stage:kernel                  yes           -             -             -             yes           yes
    stage:fusion                  yes           -             -             -             yes           yes
    feature:flow                  yes           -             -             -             yes           yes
    feature:verification          -             yes           -             -             yes           yes
    feature:pass_audits           -             -             yes           -             yes           yes
    feature:fold                  -             -             -             yes           yes           yes
    feature:expression_detail     -             -             -             yes           yes           yes
    feature:loop_ir               -             -             -             -             -             -
    feature:codegen               -             -             -             -             -             - |}]

let%expect_test "loop_ir and codegen are not merely unrequested" =
  (* Nothing to request: they are permanently unimplemented, so
     [Not_requested] would read as "you did not ask" for a thing that cannot be
     asked for. The row above shows both refusing it; this states why. *)
  let with_status key status =
    {
      base with
      S.Session.capabilities =
        List.map
          (fun (c : S.Capability.t) ->
            if c.S.Capability.key = key then { c with S.Capability.status }
            else c)
          base.S.Session.capabilities;
    }
  in
  check "codegen not_requested"
    (with_status (S.Capability.Feature S.Capability.Codegen)
       S.Capability.Not_requested);
  check "codegen over_limit"
    (with_status (S.Capability.Feature S.Capability.Codegen)
       (S.Capability.Unavailable
          { reason = S.Capability.Over_limit; detail = None }));
  check "fold not_requested"
    (with_status (S.Capability.Feature S.Capability.Fold)
       S.Capability.Not_requested);
  [%expect
    {|
    codegen not_requested          capability feature:codegen carries a payload its key does not admit
    codegen over_limit             capability feature:codegen carries a payload its key does not admit
    fold not_requested             ok |}]

let%expect_test "absence is a producer defect, not a silence" =
  let missing =
    { base with S.Session.capabilities = List.tl base.S.Session.capabilities }
  in
  check "one key dropped" missing;
  let duplicated =
    {
      base with
      S.Session.capabilities =
        base.S.Session.capabilities @ [ List.hd base.S.Session.capabilities ];
    }
  in
  check "one key twice" duplicated;
  [%expect
    {|
    one key dropped                missing capability stage:source
    one key twice                  duplicate capability stage:source |}]

(* --- what validate resolves --- *)

let%expect_test "graph ids are unique and every reference resolves" =
  let dup =
    {
      base with
      S.Session.graph_collections =
        [ collection [ graph "g/native/000" []; graph "g/native/000" [] ] ];
    }
  in
  check "duplicate graph" dup;
  let dangling_view =
    {
      base with
      S.Session.views =
        List.map
          (fun (v : S.View.t) -> { v with S.View.graph = "g/nope" })
          base.S.Session.views;
    }
  in
  check "view names no graph" dangling_view;
  check "default_view names no view"
    { base with S.Session.default_view = "v/nope" };
  [%expect
    {|
    duplicate graph                duplicate graph id g/native/000
    view names no graph            unknown graph g/nope
    default_view names no view     unknown view v/nope |}]

let%expect_test "an edge must name a node AND one of its output slots" =
  (* Resolving the node is not enough: an edge pointing at output 3 of a
     single-output node renders as a connection carrying nothing. *)
  let bad_node =
    {
      base with
      S.Session.graph_collections =
        [
          collection
            [ graph "g/native/000" [ node ~incoming:[ ("n9", "0") ] "n1" ] ];
        ];
    }
  in
  check "edge from unknown node" bad_node;
  let bad_slot =
    {
      base with
      S.Session.graph_collections =
        [
          collection
            [
              graph "g/native/000"
                [ node "n0"; node ~incoming:[ ("n0", "3") ] "n1" ];
            ];
        ];
    }
  in
  check "edge from absent slot" bad_slot;
  [%expect
    {|
    edge from unknown node         unknown node n9 in graph g/native/000
    edge from absent slot          edge in graph g/native/000 names no output slot of node n0 |}]

let%expect_test "node data is graph-addressed and node-keyed" =
  let with_data graph results =
    {
      base with
      S.Session.node_data_sets =
        [ { S.Node_data_set.name = "cost"; graph; results } ];
    }
  in
  let v x = { S.Node_data_set.value = x; label = None } in
  check "resolves" (with_data "g/native/000" [ ("n0", v 1.) ]);
  check "unknown graph" (with_data "g/nope" [ ("n0", v 1.) ]);
  check "unknown node" (with_data "g/native/000" [ ("n9", v 1.) ]);
  [%expect
    {|
    resolves                       ok
    unknown graph                  unknown graph g/nope
    unknown node                   unknown node n9 in graph g/native/000 |}]

(* --- the mapping rule, and the scope that makes it right --- *)

let two_pane_session =
  let left = graph "g/native/000" [ node "n0"; node "n1" ] in
  let right4d = graph "g/native4d/000" [ node "m0" ] in
  let rightsym = graph "g/symbolic/000" [ node "k0" ] in
  let pane collection graph = { S.Pane_state.collection; graph } in
  let comparison id right entries =
    {
      S.Comparison.id;
      label = id;
      left = pane "mltorch:m" "g/native/000";
      right = pane "mltorch:m" right;
      sync = { S.Sync_navigation.entries; show_diff_highlights = true };
      overlays_left = [];
      overlays_right = [];
    }
  in
  {
    base with
    S.Session.graph_collections = [ collection [ left; right4d; rightsym ] ];
    comparisons =
      [
        comparison "c/4d" "g/native4d/000"
          [ { S.Mapping_entry.left = [ "n0"; "n1" ]; right = [ "m0" ] } ];
        comparison "c/sym" "g/symbolic/000"
          [ { S.Mapping_entry.left = [ "n0"; "n1" ]; right = [ "k0" ] } ];
      ];
  }

let%expect_test "one node may map in two comparisons, but not twice in one" =
  (* The scope is what makes this right. Canonical Native is the LEFT pane of
     both comparisons hanging off that state, so its nodes legitimately appear
     on the left of two mapping sets — a global check would reject the very
     branch the flow spine exists to show. *)
  check "same nodes, two comparisons" two_pane_session;
  let twice_in_one =
    {
      two_pane_session with
      S.Session.comparisons =
        List.map
          (fun (c : S.Comparison.t) ->
            if String.equal c.S.Comparison.id "c/4d" then
              {
                c with
                S.Comparison.sync =
                  {
                    c.S.Comparison.sync with
                    S.Sync_navigation.entries =
                      c.S.Comparison.sync.S.Sync_navigation.entries
                      @ [ { S.Mapping_entry.left = [ "n0" ]; right = [] } ];
                  };
              }
            else c)
          two_pane_session.S.Session.comparisons;
    }
  in
  check "same node twice in one" twice_in_one;
  [%expect
    {|
    same nodes, two comparisons    ok
    same node twice in one         node n0 appears in two mapping entries of comparison c/4d |}]

(* --- the flow's half that only this scope can check --- *)

let with_flow session comparison_of_4d =
  let st id graph layer label produced_by =
    { Me_flow.State.id; graph; layer; label; produced_by }
  in
  {
    session with
    S.Session.flow =
      Some
        {
          Me_flow.states =
            [
              st "s/pt2/000" "g/pt2/000" Me_ids.Layer.Pt2 "src" None;
              st "s/native/000" "g/native/000" Me_ids.Layer.Native "native"
                (Some "t/native/000");
              st "s/native4d/000" "g/native4d/000" Me_ids.Layer.Native4d "4d"
                (Some "t/native4d/000");
            ];
          transitions =
            [
              {
                Me_flow.Transition.id = "t/native/000";
                before = "s/pt2/000";
                after = "s/native/000";
                kind = Me_flow.Transition.Import;
                comparison = None;
              };
              {
                Me_flow.Transition.id = "t/native4d/000";
                before = "s/native/000";
                after = "s/native4d/000";
                kind = Me_flow.Transition.Cross_dialect;
                comparison = comparison_of_4d;
              };
            ];
          graph = "g/flow";
        };
  }

let%expect_test "a transition's comparison must be over its own two graphs" =
  (* Resolving is not enough. A comparison naming two unrelated graphs resolves
     perfectly and then shows the wrong diff, which is a rendering nobody can
     tell is wrong. *)
  check "panes match endpoints" (with_flow two_pane_session (Some "c/4d"));
  check "panes name other graphs" (with_flow two_pane_session (Some "c/sym"));
  check "comparison does not exist" (with_flow two_pane_session (Some "c/nope"));
  check "no comparison" (with_flow two_pane_session None);
  [%expect
    {|
    panes match endpoints          ok
    panes name other graphs        transition t/native4d/000 names a comparison over other graphs
    comparison does not exist      unknown comparison c/nope
    no comparison                  ok |}]

(* --- determinism, and the aggregates --- *)

let encode s =
  match Jsont_bytesrw.encode_string S.Session.jsont s with
  | Ok j -> j
  | Error e -> failwith ("encode: " ^ e)

let%expect_test "the epoch is runtime state and never reaches the document" =
  (* Two loads of the same bytes under different runtime epochs must produce
     byte-identical session JSON. That is why the epoch lives in [Runtime] and
     not in [Session]: a field cannot be forgotten out of a document it was
     never in. *)
  let a =
    {
      S.Runtime.epoch = "11111111-1111-1111-1111-111111111111";
      limits;
      session = base;
    }
  in
  let b =
    {
      S.Runtime.epoch = "22222222-2222-2222-2222-222222222222";
      limits;
      session = base;
    }
  in
  Printf.printf "identical: %b\n"
    (String.equal (encode a.S.Runtime.session) (encode b.S.Runtime.session));
  [%expect {| identical: true |}]

let%expect_test "the document, in full" =
  print_endline (encode base);
  [%expect
    {| {"schemaVersion":1,"producer":{"tool":"mltorch","sessionSchema":1},"model":{"name":"resnet18","sourceKind":"pt2","sourceBytes":"46000000","pt2GraphCount":1,"opTargets":2},"graphCollections":[{"label":"mltorch:m","graphs":[{"id":"g/native/000","nodes":[{"id":"n0","label":"n0","namespace":"","incomingEdges":[],"outputsMetadata":[{"id":"0","attrs":[]}]},{"id":"n1","label":"n1","namespace":"","incomingEdges":[{"sourceNodeId":"n0","sourceNodeOutputId":"0","targetNodeInputId":"0"}],"outputsMetadata":[{"id":"0","attrs":[]}]}]}]}],"views":[{"id":"v/native","label":"Initial Native","kind":"stage:initial_native","collection":"mltorch:m","graph":"g/native/000"}],"comparisons":[],"nodeDataSets":[],"capabilities":[{"key":"stage:source","status":{"state":"available","payload":{"kind":"graph","graph":"g/native/000"}}},{"key":"stage:initial_native","status":{"state":"available","payload":{"kind":"graph","graph":"g/native/000"}}},{"key":"stage:canonical","status":{"state":"available","payload":{"kind":"graph","graph":"g/native/000"}}},{"key":"stage:native4d","status":{"state":"available","payload":{"kind":"graph","graph":"g/native/000"}}},{"key":"stage:stage_program","status":{"state":"available","payload":{"kind":"graph","graph":"g/native/000"}}},{"key":"stage:kernel","status":{"state":"available","payload":{"kind":"graph","graph":"g/native/000"}}},{"key":"stage:fusion","status":{"state":"available","payload":{"kind":"graph","graph":"g/native/000"}}},{"key":"feature:flow","status":{"state":"available","payload":{"kind":"graph","graph":"g/native/000"}}},{"key":"feature:verification","status":{"state":"available","payload":{"kind":"verification_summary","verificationSummary":[]}}},{"key":"feature:pass_audits","status":{"state":"available","payload":{"kind":"pass_audit_status","passAuditStatus":{"retainedReports":"3","omittedReports":"0","omittedCounts":[]}}}},{"key":"feature:fold","status":{"state":"available","payload":{"kind":"present"}}},{"key":"feature:expression_detail","status":{"state":"available","payload":{"kind":"present"}}},{"key":"feature:loop_ir","status":{"state":"unavailable","reason":"not_implemented"}},{"key":"feature:codegen","status":{"state":"unavailable","reason":"not_implemented"}}],"diagnostics":[],"defaultView":"v/native"} |}]

let%expect_test "aggregates are checked before the walks they bound" =
  let tight =
    Core.or_raise Me_limits.pp_error
      (Me_limits.Limits.create ~max_graphs:1 limits)
  in
  Format.printf "graphs %a@." pp_ok
    (S.Session.validate ~limits:tight
       {
         base with
         S.Session.graph_collections =
           [ collection [ graph "a" []; graph "b" [] ] ];
       });
  (* [max_views:0] is not a way to write this: [Limits.create] refuses a
     non-positive ceiling, so the fixture would raise rather than exercise the
     check. A second view against a ceiling of one is the same test and is
     constructible. *)
  let tight_v =
    Core.or_raise Me_limits.pp_error
      (Me_limits.Limits.create ~max_views:1 limits)
  in
  Format.printf "views  %a@." pp_ok
    (S.Session.validate ~limits:tight_v
       {
         base with
         S.Session.views =
           base.S.Session.views
           @ [
               {
                 S.View.id = "v/flow";
                 label = "Flow";
                 kind = S.View.Flow;
                 collection = "mltorch:m";
                 graph = "g/native/000";
               };
             ];
       });
  [%expect
    {|
    graphs session graphs = 2 is over the ceiling
    views  session views = 2 is over the ceiling |}]
