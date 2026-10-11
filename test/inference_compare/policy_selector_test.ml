open Inference_compare_schema
module Plan = Inference_compare_plan
module Pol = Inference_compare_policy
module P = Plan_fixtures
module C = Chain_fixture
module TS = Tensor_selector

let structural = C.structural C.attention

let ctx ?(limits = P.limits) s : Pol.Selector_eval.ctx =
  {
    left = Plan.Structural.left s;
    right = Plan.Structural.right s;
    partition = Plan.Structural.partition s;
    limits;
  }

let sites s =
  match Pol.Comparison_sites.build s with
  | Ok l -> l
  | Error e ->
      Format.kasprintf failwith "sites: %a" Pol.Policy_error.pp
        (Err.Error.kind e)

let np ?node ?op ?(dialect = "aten") () : TS.Node_pattern.t =
  { node; operator = Option.map (P.operator dialect) op }

let in_port i = Input_port.of_index i
let out_port i = Output_port.of_index i

let prod ?(scope = TS.Endpoint_graph) ?port p : TS.t =
  Produced_by
    { scope; producer = { node = p; port = Option.bind port out_port } }

let cons ?(scope = TS.Endpoint_graph) ?port p : TS.t =
  Consumed_by
    { scope; consumer = { node = p; port = Option.bind port in_port } }

let on ?(q = Metric_selector.Any_anchor) side t : Metric_selector.t =
  On_side { side; anchors = q; tensor = t }

let rule_id = P.text (Id.Metric_rule.of_string "r")

let select ?(s = structural) name (sel : Metric_selector.t) =
  let ctx = ctx s in
  let all = sites s in
  match Pol.Selector_eval.check ctx all ~rule:rule_id sel with
  | Error e ->
      Format.printf "%s: REJECTED %a@." name Pol.Policy_error.pp
        (Err.Error.kind e)
  | Ok () ->
      let hit =
        List.filter_map
          (fun (site : Comparison_site.t) ->
            Option.map
              (fun w -> (site, w))
              (Pol.Selector_eval.eval ctx site sel))
          all
      in
      Format.printf "%s: [%s]@." name
        (String.concat " "
           (List.map
              (fun ((s : Comparison_site.t), _) ->
                Id.Comparison.to_string s.comparison)
              hit))

let%expect_test "comparison sites of the attention fixture" =
  List.iter
    (fun (s : Comparison_site.t) ->
      Format.printf "%a unit=%a left=[%a] right=[%a] finals=[%a]@."
        Id.Comparison.pp s.comparison Id.Unit.pp s.unit
        Fmt.(
          list ~sep:comma
            (using
               (fun (r : Tensor_ref.t) -> Id.Tensor_local.to_string r.local_id)
               string))
        s.left_anchors
        Fmt.(
          list ~sep:comma
            (using
               (fun (r : Tensor_ref.t) -> Id.Tensor_local.to_string r.local_id)
               string))
        s.right_anchors
        Fmt.(list ~sep:comma Id.Output_occurrence.pp)
        s.final_occurrences)
    (sites structural);
  [%expect
    {|
    cmp_a unit=u_add left=[a] right=[a_r] finals=[]
    cmp_m unit=u_mm left=[m] right=[m_r] finals=[]
    cmp_mm unit=u_mul left=[mm] right=[mm_r] finals=[out1]
    cmp_o unit=u_att left=[o] right=[o_r] finals=[]
    cmp_s unit=u_sm left=[s] right=[s_r] finals=[]
    cmp_t unit=u_tail left=[t] right=[t_r] finals=[out0] |}]

let%expect_test "producer and use ports select the intended sites" =
  select "matmul result used at add input 0"
    (on Left
       (All_of [ prod (np ~op:"matmul" ()); cons ~port:0 (np ~op:"add" ()) ]));
  select "matmul result used at add input 1"
    (on Left
       (All_of [ prod (np ~op:"matmul" ()); cons ~port:1 (np ~op:"add" ()) ]));
  select "softmax result used as matmul weights"
    (on Left
       (All_of [ prod (np ~op:"softmax" ()); cons ~port:0 (np ~op:"matmul" ()) ]));
  select "softmax result used at matmul input 1"
    (on Left
       (All_of [ prod (np ~op:"softmax" ()); cons ~port:1 (np ~op:"matmul" ()) ]));
  select "any matmul result" (on Left (prod (np ~op:"matmul" ())));
  select "matmul, node identity and operator together"
    (on Left (prod (np ~node:(P.node C.left_key "n_att") ~op:"matmul" ())));
  select "node identity with the wrong operator"
    (on Left (prod (np ~node:(P.node C.left_key "n_att") ~op:"add" ())));
  select "producer output port 0" (on Left (prod ~port:0 (np ~op:"matmul" ())));
  [%expect
    {|
    matmul result used at add input 0: [cmp_m]
    matmul result used at add input 1: []
    softmax result used as matmul weights: [cmp_s]
    softmax result used at matmul input 1: []
    any matmul result: [cmp_m cmp_o]
    matmul, node identity and operator together: [cmp_o]
    node identity with the wrong operator: []
    producer output port 0: [cmp_m cmp_o] |}]

let%expect_test "same-use semantics versus independent predicates" =
  (* m is used by add (port 0) and mul (ports 0, 1). Within one All_of the
     consumer predicate pins a single use; two separate Consumed_by predicates
     may be satisfied by different uses. *)
  select "a single use that is add at port 1 (x, not m)"
    (on Left (cons ~port:1 (np ~op:"add" ())));
  select "a single use that is add at port 0"
    (on Left (cons ~port:0 (np ~op:"add" ())));
  select "independent predicates: an add use and a port-1 use"
    (on Left (All_of [ cons (np ~op:"add" ()); cons ~port:1 (np ()) ]));
  select "independent predicates: add and mul uses"
    (on Left (All_of [ cons (np ~op:"add" ()); cons (np ~op:"mul" ()) ]));
  select "a single use that is mul at port 1"
    (on Left (cons ~port:1 (np ~op:"mul" ())));
  [%expect
    {|
    a single use that is add at port 1 (x, not m): []
    a single use that is add at port 0: [cmp_m]
    independent predicates: an add use and a port-1 use: [cmp_m]
    independent predicates: add and mul uses: [cmp_m]
    a single use that is mul at port 1: [cmp_m] |}]

let%expect_test "fan-out counts operand occurrences" =
  let fan scope min max : TS.t =
    Fan_out { scope; count = { minimum = min; maximum = max } }
  in
  select "exactly three uses (repeated operand counted twice)"
    (on Left (fan Endpoint_graph 3L (Some 3L)));
  select "exactly two uses" (on Left (fan Endpoint_graph 2L (Some 2L)));
  select "at least two uses" (on Left (fan Endpoint_graph 2L None));
  select "zero uses (outputs with no consumer)"
    (on Left (fan Endpoint_graph 0L (Some 0L)));
  select "uses inside the producing cluster"
    (on Left (fan Within_cluster 1L None));
  [%expect
    {|
    exactly three uses (repeated operand counted twice): [cmp_m]
    exactly two uses: []
    at least two uses: [cmp_m]
    zero uses (outputs with no consumer): [cmp_mm cmp_t]
    uses inside the producing cluster: [] |}]

let%expect_test "connected paths" =
  let hop ?op () : TS.Hop.t =
    { node = np ?op (); input_port = None; output_port = None }
  in
  let sink ?op () : TS.Consumer.t = { node = np ?op (); port = None } in
  let flow ?(scope = TS.Endpoint_graph) through sink : TS.t =
    Flows_to { scope; through; sink }
  in
  select "o -> reshape -> sum"
    (on Left (flow [ hop ~op:"reshape" () ] (sink ~op:"sum" ())));
  select "direct consumer is reshape"
    (on Left (flow [] (sink ~op:"reshape" ())));
  select "wrong order"
    (on Left (flow [ hop ~op:"sum" () ] (sink ~op:"reshape" ())));
  select "any hop then sum" (on Left (flow [ hop () ] (sink ~op:"sum" ())));
  select "m -> mul -> (nothing)" (on Left (flow [ hop ~op:"mul" () ] (sink ())));
  select "path leaves the producing cluster"
    (on Left
       (flow ~scope:Within_cluster [ hop ~op:"reshape" () ] (sink ~op:"sum" ())));
  (* matmul output flowing through two ops reaches the sum: a -> softmax -> s ->
     matmul -> o -> reshape -> r -> sum *)
  select "a reaches sum through softmax, matmul, reshape"
    (on Left
       (flow
          [ hop ~op:"softmax" (); hop ~op:"matmul" (); hop ~op:"reshape" () ]
          (sink ~op:"sum" ())));
  (* the witness names the chain *)
  let all = sites structural in
  let site =
    Option.get (Pol.Comparison_sites.find all (P.comparison "cmp_o"))
  in
  (match
     Pol.Selector_eval.eval (ctx structural) site
       (on Left (flow [ hop ~op:"reshape" () ] (sink ~op:"sum" ())))
   with
  | Some [ w ] ->
      let path = List.hd w.paths in
      Format.printf "witness: anchor=%s hops=%d out=%s sink=%s@."
        (Id.Tensor_local.to_string w.anchor.local_id)
        (List.length path.through)
        (String.concat ","
           (List.map
              (fun (h : Topology_witness.Hop.t) ->
                Id.Tensor_local.to_string h.output.local_id)
              path.through))
        (Id.Node_local.to_string path.sink.node.local_id)
  | _ -> Format.printf "no witness@.");
  [%expect
    {|
    o -> reshape -> sum: [cmp_o]
    direct consumer is reshape: [cmp_o]
    wrong order: []
    any hop then sum: [cmp_o]
    m -> mul -> (nothing): []
    path leaves the producing cluster: []
    a reaches sum through softmax, matmul, reshape: [cmp_a]
    witness: anchor=o hops=1 out=r sink=n_sum |}]

let%expect_test "boundaries and Boolean combinations" =
  select "graph outputs" (on Left (Boundary Graph_output));
  select "cluster outputs" (on Left (Boundary Cluster_output));
  select "cluster inputs (an anchor is an output, not an input)"
    (on Left (Boundary Cluster_input));
  select "not a graph output" (on Left (Not (Boundary Graph_output)));
  select "graph output or matmul result"
    (Any_of
       [ on Left (Boundary Graph_output); on Left (prod (np ~op:"matmul" ())) ]);
  select "specific tensor" (on Left (Tensor (P.tensor C.left_key "s")));
  select "unit and not final output"
    (All_of
       [
         Unit (P.uid "u_tail");
         Not (Final_output (P.text (Id.Output_occurrence.of_string "out0")));
       ]);
  select "final output occurrence"
    (Final_output (P.text (Id.Output_occurrence.of_string "out1")));
  select "explicit comparison" (Comparison (P.comparison "cmp_a"));
  select "canonical value" (Canonical_value (P.ckey "s"));
  [%expect
    {|
    graph outputs: [cmp_mm cmp_t]
    cluster outputs: [cmp_a cmp_m cmp_mm cmp_o cmp_s cmp_t]
    cluster inputs (an anchor is an output, not an input): []
    not a graph output: [cmp_a cmp_m cmp_o cmp_s]
    graph output or matmul result: [cmp_m cmp_mm cmp_o cmp_t]
    specific tensor: [cmp_s]
    unit and not final output: []
    final output occurrence: [cmp_mm]
    explicit comparison: [cmp_a]
    canonical value: [cmp_s] |}]

let%expect_test "reference-side selectors work across operator vocabularies" =
  (* Left is aten (matmul, softmax); right is native (gemm, softmax_part). *)
  select "left structure selects" (on Left (prod (np ~op:"softmax" ())));
  select "right names the same node differently"
    (on Right (prod (np ~dialect:"native" ~op:"softmax_part" ())));
  select "right does not know the aten name"
    (on Right (prod (np ~op:"softmax" ())));
  select "both sides together"
    (All_of
       [
         on Left (prod (np ~op:"matmul" ()));
         on Right (prod (np ~dialect:"native" ~op:"gemm" ()));
       ]);
  [%expect
    {|
    left structure selects: [cmp_s]
    right names the same node differently: [cmp_s]
    right does not know the aten name: []
    both sides together: [cmp_m cmp_o] |}]

let%expect_test "any versus every anchor on a packed site" =
  (* A site with two left anchors, m (matmul result used at add port 0) and o
     (matmul result used by reshape): Any matches on m, Every needs both. *)
  let base =
    Option.get
      (Pol.Comparison_sites.find (sites structural) (P.comparison "cmp_m"))
  in
  let site =
    {
      base with
      left_anchors = [ P.tensor C.left_key "m"; P.tensor C.left_key "o" ];
    }
  in
  let try_ name q t =
    let r = Pol.Selector_eval.eval (ctx structural) site (on ~q Left t) in
    Format.printf "%s: %s@." name
      (match r with
      | None -> "no match"
      | Some ws -> Printf.sprintf "match, %d witness(es)" (List.length ws))
  in
  let both = prod (np ~op:"matmul" ()) in
  let used_by_add : TS.t = All_of [ both; cons ~port:0 (np ~op:"add" ()) ] in
  try_ "any anchor, matmul" Any_anchor both;
  try_ "every anchor, matmul" Every_anchor both;
  try_ "any anchor, matmul used by add" Any_anchor used_by_add;
  try_ "every anchor, matmul used by add" Every_anchor used_by_add;
  let none = { site with left_anchors = [] } in
  Format.printf "every anchor on a site with no anchors: %s@."
    (match
       Pol.Selector_eval.eval (ctx structural) none
         (on ~q:Every_anchor Left Any_tensor)
     with
    | None -> "no match"
    | Some _ -> "match");
  [%expect
    {|
    any anchor, matmul: match, 2 witness(es)
    every anchor, matmul: match, 2 witness(es)
    any anchor, matmul used by add: match, 1 witness(es)
    every anchor, matmul used by add: no match
    every anchor on a site with no anchors: no match |}]

let%expect_test "selectors are validated before use" =
  let oversize n =
    let rec build k : Metric_selector.t =
      if k = 0 then Any_site else Not (build (k - 1))
    in
    build n
  in
  select "unknown comparison" (Comparison (P.comparison "nope"));
  select "unknown canonical value" (Canonical_value (P.ckey "zz"));
  select "unknown unit" (Unit (P.uid "u_nope"));
  select "unknown output occurrence"
    (Final_output (P.text (Id.Output_occurrence.of_string "out9")));
  select "unknown tensor" (on Left (Tensor (P.tensor C.left_key "nope")));
  select "port beyond the node's arity"
    (on Left (cons ~port:5 (np ~node:(P.node C.left_key "n_add") ())));
  select "result port beyond arity"
    (on Left (prod ~port:3 (np ~node:(P.node C.left_key "n_mm") ())));
  select "node from another graph"
    (on Left (prod (np ~node:(P.node C.right_key "n_mm_r") ())));
  select "depth within limit" (oversize 10);
  select "depth over limit" (oversize 40);
  let long =
    on Left
      (Flows_to
         {
           scope = Endpoint_graph;
           through =
             List.init 20 (fun _ ->
                 { TS.Hop.node = np (); input_port = None; output_port = None });
           sink = { node = np (); port = None };
         })
  in
  select "path longer than the limit" long;
  [%expect
    {|
    unknown comparison: REJECTED invalid_selector rule=r
    unknown canonical value: REJECTED invalid_selector rule=r
    unknown unit: REJECTED invalid_selector rule=r
    unknown output occurrence: REJECTED invalid_selector rule=r
    unknown tensor: REJECTED invalid_selector rule=r
    port beyond the node's arity: REJECTED invalid_selector rule=r
    result port beyond arity: REJECTED invalid_selector rule=r
    node from another graph: []
    depth within limit: [cmp_a cmp_m cmp_mm cmp_o cmp_s cmp_t]
    depth over limit: REJECTED resource_exhausted rule=r limit=16 actual=41
    path longer than the limit: REJECTED resource_exhausted rule=r limit=8 actual=20 |}]
