(* [Me_detail]: expression detail, and the merge (.ai/model_explorer_design.md).

   The rules here are all about what {!Me_detail.apply} must REFUSE, and none of
   them is visible in a merge that happens to succeed:

   - a delta whose graph and view do not carry the key's id — the case that
     exists because the payload deliberately has no key field of its own to
     compare against;
   - a key that names no value in its parent graph, which is a valid request
     about something absent rather than a malformed one;
   - a re-request REPLACING rather than accumulating, so aggregates cannot be
     inflated by asking twice;
   - two deltas that each pass alone where the second merge does not. *)

module MR = Me_request
module ME = Model_explorer
module L = Me_limits.Limits

let limits = L.untrusted

(* A real session, from the same tiny model the export suite uses: a hand-built
   one would be a session nobody produces, and the detail rules are about
   merging into what the exporter actually emits. *)
let model =
  Printf.sprintf
    {|{"graph_module":{"graph":{
        "inputs":[{"as_tensor":{"name":"x"}}],
        "outputs":[{"as_tensor":{"name":"y"}}],
        "nodes":[{"target":"torch.ops.aten.relu.default",
                  "inputs":[{"name":"self","arg":{"as_tensor":{"name":"x"}},"kind":1}],
                  "outputs":[{"as_tensor":{"name":"w"}}],
                  "metadata":{}},
                 {"target":"torch.ops.aten.relu.default",
                  "inputs":[{"name":"self","arg":{"as_tensor":{"name":"w"}},"kind":1}],
                  "outputs":[{"as_tensor":{"name":"y"}}],
                  "metadata":{}}],
        "tensor_values":{"x":%s,"w":%s,"y":%s},
        "sym_int_values":{},"sym_bool_values":{},"is_single_tensor_return":true},
      "signature":{"input_specs":[{"user_input":{"arg":{"as_tensor":{"name":"x"}}}}],
                   "output_specs":[{"user_output":{"arg":{"as_tensor":{"name":"y"}}}}]},
      "module_call_graph":[]},
      "opset_version":{"aten":15},"range_constraints":{},
      "schema_version":{"major":8,"minor":5}}|}
    {|{"dtype":7,"sizes":[{"as_int":1},{"as_int":4}],"requires_grad":false,"device":{"type":"cpu"},"strides":[{"as_int":4},{"as_int":1}],"storage_offset":{"as_int":0},"layout":7}|}
    {|{"dtype":7,"sizes":[{"as_int":1},{"as_int":4}],"requires_grad":false,"device":{"type":"cpu"},"strides":[{"as_int":4},{"as_int":1}],"storage_offset":{"as_int":0},"layout":7}|}
    {|{"dtype":7,"sizes":[{"as_int":1},{"as_int":4}],"requires_grad":false,"device":{"type":"cpu"},"strides":[{"as_int":4},{"as_int":1}],"storage_offset":{"as_int":0},"layout":7}|}

let session_of ~limits =
  Err.or_raise ~pp_error:Me_export.pp_error
    (Me_export.session ~limits
       ~options:
         {
           Me_export.Options.stages = Me_session.Capability.all_stages;
           fold = false;
           verify_symbolic = None;
           name = "tiny";
           source_bytes = Int64.of_int (String.length model);
           source_sha256 = None;
         }
       ~bytes:model)

let session = session_of ~limits
let kernel_id = Me_ids.graph Me_ids.Layer.Kernel 0

(* The value nodes the kernel graph actually offers -- read off the session, not
   guessed, so the fixture cannot drift from the projection. *)
let kernel_value_nodes s =
  List.concat_map
    (fun (c : ME.GraphCollection.t) ->
      List.concat_map
        (fun (g : ME.Graph.t) ->
          if String.equal g.ME.Graph.id kernel_id then
            List.filter_map
              (fun (n : ME.GraphNode.t) ->
                let id = n.ME.GraphNode.id in
                if String.length id > 1 && id.[0] = 'v' then
                  int_of_string_opt (String.sub id 1 (String.length id - 1))
                else None)
              g.ME.Graph.nodes
          else [])
        c.ME.GraphCollection.graphs)
    s.Me_session.Session.graph_collections

let%expect_test "the kernel graph offers value nodes to ask about" =
  Printf.printf "%s\n"
    (String.concat " " (List.map string_of_int (kernel_value_nodes session)));
  [%expect {| 1 2 |}]

let key ?(parent = kernel_id) v =
  Err.or_raise ~pp_error:MR.Request.pp_error
    (MR.Detail_key.create ~limits ~parent_graph:parent
       ~value:(Graph_ir.Tensor_id.of_int v))

let collection = "mltorch:tiny"

let delta ?graph ?view k =
  let id = MR.Detail_key.id k in
  {
    Me_detail.Delta.schema_version = 1;
    collection;
    graph =
      Option.value graph
        ~default:
          (ME.Graph.create ~id
             ~nodes:
               [
                 ME.GraphNode.create ~id:"e0" ~label:"const 1" ~namespace:""
                   ~incomingEdges:[] ~outputsMetadata:[] ();
               ]
             ());
    view =
      Option.value view
        ~default:
          {
            Me_session.View.id;
            label = "expression";
            kind = Me_session.View.Stage Me_session.Capability.Kernel;
            collection;
            graph = id;
          };
    node_data = [];
    diagnostics = [];
  }

let pp ppf r =
  Core.Pretty.err_result
    ~ok:(fun ppf (s : Me_session.Session.t) ->
      Fmt.pf ppf "graphs=%d views=%d"
        (List.length
           (List.concat_map
              (fun (c : ME.GraphCollection.t) -> c.ME.GraphCollection.graphs)
              s.Me_session.Session.graph_collections))
        (List.length s.Me_session.Session.views))
    ~error:Me_detail.pp_error ppf r

(* --- two details, then a replacement --- *)

let%expect_test "two details on two different value nodes" =
  let values = kernel_value_nodes session in
  let a = key (List.nth values 0) and b = key (List.nth values 1) in
  let after_a = Me_detail.apply ~key:a ~limits session (delta a) in
  Format.printf "one   %a@." pp after_a;
  (match after_a with
  | Error _ -> ()
  | Ok s ->
      Format.printf "two   %a@." pp (Me_detail.apply ~key:b ~limits s (delta b)));
  [%expect {|
    one   graphs=8 views=8
    two   graphs=9 views=9 |}]

let%expect_test "re-requesting one REPLACES it" =
  (* Aggregates are counted over what is installed, so an accumulating merge
     would let a caller inflate them by asking twice. The graph and the view
     carry the same id, so one predicate removes both. *)
  let a = key (List.hd (kernel_value_nodes session)) in
  let once =
    Err.or_raise ~pp_error:Me_detail.pp_error
      (Me_detail.apply ~key:a ~limits session (delta a))
  in
  Format.printf "once  %a@." pp (Ok once);
  Format.printf "twice %a@." pp (Me_detail.apply ~key:a ~limits once (delta a));
  [%expect {|
    once  graphs=8 views=8
    twice graphs=8 views=8 |}]

let%expect_test "the parent node gains subGraphIds, and only then" =
  (* A fresh session carries none: a link to a graph that is not installed is a
     dangling reference the session validator rejects, so the graph, the view
     and the link commit together or not at all. *)
  let a = key (List.hd (kernel_value_nodes session)) in
  let node =
    Me_ids.value_node
      (Graph_ir.Tensor_id.of_int (List.hd (kernel_value_nodes session)))
  in
  let show label (s : Me_session.Session.t) =
    List.iter
      (fun (c : ME.GraphCollection.t) ->
        List.iter
          (fun (g : ME.Graph.t) ->
            if String.equal g.ME.Graph.id kernel_id then
              List.iter
                (fun (n : ME.GraphNode.t) ->
                  if String.equal n.ME.GraphNode.id node then
                    Printf.printf "%-7s %s -> %s\n" label node
                      (Core.Pretty.to_string
                         (Core.Pretty.option_or ~none:"absent" (fun ppf l ->
                              Fmt.pf ppf "[%s]" (String.concat " " l)))
                         n.ME.GraphNode.subgraphIds))
                g.ME.Graph.nodes)
          c.ME.GraphCollection.graphs)
      s.Me_session.Session.graph_collections
  in
  show "before" session;
  show "after"
    (Err.or_raise ~pp_error:Me_detail.pp_error
       (Me_detail.apply ~key:a ~limits session (delta a)));
  [%expect
    {|
    before  v1 -> absent
    after   v1 -> [expr/g/kernel/000/t1/t1] |}]

(* --- what it refuses --- *)

let%expect_test "a delta whose ids are not the key's" =
  (* The payload carries NO key of its own, so this is the only check that can
     catch a shell announcing key A with a payload for key B -- and it works
     because the validated key arrives as an argument rather than in the
     document. *)
  let values = kernel_value_nodes session in
  let a = key (List.nth values 0) and b = key (List.nth values 1) in
  let mismatched =
    { (delta a) with Me_detail.Delta.graph = (delta b).Me_detail.Delta.graph }
  in
  Format.printf "%a@." pp (Me_detail.apply ~key:a ~limits session mismatched);
  [%expect {| the delta's graph and view do not carry the key's id |}]

let%expect_test "a key naming no value in that graph" =
  (* Well-formed, and about something absent -- which is a different fact from a
     malformed key and carries a different code on the wire. *)
  Format.printf "absent value %a@." pp
    (let k = key 9999 in
     Me_detail.apply ~key:k ~limits session (delta k));
  Format.printf "absent graph %a@." pp
    (let k =
       key ~parent:"g/native/000" (List.hd (kernel_value_nodes session))
     in
     Me_detail.apply ~key:k ~limits session (delta k));
  [%expect
    {|
    absent value the key names no value in that graph
    absent graph the key names no value in that graph |}]

let%expect_test "each passes alone, the SECOND merge does not" =
  (* The aggregate is over every installed detail, not over the delta in hand,
     which is the whole reason it is checked on the merged session. *)
  let tight =
    Err.or_raise ~pp_error:Me_limits.pp_error
      (L.create ~max_detail_graphs:1 limits)
  in
  let s = session_of ~limits:tight in
  let values = kernel_value_nodes s in
  let a = key (List.nth values 0) and b = key (List.nth values 1) in
  let first = Me_detail.apply ~key:a ~limits:tight s (delta a) in
  Format.printf "first  %a@." pp first;
  (match first with
  | Error _ -> ()
  | Ok s' ->
      Format.printf "second %a@." pp
        (Me_detail.apply ~key:b ~limits:tight s' (delta b)));
  [%expect
    {|
    first  graphs=8 views=8
    second detail detailGraphs = 2 is over the ceiling |}]

(* --- the expression graph --- *)

let%expect_test "an expression becomes one node per AST node" =
  let body =
    Expr.Value.add
      (Expr.Value.mul (Expr.Value.const 2.) (Expr.Value.const 3.))
      (Expr.Value.round_f32 (Expr.Value.const 4.))
  in
  let v =
    {
      Kernel.Value.id = Graph_ir.Tensor_id.of_int 7;
      sg =
        Tensor_sig.create
          ~id:(Graph_ir.Tensor_id.of_int 7)
          ~name:""
          ~shape:(Vec6.shape ~n:1 ~t:1 ~d:1 ~h:1 ~w:1 ~c:1)
          ~fmt:(Payload.Fmt Payload.F32) ();
      body;
      result = Kernel.Result_conversion.Round_f32;
    }
  in
  Format.printf "%a@."
    (Core.Pretty.err_result
       ~ok:(fun ppf (g : ME.Graph.t) ->
         Fmt.pf ppf "%s@\n%a" g.ME.Graph.id
           (Fmt.list ~sep:(Fmt.any "@\n") (fun ppf (n : ME.GraphNode.t) ->
                Fmt.pf ppf "  %-3s %-9s from [%s]" n.ME.GraphNode.id
                  n.ME.GraphNode.label
                  (String.concat " "
                     (List.map
                        (fun (e : ME.IncomingEdge.t) ->
                          e.ME.IncomingEdge.sourceNodeId)
                        (Option.value n.ME.GraphNode.incomingEdges ~default:[])))))
           g.ME.Graph.nodes)
       ~error:Me_detail.pp_error)
    (Me_detail.of_value ~limits ~key:(key 7) v);
  [%expect
    {|
    expr/g/kernel/000/t7/t7
      e0  +         from []
      e1  *         from [e0]
      e2  const 2   from [e1]
      e3  const 3   from [e1]
      e4  round_f32 from [e0]
      e5  const 4   from [e4] |}]

let%expect_test "the size ceiling is checked BEFORE the walk" =
  let tight =
    Err.or_raise ~pp_error:Me_limits.pp_error
      (L.create ~max_detail_nodes:2 limits)
  in
  let body = Expr.Value.add (Expr.Value.const 1.) (Expr.Value.const 2.) in
  let v =
    {
      Kernel.Value.id = Graph_ir.Tensor_id.of_int 7;
      sg =
        Tensor_sig.create
          ~id:(Graph_ir.Tensor_id.of_int 7)
          ~name:""
          ~shape:(Vec6.shape ~n:1 ~t:1 ~d:1 ~h:1 ~w:1 ~c:1)
          ~fmt:(Payload.Fmt Payload.F32) ();
      body;
      result = Kernel.Result_conversion.Round_f32;
    }
  in
  Format.printf "%a@."
    (Core.Pretty.err_result ~ok:(Fmt.any "built") ~error:Me_detail.pp_error)
    (Me_detail.of_value ~limits:tight ~key:(key 7) v);
  [%expect {| detail expressionNodes = 3 is over the ceiling |}]
