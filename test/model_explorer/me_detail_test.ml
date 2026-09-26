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
           generated_js = None;
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

let operator_key node =
  Err.or_raise ~pp_error:MR.Request.pp_error
    (MR.Detail_key.create_operator ~limits ~parent_graph:"g/native/001"
       ~node:(Graph_ir.Node_id.of_int node))

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
    one   graphs=10 views=10
    two   graphs=11 views=11 |}]

let%expect_test "an operator detail links its canonical Native parent" =
  let k = operator_key 0 in
  Format.printf "%a@." pp (Me_detail.apply ~key:k ~limits session (delta k));
  [%expect {| graphs=9 views=9 |}]

let%expect_test "an operator detail also links its Stage and Kernel values" =
  let stage_id = Me_ids.graph Me_ids.Layer.Symbolic 0 in
  let k = operator_key 0 in
  let linked =
    Err.or_raise ~pp_error:Me_detail.pp_error
      (Me_detail.apply ~key:k ~limits session (delta k))
  in
  List.iter
    (fun (c : ME.GraphCollection.t) ->
      List.iter
        (fun (g : ME.Graph.t) ->
          if
            String.equal g.ME.Graph.id stage_id
            || String.equal g.ME.Graph.id kernel_id
          then
            List.iter
              (fun (n : ME.GraphNode.t) ->
                let origin =
                  List.find_map
                    (fun (attr : ME.NodeAttribute.t) ->
                      if
                        String.equal attr.ME.NodeAttribute.key
                          "canonical_native_node"
                      then
                        match attr.ME.NodeAttribute.value with
                        | ME.NodeAttributeValue.Str value -> Some value
                        | ME.NodeAttributeValue.NodeIds _
                        | ME.NodeAttributeValue.NodeWithAttrs _ ->
                            None
                      else None)
                    (Option.value n.ME.GraphNode.attrs ~default:[])
                in
                if origin = Some "n0" then
                  Fmt.pr "%s %s -> [%s]@." g.ME.Graph.id n.ME.GraphNode.id
                    (String.concat " "
                       (Option.value n.ME.GraphNode.subgraphIds ~default:[])))
              g.ME.Graph.nodes)
        c.ME.GraphCollection.graphs)
    linked.Me_session.Session.graph_collections;
  [%expect
    {|
    g/symbolic/000 v1 -> [expr/g/native/001/n0]
    g/kernel/000 v1 -> [expr/g/native/001/n0] |}]

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
    once  graphs=10 views=10
    twice graphs=10 views=10 |}]

let%expect_test "the initial session already links the parent node" =
  (* The canonical operator detail is part of the initial session, so the
     Stage/Kernel value starts with its native Model Explorer link. A later
     value delta remains an additional, independently keyed link. *)
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
    before  v1 -> [expr/g/native/001/n0]
    after   v1 -> [expr/g/kernel/000/t1/t1 expr/g/native/001/n0] |}]

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

(* The eager expression graphs degrade at the aggregate ceilings instead of
   failing the session on the graph that crosses them: what fits is installed
   and one diagnostic records the rest. *)
let session_with tight =
  Me_export.session ~limits:tight
    ~options:
      {
        Me_export.Options.stages = Me_session.Capability.all_stages;
        fold = false;
        generated_js = None;
        verify_symbolic = None;
        name = "tiny";
        source_bytes = Int64.of_int (String.length model);
        source_sha256 = None;
      }
    ~bytes:model

let%expect_test "the initial details degrade at the aggregate ceilings" =
  let all_graphs s =
    List.concat_map
      (fun (c : ME.GraphCollection.t) -> c.ME.GraphCollection.graphs)
      s.Me_session.Session.graph_collections
  in
  let detail_count s =
    List.length
      (List.filter
         (fun (g : ME.Graph.t) ->
           String.starts_with ~prefix:"expr/" g.ME.Graph.id)
         (all_graphs s))
  in
  let baseline = session in
  let total = List.length (all_graphs baseline) in
  let details = detail_count baseline in
  Format.printf "graphs=%d details=%d@." total details;
  let show label tight =
    let tight = Err.or_raise ~pp_error:Me_limits.pp_error tight in
    match session_with tight with
    | Error e ->
        Format.printf "%s: %a@." label Me_export.pp_error (Err.Error.kind e)
    | Ok s ->
        let omitted =
          List.filter
            (fun (d : Me_limits.Diagnostic.t) ->
              d.Me_limits.Diagnostic.code = Me_limits.Diagnostic.Code.Over_limit)
            s.Me_session.Session.diagnostics
        in
        Format.printf "%s: graphs=%d details=%d omitted-diagnostics=%d@." label
          (List.length (all_graphs s))
          (detail_count s) (List.length omitted)
  in
  show "detail ceiling = details" (L.create ~max_detail_graphs:details limits);
  show "detail ceiling = 1" (L.create ~max_detail_graphs:1 limits);
  show "graph ceiling = total" (L.create ~max_graphs:total limits);
  show "graph ceiling = total - 1" (L.create ~max_graphs:(total - 1) limits);
  show "graph ceiling = no room" (L.create ~max_graphs:(total - details) limits);
  [%expect
    {|
    graphs=9 details=2
    detail ceiling = details: graphs=9 details=2 omitted-diagnostics=0
    detail ceiling = 1: graphs=8 details=1 omitted-diagnostics=1
    graph ceiling = total: graphs=9 details=2 omitted-diagnostics=0
    graph ceiling = total - 1: graphs=8 details=1 omitted-diagnostics=1
    graph ceiling = no room: graphs=7 details=0 omitted-diagnostics=1 |}]

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
      computation = Region_group.Ref.Solo (Region_program.pixel body);
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
      e0  round_f32 from []
      e1  region    from [e0]
      e2  emitter   from [e1]
      e3  +         from [e2]
      e4  *         from [e3]
      e5  const 2   from [e4]
      e6  const 3   from [e4]
      e7  round_f32 from [e3]
      e8  const 4   from [e7] |}]

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
      computation = Region_group.Ref.Solo (Region_program.pixel body);
      result = Kernel.Result_conversion.Round_f32;
    }
  in
  Format.printf "%a@."
    (Core.Pretty.err_result ~ok:(Fmt.any "built") ~error:Me_detail.pp_error)
    (Me_detail.of_value ~limits:tight ~key:(key 7) v);
  [%expect {| detail expressionNodes = 3 is over the ceiling |}]

let%expect_test "a Region detail includes its locals and emitter" =
  let partition =
    Err.or_raise ~pp_error:Region_partition.pp_error
      (Region_partition.of_whole_axes [ Expr.Axis.C ])
  in
  let program =
    Err.or_raise ~pp_error:Region_program.pp_error
      (Region_program.Builder.run
         (Region_program.Builder.scalar (Expr.Value.const 2.) (fun local ->
              Region_program.Builder.finish ~max_size:16 ~max_depth:8 ~partition
                ~output:(Expr.Value.add local (Expr.Value.const 1.)))))
  in
  let v =
    {
      Kernel.Value.id = Graph_ir.Tensor_id.of_int 7;
      sg =
        Tensor_sig.create
          ~id:(Graph_ir.Tensor_id.of_int 7)
          ~name:""
          ~shape:(Vec6.shape ~n:1 ~t:1 ~d:1 ~h:1 ~w:1 ~c:2)
          ~fmt:(Payload.Fmt Payload.F32) ();
      computation = Region_group.Ref.Solo program;
      result = Kernel.Result_conversion.Round_f32;
    }
  in
  let graph =
    Err.or_raise ~pp_error:Me_detail.pp_error
      (Me_detail.of_value ~limits ~key:(key 7) v)
  in
  List.iter
    (fun (n : ME.GraphNode.t) ->
      Printf.printf "%s %s\n" n.ME.GraphNode.id n.ME.GraphNode.label)
    graph.ME.Graph.nodes;
  [%expect
    {|
    e0 round_f32
    e1 region
    e2 local l0
    e3 l0
    e4 const 2
    e5 emitter
    e6 +
    e7 local
    e8 const 1 |}]

(* --- generated JavaScript --- *)

let multi_output_node () =
  let g = Native_test.Graph_fixtures.multi_output () in
  let node =
    List.find
      (fun (n : Graph_ir.node) -> List.length n.Graph_ir.Node.outputs > 1)
      g.Graph_ir.Graph.nodes
  in
  (g, node)

let pp_js_attr ppf : Me_detail.Js_attr.t -> unit = function
  | Emitted text -> Fmt.pf ppf "Emitted (%d bytes)" (String.length text)
  | Unavailable e ->
      Fmt.pf ppf "Unavailable %a" Loop_ir.Loop_node_program.pp_error e

let direct ~passes g node ~output : Me_detail.Js_attr.t =
  match Loop_ir.Loop_node_program.lower ~passes g node ~output with
  | Ok program -> Emitted (Loop_ir.Loop_js.emit program)
  | Error e -> Unavailable (Err.Error.kind e)

let equal_attr (a : Me_detail.Js_attr.t) (b : Me_detail.Js_attr.t) =
  match (a, b) with
  | Emitted a, Emitted b -> String.equal a b
  | Unavailable _, Unavailable _ -> true
  | (Emitted _ | Unavailable _), _ -> false

let%expect_test
    "generated_js (optimized and raw) is exactly Loop_js.emit of \
     Loop_node_program.lower, per output" =
  let g, node = multi_output_node () in
  List.iter
    (fun (ordinal, _) ->
      List.iter
        (fun (label, passes) ->
          let via_detail =
            Me_detail.generated_js ~passes g node ~output:ordinal
          in
          let expected = direct ~passes g node ~output:ordinal in
          Fmt.pr "output %a %-9s agree=%b %a@." Output_ordinal.pp ordinal label
            (equal_attr via_detail expected)
            pp_js_attr via_detail)
        [ ("optimized", Loop_ir.Loop_opt.passes); ("raw", []) ])
    (Output_ordinal.indexed node.Graph_ir.Node.outputs);
  [%expect
    {|
    output 0 optimized agree=true Emitted (729 bytes)
    output 0 raw       agree=true Emitted (1076 bytes)
    output 1 optimized agree=true Emitted (1187 bytes)
    output 1 raw       agree=true Emitted (1558 bytes) |}]

(* Mutation proof: if [of_operator] ever attached output 0's JS to every
   [out<i>] node, these two outputs' [js] texts would stop differing, and this
   test would go red. Output 0 (the pooled value) and output 1 (its index,
   converted to int64) are genuinely different kernels, so a same-JS bug
   cannot produce this pattern by accident. Run for both the default
   (optimized) and the raw toggle, since the two are exclusive: at most one
   [js] family is ever attached to a built graph, never both at once. *)
let%expect_test
    "of_operator decorates each out<i> with its own js, not output 0's for \
     every output -- true for both the raw/optimized toggle" =
  let g, node = multi_output_node () in
  let outputs =
    List.map
      (fun id ->
        {
          Kernel.Value.id;
          sg =
            Tensor_sig.create ~id ~name:""
              ~shape:(Vec6.shape ~n:1 ~t:1 ~d:1 ~h:1 ~w:1 ~c:1)
              ~fmt:(Payload.Fmt Payload.F32) ();
          computation =
            Region_group.Ref.Solo (Region_program.pixel (Expr.Value.const 0.));
          result = Kernel.Result_conversion.Round_f32;
        })
      node.Graph_ir.Node.outputs
  in
  let attr_value (n : ME.GraphNode.t) key =
    List.find_map
      (fun (a : ME.NodeAttribute.t) ->
        if String.equal a.ME.NodeAttribute.key key then
          match a.ME.NodeAttribute.value with
          | ME.NodeAttributeValue.Str v -> Some v
          | ME.NodeAttributeValue.NodeIds _
          | ME.NodeAttributeValue.NodeWithAttrs _ ->
              None
        else None)
      (Option.value n.ME.GraphNode.attrs ~default:[])
  in
  List.iter
    (fun (label, passes) ->
      let generated =
        List.map
          (fun (ordinal, _) ->
            Me_detail.generated_js ~passes g node ~output:ordinal)
          (Output_ordinal.indexed node.Graph_ir.Node.outputs)
      in
      let graph =
        Err.or_raise ~pp_error:Me_detail.pp_error
          (Me_detail.of_operator ~limits ~key:(operator_key 0) ~outputs
             ~generated_js:generated ())
      in
      let out_nodes =
        List.filter
          (fun (n : ME.GraphNode.t) ->
            String.length n.ME.GraphNode.id > 3
            && String.sub n.ME.GraphNode.id 0 3 = "out")
          graph.ME.Graph.nodes
      in
      let js = List.map (fun n -> Option.get (attr_value n "js")) out_nodes in
      Fmt.pr "%-9s out0 js <> out1 js: %b@." label
        (not (String.equal (List.nth js 0) (List.nth js 1))))
    [ ("optimized", Loop_ir.Loop_opt.passes); ("raw", []) ];
  [%expect
    {|
    optimized out0 js <> out1 js: true
    raw       out0 js <> out1 js: true |}]

(* A tight [Kernel.Limits.t] (not [Me_limits.Limits.t], which only bounds the
   rendered attribute -- this is the same budget [Loop_node_program.kernel]'s
   own [Kernel_adapt.of_stage_program] admits against) forces a genuine
   [`Adapt] failure: [max_size:1] is under the max-pool node's own per-
   expression size, not merely under some proxy. *)
let tight_kernel_limits =
  let d = Kernel.Limits.default in
  Err.or_raise ~pp_error:Kernel.Limits.pp_error
    (Kernel.Limits.create ~max_size:1 ~max_depth:d.max_depth ~max_values:1
       ~max_dep_depth:d.max_dep_depth ~max_inputs:d.max_inputs
       ~max_outputs:d.max_outputs ~max_extent:d.max_extent
       ~max_numel:d.max_numel ~max_bytes:d.max_bytes
       ~max_local_slots:d.max_local_slots ~max_scan_state:d.max_scan_state
       ~max_scan_updates_per_key:d.max_scan_updates_per_key
       ~max_scan_updates_total:d.max_scan_updates_total)

let%expect_test "an output that does not lower is Unavailable, never silent" =
  let g, node = multi_output_node () in
  List.iter
    (fun (ordinal, _) ->
      Fmt.pr "output %a %a@." Output_ordinal.pp ordinal pp_js_attr
        (Me_detail.generated_js ~limits:tight_kernel_limits g node
           ~output:ordinal))
    (Output_ordinal.indexed node.Graph_ir.Node.outputs);
  [%expect
    {|
    output 0 Unavailable t1: size exceeds limit 1
    output 1 Unavailable t1: size exceeds limit 1 |}]

(* [of_operator] renders [Unavailable] as [js_unavailable], with the lowering
   error's own text -- never [js] on a silently-degraded value, and never a
   bare empty attribute. *)
let%expect_test "of_operator renders Unavailable as js_unavailable" =
  let g, node = multi_output_node () in
  let outputs =
    List.map
      (fun id ->
        {
          Kernel.Value.id;
          sg =
            Tensor_sig.create ~id ~name:""
              ~shape:(Vec6.shape ~n:1 ~t:1 ~d:1 ~h:1 ~w:1 ~c:1)
              ~fmt:(Payload.Fmt Payload.F32) ();
          computation =
            Region_group.Ref.Solo (Region_program.pixel (Expr.Value.const 0.));
          result = Kernel.Result_conversion.Round_f32;
        })
      node.Graph_ir.Node.outputs
  in
  let generated =
    List.map
      (fun (ordinal, _) ->
        Me_detail.generated_js ~limits:tight_kernel_limits g node
          ~output:ordinal)
      (Output_ordinal.indexed node.Graph_ir.Node.outputs)
  in
  let graph =
    Err.or_raise ~pp_error:Me_detail.pp_error
      (Me_detail.of_operator ~limits ~key:(operator_key 0) ~outputs
         ~generated_js:generated ())
  in
  let attr_value (n : ME.GraphNode.t) key =
    List.find_map
      (fun (a : ME.NodeAttribute.t) ->
        if String.equal a.ME.NodeAttribute.key key then
          match a.ME.NodeAttribute.value with
          | ME.NodeAttributeValue.Str v -> Some v
          | ME.NodeAttributeValue.NodeIds _
          | ME.NodeAttributeValue.NodeWithAttrs _ ->
              None
        else None)
      (Option.value n.ME.GraphNode.attrs ~default:[])
  in
  List.iter
    (fun (n : ME.GraphNode.t) ->
      if
        String.length n.ME.GraphNode.id > 2
        && String.sub n.ME.GraphNode.id 0 3 = "out"
      then
        Fmt.pr "%s js=%b js_unavailable=%a@." n.ME.GraphNode.id
          (Option.is_some (attr_value n "js"))
          (Core.Pretty.option_or ~none:"absent" Fmt.string)
          (attr_value n "js_unavailable"))
    graph.ME.Graph.nodes;
  [%expect
    {|
    out0 js=false js_unavailable=t1: size exceeds limit 1
    out1 js=false js_unavailable=t1: size exceeds limit 1 |}]

(* The rendered attribute ceiling ([Me_limits.Limits.max_attr_chars]), distinct
   from the Kernel budget above: a program that lowers and emits FINE still
   gets cut for display, with [js_truncated] recording that it was. *)
let%expect_test "js_truncated at a small max_attr_chars profile" =
  let g, node = multi_output_node () in
  let tight =
    Err.or_raise ~pp_error:Me_limits.pp_error
      (L.create ~max_attr_chars:16 limits)
  in
  let outputs =
    List.map
      (fun id ->
        {
          Kernel.Value.id;
          sg =
            Tensor_sig.create ~id ~name:""
              ~shape:(Vec6.shape ~n:1 ~t:1 ~d:1 ~h:1 ~w:1 ~c:1)
              ~fmt:(Payload.Fmt Payload.F32) ();
          computation =
            Region_group.Ref.Solo (Region_program.pixel (Expr.Value.const 0.));
          result = Kernel.Result_conversion.Round_f32;
        })
      node.Graph_ir.Node.outputs
  in
  let generated =
    List.map
      (fun (ordinal, _) -> Me_detail.generated_js g node ~output:ordinal)
      (Output_ordinal.indexed node.Graph_ir.Node.outputs)
  in
  let graph =
    Err.or_raise ~pp_error:Me_detail.pp_error
      (Me_detail.of_operator ~limits:tight ~key:(operator_key 0) ~outputs
         ~generated_js:generated ())
  in
  let attr_value (n : ME.GraphNode.t) key =
    List.find_map
      (fun (a : ME.NodeAttribute.t) ->
        if String.equal a.ME.NodeAttribute.key key then
          match a.ME.NodeAttribute.value with
          | ME.NodeAttributeValue.Str v -> Some v
          | ME.NodeAttributeValue.NodeIds _
          | ME.NodeAttributeValue.NodeWithAttrs _ ->
              None
        else None)
      (Option.value n.ME.GraphNode.attrs ~default:[])
  in
  List.iter
    (fun (n : ME.GraphNode.t) ->
      if
        String.length n.ME.GraphNode.id > 2
        && String.sub n.ME.GraphNode.id 0 3 = "out"
      then
        Fmt.pr "%s js_len=%d js_truncated=%b@." n.ME.GraphNode.id
          (String.length (Option.get (attr_value n "js")))
          (Option.is_some (attr_value n "js_truncated")))
    graph.ME.Graph.nodes;
  [%expect
    {|
    out0 js_len=16 js_truncated=true
    out1 js_len=16 js_truncated=true |}]
