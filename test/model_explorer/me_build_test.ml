(* [Me_build] / [Me_native]: the projection (.ai/model_explorer_design.md).

   Two things here are not visible in a rendering and are the reason the module
   is shaped as it is: that an attribute's bound stops the PRINTER rather than
   trimming its output, and that namespaces come off the [Group] tree through
   the encoder, so a group label containing a slash cannot move a node. *)

module ME = Model_explorer

let limits = Me_limits.Limits.untrusted

let pp_err ppf r =
  Core.Pretty.core_result ~ok:(Fmt.any "ok") ~error:Me_build.pp_error ppf r

(* --- the bounded printer --- *)

let%expect_test "the cap stops the printer, it does not trim the result" =
  (* A printer that would emit far more than the cap, instrumented to count how
     much it was ASKED to produce. Under [asprintf]-then-truncate the counter
     reaches the end; here it stops, which is the whole difference between a
     bound on the output and a bound on the work. *)
  let emitted = ref 0 in
  let pp fmt () =
    for i = 0 to 999 do
      incr emitted;
      Fmt.pf fmt "%d," i
    done
  in
  let text, capped = Me_build.bounded ~max:16 pp () in
  Printf.printf "%S capped=%b, printer reached %d of 1000 items\n" text capped
    !emitted;
  emitted := 0;
  let text2, capped2 = Me_build.bounded ~max:100_000 pp () in
  Printf.printf "len=%d capped=%b, printer reached %d of 1000\n"
    (String.length text2) capped2 !emitted;
  [%expect
    {|
    "0,1,2,3,4,5,6,7," capped=true, printer reached 30 of 1000 items
    len=3890 capped=false, printer reached 1000 of 1000 |}]

let%expect_test "a cap of one byte still yields a whole byte" =
  let text, capped = Me_build.bounded ~max:1 Fmt.string "abcdef" in
  Printf.printf "%S %b\n" text capped;
  [%expect {| "a" true |}]

(* --- a graph to project --- *)

let tid = Graph_ir.Tensor_id.of_int
let nid = Graph_ir.Node_id.of_int

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

(* t0 input, t1 constant, t2 = relu t0, t3 = relu t2; output t3. Two nodes in a
   labelled group, so the namespace has something to say. *)
let graph ?(group_label = Some "features") () =
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
                label = group_label;
                items =
                  [ Graph_ir.Group.Node (nid 0); Graph_ir.Group.Node (nid 1) ];
              };
          ];
      };
    tensors =
      List.fold_left
        (fun m i -> Graph_ir.Tensor_id.Map.add (tid i) (sig_ i) m)
        Graph_ir.Tensor_id.Map.empty [ 0; 1; 2; 3 ];
    inputs = [ tid 0; tid 1 ];
    input_kinds =
      Graph_ir.Tensor_id.Map.add (tid 1) Graph_ir.Input.Constant
        (Graph_ir.Tensor_id.Map.add (tid 0) Graph_ir.Input.Input
           Graph_ir.Tensor_id.Map.empty);
    outputs = [ tid 3 ];
  }

let show g =
  match Me_native.graph ~limits ~id:"g/native/000" g with
  | Error e -> Format.printf "%a@." pp_err (Error e)
  | Ok mg ->
      List.iter
        (fun (n : ME.GraphNode.t) ->
          Printf.printf "%-12s %-8s ns=%-16s in=[%s]%s\n" n.ME.GraphNode.id
            n.ME.GraphNode.label n.ME.GraphNode.namespace
            (String.concat " "
               (List.map
                  (fun (e : ME.IncomingEdge.t) ->
                    e.ME.IncomingEdge.sourceNodeId ^ ":"
                    ^ e.ME.IncomingEdge.sourceNodeOutputId)
                  (Option.value n.ME.GraphNode.incomingEdges ~default:[])))
            (match n.ME.GraphNode.attrs with
            | None | Some [] -> ""
            | Some attrs ->
                " "
                ^ String.concat ","
                    (List.map
                       (fun (a : ME.NodeAttribute.t) ->
                         a.ME.NodeAttribute.key ^ "="
                         ^
                         match a.ME.NodeAttribute.value with
                         | ME.NodeAttributeValue.Str s -> s
                         | _ -> "<non-string>")
                       attrs)))
        mg.ME.Graph.nodes

let%expect_test "nodes, boundaries and edges" =
  (* Graph inputs and outputs become PINNED boundary nodes, so an edge from an
     input is an ordinary edge rather than a case every consumer must know
     about, and a comparison matches them by id like any other node. *)
  show (graph ());
  [%expect
    {|
    in:t0        input    ns=                 in=[]
    const:t1     constant ns=                 in=[]
    n0           Relu     ns=features#g1      in=[in:t0:0] params=relu x=t0
    n1           Relu     ns=features#g1      in=[n0:0] params=relu x=t2
    out:t3       output   ns=                 in=[n1:0] |}]

(* --- namespaces --- *)

let%expect_test "a group label containing a slash cannot move a node" =
  (* The renderer splits [namespace] on ['/'], so an unencoded label would put
     the node one level deeper than the hierarchy says. This is the case the
     encoder exists for, and it is invisible in any rendering that happens not
     to use such a label. *)
  show (graph ~group_label:(Some "features/0") ());
  [%expect
    {|
    in:t0        input    ns=                 in=[]
    const:t1     constant ns=                 in=[]
    n0           Relu     ns=features%2F0#g1  in=[in:t0:0] params=relu x=t0
    n1           Relu     ns=features%2F0#g1  in=[n0:0] params=relu x=t2
    out:t3       output   ns=                 in=[n1:0] |}]

let%expect_test "an unlabelled group still names a level" =
  show (graph ~group_label:None ());
  [%expect
    {|
    in:t0        input    ns=                 in=[]
    const:t1     constant ns=                 in=[]
    n0           Relu     ns=g1               in=[in:t0:0] params=relu x=t0
    n1           Relu     ns=g1               in=[n0:0] params=relu x=t2
    out:t3       output   ns=                 in=[n1:0] |}]

(* --- failures --- *)

let%expect_test "an operand nothing produces is named, not silently dropped" =
  let g = graph () in
  let g = { g with Graph_ir.Graph.nodes = [ relu ~id:0 ~x:9 ~out:2 ] } in
  Format.printf "%a@." pp_err (Me_native.graph ~limits ~id:"g" g);
  [%expect {| tensor t9 has no producer |}]

let%expect_test "the ceilings, checked before the walks" =
  let tight =
    Core.or_raise Me_limits.pp_error
      (Me_limits.Limits.create ~max_nodes_per_graph:2 limits)
  in
  Format.printf "nodes %a@." pp_err
    (Me_native.graph ~limits:tight ~id:"g" (graph ()));
  let tight_e =
    Core.or_raise Me_limits.pp_error
      (Me_limits.Limits.create ~max_edges_per_graph:1 limits)
  in
  Format.printf "edges %a@." pp_err
    (Me_native.graph ~limits:tight_e ~id:"g" (graph ()));
  [%expect
    {|
    nodes graph nodes = 5 is over the ceiling
    edges graph edges = 3 is over the ceiling |}]

let%expect_test "an over-long parameter render is capped and says so" =
  let tight =
    Core.or_raise Me_limits.pp_error
      (Me_limits.Limits.create ~max_attr_chars:4 limits)
  in
  (match Me_native.graph ~limits:tight ~id:"g" (graph ()) with
  | Error e -> Format.printf "%a@." pp_err (Error e)
  | Ok mg ->
      List.iter
        (fun (n : ME.GraphNode.t) ->
          match n.ME.GraphNode.attrs with
          | Some attrs when n.ME.GraphNode.id = "n0" ->
              List.iter
                (fun (a : ME.NodeAttribute.t) ->
                  Printf.printf "%s=%s\n" a.ME.NodeAttribute.key
                    (match a.ME.NodeAttribute.value with
                    | ME.NodeAttributeValue.Str s -> s
                    | _ -> "?"))
                attrs
          | _ -> ())
        mg.ME.Graph.nodes);
  [%expect {|
    params=relu
    params_truncated=true |}]

(* --- the projection is a valid session payload --- *)

let%expect_test "what Me_build produces, Me_session accepts" =
  (* The two halves have to agree about edges and slots, and neither test alone
     would show a disagreement: [Me_build] would happily emit a slot index
     [Me_session.validate] rejects. *)
  match Me_native.graph ~limits ~id:"g/native/000" (graph ()) with
  | Error e -> Format.printf "%a@." pp_err (Error e)
  | Ok mg ->
      let collection =
        ME.GraphCollection.create ~label:"mltorch:m" ~graphs:[ mg ] ()
      in
      let capabilities =
        List.map
          (fun k ->
            {
              Me_session.Capability.key = k;
              status =
                (match k with
                | Me_session.Capability.Feature Me_session.Capability.Loop_ir
                | Me_session.Capability.Feature Me_session.Capability.Codegen ->
                    Me_session.Capability.Unavailable
                      {
                        reason = Me_session.Capability.Not_implemented;
                        detail = None;
                      }
                | _ -> Me_session.Capability.Not_requested);
            })
          Me_session.Capability.all_keys
      in
      let session =
        {
          Me_session.Session.schema_version = 1;
          producer =
            { Me_session.Producer.tool = "mltorch"; session_schema = 1 };
          model =
            {
              Me_session.Model_summary.name = "fixture";
              source_kind = Me_session.Model_summary.Json;
              source_bytes = 0L;
              source_sha256 = None;
              pt2_graph_count = 1;
              op_targets = 2;
            };
          graph_collections = [ collection ];
          views =
            [
              {
                Me_session.View.id = "v";
                label = "Initial Native";
                kind =
                  Me_session.View.Stage Me_session.Capability.Initial_native;
                collection = "mltorch:m";
                graph = "g/native/000";
              };
            ];
          comparisons = [];
          node_data_sets = [];
          flow = None;
          capabilities;
          diagnostics = [];
          default_view = "v";
        }
      in
      Format.printf "%a@."
        (Core.Pretty.core_result ~ok:(Fmt.any "valid session")
           ~error:Me_session.Session.pp_error)
        (Me_session.Session.validate ~limits session);
      [%expect {| valid session |}]
