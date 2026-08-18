(* The flow spine, as a graph. See the .mli. *)

module ME = Model_explorer

type error = [ Me_limits.over_limit_error | Me_flow.error ]

let pp_error fmt : [< error ] -> unit = function
  | `Over_limit o -> Me_limits.Over_limit.pp fmt o
  | #Me_flow.error as e -> Me_flow.pp_error fmt e

let over_limit = Me_limits.check ~scope:Me_limits.Scope.Flow
let kv key value = ME.KeyValue.create ~key ~value

let attr key value =
  ME.NodeAttribute.create ~key ~value:(ME.NodeAttributeValue.Str value)

(* One slot, on every node, leaves included: [Me_session.validate] reads a
   node's slot count as the length of this list, so a node without one admits no
   incoming edge at all. *)
let slot attrs = [ ME.MetadataItem.create ~id:"0" ~attrs ]

let edge_from source =
  ME.IncomingEdge.create ~sourceNodeId:source ~sourceNodeOutputId:"0"
    ~targetNodeInputId:"0" ()

(* The pass detail a [Pass] transition carries and the others do not. Bounded
   like every other rendered attribute, and absent rather than empty for a kind
   that has no execution to name. *)
let kind_attrs ~limits (kind : Me_flow.Transition.kind) =
  match kind with
  | Me_flow.Transition.Pass e ->
      let text, _ =
        Me_build.bounded ~max:limits.Me_limits.Limits.max_attr_chars
          Me_flow.Pass_execution.pp e
      in
      [ attr "execution" text ]
  | Me_flow.Transition.Import | Me_flow.Transition.Pack
  | Me_flow.Transition.Cross_dialect | Me_flow.Transition.Adapt ->
      []

let graph ~limits (flow : Me_flow.t) =
  let open Err.Syntax in
  (* The DAG contract first -- it is where [max_states] and [max_transitions]
     are checked, and every walk below is linear in those two. *)
  let* () = Me_flow.validate ~limits flow in
  let states = List.length flow.Me_flow.states
  and transitions = List.length flow.Me_flow.transitions in
  (* Then this graph's own two ceilings, still before a single node exists.
     They are wire-selectable independently of [max_states]/[max_transitions],
     and no later validator counts one graph's nodes or edges -- so a projection
     that skipped this would be the one graph in the session on which
     [max_nodes_per_graph] means nothing. *)
  let* () =
    over_limit Me_limits.Field.Nodes (states + transitions)
      ~ceiling:limits.Me_limits.Limits.max_nodes_per_graph
  in
  let* () =
    over_limit Me_limits.Field.Edges (2 * transitions)
      ~ceiling:limits.Me_limits.Limits.max_edges_per_graph
  in
  (* [Me_flow.validate] has already proved every endpoint resolves, so this
     table cannot miss -- but it is read through [find_opt] and a miss is a
     defect rather than a silent empty edge list. *)
  let graph_of_state = Hashtbl.create 16 in
  List.iter
    (fun (s : Me_flow.State.t) ->
      Hashtbl.replace graph_of_state s.Me_flow.State.id s)
    flow.Me_flow.states;
  (* A state's incoming edge comes from the transition that produced it, which
     is exactly [produced_by]: the root has none, and [validate] has proved
     every other one agrees with the transition's own [after]. *)
  let state_node (s : Me_flow.State.t) =
    ME.GraphNode.create ~id:s.Me_flow.State.id ~label:s.Me_flow.State.label
      ~namespace:""
      ~incomingEdges:
        (match s.Me_flow.State.produced_by with
        | None -> []
        | Some t -> [ edge_from t ])
      ~attrs:
        [
          attr "layer" (Me_ids.Layer.to_string s.Me_flow.State.layer);
          attr "graph" s.Me_flow.State.graph;
          attr "view" s.Me_flow.State.view;
        ]
      ~outputsMetadata:
        (slot
           [ kv "state" s.Me_flow.State.id; kv "graph" s.Me_flow.State.graph ])
      ()
  in
  let transition_node (t : Me_flow.Transition.t) =
    ME.GraphNode.create ~id:t.Me_flow.Transition.id
      ~label:(Me_flow.Transition.kind_name t.Me_flow.Transition.kind)
      ~namespace:""
      ~incomingEdges:[ edge_from t.Me_flow.Transition.before ]
      ~attrs:
        (attr "kind" (Me_flow.Transition.kind_name t.Me_flow.Transition.kind)
         :: kind_attrs ~limits t.Me_flow.Transition.kind
        (* Named only when it exists. An absent comparison is honest absence,
           never an identity claim, so it gets no attribute rather than one
           saying "none". *)
        @
        match t.Me_flow.Transition.comparison with
        | None -> []
        | Some c -> [ attr "comparison" c ])
      ~outputsMetadata:(slot [ kv "transition" t.Me_flow.Transition.id ])
      ()
  in
  Err.return
    (ME.Graph.create ~id:flow.Me_flow.graph
       ~nodes:
         (List.map state_node flow.Me_flow.states
         @ List.map transition_node flow.Me_flow.transitions)
       ())
