(* Expression detail. See the .mli. *)

module ME = Model_explorer

type error =
  [ `Key_disagrees_with_ids
  | `Unsupported_detail_key
  | Me_limits.over_limit_error
  | `Document of Me_session.Session.error ]

let pp_error fmt : [< error ] -> unit = function
  | `Key_disagrees_with_ids ->
      Fmt.string fmt "the delta's graph and view do not carry the key's id"
  | `Unsupported_detail_key ->
      Fmt.string fmt "the key names no value in that graph"
  | `Over_limit o -> Me_limits.Over_limit.pp fmt o
  | `Document e -> Me_session.Session.pp_error fmt e

let count = Me_limits.check ~scope:Me_limits.Scope.Detail

(* --- the expression graph ----------------------------------------------- *)

let attr key value =
  ME.NodeAttribute.create ~key ~value:(ME.NodeAttributeValue.Str value)

(* The node's own label: the constructor, and for the operators the symbol,
   because "binary" alone tells a reader nothing a picture needs. *)
let label (v : Expr.Value.t) =
  match v with
  | Expr.Value.Const c -> Printf.sprintf "const %g" c
  | Expr.Value.Binary (op, _, _) -> Expr.Value.binary_sym op
  | Expr.Value.Unary (op, _) -> Expr.Value.unary_name op
  | Expr.Value.Select _ -> "select"
  | Expr.Value.Value_of_index _ -> "index"
  | Expr.Value.Load (src, _) -> Core.Pretty.to_string Expr.Source.pp src
  | Expr.Value.Round_f32 _ -> "round_f32"
  | Expr.Value.Reduce r -> Expr.Reduction.kind_name r.Expr.Reduction.kind
  | Expr.Value.Intrinsic _ -> "max_pool"

(* The children a node has AS VALUES. A [Select]'s condition and a [Reduce]'s
   bounds are index-level terms, not value-level ones, so they become attributes
   below rather than nodes: putting two languages in one graph would make the
   picture claim a dataflow the expression does not have. *)
let children (v : Expr.Value.t) =
  match v with
  | Expr.Value.Const _ | Expr.Value.Value_of_index _ | Expr.Value.Load _
  | Expr.Value.Intrinsic _ ->
      []
  | Expr.Value.Binary (_, a, b) -> [ a; b ]
  | Expr.Value.Unary (_, a) -> [ a ]
  | Expr.Value.Round_f32 a -> [ a ]
  | Expr.Value.Select (Expr.Bool.Value_lt (a, b), t, f) -> [ a; b; t; f ]
  | Expr.Value.Select (Expr.Bool.Index_eq _, t, f) -> [ t; f ]
  | Expr.Value.Reduce r -> [ r.Expr.Reduction.body ]

(* What a node says beyond its label: the subtree rooted there, rendered and
   bounded.

   Every node gets the same attribute, and that is deliberate. The alternative
   was per-constructor attributes for the INDEX-level parts -- a [Select]'s
   condition, a [Reduce]'s bounds -- and [Expr.Pp.index] takes a [names]
   function precisely so its output cannot depend on allocation history. This
   walk holds no scoped naming environment, so it has no honest one to give;
   [Expr.Pp.value] builds its own, which is why routing everything through it is
   correct rather than merely convenient. For a leaf the rendering IS the node;
   for an interior one it is what that branch computes. *)
let attrs (v : Expr.Value.t) =
  let text, capped = Me_build.bounded ~max:256 Expr.Pp.value v in
  attr "expr" text :: (if capped then [ attr "expr_truncated" "true" ] else [])

let of_value ~limits ~key (v : Kernel.Value.t) =
  let open Err.Syntax in
  let body = v.Kernel.Value.body in
  (* BEFORE the walk. [Fold.size] is an unmetered traversal but allocates
     nothing, while building the nodes allocates per node -- and an expression is
     exactly the shape whose size is not apparent from the thing that names it.

     It counts INDEX TREES too, so it exceeds the number of value nodes this
     produces: a convolution is 88 by that measure and 8 nodes here. The bound is
     therefore conservative, deliberately -- the index trees are what the bounded
     per-node rendering pays for, so they belong in the figure the ceiling
     governs -- and the field is named for what was measured rather than for what
     was built. *)
  let+ () =
    count Me_limits.Field.Expression_nodes (Expr.Fold.size body)
      ~ceiling:limits.Me_limits.Limits.max_detail_nodes
  in
  (* Pre-order, so a node's id orders it the way a reader reads the expression,
     and the ROOT is node 0 whatever the tree looks like. *)
  let nodes = ref [] in
  let next = ref 0 in
  let rec walk parent v =
    let id = Printf.sprintf "e%d" !next in
    incr next;
    nodes :=
      ME.GraphNode.create ~id ~label:(label v) ~namespace:""
        ~incomingEdges:
          (match parent with
        | None -> []
        | Some p ->
            [
              ME.IncomingEdge.create ~sourceNodeId:p ~sourceNodeOutputId:"0"
                ~targetNodeInputId:"0" ();
            ])
          (* One output, slot 0: every node in this walk is a single-valued
             expression, whether a leaf or the parent of children referencing
             it -- so this is not conditional on having a child, the way a
             real tensor op's outputs can differ node to node. Empty attrs:
             there is no shape or SSA name to report here, unlike a PT2 or
             kernel node -- [attrs v] above already carries what this node
             IS. *)
        ~outputsMetadata:[ ME.MetadataItem.create ~id:"0" ~attrs:[] ]
        ~attrs:(attrs v) ()
      :: !nodes;
    List.iter (walk (Some id)) (children v)
  in
  walk None body;
  ME.Graph.create ~id:(Me_request.Detail_key.id key) ~nodes:(List.rev !nodes) ()

(* --- the delta ----------------------------------------------------------- *)

module Delta = struct
  type t = {
    schema_version : int;
    collection : string;
    graph : ME.Graph.t;
    view : Me_session.View.t;
    node_data : Me_session.Node_data_set.t list;
    diagnostics : Me_limits.Diagnostic.t list;
  }

  let jsont =
    Jsont.Object.map ~kind:"detailDelta"
      (fun schema_version collection graph view node_data diagnostics ->
        { schema_version; collection; graph; view; node_data; diagnostics })
    |> Jsont.Object.mem "schemaVersion" Jsont.int ~enc:(fun t ->
        t.schema_version)
    |> Jsont.Object.mem "collection" Jsont.string ~enc:(fun t -> t.collection)
    |> Jsont.Object.mem "graph" ME.Graph.jsont ~enc:(fun t -> t.graph)
    |> Jsont.Object.mem "view" Me_session.View.jsont ~enc:(fun t -> t.view)
    |> Jsont.Object.mem "nodeData" (Jsont.list Me_session.Node_data_set.jsont)
         ~dec_absent:[] ~enc:(fun t -> t.node_data)
    |> Jsont.Object.mem "diagnostics" (Jsont.list Me_limits.Diagnostic.jsont)
         ~dec_absent:[] ~enc:(fun t -> t.diagnostics)
    |> Jsont.Object.finish
end

(* --- merging ------------------------------------------------------------- *)

(* A detail graph is one this module produced, and its id is the key's. So
   "which details are installed" is answerable from the session alone, with no
   second list to keep in step. *)
let is_detail (g : ME.Graph.t) =
  let id = g.ME.Graph.id in
  String.length id >= 5 && String.sub id 0 5 = "expr/"

let apply ~key ~limits (s : Me_session.Session.t) (d : Delta.t) =
  let open Err.Syntax in
  let expected = Me_request.Detail_key.id key in
  let* () =
    if
      String.equal d.Delta.graph.ME.Graph.id expected
      && String.equal d.Delta.view.Me_session.View.id expected
    then Err.return ()
    else Err.fail `Key_disagrees_with_ids
  in
  let parent = key.Me_request.Detail_key.parent_graph in
  let node = Me_ids.value_node key.Me_request.Detail_key.value in
  let* () =
    if
      List.exists
        (fun (c : ME.GraphCollection.t) ->
          List.exists
            (fun (g : ME.Graph.t) ->
              String.equal g.ME.Graph.id parent
              && List.exists
                   (fun (n : ME.GraphNode.t) ->
                     String.equal n.ME.GraphNode.id node)
                   g.ME.Graph.nodes)
            c.ME.GraphCollection.graphs)
        s.Me_session.Session.graph_collections
    then Err.return ()
    else Err.fail `Unsupported_detail_key
  in
  (* The delta ALONE. A merged check would let a delta that is itself over the
     ceiling pass whenever the session it joins is small. *)
  let* () =
    count Me_limits.Field.Detail_nodes
      (List.length d.Delta.graph.ME.Graph.nodes)
      ~ceiling:limits.Me_limits.Limits.max_detail_nodes
  in
  (* REPLACEMENT by key, so a re-request cannot inflate the aggregates: only
     the committed result counts. The graph and the view carry the same id, so
     one predicate removes both. *)
  let drop_previous xs id_of =
    List.filter (fun x -> not (String.equal (id_of x) expected)) xs
  in
  let collections =
    List.map
      (fun (c : ME.GraphCollection.t) ->
        if not (String.equal c.ME.GraphCollection.label d.Delta.collection) then
          c
        else
          let graphs =
            drop_previous c.ME.GraphCollection.graphs (fun (g : ME.Graph.t) ->
                g.ME.Graph.id)
            @ [ d.Delta.graph ]
          in
          (* [subGraphIds] on the parent node and the graph itself arrive
             TOGETHER with the graph, because a detail commits all of them or
             none: a link to a graph that is not installed is a dangling
             reference the session validator rejects, and a graph nobody links
             to is unreachable. *)
          let graphs =
            List.map
              (fun (g : ME.Graph.t) ->
                if not (String.equal g.ME.Graph.id parent) then g
                else
                  {
                    g with
                    ME.Graph.nodes =
                      List.map
                        (fun (n : ME.GraphNode.t) ->
                          if not (String.equal n.ME.GraphNode.id node) then n
                          else
                            {
                              n with
                              ME.GraphNode.subgraphIds =
                                Some
                                  (List.sort_uniq compare
                                     (expected
                                     :: Option.value n.ME.GraphNode.subgraphIds
                                          ~default:[]));
                            })
                        g.ME.Graph.nodes;
                  })
              graphs
          in
          { c with ME.GraphCollection.graphs })
      s.Me_session.Session.graph_collections
  in
  let views =
    drop_previous s.Me_session.Session.views (fun (v : Me_session.View.t) ->
        v.Me_session.View.id)
    @ [ d.Delta.view ]
  in
  let merged =
    {
      s with
      Me_session.Session.graph_collections = collections;
      views;
      node_data_sets = s.Me_session.Session.node_data_sets @ d.Delta.node_data;
      diagnostics = s.Me_session.Session.diagnostics @ d.Delta.diagnostics;
    }
  in
  (* The AGGREGATES, over every installed detail rather than over this one. *)
  let all_graphs =
    List.concat_map
      (fun (c : ME.GraphCollection.t) -> c.ME.GraphCollection.graphs)
      collections
  in
  let* () =
    count Me_limits.Field.Detail_graphs
      (List.length (List.filter is_detail all_graphs))
      ~ceiling:limits.Me_limits.Limits.max_detail_graphs
  in
  let* () =
    count Me_limits.Field.Graphs (List.length all_graphs)
      ~ceiling:limits.Me_limits.Limits.max_graphs
  in
  let+ () =
    Err.map_error
      (fun e -> `Document e)
      (Me_session.Session.validate ~limits merged)
  in
  merged
