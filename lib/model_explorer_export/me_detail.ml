(* Expression detail. See the .mli. *)

module ME = Model_explorer

type error =
  [ `Document of Me_session.Session.error
  | `Key_disagrees_with_ids
  | Me_limits.over_limit_error
  | `Unsupported_detail_key ]

let pp_error fmt : [< error ] -> unit = function
  | `Document e -> Me_session.Session.pp_error fmt e
  | `Key_disagrees_with_ids ->
      Fmt.string fmt "the delta's graph and view do not carry the key's id"
  | `Over_limit o -> Me_limits.Over_limit.pp fmt o
  | `Unsupported_detail_key ->
      Fmt.string fmt "the key names no value in that graph"

let count = Me_limits.check ~scope:Me_limits.Scope.Detail

(* --- the expression graph ----------------------------------------------- *)

let attr key value =
  ME.NodeAttribute.create ~key ~value:(ME.NodeAttributeValue.Str value)

(* Detail edges are decomposition edges.  The renderer's slots stay opaque
   identifiers; the role is edge metadata, where a reader can distinguish an
   arithmetic operand from a reduction bound or coordinate component. *)
let edge ~parent ~role =
  ME.IncomingEdge.create ~sourceNodeId:parent ~sourceNodeOutputId:"0"
    ~targetNodeInputId:"0"
    ~metadata:(ME.String_map.singleton "role" role)
    ()

type binding = { id : string; name : string }

type scope = {
  reducers : binding Expr.Reduce_var.Map.t;
  locals : binding Expr.Local_var.Map.t;
}

let empty_scope =
  { reducers = Expr.Reduce_var.Map.empty; locals = Expr.Local_var.Map.empty }

let raw pp v = Core.Pretty.to_string pp v

let of_value ~limits ~key (v : Kernel.Value.t) =
  let open Err.Syntax in
  (* This is the output measure, rather than [Expr.Fold.size]: presentation
     roots, binders, Boolean terms, and index terms all become graph nodes.
     The actual ceiling is deliberately checked before the mutable emitter
     below allocates graph nodes. *)
  let emitted = ref 0 in
  let charge () =
    if !emitted >= limits.Me_limits.Limits.max_detail_nodes then false
    else begin
      incr emitted;
      true
    end
  in
  let rec measure_index : type r. r Expr.Index.t -> bool = function
    | Expr.Index.Add (a, b) | Expr.Index.Max (a, b) | Expr.Index.Min (a, b) ->
        charge () && measure_index a && measure_index b
    | Expr.Index.Assume_position a | Expr.Index.Clamp_low a ->
        charge () && measure_index a
    | Expr.Index.Of_position a -> charge () && measure_index a
    | Expr.Index.Ceil_div_pos (a, _) | Expr.Index.Floor_div_pos (a, _) ->
        charge () && measure_index a
    | Expr.Index.Data (_, coord, _) ->
        charge ()
        && List.for_all
             (fun axis -> measure_index (Expr.Coord.get coord axis))
             Expr.Axis.all
    | Expr.Index.Const _ | Expr.Index.Output _ | Expr.Index.Reduce _
    | Expr.Index.Zero ->
        charge ()
    | Expr.Index.Scale (_, a) -> charge () && measure_index a
  and measure_bool = function
    | Expr.Bool.Index_eq (a, b) ->
        charge () && measure_index a && measure_index b
    | Expr.Bool.Value_lt (a, b) ->
        charge () && measure_value a && measure_value b
  and measure_value = function
    | Expr.Value.Binary (_, a, b) ->
        charge () && measure_value a && measure_value b
    | Expr.Value.Const _ | Expr.Value.Local _ -> charge ()
    | Expr.Value.Intrinsic (Expr.Intrinsic.Max_pool p) ->
        charge ()
        && List.for_all
             (fun axis -> measure_index (Expr.Coord.get p.out axis))
             Expr.Axis.all
    | Expr.Value.Local_at (_, i) -> charge () && measure_index i
    | Expr.Value.Local_scan_at (_, row, lane) ->
        charge () && measure_index row && measure_index lane
    | Expr.Value.Scan_at (scan, row, lane) ->
        charge () && measure_index row && measure_index lane && charge ()
        && charge () && measure_value scan.init && charge () && charge ()
        && charge () && charge () && charge () && measure_value scan.update
    | Expr.Value.Load (_, coord) ->
        charge ()
        && List.for_all
             (fun axis -> measure_index (Expr.Coord.get coord axis))
             Expr.Axis.all
    | Expr.Value.Reduce r ->
        charge () && measure_index r.lo && measure_index r.hi && charge ()
        && measure_value r.body
    | Expr.Value.Round_f32 a | Expr.Value.Unary (_, a) ->
        charge () && measure_value a
    | Expr.Value.Select (b, t, f) ->
        charge () && measure_bool b && measure_value t && measure_value f
    | Expr.Value.Value_of_index i -> charge () && measure_index i
  in
  let measure_region p =
    charge ()
    && List.for_all
         (fun local ->
           charge ()
           &&
           match local.Region_local.rhs with
           | Region_local.Rhs.Scalar body -> charge () && measure_value body
           | Region_local.Rhs.Vector { body; _ } ->
               charge () && measure_value body
           | Region_local.Rhs.Scan scan ->
               (* The local binder, then the same [Scan_at] presentation that
                  the emitter below uses for an unspecialized trace local. *)
               charge ()
               && measure_value
                    (Expr.Value.scan_at scan ~row:Expr.Index.zero
                       ~lane:Expr.Index.zero))
         (Region_program.locals p)
    && charge ()
    && measure_value (Region_program.output p)
  in
  let* () =
    if charge () && measure_region v.Kernel.Value.computation then Err.return ()
    else
      count Me_limits.Field.Expression_nodes (!emitted + 1)
        ~ceiling:limits.Me_limits.Limits.max_detail_nodes
  in
  let nodes = ref [] and next = ref 0 in
  let add ?parent ?(role = "contains") ~language ~constructor ~label
      ?(attrs = []) () =
    let id = Printf.sprintf "e%d" !next in
    incr next;
    let attrs =
      attr "language" language :: attr "constructor" constructor :: attrs
    in
    nodes :=
      ME.GraphNode.create ~id ~label ~namespace:""
        ~incomingEdges:
          (Option.to_list (Option.map (fun p -> edge ~parent:p ~role) parent))
        ~outputsMetadata:[ ME.MetadataItem.create ~id:"0" ~attrs:[] ]
        ~attrs ()
      :: !nodes;
    id
  in
  let binder ?parent ~role ~kind ~name () =
    let id =
      add ?parent ~role ~language:"binding" ~constructor:kind ~label:name ()
    in
    { id; name }
  in
  let binder_attrs scope reducer =
    match Expr.Reduce_var.Map.find_opt reducer scope.reducers with
    | Some b -> [ attr "bound_by" b.id; attr "binding" b.name ]
    | None ->
        [ attr "scope" "free"; attr "binding" (raw Expr.Reduce_var.pp reducer) ]
  in
  let local_attrs scope local =
    match Expr.Local_var.Map.find_opt local scope.locals with
    | Some b -> [ attr "bound_by" b.id; attr "binding" b.name ]
    | None ->
        [ attr "scope" "free"; attr "binding" (raw Expr.Local_var.pp local) ]
  in
  let rec walk_index : type r.
      scope -> parent:string -> role:string -> r Expr.Index.t -> unit =
   fun scope ~parent ~role index ->
    match index with
    | Expr.Index.Add (a, b) ->
        let id =
          add ~parent ~role ~language:"index" ~constructor:"add" ~label:"+"
            ~attrs:[ attr "role" "delta" ]
            ()
        in
        walk_index scope ~parent:id ~role:"lhs" a;
        walk_index scope ~parent:id ~role:"rhs" b
    | Expr.Index.Assume_position a ->
        let id =
          add ~parent ~role ~language:"index" ~constructor:"assume_position"
            ~label:"assume_position"
            ~attrs:[ attr "role" "position" ]
            ()
        in
        walk_index scope ~parent:id ~role:"operand" a
    | Expr.Index.Ceil_div_pos (a, d) ->
        let id =
          add ~parent ~role ~language:"index" ~constructor:"ceil_div_pos"
            ~label:(Fmt.str "ceil_div %d" d)
            ~attrs:[ attr "role" "delta" ]
            ()
        in
        walk_index scope ~parent:id ~role:"operand" a
    | Expr.Index.Clamp_low a ->
        let id =
          add ~parent ~role ~language:"index" ~constructor:"clamp_low"
            ~label:"clamp_low"
            ~attrs:[ attr "role" "position" ]
            ()
        in
        walk_index scope ~parent:id ~role:"operand" a
    | Expr.Index.Const n ->
        ignore
          (add ~parent ~role ~language:"index" ~constructor:"const"
             ~label:(Fmt.str "const %d" n)
             ~attrs:[ attr "role" "delta" ]
             ())
    | Expr.Index.Data (src, coord, extent) ->
        let id =
          add ~parent ~role ~language:"index" ~constructor:"data"
            ~label:(Fmt.str "data %a" Expr.Source.pp src)
            ~attrs:
              [ attr "role" "position"; attr "extent" (string_of_int extent) ]
            ()
        in
        walk_coord scope ~parent:id coord
    | Expr.Index.Floor_div_pos (a, d) ->
        let id =
          add ~parent ~role ~language:"index" ~constructor:"floor_div_pos"
            ~label:(Fmt.str "floor_div %d" d)
            ~attrs:[ attr "role" "delta" ]
            ()
        in
        walk_index scope ~parent:id ~role:"operand" a
    | (Expr.Index.Max (a, b) | Expr.Index.Min (a, b)) as term ->
        let constructor, label =
          match term with
          | Expr.Index.Max _ -> ("max", "max")
          | Expr.Index.Min _ -> ("min", "min")
          | _ -> assert false
        in
        let id =
          add ~parent ~role ~language:"index" ~constructor ~label
            ~attrs:[ attr "role" "delta" ]
            ()
        in
        walk_index scope ~parent:id ~role:"lhs" a;
        walk_index scope ~parent:id ~role:"rhs" b
    | Expr.Index.Of_position a ->
        let id =
          add ~parent ~role ~language:"index" ~constructor:"of_position"
            ~label:"of_position"
            ~attrs:[ attr "role" "delta" ]
            ()
        in
        walk_index scope ~parent:id ~role:"operand" a
    | Expr.Index.Output axis ->
        ignore
          (add ~parent ~role ~language:"index" ~constructor:"output"
             ~label:(Fmt.str "output %s" (Expr.Axis.to_string axis))
             ~attrs:[ attr "role" "position" ]
             ())
    | Expr.Index.Reduce reducer ->
        ignore
          (add ~parent ~role ~language:"index" ~constructor:"reduce"
             ~label:"reduce"
             ~attrs:(attr "role" "position" :: binder_attrs scope reducer)
             ())
    | Expr.Index.Scale (k, a) ->
        let id =
          add ~parent ~role ~language:"index" ~constructor:"scale"
            ~label:(Fmt.str "scale %d" k)
            ~attrs:[ attr "role" "delta" ]
            ()
        in
        walk_index scope ~parent:id ~role:"operand" a
    | Expr.Index.Zero ->
        ignore
          (add ~parent ~role ~language:"index" ~constructor:"zero" ~label:"zero"
             ~attrs:[ attr "role" "position" ]
             ())
  and walk_coord scope ~parent coord =
    List.iter
      (fun axis ->
        walk_index scope ~parent
          ~role:("coord:" ^ Expr.Axis.to_string axis)
          (Expr.Coord.get coord axis))
      Expr.Axis.all
  and walk_bool scope ~parent ~role = function
    | Expr.Bool.Index_eq (a, b) ->
        let id =
          add ~parent ~role ~language:"bool" ~constructor:"index_eq"
            ~label:"index_eq" ()
        in
        walk_index scope ~parent:id ~role:"lhs" a;
        walk_index scope ~parent:id ~role:"rhs" b
    | Expr.Bool.Value_lt (a, b) ->
        let id =
          add ~parent ~role ~language:"bool" ~constructor:"value_lt"
            ~label:"value_lt" ()
        in
        walk_value scope ~parent:id ~role:"lhs" a;
        walk_value scope ~parent:id ~role:"rhs" b
  and walk_value scope ~parent ~role = function
    | Expr.Value.Binary (op, a, b) ->
        let id =
          add ~parent ~role ~language:"value" ~constructor:"binary"
            ~label:(Expr.Value.binary_sym op) ()
        in
        walk_value scope ~parent:id ~role:"lhs" a;
        walk_value scope ~parent:id ~role:"rhs" b
    | Expr.Value.Const c ->
        ignore
          (add ~parent ~role ~language:"value" ~constructor:"const"
             ~label:(Fmt.str "const %g" c) ())
    | Expr.Value.Intrinsic (Expr.Intrinsic.Max_pool p) ->
        let id =
          add ~parent ~role ~language:"value" ~constructor:"max_pool"
            ~label:"max_pool"
            ~attrs:
              [
                attr "source" (raw Expr.Source.pp p.source);
                attr "result" (Expr.Intrinsic.Max_pool.result_name p.result);
              ]
            ()
        in
        walk_coord scope ~parent:id p.out
    | Expr.Value.Local local ->
        ignore
          (add ~parent ~role ~language:"value" ~constructor:"local"
             ~label:"local" ~attrs:(local_attrs scope local) ())
    | Expr.Value.Local_at (local, index) ->
        let id =
          add ~parent ~role ~language:"value" ~constructor:"local_at"
            ~label:"local_at" ~attrs:(local_attrs scope local) ()
        in
        walk_index scope ~parent:id ~role:"lane" index
    | Expr.Value.Local_scan_at (local, row, lane) ->
        let id =
          add ~parent ~role ~language:"value" ~constructor:"local_scan_at"
            ~label:"trace_at" ~attrs:(local_attrs scope local) ()
        in
        walk_index scope ~parent:id ~role:"row" row;
        walk_index scope ~parent:id ~role:"lane" lane
    | Expr.Value.Load (src, coord) ->
        let id =
          add ~parent ~role ~language:"value" ~constructor:"load"
            ~label:(Fmt.str "load %a" Expr.Source.pp src)
            ()
        in
        walk_coord scope ~parent:id coord
    | Expr.Value.Reduce reduction ->
        let id =
          add ~parent ~role ~language:"value" ~constructor:"reduce"
            ~label:(Expr.Reduction.kind_name reduction.kind)
            ()
        in
        walk_index scope ~parent:id ~role:"lower" reduction.lo;
        walk_index scope ~parent:id ~role:"upper" reduction.hi;
        let b =
          binder ~parent:id ~role:"binder" ~kind:"reducer" ~name:"reducer" ()
        in
        let scope =
          {
            scope with
            reducers = Expr.Reduce_var.Map.add reduction.var b scope.reducers;
          }
        in
        walk_value scope ~parent:id ~role:"body" reduction.body
    | Expr.Value.Round_f32 a ->
        let id =
          add ~parent ~role ~language:"value" ~constructor:"round_f32"
            ~label:"round_f32" ()
        in
        walk_value scope ~parent:id ~role:"operand" a
    | Expr.Value.Scan_at (scan, row, lane) ->
        let id =
          add ~parent ~role ~language:"value" ~constructor:"scan_at"
            ~label:"scan"
            ~attrs:
              [
                attr "width" (string_of_int scan.width);
                attr "steps" (string_of_int scan.steps);
              ]
            ()
        in
        walk_index scope ~parent:id ~role:"row" row;
        walk_index scope ~parent:id ~role:"lane" lane;
        let init =
          add ~parent:id ~role:"init" ~language:"presentation"
            ~constructor:"scan_init" ~label:"init" ()
        in
        let init_b =
          binder ~parent:init ~role:"binder" ~kind:"scan_lane" ~name:"lane" ()
        in
        let init_scope =
          {
            scope with
            reducers = Expr.Reduce_var.Map.add scan.lane init_b scope.reducers;
          }
        in
        walk_value init_scope ~parent:init ~role:"body" scan.init;
        let update =
          add ~parent:id ~role:"update" ~language:"presentation"
            ~constructor:"scan_update" ~label:"update" ()
        in
        let lane_b =
          binder ~parent:update ~role:"binder" ~kind:"scan_lane" ~name:"lane" ()
        in
        let step_b =
          binder ~parent:update ~role:"binder" ~kind:"scan_step" ~name:"step" ()
        in
        let prev_b =
          binder ~parent:update ~role:"binder" ~kind:"scan_previous"
            ~name:"previous" ()
        in
        let update_scope =
          {
            reducers =
              scope.reducers
              |> Expr.Reduce_var.Map.add scan.lane lane_b
              |> Expr.Reduce_var.Map.add scan.step step_b;
            locals = Expr.Local_var.Map.add scan.prev prev_b scope.locals;
          }
        in
        walk_value update_scope ~parent:update ~role:"body" scan.update
    | Expr.Value.Select (condition, t, f) ->
        let id =
          add ~parent ~role ~language:"value" ~constructor:"select"
            ~label:"select" ()
        in
        walk_bool scope ~parent:id ~role:"condition" condition;
        walk_value scope ~parent:id ~role:"true_branch" t;
        walk_value scope ~parent:id ~role:"false_branch" f
    | Expr.Value.Unary (op, a) ->
        let id =
          add ~parent ~role ~language:"value" ~constructor:"unary"
            ~label:(Expr.Value.unary_name op) ()
        in
        walk_value scope ~parent:id ~role:"operand" a
    | Expr.Value.Value_of_index index ->
        let id =
          add ~parent ~role ~language:"value" ~constructor:"value_of_index"
            ~label:"index" ()
        in
        walk_index scope ~parent:id ~role:"operand" index
  in
  let root =
    add ~language:"presentation" ~constructor:"result"
      ~label:(Kernel.Result_conversion.name v.Kernel.Value.result)
      ()
  in
  let region =
    add ~parent:root ~role:"computation" ~language:"region"
      ~constructor:"region" ~label:"region" ()
  in
  let scope = ref empty_scope in
  List.iteri
    (fun i (local : Region_local.t) ->
      let name = Fmt.str "l%d" i in
      let local_node =
        add ~parent:region ~role:(Fmt.str "local:%d" i) ~language:"region"
          ~constructor:"local" ~label:("local " ^ name) ()
      in
      let b = binder ~parent:local_node ~role:"binder" ~kind:"local" ~name () in
      scope :=
        {
          !scope with
          locals = Expr.Local_var.Map.add local.Region_local.id b !scope.locals;
        };
      match local.Region_local.rhs with
      | Region_local.Rhs.Scalar body ->
          walk_value !scope ~parent:local_node ~role:"body" body
      | Region_local.Rhs.Vector { body; var; _ } ->
          let vector =
            binder ~parent:local_node ~role:"binder" ~kind:"vector_lane"
              ~name:(name ^ "_lane") ()
          in
          let scoped =
            {
              !scope with
              reducers = Expr.Reduce_var.Map.add var vector !scope.reducers;
            }
          in
          walk_value scoped ~parent:local_node ~role:"body" body
      | Region_local.Rhs.Scan scan ->
          walk_value !scope ~parent:local_node ~role:"body"
            (Expr.Value.scan_at scan ~row:Expr.Index.zero ~lane:Expr.Index.zero))
    (Region_program.locals v.Kernel.Value.computation);
  let emitter =
    add ~parent:region ~role:"emitter" ~language:"region" ~constructor:"emitter"
      ~label:"emitter" ()
  in
  walk_value !scope ~parent:emitter ~role:"body"
    (Region_program.output v.Kernel.Value.computation);
  Err.return
    (ME.Graph.create
       ~id:(Me_request.Detail_key.id key)
       ~nodes:(List.rev !nodes) ())

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
