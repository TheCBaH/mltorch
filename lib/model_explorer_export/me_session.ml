(* Visualization_session v1. See the .mli. *)

module Producer = struct
  type t = { tool : string; session_schema : int }

  let jsont =
    Jsont.Object.map ~kind:"producer" (fun tool session_schema ->
        { tool; session_schema })
    |> Jsont.Object.mem "tool" Jsont.string ~enc:(fun p -> p.tool)
    |> Jsont.Object.mem "sessionSchema" Jsont.int ~enc:(fun p ->
        p.session_schema)
    |> Jsont.Object.finish
end

module Model_summary = struct
  type source_kind = Pt2 | Json

  type t = {
    name : string;
    source_kind : source_kind;
    source_bytes : int64;
    source_sha256 : string option;
    pt2_graph_count : int;
    op_targets : int;
  }

  let source_kind_jsont =
    Jsont.enum ~kind:"source_kind" [ ("pt2", Pt2); ("json", Json) ]

  let jsont =
    Jsont.Object.map ~kind:"model_summary"
      (fun
        name
        source_kind
        source_bytes
        source_sha256
        pt2_graph_count
        op_targets
      ->
        {
          name;
          source_kind;
          source_bytes;
          source_sha256;
          pt2_graph_count;
          op_targets;
        })
    |> Jsont.Object.mem "name" Jsont.string ~enc:(fun m -> m.name)
    |> Jsont.Object.mem "sourceKind" source_kind_jsont ~enc:(fun m ->
        m.source_kind)
    (* [int64_as_string], never the adaptive [int64]: a byte count past 2^53
       loses precision as a JSON number, and this one is a file size. *)
    |> Jsont.Object.mem "sourceBytes" Jsont.int64_as_string ~enc:(fun m ->
        m.source_bytes)
    |> Jsont.Object.opt_mem "sourceSha256" Jsont.string ~enc:(fun m ->
        m.source_sha256)
    |> Jsont.Object.mem "pt2GraphCount" Jsont.int ~enc:(fun m ->
        m.pt2_graph_count)
    |> Jsont.Object.mem "opTargets" Jsont.int ~enc:(fun m -> m.op_targets)
    |> Jsont.Object.finish
end

module Pane_state = struct
  type t = { collection : string; graph : string }

  let jsont =
    Jsont.Object.map ~kind:"pane_state" (fun collection graph ->
        { collection; graph })
    |> Jsont.Object.mem "collection" Jsont.string ~enc:(fun p -> p.collection)
    |> Jsont.Object.mem "graph" Jsont.string ~enc:(fun p -> p.graph)
    |> Jsont.Object.finish
end

module Mapping_entry = struct
  type t = { left : string list; right : string list }

  let jsont =
    Jsont.Object.map ~kind:"mapping_entry" (fun left right -> { left; right })
    |> Jsont.Object.mem "left" (Jsont.list Jsont.string) ~enc:(fun e -> e.left)
    |> Jsont.Object.mem "right" (Jsont.list Jsont.string) ~enc:(fun e ->
        e.right)
    |> Jsont.Object.finish
end

module Sync_navigation = struct
  type t = {
    entries : Mapping_entry.t list;
    show_diff_highlights : bool;
    match_node_id_fallback : bool;
  }

  let jsont =
    Jsont.Object.map ~kind:"sync_navigation"
      (fun entries show_diff_highlights match_node_id_fallback ->
        { entries; show_diff_highlights; match_node_id_fallback })
    |> Jsont.Object.mem "entries" (Jsont.list Mapping_entry.jsont)
         ~enc:(fun s -> s.entries)
    |> Jsont.Object.mem "showDiffHighlights" Jsont.bool ~enc:(fun s ->
        s.show_diff_highlights)
    |> Jsont.Object.mem "matchNodeIdFallback" Jsont.bool ~enc:(fun s ->
        s.match_node_id_fallback)
    |> Jsont.Object.finish
end

(* Both carry a [state] and a [view] label, so they are separate modules rather
   than one flat pair of types: distinct namespaces are how this repo keeps
   labels unique instead of silencing warning 30. *)
module Flow_state_view = struct
  type t = { state : string; view : string }
end

module Flow_view_graph = struct
  type t = {
    state : string;
    view : string;
    state_graph : string;
    view_graph : string;
  }
end

(* Which of the two facts that must agree with the spine disagreed. A closed
   variant rather than a string: the printer needs to say which, and a caller
   that only reads the rendered text cannot act on it. *)
module Flow_destination = struct
  type part = Flow_view | Flow_capability
  type t = { part : part; declared : string; named : string option }
end

module Comparison = struct
  type t = {
    id : string;
    label : string;
    left : Pane_state.t;
    right : Pane_state.t;
    sync : Sync_navigation.t;
    overlays_left : Model_explorer.EdgeOverlay.t list;
    overlays_right : Model_explorer.EdgeOverlay.t list;
  }

  let jsont =
    Jsont.Object.map ~kind:"comparison"
      (fun id label left right sync overlays_left overlays_right ->
        { id; label; left; right; sync; overlays_left; overlays_right })
    |> Jsont.Object.mem "id" Jsont.string ~enc:(fun c -> c.id)
    |> Jsont.Object.mem "label" Jsont.string ~enc:(fun c -> c.label)
    |> Jsont.Object.mem "left" Pane_state.jsont ~enc:(fun c -> c.left)
    |> Jsont.Object.mem "right" Pane_state.jsont ~enc:(fun c -> c.right)
    |> Jsont.Object.mem "sync" Sync_navigation.jsont ~enc:(fun c -> c.sync)
    |> Jsont.Object.mem "overlaysLeft"
         (Jsont.list Model_explorer.EdgeOverlay.jsont) ~enc:(fun c ->
           c.overlays_left)
    |> Jsont.Object.mem "overlaysRight"
         (Jsont.list Model_explorer.EdgeOverlay.jsont) ~enc:(fun c ->
           c.overlays_right)
    |> Jsont.Object.finish
end

module Node_data_set = struct
  type value = { value : float; label : string option }
  type t = { name : string; graph : string; results : (string * value) list }

  let value_jsont =
    Jsont.Object.map ~kind:"node_data_value" (fun value label ->
        { value; label })
    |> Jsont.Object.mem "value" Jsont.number ~enc:(fun v -> v.value)
    |> Jsont.Object.opt_mem "label" Jsont.string ~enc:(fun v -> v.label)
    |> Jsont.Object.finish

  (* An ARRAY of {nodeId, ...}, not an object keyed by node id: the canonical
     order is part of the determinism claim and a JSON object would be
     reordered by whatever map rebuilt it. Same reasoning as
     [Pass.Outcome_counts.jsont]. *)
  let result_jsont =
    Jsont.Object.map ~kind:"node_data_result" (fun node_id value ->
        (node_id, value))
    |> Jsont.Object.mem "nodeId" Jsont.string ~enc:fst
    |> Jsont.Object.mem "value" value_jsont ~enc:snd
    |> Jsont.Object.finish

  let jsont =
    Jsont.Object.map ~kind:"node_data_set" (fun name graph results ->
        { name; graph; results })
    |> Jsont.Object.mem "name" Jsont.string ~enc:(fun s -> s.name)
    |> Jsont.Object.mem "graph" Jsont.string ~enc:(fun s -> s.graph)
    |> Jsont.Object.mem "results" (Jsont.list result_jsont) ~enc:(fun s ->
        s.results)
    |> Jsont.Object.finish
end

module Capability = struct
  type graph_stage =
    | Source
    | Initial_native
    | Canonical
    | Native4d
    | Stage_program
    | Kernel
    | Fusion

  type feature =
    | Flow
    | Verification
    | Pass_audits
    | Fold
    | Expression_detail
    | Loop_ir
    | Codegen

  type key = Graph_stage of graph_stage | Feature of feature

  let stage_name = function
    | Source -> "source"
    | Initial_native -> "initial_native"
    | Canonical -> "canonical"
    | Native4d -> "native4d"
    | Stage_program -> "stage_program"
    | Kernel -> "kernel"
    | Fusion -> "fusion"

  let feature_name = function
    | Flow -> "flow"
    | Verification -> "verification"
    | Pass_audits -> "pass_audits"
    | Fold -> "fold"
    | Expression_detail -> "expression_detail"
    | Loop_ir -> "loop_ir"
    | Codegen -> "codegen"

  let key_name = function
    | Graph_stage s -> "stage:" ^ stage_name s
    | Feature f -> "feature:" ^ feature_name f

  (* Successor chains again, for the reason [Diagnostic.Code] has one: a key
     added to the type and not to [all_keys] is a key [validate] would never
     miss, so completeness would silently stop meaning what it says. *)
  let next_stage = function
    | Source -> Some Initial_native
    | Initial_native -> Some Canonical
    | Canonical -> Some Native4d
    | Native4d -> Some Stage_program
    | Stage_program -> Some Kernel
    | Kernel -> Some Fusion
    | Fusion -> None

  let next_feature = function
    | Flow -> Some Verification
    | Verification -> Some Pass_audits
    | Pass_audits -> Some Fold
    | Fold -> Some Expression_detail
    | Expression_detail -> Some Loop_ir
    | Loop_ir -> Some Codegen
    | Codegen -> None

  let chain next first =
    let rec walk acc c =
      match next c with
      | None -> List.rev (c :: acc)
      | Some n -> walk (c :: acc) n
    in
    walk [] first

  let all_stages = chain next_stage Source
  let all_features = chain next_feature Flow

  let all_keys =
    List.map (fun s -> Graph_stage s) all_stages
    @ List.map (fun f -> Feature f) all_features

  module Pass_audit_status = struct
    type t = {
      retained_reports : int64;
      omitted_reports : int64;
      omitted_counts : Pass.Outcome_counts.t;
    }

    let jsont =
      Jsont.Object.map ~kind:"pass_audit_status"
        (fun retained_reports omitted_reports omitted_counts ->
          { retained_reports; omitted_reports; omitted_counts })
      |> Jsont.Object.mem "retainedReports" Jsont.int64_as_string ~enc:(fun t ->
          t.retained_reports)
      |> Jsont.Object.mem "omittedReports" Jsont.int64_as_string ~enc:(fun t ->
          t.omitted_reports)
      (* Through [Pass.Outcome_counts.jsont] in BOTH directions, so a malformed
         binding cannot become an unchecked map on the way in. It surfaces as a
         Jsont decode message rather than the typed [`Invalid_counts], because a
         [Jsont.t] has no typed error channel. *)
      |> Jsont.Object.mem "omittedCounts" Pass.Outcome_counts.jsont
           ~enc:(fun t -> t.omitted_counts)
      |> Jsont.Object.finish
  end

  type payload =
    | Graph of string
    | Verification_summary of Pass.Outcome_counts.t
    | Pass_audit_status of Pass_audit_status.t
    | Present

  type reason =
    | Unsupported_operator
    | Unsupported_input
    | Unsupported_graph_shape
    | Outside_dialect_domain
    | Over_limit
    | Requires_payloads
    | Prerequisite_unavailable
    | Not_implemented

  type status =
    | Available of payload
    | Unavailable of { reason : reason; detail : string option }
    | Not_requested

  type t = { key : key; status : status }

  (* The table IS the specification, so it is one exhaustive match rather than
     a set of guards: a key added to [key] stops this compiling. *)
  let compatible c =
    match (c.key, c.status) with
    | Feature Loop_ir, Unavailable { reason = Not_implemented; _ }
    | Feature Codegen, Unavailable { reason = Not_implemented; _ } ->
        true
    (* Never available and never merely unrequested: there is nothing to
       request. *)
    | Feature Loop_ir, _ | Feature Codegen, _ -> false
    | _, Not_requested -> true
    | _, Unavailable _ -> true
    | Graph_stage _, Available (Graph _) -> true
    | Feature Flow, Available (Graph _) -> true
    | Feature Verification, Available (Verification_summary _) -> true
    | Feature Pass_audits, Available (Pass_audit_status _) -> true
    | Feature Fold, Available Present -> true
    | Feature Expression_detail, Available Present -> true
    | _, Available _ -> false

  let units =
    "Verification_summary counts verifier clusters in the composed report; \
     Pass_audit_status counts audit reports in retained_reports and \
     omitted_reports, and clusters in omitted_counts."

  (* --- wire --- *)

  let key_jsont =
    Jsont.enum ~kind:"capability_key"
      (List.map (fun k -> (key_name k, k)) all_keys)

  let reason_jsont =
    Jsont.enum ~kind:"capability_reason"
      [
        ("unsupported_operator", Unsupported_operator);
        ("unsupported_input", Unsupported_input);
        ("unsupported_graph_shape", Unsupported_graph_shape);
        ("outside_dialect_domain", Outside_dialect_domain);
        ("over_limit", Over_limit);
        ("requires_payloads", Requires_payloads);
        ("prerequisite_unavailable", Prerequisite_unavailable);
        ("not_implemented", Not_implemented);
      ]

  (* A tagged union rather than four optional members, so a payload that names
     no kind is not representable on the wire either. *)
  let payload_jsont =
    let kind_of = function
      | Graph _ -> "graph"
      | Verification_summary _ -> "verification_summary"
      | Pass_audit_status _ -> "pass_audit_status"
      | Present -> "present"
    in
    Jsont.Object.map ~kind:"capability_payload"
      (fun kind graph verification pass_audits ->
        match (kind, graph, verification, pass_audits) with
        | "graph", Some g, None, None -> Graph g
        | "verification_summary", None, Some v, None -> Verification_summary v
        | "pass_audit_status", None, None, Some p -> Pass_audit_status p
        | "present", None, None, None -> Present
        | _ ->
            Jsont.Error.msgf Jsont.Meta.none
              "capability payload %S does not carry its own field" kind)
    |> Jsont.Object.mem "kind" Jsont.string ~enc:kind_of
    |> Jsont.Object.opt_mem "graph" Jsont.string ~enc:(function
      | Graph g -> Some g
      | _ -> None)
    |> Jsont.Object.opt_mem "verificationSummary" Pass.Outcome_counts.jsont
         ~enc:(function
         | Verification_summary v -> Some v
         | _ -> None)
    |> Jsont.Object.opt_mem "passAuditStatus" Pass_audit_status.jsont
         ~enc:(function
         | Pass_audit_status p -> Some p
         | _ -> None)
    |> Jsont.Object.finish

  let status_jsont =
    Jsont.Object.map ~kind:"capability_status"
      (fun state payload reason detail ->
        match (state, payload, reason) with
        | "available", Some p, None -> Available p
        | "unavailable", None, Some reason -> Unavailable { reason; detail }
        | "not_requested", None, None -> Not_requested
        | _ ->
            Jsont.Error.msgf Jsont.Meta.none
              "capability status %S does not carry its own field" state)
    |> Jsont.Object.mem "state" Jsont.string ~enc:(function
      | Available _ -> "available"
      | Unavailable _ -> "unavailable"
      | Not_requested -> "not_requested")
    |> Jsont.Object.opt_mem "payload" payload_jsont ~enc:(function
      | Available p -> Some p
      | _ -> None)
    |> Jsont.Object.opt_mem "reason" reason_jsont ~enc:(function
      | Unavailable { reason; _ } -> Some reason
      | _ -> None)
    |> Jsont.Object.opt_mem "detail" Jsont.string ~enc:(function
      | Unavailable { detail; _ } -> detail
      | _ -> None)
    |> Jsont.Object.finish

  let jsont =
    Jsont.Object.map ~kind:"capability" (fun key status -> { key; status })
    |> Jsont.Object.mem "key" key_jsont ~enc:(fun c -> c.key)
    |> Jsont.Object.mem "status" status_jsont ~enc:(fun c -> c.status)
    |> Jsont.Object.finish
end

module View = struct
  type kind = Stage of Capability.graph_stage | Flow | Compare

  type t = {
    id : string;
    label : string;
    kind : kind;
    collection : string;
    graph : string;
  }

  let kind_jsont =
    Jsont.enum ~kind:"view_kind"
      (List.map
         (fun s -> ("stage:" ^ Capability.stage_name s, Stage s))
         Capability.all_stages
      @ [ ("flow", Flow); ("compare", Compare) ])

  let jsont =
    Jsont.Object.map ~kind:"view" (fun id label kind collection graph ->
        { id; label; kind; collection; graph })
    |> Jsont.Object.mem "id" Jsont.string ~enc:(fun v -> v.id)
    |> Jsont.Object.mem "label" Jsont.string ~enc:(fun v -> v.label)
    |> Jsont.Object.mem "kind" kind_jsont ~enc:(fun v -> v.kind)
    |> Jsont.Object.mem "collection" Jsont.string ~enc:(fun v -> v.collection)
    |> Jsont.Object.mem "graph" Jsont.string ~enc:(fun v -> v.graph)
    |> Jsont.Object.finish
end

module Session = struct
  type t = {
    schema_version : int;
    producer : Producer.t;
    model : Model_summary.t;
    graph_collections : Model_explorer.GraphCollection.t list;
    views : View.t list;
    comparisons : Comparison.t list;
    node_data_sets : Node_data_set.t list;
    flow : Me_flow.t option;
    capabilities : Capability.t list;
    diagnostics : Me_limits.Diagnostic.t list;
    default_view : string;
  }

  type graph_node = { graph : string; node : string }
  type placed_graph = { graph : string; collection : string }
  type mapped_node = { comparison : string; node : string }

  type error =
    [ `Duplicate_graph of string
    | `Duplicate_node of graph_node
    | `Unknown_graph of string
    | `Unknown_node of graph_node
    | `Wrong_collection of placed_graph
    | `Slot_mismatch of graph_node
    | `Unknown_view of string
    | `Duplicate_view of string
    | `Duplicate_comparison of string
    | `Unknown_comparison of string
    | `Node_in_two_entries of mapped_node
    | `Duplicate_flow_view of string
    | `Flow_destination_disagrees of Flow_destination.t
    | `Unexpected_flow_view of string
    | `Unexpected_flow_capability of string
    | `Flow_view_unknown of Flow_state_view.t
    | `Flow_view_not_stage of Flow_state_view.t
    | `Flow_view_graph_disagrees of Flow_view_graph.t
    | `Comparison_panes_disagree of string
    | `Duplicate_capability of string
    | `Missing_capability of string
    | `Incompatible_capability of string
    | Me_limits.over_limit_error
    | Me_flow.error ]

  let pp_error fmt : [< error ] -> unit = function
    | `Duplicate_graph id -> Fmt.pf fmt "duplicate graph id %s" id
    | `Duplicate_node { graph; node } ->
        Fmt.pf fmt "duplicate node id %s in graph %s" node graph
    | `Unknown_graph id -> Fmt.pf fmt "unknown graph %s" id
    | `Unknown_node { graph; node } ->
        Fmt.pf fmt "unknown node %s in graph %s" node graph
    | `Wrong_collection { graph; collection } ->
        Fmt.pf fmt "graph %s is not in collection %s" graph collection
    | `Slot_mismatch { graph; node } ->
        Fmt.pf fmt "edge in graph %s names no output slot of node %s" graph node
    | `Unknown_view id -> Fmt.pf fmt "unknown view %s" id
    | `Duplicate_view id -> Fmt.pf fmt "duplicate view id %s" id
    | `Duplicate_comparison id -> Fmt.pf fmt "duplicate comparison id %s" id
    | `Unknown_comparison id -> Fmt.pf fmt "unknown comparison %s" id
    | `Node_in_two_entries { comparison; node } ->
        Fmt.pf fmt "node %s appears in two mapping entries of comparison %s"
          node comparison
    | `Duplicate_flow_view id -> Fmt.pf fmt "duplicate flow view %s" id
    | `Flow_destination_disagrees { Flow_destination.part; declared; named }
      -> (
        let what =
          match part with
          | Flow_destination.Flow_view -> "flow view"
          | Flow_destination.Flow_capability -> "feature:flow capability"
        in
        match named with
        | None ->
            Fmt.pf fmt "the flow declares graph %s but there is no %s" declared
              what
        | Some g ->
            Fmt.pf fmt "the flow declares graph %s but its %s names %s" declared
              what g)
    | `Unexpected_flow_view id ->
        Fmt.pf fmt "flow view %s in a session that declares no flow" id
    | `Unexpected_flow_capability g ->
        Fmt.pf fmt
          "feature:flow offers graph %s in a session that declares no flow" g
    | `Flow_view_unknown { Flow_state_view.state; view } ->
        Fmt.pf fmt "flow state %s names unknown view %s" state view
    | `Flow_view_not_stage { Flow_state_view.state; view } ->
        Fmt.pf fmt "flow state %s names view %s, which is not a stage view"
          state view
    | `Flow_view_graph_disagrees
        { Flow_view_graph.state; view; state_graph; view_graph } ->
        Fmt.pf fmt "flow state %s is graph %s but its view %s shows graph %s"
          state state_graph view view_graph
    | `Comparison_panes_disagree t ->
        Fmt.pf fmt "transition %s names a comparison over other graphs" t
    | `Duplicate_capability k -> Fmt.pf fmt "duplicate capability %s" k
    | `Missing_capability k -> Fmt.pf fmt "missing capability %s" k
    | `Incompatible_capability k ->
        Fmt.pf fmt "capability %s carries a payload its key does not admit" k
    | `Over_limit o -> Me_limits.Over_limit.pp fmt o
    | #Me_flow.error as e -> Me_flow.pp_error fmt e

  (* --- an index over the graph payload, built once --- *)

  module Graph_index = struct
    type entry = {
      collection : string;
      nodes : (string, int) Hashtbl.t;  (** node id -> output slot count *)
    }

    let of_session s =
      let open Err.Syntax in
      let graphs = Hashtbl.create 16 in
      let+ () =
        Err.List.iter
          (fun (c : Model_explorer.GraphCollection.t) ->
            let collection = c.Model_explorer.GraphCollection.label in
            Err.List.iter
              (fun (g : Model_explorer.Graph.t) ->
                if Hashtbl.mem graphs g.Model_explorer.Graph.id then
                  Err.fail (`Duplicate_graph g.Model_explorer.Graph.id)
                else begin
                  let nodes = Hashtbl.create 64 in
                  let* () =
                    Err.List.iter
                      (fun (n : Model_explorer.GraphNode.t) ->
                        let id = n.Model_explorer.GraphNode.id in
                        if Hashtbl.mem nodes id then
                          Err.fail
                            (`Duplicate_node
                               { graph = g.Model_explorer.Graph.id; node = id })
                        else begin
                          Hashtbl.replace nodes id
                            (List.length
                               (Option.value
                                  n.Model_explorer.GraphNode.outputsMetadata
                                  ~default:[]));
                          Err.return ()
                        end)
                      g.Model_explorer.Graph.nodes
                  in
                  Hashtbl.add graphs g.Model_explorer.Graph.id
                    { collection; nodes };
                  Err.return ()
                end)
              c.Model_explorer.GraphCollection.graphs)
          s.graph_collections
      in
      graphs

    let graph t id =
      match Hashtbl.find_opt t id with
      | Some g -> Err.return g
      | None -> Err.fail (`Unknown_graph id)

    (* Same as [graph], plus the check a bare id lookup cannot make: that the
       DECLARING side (a view, a pane) named the collection the graph is
       actually in, not merely a graph id that happens to exist somewhere. *)
    let graph_in t ~collection id =
      let open Err.Syntax in
      let* g = graph t id in
      if g.collection = collection then Err.return g
      else Err.fail (`Wrong_collection { graph = id; collection })

    let node t ~graph:gid id =
      let open Err.Syntax in
      let* g = graph t gid in
      match Hashtbl.find_opt g.nodes id with
      | Some slots -> Err.return slots
      | None -> Err.fail (`Unknown_node { graph = gid; node = id })
  end

  let validate ~limits s =
    let open Err.Syntax in
    let count = Me_limits.check ~scope:Me_limits.Scope.Session in
    (* [max_total_nodes]/[max_total_edges] are [int64] fields -- a real model's
       graphs cannot approach that domain, but this validator's whole point is
       an UNTRUSTED document, and a sum accumulated in [int] would be exactly
       the kind of narrowing this codebase does not do. Checked before each
       addition, not after the fold: the point of the ceiling is to stop
       accumulating, not to notice afterwards that it was crossed. *)
    let count64 field total ~ceiling =
      let+ () =
        Me_limits.check64 ~scope:Me_limits.Scope.Session field total ~ceiling
      in
      total
    in
    let all_graphs =
      List.concat_map
        (fun (c : Model_explorer.GraphCollection.t) ->
          c.Model_explorer.GraphCollection.graphs)
        s.graph_collections
    in
    (* --- aggregates first, before the walks that are linear in them --- *)
    let* () =
      let* () =
        count Me_limits.Field.Views (List.length s.views)
          ~ceiling:limits.Me_limits.Limits.max_views
      in
      let* () =
        count Me_limits.Field.Comparisons
          (List.length s.comparisons)
          ~ceiling:limits.Me_limits.Limits.max_comparisons
      in
      let* () =
        count Me_limits.Field.Node_data_sets
          (List.length s.node_data_sets)
          ~ceiling:limits.Me_limits.Limits.max_node_data_sets
      in
      let* () =
        count Me_limits.Field.Diagnostics
          (List.length s.diagnostics)
          ~ceiling:limits.Me_limits.Limits.max_diagnostics
      in
      let* () =
        count Me_limits.Field.Graphs (List.length all_graphs)
          ~ceiling:limits.Me_limits.Limits.max_graphs
      in
      let* _ =
        Err.List.fold_left
          (fun (nodes, edges) (g : Model_explorer.Graph.t) ->
            let* nodes =
              count64 Me_limits.Field.Total_nodes
                (Int64.add nodes
                   (Int64.of_int (List.length g.Model_explorer.Graph.nodes)))
                ~ceiling:limits.Me_limits.Limits.max_total_nodes
            in
            let+ edges =
              count64 Me_limits.Field.Total_edges
                (Int64.add edges
                   (Int64.of_int
                      (List.fold_left
                         (fun acc (n : Model_explorer.GraphNode.t) ->
                           acc
                           + List.length
                               (Option.value
                                  n.Model_explorer.GraphNode.incomingEdges
                                  ~default:[]))
                         0 g.Model_explorer.Graph.nodes)))
                ~ceiling:limits.Me_limits.Limits.max_total_edges
            in
            (nodes, edges))
          (0L, 0L) all_graphs
      in
      (* Every overlay's edges, over every comparison and both sides -- the
         per-overlay ceiling is [Me_fusion]'s own, checked when an overlay is
         built; nothing sums what an untrusted document could claim across all
         of them. *)
      let+ _ =
        Err.List.fold_left
          (fun total (c : Comparison.t) ->
            let total =
              total
              + List.fold_left
                  (fun acc (o : Model_explorer.EdgeOverlay.t) ->
                    acc + List.length o.Model_explorer.EdgeOverlay.edges)
                  0
                  (c.Comparison.overlays_left @ c.Comparison.overlays_right)
            in
            let+ () =
              count Me_limits.Field.Overlay_edges_total total
                ~ceiling:limits.Me_limits.Limits.max_overlay_edges_total
            in
            total)
          0 s.comparisons
      in
      ()
    in
    let* graphs = Graph_index.of_session s in
    (* --- every edge resolves with a matching output slot, and every node
       stays under its per-node ceilings --- *)
    let namespace_depth ns =
      if ns = "" then 0 else List.length (String.split_on_char '/' ns)
    in
    let* () =
      Err.List.iter
        (fun (c : Model_explorer.GraphCollection.t) ->
          Err.List.iter
            (fun (g : Model_explorer.Graph.t) ->
              let gid = g.Model_explorer.Graph.id in
              let* () =
                Err.List.iter
                  (fun sub ->
                    Err.map_error
                      (fun e -> e)
                      (let+ _ = Graph_index.graph graphs sub in
                       ()))
                  (Option.value g.Model_explorer.Graph.subGraphIds ~default:[])
              in
              Err.List.iter
                (fun (n : Model_explorer.GraphNode.t) ->
                  let* () =
                    count Me_limits.Field.Attrs_per_node
                      (List.length
                         (Option.value n.Model_explorer.GraphNode.attrs
                            ~default:[]))
                      ~ceiling:limits.Me_limits.Limits.max_attrs_per_node
                  in
                  let* () =
                    count Me_limits.Field.Metadata_items_per_node
                      (List.length
                         (Option.value n.Model_explorer.GraphNode.inputsMetadata
                            ~default:[]))
                      ~ceiling:
                        limits.Me_limits.Limits.max_metadata_items_per_node
                  in
                  let* () =
                    count Me_limits.Field.Outputs_metadata_per_node
                      (List.length
                         (Option.value
                            n.Model_explorer.GraphNode.outputsMetadata
                            ~default:[]))
                      ~ceiling:
                        limits.Me_limits.Limits.max_outputs_metadata_per_node
                  in
                  let* () =
                    count Me_limits.Field.Namespace_depth
                      (namespace_depth n.Model_explorer.GraphNode.namespace)
                      ~ceiling:limits.Me_limits.Limits.max_namespace_depth
                  in
                  Err.List.iter
                    (fun (e : Model_explorer.IncomingEdge.t) ->
                      let src = e.Model_explorer.IncomingEdge.sourceNodeId in
                      let* slots = Graph_index.node graphs ~graph:gid src in
                      (* The slot has to EXIST, not merely be named: an edge
                         pointing at output 3 of a node with one output renders
                         as a connection that carries nothing. *)
                      match
                        int_of_string_opt
                          e.Model_explorer.IncomingEdge.sourceNodeOutputId
                      with
                      (* The slot must actually EXIST, not merely be
                         nonnegative: zero [outputsMetadata] means zero
                         slots, and an edge naming any index into that is a
                         connection that carries nothing. *)
                      | Some i when i >= 0 && i < slots -> Err.return ()
                      | _ ->
                          Err.fail (`Slot_mismatch { graph = gid; node = src }))
                    (Option.value n.Model_explorer.GraphNode.incomingEdges
                       ~default:[]))
                g.Model_explorer.Graph.nodes)
            c.Model_explorer.GraphCollection.graphs)
        s.graph_collections
    in
    (* --- views --- *)
    let view_ids = Hashtbl.create 16 in
    let* () =
      Err.List.iter
        (fun (v : View.t) ->
          if Hashtbl.mem view_ids v.View.id then
            Err.fail (`Duplicate_view v.View.id)
          else begin
            Hashtbl.add view_ids v.View.id v;
            let+ _ =
              Graph_index.graph_in graphs ~collection:v.View.collection
                v.View.graph
            in
            ()
          end)
        s.views
    in
    let* () =
      if Hashtbl.mem view_ids s.default_view then Err.return ()
      else Err.fail (`Unknown_view s.default_view)
    in
    (* --- comparisons --- *)
    let comparisons = Hashtbl.create 16 in
    let* () =
      Err.List.iter
        (fun (c : Comparison.t) ->
          if Hashtbl.mem comparisons c.Comparison.id then
            Err.fail (`Duplicate_comparison c.Comparison.id)
          else begin
            Hashtbl.add comparisons c.Comparison.id c;
            let* () =
              count Me_limits.Field.Mapping_entries_per_comparison
                (List.length c.Comparison.sync.Sync_navigation.entries)
                ~ceiling:
                  limits.Me_limits.Limits.max_mapping_entries_per_comparison
            in
            let* _ =
              Graph_index.graph_in graphs
                ~collection:c.Comparison.left.Pane_state.collection
                c.Comparison.left.Pane_state.graph
            in
            let* _ =
              Graph_index.graph_in graphs
                ~collection:c.Comparison.right.Pane_state.collection
                c.Comparison.right.Pane_state.graph
            in
            (* At most one entry per node PER SIDE, scoped to THIS comparison.
               Not globally: canonical Native is the left pane of two
               comparisons, so its nodes legitimately appear on the left of two
               mapping sets, and a global check would reject the branch the
               flow spine exists to show. *)
            let seen = Hashtbl.create 64 in
            Err.List.iter
              (fun (e : Mapping_entry.t) ->
                Err.List.iter
                  (fun (side, ids, pane_graph) ->
                    Err.List.iter
                      (fun id ->
                        let k = side ^ "\000" ^ id in
                        if Hashtbl.mem seen k then
                          Err.fail
                            (`Node_in_two_entries
                               { comparison = c.Comparison.id; node = id })
                        else begin
                          Hashtbl.add seen k ();
                          (* A mapping-entry member names a node, not merely a
                             string: an id nobody's pane graph holds validates
                             today and navigates nowhere. *)
                          let+ _ =
                            Graph_index.node graphs ~graph:pane_graph id
                          in
                          ()
                        end)
                      ids)
                  [
                    ( "l",
                      e.Mapping_entry.left,
                      c.Comparison.left.Pane_state.graph );
                    ( "r",
                      e.Mapping_entry.right,
                      c.Comparison.right.Pane_state.graph );
                  ])
              c.Comparison.sync.Sync_navigation.entries
          end)
        s.comparisons
    in
    (* --- node data, graph-addressed and node-keyed --- *)
    let* () =
      Err.List.iter
        (fun (d : Node_data_set.t) ->
          let* () =
            count Me_limits.Field.Node_data_results
              (List.length d.Node_data_set.results)
              ~ceiling:limits.Me_limits.Limits.max_node_data_results_per_graph
          in
          Err.List.iter
            (fun (node, _) ->
              let+ _ =
                Graph_index.node graphs ~graph:d.Node_data_set.graph node
              in
              ())
            d.Node_data_set.results)
        s.node_data_sets
    in
    (* --- capabilities: every key exactly once, with a compatible payload --- *)
    let seen_key = Hashtbl.create 16 in
    let* () =
      Err.List.iter
        (fun (c : Capability.t) ->
          let name = Capability.key_name c.Capability.key in
          if Hashtbl.mem seen_key name then
            Err.fail (`Duplicate_capability name)
          else begin
            Hashtbl.add seen_key name ();
            if Capability.compatible c then Err.return ()
            else Err.fail (`Incompatible_capability name)
          end)
        s.capabilities
    in
    let* () =
      Err.List.iter
        (fun k ->
          let name = Capability.key_name k in
          if Hashtbl.mem seen_key name then Err.return ()
          else Err.fail (`Missing_capability name))
        Capability.all_keys
    in
    (* --- the flow, and the half of it only this scope can check --- *)
    (* A flow DESTINATION is three facts that must name one graph: the spine's
       own [graph], the [View.Flow] that opens it, and the [feature:flow]
       capability that advertises it. Nothing tied them together before, so a
       document could pass with a spine whose graph no collection held, a view
       pointing elsewhere, or a capability naming a third graph -- and the
       browser would have been the first thing to notice. *)
    let flow_views =
      List.filter (fun (v : View.t) -> v.View.kind = View.Flow) s.views
    in
    let flow_capability_graph =
      List.fold_left
        (fun acc (c : Capability.t) ->
          match (c.Capability.key, c.Capability.status) with
          | ( Capability.Feature Capability.Flow,
              Capability.Available (Capability.Graph g) ) ->
              Some g
          | _ -> acc)
        None s.capabilities
    in
    let disagrees part declared named =
      Err.fail
        (`Flow_destination_disagrees { Flow_destination.part; declared; named })
    in
    match s.flow with
    | None -> (
        (* No spine, so neither of the other two may claim one. A capability
           alone never creates a destination. *)
        let* () =
          match flow_views with
          | [] -> Err.return ()
          | v :: _ -> Err.fail (`Unexpected_flow_view v.View.id)
        in
        match flow_capability_graph with
        | None -> Err.return ()
        | Some g -> Err.fail (`Unexpected_flow_capability g))
    | Some flow ->
        let* () = Me_flow.validate ~limits flow in
        let declared = flow.Me_flow.graph in
        (* The view first: it is what names the collection the graph must be
           found in, so there is nothing to resolve the graph against until it
           is known to be unique. *)
        let* flow_view =
          match flow_views with
          | [ v ] -> Err.return v
          | [] -> disagrees Flow_destination.Flow_view declared None
          | _ :: v :: _ -> Err.fail (`Duplicate_flow_view v.View.id)
        in
        let* () =
          if flow_view.View.graph = declared then Err.return ()
          else
            disagrees Flow_destination.Flow_view declared
              (Some flow_view.View.graph)
        in
        (* In the collection the view declares -- a graph id that exists in some
           OTHER collection is the same defect [`Wrong_collection] names for a
           view or a pane, and must be one here too. *)
        let* _ =
          Graph_index.graph_in graphs ~collection:flow_view.View.collection
            declared
        in
        let* () =
          match flow_capability_graph with
          | Some g when g = declared -> Err.return ()
          | other -> disagrees Flow_destination.Flow_capability declared other
        in
        (* Every state opens an EXPLICIT view. Checked here and not in
           [Me_flow.validate] for the same reason [Transition.comparison] is:
           that module cannot see the view table. Resolving is not enough --
           a [flow] or [compare] view is not somewhere a state can be opened,
           and a stage view over some other graph would silently show the wrong
           representation for this point in the spine. *)
        let* () =
          Err.List.iter
            (fun (st : Me_flow.State.t) ->
              let state = st.Me_flow.State.id
              and view = st.Me_flow.State.view in
              match Hashtbl.find_opt view_ids view with
              | None ->
                  Err.fail (`Flow_view_unknown { Flow_state_view.state; view })
              | Some (v : View.t) -> (
                  match v.View.kind with
                  | View.Flow | View.Compare ->
                      Err.fail
                        (`Flow_view_not_stage { Flow_state_view.state; view })
                  | View.Stage _ ->
                      if v.View.graph = st.Me_flow.State.graph then
                        Err.return ()
                      else
                        Err.fail
                          (`Flow_view_graph_disagrees
                             {
                               Flow_view_graph.state;
                               view;
                               state_graph = st.Me_flow.State.graph;
                               view_graph = v.View.graph;
                             })))
            flow.Me_flow.states
        in
        let state_graph =
          let t = Hashtbl.create 16 in
          List.iter
            (fun (st : Me_flow.State.t) ->
              Hashtbl.replace t st.Me_flow.State.id st.Me_flow.State.graph)
            flow.Me_flow.states;
          t
        in
        Err.List.iter
          (fun (tr : Me_flow.Transition.t) ->
            match tr.Me_flow.Transition.comparison with
            | None -> Err.return ()
            | Some cid -> (
                match Hashtbl.find_opt comparisons cid with
                | None -> Err.fail (`Unknown_comparison cid)
                | Some c ->
                    (* Resolving is not enough. A comparison naming two
                       unrelated graphs resolves perfectly and shows the wrong
                       diff, so the panes must be the transition's own
                       endpoints. *)
                    let before =
                      Hashtbl.find_opt state_graph tr.Me_flow.Transition.before
                    and after =
                      Hashtbl.find_opt state_graph tr.Me_flow.Transition.after
                    in
                    if
                      before = Some c.Comparison.left.Pane_state.graph
                      && after = Some c.Comparison.right.Pane_state.graph
                    then Err.return ()
                    else
                      Err.fail
                        (`Comparison_panes_disagree tr.Me_flow.Transition.id)))
          flow.Me_flow.transitions

  (* --- the wire --- *)
  let jsont =
    Jsont.Object.map ~kind:"visualization_session"
      (fun
        schema_version
        producer
        model
        graph_collections
        views
        comparisons
        node_data_sets
        flow
        capabilities
        diagnostics
        default_view
      ->
        {
          schema_version;
          producer;
          model;
          graph_collections;
          views;
          comparisons;
          node_data_sets;
          flow;
          capabilities;
          diagnostics;
          default_view;
        })
    |> Jsont.Object.mem "schemaVersion" Jsont.int ~enc:(fun s ->
        s.schema_version)
    |> Jsont.Object.mem "producer" Producer.jsont ~enc:(fun s -> s.producer)
    |> Jsont.Object.mem "model" Model_summary.jsont ~enc:(fun s -> s.model)
    |> Jsont.Object.mem "graphCollections"
         (Jsont.list Model_explorer.GraphCollection.jsont) ~enc:(fun s ->
           s.graph_collections)
    |> Jsont.Object.mem "views" (Jsont.list View.jsont) ~enc:(fun s -> s.views)
    |> Jsont.Object.mem "comparisons" (Jsont.list Comparison.jsont)
         ~enc:(fun s -> s.comparisons)
    |> Jsont.Object.mem "nodeDataSets" (Jsont.list Node_data_set.jsont)
         ~enc:(fun s -> s.node_data_sets)
    (* The spine is DATA on the wire, not only a rendered graph: [flow_nav.js]
       maps a selected node id to a [Transition.id] to its [Comparison], and
       [Transition.comparison] exists nowhere else. Encoding the bipartite
       graph alone would leave every transition node unclickable. *)
    |> Jsont.Object.opt_mem "flow" Me_flow.jsont ~enc:(fun s -> s.flow)
    |> Jsont.Object.mem "capabilities" (Jsont.list Capability.jsont)
         ~enc:(fun s -> s.capabilities)
    |> Jsont.Object.mem "diagnostics" (Jsont.list Me_limits.Diagnostic.jsont)
         ~enc:(fun s -> s.diagnostics)
    |> Jsont.Object.mem "defaultView" Jsont.string ~enc:(fun s ->
        s.default_view)
    |> Jsont.Object.finish
end

module Runtime = struct
  type t = { epoch : string; limits : Me_limits.Limits.t; session : Session.t }
end
