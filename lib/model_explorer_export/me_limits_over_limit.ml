(* The over-limit domain (Scope/Field/Over_limit) and its [check]/[check64]
   entry points, split from me_limits.ml. Standalone: used by
   validators elsewhere in the tree (Me_session, Me_flow, ...), not by the
   rest of Me_limits itself. *)

(* --- the over-limit domain, owned here rather than per validator --- *)

module Scope = struct
  type t =
    | Detail
    | Flow
    | Fusion
    | Graph
    | Navigation
    | Session
    | Source_graph
    | Value_graph
    | Verification

  let to_string = function
    | Detail -> "detail"
    | Flow -> "flow"
    | Fusion -> "fusion"
    | Graph -> "graph"
    | Navigation -> "navigation"
    | Session -> "session"
    | Source_graph -> "source graph"
    | Value_graph -> "value graph"
    | Verification -> "verification"

  (* This successor relation defines the stable traversal exported by [all]. *)
  let next = function
    | Session -> Some Graph
    | Graph -> Some Value_graph
    | Value_graph -> Some Source_graph
    | Source_graph -> Some Detail
    | Detail -> Some Navigation
    | Navigation -> Some Fusion
    | Fusion -> Some Flow
    | Flow -> Some Verification
    | Verification -> None

  let all =
    let rec walk acc c =
      match next c with
      | None -> List.rev (c :: acc)
      | Some n -> walk (c :: acc) n
    in
    walk [] Session
end

module Field = struct
  type t =
    | Attrs_per_node
    | Comparisons
    | Detail_graphs
    | Detail_nodes
    | Diagnostics
    | Edges
    | Expression_nodes
    | Graphs
    | Group_node_attributes
    | Mapping_entries_per_comparison
    | Mapping_members
    | Mapping_members_per_entry
    | Metadata_items_per_node
    | Namespace_depth
    | Node_data_results
    | Node_data_sets
    | Nodes
    | Outputs_metadata_per_node
    | Overlay_edges
    | Overlay_edges_total
    | States
    | Total_edges
    | Total_nodes
    | Transitions
    | Views

  let to_string = function
    | Attrs_per_node -> "attrsPerNode"
    | Comparisons -> "comparisons"
    | Detail_graphs -> "detailGraphs"
    | Detail_nodes -> "detailNodes"
    | Diagnostics -> "diagnostics"
    | Edges -> "edges"
    | Expression_nodes -> "expressionNodes"
    | Graphs -> "graphs"
    | Group_node_attributes -> "groupNodeAttributes"
    | Mapping_entries_per_comparison -> "mappingEntriesPerComparison"
    | Mapping_members -> "mappingMembers"
    | Mapping_members_per_entry -> "mappingMembersPerEntry"
    | Metadata_items_per_node -> "metadataItemsPerNode"
    | Namespace_depth -> "namespaceDepth"
    | Node_data_results -> "nodeDataResults"
    | Node_data_sets -> "nodeDataSets"
    | Nodes -> "nodes"
    | Outputs_metadata_per_node -> "outputsMetadataPerNode"
    | Overlay_edges -> "overlayEdges"
    | Overlay_edges_total -> "overlayEdgesTotal"
    | States -> "states"
    | Total_edges -> "totalEdges"
    | Total_nodes -> "totalNodes"
    | Transitions -> "transitions"
    | Views -> "views"

  (* This successor relation defines the stable traversal exported by [all]. *)
  let next = function
    | Views -> Some Comparisons
    | Comparisons -> Some Node_data_sets
    | Node_data_sets -> Some Diagnostics
    | Diagnostics -> Some Graphs
    | Graphs -> Some Nodes
    | Nodes -> Some Edges
    | Edges -> Some Total_nodes
    | Total_nodes -> Some Total_edges
    | Total_edges -> Some Attrs_per_node
    | Attrs_per_node -> Some Metadata_items_per_node
    | Metadata_items_per_node -> Some Outputs_metadata_per_node
    | Outputs_metadata_per_node -> Some Namespace_depth
    | Namespace_depth -> Some Mapping_entries_per_comparison
    | Mapping_entries_per_comparison -> Some Mapping_members
    | Mapping_members -> Some Mapping_members_per_entry
    | Mapping_members_per_entry -> Some States
    | States -> Some Transitions
    | Transitions -> Some Detail_graphs
    | Detail_graphs -> Some Detail_nodes
    | Detail_nodes -> Some Expression_nodes
    | Expression_nodes -> Some Group_node_attributes
    | Group_node_attributes -> Some Node_data_results
    | Node_data_results -> Some Overlay_edges
    | Overlay_edges -> Some Overlay_edges_total
    | Overlay_edges_total -> None

  let all =
    let rec walk acc c =
      match next c with
      | None -> List.rev (c :: acc)
      | Some n -> walk (c :: acc) n
    in
    walk [] Views
end

module Over_limit = struct
  type t = { scope : Scope.t; field : Field.t; count : int64 }

  let pp fmt { scope; field; count } =
    Fmt.pf fmt "%s %s = %Ld is over the ceiling" (Scope.to_string scope)
      (Field.to_string field) count
end

type over_limit_error = [ `Over_limit of Over_limit.t ]

let over_limit ~scope field count =
  Err.fail (`Over_limit { Over_limit.scope; field; count })

let check ~scope field n ~ceiling =
  if n > ceiling then over_limit ~scope field (Int64.of_int n)
  else Err.return ()

let check64 ~scope field n ~ceiling =
  if Int64.compare n ceiling > 0 then over_limit ~scope field n
  else Err.return ()
