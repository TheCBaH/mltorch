(* The small pane/comparison/flow wire types: Producer, Model_summary,
   Pane_state, Mapping_entry, Sync_navigation, Flow_state_view,
   Flow_view_graph, Flow_destination, Comparison, Node_data_set. Split out
   of me_session.ml; see me_session_capability.ml for
   Capability/View and me_session_document.ml for the Session document
   itself. *)

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
  type source_kind = Json | Pt2

  type t = {
    name : string;
    source_kind : source_kind;
    source_bytes : int64;
    source_sha256 : string option;
    pt2_graph_count : int;
    op_targets : int;
  }

  let source_kind_jsont =
    Jsont.enum ~kind:"source_kind" [ ("json", Json); ("pt2", Pt2) ]

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
  type part = Flow_capability | Flow_view
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
