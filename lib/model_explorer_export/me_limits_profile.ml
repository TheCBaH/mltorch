(* The tunable [Limits.t] profile: fields, validation against [Hard],
   derivation of [response_live_bytes], and the four built-in profiles
   (trusted/untrusted/small/large). Split from me_limits.ml.
   Depends on me_limits_error.ml and me_limits_hard.ml; not published
   under a [.mli] of its own, so every field/value here stays internally
   transparent -- only the outer me_limits.mli makes [Limits.t] private. *)

open Me_limits_error
open Me_limits_hard

(* --- profiles --- *)

module Limits = struct
  type t = {
    max_json_bytes : int64;
    max_pt2_bytes : int64;
    zip : Pt2_zip.Limits.t;
    max_nodes_per_graph : int;
    max_edges_per_graph : int;
    max_groups_per_graph : int;
    max_attrs_per_node : int;
    max_metadata_items_per_node : int;
    max_outputs_metadata_per_node : int;
    max_namespace_depth : int;
    max_namespace_component_bytes : int;
    max_label_bytes : int;
    max_id_bytes : int;
    max_attr_chars : int;
    max_url_bytes : int;
    max_graphs : int;
    max_total_nodes : int64;
    max_total_edges : int64;
    max_views : int;
    max_comparisons : int;
    max_node_data_sets : int;
    max_states : int;
    max_transitions : int;
    max_mapping_entries_per_comparison : int;
    max_mapping_members_per_entry : int;
    max_mapping_members_total : int;
    max_node_data_results_per_graph : int;
    max_overlay_edges_per_overlay : int;
    max_overlay_edges_total : int;
    max_diagnostics : int;
    max_diagnostic_bytes : int;
    max_session_bytes : int;
    max_trace_entries : int;
    max_audit_reports : int;
    max_detail_nodes : int;
    max_detail_graphs : int;
    max_detail_bytes : int;
    response_live_bytes : int64;
  }

  (* Every tunable field, once. [create] checks each against its [Hard]
     ceiling and [Wire_limits.of_limits] compares each against the same field
     of a ceiling profile, both reading through [get] rather than by name — so
     the two cannot come to check different sets. Widths unify at [int64]:
     [Int64.of_int] is total, and comparing there means the 32-bit backend
     never narrows a ceiling before testing it. *)
  type field = {
    name : string;
    get : t -> int64;
    hard : int64;
    assign : t -> int64 -> t;
        (* Narrows an [int] field, and is therefore sound only after the value
           has been bounded. [assign_checked] below is the only caller and does
           that bounding; nothing else may use this directly. A wire value of
           2^40 for an [int] field would otherwise wrap under jsoo before
           anything looked at it -- CLAUDE.md's rule, in the one place this
           record could break it. *)
  }

  let i = Int64.of_int

  let fields =
    [
      {
        name = "max_json_bytes";
        get = (fun t -> t.max_json_bytes);
        hard = Hard.max_json_bytes;
        assign = (fun t v -> { t with max_json_bytes = v });
      };
      {
        name = "max_pt2_bytes";
        get = (fun t -> t.max_pt2_bytes);
        hard = Hard.max_pt2_bytes;
        assign = (fun t v -> { t with max_pt2_bytes = v });
      };
      {
        name = "max_nodes_per_graph";
        get = (fun t -> i t.max_nodes_per_graph);
        hard = i Hard.max_nodes_per_graph;
        assign = (fun t v -> { t with max_nodes_per_graph = Int64.to_int v });
      };
      {
        name = "max_edges_per_graph";
        get = (fun t -> i t.max_edges_per_graph);
        hard = i Hard.max_edges_per_graph;
        assign = (fun t v -> { t with max_edges_per_graph = Int64.to_int v });
      };
      {
        name = "max_groups_per_graph";
        get = (fun t -> i t.max_groups_per_graph);
        hard = i Hard.max_groups_per_graph;
        assign = (fun t v -> { t with max_groups_per_graph = Int64.to_int v });
      };
      {
        name = "max_attrs_per_node";
        get = (fun t -> i t.max_attrs_per_node);
        hard = i Hard.max_attrs_per_node;
        assign = (fun t v -> { t with max_attrs_per_node = Int64.to_int v });
      };
      {
        name = "max_metadata_items_per_node";
        get = (fun t -> i t.max_metadata_items_per_node);
        hard = i Hard.max_metadata_items_per_node;
        assign =
          (fun t v -> { t with max_metadata_items_per_node = Int64.to_int v });
      };
      {
        name = "max_outputs_metadata_per_node";
        get = (fun t -> i t.max_outputs_metadata_per_node);
        hard = i Hard.max_outputs_metadata_per_node;
        assign =
          (fun t v -> { t with max_outputs_metadata_per_node = Int64.to_int v });
      };
      {
        name = "max_namespace_depth";
        get = (fun t -> i t.max_namespace_depth);
        hard = i Hard.max_namespace_depth;
        assign = (fun t v -> { t with max_namespace_depth = Int64.to_int v });
      };
      {
        name = "max_namespace_component_bytes";
        get = (fun t -> i t.max_namespace_component_bytes);
        hard = i Hard.max_namespace_component_bytes;
        assign =
          (fun t v -> { t with max_namespace_component_bytes = Int64.to_int v });
      };
      {
        name = "max_label_bytes";
        get = (fun t -> i t.max_label_bytes);
        hard = i Hard.max_label_bytes;
        assign = (fun t v -> { t with max_label_bytes = Int64.to_int v });
      };
      {
        name = "max_id_bytes";
        get = (fun t -> i t.max_id_bytes);
        hard = i Hard.max_id_bytes;
        assign = (fun t v -> { t with max_id_bytes = Int64.to_int v });
      };
      {
        name = "max_attr_chars";
        get = (fun t -> i t.max_attr_chars);
        hard = i Hard.max_attr_chars;
        assign = (fun t v -> { t with max_attr_chars = Int64.to_int v });
      };
      {
        name = "max_url_bytes";
        get = (fun t -> i t.max_url_bytes);
        hard = i Hard.max_url_bytes;
        assign = (fun t v -> { t with max_url_bytes = Int64.to_int v });
      };
      {
        name = "max_graphs";
        get = (fun t -> i t.max_graphs);
        hard = i Hard.max_graphs;
        assign = (fun t v -> { t with max_graphs = Int64.to_int v });
      };
      {
        name = "max_total_nodes";
        get = (fun t -> t.max_total_nodes);
        hard = Hard.max_total_nodes;
        assign = (fun t v -> { t with max_total_nodes = v });
      };
      {
        name = "max_total_edges";
        get = (fun t -> t.max_total_edges);
        hard = Hard.max_total_edges;
        assign = (fun t v -> { t with max_total_edges = v });
      };
      {
        name = "max_views";
        get = (fun t -> i t.max_views);
        hard = i Hard.max_views;
        assign = (fun t v -> { t with max_views = Int64.to_int v });
      };
      {
        name = "max_comparisons";
        get = (fun t -> i t.max_comparisons);
        hard = i Hard.max_comparisons;
        assign = (fun t v -> { t with max_comparisons = Int64.to_int v });
      };
      {
        name = "max_node_data_sets";
        get = (fun t -> i t.max_node_data_sets);
        hard = i Hard.max_node_data_sets;
        assign = (fun t v -> { t with max_node_data_sets = Int64.to_int v });
      };
      {
        name = "max_states";
        get = (fun t -> i t.max_states);
        hard = i Hard.max_states;
        assign = (fun t v -> { t with max_states = Int64.to_int v });
      };
      {
        name = "max_transitions";
        get = (fun t -> i t.max_transitions);
        hard = i Hard.max_transitions;
        assign = (fun t v -> { t with max_transitions = Int64.to_int v });
      };
      {
        name = "max_mapping_entries_per_comparison";
        get = (fun t -> i t.max_mapping_entries_per_comparison);
        hard = i Hard.max_mapping_entries_per_comparison;
        assign =
          (fun t v ->
            { t with max_mapping_entries_per_comparison = Int64.to_int v });
      };
      {
        name = "max_mapping_members_per_entry";
        get = (fun t -> i t.max_mapping_members_per_entry);
        hard = i Hard.max_mapping_members_per_entry;
        assign =
          (fun t v -> { t with max_mapping_members_per_entry = Int64.to_int v });
      };
      {
        name = "max_mapping_members_total";
        get = (fun t -> i t.max_mapping_members_total);
        hard = i Hard.max_mapping_members_total;
        assign =
          (fun t v -> { t with max_mapping_members_total = Int64.to_int v });
      };
      {
        name = "max_node_data_results_per_graph";
        get = (fun t -> i t.max_node_data_results_per_graph);
        hard = i Hard.max_node_data_results_per_graph;
        assign =
          (fun t v ->
            { t with max_node_data_results_per_graph = Int64.to_int v });
      };
      {
        name = "max_overlay_edges_per_overlay";
        get = (fun t -> i t.max_overlay_edges_per_overlay);
        hard = i Hard.max_overlay_edges_per_overlay;
        assign =
          (fun t v -> { t with max_overlay_edges_per_overlay = Int64.to_int v });
      };
      {
        name = "max_overlay_edges_total";
        get = (fun t -> i t.max_overlay_edges_total);
        hard = i Hard.max_overlay_edges_total;
        assign =
          (fun t v -> { t with max_overlay_edges_total = Int64.to_int v });
      };
      {
        name = "max_diagnostics";
        get = (fun t -> i t.max_diagnostics);
        hard = i Hard.max_diagnostics;
        assign = (fun t v -> { t with max_diagnostics = Int64.to_int v });
      };
      {
        name = "max_diagnostic_bytes";
        get = (fun t -> i t.max_diagnostic_bytes);
        hard = i Hard.max_diagnostic_bytes;
        assign = (fun t v -> { t with max_diagnostic_bytes = Int64.to_int v });
      };
      {
        name = "max_session_bytes";
        get = (fun t -> i t.max_session_bytes);
        hard = i Hard.max_session_bytes;
        assign = (fun t v -> { t with max_session_bytes = Int64.to_int v });
      };
      {
        name = "max_trace_entries";
        get = (fun t -> i t.max_trace_entries);
        hard = i Hard.max_trace_entries;
        assign = (fun t v -> { t with max_trace_entries = Int64.to_int v });
      };
      {
        name = "max_audit_reports";
        get = (fun t -> i t.max_audit_reports);
        hard = i Hard.max_audit_reports;
        assign = (fun t v -> { t with max_audit_reports = Int64.to_int v });
      };
      {
        name = "max_detail_nodes";
        get = (fun t -> i t.max_detail_nodes);
        hard = i Hard.max_detail_nodes;
        assign = (fun t v -> { t with max_detail_nodes = Int64.to_int v });
      };
      {
        name = "max_detail_graphs";
        get = (fun t -> i t.max_detail_graphs);
        hard = i Hard.max_detail_graphs;
        assign = (fun t v -> { t with max_detail_graphs = Int64.to_int v });
      };
      {
        name = "max_detail_bytes";
        get = (fun t -> i t.max_detail_bytes);
        hard = i Hard.max_detail_bytes;
        assign = (fun t v -> { t with max_detail_bytes = Int64.to_int v });
      };
    ]

  (* The nested archive profile. Its own ceilings are [Pt2_zip.Limits.trusted],
     which is that module's [Hard] made into a profile, so there is no second
     copy of them here. [allow_encrypted] is absent because it is not a
     ceiling: [Pt2_zip.Limits.create] pins it to [false] and no profile may set
     it. *)
  let zip_fields =
    [
      ( "max_entries",
        (fun (z : Pt2_zip.Limits.t) -> i z.max_entries),
        fun (z : Pt2_zip.Limits.t) v -> { z with max_entries = Int64.to_int v }
      );
      ( "max_entry_bytes",
        (fun (z : Pt2_zip.Limits.t) -> z.max_entry_bytes),
        fun (z : Pt2_zip.Limits.t) v -> { z with max_entry_bytes = v } );
      ( "max_total_bytes",
        (fun (z : Pt2_zip.Limits.t) -> z.max_total_bytes),
        fun (z : Pt2_zip.Limits.t) v -> { z with max_total_bytes = v } );
      ( "max_path_bytes",
        (fun (z : Pt2_zip.Limits.t) -> i z.max_path_bytes),
        fun (z : Pt2_zip.Limits.t) v ->
          { z with max_path_bytes = Int64.to_int v } );
      ( "max_path_depth",
        (fun (z : Pt2_zip.Limits.t) -> i z.max_path_depth),
        fun (z : Pt2_zip.Limits.t) v ->
          { z with max_path_depth = Int64.to_int v } );
    ]

  let check name value ceiling =
    if Int64.compare value 0L <= 0 || Int64.compare value ceiling > 0 then
      Err.fail (`Invalid_limit { Invalid.name; value })
    else Err.return ()

  (* Against a ceiling supplied as a profile, so [create] (ceiling = the [Hard]
     figures) and [Wire_limits.of_limits] (ceiling = [untrusted]) run the same
     traversal over the same field set.

     ACCUMULATING, alone among this repo's validators, and the two conditions
     that make it safe are worth naming because neither holds for
     [Me_session.validate] or [Me_flow.validate]. The work is bounded by a
     compile-time constant -- [fields] and [zip_fields], 41 entries, not
     anything the document controls -- so running every element cannot be turned
     into a denial of service by the input. And each check is an independent
     [Int64.compare]: none is a precondition of another, so there is no ceiling
     that has to pass before the rest may run.

     [fold_errors] is what keeps this local. Without it, accumulating here would
     put an error LIST in the type of [validate], [create] and
     [Wire_limits.of_limits], i.e. on the wire; instead the batch collapses into
     one payload of this module's own domain, and only [pp_error] had to
     change. An operator with three bad fields learns about three. *)
  let check_against ~zip_ceiling ~field_ceiling t =
    Err.Accum.iter
      (fun (name, value, ceiling) -> check name value ceiling)
      (List.map (fun f -> (f.name, f.get t, field_ceiling f)) fields
      @ List.map
          (fun (name, get, _) -> ("zip." ^ name, get t.zip, get zip_ceiling))
          zip_fields)
    |> Err.Accum.fold_errors
         (fun (es : [ `Invalid_limit of Invalid.t ] Err.Accum.errors) ->
           match List.map Err.Error.kind es with
           (* One bad field stays the singular row: the plural form exists to
              report a SET, and would otherwise churn every existing message. *)
           | [ (`Invalid_limit _ as one) ] -> one
           | invalid ->
               `Invalid_limits (List.map (fun (`Invalid_limit i) -> i) invalid))

  let derive t =
    let open Err.Syntax in
    let* peak =
      response_live_bytes ~max_session_bytes:t.max_session_bytes
        ~max_detail_bytes:t.max_detail_bytes
    in
    let+ () =
      if Int64.compare peak Hard.jsoo_safe_bytes > 0 then
        Err.fail
          (`Invalid_limit { Invalid.name = "response_live_bytes"; value = peak })
      else Err.return ()
    in
    { t with response_live_bytes = peak }

  let validate t =
    let open Err.Syntax in
    let* () =
      check_against ~zip_ceiling:Pt2_zip.Limits.trusted
        ~field_ceiling:(fun f -> f.hard)
        t
    in
    derive t

  (* The relations a profile must satisfy before the BROWSER may read a
     document produced under it. Not part of [validate]: a profile is a valid
     profile without them — [trusted] is one, and satisfies neither — and
     folding them in would remove the only profiles that can prove the check is
     live. Asserted where each wire-selectable profile is constructed, so no
     document is ever acquired under a profile that could not fit.

     The first two are the load-bearing ones, and both can fail. The third is
     NOT independent of them and is not claimed to be: the peak is monotone in
     the document ceilings and [Hard.max_response_live_bytes] is that same peak
     evaluated at [max_response_document_bytes], so anything passing the first
     two passes the third by construction. It is kept because it states
     directly the property the other two are a means to, and it becomes live
     the moment a phase gains a term the document ceilings do not drive. The
     equality it rests on is pinned by a test, so that day is visible. *)
  let within_hard_response t =
    let open Err.Syntax in
    let* () =
      check "max_session_bytes" (i t.max_session_bytes)
        (i Hard.max_response_document_bytes)
    in
    let* () =
      check "max_detail_bytes" (i t.max_detail_bytes)
        (i Hard.max_response_document_bytes)
    in
    check "response_live_bytes" t.response_live_bytes
      Hard.max_response_live_bytes

  (* Apply ONE wire override, bounded before it is narrowed.
     [assign] is what narrows, so nothing may call it directly: a wire value of
     2^40 for an [int] field would wrap under jsoo before any check saw it, and
     a check on a wrapped value is not a bound (CLAUDE.md). The ceiling is a
     validated profile and so is itself no looser than [hard], which is why
     bounding against it bounds against [hard] too. *)
  let assign_checked ~ceiling t f v =
    let open Err.Syntax in
    let+ () = check f.name v (f.get ceiling) in
    f.assign t v

  let assign_zip_checked ~ceiling t (name, get, set) v =
    let open Err.Syntax in
    let+ () = check ("zip." ^ name) v (get ceiling.zip) in
    { t with zip = set t.zip v }

  (* The wire NAME of a field is its diagnostic name, dotted for a nested one --
     the same string [Invalid.t] carries. One name, so a rejection can be
     matched back to the member that caused it without a second table. *)
  let field_named name =
    match List.find_opt (fun f -> String.equal f.name name) fields with
    | Some f -> Some (`Field f)
    | None ->
        let prefixed = "zip." in
        if
          String.length name > String.length prefixed
          && String.sub name 0 (String.length prefixed) = prefixed
        then
          let bare =
            String.sub name (String.length prefixed)
              (String.length name - String.length prefixed)
          in
          Option.map
            (fun z -> `Zip z)
            (List.find_opt (fun (n, _, _) -> String.equal n bare) zip_fields)
        else None

  (* Every override the wire may carry, as the flat name/value pairs a profile
     DIFFERS from its base by. Absent means "as the base", which is what makes
     the base itself encode as [{}] and two equal profiles encode to equal
     bytes -- the determinism §4 quantifies over. *)
  let overrides ~base t =
    List.filter_map
      (fun f ->
        if Int64.equal (f.get t) (f.get base) then None
        else Some (f.name, f.get t))
      fields
    @ List.filter_map
        (fun (name, get, _) ->
          if Int64.equal (get t.zip) (get base.zip) then None
          else Some ("zip." ^ name, get t.zip))
        zip_fields

  let apply_overrides ~ceiling ~base pairs =
    Err.List.fold_left
      (fun t (name, v) ->
        match field_named name with
        | Some (`Field f) -> assign_checked ~ceiling t f v
        | Some (`Zip z) -> assign_zip_checked ~ceiling t z v
        | None -> Err.fail (`Invalid_limit { Invalid.name; value = v }))
      base pairs

  let create ?max_json_bytes ?max_pt2_bytes ?zip ?max_nodes_per_graph
      ?max_edges_per_graph ?max_groups_per_graph ?max_attrs_per_node
      ?max_metadata_items_per_node ?max_outputs_metadata_per_node
      ?max_namespace_depth ?max_namespace_component_bytes ?max_label_bytes
      ?max_id_bytes ?max_attr_chars ?max_url_bytes ?max_graphs ?max_total_nodes
      ?max_total_edges ?max_views ?max_comparisons ?max_node_data_sets
      ?max_states ?max_transitions ?max_mapping_entries_per_comparison
      ?max_mapping_members_per_entry ?max_mapping_members_total
      ?max_node_data_results_per_graph ?max_overlay_edges_per_overlay
      ?max_overlay_edges_total ?max_diagnostics ?max_diagnostic_bytes
      ?max_session_bytes ?max_trace_entries ?max_audit_reports ?max_detail_nodes
      ?max_detail_graphs ?max_detail_bytes base =
    let d v default = Option.value v ~default in
    validate
      {
        max_json_bytes = d max_json_bytes base.max_json_bytes;
        max_pt2_bytes = d max_pt2_bytes base.max_pt2_bytes;
        zip = d zip base.zip;
        max_nodes_per_graph = d max_nodes_per_graph base.max_nodes_per_graph;
        max_edges_per_graph = d max_edges_per_graph base.max_edges_per_graph;
        max_groups_per_graph = d max_groups_per_graph base.max_groups_per_graph;
        max_attrs_per_node = d max_attrs_per_node base.max_attrs_per_node;
        max_metadata_items_per_node =
          d max_metadata_items_per_node base.max_metadata_items_per_node;
        max_outputs_metadata_per_node =
          d max_outputs_metadata_per_node base.max_outputs_metadata_per_node;
        max_namespace_depth = d max_namespace_depth base.max_namespace_depth;
        max_namespace_component_bytes =
          d max_namespace_component_bytes base.max_namespace_component_bytes;
        max_label_bytes = d max_label_bytes base.max_label_bytes;
        max_id_bytes = d max_id_bytes base.max_id_bytes;
        max_attr_chars = d max_attr_chars base.max_attr_chars;
        max_url_bytes = d max_url_bytes base.max_url_bytes;
        max_graphs = d max_graphs base.max_graphs;
        max_total_nodes = d max_total_nodes base.max_total_nodes;
        max_total_edges = d max_total_edges base.max_total_edges;
        max_views = d max_views base.max_views;
        max_comparisons = d max_comparisons base.max_comparisons;
        max_node_data_sets = d max_node_data_sets base.max_node_data_sets;
        max_states = d max_states base.max_states;
        max_transitions = d max_transitions base.max_transitions;
        max_mapping_entries_per_comparison =
          d max_mapping_entries_per_comparison
            base.max_mapping_entries_per_comparison;
        max_mapping_members_per_entry =
          d max_mapping_members_per_entry base.max_mapping_members_per_entry;
        max_mapping_members_total =
          d max_mapping_members_total base.max_mapping_members_total;
        max_node_data_results_per_graph =
          d max_node_data_results_per_graph base.max_node_data_results_per_graph;
        max_overlay_edges_per_overlay =
          d max_overlay_edges_per_overlay base.max_overlay_edges_per_overlay;
        max_overlay_edges_total =
          d max_overlay_edges_total base.max_overlay_edges_total;
        max_diagnostics = d max_diagnostics base.max_diagnostics;
        max_diagnostic_bytes = d max_diagnostic_bytes base.max_diagnostic_bytes;
        max_session_bytes = d max_session_bytes base.max_session_bytes;
        max_trace_entries = d max_trace_entries base.max_trace_entries;
        max_audit_reports = d max_audit_reports base.max_audit_reports;
        max_detail_nodes = d max_detail_nodes base.max_detail_nodes;
        max_detail_graphs = d max_detail_graphs base.max_detail_graphs;
        max_detail_bytes = d max_detail_bytes base.max_detail_bytes;
        (* Overwritten by [derive]; the only way to obtain a [t] outside this
           module is through [create], so no caller can supply one. *)
        response_live_bytes = 0L;
      }

  (* The widest profile there is. Built as a literal because there is no
     profile to override yet, then put through the same [validate] as every
     other one.

     Every field sits at its [Hard] ceiling EXCEPT the two document sizes,
     which sit at half of theirs: the phase sums at 32MB come to roughly 1.6GB,
     so an all-at-[Hard] profile would not be constructible at all. That is the
     [Hard] ceilings doing their job rather than a compromise — a profile
     nobody can build is not a profile — and it is what leaves [create]'s
     rejection reachable from above. *)
  let trusted =
    Err.or_raise ~pp_error
      (validate
         {
           max_json_bytes = Hard.max_json_bytes;
           max_pt2_bytes = Hard.max_pt2_bytes;
           zip = Pt2_zip.Limits.trusted;
           max_nodes_per_graph = Hard.max_nodes_per_graph;
           max_edges_per_graph = Hard.max_edges_per_graph;
           max_groups_per_graph = Hard.max_groups_per_graph;
           max_attrs_per_node = Hard.max_attrs_per_node;
           max_metadata_items_per_node = Hard.max_metadata_items_per_node;
           max_outputs_metadata_per_node = Hard.max_outputs_metadata_per_node;
           max_namespace_depth = Hard.max_namespace_depth;
           max_namespace_component_bytes = Hard.max_namespace_component_bytes;
           max_label_bytes = Hard.max_label_bytes;
           max_id_bytes = Hard.max_id_bytes;
           max_attr_chars = Hard.max_attr_chars;
           max_url_bytes = Hard.max_url_bytes;
           max_graphs = Hard.max_graphs;
           max_total_nodes = Hard.max_total_nodes;
           max_total_edges = Hard.max_total_edges;
           max_views = Hard.max_views;
           max_comparisons = Hard.max_comparisons;
           max_node_data_sets = Hard.max_node_data_sets;
           max_states = Hard.max_states;
           max_transitions = Hard.max_transitions;
           max_mapping_entries_per_comparison =
             Hard.max_mapping_entries_per_comparison;
           max_mapping_members_per_entry = Hard.max_mapping_members_per_entry;
           max_mapping_members_total = Hard.max_mapping_members_total;
           max_node_data_results_per_graph =
             Hard.max_node_data_results_per_graph;
           max_overlay_edges_per_overlay = Hard.max_overlay_edges_per_overlay;
           max_overlay_edges_total = Hard.max_overlay_edges_total;
           max_diagnostics = Hard.max_diagnostics;
           max_diagnostic_bytes = Hard.max_diagnostic_bytes;
           max_session_bytes = 0x100_0000;
           max_trace_entries = Hard.max_trace_entries;
           max_audit_reports = Hard.max_audit_reports;
           max_detail_nodes = Hard.max_detail_nodes;
           max_detail_graphs = Hard.max_detail_graphs;
           max_detail_bytes = 0x100_0000;
           response_live_bytes = 0L;
         })

  (* [create], then the browser relations. Every profile built here is one a
     document may be acquired under, so the check runs at construction and its
     failure is a load-time error rather than a rejected request. *)
  let wire_selectable r =
    let open Err.Syntax in
    let* t = r in
    let+ () = within_hard_response t in
    t

  (* The default for every file-shaped input, CLI included, and the ceiling the
     wire is measured against. Provisional and conservative — every figure
     clears what a real vision model needs by an order of magnitude, and the
     release profile is calibrated after Stages 2-4. *)
  let untrusted =
    Err.or_raise ~pp_error
      (wire_selectable
      @@ create ~max_json_bytes:0x1000_0000L ~max_pt2_bytes:0x1000_0000L
           ~zip:Pt2_zip.Limits.untrusted ~max_nodes_per_graph:0x2_0000
           ~max_edges_per_graph:0x4_0000 ~max_groups_per_graph:0x1_0000
           ~max_attrs_per_node:256 ~max_metadata_items_per_node:256
           ~max_outputs_metadata_per_node:256 ~max_namespace_depth:32
           ~max_namespace_component_bytes:256 ~max_label_bytes:1024
           ~max_id_bytes:1024 ~max_attr_chars:0x2000 ~max_url_bytes:2048
           ~max_graphs:1024 ~max_total_nodes:0x100_0000L
           ~max_total_edges:0x200_0000L ~max_views:256 ~max_comparisons:64
           ~max_node_data_sets:64 ~max_states:256 ~max_transitions:1024
           ~max_mapping_entries_per_comparison:0x2_0000
           ~max_mapping_members_per_entry:64
           ~max_mapping_members_total:0x20_0000
           ~max_node_data_results_per_graph:0x2_0000
           ~max_overlay_edges_per_overlay:0x2_0000
           ~max_overlay_edges_total:0x20_0000 ~max_diagnostics:32
           ~max_diagnostic_bytes:512 ~max_session_bytes:0x40_0000
           ~max_trace_entries:0x4_0000 ~max_audit_reports:0x1000
           ~max_detail_nodes:0x4000 ~max_detail_graphs:256
           ~max_detail_bytes:0x10_0000 trusted)

  (* Fieldwise no looser than [untrusted], so it is wire-selectable. *)
  let small =
    Err.or_raise ~pp_error
      (wire_selectable
      @@ create ~max_json_bytes:0x200_0000L ~max_pt2_bytes:0x200_0000L
           ~max_nodes_per_graph:0x4000 ~max_edges_per_graph:0x8000
           ~max_groups_per_graph:0x2000 ~max_attrs_per_node:64
           ~max_metadata_items_per_node:64 ~max_outputs_metadata_per_node:64
           ~max_namespace_depth:16 ~max_namespace_component_bytes:128
           ~max_label_bytes:256 ~max_id_bytes:256 ~max_attr_chars:0x800
           ~max_url_bytes:1024 ~max_graphs:128 ~max_total_nodes:0x10_0000L
           ~max_total_edges:0x20_0000L ~max_views:64 ~max_comparisons:16
           ~max_node_data_sets:16 ~max_states:64 ~max_transitions:256
           ~max_mapping_entries_per_comparison:0x4000
           ~max_mapping_members_per_entry:16 ~max_mapping_members_total:0x4_0000
           ~max_node_data_results_per_graph:0x4000
           ~max_overlay_edges_per_overlay:0x4000
           ~max_overlay_edges_total:0x4_0000 ~max_diagnostics:16
           ~max_diagnostic_bytes:256 ~max_session_bytes:0x8_0000
           ~max_trace_entries:0x8000 ~max_audit_reports:256
           ~max_detail_nodes:0x1000 ~max_detail_graphs:64
           ~max_detail_bytes:0x4_0000 untrusted)

  (* Between [untrusted] and [Hard], for internal callers holding data they
     produced. Programmatic-only: [Wire_limits] rejects it, which is what makes
     [--limits untrusted|small] a guarantee rather than a UI convention. *)
  let large =
    Err.or_raise ~pp_error
      (create ~max_nodes_per_graph:Hard.max_nodes_per_graph
         ~max_edges_per_graph:Hard.max_edges_per_graph
         ~max_groups_per_graph:Hard.max_groups_per_graph
         ~max_total_nodes:Hard.max_total_nodes
         ~max_total_edges:Hard.max_total_edges ~max_session_bytes:0x100_0000
         ~max_detail_bytes:0x100_0000 untrusted)
end
