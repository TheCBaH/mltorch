(* Ceilings and profiles for the Model Explorer export path. See the .mli for
   what the three layers are and why they are three. *)

module Invalid = struct
  type t = { name : string; value : int64 }
end

type live_error = [ `Live_overflow of string ]
type error = [ `Invalid_limit of Invalid.t | live_error ]

let pp_error fmt : [< error ] -> unit = function
  | `Invalid_limit { Invalid.name; value } ->
      Fmt.pf fmt "invalid limit %s = %Ld" name value
  | `Live_overflow phase -> Fmt.pf fmt "live allocation overflow in %s" phase

(* --- checked int64 arithmetic over the allocation phases ---

   Every product and every sum below is tested BEFORE it is performed. A
   wrapped total passes a [<=] test, which is the whole reason these figures
   are int64 and not int: the peaks are hundreds of megabytes and the browser
   backend's [int] is 32 bits. *)

module Phase = struct
  (* The request builder's three phases and the response path's four. They
     share one error row because they share one failure: a sum that cannot be
     represented, reported by name so the formula that overflowed is
     identifiable without re-deriving it. *)
  type t = Convert | Encode | Copy | Decode | Build | Commit | Install

  let name = function
    | Convert -> "P_convert"
    | Encode -> "P_encode"
    | Copy -> "P_copy"
    | Decode -> "R_decode"
    | Build -> "R_build"
    | Commit -> "R_commit"
    | Install -> "R_install"
end

let overflow phase = Core.fail (`Live_overflow (Phase.name phase))

(* A term is [coefficient * quantity]. The coefficients are single-digit hard
   constants; the quantities are the only operands a profile can move, which is
   why the check is on the product rather than on the result. A negative
   operand is reported here too: it is not a representable live total, and
   letting it through would produce a small positive answer for it. *)
type term = { coefficient : int; quantity : int64 }

let term_bytes phase { coefficient; quantity } =
  let k = Int64.of_int coefficient in
  if Int64.compare k 0L < 0 || Int64.compare quantity 0L < 0 then overflow phase
  else if Int64.equal k 0L then Core.return 0L
  else if Int64.compare quantity (Int64.div Int64.max_int k) > 0 then
    overflow phase
  else Core.return (Int64.mul k quantity)

let sum_terms phase terms =
  Core.List.fold_left
    (fun acc t ->
      let open Core.Syntax in
      let* v = term_bytes phase t in
      if Int64.compare v (Int64.sub Int64.max_int acc) > 0 then overflow phase
      else Core.return (Int64.add acc v))
    0L terms

let max_of = List.fold_left (fun a b -> if Int64.compare b a > 0 then b else a)

(* --- the hard scalars ---

   Exposed as [Hard] below, once the two aggregates derived from them exist. *)

module Hard_scalars = struct
  (* Under js_of_ocaml [Sys.int_size = 32] and [max_int = 2147483647]
     (measured under node; [Sys.max_string_length] is 2147483643). A live total
     that is ever narrowed to [int] must stay inside that domain with room for
     the arithmetic that precedes the narrowing, so this is half of [max_int]:
     the sum of any two totals bounded by it is still representable. *)
  let jsoo_safe_bytes = 0x4000_0000L

  (* Request identity is protocol, not policy: both figures are consequences of
     the [<epoch>-<seq>] grammar, which [Request_id.of_string] checks directly.
     A UUID is 36 bytes; 10 is the widest decimal sequence below 2^32. *)
  let max_epoch_bytes = 36
  let max_request_id_bytes = 47
  let max_seq_exclusive = 4_294_967_296L

  (* Small, because every removal step runs a [disconnectedCallback] and that
     is user code which may append a child. *)
  let max_restore_steps = 8

  (* One encoded request: an id, a name, two URLs, a digest, a detail key and a
     normalised stage list. Kilobyte-scale with three orders of magnitude of
     slack, and small enough that [3 *] it stays trivially inside
     [jsoo_safe_bytes]. *)
  let max_request_json_bytes = 0x1_0000

  (* One response envelope's metadata. Its variable half is the diagnostics
     list, and the relation checked at the bottom of this module is what ties
     the two together. *)
  let max_response_meta_bytes = 0x10_0000

  (* The ceiling a response document may never exceed, whatever profile its
     metadata claims. A session document for a vision model is single-digit
     megabytes at most (resnet18's model.json is 226 263 B).

     It is deliberately BELOW the [max_session_bytes]/[max_detail_bytes] field
     ceilings, and that gap is what keeps [within_hard_response] a check that
     can fail: a profile may be built above this figure — [trusted] is — and is
     then not one whose documents the browser may read. Collapsing the two
     would make the relation true by construction and its test vacuous. *)
  let max_response_document_bytes = 0x80_0000

  (* Expansion factors: heap bytes per byte of document. Frozen conservatively
     from reasoning because they are INPUTS to the hard derivations below — a
     factor that changed after calibration would change the value it was used
     to derive. Measurement may only tighten a profile. *)

  let js_string_expansion = 2
  let js_value_expansion = 8
  let session_expansion = 8
  let graph_expansion = 8
  let render_state_expansion = 1

  (* Escaping a sanitised string for JSON. [Diagnostic.create] replaces control
     bytes and invalid sequences with U+FFFD before the length rule applies, so
     the six-byte u-escape form is unreachable and only the quote and the
     backslash expand, each to two bytes. *)
  let json_escape_expansion = 2

  (* The cardinality of [Me_session.Capability.graph_stage], which bounds a
     request's normalised stage list. Stated here because the request peak is a
     hard constant and that module does not exist yet; when it does, its test
     asserts the two agree. *)
  let max_request_stages = 7
  let max_stage_bytes = 64

  (* A source digest is a hex SHA-256. *)
  let max_digest_bytes = 64

  (* Per-field ceilings, one per tunable [Limits.t] field. *)

  (* The bounded read in [Pt2_archive] clamps to 512MB on both backends; a
     document ceiling above the clamp would be a limit that does not limit. *)
  let max_json_bytes = 0x2000_0000L
  let max_pt2_bytes = 0x2000_0000L
  let max_nodes_per_graph = 0x10_0000
  let max_edges_per_graph = 0x40_0000
  let max_groups_per_graph = 0x10_0000
  let max_attrs_per_node = 1024
  let max_metadata_items_per_node = 1024
  let max_outputs_metadata_per_node = 1024
  let max_namespace_depth = 64
  let max_namespace_component_bytes = 1024
  let max_label_bytes = 4096
  let max_id_bytes = 4096
  let max_attr_chars = 0x1_0000

  (* The interoperable URL ceiling: below every browser and proxy figure. *)
  let max_url_bytes = 2048
  let max_graphs = 4096
  let max_total_nodes = 0x400_0000L
  let max_total_edges = 0x1000_0000L
  let max_views = 4096
  let max_comparisons = 1024
  let max_node_data_sets = 1024
  let max_states = 4096
  let max_transitions = 16384
  let max_mapping_entries_per_comparison = 0x10_0000
  let max_mapping_members_per_entry = 1024
  let max_mapping_members_total = 0x100_0000
  let max_node_data_results_per_graph = 0x10_0000
  let max_overlay_edges_per_overlay = 0x10_0000
  let max_overlay_edges_total = 0x100_0000
  let max_diagnostics = 64
  let max_diagnostic_bytes = 1024

  (* Above [max_response_document_bytes] on purpose (see there), and above the
     widest profile the response peak admits: at 32MB the [R_install] sum is
     roughly 1.6GB, so [derive]'s [jsoo_safe_bytes] rejection is reachable
     through [create] rather than being dead code guarding an impossible
     profile. *)
  let max_session_bytes = 0x200_0000
  let max_detail_bytes = 0x200_0000
  let max_trace_entries = 0x10_0000
  let max_audit_reports = 0x1_0000
  let max_detail_nodes = 0x1_0000
  let max_detail_graphs = 1024
end

(* --- the two derived aggregates --- *)

(* Every textual field the builder retains at once, plus the widest single one
   it converts. Builder-owned allocations only: the raw JavaScript object the
   caller passed in is not ours to bound and is not counted. *)
let request_retained_terms =
  let open Hard_scalars in
  [
    { coefficient = 1; quantity = Int64.of_int max_request_id_bytes };
    { coefficient = 1; quantity = Int64.of_int max_label_bytes };
    { coefficient = 2; quantity = Int64.of_int max_url_bytes };
    { coefficient = 1; quantity = Int64.of_int max_digest_bytes };
    { coefficient = 1; quantity = Int64.of_int max_id_bytes };
    {
      coefficient = max_request_stages;
      quantity = Int64.of_int max_stage_bytes;
    };
  ]

(* The widest single converted field, which is live once more as a temporary
   while it is being converted. *)
let request_widest_field = Hard_scalars.max_label_bytes

let request_live_bytes () =
  let open Core.Syntax in
  let open Hard_scalars in
  let json = Int64.of_int max_request_json_bytes in
  let* retained_convert = sum_terms Phase.Convert request_retained_terms in
  let* convert =
    sum_terms Phase.Convert
      [
        { coefficient = 1; quantity = retained_convert };
        { coefficient = 3; quantity = Int64.of_int request_widest_field };
      ]
  in
  let* retained_encode = sum_terms Phase.Encode request_retained_terms in
  let* encode =
    sum_terms Phase.Encode
      [
        { coefficient = 1; quantity = retained_encode };
        (* Writer capacity, with a factor-of-two growth allowance: [Bytesrw]'s
           writer is a filter over a [Buffer], whose growth doubles. *)
        { coefficient = 2; quantity = json };
        (* The encoded string. *)
        { coefficient = 1; quantity = json };
      ]
  in
  let* retained_copy = sum_terms Phase.Copy request_retained_terms in
  let+ copy =
    sum_terms Phase.Copy
      [
        (* The retained fields are included: nothing makes the request record
           unreachable before the copy runs, and a peak formula may not assume
           a lifetime the code does not establish. *)
        { coefficient = 1; quantity = retained_copy };
        (* The encoded string and the destination ArrayBuffer. *)
        { coefficient = 2; quantity = json };
      ]
  in
  max_of 0L [ convert; encode; copy ]

let response_live_bytes ~max_session_bytes ~max_detail_bytes =
  (* One document at a time crosses the boundary, so the peak is driven by
     whichever kind this profile admits more of. Bound before the [open] below,
     which carries fields of the same two names. *)
  let doc = Int64.of_int (max max_session_bytes max_detail_bytes) in
  let open Core.Syntax in
  let open Hard_scalars in
  let render_parsed = render_state_expansion * js_value_expansion in
  let* decode =
    sum_terms Phase.Decode
      [
        (* payload ArrayBuffer + decoded JS string + OCaml string *)
        { coefficient = 1; quantity = doc };
        { coefficient = js_string_expansion; quantity = doc };
        { coefficient = 1; quantity = doc };
      ]
  in
  let* build =
    sum_terms Phase.Build
      [
        (* The JS string is still the caller's argument: [prepareOpen] is
           synchronous and holds both of its arguments for the whole call, so a
           release here is one the API cannot perform. *)
        { coefficient = js_string_expansion; quantity = doc };
        { coefficient = 1; quantity = doc };
        { coefficient = session_expansion; quantity = doc };
      ]
  in
  let* commit =
    sum_terms Phase.Commit
      [
        { coefficient = js_string_expansion; quantity = doc };
        (* candidate and retained sessions *)
        { coefficient = 2 * session_expansion; quantity = doc };
        (* the encoded render state, its JS-parsed form, and the previous one
           JavaScript still holds *)
        { coefficient = render_state_expansion; quantity = doc };
        { coefficient = 2 * render_parsed; quantity = doc };
      ]
  in
  let+ install =
    sum_terms Phase.Install
      [
        { coefficient = 2 * session_expansion; quantity = doc };
        (* The queued buffers and their metadata, at the STRUCTURAL bound of
           one per kind. Deliberately conservative rather than tight: the
           reachable maximum is one, since a build is refused while a ticket of
           that kind is present, but over-reserving one document costs nothing
           and survives a later state model that admits a successor. *)
        { coefficient = 2; quantity = Int64.of_int max_response_document_bytes };
        { coefficient = 2; quantity = Int64.of_int max_response_meta_bytes };
        { coefficient = render_state_expansion; quantity = doc };
        { coefficient = 2 * render_parsed; quantity = doc };
        (* the JS document string, still the caller's *)
        { coefficient = js_string_expansion; quantity = doc };
        (* both processed graphs, old and new, each with its element *)
        { coefficient = 2 * graph_expansion; quantity = doc };
      ]
  in
  max_of 0L [ decode; build; commit; install ]

module Hard = struct
  include Hard_scalars

  let max_request_live_bytes = Core.or_raise pp_error (request_live_bytes ())

  (* Derived from hard scalars — the phase sums evaluated at
     [max_response_document_bytes] — not from any profile's peak, which would
     make a [Hard] constant depend on the profile [Hard] constrains. *)
  let max_response_live_bytes =
    Core.or_raise pp_error
      (response_live_bytes ~max_session_bytes:max_response_document_bytes
         ~max_detail_bytes:max_response_document_bytes)
end

(* The four relations between the constants, checked once here rather than per
   input. Placing them on the constants is what keeps them out of the hot path
   while still being the thing that makes the per-field ceilings sound: with
   all four true, any set of fields the builder accepts fits, and no aggregate
   arithmetic ever runs on attacker-influenced values. *)
let () =
  let open Hard in
  let relation name value ceiling =
    if Int64.compare value ceiling > 0 then
      Core.fail (`Invalid_limit { Invalid.name; value })
    else Core.return ()
  in
  Core.or_raise pp_error
    (let open Core.Syntax in
     (* One worst-case [Js.to_string]: the JS string, its UTF-8 measure and the
        OCaml result. *)
     let* one_conversion =
       sum_terms Phase.Convert
         [ { coefficient = 3; quantity = Int64.of_int max_request_json_bytes } ]
     in
     let* () =
       relation "3 * max_request_json_bytes" one_conversion jsoo_safe_bytes
     in
     let* () =
       relation "max_request_live_bytes" max_request_live_bytes jsoo_safe_bytes
     in
     let* () =
       relation "max_response_live_bytes" max_response_live_bytes
         jsoo_safe_bytes
     in
     (* The diagnostics list is the only variable part of a response envelope's
        metadata, so the meta ceiling has to admit as many as a profile may
        emit, each carrying a message and a graph id, escaped. *)
     let* per_diagnostic =
       sum_terms Phase.Encode
         [
           {
             coefficient = json_escape_expansion;
             quantity = Int64.of_int (max_diagnostic_bytes + max_id_bytes);
           };
         ]
     in
     let* diagnostics_bytes =
       sum_terms Phase.Encode
         [ { coefficient = max_diagnostics; quantity = per_diagnostic } ]
     in
     relation "max_diagnostics * diagnostic bytes" diagnostics_bytes
       (Int64.of_int max_response_meta_bytes))

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
      Core.fail (`Invalid_limit { Invalid.name; value })
    else Core.return ()

  (* Against a ceiling supplied as a profile, so [create] (ceiling = the [Hard]
     figures) and [Wire_limits.of_limits] (ceiling = [untrusted]) run the same
     traversal over the same field set. *)
  let check_against ~zip_ceiling ~field_ceiling t =
    let open Core.Syntax in
    let* () =
      Core.List.iter (fun f -> check f.name (f.get t) (field_ceiling f)) fields
    in
    Core.List.iter
      (fun (name, get, _) ->
        check ("zip." ^ name) (get t.zip) (get zip_ceiling))
      zip_fields

  let derive t =
    let open Core.Syntax in
    let* peak =
      response_live_bytes ~max_session_bytes:t.max_session_bytes
        ~max_detail_bytes:t.max_detail_bytes
    in
    let+ () =
      if Int64.compare peak Hard.jsoo_safe_bytes > 0 then
        Core.fail
          (`Invalid_limit { Invalid.name = "response_live_bytes"; value = peak })
      else Core.return ()
    in
    { t with response_live_bytes = peak }

  let validate t =
    let open Core.Syntax in
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
    let open Core.Syntax in
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
    let open Core.Syntax in
    let+ () = check f.name v (f.get ceiling) in
    f.assign t v

  let assign_zip_checked ~ceiling t (name, get, set) v =
    let open Core.Syntax in
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
    Core.List.fold_left
      (fun t (name, v) ->
        match field_named name with
        | Some (`Field f) -> assign_checked ~ceiling t f v
        | Some (`Zip z) -> assign_zip_checked ~ceiling t z v
        | None -> Core.fail (`Invalid_limit { Invalid.name; value = v }))
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
    Core.or_raise pp_error
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
    let open Core.Syntax in
    let* t = r in
    let+ () = within_hard_response t in
    t

  (* The default for every file-shaped input, CLI included, and the ceiling the
     wire is measured against. Provisional and conservative — every figure
     clears what a real vision model needs by an order of magnitude, and the
     release profile is calibrated after Stages 2-4. *)
  let untrusted =
    Core.or_raise pp_error
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
    Core.or_raise pp_error
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
    Core.or_raise pp_error
      (create ~max_nodes_per_graph:Hard.max_nodes_per_graph
         ~max_edges_per_graph:Hard.max_edges_per_graph
         ~max_groups_per_graph:Hard.max_groups_per_graph
         ~max_total_nodes:Hard.max_total_nodes
         ~max_total_edges:Hard.max_total_edges ~max_session_bytes:0x100_0000
         ~max_detail_bytes:0x100_0000 untrusted)
end

(* [Jsont.Object.as_string_map] yields a [Stdlib.Map.Make(String)], and
   [Map.Make] is applicative, so this local instance IS that type. *)
module Wire_map = Map.Make (String)

module Wire_limits = struct
  type t = Limits.t

  let of_limits ~ceiling l =
    let open Core.Syntax in
    let* () =
      Limits.check_against ~zip_ceiling:ceiling.Limits.zip
        ~field_ceiling:(fun f -> f.Limits.get ceiling)
        l
    in
    (* Through [Limits.create] as well, so that every wire profile is a
       create-validated profile whatever route produced [l]. *)
    Limits.create l

  let limits t = t

  (* The wire carries a TIGHTENING of [untrusted], never a whole profile: a
     [Wire_limits.t] is by construction no looser, so a field equal to
     [untrusted]'s carries no information and is not sent. [untrusted] itself
     therefore encodes as [{}].

     FLAT, keyed by the field's own diagnostic name -- [zip.max_entries], not a
     nested object -- so the member a rejection names and the member that
     carried it are literally the same string.

     Values are JSON STRINGS through [int64_as_string]: two of these fields are
     [int64] with values past 2^31, and a JSON number would be read back
     through a 32-bit [int] under jsoo. *)
  let member_jsont =
    Jsont.Object.as_string_map ~kind:"limits" Jsont.int64_as_string

  let jsont =
    Jsont.map ~kind:"wireLimits"
      ~dec:(fun members ->
        let pairs = Wire_map.bindings members in
        match
          let open Core.Syntax in
          let* l =
            Limits.apply_overrides ~ceiling:Limits.untrusted
              ~base:Limits.untrusted pairs
          in
          (* [of_limits] rather than a bare [create]: the decoded value has to
             be a value the ENCODER could have produced, and only the ceiling
             check makes that true. *)
          of_limits ~ceiling:Limits.untrusted l
        with
        | Ok l -> l
        | Error e ->
            Jsont.Error.msgf Jsont.Meta.none "%a" pp_error e.Core.Error.kind)
      ~enc:(fun t ->
        List.fold_left
          (fun m (name, v) -> Wire_map.add name v m)
          Wire_map.empty
          (Limits.overrides ~base:Limits.untrusted t))
      member_jsont
end

(* --- diagnostics --- *)

module Diagnostic = struct
  module Code = struct
    type t =
      | Over_limit
      | Malformed_request
      | Invalid_limits
      | Invalid_source
      | Malformed_response
      | Request_in_flight
      | Inconsistent_mount
      | Settlement_mismatch
      | Buffer_mismatch
      | Not_an_array_buffer
      | Stale_epoch
      | Unsupported_detail_key
      | Key_disagrees_with_ids
      | Unsupported_operator
      | Unsupported_input
      | Unsupported_graph_shape
      | Outside_dialect_domain
      | Requires_payloads
      | Prerequisite_unavailable
      | Not_implemented
      | Internal

    let to_string = function
      | Over_limit -> "over_limit"
      | Malformed_request -> "malformed_request"
      | Invalid_limits -> "invalid_limits"
      | Invalid_source -> "invalid_source"
      | Malformed_response -> "malformed_response"
      | Request_in_flight -> "request_in_flight"
      | Inconsistent_mount -> "inconsistent_mount"
      | Settlement_mismatch -> "settlement_mismatch"
      | Buffer_mismatch -> "buffer_mismatch"
      | Not_an_array_buffer -> "not_an_array_buffer"
      | Stale_epoch -> "stale_epoch"
      | Unsupported_detail_key -> "unsupported_detail_key"
      | Key_disagrees_with_ids -> "key_disagrees_with_ids"
      | Unsupported_operator -> "unsupported_operator"
      | Unsupported_input -> "unsupported_input"
      | Unsupported_graph_shape -> "unsupported_graph_shape"
      | Outside_dialect_domain -> "outside_dialect_domain"
      | Requires_payloads -> "requires_payloads"
      | Prerequisite_unavailable -> "prerequisite_unavailable"
      | Not_implemented -> "not_implemented"
      | Internal -> "internal"

    (* [all] is built by walking this successor chain rather than being written
       out beside the type, so that adding a constructor cannot leave the
       vocabulary behind: the match below is exhaustive, so the new code needs
       an arm, and reaching it needs a predecessor to name it. A hand-written
       list would compile perfectly while [of_string] silently stopped
       recognising the new tag — which, for a closed vocabulary meant to be
       switched on, turns a named condition into an unknown one. *)
    let next = function
      | Over_limit -> Some Malformed_request
      | Malformed_request -> Some Invalid_limits
      | Invalid_limits -> Some Invalid_source
      | Invalid_source -> Some Malformed_response
      | Malformed_response -> Some Request_in_flight
      | Request_in_flight -> Some Inconsistent_mount
      | Inconsistent_mount -> Some Settlement_mismatch
      | Settlement_mismatch -> Some Buffer_mismatch
      | Buffer_mismatch -> Some Not_an_array_buffer
      | Not_an_array_buffer -> Some Stale_epoch
      | Stale_epoch -> Some Unsupported_detail_key
      | Unsupported_detail_key -> Some Key_disagrees_with_ids
      | Key_disagrees_with_ids -> Some Unsupported_operator
      | Unsupported_operator -> Some Unsupported_input
      | Unsupported_input -> Some Unsupported_graph_shape
      | Unsupported_graph_shape -> Some Outside_dialect_domain
      | Outside_dialect_domain -> Some Requires_payloads
      | Requires_payloads -> Some Prerequisite_unavailable
      | Prerequisite_unavailable -> Some Not_implemented
      | Not_implemented -> Some Internal
      | Internal -> None

    let all =
      let rec walk acc c =
        match next c with
        | None -> List.rev (c :: acc)
        | Some n -> walk (c :: acc) n
      in
      walk [] Over_limit

    let assoc = List.map (fun c -> (to_string c, c)) all

    (* Total in its failure, so an unknown tag from a future producer decodes
       as a named error rather than being silently dropped. *)
    let of_string s = List.assoc_opt s assoc
    let jsont = Jsont.enum ~kind:"diagnostic_code" assoc
  end

  type t = {
    code : Code.t;
    message : string;
    graph : string option;
    truncated : bool;
  }

  (* Sanitisation is REPLACEMENT, not only truncation. [of_exn] admits
     arbitrary OCaml bytes, which need not be valid UTF-8 at all, and
     [Jsont_bytesrw]'s encoder does NOT validate them — it passes them through
     and emits a document that is not JSON. So nothing in OCaml catches it, and
     the failure surfaces in the browser as a malformed response: the very
     condition the diagnostic existed to report. Invalid sequences and control
     bytes therefore become U+FFFD BEFORE the length rule applies, and the
     length rule then cuts on a scalar-value boundary. *)

  let is_control u =
    let c = Uchar.to_int u in
    c < 0x20 || c = 0x7F

  let utf_8_width u =
    let c = Uchar.to_int u in
    if c < 0x80 then 1
    else if c < 0x800 then 2
    else if c < 0x10000 then 3
    else 4

  let sanitise ~max s =
    let buf = Buffer.create (min (String.length s) max) in
    let truncated = ref false in
    let i = ref 0 in
    let n = String.length s in
    (try
       while !i < n do
         let d = String.get_utf_8_uchar s !i in
         let u =
           if Uchar.utf_decode_is_valid d then
             let u = Uchar.utf_decode_uchar d in
             if is_control u then Uchar.rep else u
           else Uchar.rep
         in
         (* Replacement can WIDEN — one stray byte becomes three — so the
            ceiling is tested against what is about to be written, not against
            what was read. *)
         if Buffer.length buf + utf_8_width u > max then begin
           truncated := true;
           raise Exit
         end;
         Buffer.add_utf_8_uchar buf u;
         i := !i + Uchar.utf_decode_length d
       done
     with Exit -> ());
    (Buffer.contents buf, !truncated)

  (* An identifier is either exactly right or absent. Truncating one yields a
     different identifier, which either resolves to the wrong graph or fails to
     resolve while looking authoritative; sanitising one has the same defect for
     the same reason. So this is a predicate and its failure is a DROP. *)
  let acceptable_id ~max s =
    String.length s <= max
    &&
    let ok = ref true and i = ref 0 in
    let n = String.length s in
    while !ok && !i < n do
      let d = String.get_utf_8_uchar s !i in
      if
        (not (Uchar.utf_decode_is_valid d))
        || is_control (Uchar.utf_decode_uchar d)
      then ok := false
      else i := !i + Uchar.utf_decode_length d
    done;
    !ok

  let build ~limits ?graph code message =
    let message, message_truncated =
      sanitise ~max:limits.Limits.max_diagnostic_bytes message
    in
    let graph, graph_dropped =
      match graph with
      | None -> (None, false)
      | Some g when acceptable_id ~max:limits.Limits.max_id_bytes g ->
          (Some g, false)
      | Some _ -> (None, true)
    in
    (* The drop sets [truncated] too, so a missing graph is visible rather than
       indistinguishable from one that was never supplied. *)
    { code; message; graph; truncated = message_truncated || graph_dropped }

  let create ~limits ?graph code message = build ~limits ?graph code message
  let of_exn ~limits e = build ~limits Code.Internal (Printexc.to_string e)

  let jsont =
    Jsont.Object.map ~kind:"diagnostic" (fun code message graph truncated ->
        (* Through the same constructor as the producer, under HARD limits: a
           [Jsont.t] carries no profile, and the profile-level bound is applied
           by whoever holds one. Whatever the wire claimed about [truncated] is
           kept beside what re-sanitising found, since a producer that already
           cut something knows that and this decoder cannot. *)
        let d = build ~limits:Limits.trusted ?graph code message in
        { d with truncated = d.truncated || truncated })
    |> Jsont.Object.mem "code" Code.jsont ~enc:(fun d -> d.code)
    |> Jsont.Object.mem "message" Jsont.string ~enc:(fun d -> d.message)
    |> Jsont.Object.opt_mem "graph" Jsont.string ~enc:(fun d -> d.graph)
    |> Jsont.Object.mem "truncated" Jsont.bool ~dec_absent:false ~enc:(fun d ->
        d.truncated)
    |> Jsont.Object.finish

  let pp fmt d =
    Fmt.pf fmt "%s: %s%a%s" (Code.to_string d.code) d.message
      (Core.Pretty.option_or ~none:"" (fun fmt g -> Fmt.pf fmt " [%s]" g))
      d.graph
      (if d.truncated then " (truncated)" else "")
end
