(* Checked int64 arithmetic over the allocation phases, the hard scalars,
   and the two derived peak calculators (request/response live bytes),
   split from me_limits.ml. Depends on me_limits_error.ml
   for [Invalid]/[error]/[pp_error]; nothing here depends on the profile
   type in me_limits_profile.ml (the dependency runs the other way). *)

open Me_limits_error

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

let overflow phase = Err.fail (`Live_overflow (Phase.name phase))

(* A term is [coefficient * quantity]. The coefficients are single-digit hard
   constants; the quantities are the only operands a profile can move, which is
   why the check is on the product rather than on the result. A negative
   operand is reported here too: it is not a representable live total, and
   letting it through would produce a small positive answer for it. *)
type term = { coefficient : int; quantity : int64 }

let term_bytes phase { coefficient; quantity } =
  let k = Int64.of_int coefficient in
  if Int64.compare k 0L < 0 || Int64.compare quantity 0L < 0 then overflow phase
  else if Int64.equal k 0L then Err.return 0L
  else if Int64.compare quantity (Int64.div Int64.max_int k) > 0 then
    overflow phase
  else Err.return (Int64.mul k quantity)

let sum_terms phase terms =
  Err.List.fold_left
    (fun acc t ->
      let open Err.Syntax in
      let* v = term_bytes phase t in
      if Int64.compare v (Int64.sub Int64.max_int acc) > 0 then overflow phase
      else Err.return (Int64.add acc v))
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

  (* How many CANCELLED Model Explorer installs the browser may keep connected.
     The pinned custom element has no abort or dispose API and disconnecting one
     that is still processing destroys a live Angular component mid-flight, so a
     cancelled install cannot be removed on demand — it is hidden, stripped of
     authority, and removed only once its expected [modelGraphProcessed] proves
     removal is safe. This is the ceiling on how many may be awaiting that
     proof, and it is the reason [R_install] below is no longer a two-element
     sum. See [cancel.md] and .ai/model_explorer_design.md §3.

     [R_install] budgets [2 + this], which is deliberately ONE MORE than the
     reachable population: a candidate is admitted only while fewer than this
     many are quarantined, so a current, an active and a full quarantine cannot
     coexist and the true maximum is [1 + this]. The same over-reservation, for
     the same reason, as the queued-buffer terms below. Structural bound, not
     tight maximum.

     Three is the largest value that fits: at four, [Limits.trusted]'s peak
     reaches 1 126 170 624, above [jsoo_safe_bytes], and the profile would stop
     being constructible. *)
  let max_quarantined_elements = 3

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
  let open Err.Syntax in
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
  let open Err.Syntax in
  let open Hard_scalars in
  (* The document a JAVASCRIPT object can be holding, which is not [doc]: no
     document above [max_response_document_bytes] reaches the browser at all.
     Two independent checks enforce it — [within_hard_response] decides which
     profiles are wire-selectable, and the page rejects an over-size payload
     before decoding it (web/app/coordinator.js). [trusted] and [large] sit
     above that ceiling precisely because they are native-only, so budgeting
     their JS-side terms at [doc] would reserve for objects that cannot exist.
     Harmless while the coefficient was 2; decisive once [R_install] retains a
     bounded quarantine. *)
  let jsdoc =
    let ceiling = Int64.of_int max_response_document_bytes in
    if Int64.compare doc ceiling < 0 then doc else ceiling
  in
  let render_parsed = render_state_expansion * js_value_expansion in
  (* Current + active + a full quarantine. See [max_quarantined_elements]: one
     more than can actually be reached, on purpose. *)
  let elements = 2 + max_quarantined_elements in
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
        (* One parsed render state per retained element: the element holds the
           [graphCollections] it was assigned for as long as it is connected. *)
        { coefficient = elements * render_parsed; quantity = jsdoc };
        (* the JS document string, still the caller's *)
        { coefficient = js_string_expansion; quantity = jsdoc };
        (* One processed graph per retained element — the current one, the
           install in flight, and every cancelled install still awaiting the
           event that authorises its removal. Distinct from the term above: that
           is the input the element was given, this is what the visualizer
           computes from it. *)
        { coefficient = elements * graph_expansion; quantity = jsdoc };
      ]
  in
  max_of 0L [ decode; build; commit; install ]

module Hard = struct
  include Hard_scalars

  let max_request_live_bytes = Err.or_raise ~pp_error (request_live_bytes ())

  (* Derived from hard scalars — the phase sums evaluated at
     [max_response_document_bytes] — not from any profile's peak, which would
     make a [Hard] constant depend on the profile [Hard] constrains. *)
  let max_response_live_bytes =
    Err.or_raise ~pp_error
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
      Err.fail (`Invalid_limit { Invalid.name; value })
    else Err.return ()
  in
  Err.or_raise ~pp_error
    (let open Err.Syntax in
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
