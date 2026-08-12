(* [Me_limits]: the hard ceilings, the profiles built against them, the derived
   response peak, and the wire subset (.ai/model_explorer_design.md).

   Runs under node as well as natively because this module exists for the
   32-bit backend: every aggregate here is [int64] precisely so it does not
   wrap where [int] would, and a native-only suite would exercise the one
   backend on which the wrapping cannot happen. The goldens are shared, so a
   figure that differed per backend fails the build rather than being averaged
   over.

   The peak calculator's own overflow arm is NOT driven from here. Its inputs
   are [int], so reaching it needs a document ceiling near [max_int] — a
   different number on each backend, hence a golden that cannot be shared. What
   is testable on both, and is, is that every product and sum it performs is
   checked: the [jsoo_safe_bytes] rejection below runs through the same
   arithmetic. *)

(* Written as syntactic functions rather than partial applications so they stay
   polymorphic in the success type: every profile constructor below returns a
   different one, and only the failure side is ever rendered. *)
let pp_ok ppf r =
  Core.Pretty.err_result ~ok:(Fmt.any "ok") ~error:Me_limits.pp_error ppf r

let show r = Format.printf "%a@." pp_ok r
let show_labelled label r = Format.printf "%s: %a@." label pp_ok r

let show_bytes r =
  Format.printf "%a@."
    (Core.Pretty.err_result ~ok:Fmt.int64 ~error:Me_limits.pp_error)
    r

(* --- the constants --- *)

let%expect_test "hard ceilings" =
  let open Me_limits.Hard in
  Printf.printf "jsoo_safe_bytes             %Ld\n" jsoo_safe_bytes;
  Printf.printf "max_epoch_bytes             %d\n" max_epoch_bytes;
  Printf.printf "max_request_id_bytes        %d\n" max_request_id_bytes;
  Printf.printf "max_seq_exclusive           %Ld\n" max_seq_exclusive;
  Printf.printf "max_restore_steps           %d\n" max_restore_steps;
  Printf.printf "max_request_json_bytes      %d\n" max_request_json_bytes;
  Printf.printf "max_response_meta_bytes     %d\n" max_response_meta_bytes;
  Printf.printf "max_response_document_bytes %d\n" max_response_document_bytes;
  Printf.printf "max_request_live_bytes      %Ld\n" max_request_live_bytes;
  Printf.printf "max_response_live_bytes     %Ld\n" max_response_live_bytes;
  [%expect
    {|
    jsoo_safe_bytes             1073741824
    max_epoch_bytes             36
    max_request_id_bytes        47
    max_seq_exclusive           4294967296
    max_restore_steps           8
    max_request_json_bytes      65536
    max_response_meta_bytes     1048576
    max_response_document_bytes 8388608
    max_request_live_bytes      209455
    max_response_live_bytes     446693376
    |}]

let%expect_test "the request-id constants are consequences of the grammar" =
  (* 36 is a UUID; 47 is that plus a separator plus the widest decimal sequence
     below 2^32. Stated as arithmetic rather than as two literals, so a change
     to one that is not a change to the other fails here. *)
  let widest_seq = String.length (Printf.sprintf "%Lu" 4_294_967_295L) in
  Printf.printf "%d + 1 + %d = %d\n" Me_limits.Hard.max_epoch_bytes widest_seq
    (Me_limits.Hard.max_epoch_bytes + 1 + widest_seq);
  Printf.printf "max_request_id_bytes = %d\n"
    Me_limits.Hard.max_request_id_bytes;
  Printf.printf "2^32 = %Ld, exclusive\n" Me_limits.Hard.max_seq_exclusive;
  [%expect
    {|
    36 + 1 + 10 = 47
    max_request_id_bytes = 47
    2^32 = 4294967296, exclusive
    |}]

let%expect_test "the four relations between the constants" =
  (* Module initialisation already raised if any of these failed — reaching
     this line at all is the assertion. What the figures add is the margin: a
     relation that holds by a hair is one a later ceiling change breaks
     silently, so the headroom is pinned too. *)
  let open Me_limits.Hard in
  let margin name value ceiling =
    Printf.printf "%-34s %13Ld <= %10Ld\n" name value ceiling
  in
  margin "3 * max_request_json_bytes"
    (Int64.of_int (3 * max_request_json_bytes))
    jsoo_safe_bytes;
  margin "max_request_live_bytes" max_request_live_bytes jsoo_safe_bytes;
  margin "max_response_live_bytes" max_response_live_bytes jsoo_safe_bytes;
  margin "max_diagnostics * escaped bytes"
    (Int64.of_int
       (max_diagnostics * json_escape_expansion
       * (max_diagnostic_bytes + max_id_bytes)))
    (Int64.of_int max_response_meta_bytes);
  [%expect
    {|
    3 * max_request_json_bytes                196608 <= 1073741824
    max_request_live_bytes                    209455 <= 1073741824
    max_response_live_bytes                446693376 <= 1073741824
    max_diagnostics * escaped bytes           655360 <=    1048576
    |}]

(* --- the response peak --- *)

let%expect_test "the peak is monotone in both of its inputs" =
  (* Tightening either input can only lower the result. Asserted over a grid
     rather than at one point, because the calculator takes the maximum of two
     document kinds and a monotonicity bug would show as a step that goes the
     wrong way only where the maximum switches. *)
  let mib n = n * 0x10_0000 in
  (* [or_raise], not a printed sentinel: every point on this grid is inside the
     admitted domain, so a failure here is a broken calculator rather than a
     result to render, and a -1 folded into the comparisons would let it read
     as monotone. *)
  let peak s d =
    Err.or_raise ~pp_error:Me_limits.pp_error
      (Me_limits.response_live_bytes ~max_session_bytes:s ~max_detail_bytes:d)
  in
  let sizes = [ 1; 2; 4; 8; 16 ] in
  List.iter
    (fun s ->
      List.iter (fun d -> Printf.printf "%12Ld" (peak (mib s) (mib d))) sizes;
      print_newline ())
    sizes;
  let non_decreasing row =
    List.for_all2
      (fun a b -> Int64.compare a b <= 0)
      (List.filteri (fun i _ -> i < List.length row - 1) row)
      (List.tl row)
  in
  let rows =
    List.map (fun s -> List.map (fun d -> peak (mib s) (mib d)) sizes) sizes
  in
  let columns =
    List.mapi (fun j _ -> List.map (fun row -> List.nth row j) rows) sizes
  in
  Printf.printf "rows non-decreasing: %b\ncolumns non-decreasing: %b\n"
    (List.for_all non_decreasing rows)
    (List.for_all non_decreasing columns);
  [%expect
    {|
        72351744   125829120   232783872   446693376   874512384
       125829120   125829120   232783872   446693376   874512384
       232783872   232783872   232783872   446693376   874512384
       446693376   446693376   446693376   446693376   874512384
       874512384   874512384   874512384   874512384   874512384
    rows non-decreasing: true
    columns non-decreasing: true
    |}]

let%expect_test "the peak the hard constant is derived from" =
  (* [Hard.max_response_live_bytes] is the phase sums at
     [max_response_document_bytes] — hard scalars only, never a profile's peak,
     which would make a [Hard] constant depend on the profile it constrains. *)
  show_bytes
    (Me_limits.response_live_bytes
       ~max_session_bytes:Me_limits.Hard.max_response_document_bytes
       ~max_detail_bytes:Me_limits.Hard.max_response_document_bytes);
  Printf.printf "%Ld\n" Me_limits.Hard.max_response_live_bytes;
  [%expect {|
    446693376
    446693376
    |}]

(* --- profiles --- *)

let%expect_test "a profile may tighten, and never widen past Hard" =
  let base = Me_limits.Limits.untrusted in
  show (Me_limits.Limits.create ~max_nodes_per_graph:1024 base);
  show
    (Me_limits.Limits.create
       ~max_nodes_per_graph:(Me_limits.Hard.max_nodes_per_graph + 1)
       base);
  show (Me_limits.Limits.create ~max_nodes_per_graph:0 base);
  show (Me_limits.Limits.create ~max_total_nodes:0x8000_0000_0000L base);
  show (Me_limits.Limits.create ~max_url_bytes:(-1) base);
  [%expect
    {|
    ok
    invalid limit max_nodes_per_graph = 1048577
    invalid limit max_nodes_per_graph = 0
    invalid limit max_total_nodes = 140737488355328
    invalid limit max_url_bytes = -1
    |}]

let%expect_test "the nested archive profile is checked field by field" =
  (* The zip ceilings are [Pt2_zip.Limits.trusted] — that module's own [Hard]
     made into a profile — so there is no second copy of them here and a
     [zip] a caller assembled by hand cannot smuggle a value past. *)
  let base = Me_limits.Limits.untrusted in
  show (Me_limits.Limits.create ~zip:Pt2_zip.Limits.trusted base);
  show (Me_limits.Limits.create ~zip:Pt2_zip.Limits.untrusted base);
  [%expect {|
    ok
    ok
    |}]

let%expect_test "a profile whose peak does not fit is refused when it is built"
    =
  (* Reachable from above: the two document field ceilings sit at 32MB while
     the phase sums admit roughly 20MB, so [create] rejects before any input is
     acquired rather than after a document has been read. *)
  let base = Me_limits.Limits.trusted in
  show
    (Me_limits.Limits.create ~max_session_bytes:0x200_0000
       ~max_detail_bytes:0x200_0000 base);
  show
    (Me_limits.Limits.create ~max_session_bytes:0x100_0000
       ~max_detail_bytes:0x100_0000 base);
  [%expect {|
    invalid limit response_live_bytes = 1730150400
    ok
    |}]

let%expect_test "the derived field is stored, not recomputed" =
  let peak (t : Me_limits.Limits.t) = t.Me_limits.Limits.response_live_bytes in
  Printf.printf "untrusted %Ld\nsmall     %Ld\nlarge     %Ld\ntrusted   %Ld\n"
    (peak Me_limits.Limits.untrusted)
    (peak Me_limits.Limits.small)
    (peak Me_limits.Limits.large)
    (peak Me_limits.Limits.trusted);
  [%expect
    {|
    untrusted 232783872
    small     45613056
    large     874512384
    trusted   874512384
    |}]

let%expect_test "which profiles the browser may read a document under" =
  (* [trusted] and [large] satisfy none of the three relations; [untrusted] and
     [small] are checked against them where they are constructed, so reaching
     this test at all already asserts the passing half. The failing half is
     what makes the check evidence rather than decoration. *)
  let check name t =
    show_labelled name (Me_limits.Limits.within_hard_response t)
  in
  check "untrusted" Me_limits.Limits.untrusted;
  check "small" Me_limits.Limits.small;
  check "large" Me_limits.Limits.large;
  check "trusted" Me_limits.Limits.trusted;
  [%expect
    {|
    untrusted: ok
    small: ok
    large: invalid limit max_session_bytes = 16777216
    trusted: invalid limit max_session_bytes = 16777216
    |}]

(* --- the wire subset --- *)

let%expect_test "only profiles no looser than untrusted cross the wire" =
  (* This is also the fieldwise ordering assertion for [small]: [of_limits]
     compares every scalar and every nested zip field against the ceiling, so
     "small is no looser than untrusted" is exactly what its success means. *)
  let against name t =
    show_labelled name
      (Me_limits.Wire_limits.of_limits ~ceiling:Me_limits.Limits.untrusted t)
  in
  against "untrusted" Me_limits.Limits.untrusted;
  against "small" Me_limits.Limits.small;
  against "large" Me_limits.Limits.large;
  against "trusted" Me_limits.Limits.trusted;
  [%expect
    {|
    untrusted: ok
    small: ok
    large: 7 invalid limits: max_nodes_per_graph = 1048576,
             max_edges_per_graph = 4194304, max_groups_per_graph = 1048576,
             max_total_nodes = 67108864, max_total_edges = 268435456,
             max_session_bytes = 16777216, max_detail_bytes = 16777216
    trusted: 40 invalid limits: max_json_bytes = 536870912,
               max_pt2_bytes = 536870912, max_nodes_per_graph = 1048576,
               max_edges_per_graph = 4194304, max_groups_per_graph = 1048576,
               max_attrs_per_node = 1024, max_metadata_items_per_node = 1024,
               max_outputs_metadata_per_node = 1024, max_namespace_depth = 64,
               max_namespace_component_bytes = 1024, max_label_bytes = 4096,
               max_id_bytes = 4096, max_attr_chars = 65536, max_graphs = 4096,
               max_total_nodes = 67108864, max_total_edges = 268435456,
               max_views = 4096, max_comparisons = 1024,
               max_node_data_sets = 1024, max_states = 4096,
               max_transitions = 16384,
               max_mapping_entries_per_comparison = 1048576,
               max_mapping_members_per_entry = 1024,
               max_mapping_members_total = 16777216,
               max_node_data_results_per_graph = 1048576,
               max_overlay_edges_per_overlay = 1048576,
               max_overlay_edges_total = 16777216, max_diagnostics = 64,
               max_diagnostic_bytes = 1024, max_session_bytes = 16777216,
               max_trace_entries = 1048576, max_audit_reports = 65536,
               max_detail_nodes = 65536, max_detail_graphs = 1024,
               max_detail_bytes = 16777216, zip.max_entries = 65536,
               zip.max_entry_bytes = 1073741824,
               zip.max_total_bytes = 4294967296, zip.max_path_bytes = 65536,
               zip.max_path_depth = 64
    |}]

let%expect_test "one field between untrusted and Hard is enough to refuse" =
  (* A hand-built profile that is valid — below every [Hard] ceiling — and
     still not wire-eligible. This is the case the type exists for: a value
     below [Hard] can be far looser than [untrusted], and since the request IS
     the wire the check cannot be delegated to the page that built it. *)
  let open Err.Syntax in
  let widened field r =
    show_labelled field
      (r >>= Me_limits.Wire_limits.of_limits ~ceiling:Me_limits.Limits.untrusted)
  in
  widened "max_attrs_per_node"
    (Me_limits.Limits.create ~max_attrs_per_node:257 Me_limits.Limits.untrusted);
  widened "max_views"
    (Me_limits.Limits.create ~max_views:257 Me_limits.Limits.untrusted);
  (* And a nested one, which is the half a scalar-only comparison would miss.
     Exactly one nested field, so this stays the singular row: passing the whole
     of [Pt2_zip.Limits.trusted] widens all five at once, which is the next
     test. *)
  widened "zip.max_entries"
    (Me_limits.Limits.create
       ~zip:
         {
           Me_limits.Limits.untrusted.Me_limits.Limits.zip with
           Pt2_zip.Limits.max_entries =
             Pt2_zip.Limits.trusted.Pt2_zip.Limits.max_entries;
         }
       Me_limits.Limits.untrusted);
  [%expect
    {|
    max_attrs_per_node: invalid limit max_attrs_per_node = 257
    max_views: invalid limit max_views = 257
    zip.max_entries: invalid limit zip.max_entries = 65536
    |}]

(* The fieldwise sweep ACCUMULATES: an operator whose profile has several fields
   out of bounds learns about all of them, rather than one per round trip. This
   is the only validator here that does — see [Me_limits.Wire_limits.of_limits]
   for the two conditions that make it safe, neither of which holds for the
   document validators.

   The plural row is not decoration: before [Err.Accum], the previous test's
   whole-record case reported ONLY [zip.max_entries] and the other four
   rejections were invisible. *)
let%expect_test "every field out of bounds is reported, not the first" =
  let open Err.Syntax in
  show_labelled "zip (whole record)"
    (Me_limits.Limits.create ~zip:Pt2_zip.Limits.trusted
       Me_limits.Limits.untrusted
    >>= Me_limits.Wire_limits.of_limits ~ceiling:Me_limits.Limits.untrusted);
  show_labelled "two scalars"
    (Me_limits.Limits.create ~max_views:257 ~max_attrs_per_node:257
       Me_limits.Limits.untrusted
    >>= Me_limits.Wire_limits.of_limits ~ceiling:Me_limits.Limits.untrusted);
  [%expect
    {|
    zip (whole record): 5 invalid limits: zip.max_entries = 65536,
                          zip.max_entry_bytes = 1073741824,
                          zip.max_total_bytes = 4294967296,
                          zip.max_path_bytes = 65536, zip.max_path_depth = 64
    two scalars: 2 invalid limits: max_attrs_per_node = 257, max_views = 257
    |}]

let%expect_test "a wire profile is still a profile" =
  let w =
    Err.or_raise ~pp_error:Me_limits.pp_error
      (Me_limits.Wire_limits.of_limits ~ceiling:Me_limits.Limits.untrusted
         Me_limits.Limits.small)
  in
  let l = Me_limits.Wire_limits.limits w in
  Printf.printf "%d %Ld\n" l.Me_limits.Limits.max_session_bytes
    l.Me_limits.Limits.response_live_bytes;
  [%expect {| 524288 45613056 |}]
