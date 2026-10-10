(* JSON setup and observations for end-to-end cram assertions, without Python. *)
open Cram_json
module G = Cram_probe_graph
module S = Cram_probe_session

let width x =
  match int_of_string_opt x with
  | Some n when n >= 0 && n <= 100 -> n
  | _ -> get (invalid "bounded formatting width required")

let fixture = function
  | "group2" -> Cram_group_fixtures.group2 ()
  | "group3" -> Cram_group_fixtures.group3 ()
  | "group5" -> Cram_group_fixtures.group5 ()
  | "group6" -> Cram_group_fixtures.group6 ()
  | "outside6" -> Cram_group_fixtures.outside6 ()
  | "empty-pad" -> Cram_group_fixtures.empty_pad ()
  | "empty-slice" -> Cram_group_fixtures.empty_slice ()
  | "group7" -> Cram_group_fixtures.group7 ()
  | "outside7" -> Cram_group_fixtures.outside7 ()
  | "group8" -> Cram_group_fixtures.group8 ()
  | "unsupported" -> Cram_refusal_fixtures.unsupported ()
  | "malformed" -> Cram_refusal_fixtures.malformed ()
  | "malformed-rows" -> Cram_refusal_fixtures.malformed_rows ()
  | "unbind" -> Cram_refusal_fixtures.unbind ()
  | _ -> get (invalid "unknown fixture")

let run = function
  | [ "fixture"; name ] -> fixture name
  | [ "repeat-g" ] -> print_endline (String.make 300 'g')
  | "graph-equality" :: paths -> G.graph_equality paths
  | [ "caps"; path ] -> G.capabilities (read path)
  | [ "caps-diags"; path ] -> G.capabilities ~diags:true (read path)
  | [ "brief"; path ] -> G.capabilities ~brief:true ~diags:true (read path)
  | [ "source"; n; path ] -> G.source (width n) (read path)
  | [ "params"; n; path ] -> G.params (width n) (read path)
  | [ "edges"; n; path ] -> G.edges (width n) (read path)
  | [ "shapes"; n; path ] -> G.shape (width n) (-1) (read path)
  | [ "shapes-ns"; n; ns; path ] -> G.shape (width n) (width ns) (read path)
  | [ "outputs"; n; path ] -> G.output_shapes (width n) (read path)
  | [ "detail"; path ] -> G.detail (read path)
  | [ "detail-count"; path ] ->
      Printf.printf "%d nodes\n"
        (List.length (rows "nodes" (at "graph" (read path))))
  | [ "js-off"; path ] -> S.js_off (read path)
  | [ "js-present"; path ] -> S.js_present (read path)
  | [ "js-raw"; opt; raw ] ->
      let opt = S.js_of (read opt) and raw = S.js_of (read raw) in
      Printf.printf "raw len %d differs from optimized %s\n" (String.length raw)
        (py_bool (raw <> opt))
  | [ "js-custom"; opt; raw; custom ] ->
      let opt = S.js_of (read opt)
      and raw = S.js_of (read raw)
      and custom = S.js_of (read custom) in
      Printf.printf "custom differs from optimized %s and from raw %s\n"
        (py_bool (custom <> opt))
        (py_bool (custom <> raw))
  | [ "js-equality"; off; opt ] ->
      Printf.printf
        "identical once the js attributes and capability are stripped %s\n"
        (py_bool (equal (S.strip_js (read off)) (S.strip_js (read opt))))
  | [ "constant-equality"; a; b ] ->
      Printf.printf "identical once constant namespaces are ignored %s\n"
        (py_bool
           (equal (S.strip_constants (read a)) (S.strip_constants (read b))))
  | [ "summary"; path ] -> S.summary (read path)
  | [ "source-summary"; path ] -> S.source_summary (read path)
  | [ "capability"; key; path ] -> S.capability (read path) key
  | [ "capability-empty"; key; path ] ->
      S.capability ~missing:"" (read path) key
  | [ "native4d"; path ] -> S.native4d (read path)
  | [ "source-kind"; path ] ->
      print_endline (field "sourceKind" (at "model" (read path)))
  | [ "frontier"; name; path ] -> S.frontier name (read path)
  | [ "source-counts"; path ] -> S.source_counts (read path)
  | [ "namespaces"; path ] -> S.namespaces (read path)
  | [ "outside"; path ] -> S.outside (read path)
  | [ "transitions"; path ] -> S.transitions (read path)
  | [ "flow"; path ] -> S.flow_observations (read path)
  | [ "default-view"; path ] -> print_endline (field "defaultView" (read path))
  | [ "state-views"; path ] -> S.state_views (read path)
  | [ "verification"; path ] -> S.verification (read path)
  | [ "node-data"; path ] -> S.node_data (read path)
  | [ "no-node-data"; path ] -> S.no_node_data (read path)
  | [ "fusion"; path ] -> S.fusion (read path)
  | [ "fanout"; path ] -> S.fanout (read path)
  | [ "collections"; path ] ->
      let xs = arr (read path) in
      let first =
        match xs with x :: _ -> x | [] -> get (invalid "empty collection")
      in
      Printf.printf "list %d %s %d\n" (List.length xs) (field "label" first)
        (List.length (rows "graphs" first))
  | [ "constants"; path ] -> S.constants ~grouped:false (read path)
  | [ "grouped-constants"; path ] -> S.constants ~grouped:true (read path)
  | [ "diagnostics"; path ] ->
      List.iter
        (fun d ->
          Printf.printf "%s | %s | %s | truncated %s\n" (field "code" d)
            (field "message" d)
            (match opt "graph" d with Jsont.Null _ -> "None" | v -> str v)
            (repr (at "truncated" d)))
        (rows "diagnostics" (read path))
  | [ "unsupported-summary"; path ] ->
      let s = read path in
      Printf.printf
        "graphs %s\nviews %s default %s\ncomparisons %s flow %s\nopTargets %s\n"
        (ids (S.graphs s))
        (ids (rows "views" s))
        (field "defaultView" s)
        (repr (at "comparisons" s))
        (repr (opt "flow" s))
        (repr (at "opTargets" (at "model" s)))
  | [ "unsupported-input"; path ] ->
      let s = read path in
      S.capability s "stage:initial_native";
      let d =
        match rows "diagnostics" s with
        | d :: _ -> d
        | [] -> get (invalid "no input diagnostic")
      in
      Printf.printf "%s | %s\n" (field "code" d) (field "message" d)
  | [ "sdpa-diagnostics"; path ] ->
      List.iter
        (fun d ->
          let m = field "message" d in
          if contains m "sdpa" || contains m "batch axis" then
            Printf.printf "%s | %s\n" (field "code" d) m)
        (rows "diagnostics" (read path))
  | [ "unbind-source"; path ] -> G.unbind true (read path)
  | [ "unbind-native"; path ] -> G.unbind false (read path)
  | [ "mobilenetv1"; path ] -> S.mobilenetv1 (read path)
  | _ -> get (invalid "unknown cram observation or arguments")

let () =
  try run (List.tl (Array.to_list Sys.argv)) with
  | Err.Exn.E e ->
      Printf.eprintf "cram observation: %s\n" (Fmt.str "%a" Err.Exn.pp_kind e);
      exit 2
  | Sys_error e ->
      Printf.eprintf "cram observation: %s\n" e;
      exit 2
