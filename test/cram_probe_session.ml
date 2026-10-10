open Cram_json

let graphs s = List.concat_map (rows "graphs") (rows "graphCollections" s)
let flow s = at "flow" s
let dataset s name = by "name" name (rows "nodeDataSets" s)

let summary s =
  Printf.printf "views=%d comparisons=%d capabilities=%d graphs=%d\n"
    (List.length (rows "views" s))
    (List.length (rows "comparisons" s))
    (List.length (rows "capabilities" s))
    (List.length (graphs s))

let source_summary s =
  let model = at "model" s in
  Printf.printf "sourceKind %s\nsha256 %s\n" (field "sourceKind" model)
    (match opt "sourceSha256" model with
    | Jsont.Null _ -> "absent"
    | v -> str v);
  let count p xs = List.length (List.filter p xs) in
  let views = rows "views" s and gs = graphs s in
  Printf.printf
    "views=%d expressionViews=%d comparisons=%d graphs=%d expressionGraphs=%d\n"
    (count primary views)
    (count (fun x -> not (primary x)) views)
    (List.length (rows "comparisons" s))
    (count primary gs)
    (count (fun x -> not (primary x)) gs)

let capability ?(missing = "None") s id =
  let c = cap s id in
  Printf.printf "%s %s\n" (field "state" c)
    (match opt "reason" c with Jsont.Null _ -> missing | v -> str v)

let native4d s =
  Printf.printf "graphs %s\n"
    (list
       (List.map
          (fun g ->
            tuple
              [
                quote (field "id" g);
                string_of_int (List.length (rows "nodes" g));
              ])
          (List.filter primary (graphs s))));
  Printf.printf "views %s\n" (ids (List.filter primary (rows "views" s)));
  Printf.printf "states %s\n" (ids (rows "states" (flow s)));
  Printf.printf "transitions %s\n"
    (list
       (List.map
          (fun t ->
            tuple [ quote (field "id" t); quote (field "kind" (at "kind" t)) ])
          (rows "transitions" (flow s))));
  Printf.printf "native4d %s\n"
    (list [ quote (field "state" (cap s "stage:native4d")) ])

let source_counts s =
  let labels =
    List.map
      (fun n ->
        let x = field "label" n in
        if List.mem x [ "input"; "constant"; "output" ] then x else "op")
      (nodes s "pt2/root")
  in
  Printf.printf "source %s\n" (counts (count_by labels));
  List.iter
    (fun id ->
      let sync = at "sync" (by "id" id (rows "comparisons" s)) in
      Printf.printf
        "%s entries %d matchNodeIdFallback %s showDiffHighlights %s\n" id
        (List.length (rows "entries" sync))
        (repr (at "matchNodeIdFallback" sync))
        (repr (at "showDiffHighlights" sync)))
    [ "c/import"; "c/canonical" ]

let namespaces s =
  let seen =
    List.fold_left
      (fun acc n ->
        let ns = field "namespace" n in
        if ns = "" || List.mem ns acc then acc else acc @ [ ns ])
      [] (nodes s "pt2/root")
  in
  let rec take n xs =
    match (n, xs) with 0, _ | _, [] -> [] | n, x :: xs -> x :: take (n - 1) xs
  in
  print_endline (list (List.map quote (take 6 seen)))

let outside s =
  print_endline
    (list
       (List.filter_map
          (fun d ->
            if field "code" d = "outside_dialect_domain" then
              Some (quote (field "message" d))
            else None)
          (rows "diagnostics" s)));
  Printf.printf "graphs %s\nstates %s\n"
    (ids (graphs s))
    (ids (rows "states" (flow s)))

let transitions s =
  Printf.printf "states %s\n" (ids (rows "states" (flow s)));
  Printf.printf "transitions %s\n"
    (list
       (List.map
          (fun t ->
            tuple
              [
                quote (field "id" t);
                quote (field "kind" (at "kind" t));
                repr (opt "comparison" t);
              ])
          (rows "transitions" (flow s))))

let flow_observations s =
  let f = flow s in
  let gs = nodes s (field "graph" f) in
  Printf.printf "view %s\n"
    (list
       (List.filter_map
          (fun v ->
            if field "kind" v = "flow" then
              Some (tuple [ quote (field "id" v); quote (field "graph" v) ])
            else None)
          (rows "views" s)));
  Printf.printf "capability %s\n"
    (field "graph" (at "payload" (cap s "feature:flow")));
  let n = List.length (rows "transitions" f) in
  Printf.printf
    "nodes %d states+transitions %d\n\
     edges %d 2*transitions %d\n\
     every node has one slot %s\n"
    (List.length gs)
    (List.length (rows "states" f) + n)
    (List.fold_left
       (fun a g -> a + List.length (optional_rows "incomingEdges" g))
       0 gs)
    (2 * n)
    (py_bool
       (List.for_all
          (fun g -> List.length (optional_rows "outputsMetadata" g) = 1)
          gs))

let state_views s =
  List.iter
    (fun st ->
      let v = by "id" (field "view" st) (rows "views" s) in
      Printf.printf "%s -> %s %s graph-agrees %s\n" (field "id" st)
        (field "view" st) (field "kind" v)
        (py_bool (field "graph" v = field "graph" st)))
    (rows "states" (flow s))

let verification s =
  List.iter
    (fun c ->
      let key = field "key" c in
      if List.mem key [ "feature:verification"; "feature:pass_audits" ] then
        let p = at "payload" (at "status" c) in
        if field "kind" p = "verification_summary" then
          Printf.printf "%s %s\n" key
            (list
               (List.map
                  (fun b ->
                    tuple [ quote (field "label" b); repr (at "count" b) ])
                  (rows "verificationSummary" p)))
        else
          let a = at "passAuditStatus" p in
          Printf.printf "%s %s %s %s\n" key
            (field "retainedReports" a)
            (field "omittedReports" a)
            (repr (at "omittedCounts" a)))
    (rows "capabilities" s)

let node_data s =
  let d =
    match rows "nodeDataSets" s with
    | d :: _ -> d
    | [] -> get (invalid "nodeDataSets is empty")
  in
  Printf.printf "nodeData %s over %s %d nodes\n" (field "name" d)
    (field "graph" d)
    (List.length (rows "results" d));
  let gna = get (members (at "groupNodeAttributes" (graph s "g/native/001"))) in
  Printf.printf "groups %d\n" (List.length gna);
  let root =
    get (Err.of_option (`Metadata_missing "root group") (List.assoc_opt "" gna))
  in
  Printf.printf "root   %s\n"
    (list
       (List.map
          (fun (k, v) -> tuple [ quote k; repr v ])
          (List.sort compare (get (members root)))));
  Printf.printf "batch_norm groups %d\n"
    (List.length (List.filter (fun (k, _) -> contains k "batch_norm") gna))

let no_node_data s =
  print_endline
    (list
       (List.filter_map
          (fun c ->
            if
              List.mem (field "key" c)
                [ "feature:verification"; "feature:pass_audits" ]
            then Some (quote (field "state" (at "status" c)))
            else None)
          (rows "capabilities" s)));
  Printf.printf "nodeDataSets %s\ngroupNodeAttributes %s\n"
    (list (List.map (fun d -> quote (field "name" d)) (rows "nodeDataSets" s)))
    (repr (opt "groupNodeAttributes" (graph s "g/native/001")))

let fusion s =
  let overlay =
    match
      rows "edgeOverlaysDataListLeftPane"
        (at "tasksData" (graph s "g/kernel/000"))
    with
    | x :: _ -> x
    | [] -> get (invalid "overlay missing")
  in
  Printf.printf "overlay %s over %s %s\n" (field "name" overlay)
    (field "graphName" overlay)
    (list
       (List.map
          (fun o ->
            tuple
              [
                quote (field "name" o);
                string_of_int (List.length (rows "edges" o));
              ])
          (rows "overlays" overlay)));
  let labels =
    List.map
      (fun r ->
        let label = field "label" (at "value" r) in
        match String.split_on_char '(' label with
        | x :: _ -> String.trim x
        | [] -> label)
      (rows "results" (dataset s "fusion"))
  in
  Printf.printf "placement %s\n" (counts (count_by labels));
  print_endline
    (list
       (List.filter_map
          (fun d ->
            let m = field "message" d in
            if contains m "virtual" then Some (quote m) else None)
          (rows "diagnostics" s)))

let fanout s =
  match
    List.find_opt
      (fun r -> contains (field "label" (at "value" r)) ">= 2")
      (rows "results" (dataset s "fusion"))
  with
  | Some r ->
      Printf.printf "%s %s\n" (field "nodeId" r) (field "label" (at "value" r))
  | None -> ()

let constants ~grouped s =
  let ns = nodes s "pt2/root" in
  Printf.printf "root constants %d\n"
    (List.length
       (List.filter
          (fun n ->
            String.starts_with ~prefix:"const:" (field "id" n)
            && field "namespace" n = "")
          ns));
  if grouped then Printf.printf "op node count unchanged %d\n" (List.length ns)

let strip_constants s =
  map_nodes
    (fun n ->
      if String.starts_with ~prefix:"const:" (field "id" n) then
        update "namespace" (J.string "") n
      else n)
    s

let js_keys = [ "js"; "js_truncated"; "js_unavailable" ]

let js_of s =
  let n = by "id" "out0" (nodes s "expr/g/native/001/n0") in
  str (get (Err.of_option (`Metadata_missing "js") (attr "js" n)))

let js_off s =
  let count =
    List.fold_left
      (fun acc g ->
        List.fold_left
          (fun acc n ->
            acc
            + List.length
                (List.filter (fun (key, _) -> List.mem key js_keys) (attrs n)))
          acc (rows "nodes" g))
      0 (graphs s)
  in
  Printf.printf "js attrs %d %s\n" count
    (field "state" (cap s "feature:generated_js"))

let js_present s =
  let c = cap s "feature:generated_js" in
  Printf.printf "%s %s\n" (field "state" c) (field "kind" (at "payload" c));
  let n = by "id" "out0" (nodes s "expr/g/native/001/n0") in
  Printf.printf "%s %s %s %d\n"
    (py_bool (attr "js" n <> None))
    (py_bool (attr "js_truncated" n <> None))
    (py_bool (attr "js_unavailable" n <> None))
    (String.length (js_of s))

let strip_js s =
  let s =
    map_nodes
      (fun n ->
        update "attrs"
          (J.list
             (List.filter
                (fun a -> not (List.mem (field "key" a) js_keys))
                (optional_rows "attrs" n)))
          n)
      s
  in
  update "capabilities"
    (J.list
       (List.filter
          (fun c -> field "key" c <> "feature:generated_js")
          (rows "capabilities" s)))
    s

let frontier name s =
  List.iter
    (fun key ->
      let st = cap s key in
      if field "state" st = "available" then
        Printf.printf "%s %s available %s\n" name key
          (field "kind" (at "payload" st))
      else
        let reason = field "reason" st in
        let detail =
          match
            List.find_opt
              (fun d -> field "code" d = reason)
              (rows "diagnostics" s)
          with
          | None -> ""
          | Some d -> ": " ^ field "message" d
        in
        Printf.printf "%s %s unavailable %s%s\n" name key reason detail)
    [ "stage:initial_native"; "stage:native4d" ]

let mobilenetv1 s =
  Printf.printf "native4d: %s\nverification: %s\n"
    (field "state" (cap s "stage:native4d"))
    (field "state" (cap s "feature:verification"));
  List.iter
    (fun id ->
      let ns = nodes s id in
      let count name =
        List.length (List.filter (fun n -> field "label" n = name) ns)
      in
      Printf.printf "%s nodes=%d batch_norm=%d add=%d sqrt=%d\n" id
        (List.length ns) (count "Batch_norm") (count "Add") (count "Sqrt"))
    [ "g/native/001"; "g/native4d/000" ];
  Printf.printf "refuted: %s\n"
    (py_bool
       (List.exists
          (fun r -> contains (field "label" r) "refuted")
          (rows "verificationSummary"
             (at "payload" (cap s "feature:verification")))))
