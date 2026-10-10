open Cram_json

let diagnostics ?(prefixed = false) ?(shape = true) s =
  List.iter
    (fun d ->
      if shape || field "code" d <> "unsupported_graph_shape" then
        Printf.printf "%s%s | %s\n"
          (if prefixed then "  diagnostic: " else "")
          (field "code" d) (field "message" d))
    (rows "diagnostics" s)

let capabilities ?(brief = false) ?(diags = false) s =
  List.iter
    (fun c ->
      let key = field "key" c and st = at "status" c in
      if
        (not brief)
        || List.mem key
             [ "stage:source"; "stage:initial_native"; "stage:native4d" ]
      then
        let state = field "state" st in
        let detail =
          match state with
          | "available" -> " " ^ field "kind" (at "payload" st)
          | "unavailable" -> " " ^ field "reason" st
          | _ -> ""
        in
        Printf.printf "%-*s %s%s\n" (if brief then 24 else 28) key state detail)
    (rows "capabilities" s);
  if diags then diagnostics ~prefixed:true ~shape:false s

let source width s =
  List.iter
    (fun n ->
      Printf.printf "%-*s ns=%-6s in=%d\n" width (field "label" n)
        (field "namespace" n)
        (List.length (optional_rows "incomingEdges" n)))
    (operations s "pt2/root")

let params width s =
  List.iter
    (fun n ->
      match attr "params" n with
      | None -> ()
      | Some v ->
          Printf.printf "%-3s %-*s %s\n" (field "id" n) width (field "label" n)
            (str v))
    (nodes s "g/native/000")

let shapes n =
  List.concat_map
    (fun m ->
      List.filter_map
        (fun a -> if field "key" a = "shape" then Some (at "value" a) else None)
        (optional_rows "attrs" m))
    (optional_rows "outputsMetadata" n)

let shape width namespace_width s =
  List.iter
    (fun n ->
      let shape = match List.rev (shapes n) with [] -> "" | v :: _ -> str v in
      Printf.printf "%-3s %-*s %s%s\n" (field "id" n) width (field "label" n)
        (if namespace_width < 0 then ""
         else Printf.sprintf "ns=%-*s " namespace_width (field "namespace" n))
        shape)
    (operations s "g/native/000")

let output_shapes width s =
  List.iter
    (fun n ->
      let xs = shapes n in
      Printf.printf "%-3s %-*s outputs=%d %s\n" (field "id" n) width
        (field "label" n) (List.length xs)
        (list (List.map repr xs)))
    (operations s "g/native/000")

let edges width s =
  List.iter
    (fun n ->
      let edges =
        List.map
          (fun e ->
            tuple
              (List.map
                 (fun key -> quote (field key e))
                 [ "sourceNodeId"; "sourceNodeOutputId"; "targetNodeInputId" ]))
          (optional_rows "incomingEdges" n)
      in
      Printf.printf "%-3s %-*s %s\n" (field "id" n) width (field "label" n)
        (list edges))
    (operations s "g/native/000")

let detail d =
  let g = at "graph" d in
  Printf.printf "%s %s %s\n" (field "collection" d) (field "id" g)
    (py_bool (field "id" (at "view" d) = field "id" g));
  List.iter
    (fun id ->
      let n = by "id" id (rows "nodes" g) in
      let attribute key =
        str (get (Err.of_option (`Metadata_missing key) (attr key n)))
      in
      let role =
        match optional_rows "incomingEdges" n with
        | [] -> "root"
        | e :: _ -> (
            match opt "metadata" e with
            | Jsont.Null _ -> "root"
            | m -> (
                match opt "role" m with Jsont.Null _ -> "root" | v -> str v))
      in
      Printf.printf "  %s %s %s %s %s\n" id (field "label" n)
        (attribute "language") (attribute "constructor") role)
    [ "e0"; "e1"; "e4"; "e55"; "e67" ]

let graph_equality paths =
  match paths with
  | [ a; b; c ] ->
      let a = read a and b = read b and c = read c in
      List.iter
        (fun id ->
          Printf.printf "%s source=archive:%s archive=preloaded:%s\n" id
            (py_bool (equal (graph a id) (graph b id)))
            (py_bool (equal (graph b id) (graph c id))))
        [ "g/native/001"; "g/native4d/000" ]
  | _ -> get (invalid "three graph-equality paths required")

let unbind source s =
  List.iter
    (fun n ->
      if contains (String.lowercase_ascii (field "label" n)) "unbind" then
        if source then (
          Printf.printf "label: %s\n" (field "label" n);
          List.iter
            (fun m ->
              Printf.printf "  slot %s ssa %s\n" (field "id" m)
                (list
                   (List.filter_map
                      (fun a ->
                        if field "key" a = "ssa" then Some (repr (at "value" a))
                        else None)
                      (rows "attrs" m))))
            (optional_rows "outputsMetadata" n))
        else
          let slots =
            List.map (field "id") (optional_rows "outputsMetadata" n)
          in
          Printf.printf "label: %s slots: %s unique: %s\n" (field "label" n)
            (list (List.map quote slots))
            (py_bool
               (List.length (List.sort_uniq compare slots) = List.length slots)))
    (nodes s (if source then "pt2/root" else "g/native/000"))
