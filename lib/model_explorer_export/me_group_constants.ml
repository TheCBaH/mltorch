(* Constant-namespace grouping. See the .mli. *)

module ME = Model_explorer

type mode = Explicit | Grouped

let const_prefix = "const:"
let const_prefix_len = String.length const_prefix

let is_constant_id id =
  String.length id >= const_prefix_len
  && String.equal (String.sub id 0 const_prefix_len) const_prefix

let split_namespace ns =
  if String.equal ns "" then [] else String.split_on_char '/' ns

let join_namespace comps = String.concat "/" comps

(* Element-wise prefix over an arbitrary number of component lists. Only ever
   folded over at least one list ([consumer_namespaces] never stores an empty
   list), so there is no empty-input case to define. *)
let common_prefix lists =
  let rec prefix2 a b =
    match (a, b) with
    | x :: xs, y :: ys when String.equal x y -> x :: prefix2 xs ys
    | _ -> []
  in
  match lists with
  | [] -> []
  | first :: rest -> List.fold_left prefix2 first rest

(* Every node whose id names a constant, mapped to the distinct namespaces of
   the nodes that consume it. Distinct, and in first-seen order: a constant
   fed to the same module twice must not out-vote a single other consumer in
   the common-prefix fold, and the order has no other effect on the result. *)
let consumer_namespaces (g : ME.Graph.t) =
  let table = Hashtbl.create 64 in
  List.iter
    (fun (n : ME.GraphNode.t) ->
      match n.ME.GraphNode.incomingEdges with
      | None -> ()
      | Some edges ->
          List.iter
            (fun (e : ME.IncomingEdge.t) ->
              let src = e.ME.IncomingEdge.sourceNodeId in
              if is_constant_id src then
                let seen =
                  Option.value (Hashtbl.find_opt table src) ~default:[]
                in
                if not (List.mem n.ME.GraphNode.namespace seen) then
                  Hashtbl.replace table src (n.ME.GraphNode.namespace :: seen))
            edges)
    g.ME.Graph.nodes;
  table

(* [None | Some []] (no consumer) falls to [parameters] alongside a real
   ownership conflict rather than staying unchanged: MobileNet's unused
   `num_batches_tracked` buffers are exactly this case, 52 of them, and left
   scattered at namespace [""] they alone kept the root rank as wide as the
   ungrouped baseline. See the .mli. *)
let namespace_for table id =
  match Hashtbl.find_opt table id with
  | None | Some [] -> Some "parameters"
  | Some [ only ] -> Some only
  | Some many -> (
      match common_prefix (List.map split_namespace many) with
      | [] -> Some "parameters"
      | comps -> Some (join_namespace comps))

let apply mode (g : ME.Graph.t) =
  match mode with
  | Explicit -> g
  | Grouped ->
      let table = consumer_namespaces g in
      let nodes =
        List.map
          (fun (n : ME.GraphNode.t) ->
            if is_constant_id n.ME.GraphNode.id then
              match namespace_for table n.ME.GraphNode.id with
              | None -> n
              | Some namespace -> { n with ME.GraphNode.namespace }
            else n)
          g.ME.Graph.nodes
      in
      { g with ME.Graph.nodes }
