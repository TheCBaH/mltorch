open Inference_compare_schema
module Err_ = Plan_error

module Producer = struct
  type t =
    | Node of { node : Node_ref.t; port : Output_port.t }
    | Source of Graph_source.t
end

module Use = struct
  type t = { node : Node_ref.t; port : Input_port.t }
end

type t = {
  side : Side.t;
  graph : Graph_desc.t;
  node_table : Node_desc.t Node_ref.Map.t;
  producers : Producer.t Tensor_ref.Map.t;
  uses : Use.t list Tensor_ref.Map.t;
  specs : Tensor_spec.t Tensor_ref.Map.t;
  finals : Tensor_ref.Set.t;
  order : Node_ref.t list;
}

let ( let* ) = Err.bind
let fail e = Plan_error.fail e

let port_exn = function
  | Some p -> p
  | None -> invalid_arg "Graph_index: port beyond Port.max_index"

let check_limit ~(limits : Resource_limits.t) side limit bound actual =
  ignore limits;
  if Int64.compare actual bound > 0 then
    fail
      (Plan_error.make ~side
         ~detail:(Limit { limit; bound; actual })
         Resource_exhausted)
  else Ok ()

let build ~(limits : Resource_limits.t) ~side (g : Graph_desc.t) =
  let key = g.key in
  let nodes_n = Int64.of_int (List.length g.nodes) in
  let edges_n =
    List.fold_left
      (fun acc (n : Node_desc.t) ->
        acc + List.length n.operands + List.length n.control_predecessors)
      0 g.nodes
    |> Int64.of_int
  in
  let* () = check_limit ~limits side Plan_error.Limit.Nodes limits.max_nodes nodes_n in
  let* () = check_limit ~limits side Plan_error.Limit.Edges limits.max_edges edges_n in
  let same_graph (k : Graph_key.t) = Graph_key.equal k key in
  let stale_node r =
    fail (Plan_error.make ~side ~node:r Stale_graph)
  in
  let stale_tensor r = fail (Plan_error.make ~side ~tensor:r Stale_graph) in
  let tensor_ok (r : Tensor_ref.t) =
    if same_graph r.graph then Ok () else stale_tensor r
  in
  (* Nodes: scope, uniqueness, effects. *)
  let rec index_nodes tbl = function
    | [] -> Ok tbl
    | (n : Node_desc.t) :: rest ->
        if not (same_graph n.node.graph) then stale_node n.node
        else if Node_ref.Map.mem n.node tbl then
          fail (Plan_error.make ~side ~node:n.node Duplicate_owner)
        else (
          match n.effect with
          | Unsupported _ ->
              fail (Plan_error.make ~side ~node:n.node Unsupported_effect)
          | Metadata_assertion | Pure ->
              index_nodes (Node_ref.Map.add n.node n tbl) rest)
  in
  let* node_table = index_nodes Node_ref.Map.empty g.nodes in
  (* Producers: sources first, then node results. *)
  let rec index_sources tbl = function
    | [] -> Ok tbl
    | (s : Graph_source.t) :: rest ->
        let* () = tensor_ok s.tensor in
        if Tensor_ref.Map.mem s.tensor tbl then
          fail (Plan_error.make ~side ~tensor:s.tensor Duplicate_owner)
        else index_sources (Tensor_ref.Map.add s.tensor (Producer.Source s) tbl) rest
  in
  let* producers = index_sources Tensor_ref.Map.empty g.sources in
  let rec index_results producers = function
    | [] -> Ok producers
    | (n : Node_desc.t) :: rest ->
        let rec results i producers = function
          | [] -> Ok producers
          | (t : Tensor_ref.t) :: more ->
              let* () = tensor_ok t in
              if Tensor_ref.Map.mem t producers then
                fail (Plan_error.make ~side ~node:n.node ~tensor:t Duplicate_owner)
              else
                results (i + 1)
                  (Tensor_ref.Map.add t
                     (Producer.Node { node = n.node; port = port_exn (Output_port.of_index i) })
                     producers)
                  more
        in
        let* producers = results 0 producers n.results in
        index_results producers rest
  in
  let* producers = index_results producers g.nodes in
  (* Specs. *)
  let rec index_specs tbl = function
    | [] -> Ok tbl
    | ((t : Tensor_ref.t), spec) :: rest ->
        let* () = tensor_ok t in
        if Tensor_ref.Map.mem t tbl then
          fail (Plan_error.make ~side ~tensor:t Duplicate_owner)
        else if not (Tensor_ref.Map.mem t producers) then
          fail (Plan_error.make ~side ~tensor:t Unknown_member)
        else index_specs (Tensor_ref.Map.add t spec tbl) rest
  in
  let* specs = index_specs Tensor_ref.Map.empty g.tensor_specs in
  (* A source carries its own spec: it must agree with a table entry if any,
     and otherwise supplies it. *)
  let* specs =
    let rec go tbl = function
      | [] -> Ok tbl
      | (s : Graph_source.t) :: rest -> (
          match Tensor_ref.Map.find_opt s.tensor tbl with
          | Some sp when not (Tensor_spec.equal sp s.spec) ->
              fail
                (Plan_error.make ~side ~tensor:s.tensor
                   ~detail:(Spec_pair { expected = s.spec; actual = sp })
                   Boundary_mismatch)
          | Some _ -> go tbl rest
          | None -> go (Tensor_ref.Map.add s.tensor s.spec tbl) rest)
    in
    go specs g.sources
  in
  let* () =
    let rec go = function
      | [] -> Ok ()
      | (n : Node_desc.t) :: rest ->
          let rec results = function
            | [] -> Ok ()
            | t :: more ->
                if Tensor_ref.Map.mem t specs then results more
                else
                  fail (Plan_error.make ~side ~node:n.node ~tensor:t Unknown_member)
          in
          let* () = results n.results in
          go rest
    in
    go g.nodes
  in
  (* Operands and control predecessors resolve. *)
  let* uses =
    let rec nodes uses = function
      | [] -> Ok uses
      | (n : Node_desc.t) :: rest ->
          let rec operands i uses = function
            | [] -> Ok uses
            | (t : Tensor_ref.t) :: more ->
                let* () = tensor_ok t in
                if not (Tensor_ref.Map.mem t producers) then
                  fail (Plan_error.make ~side ~node:n.node ~tensor:t Unknown_member)
                else
                  let use = { Use.node = n.node; port = port_exn (Input_port.of_index i) } in
                  operands (i + 1)
                    (Tensor_ref.Map.update t
                       (fun l -> Some (use :: Option.value l ~default:[]))
                       uses)
                    more
          in
          let* uses = operands 0 uses n.operands in
          let* () =
            let rec ctl = function
              | [] -> Ok ()
              | (p : Node_ref.t) :: more ->
                  if not (same_graph p.graph) then stale_node p
                  else if Node_ref.Map.mem p node_table then ctl more
                  else
                    fail (Plan_error.make ~side ~node:n.node Unknown_member
                          |> fun e -> { e with Plan_error.node = Some p })
            in
            ctl n.control_predecessors
          in
          nodes uses rest
    in
    nodes Tensor_ref.Map.empty g.nodes
  in
  let uses = Tensor_ref.Map.map List.rev uses in
  (* Final outputs and their tree. *)
  let* () =
    let rec go i = function
      | [] -> Ok ()
      | (t : Tensor_ref.t) :: rest ->
          let* () = tensor_ok t in
          if Tensor_ref.Map.mem t producers then go (i + 1) rest
          else
            fail
              (Plan_error.make ~side ~tensor:t ~detail:(Position i) Unknown_member)
    in
    go 0 g.final_outputs
  in
  let* () =
    let flat = Graph_desc.Output_tree.flatten g.output_tree in
    if List.equal Tensor_ref.equal flat g.final_outputs then Ok ()
    else if List.length flat <> List.length g.final_outputs then
      fail
        (Plan_error.make ~side
           ~detail:
             (Expected_actual_count
                { expected = List.length g.final_outputs; actual = List.length flat })
           Output_order_mismatch)
    else
      let rec first i a b =
        match (a, b) with
        | x :: ar, y :: br -> if Tensor_ref.equal x y then first (i + 1) ar br else i
        | _ -> i
      in
      fail
        (Plan_error.make ~side
           ~detail:(Position (first 0 flat g.final_outputs))
           Output_order_mismatch)
  in
  (* Acyclic over data and control edges: Kahn over nodes. *)
  let indegree = ref Node_ref.Map.empty in
  let succs = ref Node_ref.Map.empty in
  let add_edge a b =
    succs :=
      Node_ref.Map.update a (fun l -> Some (b :: Option.value l ~default:[])) !succs;
    indegree :=
      Node_ref.Map.update b (fun d -> Some (1 + Option.value d ~default:0)) !indegree
  in
  List.iter
    (fun (n : Node_desc.t) ->
      indegree := Node_ref.Map.update n.node (fun d -> Some (Option.value d ~default:0)) !indegree;
      List.iter
        (fun t ->
          match Tensor_ref.Map.find_opt t producers with
          | Some (Producer.Node { node = p; _ }) -> add_edge p n.node
          | _ -> ())
        n.operands;
      List.iter (fun p -> add_edge p n.node) n.control_predecessors)
    g.nodes;
  let ready =
    List.filter_map
      (fun (n : Node_desc.t) ->
        if Node_ref.Map.find n.node !indegree = 0 then Some n.node else None)
      g.nodes
  in
  let rec kahn acc queue =
    match queue with
    | [] -> List.rev acc
    | n :: rest ->
        let next =
          List.fold_left
            (fun q s ->
              let d = Node_ref.Map.find s !indegree - 1 in
              indegree := Node_ref.Map.add s d !indegree;
              if d = 0 then s :: q else q)
            []
            (List.rev (Option.value (Node_ref.Map.find_opt n !succs) ~default:[]))
        in
        kahn (n :: acc) (rest @ List.rev next)
  in
  let order = kahn [] ready in
  if List.length order <> List.length g.nodes then
    let stuck =
      List.filter_map
        (fun (n : Node_desc.t) ->
          if Node_ref.Map.find n.node !indegree > 0 then Some n.node else None)
        g.nodes
    in
    fail (Plan_error.make ~side ~detail:(Node_cycle stuck) Unit_cycle)
  else
    Ok
      {
        side;
        graph = g;
        node_table;
        producers;
        uses;
        specs;
        finals = Tensor_ref.Set.of_list g.final_outputs;
        order;
      }

let side t = t.side
let graph t = t.graph
let key t = t.graph.key
let nodes t = t.graph.nodes
let node t r = Node_ref.Map.find_opt r t.node_table
let producer t r = Tensor_ref.Map.find_opt r t.producers
let consumers t r = Option.value (Tensor_ref.Map.find_opt r t.uses) ~default:[]
let spec t r = Tensor_ref.Map.find_opt r t.specs
let final_outputs t = t.graph.final_outputs
let is_final_output t r = Tensor_ref.Set.mem r t.finals

let is_source t r =
  match producer t r with Some (Producer.Source _) -> true | _ -> false

let topological t = t.order
