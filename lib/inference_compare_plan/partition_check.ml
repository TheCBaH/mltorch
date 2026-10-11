open Inference_compare_schema

module Boundary = struct
  type t = {
    unit : Id.Unit.t;
    side : Side.t;
    nodes : Node_ref.t list;
    inputs : Tensor_ref.t list;
    outputs : Tensor_ref.t list;
    required_outgoing : Tensor_ref.t list;
    diagnostic_outputs : Tensor_ref.t list;
  }
end

module Unit_map = Map.Make (struct
  type t = Id.Unit.t

  let compare = Id.Unit.compare
end)

type t = {
  boundaries : Boundary.t list;
  owners : Id.Unit.t Node_ref.Map.t * Id.Unit.t Node_ref.Map.t;
  units : Matched_unit.t Unit_map.t;
}

let ( let* ) = Err.bind
let fail = Plan_error.fail

let index_of side ~left ~right =
  match (side : Side.t) with Left -> left | Right -> right

let cluster_of (u : Matched_unit.t) = function
  | Side.Left -> u.left
  | Side.Right -> u.right

(* Ownership of every node of one side, and that no node is claimed twice or
   left unowned. *)
let own side idx (units : Matched_unit.t list) =
  let rec per_unit owners = function
    | [] -> Ok owners
    | (u : Matched_unit.t) :: rest ->
        let cluster = cluster_of u side in
        let rec per_node owners = function
          | [] -> Ok owners
          | (n : Node_ref.t) :: more ->
              if not (Graph_key.equal n.graph (Graph_index.key idx)) then
                fail (Plan_error.make ~side ~unit:u.unit_id ~node:n Stale_graph)
              else if Graph_index.node idx n = None then
                fail
                  (Plan_error.make ~side ~unit:u.unit_id ~node:n Unknown_member)
              else if Node_ref.Map.mem n owners then
                fail
                  (Plan_error.make ~side ~unit:u.unit_id ~node:n Duplicate_owner)
              else per_node (Node_ref.Map.add n u.unit_id owners) more
        in
        let* owners = per_node owners cluster.nodes in
        per_unit owners rest
  in
  let* owners = per_unit Node_ref.Map.empty units in
  let rec complete = function
    | [] -> Ok owners
    | (n : Node_desc.t) :: rest ->
        if Node_ref.Map.mem n.node owners then complete rest
        else fail (Plan_error.make ~side ~node:n.node Missing_owner)
  in
  complete (Graph_index.nodes idx)

let no_duplicates side unit tensors =
  let rec go seen = function
    | [] -> Ok ()
    | (t : Tensor_ref.t) :: rest ->
        if Tensor_ref.Set.mem t seen then
          fail (Plan_error.make ~side ~unit ~tensor:t Duplicate_owner)
        else go (Tensor_ref.Set.add t seen) rest
  in
  go Tensor_ref.Set.empty tensors

let boundary_of side idx owners (u : Matched_unit.t) =
  let cluster = cluster_of u side in
  let unit = u.unit_id in
  let members = Node_ref.Set.of_list cluster.nodes in
  let inside n = Node_ref.Set.mem n members in
  let produced_inside t =
    match Graph_index.producer idx t with
    | Some (Graph_index.Producer.Node { node; _ }) -> inside node
    | _ -> false
  in
  let mem_t set t = Tensor_ref.Set.mem t set in
  (* Nodes in topological order so the recomputed lists are deterministic. *)
  let ordered = List.filter inside (Graph_index.topological idx) in
  let required_inputs =
    let seen = ref Tensor_ref.Set.empty in
    List.concat_map
      (fun nref ->
        match Graph_index.node idx nref with
        | None -> []
        | Some (n : Node_desc.t) ->
            List.filter_map
              (fun t ->
                if produced_inside t || mem_t !seen t then None
                else (
                  seen := Tensor_ref.Set.add t !seen;
                  Some t))
              n.operands)
      ordered
  in
  let required_set = Tensor_ref.Set.of_list required_inputs in
  let supplied_inputs = Tensor_ref.Set.of_list cluster.inputs in
  let* () = no_duplicates side unit cluster.inputs in
  let* () =
    let rec go = function
      | [] -> Ok ()
      | (t : Tensor_ref.t) :: rest ->
          if mem_t supplied_inputs t then go rest
          else fail (Plan_error.make ~side ~unit ~tensor:t Boundary_mismatch)
    in
    go required_inputs
  in
  let* () =
    let rec go = function
      | [] -> Ok ()
      | (t : Tensor_ref.t) :: rest ->
          if mem_t required_set t then go rest
          else if
            (* An input nothing inside reads: either unknown to the graph or
               a value the cluster does not use. *)
            Graph_index.producer idx t = None
          then fail (Plan_error.make ~side ~unit ~tensor:t Unknown_member)
          else fail (Plan_error.make ~side ~unit ~tensor:t Boundary_mismatch)
    in
    go cluster.inputs
  in
  (* Outgoing: produced inside and read outside or returned. *)
  let outgoing t =
    produced_inside t
    && (Graph_index.is_final_output idx t
       || List.exists
            (fun (use : Graph_index.Use.t) -> not (inside use.node))
            (Graph_index.consumers idx t))
  in
  let required_outgoing =
    List.concat_map
      (fun nref ->
        match Graph_index.node idx nref with
        | None -> []
        | Some (n : Node_desc.t) -> List.filter outgoing n.results)
      ordered
  in
  let* () = no_duplicates side unit cluster.outputs in
  let supplied_outputs = Tensor_ref.Set.of_list cluster.outputs in
  let* () =
    let rec go = function
      | [] -> Ok ()
      | (t : Tensor_ref.t) :: rest ->
          if mem_t supplied_outputs t then go rest
          else fail (Plan_error.make ~side ~unit ~tensor:t Boundary_mismatch)
    in
    go required_outgoing
  in
  (* A graph output that is a source passed straight through belongs to the
     unit that declares it as both input and output. *)
  let required_set_out = Tensor_ref.Set.of_list required_outgoing in
  let* () =
    let rec go = function
      | [] -> Ok ()
      | (t : Tensor_ref.t) :: rest ->
          if mem_t required_set_out t then go rest
          else if produced_inside t then go rest
          else if mem_t supplied_inputs t then go rest
          else if Graph_index.producer idx t = None then
            fail (Plan_error.make ~side ~unit ~tensor:t Unknown_member)
          else
            (* Produced elsewhere and not an input: not a declared
               passthrough. *)
            fail (Plan_error.make ~side ~unit ~tensor:t Boundary_mismatch)
    in
    go cluster.outputs
  in
  ignore owners;
  let diagnostic_outputs =
    List.filter (fun t -> not (mem_t required_set_out t)) cluster.outputs
  in
  Ok
    {
      Boundary.unit;
      side;
      nodes = ordered;
      inputs = cluster.inputs;
      outputs = cluster.outputs;
      required_outgoing;
      diagnostic_outputs;
    }

let check ~left ~right (m : Mapping_artifact.t) =
  let units = m.units in
  let* left_owners = own Side.Left left units in
  let* right_owners = own Side.Right right units in
  (* Dispositions must agree with the partition they annotate. *)
  let* () =
    let owner_of (n : Node_ref.t) =
      if Graph_key.equal n.graph (Graph_index.key left) then
        Some (Side.Left, Node_ref.Map.find_opt n left_owners, left)
      else if Graph_key.equal n.graph (Graph_index.key right) then
        Some (Side.Right, Node_ref.Map.find_opt n right_owners, right)
      else None
    in
    let rec go = function
      | [] -> Ok ()
      | (d : Member_disposition.t) :: rest -> (
          match d with
          | Internal_helper { node; owner_unit } -> (
              match owner_of node with
              | None -> fail (Plan_error.make ~node Stale_graph)
              | Some (side, None, _) ->
                  fail (Plan_error.make ~side ~node Unknown_member)
              | Some (side, Some actual, _) ->
                  if Id.Unit.equal actual owner_unit then go rest
                  else
                    fail
                      (Plan_error.make ~side ~unit:owner_unit ~node
                         Boundary_mismatch))
          | Created { node; _ } | Eliminated { node; _ } -> (
              match owner_of node with
              | None -> fail (Plan_error.make ~node Stale_graph)
              | Some (side, None, _) ->
                  fail (Plan_error.make ~side ~node Unknown_member)
              | Some (_, Some _, _) -> go rest)
          | Tensor_not_compared { tensor; _ } ->
              let in_left =
                Graph_key.equal tensor.graph (Graph_index.key left)
                && Graph_index.producer left tensor <> None
              and in_right =
                Graph_key.equal tensor.graph (Graph_index.key right)
                && Graph_index.producer right tensor <> None
              in
              if in_left || in_right then go rest
              else fail (Plan_error.make ~tensor Unknown_member))
    in
    go m.dispositions
  in
  let rec boundaries acc = function
    | [] -> Ok (List.rev acc)
    | (u : Matched_unit.t) :: rest ->
        let* l = boundary_of Left left left_owners u in
        let* r = boundary_of Right right right_owners u in
        boundaries (r :: l :: acc) rest
  in
  let* bs = boundaries [] units in
  let bs =
    List.sort
      (fun (a : Boundary.t) (b : Boundary.t) ->
        match Id.Unit.compare a.unit b.unit with
        | 0 -> Side.compare a.side b.side
        | c -> c)
      bs
  in
  let units_map =
    List.fold_left
      (fun acc (u : Matched_unit.t) -> Unit_map.add u.unit_id u acc)
      Unit_map.empty units
  in
  Ok
    {
      boundaries = bs;
      owners =
        ( Node_ref.Map.map Fun.id left_owners,
          Node_ref.Map.map Fun.id right_owners );
      units = units_map;
    }

let boundaries t = t.boundaries

let boundary t unit side =
  List.find_opt
    (fun (b : Boundary.t) ->
      Id.Unit.equal b.unit unit && Side.equal b.side side)
    t.boundaries

let owner t side n =
  let l, r = t.owners in
  Node_ref.Map.find_opt n (match (side : Side.t) with Left -> l | Right -> r)

let units t = List.map fst (Unit_map.bindings t.units)
let matched_unit t u = Unit_map.find_opt u t.units
let _ = index_of
