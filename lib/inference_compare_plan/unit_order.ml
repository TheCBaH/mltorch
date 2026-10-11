open Inference_compare_schema

module Edge = struct
  type t = Plan_error.Cycle_edge.t = {
    from_unit : Id.Unit.t;
    to_unit : Id.Unit.t;
    reason : Plan_error.Edge_reason.t;
  }
end

module Lifetime = struct
  type read = { unit : Id.Unit.t; bindings : int; checks : int }

  type t = {
    canonical : Id.Canonical_value.t;
    writer : Id.Unit.t option;
    reads : read list;
    final_uses : int;
    total_uses : int;
  }
end

module Unit_map = Map.Make (struct
  type t = Id.Unit.t

  let compare = Id.Unit.compare
end)

module Unit_set = Set.Make (struct
  type t = Id.Unit.t

  let compare = Id.Unit.compare
end)

type t = {
  order : Id.Unit.t list;
  edges : Edge.t list;
  preds : Unit_set.t Unit_map.t;
  lifetimes : Lifetime.t list;
  consumers : (Id.Canonical_value.t * Id.Unit.t list) list;
}

let ( let* ) = Err.bind
let fail = Plan_error.fail

let compare_reason (a : Plan_error.Edge_reason.t) (b : Plan_error.Edge_reason.t)
    =
  match (a, b) with
  | Canonical x, Canonical y -> Id.Canonical_value.compare x y
  | Canonical _, _ -> -1
  | _, Canonical _ -> 1
  | Control x, Control y -> (
      match Side.compare x.side y.side with
      | 0 -> Node_ref.compare x.node y.node
      | c -> c)
  | Control _, _ -> -1
  | _, Control _ -> 1
  | Data x, Data y -> (
      match Side.compare x.side y.side with
      | 0 -> Tensor_ref.compare x.tensor y.tensor
      | c -> c)

let compare_edge (a : Edge.t) (b : Edge.t) =
  match Id.Unit.compare a.from_unit b.from_unit with
  | 0 -> (
      match Id.Unit.compare a.to_unit b.to_unit with
      | 0 -> compare_reason a.reason b.reason
      | c -> c)
  | c -> c

(* The evidence for a cycle: every edge inside the strongly connected
   component of the smallest unit Kahn's algorithm could not release. A single
   chain of edges would name one reason per hop and could hide that the
   opposing dependencies come from different endpoints; the whole component
   shows each side's data and control edges and the canonical edges alike. *)
let find_cycle stuck (edges : Edge.t list) =
  match Unit_set.min_elt_opt stuck with
  | None -> []
  | Some start ->
      let among =
        List.filter
          (fun (e : Edge.t) ->
            Unit_set.mem e.from_unit stuck && Unit_set.mem e.to_unit stuck)
          edges
      in
      let closure next =
        let rec go seen = function
          | [] -> seen
          | u :: rest ->
              if Unit_set.mem u seen then go seen rest
              else
                let more =
                  List.filter_map
                    (fun (e : Edge.t) ->
                      match next e with
                      | Some (a, b) when Id.Unit.equal a u -> Some b
                      | _ -> None)
                    among
                in
                go (Unit_set.add u seen) (more @ rest)
        in
        go Unit_set.empty [ start ]
      in
      let forward = closure (fun e -> Some (e.from_unit, e.to_unit)) in
      let backward = closure (fun e -> Some (e.to_unit, e.from_unit)) in
      let component = Unit_set.inter forward backward in
      List.filter
        (fun (e : Edge.t) ->
          Unit_set.mem e.from_unit component && Unit_set.mem e.to_unit component)
        among

let compute ~(limits : Resource_limits.t) ~left ~right ~partition
    ~correspondence =
  let units = Partition_check.units partition in
  let unit_count = Int64.of_int (List.length units) in
  let* () =
    if Int64.compare unit_count limits.max_units > 0 then
      fail
        (Plan_error.make
           ~detail:
             (Limit
                { limit = Units; bound = limits.max_units; actual = unit_count })
           Resource_exhausted)
    else Ok ()
  in
  (* Data and control edges, per side, from the node-to-unit ownership. *)
  let side_edges side idx =
    List.concat_map
      (fun (n : Node_desc.t) ->
        match Partition_check.owner partition side n.node with
        | None -> []
        | Some consumer ->
            let data =
              List.filter_map
                (fun t ->
                  match Graph_index.producer idx t with
                  | Some (Graph_index.Producer.Node { node = p; _ }) -> (
                      match Partition_check.owner partition side p with
                      | Some producer when not (Id.Unit.equal producer consumer)
                        ->
                          Some
                            {
                              Edge.from_unit = producer;
                              to_unit = consumer;
                              reason =
                                Plan_error.Edge_reason.Data { side; tensor = t };
                            }
                      | _ -> None)
                  | _ -> None)
                n.operands
            in
            let control =
              List.filter_map
                (fun p ->
                  match Partition_check.owner partition side p with
                  | Some producer when not (Id.Unit.equal producer consumer) ->
                      Some
                        {
                          Edge.from_unit = producer;
                          to_unit = consumer;
                          reason =
                            Plan_error.Edge_reason.Control
                              { side; node = n.node };
                        }
                  | _ -> None)
                n.control_predecessors
            in
            data @ control)
      (Graph_index.nodes idx)
  in
  let canonical_edges =
    List.concat_map
      (fun (p : Correspondence_check.Unit_plan.t) ->
        List.filter_map
          (fun k ->
            match Correspondence_check.writer correspondence k with
            | Some w ->
                Some
                  {
                    Edge.from_unit = w;
                    to_unit = p.unit;
                    reason = Plan_error.Edge_reason.Canonical k;
                  }
            | None -> None)
          p.canonical_reads)
      (Correspondence_check.unit_plans correspondence)
  in
  let edges =
    side_edges Side.Left left @ side_edges Side.Right right @ canonical_edges
    |> List.sort_uniq compare_edge
  in
  let edge_count = Int64.of_int (List.length edges) in
  let* () =
    if Int64.compare edge_count limits.max_edges > 0 then
      fail
        (Plan_error.make
           ~detail:
             (Limit
                { limit = Edges; bound = limits.max_edges; actual = edge_count })
           Resource_exhausted)
    else Ok ()
  in
  (* A unit reading a canonical value it writes is a self-dependency. *)
  let* () =
    match
      List.find_opt
        (fun (e : Edge.t) -> Id.Unit.equal e.from_unit e.to_unit)
        edges
    with
    | Some e ->
        fail
          (Plan_error.make ~unit:e.from_unit ~detail:(Cycle [ e ]) Unit_cycle)
    | None -> Ok ()
  in
  let preds =
    List.fold_left
      (fun acc (e : Edge.t) ->
        Unit_map.update e.to_unit
          (fun s ->
            Some
              (Unit_set.add e.from_unit
                 (Option.value s ~default:Unit_set.empty)))
          acc)
      Unit_map.empty edges
  in
  let indegree =
    ref
      (List.fold_left
         (fun acc u ->
           Unit_map.add u
             (Unit_set.cardinal
                (Option.value
                   (Unit_map.find_opt u preds)
                   ~default:Unit_set.empty))
             acc)
         Unit_map.empty units)
  in
  let succs =
    List.fold_left
      (fun acc (e : Edge.t) ->
        Unit_map.update e.from_unit
          (fun s ->
            Some
              (Unit_set.add e.to_unit (Option.value s ~default:Unit_set.empty)))
          acc)
      Unit_map.empty edges
  in
  (* Kahn with the smallest ready unit first: the order depends only on unit
     identifiers and edges, never on the order the artifact listed them. *)
  let ready =
    ref
      (List.fold_left
         (fun s u ->
           if Unit_map.find u !indegree = 0 then Unit_set.add u s else s)
         Unit_set.empty units)
  in
  let rec drain acc =
    match Unit_set.min_elt_opt !ready with
    | None -> List.rev acc
    | Some u ->
        ready := Unit_set.remove u !ready;
        Unit_set.iter
          (fun s ->
            let d = Unit_map.find s !indegree - 1 in
            indegree := Unit_map.add s d !indegree;
            if d = 0 then ready := Unit_set.add s !ready)
          (Option.value (Unit_map.find_opt u succs) ~default:Unit_set.empty);
        drain (u :: acc)
  in
  let order = drain [] in
  if List.length order <> List.length units then
    let stuck =
      List.fold_left
        (fun s u ->
          if Unit_map.find u !indegree > 0 then Unit_set.add u s else s)
        Unit_set.empty units
    in
    fail
      (Plan_error.make
         ?unit:(Unit_set.min_elt_opt stuck)
         ~detail:(Cycle (find_cycle stuck edges))
         Unit_cycle)
  else
    let position =
      List.mapi (fun i u -> (u, i)) order
      |> List.fold_left (fun acc (u, i) -> Unit_map.add u i acc) Unit_map.empty
    in
    let by_plan us =
      List.sort
        (fun a b ->
          Int.compare (Unit_map.find a position) (Unit_map.find b position))
        us
    in
    let plans = Correspondence_check.unit_plans correspondence in
    let count_checks u k =
      match Correspondence_check.unit_plan correspondence u with
      | None -> 0
      | Some p ->
          List.length
            (List.filter
               (fun (c : Input_check.t) ->
                 Id.Canonical_value.equal c.canonical k)
               p.input_checks)
    in
    let count_bindings (p : Correspondence_check.Unit_plan.t) k =
      let reads_k (b : Input_binding.t) =
        let r = b.from_common in
        let arg = function
          | Recipe.Canonical c -> Id.Canonical_value.equal c k
          | Boundary _ | Conversion_result _ -> false
        in
        arg r.result
        || List.exists
             (fun (n : Recipe.node) -> List.exists arg n.arguments)
             r.nodes
      in
      List.length (List.filter reads_k p.input_bindings)
    in
    let finals = Correspondence_check.final_values correspondence in
    let keys =
      List.concat_map
        (fun (p : Correspondence_check.Unit_plan.t) ->
          p.canonical_reads @ p.canonical_writes)
        plans
      @ finals
      |> List.sort_uniq Id.Canonical_value.compare
    in
    let lifetimes =
      List.map
        (fun k ->
          let reads =
            List.filter_map
              (fun (p : Correspondence_check.Unit_plan.t) ->
                if List.exists (Id.Canonical_value.equal k) p.canonical_reads
                then
                  Some
                    {
                      Lifetime.unit = p.unit;
                      bindings = count_bindings p k;
                      checks = count_checks p.unit k;
                    }
                else None)
              plans
            |> List.sort (fun (a : Lifetime.read) (b : Lifetime.read) ->
                Int.compare
                  (Unit_map.find a.unit position)
                  (Unit_map.find b.unit position))
          in
          let final_uses =
            List.length (List.filter (Id.Canonical_value.equal k) finals)
          in
          let used =
            List.fold_left
              (fun a (r : Lifetime.read) -> a + r.bindings + r.checks)
              0 reads
          in
          {
            Lifetime.canonical = k;
            writer = Correspondence_check.writer correspondence k;
            reads;
            final_uses;
            total_uses = used + final_uses;
          })
        keys
    in
    let consumers =
      List.map
        (fun (l : Lifetime.t) ->
          ( l.canonical,
            by_plan (List.map (fun (r : Lifetime.read) -> r.unit) l.reads) ))
        lifetimes
    in
    Ok { order; edges; preds; lifetimes; consumers }

let order t = t.order
let edges t = t.edges

let predecessors t u =
  match Unit_map.find_opt u t.preds with
  | None -> []
  | Some s -> Unit_set.elements s

let lifetimes t = t.lifetimes

let consumers t k =
  match
    List.find_opt (fun (c, _) -> Id.Canonical_value.equal c k) t.consumers
  with
  | Some (_, us) -> us
  | None -> []
