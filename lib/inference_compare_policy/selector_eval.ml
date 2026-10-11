open Inference_compare_schema
module Plan = Inference_compare_plan
module GI = Plan.Graph_index
module TS = Tensor_selector

type ctx = {
  left : GI.t;
  right : GI.t;
  partition : Plan.Partition_check.t;
  limits : Resource_limits.t;
}

let ( let* ) = Err.bind
let idx ctx = function Side.Left -> ctx.left | Side.Right -> ctx.right

(* ---- evidence ------------------------------------------------------------ *)

type evidence = {
  producers : (Node_ref.t * Output_port.t) list;
  consumers : Topology_witness.Use.t list;
  paths : Topology_witness.Path.t list;
  fan_outs : (TS.scope * int64) list;
}

let no_evidence = { producers = []; consumers = []; paths = []; fan_outs = [] }

let union a b =
  {
    producers = a.producers @ b.producers;
    consumers = a.consumers @ b.consumers;
    paths = a.paths @ b.paths;
    fan_outs = a.fan_outs @ b.fan_outs;
  }

(* ---- tensor selectors ---------------------------------------------------- *)

type env = {
  graph : GI.t;
  cluster : Node_ref.Set.t;
  boundary : Plan.Partition_check.Boundary.t option;
}

let env_for ctx (site : Comparison_site.t) side =
  {
    graph = idx ctx side;
    cluster =
      (match Plan.Partition_check.boundary ctx.partition site.unit side with
      | Some b -> Node_ref.Set.of_list b.nodes
      | None -> Node_ref.Set.empty);
    boundary = Plan.Partition_check.boundary ctx.partition site.unit side;
  }

let in_scope env (scope : TS.scope) n =
  match scope with
  | Endpoint_graph -> true
  | Within_cluster -> Node_ref.Set.mem n env.cluster

let node_matches env (p : TS.Node_pattern.t) n =
  (match p.node with None -> true | Some r -> Node_ref.equal r n)
  &&
  match p.operator with
  | None -> true
  | Some op -> (
      match GI.node env.graph n with
      | Some d -> Operator_ref.equal d.operator op
      | None -> false)

let port_in = function
  | None -> fun _ -> true
  | Some p -> fun q -> Input_port.equal p q

let port_out = function
  | None -> fun _ -> true
  | Some p -> fun q -> Output_port.equal p q

let mem_tensor t l = List.exists (Tensor_ref.equal t) l

let rec eval_tensor env (sel : TS.t) (t : Tensor_ref.t) : evidence option =
  match sel with
  | Any_tensor -> Some no_evidence
  | Tensor r -> if Tensor_ref.equal r t then Some no_evidence else None
  | Boundary b ->
      let hit =
        match b with
        | Cluster_input -> (
            match env.boundary with
            | Some bd -> mem_tensor t bd.inputs
            | None -> false)
        | Cluster_output -> (
            match env.boundary with
            | Some bd -> mem_tensor t bd.outputs
            | None -> false)
        | Graph_output -> GI.is_final_output env.graph t
      in
      if hit then Some no_evidence else None
  | Produced_by { scope; producer } -> (
      match GI.producer env.graph t with
      | Some (GI.Producer.Node { node; port })
        when in_scope env scope node
             && node_matches env producer.node node
             && port_out producer.port port ->
          Some { no_evidence with producers = [ (node, port) ] }
      | _ -> None)
  | Consumed_by { scope; consumer } -> (
      match
        List.find_opt
          (fun (u : GI.Use.t) ->
            in_scope env scope u.node
            && node_matches env consumer.node u.node
            && port_in consumer.port u.port)
          (GI.consumers env.graph t)
      with
      | Some u ->
          Some
            {
              no_evidence with
              consumers = [ { tensor = t; node = u.node; port = u.port } ];
            }
      | None -> None)
  | Fan_out { scope; count } ->
      let n =
        Int64.of_int
          (List.length
             (List.filter
                (fun (u : GI.Use.t) -> in_scope env scope u.node)
                (GI.consumers env.graph t)))
      in
      let above_min = Int64.compare n count.minimum >= 0 in
      let below_max =
        match count.maximum with
        | None -> true
        | Some m -> Int64.compare n m <= 0
      in
      if above_min && below_max then
        Some { no_evidence with fan_outs = [ (scope, n) ] }
      else None
  | Flows_to path -> flows env path t
  | All_of items ->
      List.fold_left
        (fun acc s ->
          match acc with
          | None -> None
          | Some e -> Option.map (union e) (eval_tensor env s t))
        (Some no_evidence) items
  | Any_of items -> List.find_map (fun s -> eval_tensor env s t) items
  | Not s -> (
      match eval_tensor env s t with Some _ -> None | None -> Some no_evidence)

(* Layered matching: [layer] maps each tensor reachable after i hops to the
   hops (newest first) of the first -- in tensor, then use, then result order --
   path found to it, so a witness is deterministic and each layer costs one
   scan of the uses and transitions of its tensors. *)
and flows env (p : TS.Path.t) t =
  let use_of x (u : GI.Use.t) : Topology_witness.Use.t =
    { tensor = x; node = u.node; port = u.port }
  in
  let step layer (h : TS.Hop.t) =
    Tensor_ref.Map.fold
      (fun x hops acc ->
        List.fold_left
          (fun acc (u : GI.Use.t) ->
            if
              in_scope env p.scope u.node
              && node_matches env h.node u.node
              && port_in h.input_port u.port
            then
              match GI.node env.graph u.node with
              | None -> acc
              | Some d ->
                  let _, acc =
                    List.fold_left
                      (fun (i, acc) r ->
                        let acc =
                          match Output_port.of_index i with
                          | Some q
                            when port_out h.output_port q
                                 && not (Tensor_ref.Map.mem r acc) ->
                              Tensor_ref.Map.add r
                                ({
                                   Topology_witness.Hop.input = use_of x u;
                                   output = r;
                                   port = q;
                                 }
                                :: hops)
                                acc
                          | _ -> acc
                        in
                        (i + 1, acc))
                      (0, acc) d.results
                  in
                  acc
            else acc)
          acc (GI.consumers env.graph x))
      layer Tensor_ref.Map.empty
  in
  let final = List.fold_left step (Tensor_ref.Map.singleton t []) p.through in
  Tensor_ref.Map.fold
    (fun x hops acc ->
      match acc with
      | Some _ -> acc
      | None ->
          List.find_map
            (fun (u : GI.Use.t) ->
              if
                in_scope env p.scope u.node
                && node_matches env p.sink.node u.node
                && port_in p.sink.port u.port
              then
                Some
                  {
                    no_evidence with
                    paths = [ { through = List.rev hops; sink = use_of x u } ];
                  }
              else None)
            (GI.consumers env.graph x))
    final None

(* ---- metric selectors ---------------------------------------------------- *)

let anchors (site : Comparison_site.t) = function
  | Side.Left -> site.left_anchors
  | Side.Right -> site.right_anchors

let witness side anchor (e : evidence) : Topology_witness.t =
  {
    side;
    anchor;
    producers = e.producers;
    consumers = e.consumers;
    paths = e.paths;
    fan_out_counts = e.fan_outs;
  }

let compare_witness (a : Topology_witness.t) (b : Topology_witness.t) =
  match Side.compare a.side b.side with
  | 0 -> Tensor_ref.compare a.anchor b.anchor
  | c -> c

let rec eval_site ctx (site : Comparison_site.t) (sel : Metric_selector.t) :
    Topology_witness.t list option =
  match sel with
  | Any_site -> Some []
  | Comparison c ->
      if Id.Comparison.equal c site.comparison then Some [] else None
  | Canonical_value k -> (
      match site.kind with
      | Canonical_candidate c when Id.Canonical_value.equal c k -> Some []
      | _ -> None)
  | Member_pair p -> (
      match site.kind with
      | Member_pair q when Id.Member_pair.equal p q -> Some []
      | _ -> None)
  | Unit u -> if Id.Unit.equal u site.unit then Some [] else None
  | Final_output o ->
      if List.exists (Id.Output_occurrence.equal o) site.final_occurrences then
        Some []
      else None
  | On_side { side; anchors = quantifier; tensor } -> (
      let env = env_for ctx site side in
      let matched =
        List.filter_map
          (fun a ->
            Option.map (fun e -> witness side a e) (eval_tensor env tensor a))
          (anchors site side)
      in
      match quantifier with
      | Any_anchor -> if matched = [] then None else Some matched
      | Every_anchor ->
          let all = anchors site side in
          if all <> [] && List.length matched = List.length all then
            Some matched
          else None)
  | All_of items ->
      List.fold_left
        (fun acc s ->
          match acc with
          | None -> None
          | Some w -> Option.map (fun w' -> w @ w') (eval_site ctx site s))
        (Some []) items
  | Any_of items -> List.find_map (eval_site ctx site) items
  | Not s -> (
      match eval_site ctx site s with Some _ -> None | None -> Some [])

let eval ctx site sel =
  Option.map (List.sort_uniq compare_witness) (eval_site ctx site sel)

(* ---- static checks ------------------------------------------------------- *)

let fail e = Policy_error.fail e

let rec tensor_size (t : TS.t) =
  match t with
  | All_of l | Any_of l -> 1 + List.fold_left (fun a x -> a + tensor_size x) 0 l
  | Not s -> 1 + tensor_size s
  | _ -> 1

let rec tensor_depth (t : TS.t) =
  match t with
  | All_of l | Any_of l ->
      1 + List.fold_left (fun a x -> max a (tensor_depth x)) 0 l
  | Not s -> 1 + tensor_depth s
  | _ -> 1

let rec size (s : Metric_selector.t) =
  match s with
  | All_of l | Any_of l -> 1 + List.fold_left (fun a x -> a + size x) 0 l
  | Not x -> 1 + size x
  | On_side { tensor; _ } -> 1 + tensor_size tensor
  | _ -> 1

let rec depth (s : Metric_selector.t) =
  match s with
  | All_of l | Any_of l -> 1 + List.fold_left (fun a x -> max a (depth x)) 0 l
  | Not x -> 1 + depth x
  | On_side { tensor; _ } -> 1 + tensor_depth tensor
  | _ -> 1

let rec paths_of (t : TS.t) =
  match t with
  | Flows_to p -> [ p ]
  | All_of l | Any_of l -> List.concat_map paths_of l
  | Not s -> paths_of s
  | _ -> []

let rec tensor_sels (s : Metric_selector.t) =
  match s with
  | On_side { tensor; side; _ } -> [ (side, tensor) ]
  | All_of l | Any_of l -> List.concat_map tensor_sels l
  | Not x -> tensor_sels x
  | _ -> []

let rec metric_refs (s : Metric_selector.t) =
  match s with
  | All_of l | Any_of l -> List.concat_map metric_refs l
  | Not x -> metric_refs x
  | x -> [ x ]

let check ctx sites ~rule sel =
  let bad ?comparison () =
    fail (Policy_error.make ~rule ?comparison Invalid_selector)
  in
  let limit what bound actual =
    fail
      (Policy_error.make ~rule
         ~detail:(Limit { limit = what; bound; actual = Int64.of_int actual })
         Resource_exhausted)
  in
  let* () =
    if Int64.compare (Int64.of_int (size sel)) ctx.limits.max_selector_nodes > 0
    then
      limit Policy_error.Limit.Selector_nodes ctx.limits.max_selector_nodes
        (size sel)
    else Ok ()
  in
  let* () =
    if
      Int64.compare (Int64.of_int (depth sel)) ctx.limits.max_selector_depth > 0
    then
      limit Policy_error.Limit.Selector_depth ctx.limits.max_selector_depth
        (depth sel)
    else Ok ()
  in
  let tensor_sels = tensor_sels sel in
  let* () =
    let rec go = function
      | [] -> Ok ()
      | (_, t) :: rest -> (
          match
            List.find_opt
              (fun (p : TS.Path.t) ->
                Int64.compare
                  (Int64.of_int (List.length p.through))
                  ctx.limits.max_path_length
                > 0)
              (paths_of t)
          with
          | Some p ->
              limit Policy_error.Limit.Path_length ctx.limits.max_path_length
                (List.length p.through)
          | None -> go rest)
    in
    go tensor_sels
  in
  (* References resolve against the bound universes. *)
  let site_exists p = List.exists p sites in
  let* () =
    let rec go = function
      | [] -> Ok ()
      | (s : Metric_selector.t) :: rest ->
          let ok =
            match s with
            | Comparison c ->
                site_exists (fun (x : Comparison_site.t) ->
                    Id.Comparison.equal x.comparison c)
            | Canonical_value k ->
                site_exists (fun x ->
                    match x.kind with
                    | Canonical_candidate c -> Id.Canonical_value.equal c k
                    | _ -> false)
            | Member_pair p ->
                site_exists (fun x ->
                    match x.kind with
                    | Member_pair q -> Id.Member_pair.equal p q
                    | _ -> false)
            | Unit u -> site_exists (fun x -> Id.Unit.equal x.unit u)
            | Final_output o ->
                site_exists (fun x ->
                    List.exists
                      (Id.Output_occurrence.equal o)
                      x.final_occurrences)
            | _ -> true
          in
          if ok then go rest else bad ()
    in
    go (metric_refs sel)
  in
  let tensor_known (r : Tensor_ref.t) =
    List.exists
      (fun g -> Graph_key.equal r.graph (GI.key g) && GI.producer g r <> None)
      [ ctx.left; ctx.right ]
  in
  let node_arity (r : Node_ref.t) =
    List.find_map
      (fun g ->
        if Graph_key.equal r.graph (GI.key g) then
          Option.map
            (fun (d : Node_desc.t) ->
              (List.length d.operands, List.length d.results))
            (GI.node g r)
        else None)
      [ ctx.left; ctx.right ]
  in
  let pattern_ok (p : TS.Node_pattern.t) ~input ~output =
    match p.node with
    | None -> true
    | Some r -> (
        match node_arity r with
        | None -> false
        | Some (ins, outs) -> (
            (match input with
              | Some q -> Input_port.to_index q < ins
              | None -> true)
            &&
            match output with
            | Some q -> Output_port.to_index q < outs
            | None -> true))
  in
  let rec tensor_ok (t : TS.t) =
    match t with
    | Any_tensor | Boundary _ | Fan_out _ -> true
    | Tensor r -> tensor_known r
    | Produced_by { producer; _ } ->
        pattern_ok producer.node ~input:None ~output:producer.port
    | Consumed_by { consumer; _ } ->
        pattern_ok consumer.node ~input:consumer.port ~output:None
    | Flows_to p ->
        List.for_all
          (fun (h : TS.Hop.t) ->
            pattern_ok h.node ~input:h.input_port ~output:h.output_port)
          p.through
        && pattern_ok p.sink.node ~input:p.sink.port ~output:None
    | All_of l | Any_of l -> List.for_all tensor_ok l
    | Not s -> tensor_ok s
  in
  if List.for_all (fun (_, t) -> tensor_ok t) tensor_sels then Ok () else bad ()
