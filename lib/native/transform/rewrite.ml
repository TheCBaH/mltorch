(* See rewrite.mli. [apply] is a pipeline of named steps in the order the design
   fixes; each is a small function below and [apply] itself reads as the list. *)

open Graph_ir
open Core.Syntax

type error =
  [ Graph_map.error
  | Graph_view.error
  | `Bad_constant_payload of Tensor_id.t
  | `Constant_payload_overwrite of Tensor_id.t
  | `Cycle of Node_id.t
  | `Discontiguous_allocation
  | `Id_reuse_with_changed_value of Tensor_id.t
  | `Not_a_constant of Tensor_id.t
  | `Overlapping_replacements of Node_id.t
  | `Stale_allocator
  | `Substitution_conflict of Tensor_id.t
  | `Substitution_cycle of Tensor_id.t
  | `Unclaimed_redefinition of Tensor_id.t
  | `Unclaimed_substitution of Tensor_id.t
  | `Unknown_node of Node_id.t ]

let pp_error ppf : [< error ] -> unit = function
  | #Graph_map.error as e -> Graph_map.pp_error ppf e
  | #Graph_view.error as e -> Graph_view.pp_error ppf e
  | `Bad_constant_payload id ->
      Fmt.pf ppf "payload for %a does not match its signature" Tensor_id.pp id
  | `Constant_payload_overwrite id ->
      Fmt.pf ppf "%a is already a constant; a modified constant needs a new id"
        Tensor_id.pp id
  | `Cycle id ->
      Fmt.pf ppf "the rewrite creates a cycle through %a" Node_id.pp id
  | `Discontiguous_allocation ->
      Fmt.string ppf "recipes were planned from a branched allocator"
  | `Id_reuse_with_changed_value id ->
      Fmt.pf ppf "%a is kept but its signature changed; that needs a new id"
        Tensor_id.pp id
  | `Not_a_constant id ->
      Fmt.pf ppf "%a is not a constant graph input" Tensor_id.pp id
  | `Overlapping_replacements id ->
      Fmt.pf ppf "two replacements both claim node %a" Node_id.pp id
  | `Stale_allocator ->
      Fmt.string ppf "the recipe was planned against a different state"
  | `Substitution_conflict id ->
      Fmt.pf ppf "%a is substituted to two different edges" Tensor_id.pp id
  | `Substitution_cycle id ->
      Fmt.pf ppf "substitution of %a is cyclic" Tensor_id.pp id
  | `Unclaimed_redefinition id ->
      Fmt.pf ppf "%a is kept but redefined without a value claim" Tensor_id.pp
        id
  | `Unclaimed_substitution id ->
      Fmt.pf ppf "%a is substituted away without a value claim" Tensor_id.pp id
  | `Unknown_node id -> Fmt.pf ppf "unknown node %a" Node_id.pp id

type 'v t = {
  graph : graph;
  constants : Tensor.packed Tensor_id.Map.t;
  ids : Id_supply.t;
  view : Graph_view.t;
}

type origin = Origin : 'v t -> origin
type 'v step = Step : 'w t * ('v, 'w) Graph_map.t -> 'v step
type 'v allocator = Allocator of Id_supply.t

type 'v recipe = {
  replacements : Recipe.replacement list;
  start : Id_supply.t;
  finish : Id_supply.t;
}

let graph t = t.graph
let constants t = t.constants
let view t = t.view
let allocator t = Allocator t.ids

let fold_result f init l =
  List.fold_left
    (fun acc x -> Core.Syntax.( let* ) acc (fun acc -> f acc x))
    (Core.return init) l

(* ---- payload validation -------------------------------------------------- *)

let same_shape a b =
  List.for_all (fun ax -> Dim.equal (Vec6.get a ax) (Vec6.get b ax)) Axis.all

let fmt_name (Payload.Fmt f) = Payload.fmt_name f

let payload_matches (sg : Tensor_sig.t) (Tensor.Tensor payload) =
  same_shape sg.shape payload.Tensor.shape
  && String.equal (fmt_name sg.fmt)
       (Payload.fmt_name payload.Tensor.payload.fmt)

let check_payloads g pairs =
  fold_result
    (fun seen (id, payload) ->
      let* () =
        if Tensor_id.Set.mem id seen then Core.fail (`Bad_constant_payload id)
        else Core.return ()
      in
      let* () =
        if Graph_ir.input_kind g id = Input.Constant then Core.return ()
        else Core.fail (`Not_a_constant id)
      in
      let* () =
        match Tensor_id.Map.find_opt id g.Graph.tensors with
        | Some sg when payload_matches sg payload -> Core.return ()
        | _ -> Core.fail (`Bad_constant_payload id)
      in
      Core.return (Tensor_id.Set.add id seen))
    Tensor_id.Set.empty pairs
  |> fun r ->
  let+ _ = r in
  ()

let origin ?(constants = []) g =
  let* view = (Graph_view.of_graph g :> (Graph_view.t, error) Core.result) in
  let* () = check_payloads g constants in
  let table =
    List.fold_left
      (fun m (id, p) -> Tensor_id.Map.add id p m)
      Tensor_id.Map.empty constants
  in
  Core.return
    (Origin { graph = g; constants = table; ids = Id_supply.of_graph g; view })

(* ---- planning ------------------------------------------------------------ *)

let plan _state (Allocator ids) builder =
  let (), replacements, finish = Recipe.run builder ids in
  Core.return ({ replacements; start = ids; finish }, Allocator finish)

let merge a b =
  let* () =
    if Id_supply.equal a.finish b.start then Core.return ()
    else Core.fail `Discontiguous_allocation
  in
  let a_nodes =
    List.fold_left
      (fun acc (r : Recipe.replacement) -> Node_id.Set.union acc r.remove)
      Node_id.Set.empty a.replacements
  in
  let* () =
    fold_result
      (fun () (r : Recipe.replacement) ->
        match Node_id.Set.choose_opt (Node_id.Set.inter a_nodes r.remove) with
        | Some id -> Core.fail (`Overlapping_replacements id)
        | None -> Core.return ())
      () b.replacements
  in
  Core.return
    {
      replacements = a.replacements @ b.replacements;
      start = a.start;
      finish = b.finish;
    }

let pp_recipe fmt r =
  Fmt.pf fmt "@[<v>%a@]"
    Fmt.(list ~sep:cut Recipe.pp_replacement)
    r.replacements

(* ---- substitution -------------------------------------------------------- *)

(* Union every replacement's rewiring, then take the transitive normal form: two
   independently matched no-ops give [t2 := t1] and [t1 := t0], and applying
   that once would leave [t2] pointing at an edge nothing defines. *)
let normalise_subst replacements =
  let* raw =
    fold_result
      (fun acc (r : Recipe.replacement) ->
        fold_result
          (fun acc (from, onto) ->
            match Tensor_id.Map.find_opt from acc with
            | Some existing when not (Tensor_id.equal existing onto) ->
                Core.fail (`Substitution_conflict from)
            | _ -> Core.return (Tensor_id.Map.add from onto acc))
          acc
          (Tensor_id.Map.bindings r.subst))
      Tensor_id.Map.empty replacements
  in
  let rec follow seen id =
    match Tensor_id.Map.find_opt id raw with
    | None -> Core.return id
    | Some next ->
        if Tensor_id.Set.mem next seen then Core.fail (`Substitution_cycle id)
        else follow (Tensor_id.Set.add next seen) next
  in
  fold_result
    (fun acc (from, _) ->
      let+ terminal = follow (Tensor_id.Set.singleton from) from in
      Tensor_id.Map.add from terminal acc)
    Tensor_id.Map.empty
    (Tensor_id.Map.bindings raw)

let subst_of map id = Option.value (Tensor_id.Map.find_opt id map) ~default:id

(* ---- assembling the new graph -------------------------------------------- *)

let removed_nodes replacements =
  List.fold_left
    (fun acc (r : Recipe.replacement) -> Node_id.Set.union acc r.remove)
    Node_id.Set.empty replacements

(* Inserted nodes get their ids here, which is why a recipe never names one. *)
let stamp ids replacements =
  List.fold_left
    (fun (ids, acc) (r : Recipe.replacement) ->
      let ids, nodes =
        List.fold_left
          (fun (ids, nodes) (ins : Recipe.insertion) ->
            let id, ids = Id_supply.node ids in
            ( ids,
              ({ Node.id; op = ins.op; outputs = ins.outputs }, ins.from)
              :: nodes ))
          (ids, []) r.insert
      in
      (ids, (r, List.rev nodes) :: acc))
    (ids, []) replacements
  |> fun (ids, acc) -> (ids, List.rev acc)

(* Splice each replacement's new nodes at its first removed node; a replacement
   that removes nothing appends. Order is fixed up by the topological sort. *)
let splice (g : graph) stamped =
  let removed = removed_nodes (List.map fst stamped) in
  let first_of (r : Recipe.replacement) =
    List.find_opt
      (fun (n : node) -> Node_id.Set.mem n.Node.id r.remove)
      g.Graph.nodes
    |> Option.map (fun (n : node) -> n.Node.id)
  in
  let anchored =
    List.filter_map
      (fun (r, nodes) ->
        Option.map (fun anchor -> (anchor, List.map fst nodes)) (first_of r))
      stamped
  in
  let floating =
    List.concat_map
      (fun (r, nodes) -> if first_of r = None then List.map fst nodes else [])
      stamped
  in
  let kept =
    List.concat_map
      (fun (n : node) ->
        let here =
          List.concat_map
            (fun (anchor, nodes) ->
              if Node_id.equal anchor n.Node.id then nodes else [])
            anchored
        in
        here @ if Node_id.Set.mem n.Node.id removed then [] else [ n ])
      g.Graph.nodes
  in
  kept @ floating

let apply_subst_node subst (n : node) =
  {
    n with
    Node.op = Graph_ir.map_operands (subst_of subst) n.Node.op;
    outputs = List.map (subst_of subst) n.Node.outputs;
  }

let dedup_ids l =
  List.fold_left
    (fun (seen, acc) id ->
      if Tensor_id.Set.mem id seen then (seen, acc)
      else (Tensor_id.Set.add id seen, id :: acc))
    (Tensor_id.Set.empty, []) l
  |> snd |> List.rev

(* ---- group maintenance --------------------------------------------------- *)

(* Each replacement's insertions land in the nearest group containing everything
   it touched — the root when it spans siblings — or in a fresh child of it. *)
let placements view ids stamped =
  List.fold_left
    (fun (ids, acc) ((r : Recipe.replacement), nodes) ->
      let touched =
        Node_id.Set.elements r.remove
        @ List.concat_map (fun (_, from) -> from) nodes
      in
      let anchor = Graph_view.common_group view touched in
      let new_ids = List.map (fun ((n : node), _) -> n.Node.id) nodes in
      match r.placement with
      | _ when new_ids = [] -> (ids, acc)
      | Recipe.Inherit -> (ids, (anchor, `Nodes new_ids) :: acc)
      | Recipe.New_group label ->
          let gid, ids = Id_supply.group ids in
          (ids, (anchor, `Group (gid, label, new_ids)) :: acc))
    (ids, []) stamped
  |> fun (ids, acc) -> (ids, List.rev acc)

let rebuild_groups (root : Group.t) ~removed ~placements ~position =
  let additions gid =
    List.concat_map
      (fun (target, what) ->
        if not (Group_id.equal target gid) then []
        else
          match what with
          | `Nodes ids -> List.map (fun id -> Group.Node id) ids
          | `Group (id, label, ids) ->
              [
                Group.Group
                  {
                    Group.id;
                    label;
                    items = List.map (fun i -> Group.Node i) ids;
                  };
              ])
      placements
  in
  (* An emptied group carries no information, so it is pruned; the root always
     survives even if the graph ends up with no nodes at all. *)
  let rec rebuild ~is_root (grp : Group.t) =
    let items =
      List.filter_map
        (function
          | Group.Node id ->
              if Node_id.Set.mem id removed then None else Some (Group.Node id)
          | Group.Group child ->
              Option.map (fun c -> Group.Group c) (rebuild ~is_root:false child))
        grp.Group.items
      @ additions grp.Group.id
    in
    if items = [] && not is_root then None
    else Some { grp with Group.items = sort_items items }
  and sort_items items =
    let key = function
      | Group.Node id -> position id
      | Group.Group g ->
          List.fold_left
            (fun acc item ->
              min acc
                (match item with Group.Node id -> position id | _ -> acc))
            max_int g.Group.items
    in
    List.stable_sort (fun a b -> Int.compare (key a) (key b)) items
  in
  Option.value
    (rebuild ~is_root:true root)
    ~default:{ root with Group.items = [] }

(* ---- preserved-id checks ------------------------------------------------- *)

let same_sig (a : Tensor_sig.t) (b : Tensor_sig.t) =
  same_shape a.shape b.shape
  && String.equal (fmt_name a.fmt) (fmt_name b.fmt)
  && a.quant = b.quant

(* Two definitions are "the same" when their operands land in the same value
   cluster, so a consumer merely rewired to an equal edge is unchanged. *)
let same_definition ~rep old_view new_view id =
  match (Graph_view.def old_view id, Graph_view.def new_view id) with
  | None, None ->
      Graph_ir.input_kind (Graph_view.graph old_view) id
      = Graph_ir.input_kind (Graph_view.graph new_view) id
  | Some (a : node), Some (b : node) ->
      Graph_ir.map_operands rep a.Node.op = Graph_ir.map_operands rep b.Node.op
  | _ -> false

(* An id may be kept only for the exact same tensor, so this runs BEFORE the
   result is validated as a graph: a recipe that takes over an id with a
   different shape should be told it reused an id, not that some node's output
   signature disagrees with its op. *)
let check_signatures ~old_tensors ~new_tensors =
  fold_result
    (fun () (id, sg) ->
      match Tensor_id.Map.find_opt id new_tensors with
      | Some new_sg when not (same_sig sg new_sg) ->
          Core.fail (`Id_reuse_with_changed_value id)
      | _ -> Core.return ())
    ()
    (Tensor_id.Map.bindings old_tensors)

let check_preserved ~rep ~claimed old_view new_view =
  let old_g = Graph_view.graph old_view and new_g = Graph_view.graph new_view in
  fold_result
    (fun () (id, _) ->
      if not (Tensor_id.Map.mem id new_g.Graph.tensors) then Core.return ()
      else if same_definition ~rep old_view new_view id then Core.return ()
      else if claimed id then Core.return ()
      else Core.fail (`Unclaimed_redefinition id))
    ()
    (Tensor_id.Map.bindings old_g.Graph.tensors)

(* ---- claim propagation --------------------------------------------------- *)

(* After a fold declares its boundary [Equivalent], every edge downstream is
   too; leaving them implicitly [Identical] would have a verifier assert
   bit-equality on the model's final output. *)
let propagate ~explicit ~old_tensors (new_g : graph) =
  List.fold_left
    (fun acc (n : node) ->
      let operand_claim id =
        Option.value
          (Tensor_id.Map.find_opt id acc)
          ~default:Correspondence.Identical
      in
      let incoming =
        List.fold_left
          (fun c id -> Correspondence.join c (operand_claim id))
          Correspondence.Identical
          (Graph_ir.operands n.Node.op)
      in
      List.fold_left
        (fun (acc, i) out ->
          let acc =
            if Tensor_id.Map.mem out acc then acc
            else if not (Tensor_id.Map.mem out old_tensors) then acc
            else
              let claim =
                Output_transfer.transfer incoming
                  (Output_transfer.classify n.Node.op ~output:i)
              in
              if claim = Correspondence.Identical then acc
              else Tensor_id.Map.add out claim acc
          in
          (acc, i + 1))
        (acc, 0) n.Node.outputs
      |> fst)
    explicit new_g.Graph.nodes

(* ---- apply --------------------------------------------------------------- *)

let tensor_ids (g : graph) =
  Tensor_id.Map.fold
    (fun id _ acc -> Tensor_id.Set.add id acc)
    g.Graph.tensors Tensor_id.Set.empty

let apply state recipe =
  let old_g = state.graph and old_view = state.view in
  (* 1. the recipe must have been planned against exactly this state *)
  let* () =
    if Id_supply.equal recipe.start state.ids then Core.return ()
    else Core.fail `Stale_allocator
  in
  let* _claimed_nodes =
    fold_result
      (fun seen (r : Recipe.replacement) ->
        fold_result
          (fun seen id ->
            if Graph_view.node old_view id = None then
              Core.fail (`Unknown_node id)
            else if Node_id.Set.mem id seen then
              Core.fail (`Overlapping_replacements id)
            else Core.return (Node_id.Set.add id seen))
          seen
          (Node_id.Set.elements r.remove))
      Node_id.Set.empty recipe.replacements
  in
  let* () =
    fold_result
      (fun () (r : Recipe.replacement) ->
        fold_result
          (fun () (id, _) ->
            if Graph_ir.input_kind old_g id = Input.Constant then
              Core.fail (`Constant_payload_overwrite id)
            else Core.return ())
          () r.constants)
      () recipe.replacements
  in
  (* 2. one normalised substitution, with every surviving source claimed *)
  let* subst = normalise_subst recipe.replacements in
  let claims =
    List.concat_map
      (fun (r : Recipe.replacement) -> r.value_claims)
      recipe.replacements
  in
  let claimed id =
    List.exists (fun (src, _, _) -> Tensor_id.equal src id) claims
  in
  let* () =
    fold_result
      (fun () (from, _) ->
        if Tensor_id.Map.mem from old_g.Graph.tensors && not (claimed from) then
          Core.fail (`Unclaimed_substitution from)
        else Core.return ())
      ()
      (Tensor_id.Map.bindings subst)
  in
  (* 3-4. stamp, splice, rewire *)
  let ids, stamped = stamp recipe.finish recipe.replacements in
  let removed = removed_nodes recipe.replacements in
  let nodes = splice old_g stamped |> List.map (apply_subst_node subst) in
  (* 5. edges: surviving signatures, then the ones the recipe defines *)
  let sub = subst_of subst in
  let tensors =
    Tensor_id.Map.fold
      (fun id sg acc ->
        if Tensor_id.equal (sub id) id then Tensor_id.Map.add id sg acc else acc)
      old_g.Graph.tensors Tensor_id.Map.empty
  in
  (* A declared signature wins over the surviving one, so a recipe that takes
     over an id with a different shape produces a graph that says so — and is
     then rejected by the id-identity check below, rather than quietly keeping
     the old signature and shipping a graph whose edge contradicts its op. *)
  let tensors =
    List.fold_left
      (fun acc (r : Recipe.replacement) ->
        List.fold_left
          (fun acc (sg : Tensor_sig.t) ->
            let id = sub sg.id in
            Tensor_id.Map.add id { sg with Tensor_sig.id } acc)
          acc r.tensors)
      tensors recipe.replacements
  in
  let new_constants =
    List.concat_map
      (fun (r : Recipe.replacement) -> r.constants)
      recipe.replacements
  in
  let inputs =
    dedup_ids (List.map sub old_g.Graph.inputs @ List.map fst new_constants)
  in
  let outputs = dedup_ids (List.map sub old_g.Graph.outputs) in
  let referenced =
    List.concat_map
      (fun (n : node) -> n.Node.outputs @ Graph_ir.operands n.Node.op)
      nodes
    @ outputs
    |> Tensor_id.Set.of_list
  in
  (* A constant nobody reads is gone; a user input is kept even if unused,
     because the graph's signature is externally meaningful. *)
  let kind_of id =
    if List.exists (fun (c, _) -> Tensor_id.equal c id) new_constants then
      Input.Constant
    else Graph_ir.input_kind old_g id
  in
  let inputs =
    List.filter
      (fun id -> Tensor_id.Set.mem id referenced || kind_of id = Input.Input)
      inputs
  in
  let input_kinds =
    List.fold_left
      (fun acc id ->
        match kind_of id with
        | Input.Constant -> Tensor_id.Map.add id Input.Constant acc
        | Input.Input -> acc)
      Tensor_id.Map.empty inputs
  in
  let live = Tensor_id.Set.union referenced (Tensor_id.Set.of_list inputs) in
  let tensors =
    Tensor_id.Map.filter (fun id _ -> Tensor_id.Set.mem id live) tensors
  in
  (* A recipe-bound payload is held to the same contract [origin] holds its own
     to, against the signature the recipe declares for the edge. Without it a
     pass that computes a parameter itself — batch-norm folding is the first —
     could bind data whose shape or format contradicts its edge, and the
     mismatch would surface only at evaluation, far from the recipe. *)
  let* () =
    fold_result
      (fun () (id, payload) ->
        match Tensor_id.Map.find_opt id tensors with
        | Some sg when payload_matches sg payload -> Core.return ()
        | _ -> Core.fail (`Bad_constant_payload id))
      () new_constants
  in
  let* () =
    check_signatures ~old_tensors:old_g.Graph.tensors ~new_tensors:tensors
  in
  (* 6. order, then structure *)
  let* nodes = (Graph_view.topo_sort nodes :> (node list, error) Core.result) in
  let position =
    let table =
      List.fold_left
        (fun (i, acc) (n : node) -> (i + 1, Node_id.Map.add n.Node.id i acc))
        (0, Node_id.Map.empty) nodes
      |> snd
    in
    fun id -> Option.value (Node_id.Map.find_opt id table) ~default:max_int
  in
  let ids, placements = placements old_view ids stamped in
  let root = rebuild_groups old_g.Graph.root ~removed ~placements ~position in
  let new_g = { Graph.nodes; root; tensors; inputs; input_kinds; outputs } in
  (* 7. the result must itself be a graph we would accept *)
  let* new_view =
    (Graph_view.of_graph new_g :> (Graph_view.t, error) Core.result)
  in
  (* 8. preserved ids: same tensor, or an explicit claim.

     The claim clusters are built FIRST, because "did this definition really
     change" is a question about value clusters, not about raw pairs: after
     trimming a chain, a consumer rewired from t2 to t0 is unchanged exactly
     when t0 and t2 ended up in one cluster. Each cluster also closes over
     identity — a surviving target is its own predecessor too — which is what
     turns [{t1} -> {t0}] into [{t0,t1} -> {t0}] and keeps the relation an
     equivalence rather than leaving t0 both explicitly mapped to and implicitly
     identical. *)
  let explicit_pairs = claims in
  let base_values =
    Correspondence.of_clusters
      (List.map
         (fun (src, dst, rel) ->
           (* The claim names the edge the recipe wired to; where that edge was
              itself substituted away, the surviving one is its normal form. *)
           let dst = sub dst in
           let closure =
             if Tensor_id.Map.mem dst old_g.Graph.tensors then
               Tensor_id.Set.of_list [ src; dst ]
             else Tensor_id.Set.singleton src
           in
           {
             Correspondence.Cluster.src = closure;
             dst = Tensor_id.Set.singleton dst;
             label = rel;
           })
         explicit_pairs)
  in
  let rep =
    let table =
      List.fold_left
        (fun acc (c : Correspondence.Cluster.t) ->
          let members = Tensor_id.Set.union c.src c.dst in
          match Tensor_id.Set.min_elt_opt members with
          | None -> acc
          | Some canonical ->
              Tensor_id.Set.fold
                (fun id acc -> Tensor_id.Map.add id canonical acc)
                members acc)
        Tensor_id.Map.empty
        (Correspondence.clusters base_values)
    in
    fun id -> Option.value (Tensor_id.Map.find_opt id table) ~default:id
  in
  let* () = check_preserved ~rep ~claimed old_view new_view in
  (* 9. claims, propagated forward *)
  let explicit =
    List.fold_left
      (fun acc (_, dst, rel) -> Tensor_id.Map.add dst rel acc)
      Tensor_id.Map.empty explicit_pairs
  in
  let propagated = propagate ~explicit ~old_tensors:old_g.Graph.tensors new_g in
  (* 10. the mapping *)
  let old_ids = tensor_ids old_g and new_ids = tensor_ids new_g in
  let pair_clusters = Correspondence.clusters base_values in
  (* Propagation only ever weakens a claim, so an edge the recipe already spoke
     about keeps the recipe's cluster. *)
  let propagated_clusters =
    Tensor_id.Map.fold
      (fun id rel acc ->
        if Tensor_id.Set.mem id old_ids && not (Tensor_id.Map.mem id explicit)
        then Correspondence.pair id id rel :: acc
        else acc)
      propagated []
  in
  let mentioned id =
    List.exists
      (fun (c : Correspondence.Cluster.t) ->
        Tensor_id.Set.mem id c.src || Tensor_id.Set.mem id c.dst)
      pair_clusters
  in
  let deleted =
    Tensor_id.Set.fold
      (fun id acc ->
        if Tensor_id.Set.mem id new_ids || mentioned id then acc
        else Correspondence.delete id :: acc)
      old_ids []
  in
  let created =
    Tensor_id.Set.fold
      (fun id acc ->
        if Tensor_id.Set.mem id old_ids || mentioned id then acc
        else Correspondence.create id :: acc)
      new_ids []
  in
  let node_clusters =
    List.concat_map
      (fun ((r : Recipe.replacement), nodes) ->
        let accounted =
          List.concat_map (fun (_, from) -> from) nodes |> Node_id.Set.of_list
        in
        List.map
          (fun ((n : node), from) -> Node_map.fused ~from n.Node.id)
          nodes
        @ (Node_id.Set.diff r.remove accounted
          |> Node_id.Set.elements |> List.map Node_map.delete))
      stamped
  in
  let provenance =
    List.concat_map
      (fun (r : Recipe.replacement) -> r.provenance)
      recipe.replacements
    |> Provenance.of_list
  in
  let map =
    {
      Graph_map.values =
        Correspondence.of_clusters
          (pair_clusters @ propagated_clusters @ deleted @ created);
      nodes = Node_map.of_clusters node_clusters;
      provenance;
    }
  in
  let* () =
    (Graph_map.validate map ~src:old_g ~dst:new_g :> (unit, error) Core.result)
  in
  let constants =
    List.fold_left
      (fun acc (id, payload) -> Tensor_id.Map.add id payload acc)
      state.constants new_constants
  in
  let constants =
    Tensor_id.Map.filter (fun id _ -> Tensor_id.Set.mem id live) constants
  in
  Core.return (Step ({ graph = new_g; constants; ids; view = new_view }, map))

(* ---- terminal packing ----------------------------------------------------- *)

module Int_map = Map.Make (Int)

(* Post-origin ids in canonical order, deduplicated, handed dense values from the
   origin watermark. Origin ids are absent from the table and so map to
   themselves — which is what keeps the two blocks disjoint, everything below the
   watermark staying put and everything at or above it landing inside
   [watermark, watermark + count). Returns the renaming and the new watermark. *)
let renaming ~is_post ~to_int ~of_int ~mark ids =
  let ordered =
    List.fold_left
      (fun (seen, acc) id ->
        let i = to_int id in
        if (not (is_post id)) || Int_map.mem i seen then (seen, acc)
        else (Int_map.add i () seen, i :: acc))
      (Int_map.empty, []) ids
    |> snd |> List.rev
  in
  let table, next =
    List.fold_left
      (fun (table, next) i -> (Int_map.add i next table, next + 1))
      (Int_map.empty, mark) ordered
  in
  ( (fun id ->
      match Int_map.find_opt (to_int id) table with
      | Some packed -> of_int packed
      | None -> id),
    next )

(* Canonical tensor order: graph inputs, then each node's outputs in topological
   order. The trailing sweep over [tensors] keeps the renaming total even for a
   graph carrying a signature that nothing defines or reads. *)
let tensor_order (g : graph) =
  g.Graph.inputs
  @ List.concat_map (fun (n : node) -> n.Node.outputs) g.Graph.nodes
  @ List.map fst (Tensor_id.Map.bindings g.Graph.tensors)

let rec group_order (grp : Group.t) =
  grp.Group.id
  :: List.concat_map
       (function Group.Node _ -> [] | Group.Group c -> group_order c)
       grp.Group.items

let rec rename_group ~node_id ~group_id (grp : Group.t) =
  {
    grp with
    Group.id = group_id grp.Group.id;
    items =
      List.map
        (function
          | Group.Node id -> Group.Node (node_id id)
          | Group.Group c -> Group.Group (rename_group ~node_id ~group_id c))
        grp.Group.items;
  }

let pack state =
  let g = state.graph in
  let t_mark, n_mark, g_mark = Id_supply.origin_marks state.ids in
  let tensor_id, tensor_next =
    renaming
      ~is_post:(Id_supply.is_post state.ids)
      ~to_int:Tensor_id.to_int ~of_int:Tensor_id.of_int ~mark:t_mark
      (tensor_order g)
  in
  let node_id, node_next =
    renaming
      ~is_post:(Id_supply.is_post_node state.ids)
      ~to_int:Node_id.to_int ~of_int:Node_id.of_int ~mark:n_mark
      (List.map (fun (n : node) -> n.Node.id) g.Graph.nodes)
  in
  let group_id, group_next =
    renaming
      ~is_post:(Id_supply.is_post_group state.ids)
      ~to_int:Group_id.to_int ~of_int:Group_id.of_int ~mark:g_mark
      (group_order g.Graph.root)
  in
  let new_g =
    {
      Graph.nodes =
        List.map
          (fun (n : node) ->
            {
              Node.id = node_id n.Node.id;
              op = Graph_ir.map_operands tensor_id n.Node.op;
              outputs = List.map tensor_id n.Node.outputs;
            })
          g.Graph.nodes;
      root = rename_group ~node_id ~group_id g.Graph.root;
      tensors =
        Tensor_id.Map.fold
          (fun id sg acc ->
            let id = tensor_id id in
            Tensor_id.Map.add id { sg with Tensor_sig.id } acc)
          g.Graph.tensors Tensor_id.Map.empty;
      inputs = List.map tensor_id g.Graph.inputs;
      input_kinds =
        Tensor_id.Map.fold
          (fun id kind acc -> Tensor_id.Map.add (tensor_id id) kind acc)
          g.Graph.input_kinds Tensor_id.Map.empty;
      outputs = List.map tensor_id g.Graph.outputs;
    }
  in
  (* A bulk renaming is exactly the kind of edit that can silently produce a
     graph nobody would accept, so the result goes through the trust boundary
     like any other. *)
  let* view =
    (Graph_view.of_graph new_g :> (Graph_view.t, error) Core.result)
  in
  let constants =
    Tensor_id.Map.fold
      (fun id payload acc -> Tensor_id.Map.add (tensor_id id) payload acc)
      state.constants Tensor_id.Map.empty
  in
  (* Only ids that actually moved are mentioned; the untouched bulk stays
     implicit, which is the whole point of leaving origin ids alone. *)
  let moved_values =
    Tensor_id.Map.fold
      (fun id _ acc ->
        let packed = tensor_id id in
        if Tensor_id.equal id packed then acc
        else Correspondence.pair id packed Correspondence.Identical :: acc)
      g.Graph.tensors []
  in
  let moved_nodes =
    List.filter_map
      (fun (n : node) ->
        let packed = node_id n.Node.id in
        if Node_id.equal n.Node.id packed then None
        else Some (Node_map.pair n.Node.id packed))
      g.Graph.nodes
  in
  let map =
    {
      Graph_map.values = Correspondence.of_clusters moved_values;
      nodes = Node_map.of_clusters moved_nodes;
      provenance = Provenance.empty;
    }
  in
  let* () =
    (Graph_map.validate map ~src:g ~dst:new_g :> (unit, error) Core.result)
  in
  let ids =
    Id_supply.repack state.ids ~tensor:tensor_next ~node:node_next
      ~group:group_next
  in
  Core.return (Step ({ graph = new_g; constants; ids; view }, map))
