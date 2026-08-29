(* See rewrite.mli. [apply] is a pipeline of named steps in the order the design
   fixes; each is a small function below and [apply] itself reads as the list. *)

open Graph_ir

(* Takes a [Side.S] rather than a [Dialect.S]: [apply] both REBUILDS a graph,
   which needs the operation table, and BUILDS A MAP, which needs the snapshot
   and transfer table. One argument cannot then disagree with itself about
   [op]. *)
module Make (S : Side.S) = struct
  (* [S.Snapshot] IS [Snapshot.Make (D)] — [Side.S] declares it as that
     application, and applicativity makes the two names denote one module — so a
     snapshot passes to both [Recipe.Make (D)] and [Graph_map.Make_pair (S) (S)]
     with no coercion. *)
  module Snap = S.Snapshot
  module View = Graph_view.Make (S.Dialect)

  (* [S.Snapshot.view] answers in exactly this module, [Side.S] naming the same
     application. *)
  module Rcp = Recipe.Make (S)
  module Rgn = Region.Make (S.Dialect)
  module Gmap = Graph_map.Make_pair (S) (S)

  type node = S.Dialect.op Graph_common.Node.t
  type graph = S.Dialect.op Graph_common.Graph.t

  open Err.Syntax

  type error =
    [ `Bad_constant_payload of Tensor_id.t
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
    | `Unknown_node of Node_id.t
    | Constant_store.error
    | Graph_map.error
    | Rcp.error
    | View.error ]

  let pp_error ppf : [< error ] -> unit = function
    | `Bad_constant_payload id ->
        Fmt.pf ppf "payload for %a does not match its signature" Tensor_id.pp id
    | `Constant_payload_overwrite id ->
        Fmt.pf ppf
          "%a is already a constant; a modified constant needs a new id"
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
        Fmt.pf ppf "%a is substituted away without a value claim" Tensor_id.pp
          id
    | `Unknown_node id -> Fmt.pf ppf "unknown node %a" Node_id.pp id
    | #Constant_store.error as e -> Constant_store.pp_error ppf e
    | #Graph_map.error as e -> Graph_map.pp_error ppf e
    | #Rcp.error as e -> Rcp.pp_error ppf e
    | #View.error as e -> View.pp_error ppf e

  (* The snapshot IS the version: it carries the validated view and the two id
     universes, so a state and the maps into and out of it are indexed by the same
     ['v] and the graph is reachable through it rather than stored twice. *)
  type 'v t = {
    constant_store : Constant_store.t;
    ids : Id_supply.t;
    snapshot : 'v Snap.t;
  }

  type origin = Origin : 'v t -> origin
  type 'v step = Step : 'w t * ('v, 'w) Graph_map.t -> 'v step
  type 'v allocator = Allocator of Id_supply.t

  type 'v recipe = {
    replacements : 'v Rcp.replacement list;
    start : Id_supply.t;
    finish : Id_supply.t;
  }

  let constants t = Constant_store.materialized t.constant_store
  let constant_store t = t.constant_store
  let graph t = Snap.graph t.snapshot
  let snapshot t = t.snapshot
  let view t = Snap.view t.snapshot
  let allocator t = Allocator t.ids

  let fold_result f init l =
    List.fold_left
      (fun acc x -> Err.Syntax.( let* ) acc (fun acc -> f acc x))
      (Err.return init) l

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
          if Tensor_id.Set.mem id seen then Err.fail (`Bad_constant_payload id)
          else Err.return ()
        in
        let* () =
          if Graph_common.input_kind g id = Input.Constant then Err.return ()
          else Err.fail (`Not_a_constant id)
        in
        let* () =
          match Tensor_id.Map.find_opt id g.Graph.tensors with
          | Some sg when payload_matches sg payload -> Err.return ()
          | _ -> Err.fail (`Bad_constant_payload id)
        in
        Err.return (Tensor_id.Set.add id seen))
      Tensor_id.Set.empty pairs
    |> fun r ->
    let+ _ = r in
    ()

  let origin ?(constant_store = Constant_store.empty) ?(constants = []) g =
    let* (Snap.Pack snapshot) = (Snap.create g :> (Snap.packed, error) Err.t) in
    let* () = check_payloads g constants in
    let* constant_store =
      fold_result
        (fun store (id, payload) ->
          match Tensor_id.Map.find_opt id g.Graph.tensors with
          | None -> Err.fail (`Bad_constant_payload id)
          | Some tensor ->
              Constant_store.bind_materialized store ~tensor payload
              |> Err.map_error (fun e -> (e :> error)))
        constant_store constants
    in
    Err.return (Origin { constant_store; ids = Id_supply.of_graph g; snapshot })

  (* ---- planning ------------------------------------------------------------ *)

  (* The state is what makes a recipe version-bound, and now it is also what
     [Rcp.existing] resolves against: a raw id from a matcher enters the typed
     world here or not at all. *)
  let plan state (Allocator ids) builder =
    let* (), replacements, finish =
      (Rcp.run builder state.snapshot ids
        :> (unit * 'v Rcp.replacement list * Id_supply.t, error) Err.t)
    in
    Err.return ({ replacements; start = ids; finish }, Allocator finish)

  let merge a b =
    let* () =
      if Id_supply.equal a.finish b.start then Err.return ()
      else Err.fail `Discontiguous_allocation
    in
    let a_nodes =
      List.fold_left
        (fun acc (r : _ Rcp.replacement) -> Node_id.Set.union acc r.remove)
        Node_id.Set.empty a.replacements
    in
    let* () =
      fold_result
        (fun () (r : _ Rcp.replacement) ->
          match Node_id.Set.choose_opt (Node_id.Set.inter a_nodes r.remove) with
          | Some id -> Err.fail (`Overlapping_replacements id)
          | None -> Err.return ())
        () b.replacements
    in
    Err.return
      {
        replacements = a.replacements @ b.replacements;
        start = a.start;
        finish = b.finish;
      }

  let pp_recipe fmt r =
    Fmt.pf fmt "@[<v>%a@]" Fmt.(list ~sep:cut Rcp.pp_replacement) r.replacements

  (* ---- the typed edit meets the raw graph builder ---------------------------

     Everything below rebuilds a [graph], and [S.Dialect.op] takes raw
     [tensor_ref]s — parameterising the IR is a non-goal
     (.ai/native_transform_versioning.md §7). So a replacement's metadata is
     projected once, here, rather than threaded through the rebuild. Nothing is
     lost by it: the ids that reach the MAP are re-lifted through the two
     snapshots further down, which is where a side confusion would matter. *)

  let subst_pairs (r : _ Rcp.replacement) =
    List.map (fun (a, b) -> (Rcp.raw_edit_edge a, Rcp.raw_edit_edge b)) r.subst

  let claim_triples (r : _ Rcp.replacement) =
    List.map
      (fun (src, dst, rel) -> (Rcp.raw_source src, Rcp.raw_target dst, rel))
      r.value_claims

  let constant_pairs (r : _ Rcp.replacement) =
    List.map (fun (t, p) -> (Rcp.raw_target t, p)) r.constants

  let literal_pairs (r : _ Rcp.replacement) =
    List.map (fun (t, p) -> (Rcp.raw_target t, p)) r.literals

  let deferred_pairs (r : _ Rcp.replacement) =
    List.map (fun (t, op) -> (Rcp.raw_target t, op)) r.deferred

  let provenance_pairs (r : _ Rcp.replacement) =
    List.map
      (fun (sources, dst) ->
        (List.map Rcp.raw_source sources, Rcp.raw_target dst))
      r.provenance

  (* ---- substitution -------------------------------------------------------- *)

  (* Union every replacement's rewiring, then take the transitive normal form: two
     independently matched no-ops give [t2 := t1] and [t1 := t0], and applying
     that once would leave [t2] pointing at an edge nothing defines. *)
  let normalise_subst replacements =
    let* raw =
      fold_result
        (fun acc (r : _ Rcp.replacement) ->
          fold_result
            (fun acc (from, onto) ->
              match Tensor_id.Map.find_opt from acc with
              | Some existing when not (Tensor_id.equal existing onto) ->
                  Err.fail (`Substitution_conflict from)
              | _ -> Err.return (Tensor_id.Map.add from onto acc))
            acc (subst_pairs r))
        Tensor_id.Map.empty replacements
    in
    let rec follow seen id =
      match Tensor_id.Map.find_opt id raw with
      | None -> Err.return id
      | Some next ->
          if Tensor_id.Set.mem next seen then Err.fail (`Substitution_cycle id)
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
      (fun acc (r : _ Rcp.replacement) -> Node_id.Set.union acc r.remove)
      Node_id.Set.empty replacements

  (* Inserted nodes get their ids here, which is why a recipe never names one. *)
  let stamp ids replacements =
    List.fold_left
      (fun (ids, acc) (r : _ Rcp.replacement) ->
        let ids, nodes =
          List.fold_left
            (fun (ids, nodes) (ins : _ Rcp.insertion) ->
              let id, ids = Id_supply.node ids in
              ( ids,
                ( {
                    Node.id;
                    op = ins.op;
                    outputs = List.map Rcp.raw_target ins.outputs;
                  },
                  ins.from )
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
    let first_of (r : _ Rcp.replacement) =
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
      Node.op = S.Dialect.map_operands (subst_of subst) n.Node.op;
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
      (fun (ids, acc) ((r : _ Rcp.replacement), nodes) ->
        let touched =
          Node_id.Set.elements r.remove
          @ List.concat_map (fun (_, from) -> from) nodes
        in
        let anchor = View.common_group view touched in
        let new_ids = List.map (fun ((n : node), _) -> n.Node.id) nodes in
        match r.placement with
        | _ when new_ids = [] -> (ids, acc)
        | Rcp.Inherit -> (ids, (anchor, `Nodes new_ids) :: acc)
        | Rcp.New_group label ->
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
            | Group.Group child ->
                Option.map
                  (fun c -> Group.Group c)
                  (rebuild ~is_root:false child)
            | Group.Node id ->
                if Node_id.Set.mem id removed then None
                else Some (Group.Node id))
          grp.Group.items
        @ additions grp.Group.id
      in
      if items = [] && not is_root then None
      else Some { grp with Group.items = sort_items items }
    and sort_items items =
      let key = function
        | Group.Group g ->
            List.fold_left
              (fun acc item ->
                min acc
                  (match item with
                  | Group.Group _ -> acc
                  | Group.Node id -> position id))
              max_int g.Group.items
        | Group.Node id -> position id
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
    match (View.def old_view id, View.def new_view id) with
    | None, None ->
        Graph_common.input_kind (View.graph old_view) id
        = Graph_common.input_kind (View.graph new_view) id
    | Some (a : node), Some (b : node) ->
        S.Dialect.map_operands rep a.Node.op
        = S.Dialect.map_operands rep b.Node.op
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
            Err.fail (`Id_reuse_with_changed_value id)
        | _ -> Err.return ())
      ()
      (Tensor_id.Map.bindings old_tensors)

  let check_preserved ~rep ~claimed old_view new_view =
    let old_g = View.graph old_view and new_g = View.graph new_view in
    fold_result
      (fun () (id, _) ->
        if not (Tensor_id.Map.mem id new_g.Graph.tensors) then Err.return ()
        else if same_definition ~rep old_view new_view id then Err.return ()
        else if claimed id then Err.return ()
        else Err.fail (`Unclaimed_redefinition id))
      ()
      (Tensor_id.Map.bindings old_g.Graph.tensors)

  (* ---- apply --------------------------------------------------------------- *)

  let apply state recipe =
    let old_snap = state.snapshot in
    let old_g = Snap.graph old_snap and old_view = Snap.view old_snap in
    (* 1. the recipe must have been planned against exactly this state *)
    let* () =
      if Id_supply.equal recipe.start state.ids then Err.return ()
      else Err.fail `Stale_allocator
    in
    let* _claimed_nodes =
      fold_result
        (fun seen (r : _ Rcp.replacement) ->
          fold_result
            (fun seen id ->
              if View.node old_view id = None then Err.fail (`Unknown_node id)
              else if Node_id.Set.mem id seen then
                Err.fail (`Overlapping_replacements id)
              else Err.return (Node_id.Set.add id seen))
            seen
            (Node_id.Set.elements r.remove))
        Node_id.Set.empty recipe.replacements
    in
    let* () =
      fold_result
        (fun () (r : _ Rcp.replacement) ->
          fold_result
            (fun () (id, _) ->
              if Graph_common.input_kind old_g id = Input.Constant then
                Err.fail (`Constant_payload_overwrite id)
              else Err.return ())
            ()
            (constant_pairs r @ literal_pairs r)
          |> fun checked ->
          Err.Syntax.( let* ) checked (fun () ->
              fold_result
                (fun () (id, _) ->
                  if Graph_common.input_kind old_g id = Input.Constant then
                    Err.fail (`Constant_payload_overwrite id)
                  else Err.return ())
                () (deferred_pairs r)))
        () recipe.replacements
    in
    (* 2. one normalised substitution, with every surviving source claimed *)
    let* subst = normalise_subst recipe.replacements in
    let claims = List.concat_map claim_triples recipe.replacements in
    let claimed id =
      List.exists (fun (src, _, _) -> Tensor_id.equal src id) claims
    in
    let* () =
      fold_result
        (fun () (from, _) ->
          if Tensor_id.Map.mem from old_g.Graph.tensors && not (claimed from)
          then Err.fail (`Unclaimed_substitution from)
          else Err.return ())
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
          if Tensor_id.equal (sub id) id then Tensor_id.Map.add id sg acc
          else acc)
        old_g.Graph.tensors Tensor_id.Map.empty
    in
    (* A declared signature wins over the surviving one, so a recipe that takes
       over an id with a different shape produces a graph that says so — and is
       then rejected by the id-identity check below, rather than quietly keeping
       the old signature and shipping a graph whose edge contradicts its op. *)
    let tensors =
      List.fold_left
        (fun acc (r : _ Rcp.replacement) ->
          List.fold_left
            (fun acc (sg : Tensor_sig.t) ->
              let id = sub sg.id in
              Tensor_id.Map.add id { sg with Tensor_sig.id } acc)
            acc r.tensors)
        tensors recipe.replacements
    in
    let new_constants = List.concat_map constant_pairs recipe.replacements in
    let new_literals = List.concat_map literal_pairs recipe.replacements in
    let new_deferred = List.concat_map deferred_pairs recipe.replacements in
    let inputs =
      dedup_ids
        (List.map sub old_g.Graph.inputs
        @ List.map fst new_constants @ List.map fst new_literals
        @ List.map fst new_deferred)
    in
    let outputs = dedup_ids (List.map sub old_g.Graph.outputs) in
    let referenced =
      List.concat_map
        (fun (n : node) -> n.Node.outputs @ S.Dialect.operands n.Node.op)
        nodes
      @ outputs
      |> Tensor_id.Set.of_list
    in
    (* A constant nobody reads is gone; a user input is kept even if unused,
       because the graph's signature is externally meaningful. *)
    let kind_of id =
      if
        List.exists (fun (c, _) -> Tensor_id.equal c id) new_constants
        || List.exists (fun (c, _) -> Tensor_id.equal c id) new_literals
        || List.exists (fun (c, _) -> Tensor_id.equal c id) new_deferred
      then Input.Constant
      else Graph_common.input_kind old_g id
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
          | Some sg when payload_matches sg payload -> Err.return ()
          | _ -> Err.fail (`Bad_constant_payload id))
        ()
        (new_constants @ new_literals)
    in
    let* () =
      check_signatures ~old_tensors:old_g.Graph.tensors ~new_tensors:tensors
    in
    (* 6. order, then structure *)
    let* nodes = (View.topo_sort nodes :> (node list, error) Err.t) in
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
    (* 7. the result must itself be a graph we would accept. The snapshot mints the
       destination version ['w], and every id below enters the map by being found
       in one of the two snapshots — which is how a source id can no longer be
       written into a destination side. *)
    let* (Snap.Pack new_snap) =
      (Snap.create new_g :> (Snap.packed, error) Err.t)
    in
    let new_view = Snap.view new_snap in
    let src_edge id =
      Snap.edge old_snap id
      |> Err.of_option (`Value_endpoint (Cluster_relation.Dangling_src id))
    in
    let dst_edge id =
      Snap.edge new_snap id
      |> Err.of_option (`Value_endpoint (Cluster_relation.Dangling_dst id))
    in
    let src_node id =
      Snap.node old_snap id
      |> Err.of_option (`Node_endpoint (Cluster_relation.Dangling_src id))
    in
    let dst_node id =
      Snap.node new_snap id
      |> Err.of_option (`Node_endpoint (Cluster_relation.Dangling_dst id))
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
    let* base_clusters =
      Err.List.map
        (fun (src, dst, rel) ->
          (* The claim names the edge the recipe wired to; where that edge was
             itself substituted away, the surviving one is its normal form. *)
          let dst = sub dst in
          let* s = src_edge src in
          let+ d = dst_edge dst in
          (* A surviving target is its own predecessor too, which is what turns
             [{t1} -> {t0}] into [{t0,t1} -> {t0}]. The old runtime test "is dst
             an id the source graph has" IS this lift. *)
          let closure =
            match Snap.edge old_snap dst with
            | Some kept -> Correspondence.Set.of_list [ s; kept ]
            | None -> Correspondence.Set.singleton s
          in
          {
            Correspondence.Cluster.src = closure;
            dst = Correspondence.Set.singleton d;
            label = rel;
          })
        explicit_pairs
    in
    let pair_clusters = Correspondence.normalise base_clusters in
    let rep =
      let table =
        List.fold_left
          (fun acc (c : (_, _) Correspondence.Cluster.t) ->
            let members =
              Tensor_id.Set.union
                (Correspondence.raws c.src)
                (Correspondence.raws c.dst)
            in
            match Tensor_id.Set.min_elt_opt members with
            | None -> acc
            | Some canonical ->
                Tensor_id.Set.fold
                  (fun id acc -> Tensor_id.Map.add id canonical acc)
                  members acc)
          Tensor_id.Map.empty pair_clusters
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
    (* THIS dialect's transfer table, not Native's: the claims being propagated
       are over the graph this rewrite just produced. *)
    let propagated =
      S.Transfer.propagate ~explicit
        ~preserved:(fun id -> Tensor_id.Map.mem id old_g.Graph.tensors)
        new_g
    in
    (* 10. the mapping *)
    (* Propagation only ever weakens a claim, so an edge the recipe already spoke
       about keeps the recipe's cluster. *)
    let* propagated_clusters =
      Tensor_id.Map.fold
        (fun id rel acc ->
          let* acc = acc in
          if Tensor_id.Map.mem id explicit then Err.return acc
          else
            match (Snap.edge old_snap id, Snap.edge new_snap id) with
            | Some s, Some d -> Err.return (Correspondence.pair s d rel :: acc)
            | _ -> Err.return acc)
        propagated (Err.return [])
    in
    let mentioned id =
      List.exists
        (fun (c : (_, _) Correspondence.Cluster.t) ->
          Tensor_id.Set.mem id (Correspondence.raws c.src)
          || Tensor_id.Set.mem id (Correspondence.raws c.dst))
        pair_clusters
    in
    let deleted =
      Correspondence.Universe.elements (Snap.edges old_snap)
      |> List.filter_map (fun s ->
          let id = Correspondence.raw s in
          if Snap.edge new_snap id <> None || mentioned id then None
          else Some (Correspondence.delete s))
    in
    let created =
      Correspondence.Universe.elements (Snap.edges new_snap)
      |> List.filter_map (fun d ->
          let id = Correspondence.raw d in
          if Snap.edge old_snap id <> None || mentioned id then None
          else Some (Correspondence.create d))
    in
    let* node_clusters =
      Err.List.fold_left
        (fun acc ((r : _ Rcp.replacement), nodes) ->
          let accounted =
            List.concat_map (fun (_, from) -> from) nodes |> Node_id.Set.of_list
          in
          let* fused =
            Err.List.map
              (fun ((n : node), from) ->
                let* from = Err.List.map src_node from in
                let+ d = dst_node n.Node.id in
                Node_map.fused ~from d)
              nodes
          in
          let+ dropped =
            Err.List.map
              (fun id ->
                let+ s = src_node id in
                Node_map.delete s)
              (Node_id.Set.diff r.remove accounted |> Node_id.Set.elements)
          in
          acc @ fused @ dropped)
        [] stamped
    in
    let* provenance =
      Err.List.fold_left
        (fun acc (sources, dst) ->
          let* sources = Err.List.map src_edge sources in
          let+ d = dst_edge dst in
          Provenance.add ~sources:(Correspondence.Set.of_list sources) d acc)
        Provenance.empty
        (List.concat_map provenance_pairs recipe.replacements)
    in
    let* map =
      Gmap.create ~src:old_snap ~dst:new_snap
        ~values:(pair_clusters @ propagated_clusters @ deleted @ created)
        ~nodes:node_clusters ~provenance
      |> Err.map_error (fun e -> (e :> error))
    in
    let* constant_store =
      Constant_store.restrict_and_rename_exports state.constant_store (fun id ->
          if Tensor_id.Set.mem id live then Some id else None)
      |> Err.map_error (fun e -> (e :> error))
    in
    let* constant_store =
      fold_result
        (fun store (id, payload) ->
          match Tensor_id.Map.find_opt id tensors with
          | None -> Err.fail (`Bad_constant_payload id)
          | Some tensor ->
              Constant_store.bind_materialized store ~tensor payload
              |> Err.map_error (fun e -> (e :> error)))
        constant_store new_constants
    in
    let* constant_store =
      fold_result
        (fun store (id, payload) ->
          match Tensor_id.Map.find_opt id tensors with
          | None -> Err.fail (`Bad_constant_payload id)
          | Some tensor ->
              Constant_store.bind_literal store ~tensor payload
              |> Err.map_error (fun e -> (e :> error)))
        constant_store new_literals
    in
    let* constant_store =
      fold_result
        (fun store (id, op) ->
          match Tensor_id.Map.find_opt id tensors with
          | None -> Err.fail (`Bad_constant_payload id)
          | Some tensor ->
              Constant_store.bind_apply store ~tensor op
              |> Err.map_error (fun e -> (e :> error)))
        constant_store new_deferred
    in
    Err.return (Step ({ constant_store; ids; snapshot = new_snap }, map))

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
        Int_map.find_opt (to_int id) table |> Option.fold ~none:id ~some:of_int),
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
         (function Group.Group c -> group_order c | Group.Node _ -> [])
         grp.Group.items

  let rec rename_group ~node_id ~group_id (grp : Group.t) =
    {
      grp with
      Group.id = group_id grp.Group.id;
      items =
        List.map
          (function
            | Group.Group c -> Group.Group (rename_group ~node_id ~group_id c)
            | Group.Node id -> Group.Node (node_id id))
          grp.Group.items;
    }

  let pack state =
    let old_snap = state.snapshot in
    let g = Snap.graph old_snap in
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
                op = S.Dialect.map_operands tensor_id n.Node.op;
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
    let* (Snap.Pack new_snap) =
      (Snap.create new_g :> (Snap.packed, error) Err.t)
    in
    let* constant_store =
      Constant_store.restrict_and_rename_exports state.constant_store (fun id ->
          Some (tensor_id id))
      |> Err.map_error (fun e -> (e :> error))
    in
    (* Only ids that actually moved are mentioned; the untouched bulk stays
       implicit, which is the whole point of leaving origin ids alone. *)
    let* moved_values =
      Err.List.fold_left
        (fun acc (id, _) ->
          let packed = tensor_id id in
          if Tensor_id.equal id packed then Err.return acc
          else
            match (Snap.edge old_snap id, Snap.edge new_snap packed) with
            | Some s, Some d ->
                Err.return
                  (Correspondence.pair s d Correspondence.Identical :: acc)
            | None, _ ->
                Err.fail (`Value_endpoint (Cluster_relation.Dangling_src id))
            | _, None ->
                Err.fail
                  (`Value_endpoint (Cluster_relation.Dangling_dst packed)))
        []
        (Tensor_id.Map.bindings g.Graph.tensors)
    in
    let* moved_nodes =
      Err.List.fold_left
        (fun acc (n : node) ->
          let packed = node_id n.Node.id in
          if Node_id.equal n.Node.id packed then Err.return acc
          else
            match (Snap.node old_snap n.Node.id, Snap.node new_snap packed) with
            | Some s, Some d -> Err.return (Node_map.pair s d :: acc)
            | None, _ ->
                Err.fail
                  (`Node_endpoint (Cluster_relation.Dangling_src n.Node.id))
            | _, None ->
                Err.fail (`Node_endpoint (Cluster_relation.Dangling_dst packed)))
        [] g.Graph.nodes
    in
    let* map =
      Gmap.create ~src:old_snap ~dst:new_snap ~values:moved_values
        ~nodes:moved_nodes ~provenance:Provenance.empty
      |> Err.map_error (fun e -> (e :> error))
    in
    let ids =
      Id_supply.repack state.ids ~tensor:tensor_next ~node:node_next
        ~group:group_next
    in
    Err.return (Step ({ constant_store; ids; snapshot = new_snap }, map))
end

include Make (Native_side)
