(* The parts of [Rewrite] that carry no version-bound abstraction: they read and
   rebuild the group tree and compute dense renamings over plain ids, never
   touching an allocator, a recipe or a snapshot, so moving them out of the
   functor forges nothing. *)

open Graph_ir

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
              Option.map (fun c -> Group.Group c) (rebuild ~is_root:false child)
          | Group.Node id ->
              if Node_id.Set.mem id removed then None else Some (Group.Node id))
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

(* Post-origin ids in canonical order, deduplicated, handed dense values from the
   origin watermark. Origin ids are absent from the table and so map to
   themselves — which is what keeps the two blocks disjoint, everything below the
   watermark staying put and everything at or above it landing inside
   [watermark, watermark + count). Returns the renaming and the new watermark. *)
module Renaming (Id : Core.Tagged_int.S) = struct
  let make ~is_post ~(mark : Id.Next.t) ids =
    let ordered =
      List.fold_left
        (fun (seen, acc) id ->
          if (not (is_post id)) || Id.Set.mem id seen then (seen, acc)
          else (Id.Set.add id seen, id :: acc))
        (Id.Set.empty, []) ids
      |> snd |> List.rev
    in
    let table, next =
      List.fold_left
        (fun (table, next) id ->
          let fresh, next = Id.Next.alloc next in
          (Id.Map.add id fresh table, next))
        (Id.Map.empty, mark) ordered
    in
    ((fun id -> Option.value (Id.Map.find_opt id table) ~default:id), next)
end

module Tensor_renaming = Renaming (Tensor_id)
module Node_renaming = Renaming (Node_id)
module Group_renaming = Renaming (Group_id)

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
