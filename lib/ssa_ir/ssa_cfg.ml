(* A program as a control-flow graph: the form the native handoff consumes.
   Blocks carry typed parameters, edges carry the arguments that bind them, and
   effects stay explicit values. [blocks] holds the entry first; the order after
   it carries no meaning. A value is defined once, by a block parameter or an
   operation result, and is visible wherever its definition dominates. *)
type t = {
  buffers : Ssa_buffer.t list;
  entry : Ssa_id.Block.t;
  blocks : Ssa_cfg_block.t list;
  scan_limits : Expr.Scan_limits.t;
  next_value : Ssa_id.Value.Next.t;
}

let find_block t id =
  List.find_opt
    (fun (b : Ssa_cfg_block.t) -> Ssa_id.Block.equal b.id id)
    t.blocks

let find_buffer t id =
  List.find_opt
    (fun (b : Ssa_buffer.t) -> Ssa_id.Buffer.equal b.id id)
    t.buffers

let predecessors t =
  List.fold_left
    (fun acc (b : Ssa_cfg_block.t) ->
      List.fold_left
        (fun acc s ->
          let ps = Option.value ~default:[] (Ssa_id.Block.Map.find_opt s acc) in
          Ssa_id.Block.Map.add s (ps @ [ b.id ]) acc)
        acc
        (Ssa_cfg_terminator.successors b.terminator))
    Ssa_id.Block.Map.empty t.blocks

(* Reverse postorder from the entry: every block after the blocks it can only
   be reached through, except across a back edge. Unreachable blocks are not
   in it. *)
let reverse_postorder t =
  let seen = ref Ssa_id.Block.Set.empty and order = ref [] in
  let rec visit id =
    if not (Ssa_id.Block.Set.mem id !seen) then (
      seen := Ssa_id.Block.Set.add id !seen;
      (match find_block t id with
      | Some b -> List.iter visit (Ssa_cfg_terminator.successors b.terminator)
      | None -> ());
      order := id :: !order)
  in
  visit t.entry;
  !order

(* The immediate dominator of each reachable block but the entry (Cooper,
   Harvey and Kennedy's iteration over reverse postorder). *)
let immediate_dominators t =
  let rpo = reverse_postorder t in
  let index =
    List.mapi (fun i id -> (id, i)) rpo
    |> List.fold_left
         (fun m (id, i) -> Ssa_id.Block.Map.add id i m)
         Ssa_id.Block.Map.empty
  in
  let preds = predecessors t in
  let idom = ref (Ssa_id.Block.Map.singleton t.entry t.entry) in
  let rank id = Ssa_id.Block.Map.find id index in
  let rec intersect a b =
    if Ssa_id.Block.equal a b then a
    else if rank a > rank b then intersect (Ssa_id.Block.Map.find a !idom) b
    else intersect a (Ssa_id.Block.Map.find b !idom)
  in
  let changed = ref true in
  while !changed do
    changed := false;
    List.iter
      (fun id ->
        if not (Ssa_id.Block.equal id t.entry) then
          let ps =
            List.filter
              (fun p -> Ssa_id.Block.Map.mem p !idom)
              (Option.value ~default:[] (Ssa_id.Block.Map.find_opt id preds))
          in
          match ps with
          | [] -> ()
          | first :: rest ->
              let d = List.fold_left intersect first rest in
              if
                match Ssa_id.Block.Map.find_opt id !idom with
                | Some old -> not (Ssa_id.Block.equal old d)
                | None -> true
              then (
                idom := Ssa_id.Block.Map.add id d !idom;
                changed := true))
      rpo
  done;
  Ssa_id.Block.Map.remove t.entry !idom

(* [dominates idom a b]: every path from the entry to [b] passes through [a]. *)
let dominates t idom a b =
  let rec up b =
    Ssa_id.Block.equal a b
    || (not (Ssa_id.Block.equal b t.entry))
       &&
       match Ssa_id.Block.Map.find_opt b idom with
       | Some d -> up d
       | None -> false
  in
  up b

(* An edge from a block with several successors to a block with several
   predecessors: nowhere to put a copy that only this edge needs. *)
let critical_edges t =
  let preds = predecessors t in
  let count id =
    List.length (Option.value ~default:[] (Ssa_id.Block.Map.find_opt id preds))
  in
  List.concat_map
    (fun (b : Ssa_cfg_block.t) ->
      let succ = Ssa_cfg_terminator.successors b.terminator in
      if List.length succ < 2 then []
      else
        List.filter_map
          (fun s -> if count s > 1 then Some (b.id, s) else None)
          succ)
    t.blocks
