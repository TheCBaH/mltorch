(* Control-flow analysis over any stage's function, given its terminators'
   successors: predecessors, reverse postorder and dominators. Iterative, so a
   long chain of blocks costs no host stack. *)

module B = Mir_id.Block

type t = {
  entry : B.t;
  succs : B.t list B.Map.t;
  preds : B.t list B.Map.t;
  rpo : B.t list;
  rank : int B.Map.t;
  idom : B.t B.Map.t;
}

let find_list m k = Option.value ~default:[] (B.Map.find_opt k m)

let reverse_postorder ~entry succs =
  let seen = ref B.Set.empty and order = ref [] in
  (* explicit stack of (block, remaining successors) *)
  let stack = ref [] in
  let push b =
    seen := B.Set.add b !seen;
    stack := (b, ref (find_list succs b)) :: !stack
  in
  push entry;
  while !stack <> [] do
    match !stack with
    | [] -> ()
    | (b, rest) :: tail -> (
        match !rest with
        | s :: more ->
            rest := more;
            if not (B.Set.mem s !seen) then push s
        | [] ->
            order := b :: !order;
            stack := tail)
  done;
  !order

let make ~entry (blocks : (B.t * B.t list) list) =
  let succs =
    List.fold_left (fun m (b, ss) -> B.Map.add b ss m) B.Map.empty blocks
  in
  let preds =
    List.fold_left
      (fun m (b, ss) ->
        List.fold_left (fun m s -> B.Map.add s (find_list m s @ [ b ]) m) m ss)
      B.Map.empty blocks
  in
  let rpo = reverse_postorder ~entry succs in
  let rank =
    List.fold_left
      (fun (m, i) b -> (B.Map.add b i m, i + 1))
      (B.Map.empty, 0) rpo
    |> fst
  in
  let idom = ref (B.Map.singleton entry entry) in
  let r b = B.Map.find b rank in
  let rec intersect a b =
    if B.equal a b then a
    else if r a > r b then intersect (B.Map.find a !idom) b
    else intersect a (B.Map.find b !idom)
  in
  let changed = ref true in
  while !changed do
    changed := false;
    List.iter
      (fun b ->
        if not (B.equal b entry) then
          let ps =
            List.filter (fun p -> B.Map.mem p !idom) (find_list preds b)
          in
          match ps with
          | [] -> ()
          | first :: rest ->
              let d = List.fold_left intersect first rest in
              if
                match B.Map.find_opt b !idom with
                | Some old -> not (B.equal old d)
                | None -> true
              then (
                idom := B.Map.add b d !idom;
                changed := true))
      rpo
  done;
  { entry; succs; preds; rpo; rank; idom = !idom }

let reachable t b = B.Map.mem b t.rank
let preds t b = find_list t.preds b
let succs t b = find_list t.succs b

(* The immediate dominator, [None] for the entry or an unreachable block. *)
let idom t b = if B.equal b t.entry then None else B.Map.find_opt b t.idom

(* [dominates t a b]: every path from the entry to [b] passes through [a]. *)
let dominates t a b =
  let rec up b =
    B.equal a b || match idom t b with Some d -> up d | None -> false
  in
  up b
