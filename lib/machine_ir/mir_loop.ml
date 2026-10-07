(* Natural loops of a control-flow graph: a back edge's target dominates its
   source, and the loop of a header is the header with every block reaching
   one of its back edges' sources without passing it — back edges to one
   header make one loop. The graph is given as each block's successors, entry
   first; a block unreachable from the entry is in no loop. *)

module Ids = Set.Make (Int)

type t = { header : Mir_id.Block.t; body : Ids.t }

let mem t b = Ids.mem (Mir_id.Block.to_int b) t.body
let key = Mir_id.Block.to_int

(* Each block's dominators, by the iterative dataflow over [blocks]. *)
let dominators ~entry (graph : (Mir_id.Block.t * Mir_id.Block.t list) list) =
  let preds = Hashtbl.create 16 in
  List.iter
    (fun (b, ss) ->
      List.iter
        (fun s ->
          Hashtbl.replace preds (key s)
            (key b :: Option.value ~default:[] (Hashtbl.find_opt preds (key s))))
        ss)
    graph;
  let all = Ids.of_list (List.map (fun (b, _) -> key b) graph) in
  let dom = Hashtbl.create 16 in
  List.iter
    (fun (b, _) ->
      Hashtbl.replace dom (key b)
        (if key b = key entry then Ids.singleton (key b) else all))
    graph;
  let changed = ref true in
  while !changed do
    changed := false;
    List.iter
      (fun (b, _) ->
        let b = key b in
        if b <> key entry then
          let d =
            Ids.add b
              (List.fold_left
                 (fun acc p -> Ids.inter acc (Hashtbl.find dom p))
                 all
                 (Option.value ~default:[] (Hashtbl.find_opt preds b)))
          in
          if not (Ids.equal d (Hashtbl.find dom b)) then (
            Hashtbl.replace dom b d;
            changed := true))
      graph
  done;
  (dom, preds)

let find ~entry graph =
  let dom, preds = dominators ~entry graph in
  let reached = Hashtbl.create 16 in
  let rec reach b =
    if not (Hashtbl.mem reached (key b)) then (
      Hashtbl.replace reached (key b) ();
      List.iter reach
        (Option.value ~default:[]
           (List.find_map
              (fun (x, ss) -> if key x = key b then Some ss else None)
              graph)))
  in
  reach entry;
  let bodies = Hashtbl.create 4 in
  List.iter
    (fun (b, ss) ->
      if Hashtbl.mem reached (key b) then
        List.iter
          (fun h ->
            if Ids.mem (key h) (Hashtbl.find dom (key b)) then (
              let body =
                ref
                  (Ids.add (key h)
                     (match Hashtbl.find_opt bodies (key h) with
                     | Some (_, body) -> body
                     | None -> Ids.empty))
              in
              let rec walk x =
                if Hashtbl.mem reached x && not (Ids.mem x !body) then (
                  body := Ids.add x !body;
                  List.iter walk
                    (Option.value ~default:[] (Hashtbl.find_opt preds x)))
              in
              walk (key b);
              Hashtbl.replace bodies (key h) (h, !body)))
          ss)
    graph;
  Hashtbl.fold (fun _ (header, body) acc -> { header; body } :: acc) bodies []

(* The loops holding [b], innermost (smallest) first. *)
let around loops b =
  List.sort
    (fun x y -> compare (Ids.cardinal x.body) (Ids.cardinal y.body))
    (List.filter (fun l -> mem l b) loops)

(* Whether [l] holds another loop's header. *)
let nests loops l =
  List.exists (fun l' -> key l'.header <> key l.header && mem l l'.header) loops
