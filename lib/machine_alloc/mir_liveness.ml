(* Liveness and live intervals over a selected virtual program. Block
   parameters are definitions at their block's start; an edge's arguments are
   uses on that edge, at the end of its source block. Order state carries no
   storage and is excluded. Sets are computed to a fixpoint; intervals follow
   linear scan on SSA form (Wimmer and Franz): blocks in reverse postorder,
   every instruction at two positions (operands read at the even one, results
   written at the odd one), ranges built backwards with lifetime holes, and a
   value live into a loop header kept live to the end of the loop's last
   block. *)

open Machine_ir
module VS = Mir_id.Value.Set

let storage (v : Mir_value.t) = Mir_type.has_storage v.Mir_value.ty

module Make (T : Mir_sel.TARGET) = struct
  module S = Mir_sel.Make (T)

  type func = (S.Stage.op, S.Stage.term) Mir_func.t
  type block = (S.Stage.op, S.Stage.term) Mir_block.t

  let ids vs =
    List.fold_left
      (fun s (v : Mir_value.t) ->
        if storage v then VS.add v.Mir_value.id s else s)
      VS.empty vs

  let instr_uses (i : S.Stage.op Mir_instr.t) = S.Stage.operands i.Mir_instr.op
  let term_uses (b : block) = S.Stage.term_values b.Mir_block.terminator
  let edges (b : block) = Mir_sel.Terminator.edges b.Mir_block.terminator

  (* Live-in and live-out sets of every block. *)
  let sets (f : func) =
    let live_in = Hashtbl.create 16 and live_out = Hashtbl.create 16 in
    let get h id =
      Option.value ~default:VS.empty
        (Hashtbl.find_opt h (Mir_id.Block.to_int id))
    in
    let changed = ref true in
    while !changed do
      changed := false;
      List.iter
        (fun (b : block) ->
          let out =
            List.fold_left
              (fun acc (e : Mir_edge.t) ->
                let t = Option.get (Mir_func.find_block f e.Mir_edge.target) in
                VS.union acc
                  (VS.union
                     (VS.diff
                        (get live_in e.Mir_edge.target)
                        (ids t.Mir_block.params))
                     (ids e.Mir_edge.args)))
              VS.empty (edges b)
          in
          let inn =
            List.fold_right
              (fun (i : S.Stage.op Mir_instr.t) live ->
                VS.union
                  (VS.diff live (ids i.Mir_instr.results))
                  (ids (instr_uses i)))
              b.Mir_block.body
              (VS.union out (ids (term_uses b)))
          in
          let inn = VS.diff inn (ids b.Mir_block.params) in
          if
            not
              (VS.equal out (get live_out b.Mir_block.id)
              && VS.equal inn (get live_in b.Mir_block.id))
          then (
            Hashtbl.replace live_out (Mir_id.Block.to_int b.Mir_block.id) out;
            Hashtbl.replace live_in (Mir_id.Block.to_int b.Mir_block.id) inn;
            changed := true))
        (List.rev f.Mir_func.blocks)
    done;
    (get live_in, get live_out)

  module Interval = struct
    type t = {
      value : Mir_value.t;
      ranges : (int * int) list;  (** half-open, ascending, disjoint *)
      uses : int list;  (** ascending use positions *)
    }
  end

  (* Each block's [from, to) positions in the linear (reverse postorder)
     order: two per instruction and two for the block's boundary. *)
  let spans (f : func) =
    let g =
      Mir_graph.make ~entry:f.Mir_func.entry
        (List.map
           (fun (b : block) ->
             ( b.Mir_block.id,
               List.map (fun (e : Mir_edge.t) -> e.Mir_edge.target) (edges b) ))
           f.Mir_func.blocks)
    in
    let order = List.filter_map (Mir_func.find_block f) g.Mir_graph.rpo in
    List.rev
      (snd
         (List.fold_left
            (fun (pos, acc) (b : block) ->
              let n = (2 * List.length b.Mir_block.body) + 2 in
              (pos + n, (b.Mir_block.id, (pos, pos + n)) :: acc))
            (0, []) order))

  (* Linear order, positions and intervals. *)
  let intervals (f : func) =
    let live_in, _ = sets f in
    let g =
      Mir_graph.make ~entry:f.Mir_func.entry
        (List.map
           (fun (b : block) ->
             ( b.Mir_block.id,
               List.map (fun (e : Mir_edge.t) -> e.Mir_edge.target) (edges b) ))
           f.Mir_func.blocks)
    in
    let order = List.filter_map (Mir_func.find_block f) g.Mir_graph.rpo in
    (* block id -> (from, to) positions *)
    let span = Hashtbl.create 16 in
    let _ =
      List.fold_left
        (fun pos (b : block) ->
          let n = (2 * List.length b.Mir_block.body) + 2 in
          Hashtbl.replace span
            (Mir_id.Block.to_int b.Mir_block.id)
            (pos, pos + n);
          pos + n)
        0 order
    in
    let span_of id = Hashtbl.find span (Mir_id.Block.to_int id) in
    let ranges = Hashtbl.create 64
    and uses = Hashtbl.create 64
    and values = Hashtbl.create 64 in
    let note (v : Mir_value.t) =
      Hashtbl.replace values (Mir_id.Value.to_int v.Mir_value.id) v
    in
    let add_range (v : Mir_value.t) a b =
      if a < b then (
        note v;
        let k = Mir_id.Value.to_int v.Mir_value.id in
        Hashtbl.replace ranges k
          ((a, b) :: Option.value ~default:[] (Hashtbl.find_opt ranges k)))
    in
    (* a definition at [a] in the block starting at [from]: the value's ranges
       that open at the block's start open at the definition instead; a
       definition never used still occupies its own position *)
    let shorten ~from (v : Mir_value.t) a =
      let k = Mir_id.Value.to_int v.Mir_value.id in
      let rs = Option.value ~default:[] (Hashtbl.find_opt ranges k) in
      let rs =
        List.filter_map
          (fun (x, y) ->
            if x = from then if y > a then Some (a, y) else None else Some (x, y))
          rs
      in
      if List.exists (fun (x, y) -> x <= a && a < y) rs then
        Hashtbl.replace ranges k rs
      else (
        Hashtbl.replace ranges k rs;
        add_range v a (a + 1))
    in
    let add_use (v : Mir_value.t) p =
      note v;
      let k = Mir_id.Value.to_int v.Mir_value.id in
      Hashtbl.replace uses k
        (p :: Option.value ~default:[] (Hashtbl.find_opt uses k))
    in
    let find_value id = Hashtbl.find_opt values (Mir_id.Value.to_int id) in
    (* every value with storage, so live sets can be turned back into values *)
    List.iter
      (fun (b : block) ->
        List.iter note (List.filter storage b.Mir_block.params);
        List.iter
          (fun (i : S.Stage.op Mir_instr.t) ->
            List.iter note (List.filter storage i.Mir_instr.results))
          b.Mir_block.body)
      f.Mir_func.blocks;
    List.iter
      (fun (b : block) ->
        let from, to_ = span_of b.Mir_block.id in
        let live =
          List.fold_left
            (fun acc (e : Mir_edge.t) ->
              let t = Option.get (Mir_func.find_block f e.Mir_edge.target) in
              VS.union acc
                (VS.union
                   (VS.diff
                      (live_in e.Mir_edge.target)
                      (ids t.Mir_block.params))
                   (ids e.Mir_edge.args)))
            VS.empty (edges b)
        in
        let live = VS.union live (ids (term_uses b)) in
        VS.iter
          (fun id ->
            Option.iter (fun v -> add_range v from to_) (find_value id))
          live;
        List.iter (fun v -> if storage v then add_use v (to_ - 1)) (term_uses b);
        List.iter
          (fun (e : Mir_edge.t) ->
            List.iter
              (fun v -> if storage v then add_use v (to_ - 1))
              e.Mir_edge.args)
          (edges b);
        (* backwards: a result shortens its range to its definition; an
           operand extends a range from the block's start to its use *)
        List.iteri
          (fun k (i : S.Stage.op Mir_instr.t) ->
            let pos = from + 2 + (2 * (List.length b.Mir_block.body - 1 - k)) in
            List.iter
              (fun r -> if storage r then shorten ~from r (pos + 1))
              i.Mir_instr.results;
            List.iter
              (fun v ->
                if storage v then (
                  (* live through the read at [pos] *)
                  add_range v from (pos + 1);
                  add_use v pos))
              (instr_uses i))
          (List.rev b.Mir_block.body);
        List.iter
          (fun p -> if storage p then shorten ~from p from)
          b.Mir_block.params;
        (* a loop header: everything live into it stays live through the loop *)
        List.iter
          (fun (p : block) ->
            let _, pt = span_of p.Mir_block.id in
            if
              pt > to_
              && List.exists
                   (fun (e : Mir_edge.t) ->
                     Mir_id.Block.equal e.Mir_edge.target b.Mir_block.id)
                   (edges p)
            then
              VS.iter
                (fun id ->
                  Option.iter (fun v -> add_range v from pt) (find_value id))
                (live_in b.Mir_block.id))
          order)
      (List.rev order);
    (* normalize: sort and merge each value's ranges *)
    Hashtbl.fold
      (fun k rs acc ->
        let rs = List.sort compare rs in
        let merged =
          List.fold_left
            (fun acc (a, b) ->
              match acc with
              | (c, d) :: rest when a <= d -> (c, max b d) :: rest
              | _ -> (a, b) :: acc)
            [] rs
          |> List.rev
        in
        {
          Interval.value = Hashtbl.find values k;
          ranges = merged;
          uses =
            List.sort_uniq compare
              (Option.value ~default:[] (Hashtbl.find_opt uses k));
        }
        :: acc)
      ranges []
    |> List.sort (fun (a : Interval.t) (b : Interval.t) ->
        compare a.Interval.ranges b.Interval.ranges)
end
