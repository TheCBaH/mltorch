(* What a kernel asks of the registers and what its allocation pays, for
   feedback to the structured SSA blocking that shaped it: the most values live
   at once in each bank of the selected program, the spill stores and reloads
   in blocks on a cycle of the physical one (its hot loops), the realized frame
   and the helper calls. Counts, never timings: interpreted execution says
   nothing of native speed. *)

open Machine_ir

type t = {
  peak : (Mir_target.Bank.t * int) list;
      (** per register bank, the most values live at one position *)
  hot_stores : (Mir_target.Bank.t * int) list;
      (** per bank, register to frame in a block on a cycle *)
  hot_loads : (Mir_target.Bank.t * int) list;
      (** per bank, frame to register in a block on a cycle *)
  by_depth : (int * int) list;
      (** per loop depth from 1, ascending: the hot stores and loads of every
          bank in blocks that many natural loops contain *)
  innermost : int;
      (** the hot stores and loads in blocks of a loop holding no other *)
  stats : Mir_alloc_stats.t;
  frame : int64 option;  (** the largest realized frame, when realized *)
  helper_calls : int;
}

let total l = List.fold_left (fun n (_, k) -> n + k) 0 l
let hot_spills t = total t.hot_stores + total t.hot_loads

(* Hot stores and loads by loop depth, then those in innermost loops. *)
let pp_depths fmt (by_depth, innermost) =
  match by_depth with
  | [] -> Fmt.string fmt "none"
  | l ->
      Fmt.pf fmt "%a; innermost %d"
        Fmt.(
          list ~sep:(any ", ") (fun fmt (d, n) -> Fmt.pf fmt "depth %d: %d" d n))
        l innermost

let pp fmt t =
  let banks =
    Fmt.(
      list ~sep:(any ", ") (fun fmt (b, n) ->
          Fmt.pf fmt "%s %d" (Mir_target.Bank.name b) n))
  in
  let hot fmt = function [] -> Fmt.string fmt "none" | l -> banks fmt l in
  Fmt.pf fmt "peak %a; hot stores %a, loads %a; %d helper calls; frame %a" banks
    t.peak hot t.hot_stores hot t.hot_loads t.helper_calls
    Fmt.(option ~none:(any "unrealized") (fmt "%Ld bytes"))
    t.frame

(* The blocks of a function that lie on a cycle. *)
let on_cycle (f : (_, _) Mir_phys.Func.t) =
  let succs id =
    match
      List.find_opt
        (fun (b : (_, _) Mir_phys.Block.t) ->
          Mir_id.Block.equal b.Mir_phys.Block.id id)
        f.Mir_phys.Func.blocks
    with
    | Some b -> Mir_phys.Term.successors b.Mir_phys.Block.terminator
    | None -> []
  in
  let reaches_itself id =
    let seen = Hashtbl.create 16 in
    let rec go = function
      | [] -> false
      | b :: rest ->
          if Mir_id.Block.equal b id then true
          else if Hashtbl.mem seen (Mir_id.Block.to_int b) then go rest
          else (
            Hashtbl.replace seen (Mir_id.Block.to_int b) ();
            go (succs b @ rest))
    in
    go (succs id)
  in
  List.filter_map
    (fun (b : (_, _) Mir_phys.Block.t) ->
      if reaches_itself b.Mir_phys.Block.id then Some b else None)
    f.Mir_phys.Func.blocks

(* Each block's natural-loop nest: how many loops contain it, and whether the
   smallest of them holds another loop's header. *)
let nest (f : (_, _) Mir_phys.Func.t) =
  let loops =
    Mir_loop.find ~entry:f.Mir_phys.Func.entry
      (List.map
         (fun (b : (_, _) Mir_phys.Block.t) ->
           ( b.Mir_phys.Block.id,
             Mir_phys.Term.successors b.Mir_phys.Block.terminator ))
         f.Mir_phys.Func.blocks)
  in
  fun id ->
    match Mir_loop.around loops id with
    | [] -> (0, false)
    | l :: _ as mine -> (List.length mine, not (Mir_loop.nests loops l))

module Make (T : Mir_sel.TARGET) = struct
  module Lv = Mir_liveness.Make (T)
  module V = Mir_phys_verify.Make (T)

  let peak (sel : Lv.S.Verified.t) =
    let events = Hashtbl.create 4 in
    List.iter
      (fun f ->
        List.iter
          (fun (i : Lv.Interval.t) ->
            match V.shape i.Lv.Interval.value.Mir_value.ty with
            | Some (((Mir_target.Bank.Fpr | Mir_target.Bank.Gpr) as bank), _) ->
                let l =
                  Option.value ~default:[] (Hashtbl.find_opt events bank)
                in
                Hashtbl.replace events bank
                  (List.concat_map
                     (fun (a, b) -> [ (a, 1); (b, -1) ])
                     i.Lv.Interval.ranges
                  @ l)
            | Some ((Mir_target.Bank.Control | Mir_target.Bank.Flags), _) | None
              ->
                ())
          (Lv.intervals f))
      (Lv.S.Verified.selected sel).Lv.S.program.Mir_program.funcs;
    List.filter_map
      (fun bank ->
        Option.map
          (fun l ->
            (* a range closing at a position frees it before one opens there *)
            let l = List.sort compare l in
            ( bank,
              snd
                (List.fold_left
                   (fun (live, top) (_, d) -> (live + d, max top (live + d)))
                   (0, 0) l) ))
          (Hashtbl.find_opt events bank))
      [ Mir_target.Bank.Fpr; Mir_target.Bank.Gpr ]

  let by_bank h =
    List.filter_map
      (fun b -> Option.map (fun n -> (b, n)) (Hashtbl.find_opt h b))
      [ Mir_target.Bank.Fpr; Mir_target.Bank.Gpr ]

  let report sel (p : (T.op, T.test) Mir_phys.Program.t) =
    let hot_stores = Hashtbl.create 2 and hot_loads = Hashtbl.create 2 in
    let by_depth = Hashtbl.create 4 and innermost = ref 0 in
    let helper_calls = ref 0 in
    let count h (v : Mir_value.t) =
      match V.shape v.Mir_value.ty with
      | Some (bank, _) ->
          Hashtbl.replace h bank
            (1 + Option.value ~default:0 (Hashtbl.find_opt h bank))
      | None -> ()
    in
    List.iter
      (fun f ->
        let nest = nest f in
        List.iter
          (fun (b : (_, _) Mir_phys.Block.t) ->
            let depth, inner = nest b.Mir_phys.Block.id in
            let spill () =
              Hashtbl.replace by_depth depth
                (1 + Option.value ~default:0 (Hashtbl.find_opt by_depth depth));
              if inner then incr innermost
            in
            List.iter
              (function
                | Mir_phys.Instr.Move { dst; src; value } -> (
                    match
                      ( Mir_alloc_stats.is_memory dst,
                        Mir_alloc_stats.is_memory src )
                    with
                    | true, false ->
                        spill ();
                        count hot_stores value
                    | false, true ->
                        spill ();
                        count hot_loads value
                    | false, false | true, true -> ())
                | Mir_phys.Instr.Exec _ | Mir_phys.Instr.Late _
                | Mir_phys.Instr.Remat _ | Mir_phys.Instr.Save _
                | Mir_phys.Instr.Sp _ ->
                    ())
              b.Mir_phys.Block.body)
          (on_cycle f);
        List.iter
          (fun (b : (_, _) Mir_phys.Block.t) ->
            List.iter
              (function
                | Mir_phys.Instr.Exec
                    { instr = { Mir_instr.op = Mir_sel.Op.Machine op; _ }; _ }
                  ->
                    List.iter
                      (function
                        | Mir_target.Reference.Call (Mir_op.Callee.Helper _) ->
                            incr helper_calls
                        | Mir_target.Reference.Call (Mir_op.Callee.Func _)
                        | Mir_target.Reference.View _ ->
                            ())
                      (T.references op)
                | _ -> ())
              b.Mir_phys.Block.body)
          f.Mir_phys.Func.blocks)
      p.Mir_phys.Program.funcs;
    {
      peak = peak sel;
      hot_stores = by_bank hot_stores;
      hot_loads = by_bank hot_loads;
      by_depth =
        List.sort compare
          (Hashtbl.fold (fun d n acc -> (d, n) :: acc) by_depth []);
      innermost = !innermost;
      stats = Mir_alloc_stats.of_program p;
      frame =
        List.fold_left
          (fun acc (f : (_, _) Mir_phys.Func.t) ->
            match (acc, f.Mir_phys.Func.frame) with
            | Some a, Some b -> Some (max a b)
            | None, x | x, None -> x)
          None p.Mir_phys.Program.funcs;
      helper_calls = !helper_calls;
    }
end
