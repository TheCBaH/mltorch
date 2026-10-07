(* Production allocation: linear scan over SSA live intervals with lifetime
   holes, splitting and spilling (after Wimmer and Franz). Intervals come from
   [Mir_liveness]; every instruction occupies two positions, operands read at
   the even one and results written at the odd one.

   - Pieces. An interval is allocated as pieces, each wholly in one register
     or wholly spilled to the value's slot. A piece is split only at an even
     position: before an instruction, where a transition move runs, or at a
     block start, where edge resolution does.
   - Spilled pieces stay correct at every use and definition: an operand in a
     slot is reloaded into the reference allocator's scratch just before its
     instruction and a result in a slot is written through the result scratch
     and stored, so no split is ever forced for legality — splitting is how a
     spilled value gets a register back before its next use.
   - Constraints. A fixed use or result blocks its register for the
     instruction (a fixed interval) and is reached by moves before or after
     it; a call's clobbered views are blocked as it writes, so a value live
     across the call avoids them bit for bit (AArch64's v8-v15 keep their low
     half); a tied or early-clobber result's interval opens at the
     instruction's read, so it shares no register with an operand, and a tied
     operand is copied into the result's register first. Condition values
     live only in the condition register and are never allocated.
   - Decisions. The register free longest wins, preferring the value's
     previous register; with none free, the occupant whose next use is
     farthest, weighted by loop depth (a use ten times deeper in loops counts
     as ten times nearer), is evicted to its slot — or the current piece is
     spilled when its own next use is farther still.
   - Resolution. Pieces that change location inside a block do so through one
     parallel copy at the split; across an edge, every value live into the
     target (block parameters from the edge's arguments) moves from where the
     source leaves it to where the target expects it, resolved by
     [Mir_parallel_copy], at the end of a block with one successor or in a
     split block of its own.
   - Slots are shared by values whose spilled pieces never overlap, size by
     size, so the frame grows with simultaneous spills, not with values.
   - A value whose instruction reads nothing and is pure, total and
     unconstrained (a constant, an address) is rematerialized: its spilled
     pieces take no slot, a reload runs the instruction again, and a store is
     dropped. *)

open Machine_ir
module Loc = Mir_phys.Loc

module type POOL = Mir_scan.POOL

module Mutation = Mir_scan.Mutation

module Make (T : Mir_sel.TARGET) (R : POOL) = struct
  include Mir_scan.Make (T) (R)

  (* --- emission ----------------------------------------------------------- *)

  let transfer (moves : Mir_parallel_copy.move list) =
    let scratch (v : Mir_value.t) =
      let bank, bits = shape v in
      Loc.Reg (R.cycle_scratch bank ~bits)
    in
    List.concat_map
      (fun (m : Mir_parallel_copy.move) ->
        match (m.Mir_parallel_copy.dst, m.Mir_parallel_copy.src) with
        | Loc.Slot _, Loc.Slot _ ->
            let bank, bits = shape m.Mir_parallel_copy.value in
            let tmp = Loc.Reg (R.copy_scratch bank ~bits) in
            [
              Mir_phys.Instr.Move
                {
                  dst = tmp;
                  src = m.Mir_parallel_copy.src;
                  value = m.Mir_parallel_copy.value;
                };
              Mir_phys.Instr.Move
                {
                  dst = m.Mir_parallel_copy.dst;
                  src = tmp;
                  value = m.Mir_parallel_copy.value;
                };
            ]
        | dst, src ->
            [
              Mir_phys.Instr.Move
                { dst; src; value = m.Mir_parallel_copy.value };
            ])
      (Mir_parallel_copy.resolve ~scratch
         (List.filter
            (fun (m : Mir_parallel_copy.move) ->
              not (Loc.equal m.Mir_parallel_copy.dst m.Mir_parallel_copy.src))
            moves))

  let func st (f : func) =
    let block id = Option.get (Mir_func.find_block f id) in
    let spans = Lv.spans f in
    let span_of id =
      snd (List.find (fun (b, _) -> Mir_id.Block.equal b id) spans)
    in
    let order = List.map fst spans in
    (* loops: a header and the end of its last back edge's source *)
    let loops =
      List.concat_map
        (fun hid ->
          let hf, ht = span_of hid in
          List.filter_map
            (fun pid ->
              let _, pt = span_of pid in
              if
                pt > ht
                && List.exists
                     (fun (e : Mir_edge.t) ->
                       Mir_id.Block.equal e.Mir_edge.target hid)
                     (Mir_sel.Terminator.edges (block pid).Mir_block.terminator)
              then Some (hf, pt)
              else None)
            order)
        order
    in
    let depth x =
      List.length (List.filter (fun (a, b) -> a <= x && x < b) loops)
    in
    (* instruction positions and their constraints *)
    let at = Hashtbl.create 64 in
    List.iter
      (fun bid ->
        let from, _ = span_of bid in
        List.iteri
          (fun j (i : S.Stage.op Mir_instr.t) ->
            Hashtbl.replace at (from + 2 + (2 * j)) i)
          (block bid).Mir_block.body)
      order;
    let fixed = ref [] in
    let opens = Hashtbl.create 16 in
    (* where a result's interval opens instead of its definition *)
    Hashtbl.iter
      (fun p (i : S.Stage.op Mir_instr.t) ->
        match i.Mir_instr.op with
        | Mir_sel.Op.Event _ | Mir_sel.Op.Undef _ -> ()
        | Mir_sel.Op.Machine op ->
            let cs = T.constraints op in
            let own = i.Mir_instr.results in
            List.iter
              (function
                | Mir_target.Constraint.Fixed_use { view; _ } ->
                    fixed :=
                      { Fixed.view; lo = p; hi = p + 1; own = [] } :: !fixed
                | Mir_target.Constraint.Fixed_result { view; _ } ->
                    fixed :=
                      { Fixed.view; lo = p + 1; hi = p + 2; own } :: !fixed
                | Mir_target.Constraint.Early_clobber k
                | Mir_target.Constraint.Tied { result = k; _ } ->
                    Hashtbl.replace opens
                      (Mir_id.Value.to_int (List.nth own k).Mir_value.id)
                      p)
              cs;
            List.iter
              (fun view ->
                fixed := { Fixed.view; lo = p + 1; hi = p + 2; own } :: !fixed)
              (T.clobbers op))
      at;
    (* the instructions a second run recreates *)
    let remat = Hashtbl.create 16 in
    List.iter
      (fun (b : block) ->
        List.iter
          (fun (i : S.Stage.op Mir_instr.t) ->
            if V.rematerializable i then
              Hashtbl.replace remat
                (Mir_id.Value.to_int (List.hd i.Mir_instr.results).Mir_value.id)
                i)
          b.Mir_block.body)
      f.Mir_func.blocks;
    let intervals = Lv.intervals f in
    let pieces =
      List.filter_map
        (fun (iv : Lv.Interval.t) ->
          let v = iv.Lv.Interval.value in
          if is_flags v then None
          else
            let ranges =
              match
                ( iv.Lv.Interval.ranges,
                  Hashtbl.find_opt opens (Mir_id.Value.to_int v.Mir_value.id) )
              with
              | (_, b) :: rest, Some o -> (o, b) :: rest
              | rs, _ -> rs
            in
            if ranges = [] then None
            else
              Some
                {
                  Piece.value = v;
                  ranges;
                  uses = iv.Lv.Interval.uses;
                  where = Where.Unassigned;
                })
        intervals
    in
    let bank_of (p : Piece.t) = fst (shape p.Piece.value) in
    let allocated =
      List.concat_map
        (fun bank ->
          scan st ~bank ~fixed:!fixed ~depth
            (List.filter (fun p -> bank_of p = bank) pieces))
        [ Mir_target.Bank.Fpr; Mir_target.Bank.Gpr ]
    in
    (* each value's pieces, by start *)
    let by_value = Hashtbl.create 64 in
    List.iter
      (fun (p : Piece.t) ->
        let k = Mir_id.Value.to_int p.Piece.value.Mir_value.id in
        Hashtbl.replace by_value k
          (p :: Option.value ~default:[] (Hashtbl.find_opt by_value k)))
      allocated;
    Hashtbl.filter_map_inplace
      (fun _ ps ->
        Some (List.sort (fun a b -> compare (Piece.start a) (Piece.start b)) ps))
      by_value;
    (* slots: shared by values whose spilled pieces never overlap *)
    let slots = ref [] and slot_of = Hashtbl.create 16 in
    let spilled_ranges ps =
      List.concat_map
        (fun (p : Piece.t) ->
          if p.Piece.where = Where.Spilled then p.Piece.ranges else [])
        ps
    in
    let candidates =
      Hashtbl.fold
        (fun k ps acc ->
          match spilled_ranges ps with
          | [] -> acc
          | _ when Hashtbl.mem remat k -> acc
          | rs -> (k, List.hd ps, rs) :: acc)
        by_value []
      |> List.sort (fun (_, _, a) (_, _, b) -> compare (List.hd a) (List.hd b))
    in
    List.iter
      (fun (k, (p : Piece.t), rs) ->
        let _, bits = shape p.Piece.value in
        let bytes = Int64.of_int (bits / 8) in
        let disjoint xs ys =
          List.for_all
            (fun (a, b) -> List.for_all (fun (c, d) -> b <= c || d <= a) ys)
            xs
        in
        match
          List.find_opt
            (fun (_, b, used) -> Int64.equal b bytes && disjoint !used rs)
            !slots
        with
        | Some (id, _, used) ->
            used := rs @ !used;
            Hashtbl.replace slot_of k id
        | None ->
            let id = Mir_id.Slot.of_int (List.length !slots) in
            slots := (id, bytes, ref rs) :: !slots;
            Hashtbl.replace slot_of k id)
      candidates;
    (* a rematerialized value's spilled location: a marker, never a frame
       slot, rewritten below into re-runs and dropped stores *)
    let marker_base = 1_000_000 in
    let slot_loc (v : Mir_value.t) =
      let _, bits = shape v in
      let k = Mir_id.Value.to_int v.Mir_value.id in
      Loc.Slot
        {
          slot =
            (if Hashtbl.mem remat k then Mir_id.Slot.of_int (marker_base + k)
             else Hashtbl.find slot_of k);
          bytes = Int64.of_int (bits / 8);
        }
    in
    let rematerialize body =
      List.filter_map
        (function
          | Mir_phys.Instr.Move { dst = Loc.Slot { slot; _ }; _ }
            when Mir_id.Slot.to_int slot >= marker_base ->
              None
          | Mir_phys.Instr.Move { dst; src = Loc.Slot { slot; _ }; _ }
            when Mir_id.Slot.to_int slot >= marker_base ->
              Some
                (Mir_phys.Instr.Remat
                   {
                     instr =
                       Hashtbl.find remat (Mir_id.Slot.to_int slot - marker_base);
                     defs = [ dst ];
                   })
          | i -> Some i)
        body
    in
    let loc_of_piece (p : Piece.t) =
      match p.Piece.where with
      | Where.In k ->
          let bank, bits = shape p.Piece.value in
          Loc.Reg (R.view bank ~bits k)
      | Where.Spilled -> slot_loc p.Piece.value
      | Where.Unassigned -> invalid_arg "Mir_linear_scan: an unassigned piece"
    in
    (* where a value is at a position *)
    let loc (v : Mir_value.t) x =
      if is_flags v then Loc.Reg T.flags_view
      else
        match
          Hashtbl.find_opt by_value (Mir_id.Value.to_int v.Mir_value.id)
        with
        | None -> invalid_arg "Mir_linear_scan: a value with no interval"
        | Some ps -> (
            match List.find_opt (fun p -> Piece.covers p x) ps with
            | Some p -> loc_of_piece p
            | None -> (
                (* a definition the interval never reaches: its first piece *)
                match ps with
                | p :: _ -> loc_of_piece p
                | [] -> invalid_arg "Mir_linear_scan: no piece"))
    in
    (* Whether a value is live at an instruction position, by dataflow: an
       interval may also cover positions where it is dead (a loop header's
       live-ins are kept live over the header's whole linear span, which need
       not hold only its loop), and a move must never read such a value. *)
    let live_in, live_out = Lv.sets f in
    let last_use = Hashtbl.create 64 in
    List.iter
      (fun bid ->
        let from, to_ = span_of bid in
        let b = block bid in
        let note x (v : Mir_value.t) =
          let k =
            (Mir_id.Block.to_int bid, Mir_id.Value.to_int v.Mir_value.id)
          in
          match Hashtbl.find_opt last_use k with
          | Some y when y >= x -> ()
          | _ -> Hashtbl.replace last_use k x
        in
        List.iteri
          (fun j (i : S.Stage.op Mir_instr.t) ->
            List.iter
              (note (from + 2 + (2 * j)))
              (S.Stage.operands i.Mir_instr.op))
          b.Mir_block.body;
        List.iter (note (to_ - 1)) (S.Stage.term_values b.Mir_block.terminator))
      order;
    let block_at x =
      List.find (fun (_, (a, b)) -> a <= x && x < b) spans |> fst
    in
    let live_at (v : Mir_value.t) x =
      let bid = block_at x in
      let defined_later =
        match
          Hashtbl.find_opt by_value (Mir_id.Value.to_int v.Mir_value.id)
        with
        | Some (p :: _) ->
            let d = Piece.start p in
            d >= x
            &&
            let a, b = span_of bid in
            a <= d && d < b
        | _ -> false
      in
      (not defined_later)
      && (Mir_id.Value.Set.mem v.Mir_value.id (live_out bid)
         ||
         match
           Hashtbl.find_opt last_use
             (Mir_id.Block.to_int bid, Mir_id.Value.to_int v.Mir_value.id)
         with
         | Some y -> y >= x
         | None -> false)
    in
    (* transitions at an even position inside a block *)
    let transitions x =
      if mutated st Mutation.Split_move then []
      else
        transfer
          (Hashtbl.fold
             (fun _ ps acc ->
               match
                 List.find_opt
                   (fun p ->
                     Piece.start p = x
                     && Piece.covers p x && live_at p.Piece.value x)
                   ps
               with
               | Some q -> (
                   match List.find_opt (fun p -> Piece.covers p (x - 1)) ps with
                   | Some p when p != q ->
                       {
                         Mir_parallel_copy.dst = loc_of_piece q;
                         src = loc_of_piece p;
                         value = q.Piece.value;
                       }
                       :: acc
                   | _ -> acc)
               | None -> acc)
             by_value [])
    in
    let instr p (i : S.Stage.op Mir_instr.t) =
      match i.Mir_instr.op with
      | Mir_sel.Op.Event _ | Mir_sel.Op.Undef _ ->
          [ Mir_phys.Instr.Exec { instr = i; uses = []; defs = [] } ]
      | Mir_sel.Op.Machine op ->
          let operands = T.uses op and cs = T.constraints op in
          let late =
            T.clobbers op <> []
            || List.exists
                 (function
                   | Mir_target.Constraint.Fixed_result _ -> true | _ -> false)
                 cs
          in
          let fixed_use k =
            List.find_map
              (function
                | Mir_target.Constraint.Fixed_use { use; view } when use = k ->
                    Some view
                | _ -> None)
              cs
          in
          let tied_result k =
            List.find_map
              (function
                | Mir_target.Constraint.Tied { result; use } when use = k ->
                    Some result
                | _ -> None)
              cs
          in
          let fixed_result k =
            List.find_map
              (function
                | Mir_target.Constraint.Fixed_result { result; view }
                  when result = k ->
                    Some view
                | _ -> None)
              cs
          in
          (* where each result is written *)
          let defs =
            List.mapi
              (fun k (r : Mir_value.t) ->
                if is_flags r then Loc.Reg T.flags_view
                else
                  match fixed_result k with
                  | Some view -> Loc.Reg view
                  | None -> (
                      let bank, bits = shape r in
                      if late then Loc.Reg (R.result_scratch bank ~bits)
                      else
                        match loc r (p + 1) with
                        | Loc.Reg _ as l -> l
                        | Loc.Slot _ | Loc.Mem _ ->
                            Loc.Reg (R.result_scratch bank ~bits)))
              i.Mir_instr.results
          in
          (* stack operands reloaded into scratch, the k-th distinct one into
             the k-th scratch *)
          let stacked =
            List.sort_uniq Mir_value.compare
              (List.filteri
                 (fun k (v : Mir_value.t) ->
                   (not (is_flags v))
                   && fixed_use k = None
                   && tied_result k = None
                   && match loc v p with Loc.Slot _ -> true | _ -> false)
                 operands)
          in
          let scratch_of (v : Mir_value.t) =
            let rec index n = function
              | [] -> None
              | w :: rest ->
                  if Mir_value.equal v w then Some n else index (n + 1) rest
            in
            Option.map
              (fun n ->
                let bank, bits = shape v in
                Loc.Reg (R.use_scratch bank ~bits n))
              (index 0 stacked)
          in
          let reloads =
            List.map
              (fun v ->
                Mir_phys.Instr.Move
                  { dst = Option.get (scratch_of v); src = loc v p; value = v })
              stacked
          in
          let uses =
            List.mapi
              (fun k (v : Mir_value.t) ->
                match (fixed_use k, tied_result k) with
                | Some view, _ -> Loc.Reg view
                | None, Some r -> List.nth defs r
                | None, None -> (
                    match scratch_of v with Some l -> l | None -> loc v p))
              operands
          in
          (* fixed and tied operands, all at once *)
          let pre =
            transfer
              (List.concat
                 (List.mapi
                    (fun k (v : Mir_value.t) ->
                      match (fixed_use k, tied_result k) with
                      | Some view, _ ->
                          [
                            {
                              Mir_parallel_copy.dst = Loc.Reg view;
                              src = loc v p;
                              value = v;
                            };
                          ]
                      | None, Some r ->
                          [
                            {
                              Mir_parallel_copy.dst = List.nth defs r;
                              src = loc v p;
                              value = v;
                            };
                          ]
                      | None, None -> [])
                    operands))
          in
          (* results that are not where their interval holds them *)
          let post =
            transfer
              (List.filter_map
                 (fun ((r : Mir_value.t), d) ->
                   (* a result nothing reads after the instruction stays where
                      it was written *)
                   if
                     is_flags r
                     || not
                          (Hashtbl.mem by_value
                             (Mir_id.Value.to_int r.Mir_value.id))
                   then None
                   else
                     let want = loc r (p + 1) in
                     if Loc.equal want d then None
                     else
                       Some { Mir_parallel_copy.dst = want; src = d; value = r })
                 (List.combine i.Mir_instr.results defs))
          in
          reloads @ pre
          @ [ Mir_phys.Instr.Exec { instr = i; uses; defs } ]
          @ post
    in
    let splits = ref [] in
    let fresh_block () =
      let id = Mir_id.Block.of_int st.next_block in
      st.next_block <- st.next_block + 1;
      id
    in
    (* an edge's moves: every value live into the target, from where the
       source leaves it to where the target expects it *)
    let edge_moves bid (e : Mir_edge.t) =
      let _, bt = span_of bid in
      let tf, _ = span_of e.Mir_edge.target in
      let t = block e.Mir_edge.target in
      let params =
        List.filter_map
          (fun ((pv : Mir_value.t), (a : Mir_value.t)) ->
            if is_flags pv then None
            else
              Some
                {
                  Mir_parallel_copy.dst = loc pv tf;
                  src = loc a (bt - 1);
                  value = a;
                })
          (List.combine t.Mir_block.params e.Mir_edge.args)
      in
      let through =
        List.filter_map
          (fun id ->
            match Hashtbl.find_opt by_value (Mir_id.Value.to_int id) with
            | Some (p :: _) when not (is_flags p.Piece.value) ->
                let v = p.Piece.value in
                Some
                  {
                    Mir_parallel_copy.dst = loc v tf;
                    src = loc v (bt - 1);
                    value = v;
                  }
            | _ -> None)
          (Mir_id.Value.Set.elements
             (Mir_id.Value.Set.diff
                (live_in e.Mir_edge.target)
                (Mir_id.Value.Set.of_list
                   (List.map
                      (fun (v : Mir_value.t) -> v.Mir_value.id)
                      t.Mir_block.params))))
      in
      transfer (params @ through)
    in
    let entry_params = (block f.Mir_func.entry).Mir_block.params in
    let arg_regs =
      R.args (List.map (fun (v : Mir_value.t) -> v.Mir_value.ty) entry_params)
    in
    let result_regs = R.results f.Mir_func.results in
    let blocks =
      List.map
        (fun (b : block) ->
          let id = b.Mir_block.id in
          let from, to_ = span_of id in
          let prologue =
            if Mir_id.Block.equal id f.Mir_func.entry then
              transfer
                (List.map2
                   (fun (p : Mir_value.t) r ->
                     {
                       Mir_parallel_copy.dst = loc p from;
                       src = Loc.Reg r;
                       value = p;
                     })
                   entry_params arg_regs)
            else []
          in
          let body =
            List.concat
              (List.mapi
                 (fun j i ->
                   let p = from + 2 + (2 * j) in
                   transitions p @ instr p i)
                 b.Mir_block.body)
          in
          let reload_test vs =
            List.mapi
              (fun k (v : Mir_value.t) ->
                if is_flags v then (Loc.Reg T.flags_view, [])
                else
                  match loc v (to_ - 1) with
                  | Loc.Reg _ as l -> (l, [])
                  | src ->
                      let bank, bits = shape v in
                      let r = Loc.Reg (R.use_scratch bank ~bits k) in
                      (r, [ Mir_phys.Instr.Move { dst = r; src; value = v } ]))
              vs
          in
          let tail, terminator =
            match b.Mir_block.terminator with
            | Mir_sel.Terminator.Jump e ->
                (edge_moves id e, Mir_phys.Term.Jump e.Mir_edge.target)
            | Mir_sel.Terminator.Branch { test; then_; else_ } ->
                let located = reload_test (T.test_uses test) in
                let via index (e : Mir_edge.t) =
                  let moves = edge_moves id e in
                  if moves = [] then e.Mir_edge.target
                  else
                    let sid = fresh_block () in
                    splits :=
                      {
                        Mir_phys.Block.id = sid;
                        origin =
                          Mir_phys.Origin.Edge
                            { Mir_phys.Edge_ref.from = id; index };
                        entry = [];
                        body = rematerialize moves;
                        terminator = Mir_phys.Term.Jump e.Mir_edge.target;
                      }
                      :: !splits;
                    sid
                in
                let then_b = via 0 then_ and else_b = via 1 else_ in
                ( List.concat_map snd located,
                  Mir_phys.Term.Branch
                    {
                      test;
                      uses = List.map fst located;
                      then_ = then_b;
                      else_ = else_b;
                    } )
            | Mir_sel.Terminator.Return { Mir_return.values; _ } ->
                ( transfer
                    (List.map2
                       (fun (v : Mir_value.t) r ->
                         {
                           Mir_parallel_copy.dst = Loc.Reg r;
                           src = loc v (to_ - 1);
                           value = v;
                         })
                       values result_regs),
                  Mir_phys.Term.Return
                    { values = List.map (fun r -> Loc.Reg r) result_regs } )
          in
          {
            Mir_phys.Block.id;
            origin = Mir_phys.Origin.Block id;
            entry =
              List.filter_map
                (fun (p : Mir_value.t) ->
                  if is_flags p then None else Some (p, loc p from))
                b.Mir_block.params;
            body = rematerialize (prologue @ body @ tail);
            terminator;
          })
        f.Mir_func.blocks
    in
    {
      Mir_phys.Func.id = f.Mir_func.id;
      name = f.Mir_func.name;
      entry = f.Mir_func.entry;
      params =
        List.combine entry_params (List.map (fun r -> Loc.Reg r) arg_regs);
      results = List.map (fun r -> Loc.Reg r) result_regs;
      slots =
        List.rev_map
          (fun (id, bytes, _) -> { Mir_phys.Slot.id; bytes; align = bytes })
          !slots;
      frame = None;
      blocks = blocks @ List.rev !splits;
    }

  let allocate ?mutation (sel : S.Verified.t) =
    let s = S.Verified.selected sel in
    let p = s.S.program in
    let max_block =
      List.fold_left
        (fun m (f : (_, _) Mir_func.t) ->
          List.fold_left
            (fun m (b : (_, _) Mir_block.t) ->
              max m (Mir_id.Block.to_int b.Mir_block.id))
            m f.Mir_func.blocks)
        0 p.Mir_program.funcs
    in
    let st = { mutation; next_block = max_block + 1 } in
    {
      Mir_phys.Program.data_model = p.Mir_program.data_model;
      regions = p.Mir_program.regions;
      views = p.Mir_program.views;
      helpers = p.Mir_program.helpers;
      funcs = List.map (func st) p.Mir_program.funcs;
      main = p.Mir_program.main;
      features = s.S.features;
    }
end
