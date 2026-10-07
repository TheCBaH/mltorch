(* The constrained reference allocator: every value lives in its own frame
   slot; each instruction reloads its operands into per-position scratch
   registers, runs with its result in a result scratch (or its fixed or tied
   register, or the condition register), and spills the result back. Block
   parameters live in their slots, so each edge is a simultaneous slot-to-slot
   transfer resolved by [Mir_parallel_copy], placed at the end of a block with
   one successor or in a split block of its own. It establishes correctness,
   not speed. *)

open Machine_ir
module Loc = Mir_phys.Loc

(* What the allocator needs of a target beyond its selected interface. *)
module type REGISTERS = sig
  val use_scratch : Mir_target.Bank.t -> bits:int -> int -> Mir_target.View.t
  (** the register for an instruction's [k]-th distinct operand *)

  val result_scratch : Mir_target.Bank.t -> bits:int -> Mir_target.View.t
  val copy_scratch : Mir_target.Bank.t -> bits:int -> Mir_target.View.t
  val cycle_scratch : Mir_target.Bank.t -> bits:int -> Mir_target.View.t

  val args : Mir_type.t list -> Mir_target.View.t list
  (** where a function's parameters arrive *)

  val results : Mir_type.t list -> Mir_target.View.t list
end

(* Fault injection for the evidence suite: one deliberate allocation defect
   each, which the checker or interpreter must catch. No consumer passes one. *)
module Mutation = struct
  type t =
    | Copy_placement  (** a branch's edge moves placed on the other edge *)
    | Cycle_scratch  (** a transfer cycle broken through the copy scratch *)
    | Reload_order
        (** an operand reloaded before the previous result's spill *)
    | Slot_reuse  (** a value spilled into the previous value's slot *)
    | Spill_width  (** a 64-bit value spilled at half width *)
    | Tie  (** a tied result written to another register *)
end

module Make (T : Mir_sel.TARGET) (R : REGISTERS) = struct
  module S = Mir_sel.Make (T)
  module V = Mir_phys_verify.Make (T)

  let shape (v : Mir_value.t) =
    match V.shape v.Mir_value.ty with
    | Some s -> s
    | None -> invalid_arg "Mir_ref_alloc: a value with no shape"

  type st = {
    mutation : Mutation.t option;
    mutable next_block : int;
    mutable slots : Mir_phys.Slot.t list;
    mutable previous : Mir_value.t option;  (** the last value spilled *)
  }

  let mutated st m = st.mutation = Some m
  let is_flags (v : Mir_value.t) = Mir_type.equal v.Mir_value.ty Mir_type.Flags

  let slot st (v : Mir_value.t) =
    let _, bits = shape v in
    let bytes = Int64.of_int (bits / 8) in
    let id = Mir_id.Slot.of_int (Mir_id.Value.to_int v.Mir_value.id) in
    if
      not
        (List.exists
           (fun (s : Mir_phys.Slot.t) ->
             Mir_id.Slot.equal s.Mir_phys.Slot.id id)
           st.slots)
    then st.slots <- { Mir_phys.Slot.id; bytes; align = bytes } :: st.slots;
    Loc.Slot { slot = id; bytes }

  let reg_of bank bits view_fn = Loc.Reg (view_fn bank ~bits)

  (* A simultaneous transfer of values between locations, as moves: slot to
     slot through the copy scratch. *)
  let transfer st (moves : Mir_parallel_copy.move list) =
    let scratch (v : Mir_value.t) =
      let bank, bits = shape v in
      if mutated st Mutation.Cycle_scratch then reg_of bank bits R.copy_scratch
      else reg_of bank bits R.cycle_scratch
    in
    List.concat_map
      (fun (m : Mir_parallel_copy.move) ->
        match (m.Mir_parallel_copy.dst, m.Mir_parallel_copy.src) with
        | Loc.Slot _, Loc.Slot _ ->
            let bank, bits = shape m.Mir_parallel_copy.value in
            let tmp = reg_of bank bits R.copy_scratch in
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
      (Mir_parallel_copy.resolve ~scratch moves)

  (* One selected instruction: reloads, the instruction, spills. *)
  let instr st (i : T.op Mir_sel.Op.t Mir_instr.t) =
    match i.Mir_instr.op with
    | Mir_sel.Op.Event _ ->
        [ Mir_phys.Instr.Exec { instr = i; uses = []; defs = [] } ]
    | Mir_sel.Op.Machine op ->
        let operands = T.uses op in
        let constraints = T.constraints op in
        (* the register each distinct operand is reloaded into *)
        let distinct =
          List.sort_uniq Mir_value.compare
            (List.filter (fun v -> not (is_flags v)) operands)
        in
        let fixed_use k =
          List.find_map
            (function
              | Mir_target.Constraint.Fixed_use { use; view } when use = k ->
                  Some view
              | _ -> None)
            constraints
        in
        let reg_for (v : Mir_value.t) =
          if is_flags v then Loc.Reg T.flags_view
          else
            let k =
              let rec index i = function
                | [] -> 0
                | w :: r -> if Mir_value.equal v w then i else index (i + 1) r
              in
              index 0 distinct
            in
            let pos =
              let rec index i = function
                | [] -> -1
                | w :: r -> if Mir_value.equal v w then i else index (i + 1) r
              in
              index 0 operands
            in
            match fixed_use pos with
            | Some view -> Loc.Reg view
            | None ->
                let bank, bits = shape v in
                Loc.Reg (R.use_scratch bank ~bits k)
        in
        let reloads =
          List.map
            (fun v ->
              Mir_phys.Instr.Move
                { dst = reg_for v; src = slot st v; value = v })
            distinct
        in
        let uses = List.map reg_for operands in
        let defs =
          List.mapi
            (fun k (r : Mir_value.t) ->
              if is_flags r then Loc.Reg T.flags_view
              else
                match
                  List.find_map
                    (function
                      | Mir_target.Constraint.Fixed_result { result; view }
                        when result = k ->
                          Some (Loc.Reg view)
                      | Mir_target.Constraint.Tied { result; use }
                        when result = k && not (mutated st Mutation.Tie) -> (
                          match List.nth uses use with
                          | Loc.Reg u ->
                              let _, bits = shape r in
                              Some
                                (Loc.Reg
                                   {
                                     u with
                                     Mir_target.View.bits;
                                     name = u.Mir_target.View.name;
                                   })
                          | Loc.Slot _ | Loc.Mem _ -> None)
                      | _ -> None)
                    constraints
                with
                | Some l -> l
                | None ->
                    let bank, bits = shape r in
                    Loc.Reg (R.result_scratch bank ~bits))
            i.Mir_instr.results
        in
        let spills =
          List.filter_map
            (fun ((r : Mir_value.t), l) ->
              if is_flags r then None
              else
                let dst =
                  match (st.mutation, st.previous) with
                  | Some Mutation.Slot_reuse, Some p
                    when Mir_type.equal p.Mir_value.ty r.Mir_value.ty ->
                      slot st p
                  | Some Mutation.Spill_width, _ when snd (shape r) = 64 -> (
                      match slot st r with
                      | Loc.Slot s -> Loc.Slot { s with bytes = 4L }
                      | l -> l)
                  | _ -> slot st r
                in
                st.previous <- Some r;
                Some (Mir_phys.Instr.Move { dst; src = l; value = r }))
            (List.combine i.Mir_instr.results defs)
        in
        reloads @ [ Mir_phys.Instr.Exec { instr = i; uses; defs } ] @ spills

  let fresh_block st =
    let id = Mir_id.Block.of_int st.next_block in
    st.next_block <- st.next_block + 1;
    id

  let func st (f : (S.Stage.op, S.Stage.term) Mir_func.t) =
    let block id = Option.get (Mir_func.find_block f id) in
    let preds id =
      List.length
        (List.filter
           (fun (b : (S.Stage.op, S.Stage.term) Mir_block.t) ->
             List.exists
               (fun (e : Mir_edge.t) -> Mir_id.Block.equal e.Mir_edge.target id)
               (Mir_sel.Terminator.edges b.Mir_block.terminator))
           f.Mir_func.blocks)
    in
    ignore preds;
    let splits = ref [] in
    (* the moves of a selected edge: its arguments' slots into the target's
       parameter slots *)
    let edge_moves (e : Mir_edge.t) =
      let t = block e.Mir_edge.target in
      transfer st
        (List.map2
           (fun (p : Mir_value.t) (a : Mir_value.t) ->
             { Mir_parallel_copy.dst = slot st p; src = slot st a; value = a })
           t.Mir_block.params e.Mir_edge.args)
    in
    let entry_params = (block f.Mir_func.entry).Mir_block.params in
    let arg_regs =
      R.args (List.map (fun (v : Mir_value.t) -> v.Mir_value.ty) entry_params)
    in
    let result_regs = R.results f.Mir_func.results in
    let blocks =
      List.map
        (fun (b : (S.Stage.op, S.Stage.term) Mir_block.t) ->
          let id = b.Mir_block.id in
          let prologue =
            if Mir_id.Block.equal id f.Mir_func.entry then
              List.map2
                (fun (p : Mir_value.t) r ->
                  Mir_phys.Instr.Move
                    { dst = slot st p; src = Loc.Reg r; value = p })
                entry_params arg_regs
            else []
          in
          let body = List.concat_map (instr st) b.Mir_block.body in
          let body =
            if mutated st Mutation.Reload_order then
              (* the first reload of a value spilled just before moves above
                 that spill *)
              let rec swap = function
                | (Mir_phys.Instr.Move { dst = Loc.Slot _; _ } as spill)
                  :: (Mir_phys.Instr.Move { src = Loc.Slot _; _ } as reload)
                  :: rest ->
                    reload :: spill :: rest
                | x :: rest -> x :: swap rest
                | [] -> []
              in
              swap body
            else body
          in
          let reload_test vs =
            List.mapi
              (fun k (v : Mir_value.t) ->
                if is_flags v then (Loc.Reg T.flags_view, [])
                else
                  let bank, bits = shape v in
                  let r = Loc.Reg (R.use_scratch bank ~bits k) in
                  ( r,
                    [
                      Mir_phys.Instr.Move
                        { dst = r; src = slot st v; value = v };
                    ] ))
              vs
          in
          let tail, terminator =
            match b.Mir_block.terminator with
            | Mir_sel.Terminator.Jump e ->
                (edge_moves e, Mir_phys.Term.Jump e.Mir_edge.target)
            | Mir_sel.Terminator.Branch { test; then_; else_ } ->
                let located = reload_test (T.test_uses test) in
                let via index (e : Mir_edge.t) other =
                  (* a nonempty transfer gets its own block *)
                  let moves =
                    edge_moves
                      (if mutated st Mutation.Copy_placement then other else e)
                  in
                  if moves = [] then e.Mir_edge.target
                  else
                    let sid = fresh_block st in
                    splits :=
                      {
                        Mir_phys.Block.id = sid;
                        origin =
                          Mir_phys.Origin.Edge
                            { Mir_phys.Edge_ref.from = id; index };
                        entry = [];
                        body = moves;
                        terminator = Mir_phys.Term.Jump e.Mir_edge.target;
                      }
                      :: !splits;
                    sid
                in
                let then_b = via 0 then_ else_ and else_b = via 1 else_ then_ in
                ( List.concat_map snd located,
                  Mir_phys.Term.Branch
                    {
                      test;
                      uses = List.map fst located;
                      then_ = then_b;
                      else_ = else_b;
                    } )
            | Mir_sel.Terminator.Return { Mir_return.values; _ } ->
                ( List.map2
                    (fun (v : Mir_value.t) r ->
                      Mir_phys.Instr.Move
                        { dst = Loc.Reg r; src = slot st v; value = v })
                    values result_regs,
                  Mir_phys.Term.Return
                    { values = List.map (fun r -> Loc.Reg r) result_regs } )
          in
          {
            Mir_phys.Block.id;
            origin = Mir_phys.Origin.Block id;
            entry = List.map (fun p -> (p, slot st p)) b.Mir_block.params;
            body = prologue @ body @ tail;
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
      slots = List.rev st.slots;
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
    let st =
      { mutation; next_block = max_block + 1; slots = []; previous = None }
    in
    {
      Mir_phys.Program.data_model = p.Mir_program.data_model;
      regions = p.Mir_program.regions;
      views = p.Mir_program.views;
      helpers = p.Mir_program.helpers;
      funcs =
        List.map
          (fun f ->
            st.slots <- [];
            func st f)
          p.Mir_program.funcs;
      main = p.Mir_program.main;
      features = s.S.features;
    }
end
