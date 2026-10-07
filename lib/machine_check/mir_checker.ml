(* Symbolic allocation checking. A state maps each register unit and frame
   slot to the virtual value it holds and the view or size it was written
   with; anything absent is unknown. Moves copy a holder; an instruction checks
   that each use's location holds that operand at that width, then forgets the
   units it clobbers and every older holder of the values it defines (an older
   loop iteration's copy is stale), then records its results. Entering a
   block's realization renames the selected edge's arguments to its
   parameters at the locations the block's entry contract claims. States meet
   by intersection to a fixpoint, so a value unknown on any incoming path is
   unknown at the join. Every selected instruction must appear, once and in
   order, in the block that realizes its own. *)

open Machine_ir
module Loc = Mir_phys.Loc

module Problem = struct
  type t =
    | Instructions
        (** a block's instructions differ from its selected block's *)
    | Missing_claim of Mir_value.t
        (** a block parameter with no entry location *)
    | Missing_value of { value : Mir_value.t; at : Loc.t }
        (** a use, move source, parameter or result whose location does not hold
            the expected value at the expected width *)
    | Realization  (** a selected block realized never or twice *)
    | Successor  (** a terminator that does not reach its selected targets *)

  let pp fmt = function
    | Instructions ->
        Fmt.string fmt "instructions differ from the selected block"
    | Missing_claim v ->
        Fmt.pf fmt "parameter %a has no entry location" Mir_value.pp v
    | Missing_value { value; at } ->
        Fmt.pf fmt "%a does not hold %a" Loc.pp at Mir_value.pp value
    | Realization ->
        Fmt.string fmt "a selected block is realized never or twice"
    | Successor -> Fmt.string fmt "a successor that is not the selected edge's"
end

type error = {
  func : Mir_id.Func.t;
  block : Mir_id.Block.t option;
  problem : Problem.t;
}

let pp_error fmt e =
  Fmt.pf fmt "%a%a: %a" Mir_id.Func.pp e.func
    Fmt.(option (any " " ++ Mir_id.Block.pp))
    e.block Problem.pp e.problem

module Make (T : Mir_sel.TARGET) = struct
  module S = Mir_sel.Make (T)

  (* Where a holder lives: a register unit or a slot. *)
  (* Where a holder lives: a register unit, a frame slot before layout, or
     realized frame bytes by their offset from the entry stack pointer. *)
  type key = Frame of int64 | Slot of int | Unit of int

  (* What a location holds: a virtual value written at a width (bits), or,
     for a register, an address — the entry stack pointer plus [k]. *)
  type holder =
    | Address of int64
    | Value of { value : Mir_value.t; width : int64 }

  module K = Map.Make (struct
    type t = key

    let compare = compare
  end)

  let unit_of (v : Mir_target.View.t) =
    Unit (Mir_id.Unit.to_int v.Mir_target.View.unit)

  (* A location's key; frame memory resolves through its base register's
     address holder, and is unknown when the base holds no address. *)
  let key_of st (l : Loc.t) =
    match l with
    | Loc.Reg v -> Some (unit_of v)
    | Loc.Slot { slot; _ } -> Some (Slot (Mir_id.Slot.to_int slot))
    | Loc.Mem { base; offset; _ } -> (
        match K.find_opt (unit_of base) st with
        | Some (Address k) -> Some (Frame (Int64.add k offset))
        | Some (Value _) | None -> None)

  let width_of = function
    | Loc.Reg v -> Int64.of_int v.Mir_target.View.bits
    | Loc.Slot { bytes; _ } | Loc.Mem { bytes; _ } -> Int64.mul 8L bytes

  let holds st l (v : Mir_value.t) =
    match Option.bind (key_of st l) (fun k -> K.find_opt k st) with
    | Some (Value h) ->
        Mir_value.equal h.value v && Int64.equal h.width (width_of l)
    | Some (Address _) | None -> false

  (* Forgets whatever a write to [l] overwrites: a register unit, a slot, or
     every frame holder whose bytes meet the written ones (all frame holders
     when the address is unknown). *)
  let kill st (l : Loc.t) =
    match (l, key_of st l) with
    | Loc.Mem { bytes; _ }, Some (Frame k) ->
        K.filter
          (fun key h ->
            match (key, h) with
            | Frame j, Value { width; _ } ->
                Int64.compare j (Int64.add k bytes) >= 0
                || Int64.compare k (Int64.add j (Int64.div width 8L)) >= 0
            | _ -> true)
          st
    | Loc.Mem _, _ ->
        K.filter (fun key _ -> match key with Frame _ -> false | _ -> true) st
    | _, Some k -> K.remove k st
    | _, None -> st

  (* A clobbered view: a register holder survives only when the clobbered
     bits lie wholly above the bits it was written with. *)
  let kill_view st (v : Mir_target.View.t) =
    let k = unit_of v in
    match K.find_opt k st with
    | Some (Value h)
      when Int64.compare (Int64.of_int v.Mir_target.View.lo) h.width >= 0 ->
        st
    | Some _ -> K.remove k st
    | None -> st

  let forget_value st (v : Mir_value.t) =
    K.filter
      (fun _ h ->
        match h with
        | Value h -> not (Mir_value.equal h.value v)
        | Address _ -> true)
      st

  let set st l v =
    match key_of st l with
    | Some k -> K.add k (Value { value = v; width = width_of l }) (kill st l)
    | None -> kill st l

  let meet a b =
    K.merge
      (fun _ x y ->
        match (x, y) with Some x, Some y when x = y -> Some x | _ -> None)
      a b

  let check (sel : S.Verified.t) (phys : (T.op, T.test) Mir_phys.Program.t) =
    let sp = (S.Verified.selected sel).S.program in
    Err.Escape.with_escape @@ fun esc ->
    List.iter
      (fun (sf : (S.Stage.op, S.Stage.term) Mir_func.t) ->
        let func = sf.Mir_func.id in
        let fail ?block problem =
          Err.Escape.throw esc { func; block; problem }
        in
        let pf =
          match
            List.find_opt
              (fun (f : (_, _) Mir_phys.Func.t) ->
                Mir_id.Func.equal f.Mir_phys.Func.id func)
              phys.Mir_phys.Program.funcs
          with
          | Some f -> f
          | None -> fail Problem.Realization
        in
        (* each selected block realized exactly once *)
        let realizer = Hashtbl.create 16 in
        List.iter
          (fun (b : (T.op, T.test) Mir_phys.Block.t) ->
            match b.Mir_phys.Block.origin with
            | Mir_phys.Origin.Block sb ->
                if Hashtbl.mem realizer (Mir_id.Block.to_int sb) then
                  fail ~block:b.Mir_phys.Block.id Problem.Realization;
                Hashtbl.replace realizer (Mir_id.Block.to_int sb) b
            | Mir_phys.Origin.Edge _ -> ())
          pf.Mir_phys.Func.blocks;
        List.iter
          (fun (b : (S.Stage.op, S.Stage.term) Mir_block.t) ->
            if not (Hashtbl.mem realizer (Mir_id.Block.to_int b.Mir_block.id))
            then fail ~block:b.Mir_block.id Problem.Realization)
          sf.Mir_func.blocks;
        let selected id = Option.get (Mir_func.find_block sf id) in
        let physical id = Option.get (Mir_phys.Func.find_block pf id) in
        (* the selected edge a physical edge carries, and its target *)
        let selected_edge (from : Mir_id.Block.t) index =
          let edges =
            Mir_sel.Terminator.edges (selected from).Mir_block.terminator
          in
          match List.nth_opt edges index with
          | Some e -> e
          | None -> fail ~block:from Problem.Successor
        in
        let realization_of (b : (T.op, T.test) Mir_phys.Block.t) =
          match b.Mir_phys.Block.origin with
          | Mir_phys.Origin.Block sb -> Some sb
          | Mir_phys.Origin.Edge _ -> None
        in
        (* structure: instructions in order, terminators reaching the selected
           targets (directly, or through one split block for that edge) *)
        List.iter
          (fun (b : (T.op, T.test) Mir_phys.Block.t) ->
            let block = b.Mir_phys.Block.id in
            let reach index target =
              match b.Mir_phys.Block.origin with
              | Mir_phys.Origin.Edge _ -> ()
              | Mir_phys.Origin.Block sb -> (
                  let e = selected_edge sb index in
                  let t = physical target in
                  match t.Mir_phys.Block.origin with
                  | Mir_phys.Origin.Block tb
                    when Mir_id.Block.equal tb e.Mir_edge.target ->
                      ()
                  | Mir_phys.Origin.Edge { Mir_phys.Edge_ref.from; index = i }
                    when Mir_id.Block.equal from sb && i = index -> (
                      match t.Mir_phys.Block.terminator with
                      | Mir_phys.Term.Jump t2
                        when realization_of (physical t2)
                             = Some e.Mir_edge.target ->
                          ()
                      | _ -> fail ~block Problem.Successor)
                  | _ -> fail ~block Problem.Successor)
            in
            match b.Mir_phys.Block.origin with
            | Mir_phys.Origin.Edge _ -> (
                List.iter
                  (function
                    | Mir_phys.Instr.Exec _ -> fail ~block Problem.Instructions
                    | Mir_phys.Instr.Late _ | Mir_phys.Instr.Move _
                    | Mir_phys.Instr.Save _ | Mir_phys.Instr.Sp _ ->
                        ())
                  b.Mir_phys.Block.body;
                match b.Mir_phys.Block.terminator with
                | Mir_phys.Term.Jump _ -> ()
                | _ -> fail ~block Problem.Successor)
            | Mir_phys.Origin.Block sb -> (
                let s = selected sb in
                let execs =
                  List.filter_map
                    (function
                      | Mir_phys.Instr.Exec { instr; _ } -> Some instr
                      | Mir_phys.Instr.Late _ | Mir_phys.Instr.Move _
                      | Mir_phys.Instr.Save _ | Mir_phys.Instr.Sp _ ->
                          None)
                    b.Mir_phys.Block.body
                in
                if
                  List.length execs <> List.length s.Mir_block.body
                  || not
                       (List.for_all2
                          (fun (a : _ Mir_instr.t) (c : _ Mir_instr.t) ->
                            Mir_id.Instr.equal a.Mir_instr.id c.Mir_instr.id
                            && a.Mir_instr.op = c.Mir_instr.op
                            && List.equal Mir_value.equal a.Mir_instr.results
                                 c.Mir_instr.results)
                          execs s.Mir_block.body)
                then fail ~block Problem.Instructions;
                match (s.Mir_block.terminator, b.Mir_phys.Block.terminator) with
                | ( Mir_sel.Terminator.Branch { test; _ },
                    Mir_phys.Term.Branch { test = t; then_; else_; _ } ) ->
                    if test <> t then fail ~block Problem.Successor;
                    reach 0 then_;
                    reach 1 else_
                | Mir_sel.Terminator.Jump _, Mir_phys.Term.Jump t -> reach 0 t
                | Mir_sel.Terminator.Return _, Mir_phys.Term.Return _ -> ()
                | _ -> fail ~block Problem.Successor))
          pf.Mir_phys.Func.blocks;
        (* dataflow *)
        let need ?block st l v =
          if not (holds st l v) then
            fail ?block (Problem.Missing_value { value = v; at = l })
        in
        let transfer ~checking (b : (T.op, T.test) Mir_phys.Block.t) st =
          let block = b.Mir_phys.Block.id in
          let need st l v = if checking then need ~block st l v in
          List.fold_left
            (fun st i ->
              match i with
              | Mir_phys.Instr.Move { dst; src; value } ->
                  need st src value;
                  if holds st src value then set st dst value else kill st dst
              | Mir_phys.Instr.Save { dst; _ } -> kill st dst
              | Mir_phys.Instr.Sp delta -> (
                  let k = unit_of T.stack_pointer in
                  match K.find_opt k st with
                  | Some (Address d) -> K.add k (Address (Int64.add d delta)) st
                  | Some (Value _) | None -> K.remove k st)
              | Mir_phys.Instr.Late { op; uses; defs } -> (
                  (* address arithmetic on an address stays an address;
                     anything else a late form writes is forgotten *)
                  let st' = List.fold_left kill_view st (T.clobbers op) in
                  let st' = List.fold_left kill st' defs in
                  match (T.address_step op, uses, defs) with
                  | Some step, [ u ], [ (Loc.Reg _ as d) ] -> (
                      match
                        Option.bind (key_of st u) (fun k -> K.find_opt k st)
                      with
                      | Some (Address a) -> (
                          match key_of st' d with
                          | Some k -> K.add k (Address (Int64.add a step)) st'
                          | None -> st')
                      | _ -> st')
                  | _ -> st')
              | Mir_phys.Instr.Exec { instr; uses; defs } -> (
                  match instr.Mir_instr.op with
                  | Mir_sel.Op.Event _ -> st
                  | Mir_sel.Op.Machine op ->
                      List.iter2 (need st) uses (T.uses op);
                      let st = List.fold_left kill_view st (T.clobbers op) in
                      let st =
                        if
                          T.writes_flags op
                          && not
                               (List.exists
                                  (fun (r : Mir_value.t) ->
                                    Mir_type.equal r.Mir_value.ty Mir_type.Flags)
                                  instr.Mir_instr.results)
                        then kill st (Loc.Reg T.flags_view)
                        else st
                      in
                      let st =
                        List.fold_left forget_value st instr.Mir_instr.results
                      in
                      let st = List.fold_left kill st defs in
                      List.fold_left2 set st defs instr.Mir_instr.results))
            st b.Mir_phys.Block.body
        in
        (* entering [t] from a block whose selected edge is [e] *)
        let enter ~checking ~from (e : Mir_edge.t option)
            (t : (T.op, T.test) Mir_phys.Block.t) st =
          match (t.Mir_phys.Block.origin, e) with
          | Mir_phys.Origin.Edge _, _ -> st
          | Mir_phys.Origin.Block tb, Some e ->
              let params = (selected tb).Mir_block.params in
              let claims = t.Mir_phys.Block.entry in
              let at p =
                List.find_map
                  (fun ((q : Mir_value.t), l) ->
                    if Mir_value.equal p q then Some l else None)
                  claims
              in
              let pairs = List.combine params e.Mir_edge.args in
              if checking then
                List.iter
                  (fun (p, a) ->
                    match at p with
                    | Some l -> need ~block:from st l a
                    | None ->
                        fail ~block:t.Mir_phys.Block.id
                          (Problem.Missing_claim p))
                  pairs;
              let ok =
                List.filter
                  (fun (p, a) ->
                    match at p with Some l -> holds st l a | None -> false)
                  pairs
              in
              let st =
                List.fold_left (fun st (p, _) -> forget_value st p) st pairs
              in
              List.fold_left
                (fun st (p, _) ->
                  match at p with Some l -> set st l p | None -> st)
                st ok
          | Mir_phys.Origin.Block _, None -> st
        in
        (* the selected edge leaving physical block [b] by successor [index] *)
        let edge_of (b : (T.op, T.test) Mir_phys.Block.t) index =
          match b.Mir_phys.Block.origin with
          | Mir_phys.Origin.Block sb -> Some (selected_edge sb index)
          | Mir_phys.Origin.Edge { Mir_phys.Edge_ref.from; index } ->
              Some (selected_edge from index)
        in
        let entry_state =
          let sentry = selected sf.Mir_func.entry in
          if
            List.length sentry.Mir_block.params
            <> List.length pf.Mir_phys.Func.params
          then fail Problem.Realization;
          List.fold_left2
            (fun st (p : Mir_value.t) ((q : Mir_value.t), l) ->
              if not (Mir_value.equal p q) then
                fail (Problem.Missing_value { value = p; at = l });
              set st l p)
            (K.singleton (unit_of T.stack_pointer) (Address 0L))
            sentry.Mir_block.params pf.Mir_phys.Func.params
        in
        let states = Hashtbl.create 16 in
        let get_in id = Hashtbl.find_opt states (Mir_id.Block.to_int id) in
        Hashtbl.replace states
          (Mir_id.Block.to_int pf.Mir_phys.Func.entry)
          entry_state;
        let changed = ref true in
        while !changed do
          changed := false;
          List.iter
            (fun (b : (T.op, T.test) Mir_phys.Block.t) ->
              match get_in b.Mir_phys.Block.id with
              | None -> ()
              | Some st ->
                  let out = transfer ~checking:false b st in
                  List.iteri
                    (fun index t ->
                      let tb = physical t in
                      let incoming =
                        enter ~checking:false ~from:b.Mir_phys.Block.id
                          (edge_of b index) tb out
                      in
                      let next =
                        if Mir_id.Block.equal t pf.Mir_phys.Func.entry then None
                        else
                          match get_in t with
                          | None -> Some incoming
                          | Some old ->
                              let m = meet old incoming in
                              if K.equal ( = ) m old then None else Some m
                      in
                      match next with
                      | Some s ->
                          Hashtbl.replace states (Mir_id.Block.to_int t) s;
                          changed := true
                      | None -> ())
                    (Mir_phys.Term.successors b.Mir_phys.Block.terminator))
            pf.Mir_phys.Func.blocks
        done;
        (* the checking pass over the converged states *)
        List.iter
          (fun (b : (T.op, T.test) Mir_phys.Block.t) ->
            match get_in b.Mir_phys.Block.id with
            | None -> ()
            | Some st -> (
                let block = b.Mir_phys.Block.id in
                let out = transfer ~checking:true b st in
                List.iteri
                  (fun index t ->
                    ignore
                      (enter ~checking:true ~from:block (edge_of b index)
                         (physical t) out))
                  (Mir_phys.Term.successors b.Mir_phys.Block.terminator);
                match b.Mir_phys.Block.terminator with
                | Mir_phys.Term.Branch { test; uses; _ } ->
                    List.iter2 (need ~block out) uses (T.test_uses test)
                | Mir_phys.Term.Return { values } -> (
                    match b.Mir_phys.Block.origin with
                    | Mir_phys.Origin.Block sb -> (
                        match (selected sb).Mir_block.terminator with
                        | Mir_sel.Terminator.Return
                            { Mir_return.values = vs; _ } ->
                            if List.length vs <> List.length values then
                              fail ~block Problem.Successor;
                            List.iter2 (need ~block out) values vs
                        | _ -> fail ~block Problem.Successor)
                    | Mir_phys.Origin.Edge _ -> fail ~block Problem.Successor)
                | Mir_phys.Term.Jump _ -> ()))
          pf.Mir_phys.Func.blocks)
      sp.Mir_program.funcs
end
