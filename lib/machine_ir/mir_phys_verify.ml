(* The allocated stage's structural rules, per target: every location fits
   the value it holds (bank and width; a condition only in the condition
   register; a slot or frame access exactly the value's bytes), no reserved
   register holds a value, operand counts match the selected form, fixed
   operands sit in their register, tied results share their use's unit, an
   early-clobber result overlaps no use, and every branch target exists. A
   realized frame access is based on the stack pointer or a reserved scratch
   register, stays inside the function's frame and encodes; every
   stack-pointer step encodes and keeps the stack aligned; a save moves a
   preserved or link register whole. Whether the right value is in each
   location is the checker's question, not this one's. *)

module P = Mir_diagnostic.Problem
module Loc = Mir_phys.Loc

module Make (T : Mir_sel.TARGET) = struct
  (* The bank and width a value of this type occupies. *)
  let shape (ty : Mir_type.t) =
    match ty with
    | Mir_type.Int Mir_width.W64 | Mir_type.Ptr -> Some (Mir_target.Bank.Gpr, 64)
    | Mir_type.Int (Mir_width.W8 | Mir_width.W16 | Mir_width.W32)
    | Mir_type.Pred ->
        Some (Mir_target.Bank.Gpr, 32)
    | Mir_type.F64 -> Some (Mir_target.Bank.Fpr, 64)
    | Mir_type.F32 -> Some (Mir_target.Bank.Fpr, 32)
    | Mir_type.Flags ->
        Some (Mir_target.Bank.Flags, T.flags_view.Mir_target.View.bits)
    | Mir_type.Vec (e, n) -> (
        (* a register-wide slice or its half *)
        match
          Int64.to_int (Mir_type.Elem.bytes e) * 8 * Mir_type.Lanes.to_int n
        with
        | (64 | 128) as bits -> Some (Mir_target.Bank.Fpr, bits)
        | _ -> None)
    | Mir_type.Mask _ | Mir_type.Order -> None

  let in_memory bits (l : Loc.t) =
    match l with
    | Loc.Slot { bytes; _ } | Loc.Mem { bytes; _ } ->
        Int64.equal bytes (Int64.of_int (bits / 8))
    | Loc.Reg _ -> false

  let fits (v : Mir_value.t) (l : Loc.t) =
    match (shape v.Mir_value.ty, l) with
    | Some (bank, bits), Loc.Reg view ->
        view.Mir_target.View.bank = bank && view.Mir_target.View.bits = bits
    | Some (Mir_target.Bank.Flags, _), (Loc.Slot _ | Loc.Mem _) -> false
    | Some (_, bits), (Loc.Slot _ | Loc.Mem _) -> in_memory bits l
    | None, _ -> false

  (* Whether running an instruction again recreates its one result: it reads
     nothing, is unordered (pure and total), and constrains, clobbers and
     changes condition state not at all. *)
  let rematerializable (i : T.op Mir_sel.Op.t Mir_instr.t) =
    match (i.Mir_instr.op, i.Mir_instr.results) with
    | Mir_sel.Op.Machine op, [ r ] ->
        T.uses op = []
        && (not (T.ordered op))
        && T.constraints op = []
        && T.clobbers op = []
        && (not (T.writes_flags op))
        && not (Mir_type.equal r.Mir_value.ty Mir_type.Flags)
    | _ -> false

  let is_reserved (v : Mir_target.View.t) =
    List.exists (Mir_target.View.overlap v) T.abi.Mir_target.Abi.reserved

  (* A value never lives in a reserved register. *)
  let value_location_ok (l : Loc.t) =
    match l with
    | Loc.Reg v -> not (is_reserved v)
    | Loc.Slot _ | Loc.Mem _ -> true

  let verify (p : (T.op, T.test) Mir_phys.Program.t) =
    Err.Escape.with_escape @@ fun esc ->
    List.iter
      (fun (f : (T.op, T.test) Mir_phys.Func.t) ->
        let reject ?block ?instr s =
          Err.Escape.throw esc
            {
              Mir_diagnostic.stage = Mir_diagnostic.Stage.Allocated;
              func = Some f.Mir_phys.Func.id;
              block;
              instr;
              problem = P.Target s;
            }
        in
        let frame_ok ?block (l : Loc.t) =
          match l with
          | Loc.Slot { slot; bytes } ->
              if
                not
                  (List.exists
                     (fun (s : Mir_phys.Slot.t) ->
                       Mir_id.Slot.equal s.Mir_phys.Slot.id slot
                       && Int64.compare bytes s.Mir_phys.Slot.bytes <= 0)
                     f.Mir_phys.Func.slots)
              then reject ?block "an undeclared slot"
          | Loc.Mem { base; offset; bytes } -> (
              if
                not
                  (Mir_target.View.equal base T.stack_pointer
                  || is_reserved base)
              then reject ?block "frame memory based on an allocatable register";
              if not (T.frame_offset_ok ~bytes offset) then
                reject ?block "a frame offset that does not encode";
              match f.Mir_phys.Func.frame with
              | None -> reject ?block "frame memory before realization"
              | Some frame ->
                  if
                    Mir_target.View.equal base T.stack_pointer
                    && Int64.compare (Int64.add offset bytes) frame > 0
                  then reject ?block "a frame access outside the frame")
          | Loc.Reg _ -> ()
        in
        let check ?block ?instr (v : Mir_value.t) l =
          if not (fits v l) then
            reject ?block ?instr "a location that does not fit its value";
          if not (value_location_ok l) then
            reject ?block ?instr "a reserved register";
          frame_ok ?block l
        in
        List.iter (fun (v, l) -> check v l) f.Mir_phys.Func.params;
        if Option.is_none (Mir_phys.Func.find_block f f.Mir_phys.Func.entry)
        then reject "no entry block";
        (* a realized frame and a pushed return address together keep the
           stack aligned *)
        (match f.Mir_phys.Func.frame with
        | Some frame ->
            if
              not
                (Int64.equal
                   (Int64.rem
                      (Int64.add frame T.call_push)
                      T.abi.Mir_target.Abi.stack_align)
                   0L)
            then reject "a frame size that breaks stack alignment"
        | None -> ());
        List.iter
          (fun (b : (T.op, T.test) Mir_phys.Block.t) ->
            let block = b.Mir_phys.Block.id in
            List.iter (fun (v, l) -> check ~block v l) b.Mir_phys.Block.entry;
            List.iter
              (function
                | Mir_phys.Instr.Move { dst; src; value } ->
                    check ~block value dst;
                    check ~block value src
                | Mir_phys.Instr.Save { dst; src } -> (
                    frame_ok ~block dst;
                    frame_ok ~block src;
                    let saved (v : Mir_target.View.t) =
                      List.exists (Mir_target.View.equal v)
                        T.abi.Mir_target.Abi.preserved
                      || (match T.link with
                        | Some l -> Mir_target.View.equal v l
                        | None -> false)
                      (* reserved scratch carries control state to and from
                         its save word *)
                      || is_reserved v
                         && not (Mir_target.View.equal v T.stack_pointer)
                    in
                    match (dst, src) with
                    | Loc.Mem { bytes; _ }, Loc.Reg v
                    | Loc.Reg v, Loc.Mem { bytes; _ } ->
                        if not (saved v) then
                          reject ~block
                            "a save of a register the convention does not \
                             preserve";
                        if
                          not
                            (Int64.equal (Int64.mul 8L bytes)
                               (Int64.of_int v.Mir_target.View.bits))
                        then reject ~block "a save narrower than its register"
                    | _ ->
                        reject ~block
                          "a save not between a register and the frame")
                | Mir_phys.Instr.Remat { instr; defs } -> (
                    if not (rematerializable instr) then
                      reject ~block ~instr:instr.Mir_instr.id
                        "a rematerialized instruction that reads, orders, \
                         constrains or clobbers";
                    match defs with
                    | [ (Loc.Reg _ as d) ] ->
                        check ~block ~instr:instr.Mir_instr.id
                          (List.hd instr.Mir_instr.results)
                          d
                    | _ ->
                        reject ~block ~instr:instr.Mir_instr.id
                          "a rematerialization not into one register")
                | Mir_phys.Instr.Sp delta ->
                    if not (T.stack_step_ok delta) then
                      reject ~block "a stack-pointer step that does not encode"
                | Mir_phys.Instr.Late { op; uses; defs } ->
                    if List.length (T.uses op) <> List.length uses then
                      reject ~block "late operand count";
                    List.iter
                      (fun (l : Loc.t) ->
                        match l with
                        | Loc.Reg v ->
                            if
                              not
                                (is_reserved v
                                || Mir_target.View.equal v T.stack_pointer
                                || v.Mir_target.View.bank
                                   = Mir_target.Bank.Control)
                            then
                              reject ~block
                                "a late form touching an allocatable register"
                        | Loc.Slot _ | Loc.Mem _ ->
                            reject ~block "a late form with a memory operand")
                      (uses @ defs)
                | Mir_phys.Instr.Exec { instr; uses; defs } -> (
                    let instr_id = instr.Mir_instr.id in
                    match instr.Mir_instr.op with
                    | Mir_sel.Op.Event _ | Mir_sel.Op.Undef _ ->
                        if uses <> [] || defs <> [] then
                          reject ~block ~instr:instr_id
                            "a target-neutral instruction with locations"
                    | Mir_sel.Op.Machine op ->
                        let operands = T.uses op in
                        let results = instr.Mir_instr.results in
                        if
                          List.length operands <> List.length uses
                          || List.length results <> List.length defs
                        then reject ~block ~instr:instr_id "operand count";
                        List.iter2 (check ~block ~instr:instr_id) operands uses;
                        List.iter2 (check ~block ~instr:instr_id) results defs;
                        List.iter
                          (function
                            | Mir_target.Constraint.Early_clobber k ->
                                if
                                  List.exists
                                    (Loc.overlap (List.nth defs k))
                                    uses
                                then
                                  reject ~block ~instr:instr_id
                                    "an early-clobber result overlaps a use"
                            | Mir_target.Constraint.Fixed_result
                                { result; view } ->
                                if
                                  not
                                    (Loc.equal (List.nth defs result)
                                       (Loc.Reg view))
                                then
                                  reject ~block ~instr:instr_id
                                    "a fixed result elsewhere"
                            | Mir_target.Constraint.Fixed_use { use; view } ->
                                if
                                  not
                                    (Loc.equal (List.nth uses use)
                                       (Loc.Reg view))
                                then
                                  reject ~block ~instr:instr_id
                                    "a fixed use elsewhere"
                            | Mir_target.Constraint.Tied { result; use } -> (
                                match
                                  (List.nth defs result, List.nth uses use)
                                with
                                | Loc.Reg d, Loc.Reg u
                                  when Mir_id.Unit.equal d.Mir_target.View.unit
                                         u.Mir_target.View.unit ->
                                    ()
                                | _ ->
                                    reject ~block ~instr:instr_id
                                      "a tied result not in its use's register"))
                          (T.constraints op);
                        List.iter
                          (function
                            | Loc.Slot _ | Loc.Mem _ ->
                                reject ~block ~instr:instr_id
                                  "an instruction operand in memory"
                            | Loc.Reg _ -> ())
                          (uses @ defs)))
              b.Mir_phys.Block.body;
            (match b.Mir_phys.Block.terminator with
            | Mir_phys.Term.Branch { test; uses; _ } ->
                let operands = T.test_uses test in
                if List.length operands <> List.length uses then
                  reject ~block "branch operand count"
                else List.iter2 (check ~block) operands uses;
                List.iter
                  (function
                    | Loc.Slot _ | Loc.Mem _ ->
                        reject ~block "a branch operand in memory"
                    | Loc.Reg _ -> ())
                  uses
            | Mir_phys.Term.Jump _ | Mir_phys.Term.Return _ -> ());
            List.iter
              (fun t ->
                if Option.is_none (Mir_phys.Func.find_block f t) then
                  reject ~block "a missing successor")
              (Mir_phys.Term.successors b.Mir_phys.Block.terminator))
          f.Mir_phys.Func.blocks)
      p.Mir_phys.Program.funcs;
    p
end
