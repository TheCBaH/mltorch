(* The allocated stage as text: each instruction's selected form with its
   operands named by the locations the allocator gave them, blocks in list
   order. A virtual value an operand names is replaced by its location where
   the form's uses and the instruction's results pair with the lists
   positionally, and printed as its id where they do not. *)

module Make (T : Mir_sel.TARGET) = struct
  type program = (T.op, T.test) Mir_phys.Program.t

  let locs = Fmt.(list ~sep:(any ", ") Mir_phys.Loc.pp)

  (* the value printer: a virtual value shown as the location it was given *)
  let value pairs fmt (v : Mir_value.t) =
    match
      List.find_opt
        (fun ((w : Mir_value.t), _) ->
          Mir_id.Value.equal w.Mir_value.id v.Mir_value.id)
        pairs
    with
    | Some (_, l) -> Mir_phys.Loc.pp fmt l
    | None -> Fmt.pf fmt "%%%a" Mir_id.Value.pp v.Mir_value.id

  let pair vs ls =
    if List.length vs = List.length ls then List.combine vs ls else []

  let pp_sel pairs fmt = function
    | Mir_sel.Op.Event (e, k) -> Fmt.pf fmt "event %s x%Ld" (Mir_event.name e) k
    | Mir_sel.Op.Machine o -> T.pp_op (value pairs) fmt o
    | Mir_sel.Op.Undef v -> Fmt.pf fmt "undef %a" Mir_id.View.pp v

  let uses_of = function
    | Mir_sel.Op.Machine o -> T.uses o
    | Mir_sel.Op.Event _ | Mir_sel.Op.Undef _ -> []

  let pp_instr fmt = function
    | Mir_phys.Instr.Exec { instr; uses; defs } ->
        let pairs =
          pair (uses_of instr.Mir_instr.op) uses
          @ pair instr.Mir_instr.results defs
        in
        if defs <> [] then Fmt.pf fmt "%a = " locs defs;
        pp_sel pairs fmt instr.Mir_instr.op
    | Mir_phys.Instr.Late { op; uses; defs } ->
        let pairs = pair (T.uses op) uses in
        if defs <> [] then Fmt.pf fmt "%a = " locs defs;
        Fmt.pf fmt "late %a" (fun fmt -> T.pp_op (value pairs) fmt) op
    | Mir_phys.Instr.Move { dst; src; _ } ->
        Fmt.pf fmt "%a = move %a" Mir_phys.Loc.pp dst Mir_phys.Loc.pp src
    | Mir_phys.Instr.Remat { instr; defs } ->
        let pairs = pair instr.Mir_instr.results defs in
        Fmt.pf fmt "%a = remat %a" locs defs (pp_sel pairs) instr.Mir_instr.op
    | Mir_phys.Instr.Save { dst; src } ->
        Fmt.pf fmt "save %a <- %a" Mir_phys.Loc.pp dst Mir_phys.Loc.pp src
    | Mir_phys.Instr.Sp k -> Fmt.pf fmt "sp %+Ld" k

  let pp_term fmt = function
    | Mir_phys.Term.Branch { test; uses; then_; else_ } ->
        let pairs = pair (T.test_uses test) uses in
        Fmt.pf fmt "branch %a, %a, %a"
          (fun fmt -> T.pp_test (value pairs) fmt)
          test Mir_id.Block.pp then_ Mir_id.Block.pp else_
    | Mir_phys.Term.Jump b -> Fmt.pf fmt "jump %a" Mir_id.Block.pp b
    | Mir_phys.Term.Return { values } -> Fmt.pf fmt "return %a" locs values

  let pp_block fmt (b : (T.op, T.test) Mir_phys.Block.t) =
    Fmt.pf fmt "@,%a(%a):" Mir_id.Block.pp b.Mir_phys.Block.id
      Fmt.(list ~sep:(any ", ") (using snd Mir_phys.Loc.pp))
      b.Mir_phys.Block.entry;
    List.iter (fun i -> Fmt.pf fmt "@,  %a" pp_instr i) b.Mir_phys.Block.body;
    Fmt.pf fmt "@,  %a" pp_term b.Mir_phys.Block.terminator

  let pp_func fmt (f : (T.op, T.test) Mir_phys.Func.t) =
    Fmt.pf fmt "func %s(%a) -> (%a)%a {" f.Mir_phys.Func.name
      Fmt.(list ~sep:(any ", ") (using snd Mir_phys.Loc.pp))
      f.Mir_phys.Func.params locs f.Mir_phys.Func.results
      Fmt.(option (fun fmt n -> pf fmt " frame %Ld" n))
      f.Mir_phys.Func.frame;
    List.iter
      (fun (s : Mir_phys.Slot.t) ->
        Fmt.pf fmt "@,  slot %a: %Ld bytes align %Ld" Mir_id.Slot.pp
          s.Mir_phys.Slot.id s.Mir_phys.Slot.bytes s.Mir_phys.Slot.align)
      f.Mir_phys.Func.slots;
    List.iter (pp_block fmt) f.Mir_phys.Func.blocks;
    Fmt.pf fmt "@,}"

  let pp fmt (p : program) =
    Fmt.pf fmt "@[<v>target %s [%a]" T.name
      Fmt.(list ~sep:(any " ") (using Mir_target.Feature.name string))
      p.Mir_phys.Program.features;
    List.iter
      (fun f ->
        let main =
          if Mir_id.Func.equal f.Mir_phys.Func.id p.Mir_phys.Program.main then
            " main"
          else ""
        in
        Fmt.pf fmt "@,%a%s %a" Mir_id.Func.pp f.Mir_phys.Func.id main pp_func f)
      p.Mir_phys.Program.funcs;
    Fmt.pf fmt "@]"
end
