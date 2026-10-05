(* How big a program is, and how much of it is checking. Nothing semantic: a
   pass's effect made visible, so a regression in what it removes is seen. *)

type t = {
  instrs : int;
  effectful : int;  (** instructions on the effect chain *)
  loads : int;  (** all loads, checked or in bounds *)
  checked : int;  (** operations that check something at run time *)
  loops : int;  (** [for] and [ordered_sum] *)
}

let zero = { instrs = 0; effectful = 0; loads = 0; checked = 0; loops = 0 }

let add a b =
  {
    instrs = a.instrs + b.instrs;
    effectful = a.effectful + b.effectful;
    loads = a.loads + b.loads;
    checked = a.checked + b.checked;
    loops = a.loops + b.loops;
  }

let of_op (op : Ssa_op.t) =
  let checks =
    match op with
    | Ssa_op.Check_access _ | Ssa_op.Check_gather _ | Ssa_op.Check_local _
    | Ssa_op.Check_scan _ | Ssa_op.Float_to_i64 _ | Ssa_op.I64_div _
    | Ssa_op.Index_add _ | Ssa_op.Index_of_i64 _ | Ssa_op.Index_scale _
    | Ssa_op.Load _ ->
        1
    | _ -> 0
  in
  let loads =
    match op with Ssa_op.Load _ | Ssa_op.Load_in_bounds _ -> 1 | _ -> 0
  in
  {
    instrs = 1;
    effectful = (if Ssa_op.effectful op then 1 else 0);
    loads;
    checked = checks;
    loops = 0;
  }

let rec of_region (r : Ssa_region.t) =
  List.fold_left
    (fun acc s ->
      add acc
        (match s with
        | Ssa_stmt.Instr i -> of_op i.Ssa_instr.op
        | Ssa_stmt.For { body; _ } | Ssa_stmt.Ordered_sum { body; _ } ->
            add { zero with loops = 1 } (of_region body)
        | Ssa_stmt.If { then_; else_; _ } ->
            add (of_region then_) (of_region else_)))
    zero r.Ssa_region.body

let of_program (p : Ssa_program.t) = of_region p.Ssa_program.entry
