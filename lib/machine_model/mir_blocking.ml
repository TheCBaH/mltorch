(* Pressure feedback to structured SSA blocking. The candidates are bounded and
   each already legal: the unblocked exact program, and the exact program with
   independent outputs blocked in each group size the SSA pass itself admits
   for some loop of the kernel. Each is lowered and taken through the target's
   production pipeline; the choice is the candidate whose hot loops spill
   least, the largest group among equals. Blocking keeps every output's
   operations and their order, so no candidate changes a result or the
   precision — a spill is reported or avoided, never hidden. *)

module Policy = struct
  type t =
    | Feedback
        (** per invocation, the candidate whose hot loops spill least, the
            largest group among equals *)
    | Group of Ssa_ir.Ssa_blocking.Group.t
        (** this group wherever a loop admits it *)
    | Unblocked
end

(* The group sizes the SSA pass considers on its own. *)
let groups = List.map Ssa_ir.Ssa_blocking.Group.of_int [ 8; 4; 2 ]

module Candidate = struct
  type t = {
    group : Ssa_ir.Ssa_blocking.Group.t;  (** 1: unblocked *)
    pressure : (Machine_alloc.Mir_pressure.t, string) result;
  }
end

module Decision = struct
  type t = {
    candidates : Candidate.t list;
    chosen : Ssa_ir.Ssa_blocking.Group.t;
  }

  let pp fmt t =
    Fmt.pf fmt "@[<v>chosen %d@,%a@]"
      (t.chosen :> int)
      Fmt.(
        list ~sep:cut (fun fmt (c : Candidate.t) ->
            Fmt.pf fmt "group %d: %a"
              (c.Candidate.group :> int)
              (Fmt.result ~ok:Machine_alloc.Mir_pressure.pp ~error:Fmt.string)
              c.Candidate.pressure))
      t.candidates
end

(* The chosen candidate: least hot spilling, then the largest group; a
   candidate whose pressure is unknown is never chosen. *)
let decide (candidates : Candidate.t list) =
  let best =
    List.fold_left
      (fun acc (c : Candidate.t) ->
        match (c.Candidate.pressure, acc) with
        | Error _, _ -> acc
        | Ok p, None -> Some (Machine_alloc.Mir_pressure.hot_spills p, c)
        | Ok p, Some (h, b) ->
            let h' = Machine_alloc.Mir_pressure.hot_spills p in
            if h' < h || (h' = h && c.Candidate.group > b.Candidate.group) then
              Some (h', c)
            else acc)
      None candidates
  in
  {
    Decision.candidates;
    chosen =
      (match best with
      | Some (_, c) -> c.Candidate.group
      | None -> Ssa_ir.Ssa_blocking.Group.of_int 1);
  }
