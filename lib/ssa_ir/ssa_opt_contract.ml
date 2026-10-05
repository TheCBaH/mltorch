(* The multiplications seen so far, by the value they define: an add whose operand
   is one of them fuses. A value is defined once and used only where it is in
   scope, so a multiplication found by id dominates the add that names it. *)

type mul = { x : Ssa_value.t; y : Ssa_value.t; lanewise : bool }

let rec classify (op : Ssa_op.t) =
  match op with
  | Ssa_op.Float_binary (Expr.Value.Mul, x, y) ->
      Some { x; y; lanewise = false }
  | Ssa_op.Lanewise inner ->
      Option.map (fun m -> { m with lanewise = true }) (classify inner)
  | _ -> None

let adds (op : Ssa_op.t) =
  match op with
  | Ssa_op.Float_binary (Expr.Value.Add, a, b) -> Some (a, b, false)
  | Ssa_op.Lanewise (Ssa_op.Float_binary (Expr.Value.Add, a, b)) ->
      Some (a, b, true)
  | _ -> None

let pass ~scalar (p : Ssa_program.t) =
  let muls : (int, mul) Hashtbl.t = Hashtbl.create 32 in
  let rule t (s : Ssa_region.t Ssa_stmt.t) =
    match s with
    | Ssa_stmt.Instr i -> (
        let key (v : Ssa_value.t) = (v.Ssa_value.id :> int) in
        (match (classify i.Ssa_instr.op, i.Ssa_instr.results) with
        | Some m, [ r ] -> Hashtbl.replace muls (key r) m
        | _ -> ());
        match adds i.Ssa_instr.op with
        | Some (a, b, lanewise) when scalar || lanewise -> (
            let operand (v : Ssa_value.t) =
              match Hashtbl.find_opt muls (key v) with
              | Some m when Bool.equal m.lanewise lanewise -> Some m
              | Some _ | None -> None
            in
            (* a multiplication on the right is taken first *)
            let fused =
              match (operand b, operand a) with
              | Some m, _ -> Some (m, a)
              | None, Some m -> Some (m, b)
              | None, None -> None
            in
            match fused with
            | None -> [ s ]
            | Some (m, other) ->
                Ssa_rewrite.mark_changed t;
                let fma = Ssa_op.Float_fma (m.x, m.y, other) in
                [
                  Ssa_stmt.Instr
                    {
                      i with
                      Ssa_instr.op =
                        (if lanewise then Ssa_op.Lanewise fma else fma);
                    };
                ])
        | Some _ | None -> [ s ])
    | Ssa_stmt.For _ | Ssa_stmt.If _ | Ssa_stmt.Ordered_sum _ -> [ s ]
  in
  Ssa_rewrite.program rule p

let contracted (p : Ssa_program.t) =
  let n = ref 0 in
  let rec region (r : Ssa_region.t) = List.iter stmt r.Ssa_region.body
  and stmt : Ssa_region.t Ssa_stmt.t -> unit = function
    | Ssa_stmt.Instr { Ssa_instr.op = Ssa_op.Float_fma _; _ }
    | Ssa_stmt.Instr { Ssa_instr.op = Ssa_op.Lanewise (Ssa_op.Float_fma _); _ }
      ->
        incr n
    | Ssa_stmt.Instr _ -> ()
    | Ssa_stmt.For { body; _ } | Ssa_stmt.Ordered_sum { body; _ } -> region body
    | Ssa_stmt.If { then_; else_; _ } ->
        region then_;
        region else_
  in
  region p.Ssa_program.entry;
  !n
