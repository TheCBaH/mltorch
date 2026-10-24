type mul = { x : Ssa_value.t; y : Ssa_value.t; lanewise : bool }

let rec product (op : Ssa_op.t) =
  match op with
  | Ssa_op.Float_binary (Expr.Value.Mul, x, y) ->
      Some { x; y; lanewise = false }
  | Ssa_op.Lanewise inner ->
      Option.map (fun m -> { m with lanewise = true }) (product inner)
  | _ -> None

let sum (op : Ssa_op.t) =
  match op with
  | Ssa_op.Float_binary (Expr.Value.Add, a, b) -> Some (a, b, false)
  | Ssa_op.Lanewise (Ssa_op.Float_binary (Expr.Value.Add, a, b)) ->
      Some (a, b, true)
  | _ -> None

let same (a : Ssa_value.t) (b : Ssa_value.t) =
  Ssa_id.Value.equal a.Ssa_value.id b.Ssa_value.id

let marks_reduction (r : Ssa_region.t) =
  List.exists
    (function
      | Ssa_stmt.Instr
          {
            Ssa_instr.op =
              ( Ssa_op.Mark Ssa_mark.Reduction
              | Ssa_op.Mark_lanes { mark = Ssa_mark.Reduction; _ } );
            _;
          } ->
          true
      | _ -> false)
    r.Ssa_region.body

(* The body's instructions by the value each defines. *)
let defining (r : Ssa_region.t) (v : Ssa_value.t) =
  List.find_map
    (function
      | Ssa_stmt.Instr ({ Ssa_instr.results = [ res ]; _ } as i) when same res v
        ->
          Some i
      | _ -> None)
    r.Ssa_region.body

(* The yields of a reduction loop's body whose accumulation fuses: each
   accumulator parameter's yield, as the fused instruction that replaces its
   add. *)
let fusions uses (body : Ssa_region.t) =
  match body.Ssa_region.params with
  | [] -> []
  | _iv :: carried ->
      List.concat
        (List.map2
           (fun (acc : Ssa_value.t) (yielded : Ssa_value.t) ->
             match defining body yielded with
             | None -> []
             | Some add -> (
                 match sum add.Ssa_instr.op with
                 | None -> []
                 | Some (a, b, lanewise) -> (
                     let term =
                       if same a acc then Some b
                       else if same b acc then Some a
                       else None
                     in
                     match term with
                     | Some term when Ssa_uses.use_count uses acc = 1 -> (
                         match
                           Option.bind (defining body term) (fun i ->
                               product i.Ssa_instr.op)
                         with
                         | Some m when Bool.equal m.lanewise lanewise ->
                             let fma = Ssa_op.Float_fma (m.x, m.y, acc) in
                             [
                               ( yielded.Ssa_value.id,
                                 if lanewise then Ssa_op.Lanewise fma else fma
                               );
                             ]
                         | Some _ | None -> [])
                     | Some _ | None -> [])))
           carried
           (List.filteri
              (fun k _ -> k < List.length carried)
              body.Ssa_region.yields))

let rec region uses (r : Ssa_region.t) =
  { r with Ssa_region.body = List.map (stmt uses) r.Ssa_region.body }

and stmt uses (s : Ssa_region.t Ssa_stmt.t) =
  match s with
  | Ssa_stmt.Instr _ -> s
  | Ssa_stmt.For f ->
      let body = region uses f.body in
      let fused = if marks_reduction body then fusions uses body else [] in
      let body =
        {
          body with
          Ssa_region.body =
            List.map
              (function
                | Ssa_stmt.Instr ({ Ssa_instr.results = [ r ]; _ } as i) as s
                  -> (
                    match List.assoc_opt r.Ssa_value.id fused with
                    | Some op -> Ssa_stmt.Instr { i with Ssa_instr.op }
                    | None -> s)
                | s -> s)
              body.Ssa_region.body;
        }
      in
      Ssa_stmt.For { f with body }
  | Ssa_stmt.If i ->
      Ssa_stmt.If
        { i with then_ = region uses i.then_; else_ = region uses i.else_ }
  | Ssa_stmt.Ordered_sum o ->
      Ssa_stmt.Ordered_sum { o with body = region uses o.body }

let program (p : Ssa_program.t) =
  let uses = Ssa_uses.build p in
  { p with Ssa_program.entry = region uses p.Ssa_program.entry }
