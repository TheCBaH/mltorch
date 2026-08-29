(* Exact scalar simplification: constant folding, pure common subexpressions
   and dead pure values.

   Folding evaluates through {!Ssa_scalar}, the same definition execution uses,
   so a folded value is the value the interpreter would have computed. A checked
   operation folds only when its operands are constants it provably succeeds on,
   and then its effect is reconnected. No algebraic identity is applied: [x + 0]
   is not [x] (signed zero), and [x * 1] is left alone too.

   A dead value is deleted only if it is pure. A checked operation, a load, a
   mark, a meter operation or a store stays whether or not its result is used:
   deleting one would change what fails and what is counted. *)

let is_pure op = not (Ssa_op.effectful op)

(* ---- folding --------------------------------------------------------------- *)

let fold (p : Ssa_program.t) =
  let consts : (int, Ssa_const.t) Hashtbl.t = Hashtbl.create 64 in
  let const_of (v : Ssa_value.t) =
    Hashtbl.find_opt consts (v.Ssa_value.id :> int)
  in
  let all_const vs = List.for_all (fun v -> const_of v <> None) vs in
  let get v =
    match const_of v with
    | Some c -> Ssa_scalar.of_const c
    | None -> invalid_arg "Ssa_opt_simplify: an operand that is not constant"
  in
  let rule t (s : Ssa_region.t Ssa_stmt.t) =
    match s with
    | Ssa_stmt.Instr i -> (
        let result () = List.hd i.Ssa_instr.results in
        let to_const (r : Ssa_value.t) value =
          match Ssa_scalar.to_const r.Ssa_value.ty value with
          | Some c -> Some c
          | None -> None
        in
        let replace_with (r : Ssa_value.t) c =
          Hashtbl.replace consts (r.Ssa_value.id :> int) c;
          Ssa_rewrite.mark_changed t;
          Ssa_stmt.Instr
            {
              i with
              Ssa_instr.op = Ssa_op.Const c;
              results = [ r ];
              token = None;
            }
        in
        let fold_checked eval =
          (* a checked operation on constants that provably succeeds *)
          match eval () with
          | Some c ->
              let r = result () in
              (match (i.Ssa_instr.token, List.rev i.Ssa_instr.results) with
              | Some token, out :: _ -> Ssa_rewrite.alias t ~from:out ~to_:token
              | _ -> ());
              [ replace_with r c ]
          | None -> [ s ]
        in
        let operands = Ssa_op.operands i.Ssa_instr.op in
        match i.Ssa_instr.op with
        | Ssa_op.Const c ->
            Hashtbl.replace consts ((result ()).Ssa_value.id :> int) c;
            [ s ]
        | Ssa_op.Select (p, a, b) when const_of p <> None -> (
            match const_of p with
            | Some (Ssa_const.Pred chosen) ->
                Ssa_rewrite.alias t ~from:(result ())
                  ~to_:(if chosen then a else b);
                []
            | _ -> [ s ])
        | Ssa_op.Index_add (a, b) when all_const [ a; b ] ->
            fold_checked (fun () ->
                match (get a, get b) with
                | Ssa_scalar.I x, Ssa_scalar.I y ->
                    let r = Int64.add x y in
                    if Ssa_const.in_index_domain r then Some (Ssa_const.Index r)
                    else None
                | _ -> None)
        | Ssa_op.Index_scale (k, a) when all_const [ a ] ->
            fold_checked (fun () ->
                match get a with
                | Ssa_scalar.I x ->
                    let r = Int64.mul k x in
                    if Ssa_const.in_index_domain r then Some (Ssa_const.Index r)
                    else None
                | _ -> None)
        | Ssa_op.Index_of_i64 a when all_const [ a ] ->
            fold_checked (fun () ->
                match get a with
                | Ssa_scalar.I x when Ssa_const.in_index_domain x ->
                    Some (Ssa_const.Index x)
                | _ -> None)
        | Ssa_op.I64_div (a, b) when all_const [ a; b ] ->
            fold_checked (fun () ->
                match (get a, get b) with
                | Ssa_scalar.I x, Ssa_scalar.I y
                  when (not (Int64.equal y 0L))
                       && not
                            (Int64.equal x Int64.min_int && Int64.equal y (-1L))
                  ->
                    Some (Ssa_const.I64 (Int64.div x y))
                | _ -> None)
        | Ssa_op.Float_to_i64 a when all_const [ a ] ->
            fold_checked (fun () ->
                match get a with
                | Ssa_scalar.F x -> (
                    match
                      Err.payload
                        (Expr.Value.i64_of_float x
                          : (int64, Expr.Value.i64_from_float_error) Err.t)
                    with
                    | Ok n -> Some (Ssa_const.I64 n)
                    | Error _ -> None)
                | _ -> None)
        | op when is_pure op && operands <> [] && all_const operands -> (
            let r = result () in
            match Ssa_scalar.eval op ~result:r.Ssa_value.ty ~get with
            | Some value -> (
                match to_const r value with
                | Some c -> [ replace_with r c ]
                | None -> [ s ])
            | None -> [ s ])
        | _ -> [ s ])
    | Ssa_stmt.If { cond; results; then_; else_ } -> (
        match const_of cond with
        | Some (Ssa_const.Pred chosen) ->
            (* only the selected arm ever runs: its body, in this scope *)
            let arm = if chosen then then_ else else_ in
            let body =
              List.map (Ssa_rewrite.resolve_deep t) arm.Ssa_region.body
            in
            List.iter2
              (fun r y -> Ssa_rewrite.alias t ~from:r ~to_:y)
              results
              (List.map (Ssa_rewrite.resolve t) arm.Ssa_region.yields);
            body
        | _ -> [ s ])
    | Ssa_stmt.For _ | Ssa_stmt.Ordered_sum _ -> [ s ]
  in
  Ssa_rewrite.program rule p

(* ---- common subexpressions ---------------------------------------------------- *)

(* A value computed again by an identical pure operation on the same operands
   is the earlier value. The table is scoped like the program: a region sees
   what its enclosing regions computed before it, never a sibling's. *)
let cse (p : Ssa_program.t) =
  let scopes = ref [ Hashtbl.create 64 ] in
  let enter () = scopes := Hashtbl.copy (List.hd !scopes) :: !scopes in
  let leave () = scopes := List.tl !scopes in
  let rule t (s : Ssa_region.t Ssa_stmt.t) =
    match s with
    | Ssa_stmt.Instr i
      when is_pure i.Ssa_instr.op && List.length i.Ssa_instr.results = 1 -> (
        let key = Ssa_op.key i.Ssa_instr.op in
        let table = List.hd !scopes in
        match Hashtbl.find_opt table key with
        | Some earlier ->
            Ssa_rewrite.alias t ~from:(List.hd i.Ssa_instr.results) ~to_:earlier;
            []
        | None ->
            Hashtbl.replace table key (List.hd i.Ssa_instr.results);
            [ s ])
    | _ -> [ s ]
  in
  Ssa_rewrite.program ~enter ~leave rule p

(* ---- dead values ------------------------------------------------------------- *)

let dce (p : Ssa_program.t) =
  let uses = Ssa_uses.build p in
  let rule t (s : Ssa_region.t Ssa_stmt.t) =
    match s with
    | Ssa_stmt.Instr i
      when is_pure i.Ssa_instr.op
           && List.for_all
                (fun r -> Ssa_uses.use_count uses r = 0)
                i.Ssa_instr.results ->
        Ssa_rewrite.mark_changed t;
        []
    | _ -> [ s ]
  in
  Ssa_rewrite.program rule p

(* One round of each, repeated while any changes anything. *)
let run (p : Ssa_program.t) =
  let rec go p changed_any rounds =
    if rounds = 0 then (p, changed_any)
    else
      let p, c1 = fold p in
      let p, c2 = cse p in
      let p, c3 = dce p in
      if c1 || c2 || c3 then go p true (rounds - 1) else (p, changed_any)
  in
  go p false 16
