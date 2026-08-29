(* Sharing loads. A load that repeats an earlier one (same buffer, same decode,
   same coordinate values) returns what the earlier one did, provided the earlier
   one dominates it and nothing between them may write what it reads. Sharing is
   sound for failure too: a repeat that would fail is preceded by the same load
   having failed already.

   The table of available loads is scoped like the program. Entering a loop
   discards what the loop may overwrite, because the first iteration is not the
   only one the body sees; leaving any statement discards what it may have
   written. The removed load's place on the effect chain is closed up, and the
   number of primitive loads falls, which is not a change of logical work. *)

let key (op : Ssa_op.t) =
  match op with
  | Ssa_op.Load { buffer; decode; _ }
  | Ssa_op.Load_in_bounds { buffer; decode; _ } ->
      let buffer_id (b : Ssa_id.Buffer.t) = (b :> int) in
      Some
        ( buffer_id buffer,
          Fmt.str "%s %s"
            (Ssa_op.Decode.name decode)
            (String.concat ","
               (List.map
                  (fun (v : Ssa_value.t) ->
                    string_of_int (v.Ssa_value.id :> int))
                  (Ssa_op.operands op))) )
  | _ -> None

let pass ~alias (p : Ssa_program.t) =
  (* each scope maps (buffer, address) to the value an earlier load produced *)
  let scopes : (int * string, Ssa_value.t) Hashtbl.t list ref =
    ref [ Hashtbl.create 32 ]
  in
  let table () = List.hd !scopes in
  let invalidate (summary : Ssa_effects.summary) =
    let t = table () in
    let dead =
      Hashtbl.fold
        (fun ((b, _) as k) _ acc ->
          if Ssa_effects.may_write alias summary (Ssa_id.Buffer.of_int b) then
            k :: acc
          else acc)
        t []
    in
    List.iter (Hashtbl.remove t) dead
  in
  let enter () = scopes := Hashtbl.copy (table ()) :: !scopes in
  let leave () = scopes := List.tl !scopes in
  (* a loop's body repeats: what it writes is not what the first iteration
     saw once the second runs *)
  let enter_stmt (s : Ssa_region.t Ssa_stmt.t) =
    match s with
    | Ssa_stmt.For { body; _ } | Ssa_stmt.Ordered_sum { body; _ } ->
        invalidate (Ssa_effects.of_region body)
    | Ssa_stmt.Instr _ | Ssa_stmt.If _ -> ()
  in
  let rule t (s : Ssa_region.t Ssa_stmt.t) =
    match s with
    | Ssa_stmt.Instr i -> (
        let summary = Ssa_effects.of_op i.Ssa_instr.op in
        match (key i.Ssa_instr.op, i.Ssa_instr.token) with
        | Some k, Some token -> (
            match Hashtbl.find_opt (table ()) k with
            | Some earlier ->
                let value = List.hd i.Ssa_instr.results in
                let out =
                  List.nth i.Ssa_instr.results
                    (List.length i.Ssa_instr.results - 1)
                in
                Ssa_rewrite.alias t ~from:value ~to_:earlier;
                Ssa_rewrite.alias t ~from:out ~to_:token;
                []
            | None ->
                Hashtbl.replace (table ()) k (List.hd i.Ssa_instr.results);
                [ s ])
        | _ ->
            invalidate summary;
            [ s ])
    | Ssa_stmt.For _ | Ssa_stmt.If _ | Ssa_stmt.Ordered_sum _ ->
        invalidate (Ssa_effects.of_stmt s);
        [ s ]
  in
  Ssa_rewrite.program ~enter ~leave ~enter_stmt rule p
