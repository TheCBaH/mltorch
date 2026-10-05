(* Removing what the ranges prove cannot happen, and loops whose trip count they
   fix.

   - A checked index operation whose result stays in the domain becomes the
     in-domain operation, which is pure: its effect result is replaced by the
     effect it consumed, so the chain is reconnected rather than broken.
   - A load whose coordinate is inside the buffer becomes a load that checks
     nothing; a check that cannot fire is deleted.
   - A loop that cannot run is replaced by its initializers, and one that runs
     exactly once by its body with the loop's parameters bound.

   Every one of these is a claim {!Ssa_range} re-derives, so a wrong range
   shows up as a program the verifier rejects or an execution that differs, never
   as a silent change. Marks are never removed: a loop that does run keeps the
   marks of its body, and one that does not never ran them. *)

let reconnect t (i : Ssa_instr.t) =
  (* the last result of an effectful operation is the effect it returns *)
  match (i.Ssa_instr.token, List.rev i.Ssa_instr.results) with
  | Some token, out :: _ -> Ssa_rewrite.alias t ~from:out ~to_:token
  | _ -> invalid_arg "Ssa_opt_guards: an effectful operation without an effect"

let without_effect (i : Ssa_instr.t) op =
  let values =
    match List.rev i.Ssa_instr.results with
    | _ :: rest -> List.rev rest
    | [] -> []
  in
  { i with Ssa_instr.op; results = values; token = None }

let subset ranges v ~lo ~hi = Ssa_range.(subset (range ranges v) ~lo ~hi)

let pass (p : Ssa_program.t) =
  let ranges = Ssa_range.analyze p in
  let rule t (s : Ssa_region.t Ssa_stmt.t) =
    match s with
    | Ssa_stmt.Instr i -> (
        let drop () =
          reconnect t i;
          []
        in
        let pure op =
          reconnect t i;
          [ Ssa_stmt.Instr (without_effect i op) ]
        in
        match i.Ssa_instr.op with
        | Ssa_op.Index_add (a, b) when Ssa_range.add_stays_in_domain ranges a b
          ->
            pure (Ssa_op.Index_add_in_domain (a, b))
        | Ssa_op.Index_scale (k, a)
          when Ssa_range.scale_stays_in_domain ranges k a ->
            pure (Ssa_op.Index_scale_in_domain (k, a))
        | Ssa_op.Load { buffer; at; decode } -> (
            match Ssa_program.find_buffer p buffer with
            | Some b when Ssa_range.in_bounds ranges b at ->
                Ssa_rewrite.mark_changed t;
                [
                  Ssa_stmt.Instr
                    {
                      i with
                      Ssa_instr.op =
                        Ssa_op.Load_in_bounds { buffer; at; decode };
                    };
                ]
            | Some _ | None -> [ s ])
        | Ssa_op.Check_access { buffer; at } -> (
            match Ssa_program.find_buffer p buffer with
            | Some b when Ssa_range.in_bounds ranges b at -> drop ()
            | Some _ | None -> [ s ])
        | Ssa_op.Check_gather { raw; extent }
          when subset ranges raw ~lo:(Int64.neg extent) ~hi:(Int64.pred extent)
          ->
            drop ()
        | Ssa_op.Check_local { at; extent; _ }
          when subset ranges at ~lo:0L ~hi:(Int64.pred extent) ->
            drop ()
        | Ssa_op.Check_scan { row; lane; row_extent; lane_extent; _ }
          when subset ranges row ~lo:0L ~hi:(Int64.pred row_extent)
               && subset ranges lane ~lo:0L ~hi:(Int64.pred lane_extent) ->
            drop ()
        | _ -> [ s ])
    | Ssa_stmt.For { lo; hi; step; inits; results; body } -> (
        match Ssa_range.trips ranges ~lo ~hi ~step with
        | Ssa_range.Zero ->
            (* the body never runs: the loop returns what it was given *)
            List.iter2
              (fun r i -> Ssa_rewrite.alias t ~from:r ~to_:i)
              results inits;
            []
        | Ssa_range.Exactly 1L ->
            (* one trip: the body with its parameters bound *)
            (match body.Ssa_region.params with
            | iv :: carried ->
                Ssa_rewrite.alias t ~from:iv ~to_:lo;
                List.iter2
                  (fun c i -> Ssa_rewrite.alias t ~from:c ~to_:i)
                  carried inits
            | [] -> ());
            let inlined =
              List.map (Ssa_rewrite.resolve_deep t) body.Ssa_region.body
            in
            List.iter2
              (fun r y -> Ssa_rewrite.alias t ~from:r ~to_:y)
              results
              (List.map (Ssa_rewrite.resolve t) body.Ssa_region.yields);
            inlined
        | Ssa_range.Exactly _ | Ssa_range.At_least_one | Ssa_range.Unknown ->
            [ s ])
    | Ssa_stmt.If _ | Ssa_stmt.Ordered_sum _ -> [ s ]
  in
  Ssa_rewrite.program rule p
