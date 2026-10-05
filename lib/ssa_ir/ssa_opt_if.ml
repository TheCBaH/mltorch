(* If-conversion: a branch whose arms only compute values that are safe to
   compute anyway becomes a [Select] over those values.

   Both arms then run. That is exact for a pure operation, which cannot fail
   and does not count (failure is intrinsic to an effectful operation, see
   {!Ssa_op.effectful}), and for a load proved in bounds, which cannot fail
   either and is not logical work. Anything else in an arm (a checked
   operation, a mark, a store, a loop, another branch) keeps the branch.

   A speculated load keeps its place on the effect chain: the then arm's loads
   come first, then the else arm's, so the chain stays linear, and the branch's
   effect result is the last of them. A select is a value a vector loop can
   hold; a branch is not, which is why this runs before the vectorizer.

   An arm longer than [max_arm] statements is left a branch: evaluating both
   arms is then a cost the select does not clearly repay. *)

let max_arm = 16

let selectable (v : Ssa_value.t) =
  match v.Ssa_value.ty with
  | Ssa_type.Scalar (Ssa_type.F32 | Ssa_type.F64 | Ssa_type.I64 | Ssa_type.Index)
    ->
      true
  | _ -> false

let is_effect (v : Ssa_value.t) = Ssa_type.equal v.Ssa_value.ty Ssa_type.Effect

(* A statement that may run on a path the branch would not take. *)
let speculable = function
  | Ssa_stmt.Instr { Ssa_instr.token = None; op; _ } ->
      not (Ssa_op.effectful op)
  | Ssa_stmt.Instr { Ssa_instr.token = Some _; op = Ssa_op.Load_in_bounds _; _ }
    ->
      true
  | Ssa_stmt.Instr _ | Ssa_stmt.For _ | Ssa_stmt.If _ | Ssa_stmt.Ordered_sum _
    ->
      false

let last l = List.nth l (List.length l - 1)

(* The effect-chain links of an arm: each tokened instruction's input and
   output. *)
let links (r : Ssa_region.t) =
  List.filter_map
    (function
      | Ssa_stmt.Instr ({ Ssa_instr.token = Some tok; _ } as i) ->
          Some (tok, last i.Ssa_instr.results)
      | _ -> None)
    r.Ssa_region.body

let last_out ~incoming = function [] -> incoming | l -> snd (last l)

(* The same body with its first tokened instruction taking [tok] instead. *)
let rethread (r : Ssa_region.t) tok =
  let seen = ref false in
  List.map
    (function
      | Ssa_stmt.Instr ({ Ssa_instr.token = Some _; _ } as i) when not !seen ->
          seen := true;
          Ssa_stmt.Instr { i with Ssa_instr.token = Some tok }
      | s -> s)
    r.Ssa_region.body

let pass (p : Ssa_program.t) =
  let rule t (s : Ssa_region.t Ssa_stmt.t) =
    match s with
    | Ssa_stmt.If { cond; results; then_; else_ }
      when List.length then_.Ssa_region.body <= max_arm
           && List.length else_.Ssa_region.body <= max_arm
           && List.for_all speculable then_.Ssa_region.body
           && List.for_all speculable else_.Ssa_region.body
           && List.length results = List.length then_.Ssa_region.yields
           && List.length results = List.length else_.Ssa_region.yields -> (
        let yields =
          List.combine results
            (List.combine then_.Ssa_region.yields else_.Ssa_region.yields)
        in
        let yes_links = links then_ and no_links = links else_ in
        (* the effect both arms start from *)
        let incoming =
          match (yes_links, no_links) with
          | (tok, _) :: _, _ | [], (tok, _) :: _ -> Some tok
          | [], [] ->
              List.find_map
                (fun ((r : Ssa_value.t), ((a : Ssa_value.t), _)) ->
                  if is_effect r then Some a else None)
                yields
        in
        let same_start =
          match (yes_links, no_links) with
          | (a, _) :: _, (b, _) :: _ -> Ssa_value.equal a b
          | _ -> true
        in
        match incoming with
        | Some e0
          when same_start
               && List.for_all
                    (fun ( (r : Ssa_value.t),
                           ((a : Ssa_value.t), (b : Ssa_value.t)) ) ->
                      if is_effect r then
                        Ssa_value.equal a (last_out ~incoming:e0 yes_links)
                        && Ssa_value.equal b (last_out ~incoming:e0 no_links)
                      else selectable r)
                    yields ->
            let yes_final = last_out ~incoming:e0 yes_links in
            let no_body =
              if yes_links <> [] && no_links <> [] then rethread else_ yes_final
              else else_.Ssa_region.body
            in
            let final =
              if no_links <> [] then last_out ~incoming:e0 no_links
              else yes_final
            in
            let selects =
              List.concat_map
                (fun ((r : Ssa_value.t), ((a : Ssa_value.t), (b : Ssa_value.t)))
                   ->
                  if is_effect r then (
                    Ssa_rewrite.alias t ~from:r ~to_:final;
                    [])
                  else
                    [
                      Ssa_stmt.Instr
                        {
                          Ssa_instr.results = [ r ];
                          op = Ssa_op.Select (cond, a, b);
                          token = None;
                          origin = Ssa_origin.Unknown;
                        };
                    ])
                yields
            in
            Ssa_rewrite.mark_changed t;
            then_.Ssa_region.body @ no_body @ selects
        | Some _ | None -> [ s ])
    | Ssa_stmt.Instr _ | Ssa_stmt.For _ | Ssa_stmt.If _ | Ssa_stmt.Ordered_sum _
      ->
        [ s ]
  in
  Ssa_rewrite.program rule p
