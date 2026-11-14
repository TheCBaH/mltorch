open Ssa_ir
open Loop_ir_test

(* The value-and-domain provenance the structural gate reads: which logical value
   each instruction came from, kept through the passes. *)

let rec instrs_of (r : Ssa_region.t) =
  List.concat_map
    (function
      | Ssa_stmt.Instr i -> [ i ]
      | Ssa_stmt.For { body; _ } | Ssa_stmt.Ordered_sum { body; _ } ->
          instrs_of body
      | Ssa_stmt.If { then_; else_; _ } -> instrs_of then_ @ instrs_of else_)
    r.Ssa_region.body

(* Instructions per origin, in the order the origins first appear. *)
let attribution (p : Ssa_program.t) =
  List.fold_left
    (fun acc (i : Ssa_instr.t) ->
      let o = i.Ssa_instr.origin in
      match List.find_opt (fun (o', _) -> Ssa_origin.equal o o') acc with
      | Some _ ->
          List.map
            (fun (o', n) ->
              if Ssa_origin.equal o o' then (o', n + 1) else (o', n))
            acc
      | None -> acc @ [ (o, 1) ])
    []
    (instrs_of p.Ssa_program.entry)

let show name p =
  Fmt.pr "%s: %a@." name
    Fmt.(
      list ~sep:(any ", ") (fun ppf (o, n) -> pf ppf "%a x%d" Ssa_origin.pp o n))
    (attribution p)

let lowered plan =
  Err.or_raise ~pp_error:Ssa_lower.Ssa_lower_plan.pp_error
    (Ssa_lower.Ssa_lower_plan.lower plan)

let%expect_test "lowering marks each stored value's instructions" =
  let both =
    Loop_programs.chain ~outputs:[ Loop_fixtures.tid 1; Loop_fixtures.tid 2 ]
  in
  show "stored" (lowered (Fusion_plan.default both));
  let only_t2 = Loop_programs.chain ~outputs:[ Loop_fixtures.tid 2 ] in
  let fused, _ = Fusion_plan.plan only_t2 in
  show "fused" (lowered fused);
  [%expect {|
    stored: value b1 x18, value b2 x18
    fused: value b2 x22 |}]

let unknown p =
  match
    List.find_opt
      (fun (o, _) -> Ssa_origin.equal o Ssa_origin.Unknown)
      (attribution p)
  with
  | Some (_, n) -> n
  | None -> 0

let%expect_test "the rewriting passes keep every origin" =
  let both =
    Loop_programs.chain ~outputs:[ Loop_fixtures.tid 1; Loop_fixtures.tid 2 ]
  in
  let p = lowered (Fusion_plan.default both) in
  show "lowered" p;
  let q, _ =
    Ssa_opt.run ~passes:[ Ssa_opt.simplify; Ssa_opt.guards; Ssa_opt.collapse ] p
  in
  show "after simplify, guards, collapse" q;
  Fmt.pr "instructions with no origin: %d@." (unknown q);
  [%expect
    {|
    lowered: value b1 x18, value b2 x18
    after simplify, guards, collapse: value b1 x8, value b2 x6
    instructions with no origin: 0 |}]

let%expect_test "an attribution swapped between two values is seen" =
  let both =
    Loop_programs.chain ~outputs:[ Loop_fixtures.tid 1; Loop_fixtures.tid 2 ]
  in
  let p = lowered (Fusion_plan.default both) in
  (* the same instructions, the same totals: the origins of the two values
     exchanged *)
  let swap (o : Ssa_origin.t) =
    match o with
    | Ssa_origin.Value b when Ssa_id.Buffer.to_int b = 1 ->
        Ssa_origin.Value (Ssa_id.Buffer.of_int 2)
    | Ssa_origin.Value b when Ssa_id.Buffer.to_int b = 2 ->
        Ssa_origin.Value (Ssa_id.Buffer.of_int 1)
    | o -> o
  in
  let rec map_region (r : Ssa_region.t) =
    {
      r with
      Ssa_region.body =
        List.map
          (function
            | Ssa_stmt.Instr i ->
                Ssa_stmt.Instr
                  { i with Ssa_instr.origin = swap i.Ssa_instr.origin }
            | Ssa_stmt.For f -> Ssa_stmt.For { f with body = map_region f.body }
            | Ssa_stmt.If f ->
                Ssa_stmt.If
                  {
                    f with
                    then_ = map_region f.then_;
                    else_ = map_region f.else_;
                  }
            | Ssa_stmt.Ordered_sum f ->
                Ssa_stmt.Ordered_sum { f with body = map_region f.body })
          r.Ssa_region.body;
    }
  in
  let swapped = { p with Ssa_program.entry = map_region p.Ssa_program.entry } in
  let totals q = List.fold_left (fun n (_, k) -> n + k) 0 (attribution q) in
  Fmt.pr "same totals: %b; same attribution: %b@."
    (totals p = totals swapped)
    (attribution p = attribution swapped);
  [%expect {| same totals: true; same attribution: false |}]
