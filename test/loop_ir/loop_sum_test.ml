open Loop_ir
open Loop_fixtures
open Loop_programs

(* A structured sum is the loop the lowering always produced, kept as one
   operation: it prints as one, expands to exactly the plain lowering, and
   checks its own descriptor. *)

let sum_plan =
  Fusion_plan.default
    (reduction_kernel Expr.Reduction.Sum ~lo:(fst full) ~hi:(snd full))

let structured () =
  Err.or_raise ~pp_error:Loop_lower.pp_error
    (Loop_lower.lower_structured sum_plan)

let plain () =
  Err.or_raise ~pp_error:Loop_lower.pp_error
    (Loop_lower.lower_unoptimized sum_plan)

let%expect_test "a sum is one statement, and expands to the plain lowering" =
  let s = structured () and p = plain () in
  Fmt.pr "structured sums: %d (plain %d)@." (Loop_sum.count s)
    (Loop_sum.count p);
  Fmt.pr "%a@." Loop_pp.program s;
  Fmt.pr "expansion is the plain lowering: %b@."
    (Stdlib.compare (Loop_sum.program s) p = 0);
  Fmt.pr "expansion has no sum: %d@." (Loop_sum.count (Loop_sum.program s));
  [%expect
    {|
    structured sums: 1 (plain 0)
    in t0 f32 [C=3]
    out t1 f32 [C=1]
    for i0 in [0, 1):
      for i1 in [0, 1):
        for i2 in [0, 1):
          for i3 in [0, 1):
            for i4 in [0, 1):
              for i5 in [0, 1):
                x0 = sum 0 over i6 in [0, 3):
                  term load t0[i0,i1,i2,i3,i4,i6]
                store t1[i0,i1,i2,i3,i4,i5] = f32(round_f32(x0))
    expansion is the plain lowering: true
    expansion has no sum: 0 |}]

let%expect_test "the interpreter runs a structured program as its expansion" =
  let s = structured () in
  let bind id =
    if Tensor_id.equal id (tid 0) then
      Some
        (f32_tensor (s1c 3) (fun c ->
             float_of_int ((Vec6.offset (s1c 3) c :> int) + 1)))
    else None
  in
  let run p =
    match Err.payload (Loop_interp.run p ~bind) with
    | Ok m ->
        Fmt.str "%a"
          Fmt.(list ~sep:sp float)
          (cells (Tensor_id.Map.find (tid 1) m) 1)
    | Error _ -> "failed"
  in
  Fmt.pr "%s@.%s@." (run s) (run (plain ()));
  [%expect {|
    6
    6 |}]

let find_sum (p : Loop_program.t) =
  let rec go = function
    | [] -> None
    | (Loop_stmt.Reduce_sum _ as s) :: _ -> Some s
    | Loop_stmt.For { body; _ } :: rest -> (
        match go body with Some s -> Some s | None -> go rest)
    | _ :: rest -> go rest
  in
  Option.get (go p.Loop_program.body)

(* Rebuilds every sum with [f acc body term]'s body and term. *)
let with_sum f (p : Loop_program.t) =
  let rec map (s : Loop_stmt.t) =
    match s with
    | Loop_stmt.Reduce_sum { var; lo; hi; acc; seed; body; term; at } ->
        let body, term = f acc body term in
        Loop_stmt.Reduce_sum { var; lo; hi; acc; seed; body; term; at }
    | Loop_stmt.For r -> Loop_stmt.For { r with body = List.map map r.body }
    | s -> s
  in
  { p with Loop_program.body = List.map map p.Loop_program.body }

let verdict p =
  match Err.payload (Loop_sum.check p) with
  | Ok () -> "ok"
  | Error e -> Fmt.str "%a" Loop_sum.pp_error e

let%expect_test "the descriptor checks" =
  let s = structured () in
  Fmt.pr "valid: %s@." (verdict s);
  ignore (find_sum s);
  Fmt.pr "term reads acc: %s@."
    (verdict
       (with_sum
          (fun acc body _ -> (body, Loop_expr.Temp (Loop_carrier.Float, acc)))
          s));
  Fmt.pr "body assigns acc: %s@."
    (verdict
       (with_sum
          (fun acc _ term ->
            ( [ Loop_stmt.Assign (Loop_carrier.Float, acc, Loop_expr.Const 0.) ],
              term ))
          s));
  [%expect
    {|
    valid: ok
    term reads acc: a sum's term or body reads its accumulator 0
    body assigns acc: a sum's body assigns its accumulator 0 |}]
