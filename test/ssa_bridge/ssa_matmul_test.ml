open Ssa_bridge
open Ssa_fixtures
open Loop_ir_test

let case (m, k, n) =
  let plan = Fusion_plan.default (matmul_kernel ~m ~k ~n) in
  let a = operand 3 (m * k) and b = operand 5 (k * n) in
  let bind = matmul_bind ~m ~k ~n ~a ~b in
  Fmt.pr "%dx%dx%d: %a" m k n Ssa_check.pp_verdict (Ssa_check.run plan ~bind);
  match Ssa_check.marks plan ~bind with
  | Ok (ssa, loop) ->
      Fmt.pr ", reductions ssa=%d loop=%d (%s)@." ssa.Ssa_check.reductions
        loop.Ssa_check.reductions
        (if ssa = loop then "same work" else "DIFFERENT WORK")
  | Error _ -> Fmt.pr ", no marks@."

let%expect_test
    "matmul agrees with the reference and the Loop work, odd shapes too" =
  List.iter case
    [ (1, 1, 1); (1, 3, 2); (3, 1, 4); (5, 7, 3); (4, 4, 4); (2, 9, 1) ];
  [%expect
    {|
    1x1x1: agree, reductions ssa=1 loop=1 (same work)
    1x3x2: agree, reductions ssa=6 loop=6 (same work)
    3x1x4: agree, reductions ssa=12 loop=12 (same work)
    5x7x3: agree, reductions ssa=105 loop=105 (same work)
    4x4x4: agree, reductions ssa=64 loop=64 (same work)
    2x9x1: agree, reductions ssa=18 loop=18 (same work)
    |}]

let%expect_test "an empty or reversed reduction range returns the seed" =
  let run ~lo ~hi =
    let plan =
      Fusion_plan.default
        (Loop_programs.reduction_kernel Expr.Reduction.Sum
           ~lo:(Expr.Index.assume_position (Expr.Index.const lo))
           ~hi:(Expr.Index.const hi))
    in
    let shape = Loop_programs.s1c 3 in
    let bind id =
      if Tensor_id.equal id (Loop_fixtures.tid 0) then
        Some (Loop_fixtures.f32_tensor shape (fun _ -> -0.))
      else None
    in
    Fmt.pr "[%d,%d): %a@." lo hi Ssa_check.pp_verdict (Ssa_check.run plan ~bind)
  in
  run ~lo:0 ~hi:0;
  run ~lo:2 ~hi:1;
  run ~lo:0 ~hi:3;
  [%expect {|
    [0,0): agree
    [2,1): agree
    [0,3): agree
    |}]

(* A signed zero is a value: an all -0. sum is +0. because the seed is +0., not
   the first term. Both executors and the reference agree on it. *)
let%expect_test "the sum is seeded at +0, so a sum of -0 is +0" =
  let plan =
    Fusion_plan.default
      (Loop_programs.reduction_kernel Expr.Reduction.Sum ~lo:Expr.Index.zero
         ~hi:(Expr.Index.const 3))
  in
  let shape = Loop_programs.s1c 3 in
  let bind id =
    if Tensor_id.equal id (Loop_fixtures.tid 0) then
      Some (Loop_fixtures.f32_tensor shape (fun _ -> -0.))
    else None
  in
  (match
     Err.payload
       (Ssa_lower.Ssa_exec.run plan
          (Err.or_raise ~pp_error:Ssa_lower.Ssa_lower_plan.pp_error
             (Ssa_lower.Ssa_lower_plan.lower plan))
          ~bind)
   with
  | Ok m ->
      let t = Tensor_id.Map.find (Loop_fixtures.tid 1) m in
      Fmt.pr "%h@." (Tensor.read t Vec6.origin)
  | Error e -> Fmt.pr "%a@." Ssa_lower.Ssa_exec.pp_error e);
  [%expect {| 0x0p+0 |}]
