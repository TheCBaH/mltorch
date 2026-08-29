open Ssa_bridge
open Ssa_fixtures
open Loop_ir_test

let two24 = Loop_programs.two24

let%expect_test "an inner f32 boundary survives, stored or inlined" =
  let shape = Loop_fixtures.shape_w 2 in
  let bind = bind_data ~shape [| two24; 1. |] in
  let check name plan =
    Fmt.pr "%s: %a@." name Ssa_check.pp_verdict (Ssa_check.run plan ~bind);
    let program =
      Err.or_raise ~pp_error:Ssa_lower.Ssa_lower_plan.pp_error
        (Ssa_lower.Ssa_lower_plan.lower plan)
    in
    match Err.payload (Ssa_lower.Ssa_exec.run plan program ~bind) with
    | Ok m ->
        let t = Tensor_id.Map.find (Loop_fixtures.tid 2) m in
        Fmt.pr "  t2 = %g %g@."
          (Tensor.read t (Vec6.coord ~n:0 ~t:0 ~d:0 ~h:0 ~w:0 ~c:0))
          (Tensor.read t (Vec6.coord ~n:0 ~t:0 ~d:0 ~h:0 ~w:1 ~c:0))
    | Error e -> Fmt.pr "  %a@." Ssa_lower.Ssa_exec.pp_error e
  in
  let both =
    Loop_programs.chain ~outputs:[ Loop_fixtures.tid 1; Loop_fixtures.tid 2 ]
  in
  check "stored" (Fusion_plan.default both);
  (* t1 is virtual for t2: the elaborator inlines it with its own rounding *)
  let only_t2 = Loop_programs.chain ~outputs:[ Loop_fixtures.tid 2 ] in
  let fused, _ = Fusion_plan.plan only_t2 in
  check "fused" fused;
  [%expect
    {|
    stored: agree
      t2 = 1.67772e+07 3
    fused: agree
      t2 = 1.67772e+07 3
    |}]

(* Index failures are asserted against the row the language reports, not the
   reference: the reference's checked domain is its host int, wider natively. *)
let index_failure kernel =
  let plan = Fusion_plan.default kernel in
  let program =
    Err.or_raise ~pp_error:Ssa_lower.Ssa_lower_plan.pp_error
      (Ssa_lower.Ssa_lower_plan.lower plan)
  in
  match
    Err.payload
      (Ssa_lower.Ssa_exec.run plan program
         ~bind:(bind_data ~shape:(Loop_fixtures.shape_w 4) [| 0.; 0.; 0.; 0. |]))
  with
  | Ok _ -> Fmt.pr "no failure@."
  | Error e -> Fmt.pr "%a@." Ssa_lower.Ssa_exec.pp_error e

let w = Expr.Index.output Expr.Axis.W
let position i = Expr.Index.assume_position i
let max_index = Expr.Index.const 0x7FFF_FFFF

let%expect_test "index operations fail where the source evaluates them" =
  (* w + (2^31 - 1) overflows first at w = 1, with the operands in source order *)
  index_failure
    (Loop_fixtures.pixel_kernel
       (Expr.Value.value_of_index
          (Expr.Index.add (Expr.Index.of_position w) max_index)));
  (* two components both overflow: the first in axis order is the one reported *)
  let at axis i = Expr.Coord.set Loop_programs.out_coord axis i in
  let wide k = position (Expr.Index.add max_index (Expr.Index.const k)) in
  index_failure
    (Loop_fixtures.pixel_kernel
       (Loop_programs.ld
          (Expr.Coord.set (at Expr.Axis.H (wide 1)) Expr.Axis.W (wide 2))));
  [%expect
    {|
    index add overflows on 1 and 2147483647
    index add overflows on 2147483647 and 1
    |}]

let%expect_test
    "the nest visits cells with C innermost, so the first failing cell is \
     stable" =
  (* t1[h,w] = t0[h+1, w+1] on 2x2: (0,0) is fine, (0,1) fails on W first; a nest
     with W outermost would reach (1,0) and report H *)
  let shifted axis =
    position
      (Expr.Index.add
         (Expr.Index.of_position (Expr.Index.output axis))
         (Expr.Index.const 1))
  in
  let coord =
    Expr.Coord.set
      (Expr.Coord.set Loop_programs.out_coord Expr.Axis.H (shifted Expr.Axis.H))
      Expr.Axis.W (shifted Expr.Axis.W)
  in
  let shape = hw 2 2 in
  let kernel = Loop_fixtures.pixel_kernel ~shape (Loop_programs.ld coord) in
  let plan = Fusion_plan.default kernel in
  let bind = bind_data ~shape [| 1.; 2.; 3.; 4. |] in
  Fmt.pr "%a@." Ssa_check.pp_verdict (Ssa_check.run plan ~bind);
  [%expect {| agree on failure: coord_out_of_range |}]
