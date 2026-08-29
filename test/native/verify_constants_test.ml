(* Split out of what was verify_test.ml see verify_fixtures.ml. *)

open Graph_ir
open Verify_fixtures

(* ---- constant payloads ----------------------------------------------------

   Binding the model's constants narrows what a proof quantifies over — every
   INPUT, for these constants, rather than every payload — so it is only
   attempted when the unqualified comparison fails. The permute passes above
   stay at plain [structural] because they never need it; [fold_const] cannot
   be proved without it, since the destination edge IS a payload the pass
   computed. *)

let w_shape = Graph_fixtures.s 3 1 1 2 2 2

let%expect_test "verify: fold_const needs the constants, and gets them" =
  check_with
    ~constants:[ (Tensor_id.of_int 1, hw_ramp w_shape) ]
    "const_permute [fold_const]"
    (Graph_fixtures.const_permute ())
    [ Fold_const.pass ];
  [%expect
    {|
    const_permute [fold_const]:
      {t1} -> {} identical: vacuous
      {t0} -> {t0} identical: proved (structural) [exhaustive]
      {t2} -> {t2} identical: proved (structural, for these constants) [exhaustive]
      {t3} -> {t3} identical: proved (structural) [exhaustive] |}]

(* Folding is only correct if the pass reproduced the source's arithmetic
   exactly, materialization included. Perturbing the DESTINATION payload is how
   to simulate a fold that computed the wrong number — perturbing the source
   constant instead would just be folded faithfully and prove.

   The refuted terms here are closed (both sides are constants), so the witness
   is the empty valuation: two closed terms that differ need no assignment to
   separate them. *)
let%expect_test "verify: a fold that computed the wrong payload is refuted" =
  let bump (Tensor.Tensor t as packed) =
    Tensor.materialize t.Tensor.shape (fun c -> Tensor.read packed c +. 1.)
  in
  let result =
    let open Err.Syntax in
    let* (Rewrite.Origin state) =
      lift_origin
        (Rewrite.origin
           ~constants:[ (Tensor_id.of_int 1, hw_ramp w_shape) ]
           (Graph_fixtures.const_permute ()))
    in
    let* (Rewrite.Step (final, map)) =
      lift_pass (Pass.run_all state [ Fold_const.pass ])
    in
    lift_verify
      (Map_verify.run map ~src:(Rewrite.snapshot state)
         ~src_constants:(Rewrite.constants state) ~dst:(Rewrite.snapshot final)
         ~dst_constants:(Tensor_id.Map.map bump (Rewrite.constants final)))
  in
  Format.printf "@[<v 2>const_permute, folded payload off by one:@,%a@]@."
    (pp_result Map_verify.Report.pp_verdicts)
    result;
  [%expect
    {|
    const_permute, folded payload off by one:
      {t1} -> {} identical: vacuous
      {t0} -> {t0} identical: proved (structural) [exhaustive]
      {t2} -> {t2} identical: refuted: value at (0): src.t2 vs dst.t2 under {} [exhaustive]
      {t3} -> {t3} identical: proved (structural) [exhaustive] |}]
