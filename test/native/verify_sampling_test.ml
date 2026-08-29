(* Split out of what was verify_test.ml see verify_fixtures.ml. *)

open Verify_fixtures

(* ---- sampling -------------------------------------------------------------

   Coverage is carried BESIDE the verdict, not folded into it, so a sampled
   proof is visibly partial whatever the verdict is. [Report.proved] demands
   [Exhaustive], while [Report.refuted] ignores coverage entirely — a
   counterexample found at a sampled coordinate is still a counterexample. *)

let%expect_test "sampling: a sampled proof does not satisfy Report.proved" =
  let sampling = { Map_verify.Budget.default with sample = Some 4 } in
  let result =
    let open Err.Syntax in
    let* (Rewrite.Origin state) =
      lift_origin (Rewrite.origin (Graph_fixtures.permute_sequence ()))
    in
    let* step = lift_pass (Pass.run_all state [ Trim_permute.pass ]) in
    let* sampled = lift_verify (Map_verify.step ~budget:sampling state step) in
    let+ full = lift_verify (Map_verify.step state step) in
    Printf.sprintf "sampled: %s / proved=%b\nexhaustive: %s / proved=%b"
      (Map_verify.Report.summary sampled)
      (Map_verify.Report.proved sampled)
      (Map_verify.Report.summary full)
      (Map_verify.Report.proved full)
  in
  Format.printf "%a@." (pp_result Fmt.string) result;
  [%expect
    {|
    sampled: 3 clusters: 2 proved (structural) [sampled 4], 1 vacuous / proved=false
    exhaustive: 3 clusters: 2 proved (structural), 1 vacuous / proved=true |}]
