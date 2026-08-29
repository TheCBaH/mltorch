(* Split out of what was verify_test.ml see verify_fixtures.ml. *)

open Graph_ir
open Verify_fixtures

(* ---- cumulative verification ----------------------------------------------

   [Pass.run_all] already threads [Graph_map.compose], so everything above
   verifies a COMPOSED map whenever it is handed more than one pass. What is
   added here is the other half: verifying each step on its own, so a failure
   names the pass that caused it, and comparing the two.

   The per-step chain is already a proof of the end-to-end claim PROVIDED
   composition is sound; the composed check is what tests that proviso. It
   catches a composition error that makes a composed claim false at the
   endpoints — it does not validate compose's algebraic contract in general,
   since an over-conservative label is legal and therefore unverifiable, and a
   cluster set that is wrong but endpoint-consistent still passes. That stays
   graph_map_test.ml's job. *)

let cumulative = Map_verify.Budget.cumulative

(* Apply passes one at a time, verifying each step against the state it started
   from, so a failure names the pass that caused it. *)
let per_step ?constants g passes =
  let open Err.Syntax in
  let* (Rewrite.Origin origin) = lift_origin (Rewrite.origin ?constants g) in
  let rec go : type v.
      v Rewrite.t ->
      Pass.t list ->
      ((string * Map_verify.Report.t) list, error) Err.t =
   fun state -> function
     | [] -> Err.return []
     | p :: rest ->
         let* (Rewrite.Step (next, _) as step) =
           lift_pass (Pass.run_all state [ p ])
         in
         let* report =
           lift_verify (Map_verify.step ~budget:cumulative state step)
         in
         let+ rest = go next rest in
         (p.Pass.name, report) :: rest
  in
  go origin passes

let composed ?constants g passes =
  let open Err.Syntax in
  let* (Rewrite.Origin origin) = lift_origin (Rewrite.origin ?constants g) in
  let* step = lift_pass (Pass.run_all origin passes) in
  lift_verify (Map_verify.step ~budget:cumulative origin step)

(* The law worth pinning: if every step verifies, composition must not turn that
   into a refutation. The converse is NOT a law — a composed [Unproved] where
   every step is [Proved] is an acceptable outcome, since the composed frontier
   spans the whole pipeline and can run out of budget where a single step does
   not. Verification strength is also not monotone under composition: two steps
   whose roundings cancel can compose to a bit-identical pair. *)
let both name ?constants g passes =
  let report =
    let open Err.Syntax in
    let* steps = per_step ?constants g passes in
    let+ composed = composed ?constants g passes in
    let lines =
      List.map
        (fun (n, r) -> Printf.sprintf "%s: %s" n (Map_verify.Report.summary r))
        steps
      @ [ Printf.sprintf "composed: %s" (Map_verify.Report.summary composed) ]
    in
    let every_step_proved =
      List.for_all (fun (_, r) -> Map_verify.Report.proved r) steps
    in
    lines
    @ [
        Printf.sprintf "law (every step proved => composed not refuted): %b"
          ((not every_step_proved) || not (Map_verify.Report.refuted composed));
      ]
  in
  Format.printf "@[<v 2>%s:@,%a@]@." name
    (pp_result (Fmt.list ~sep:Fmt.cut Fmt.string))
    report

let%expect_test "verify: each step and their composition agree" =
  both "permute_identity_chain"
    (Graph_fixtures.permute_identity_chain ())
    [ Chain_permute.pass; Trim_permute.pass ];
  [%expect
    {|
    permute_identity_chain:
      chain_permute: 5 clusters: 4 proved (structural), 1 vacuous
      trim_permute: 3 clusters: 3 proved (structural)
      composed: 4 clusters: 3 proved (structural), 1 vacuous
      law (every step proved => composed not refuted): true |}]

(* Cross-iteration composition: a fixpoint fold collapses a multi-node constant
   sub-DAG one node at a time, so the composed map is a chain of per-iteration
   maps rather than a single step's. *)
let%expect_test "verify: a fixpoint over a constant sub-DAG" =
  let shape = Graph_fixtures.s1c 3 in
  let ramp base =
    Tensor.materialize shape (fun c ->
        base +. float_of_int (Dim.to_int (Vec6.get c Axis.C)))
  in
  let constants =
    [
      (Tensor_id.of_int 1, ramp 1.);
      (Tensor_id.of_int 2, ramp 10.);
      (Tensor_id.of_int 3, ramp 100.);
    ]
  in
  both "const_arith [fixpoint fold_const]" ~constants
    (Graph_fixtures.const_arith ())
    [ Pass.fixpoint Fold_const.pass ];
  [%expect
    {|
    const_arith [fixpoint fold_const]:
      fold_const: 7 clusters: 1 proved (for these constants), 2 proved (structural), 4 vacuous
      composed: 7 clusters: 1 proved (for these constants), 2 proved (structural), 4 vacuous
      law (every step proved => composed not refuted): true |}]

(* Terminal id packing renumbers post-origin ids, including graph inputs, and
   composing its map is the {t11} -> {} then {t12} -> {t11} hazard §9 of
   native_transform_design.md warns about: a resurrected dead id would fuse two
   clusters and claim the dead edge corresponds to the packed one. Verifying
   origin -> passes -> pack end to end is the check that a resurrection would
   actually be caught, rather than just producing a plausible-looking map. *)
let%expect_test "verify: origin -> passes -> pack, composed" =
  let result =
    let open Err.Syntax in
    let* (Rewrite.Origin s0) =
      lift_origin (Rewrite.origin (Graph_fixtures.permute_identity_chain ()))
    in
    let* (Rewrite.Step (s1, m01)) =
      lift_pass (Pass.run_all s0 [ Chain_permute.pass; Trim_permute.pass ])
    in
    let* (Rewrite.Step (s2, m12)) = lift_origin (Rewrite.pack s1) in
    let+ report =
      lift_verify
        (Map_verify.run ~budget:Map_verify.Budget.cumulative
           (Graph_map.compose m01 m12)
           ~src:(Rewrite.snapshot s0) ~dst:(Rewrite.snapshot s2))
    in
    Map_verify.Report.summary report
  in
  Format.printf "permute_identity_chain, passes then pack: %a@."
    (pp_result Fmt.string) result;
  [%expect
    {| permute_identity_chain, passes then pack: 4 clusters: 3 proved (structural), 1 vacuous |}]
