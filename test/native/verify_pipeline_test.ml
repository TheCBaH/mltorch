(* Split out of what was verify_test.ml see verify_fixtures.ml. *)

open Graph_ir
open Verify_fixtures

(* ---- the pipeline hook ----------------------------------------------------

   [Pass.run_all ~verify] checks each step as it is applied, so the first
   offending pass stops the pipeline and the error names it. These live here
   rather than in pass_test.ml because what is under test is the verifier's
   effect on the driver, and the fixtures are already to hand. *)

(* Deliberately wrong: trims EVERY permute, tying its output to its input, when
   only an identity permute may be trimmed that way. The claim it leaves behind
   is [Identical] between two edges that differ. *)
let trim_any_permute =
  Pass.per_node ~name:"trim_any_permute"
    {
      Pass.on_node =
        (fun _env (n : node) ->
          match (n.Node.op, n.Node.outputs) with
          | Permute { x; _ }, [ out ] ->
              Some
                (let open Recipe in
                 let* out = existing out in
                 let* x = existing x in
                 trim ~remove:[ n.Node.id ] ~tie:[ (out, x) ])
          | _ -> None);
    }

let piped ?verify g passes =
  let open Err.Syntax in
  let* (Rewrite.Origin state) = lift_origin (Rewrite.origin g) in
  let+ (Rewrite.Step (final, _)) =
    lift_pass (Pass.run_all ?verify state passes)
  in
  Format.asprintf "%d nodes" (List.length (Rewrite.graph final).Graph.nodes)

let%expect_test "hook: a broken pass is caught, and named" =
  let g () =
    build "swap"
      Graph_builder.(
        let* a = input ~shape:s ~name:"a" () in
        let* t1 = permute Graph_fixtures.swap_hw a in
        relu t1)
  in
  (* unverified, the bad rewrite sails through *)
  Format.printf "no policy: %a@." (pp_result Fmt.string)
    (piped (g ()) [ trim_any_permute ]);
  Format.printf "%a@." (pp_result Fmt.string)
    (piped ~verify:Map_verify.Policy.Require_proved (g ()) [ trim_any_permute ]);
  [%expect
    {|
    no policy: 1 nodes
    pass trim_any_permute rejected: 2 clusters: 1 proved (structural), 1 refuted (counterexample)
      {t0, t1} -> {t0} identical: refuted: value at (1,0): src.t0 vs src.t1 under {v0(1,0)=0x1p+0, v0(1,0,0)=0x1p+1} [exhaustive]
      {t2} -> {t2} identical: proved (structural) [exhaustive] |}]

(* The two policies exist because [Unproved] and [Refuted] are different
   answers. The trim below is genuinely unproven — an i32 cell upstream blocks
   the collapse — but the verifier has exhibited no counterexample, so the
   release bar tolerates it while the development bar does not.

   The budget is what makes it unproven here, rather than a non-f32 cell. Tying
   an i32 input to a permute's f32 output is a contradiction [Graph_map.create]
   now rejects outright, and a rejected map produces no report at all for a
   policy to judge. *)
let%expect_test
    "hook: Reject_refuted tolerates unproved, Require_proved does not" =
  let g () =
    build "wide_permute"
      Graph_builder.(
        let* a =
          input ~shape:(Graph_fixtures.nhwc ~h:64 ~w:64 ~c:2) ~name:"a" ()
        in
        let* t1 = permute Graph_fixtures.identity_perm a in
        relu t1)
  in
  Format.printf "reject_refuted: %a@." (pp_result Fmt.string)
    (piped ~verify:Map_verify.Policy.Reject_refuted (g ()) [ Trim_permute.pass ]);
  Format.printf "require_proved: %a@." (pp_result Fmt.string)
    (piped ~verify:Map_verify.Policy.Require_proved (g ()) [ Trim_permute.pass ]);
  [%expect
    {|
    reject_refuted: 1 nodes
    require_proved: pass trim_permute rejected: 2 clusters: 2 unproved (too large)
                      {t0, t1} -> {t0} identical: unproved: too large (8192 coords)
                      {t2} -> {t2} identical: unproved: too large (8192 coords) |}]
