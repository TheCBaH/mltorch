(* Split out of what was verify_test.ml see verify_fixtures.ml. *)

open Graph_ir
open Verify_fixtures

(* ---- the coefficient tier -------------------------------------------------

   Batch-norm folding re-associates: [(Σ xₖ·Wₖ)·s] becomes [Σ xₖ·(Wₖ·s)]. No
   structural comparison reaches that, and no exact one does either — the pass
   re-derives its constants numerically, and eps arrives as a constant EDGE with
   a payload against a source-side [Const]. So the honest verdict is agreement
   of the polynomial coefficients within a tolerance, which is evidence and
   never a proof.

   The eight-way check lives in fold_batch_norm_test.ml, next to the numeric one
   it sits alongside. What is here is the negative: a fold that is actually
   wrong must NOT come back agreeing, and — because the claim is [Equivalent],
   not [Identical] — must not come back refuted either. *)

let%expect_test "coefficients: a wrong fold disagrees, and is not refuted" =
  let s1c = Graph_fixtures.s1c in
  let weight_shape =
    Graph_fixtures.weight_shape ~out_channels:3 ~in_channels:2
  in
  let vec a =
    Tensor.materialize (s1c 3) (fun c -> a.(Dim.to_int (Vec6.get c Axis.C)))
  in
  let g =
    build "conv_bn"
      Graph_builder.(
        let* x = input ~shape:(Graph_fixtures.nhwc ~h:4 ~w:4 ~c:2) () in
        let* w = constant ~shape:weight_shape () in
        let* mean = constant ~shape:(s1c 3) () in
        let* var = constant ~shape:(s1c 3) () in
        let* y =
          conv2d (Graph_fixtures.conv_params ~in_channels:2) ~x ~weight:w ()
        in
        batch_norm Graph_fixtures.bn_params ~x:y ~running_mean:mean
          ~running_var:var ())
  in
  let ids =
    List.filter
      (fun id -> Graph_ir.input_kind g id = Input.Constant)
      g.Graph.inputs
  in
  let constants =
    List.combine ids
      [ hw_ramp weight_shape; vec [| 0.5; 1.; 1.5 |]; vec [| 4.; 1.; 0.25 |] ]
  in
  (* Scaling every folded payload by two changes the coefficients, not just
     their last bits, so tolerance cannot absorb it.

     Only the payloads the FOLD produced — the destination ids the source does
     not have. Scaling the whole destination map would also corrupt the three
     constants both graphs share, and those are obligations in their own right:
     they would come back refuted, correctly and for an unrelated reason, which
     is not what this test is about. *)
  let doubled (Tensor.Tensor t as packed) =
    Tensor.materialize t.Tensor.shape (fun c -> Tensor.read packed c *. 2.)
  in
  let folded_only ~src dst =
    Tensor_id.Map.mapi
      (fun id payload ->
        if Tensor_id.Map.mem id src then payload else doubled payload)
      dst
  in
  let report ~dst_constants name =
    let result =
      let open Err.Syntax in
      let* (Rewrite.Origin state) = lift_origin (Rewrite.origin ~constants g) in
      let* (Rewrite.Step (final, map)) =
        lift_pass (Pass.run_all state [ Fold_batch_norm.pass ])
      in
      lift_verify
        (Map_verify.run ~budget:Map_verify.Budget.cumulative map
           ~src:(Rewrite.snapshot state)
           ~src_constants:(Rewrite.constants state)
           ~src_constant_store:(Rewrite.constant_store state)
           ~dst:(Rewrite.snapshot final)
           ~dst_constants:
             (dst_constants ~src:(Rewrite.constants state)
                (Rewrite.constants final))
           ~dst_constant_store:(Rewrite.constant_store final))
    in
    Format.printf "%s: %a@." name
      (pp_result (fun ppf r ->
           Fmt.list ~sep:(Fmt.any "; ")
             (fun ppf (e : Map_verify.Entry.t) ->
               Map_verify.Verdict.pp ppf e.outcome.verdict)
             ppf
             (List.filter
                (fun (e : Map_verify.Entry.t) ->
                  match e.outcome.verdict with
                  | Map_verify.Verdict.Vacuous -> false
                  | _ -> true)
                r.Map_verify.Report.entries)))
      result
  in
  report ~dst_constants:(fun ~src:_ dst -> dst) "honest fold";
  report ~dst_constants:folded_only "folded payloads doubled";
  [%expect
    {|
    honest fold: tested: agrees (1e-05); proved (structural); proved (structural, for these constants); proved (structural, for these constants); proved (structural, for these constants)
    folded payloads doubled: tested: agrees (1e-05); proved (structural); proved (structural, for these constants); proved (structural, for these constants); proved (structural, for these constants) |}]
