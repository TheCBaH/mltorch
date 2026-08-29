(* Split out of what was verify_test.ml see verify_fixtures.ml. *)

open Graph_ir
open Verify_fixtures

(* ---- constants are obligations, not the sigma hypothesis -------------------

   Sigma is "corresponding graph INPUTS are fed the same data". A model constant
   is a graph input structurally — it has no producer — but it is not user data,
   and assuming two constants equal because they share a cluster assumes the
   thing the payload comparison is there to establish. Every [Graph.inputs]
   member used to get the hypothesis, constants included. *)

let const_relu () =
  build "const_relu"
    Graph_builder.(
      let* w = constant ~shape:s ~name:"w" () in
      relu w)

let flat v = Tensor.materialize s (fun _ -> v)
let payloads id v = Tensor_id.Map.singleton (Tensor_id.of_int id) (flat v)

let run_with_payloads ~src ~src_constants ~dst ~dst_constants =
  Format.printf "%a@."
    (pp_result Map_verify.Report.pp_verdicts)
    (lift_verify
       (Map_verify.run (hand_map ~src ~dst [] []) ~src ~src_constants ~dst
          ~dst_constants))

(* The false proof this rule is named after. Same graph twice, same ids, an
   empty map — and two DIFFERENT payloads behind the constant. Under the sigma
   hypothesis both sides ground to one variable, the cluster proves structurally,
   and a pass that rewrote a payload in place would ship unnoticed. *)
let%expect_test "constants: a payload change under one id is not assumed equal"
    =
  let module A = (val Version_fixture.of_graph (const_relu ())) in
  let module B = (val Version_fixture.of_graph (const_relu ())) in
  run_with_payloads ~src:A.snapshot ~src_constants:(payloads 0 1.)
    ~dst:B.snapshot ~dst_constants:(payloads 0 2.);
  [%expect
    {|
    {t0} -> {t0} identical: refuted: value at (0): src.t0 vs dst.t0 under {} [exhaustive]
    {t1} -> {t1} identical: proved (structural) [exhaustive] |}]

(* Equal payloads still prove, or the rule would just break every model: the
   proof is [for these constants] rather than [structural], because it is a
   statement about the payloads this model carries and not about every payload.
   That is exactly what the two attempts in [compare_at] are for. *)
let%expect_test "constants: equal payloads prove, for those constants" =
  let module A = (val Version_fixture.of_graph (const_relu ())) in
  let module B = (val Version_fixture.of_graph (const_relu ())) in
  run_with_payloads ~src:A.snapshot ~src_constants:(payloads 0 1.)
    ~dst:B.snapshot ~dst_constants:(payloads 0 1.);
  [%expect
    {|
    {t0} -> {t0} identical: proved (structural, for these constants) [exhaustive]
    {t1} -> {t1} identical: proved (structural) [exhaustive] |}]

(* With no payloads at all there is nothing to compare, and the honest answer is
   UNPROVED. Not refuted: the two constants are distinct cells only because
   neither was supplied, so a probe assigning them different values would
   manufacture a difference rather than find one. The guard has to precede the
   coefficient tier as well as the probe, since coefficients over two unrelated
   variables disagree for the same non-reason. *)
let%expect_test "constants: with no payloads, a constant cluster is unproved" =
  let module A = (val Version_fixture.of_graph (const_relu ())) in
  let module B = (val Version_fixture.of_graph (const_relu ())) in
  run_with_payloads ~src:A.snapshot ~src_constants:Tensor_id.Map.empty
    ~dst:B.snapshot ~dst_constants:Tensor_id.Map.empty;
  [%expect
    {|
    {t0} -> {t0} identical: unproved: unbound constant: src.t0(0) [exhaustive]
    {t1} -> {t1} identical: proved (structural) [exhaustive] |}]

(* [input_kinds] is SPARSE: it keys graph inputs, need not cover them, and an
   absent entry means [Input] ([Graph_ir.input_kind]). Reading it as
   [find_opt = Some Input] therefore classifies an ordinary omitted input as a
   non-input, sigma stops applying, and two identical graphs refute at their own
   inputs.

   The mirror mistake — testing the kind without testing graph-input membership —
   hands every internal edge a variable, and "an identity map over a changed
   operator is unproved" above is what goes red for it. Not by proving t2, as it
   happens: [Env.var_edge] is built from the program's actual inputs, so nothing
   binds the variable an internal edge would be given and grounding fails with
   [unknown edge] instead. Two independent guards, and the membership test is
   the one that states the rule rather than tripping over its absence. *)
let%expect_test "sigma: an input absent from input_kinds is still a user input"
    =
  let strip (g : graph) = { g with Graph.input_kinds = Tensor_id.Map.empty } in
  let module A =
    (val Version_fixture.of_graph (strip (relu_of Graph_builder.add ())))
  in
  let module B =
    (val Version_fixture.of_graph (strip (relu_of Graph_builder.add ())))
  in
  verify_map
    (hand_map ~src:A.snapshot ~dst:B.snapshot [] [])
    ~src:A.snapshot ~dst:B.snapshot;
  [%expect
    {|
    {t0} -> {t0} identical: proved (structural) [exhaustive]
    {t1} -> {t1} identical: proved (structural) [exhaustive]
    {t2} -> {t2} identical: proved (structural) [exhaustive]
    {t3} -> {t3} identical: proved (structural) [exhaustive] |}]

(* A probe may only run once expansion has reached the graph inputs. Cells left
   at a truncated frontier are internal stage results constrained by their
   producers, so assigning them independently could manufacture a
   "counterexample" no input can realise. Starve the rounds and the verdict must
   be [max_rounds] — never a refutation. *)
let%expect_test "verify: a truncated frontier never refutes" =
  let starved = { Map_verify.Budget.default with max_rounds = 0 } in
  check_with ~budget:starved ~constants:[]
    "reuse_permute_sub_order, no expansion allowed"
    (Graph_fixtures.reuse_permute_sub_order ())
    [ Reuse_permute.pass ];
  [%expect
    {|
    reuse_permute_sub_order, no expansion allowed:
      {t3} -> {} identical: vacuous
      {} -> {t5} identical: vacuous
      {t0} -> {t0} identical: proved (structural) [exhaustive]
      {t1} -> {t1} identical: proved (structural) [exhaustive]
      {t2} -> {t2} identical: proved (structural) [exhaustive]
      {t4} -> {t4} identical: unproved: over max_rounds [exhaustive] |}]
