(* Split out of what was verify_test.ml see verify_fixtures.ml. *)

open Graph_ir
open Verify_fixtures

(* ---- what a relation claims, and what may be run against it ----------------

   [Unverifiable] is the bottom of the lattice: the two edges correspond, and
   nothing may be asserted about their values. Two consequences, and they pull in
   opposite directions, which is why each has its own test.

   Structural equality is still MEANINGFUL there. It does not assert a value
   relation, it observes that the two sides compute the same term — useful
   exactly where an upstream rewrite destroyed the value relation and a
   downstream transfer function was left alone.

   Everything below the structural tier is NOT. Coefficient agreement and the
   probe are evidence about values, and a relation that claims nothing has no
   value claim to gather evidence for or against. *)

let%expect_test "relation: an unverifiable cluster may still prove structurally"
    =
  let module A = (val Version_fixture.of_graph (relu_of Graph_builder.add ()))
  in
  let module B = (val Version_fixture.of_graph (relu_of Graph_builder.add ()))
  in
  let src id = Option.get (Snapshot.edge A.snapshot (Tensor_id.of_int id)) in
  let dst id = Option.get (Snapshot.edge B.snapshot (Tensor_id.of_int id)) in
  (* t3 has to be claimed too: it is downstream of t2, and [check_claim_closure]
     rejects a map that leaves it implicitly identical. *)
  verify_map
    (hand_map ~src:A.snapshot ~dst:B.snapshot
       [
         Correspondence.pair (src 2) (dst 2) Correspondence.Unverifiable;
         Correspondence.pair (src 3) (dst 3) Correspondence.Unverifiable;
       ]
       [])
    ~src:A.snapshot ~dst:B.snapshot;
  [%expect
    {|
    {t2} -> {t2} unverifiable: proved (structural) [exhaustive]
    {t3} -> {t3} unverifiable: proved (structural) [exhaustive]
    {t0} -> {t0} identical: proved (structural) [exhaustive]
    {t1} -> {t1} identical: proved (structural) [exhaustive] |}]

(* The mutation this kills: dropping [check_cluster]'s early return WITHOUT
   giving [settle] an [Unverifiable] terminal branch. The comparison then falls
   through to [Coeff_form.agree] and the probe, and since the label is not
   [Identical] the cluster comes back [tested: disagrees at ...] — a numerical
   verdict about a relation that asserts nothing. *)
let%expect_test "relation: an unverifiable cluster is never numerically tested"
    =
  let module A = (val Version_fixture.of_graph (relu_of Graph_builder.add ()))
  in
  let module B = (val Version_fixture.of_graph (relu_of Graph_builder.sub ()))
  in
  let src id = Option.get (Snapshot.edge A.snapshot (Tensor_id.of_int id)) in
  let dst id = Option.get (Snapshot.edge B.snapshot (Tensor_id.of_int id)) in
  verify_map
    (hand_map ~src:A.snapshot ~dst:B.snapshot
       [
         Correspondence.pair (src 2) (dst 2) Correspondence.Unverifiable;
         Correspondence.pair (src 3) (dst 3) Correspondence.Unverifiable;
       ]
       [])
    ~src:A.snapshot ~dst:B.snapshot;
  [%expect
    {|
    {t2} -> {t2} unverifiable: unproved: unsupported relation: unverifiable [exhaustive]
    {t3} -> {t3} unverifiable: proved (structural) [exhaustive]
    {t0} -> {t0} identical: proved (structural) [exhaustive]
    {t1} -> {t1} identical: proved (structural) [exhaustive] |}]
