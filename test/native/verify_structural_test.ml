(* Split out of what was verify_test.ml see verify_fixtures.ml. *)

open Verify_fixtures

(* ---- the structural passes ------------------------------------------------

   Every one of these rearranges indices without touching arithmetic, so the
   ground terms come out identical and the proof holds for all payloads. *)

let%expect_test "verify: trim / chain permute" =
  check "permute_noop [trim]"
    (Graph_fixtures.permute_noop ())
    [ Trim_permute.pass ];
  check "permute_sequence [trim]"
    (Graph_fixtures.permute_sequence ())
    [ Trim_permute.pass ];
  check "permute_identity_chain [chain;trim]"
    (Graph_fixtures.permute_identity_chain ())
    [ Chain_permute.pass; Trim_permute.pass ];
  check "permute_pair [chain]"
    (Graph_fixtures.permute_pair ())
    [ Chain_permute.pass ];
  check "permute_partial_cancel [chain;trim]"
    (Graph_fixtures.permute_partial_cancel ())
    [ Chain_permute.pass; Trim_permute.pass ];
  check "permute_shared [chain]"
    (Graph_fixtures.permute_shared ())
    [ Chain_permute.pass ];
  [%expect
    {|
    permute_noop [trim]: 2 clusters: 2 proved (structural)
    permute_sequence [trim]: 3 clusters: 2 proved (structural), 1 vacuous
    permute_identity_chain [chain;trim]: 4 clusters: 3 proved (structural), 1 vacuous
    permute_pair [chain]: 4 clusters: 3 proved (structural), 1 vacuous
    permute_partial_cancel [chain;trim]: 4 clusters: 3 proved (structural), 1 vacuous
    permute_shared [chain]: 5 clusters: 5 proved (structural) |}]

let%expect_test "verify: bypass / sink permute" =
  check "sink_permute_unary [sink]"
    (Graph_fixtures.sink_permute_unary ())
    [ Sink_permute.pass ];
  check "sink_permute_binary [sink]"
    (Graph_fixtures.sink_permute_binary ())
    [ Sink_permute.pass ];
  check "sink_permute_broadcast [sink]"
    (Graph_fixtures.sink_permute_broadcast ())
    [ Sink_permute.pass ];
  check "sink_permute_mean_basic [sink_mean]"
    (Graph_fixtures.sink_permute_mean_basic ())
    [ Sink_permute_mean.pass ];
  [%expect
    {|
    sink_permute_unary [sink]: 5 clusters: 3 proved (structural), 2 vacuous
    sink_permute_binary [sink]: 8 clusters: 5 proved (structural), 3 vacuous
    sink_permute_broadcast [sink]: 8 clusters: 5 proved (structural), 3 vacuous
    sink_permute_mean_basic [sink_mean]: 4 clusters: 2 proved (structural), 2 vacuous |}]

(* [reuse_permute] is what forces a boundary to be crossed when only ONE side
   names it. [reuse_permute_basic] rewrites [t4 = add(P(t0), t1)] into
   [t4 = P(add(t0, Q(t1)))], reusing an existing [t3 = Q(t1)], so the output
   cluster's source side reads t1 where its destination side reads t3. Both are
   non-vacuous clusters, so a rule that stopped at every correspondence variable
   would leave the two sides as functions of DIFFERENT variables — unequal, and
   a probe assigning them independently would "separate" a correct rewrite,
   since [t3 = Q(t1)] is precisely the fact a local frontier drops. Crossing the
   one-sided variable recovers it. See .ai/native_transform_local_verify_plan.md
   §13.

   Sub and Div are the load-bearing ones — transposing a rebuilt op's operands
   would silently negate or invert rather than fail to type-check. *)
let%expect_test "verify: reuse_permute, including the non-commutative ops" =
  check "reuse_permute_basic"
    (Graph_fixtures.reuse_permute_basic ())
    [ Reuse_permute.pass ];
  check "reuse_permute_sub_order"
    (Graph_fixtures.reuse_permute_sub_order ())
    [ Reuse_permute.pass ];
  check "reuse_permute_div_order"
    (Graph_fixtures.reuse_permute_div_order ())
    [ Reuse_permute.pass ];
  check "reuse_permute_self_inverse"
    (Graph_fixtures.reuse_permute_self_inverse ())
    [ Reuse_permute.pass ];
  [%expect
    {|
    reuse_permute_basic: 6 clusters: 4 proved (structural), 2 vacuous
    reuse_permute_sub_order: 6 clusters: 4 proved (structural), 2 vacuous
    reuse_permute_div_order: 6 clusters: 4 proved (structural), 2 vacuous
    reuse_permute_self_inverse: 3 clusters: 3 proved (structural) |}]
