(* Eliminating redundant permutes: trimming a run that composes to the
   identity, fusing adjacent permutes into one, and the combination of both
   under a fixpoint. Split from permute_passes_test.ml. *)

open Permute_pass_fixtures

(* ---- trimming a cancelling run ------------------------------------------- *)

let%expect_test "trim_permute: a single identity permute" =
  run (Graph_fixtures.permute_noop ()) [ Trim_permute.pass ];
  [%expect
    {|
    before:
      graph
      inputs: [t0 f32 [H=2 W=3 C=4] ->[n0]]
      nodes:
        n0: [t1 f32 [H=2 W=3 C=4] ->[n1]] = permute x=t0 perm=[]
        n1: [t2 f32 [H=2 W=3 C=4]] = relu x=t1 <-n0
      outputs: [t2 f32 [H=2 W=3 C=4] <-n1]
    after:
      graph
      inputs: [t0 f32 [H=2 W=3 C=4] ->[n1]]
      nodes:
        n1: [t2 f32 [H=2 W=3 C=4]] = relu x=t0
      outputs: [t2 f32 [H=2 W=3 C=4] <-n1]
    map:
      values:
        {t0, t1} -> {t0} identical
      nodes:
        {n0} -> {}
      provenance:
        none |}]

let%expect_test "trim_permute: a pair composing to the identity" =
  (* Neither node is a no-op alone, and the intermediate edge is NOT equal to
     the input — so it is deleted rather than tied into the cluster. Getting
     that wrong would tell a checker that t1 and t0 hold the same values. *)
  run (Graph_fixtures.permute_sequence ()) [ Trim_permute.pass ];
  [%expect
    {|
    before:
      graph
      inputs: [t0 f32 [H=2 W=3 C=4] ->[n0]]
      nodes:
        n0: [t1 f32 [H=3 W=4 C=2] ->[n1]] = permute x=t0 perm=[H<-W, W<-C, C<-H]
        n1: [t2 f32 [H=2 W=3 C=4] ->[n2]] =
          permute x=t1 <-n0 perm=[H<-C, W<-H, C<-W]
        n2: [t3 f32 [H=2 W=3 C=4]] = relu x=t2 <-n1
      outputs: [t3 f32 [H=2 W=3 C=4] <-n2]
    after:
      graph
      inputs: [t0 f32 [H=2 W=3 C=4] ->[n2]]
      nodes:
        n2: [t3 f32 [H=2 W=3 C=4]] = relu x=t0
      outputs: [t3 f32 [H=2 W=3 C=4] <-n2]
    map:
      values:
        {t0, t2} -> {t0} identical
        {t1} -> {} identical
      nodes:
        {n0} -> {}
        {n1} -> {}
      provenance:
        none |}]

let%expect_test "trim_permute: what a partially cancelling chain matches" =
  (* Only the first two of the three permutes cancel, and the match is found by
     anchoring on the middle edge — the run ending at the last edge composes to
     [swap_hw] and is rejected. *)
  matches Trim_permute.pattern (Graph_fixtures.permute_partial_cancel ());
  [%expect
    {|
    matched nodes: [n0, n1]
            inputs: [t0]
            outputs: [t2]
            interior: [t1]
            convex: true:
      graph
      inputs: [t0 f32 [H=2 W=3 C=4] ->[n0]]
      nodes:
        n0: [t1 f32 [H=3 W=4 C=2] ->[n1]] = permute x=t0 perm=[H<-W, W<-C, C<-H]
        n1: [t2 f32 [H=2 W=3 C=4]] = permute x=t1 <-n0 perm=[H<-C, W<-H, C<-W]
      outputs: [t2 f32 [H=2 W=3 C=4] <-n1] |}]

let%expect_test "trim_permute: the run cancelling is looked for at every edge" =
  run (Graph_fixtures.permute_partial_cancel ()) [ Trim_permute.pass ];
  [%expect
    {|
    before:
      graph
      inputs: [t0 f32 [H=2 W=3 C=4] ->[n0]]
      nodes:
        n0: [t1 f32 [H=3 W=4 C=2] ->[n1]] = permute x=t0 perm=[H<-W, W<-C, C<-H]
        n1: [t2 f32 [H=2 W=3 C=4] ->[n2]] =
          permute x=t1 <-n0 perm=[H<-C, W<-H, C<-W]
        n2: [t3 f32 [H=3 W=2 C=4] ->[n3]] = permute x=t2 <-n1 perm=[H<-W, W<-H]
        n3: [t4 f32 [H=3 W=2 C=4]] = relu x=t3 <-n2
      outputs: [t4 f32 [H=3 W=2 C=4] <-n3]
    after:
      graph
      inputs: [t0 f32 [H=2 W=3 C=4] ->[n2]]
      nodes:
        n2: [t3 f32 [H=3 W=2 C=4] ->[n3]] = permute x=t0 perm=[H<-W, W<-H]
        n3: [t4 f32 [H=3 W=2 C=4]] = relu x=t3 <-n2
      outputs: [t4 f32 [H=3 W=2 C=4] <-n3]
    map:
      values:
        {t0, t2} -> {t0} identical
        {t1} -> {} identical
      nodes:
        {n0} -> {}
        {n1} -> {}
      provenance:
        none |}]

let%expect_test "trim_permute: a shared intermediate edge is not trimmable" =
  (* The run still cancels, but t1 has a second consumer; removing n0 would
     leave it dangling. The chain stops at t1, and the one-node run that
     remains does not cancel. *)
  matches Trim_permute.pattern (Graph_fixtures.permute_shared ());
  [%expect {| no match |}]

let%expect_test "trim_permute: a run that composes to something else" =
  matches Trim_permute.pattern (Graph_fixtures.permute_pair ());
  [%expect {| no match |}]

let%expect_test "trim_permute: one sweep takes only the shortest run" =
  (* Matches are greedy and disjoint: anchoring on t1 finds the one-node run
     first, and the longer runs anchored on t2 and t3 overlap it and are
     dropped. A fixpoint (next test) unwinds the rest. *)
  run ~show_before:false
    (Graph_fixtures.permute_identity_chain ())
    [ Trim_permute.pass ];
  [%expect
    {|
    after:
      graph
      inputs: [t0 f32 [H=2 W=3 C=4] ->[n1]]
      nodes:
        n1: [t2 f32 [H=2 W=3 C=4] ->[n2]] = permute x=t0 perm=[]
        n2: [t3 f32 [H=2 W=3 C=4] ->[n3]] = permute x=t2 <-n1 perm=[]
        n3: [t4 f32 [H=2 W=3 C=4]] = relu x=t3 <-n2
      outputs: [t4 f32 [H=2 W=3 C=4] <-n3]
    map:
      values:
        {t0, t1} -> {t0} identical
      nodes:
        {n0} -> {}
      provenance:
        none |}]

let%expect_test "trim_permute: under a fixpoint the whole chain goes" =
  run ~show_before:false
    (Graph_fixtures.permute_identity_chain ())
    [ Pass.fixpoint Trim_permute.pass ];
  [%expect
    {|
    after:
      graph
      inputs: [t0 f32 [H=2 W=3 C=4] ->[n3]]
      nodes:
        n3: [t4 f32 [H=2 W=3 C=4]] = relu x=t0
      outputs: [t4 f32 [H=2 W=3 C=4] <-n3]
    map:
      values:
        {t0, t1, t2, t3} -> {t0} identical
      nodes:
        {n0} -> {}
        {n1} -> {}
        {n2} -> {}
      provenance:
        none |}]

(* ---- fusing adjacent permutes -------------------------------------------- *)

let%expect_test "chain_permute: two permutes become one" =
  (* The fused node takes over t2: same tensor, same shape, so keeping the id is
     legal — but its definition changed, which the recipe has to claim. The
     intermediate t1 is deleted. *)
  run (Graph_fixtures.permute_pair ()) [ Chain_permute.pass ];
  [%expect
    {|
    before:
      graph
      inputs: [t0 f32 [H=2 W=3 C=4] ->[n0]]
      nodes:
        n0: [t1 f32 [H=3 W=2 C=4] ->[n1]] = permute x=t0 perm=[H<-W, W<-H]
        n1: [t2 f32 [H=3 W=4 C=2] ->[n2]] = permute x=t1 <-n0 perm=[W<-C, C<-W]
        n2: [t3 f32 [H=3 W=4 C=2]] = relu x=t2 <-n1
      outputs: [t3 f32 [H=3 W=4 C=2] <-n2]
    after:
      graph
      inputs: [t0 f32 [H=2 W=3 C=4] ->[n3]]
      nodes:
        n3: [t2 f32 [H=3 W=4 C=2] ->[n2]] = permute x=t0 perm=[H<-W, W<-C, C<-H]
        n2: [t3 f32 [H=3 W=4 C=2]] = relu x=t2 <-n3
      outputs: [t3 f32 [H=3 W=4 C=2] <-n2]
    map:
      values:
        {t1} -> {} identical
      nodes:
        {n0, n1} -> {n3}
      provenance:
        none |}]

let%expect_test "chain_permute: a shared intermediate edge is not fusable" =
  matches Chain_permute.pattern (Graph_fixtures.permute_shared ());
  [%expect {| no match |}]

let%expect_test "chain_permute then trim_permute collapse a whole run" =
  (* Fusing normalises the run to one node; trimming then recognises it as the
     identity. Same destination graph as the fixpoint above, but a WEAKER
     mapping: this route deletes t1 and t2, where trimming put them in t0's
     cluster. Both are sound — a deletion asserts nothing — and the difference is
     honest rather than a defect. Fusing two permutes genuinely does not know
     that the intermediate equalled the input; in general it does not. So the
     precision of a mapping depends on the route taken, and a verifier gets
     fewer clusters to check here, never a false one. *)
  run ~show_before:false
    (Graph_fixtures.permute_identity_chain ())
    [ Pass.fixpoint Chain_permute.pass; Trim_permute.pass ];
  [%expect
    {|
    after:
      graph
      inputs: [t0 f32 [H=2 W=3 C=4] ->[n3]]
      nodes:
        n3: [t4 f32 [H=2 W=3 C=4]] = relu x=t0
      outputs: [t4 f32 [H=2 W=3 C=4] <-n3]
    map:
      values:
        {t0, t3} -> {t0} identical
        {t1} -> {} identical
        {t2} -> {} identical
      nodes:
        {n0, n1, n2} -> {}
      provenance:
        none |}]
