(* Reusing an existing alternate layout instead of inserting a fresh
   permute: unwrapping an operand, competing matches, wide fan-out, and
   non-commutative operand order (sub/div). Split from permute_passes_test.ml. *)

open Permute_pass_fixtures

(* ---- reusing an existing alternate layout -------------------------------- *)

let%expect_test "reuse_permute: unwrap one operand, reuse the other" =
  run (Graph_fixtures.reuse_permute_basic ()) [ Reuse_permute.pass ];
  [%expect
    {|
    before:
      graph
      inputs: [t0 f32 [H=2 W=3 C=4] ->[n0], t1 f32 [H=3 W=4 C=2] ->[n1, n3]]
      nodes:
        n0: [t2 f32 [H=3 W=4 C=2] ->[n3]] = permute x=t0 perm=[H<-W, W<-C, C<-H]
        n1: [t3 f32 [H=2 W=3 C=4] ->[n2]] = permute x=t1 perm=[H<-C, W<-H, C<-W]
        n2: [] = discard x=t3 <-n1
        n3: [t4 f32 [H=3 W=4 C=2]] = add a=t2 <-n0 b=t1
      outputs: [t4 f32 [H=3 W=4 C=2] <-n3]
    after:
      graph
      inputs: [t0 f32 [H=2 W=3 C=4] ->[n4], t1 f32 [H=3 W=4 C=2] ->[n1]]
      nodes:
        n1: [t3 f32 [H=2 W=3 C=4] ->[n2, n4]] =
          permute x=t1 perm=[H<-C, W<-H, C<-W]
        n4: [t5 f32 [H=2 W=3 C=4] ->[n5]] = add a=t0 b=t3 <-n1
        n2: [] = discard x=t3 <-n1
        n5: [t4 f32 [H=3 W=4 C=2]] = permute x=t5 <-n4 perm=[H<-W, W<-C, C<-H]
      outputs: [t4 f32 [H=3 W=4 C=2] <-n5]
    map:
      values:
        {t2} -> {} identical
        {} -> {t5} identical
      nodes:
        {n0} -> {n5}
        {n3} -> {n4}
      provenance:
        none |}]

let%expect_test "reuse_permute: two matches competing over one producer" =
  (* [Found in review, P1]: [add1] reuses [qb] without removing it, [add2]
     would unwrap (remove) that very [qb] — both are found in the SAME sweep,
     over the untouched graph, before either recipe has run. Claiming [qb] on
     the reuse side is what makes [Pattern.scan]'s ordinary greedy-disjoint
     rule drop the overlapping one instead of merging both recipes into a
     dangling reference. Only [add1] resolves; [add2] never does, even under
     a fixpoint, because after [add1]'s rewrite [qb]'s output gains a SECOND
     consumer (the rebuilt [add1] itself now reads it too) and is no longer
     interior — a real consequence, not a bug: [add2] correctly keeps
     declining rather than being unsafely forced through. *)
  run
    (Graph_fixtures.reuse_permute_competing_matches ())
    [ Pass.fixpoint Reuse_permute.pass ];
  [%expect
    {|
    before:
      graph
      inputs:
        [t0 f32 [H=2 W=3 C=4] ->[n0], t1 f32 [H=3 W=4 C=2] ->[n1, n4],
         t2 f32 [H=2 W=3 C=4] ->[n2, n5]]
      nodes:
        n0: [t3 f32 [H=3 W=4 C=2] ->[n4]] = permute x=t0 perm=[H<-W, W<-C, C<-H]
        n1: [t4 f32 [H=2 W=3 C=4] ->[n5]] = permute x=t1 perm=[H<-C, W<-H, C<-W]
        n2: [t5 f32 [H=3 W=4 C=2] ->[n3]] = permute x=t2 perm=[H<-W, W<-C, C<-H]
        n3: [] = discard x=t5 <-n2
        n4: [t6 f32 [H=3 W=4 C=2]] = add a=t3 <-n0 b=t1
        n5: [t7 f32 [H=2 W=3 C=4]] = add a=t4 <-n1 b=t2
      outputs: [t6 f32 [H=3 W=4 C=2] <-n4, t7 f32 [H=2 W=3 C=4] <-n5]
    after:
      graph
      inputs:
        [t0 f32 [H=2 W=3 C=4] ->[n6], t1 f32 [H=3 W=4 C=2] ->[n1],
         t2 f32 [H=2 W=3 C=4] ->[n2, n5]]
      nodes:
        n1: [t4 f32 [H=2 W=3 C=4] ->[n5, n6]] =
          permute x=t1 perm=[H<-C, W<-H, C<-W]
        n2: [t5 f32 [H=3 W=4 C=2] ->[n3]] = permute x=t2 perm=[H<-W, W<-C, C<-H]
        n6: [t8 f32 [H=2 W=3 C=4] ->[n7]] = add a=t0 b=t4 <-n1
        n3: [] = discard x=t5 <-n2
        n5: [t7 f32 [H=2 W=3 C=4]] = add a=t4 <-n1 b=t2
        n7: [t6 f32 [H=3 W=4 C=2]] = permute x=t8 <-n6 perm=[H<-W, W<-C, C<-H]
      outputs: [t6 f32 [H=3 W=4 C=2] <-n7, t7 f32 [H=2 W=3 C=4] <-n5]
    map:
      values:
        {t3} -> {} identical
        {} -> {t8} identical
      nodes:
        {n0} -> {n7}
        {n4} -> {n6}
      provenance:
        none |}]

let%expect_test "reuse_permute: wide fan-out reuse is not serialized" =
  (* [Found in review, P1]: 16 independent matches all reading (never
     removing) the same [Q(b)] used to be forced one-per-sweep by the
     exclusive [claim] every reuse took — needing 16 sweeps, which exactly
     exhausted [Pass.fixpoint]'s default 16-unit fuel with `Not_converged`
     on a graph that was never actually stuck. [claim_shared]'s read/read
     tolerance is what lets every one of them resolve in a SINGLE sweep, so
     a single (non-fixpoint) application of the pass is enough — and a
     second, quiet run confirms nothing more is left to do. *)
  let g = Graph_fixtures.reuse_permute_wide_fanout () in
  let before = List.length (Graph_ir.nodes g) in
  (match rewritten g [ Reuse_permute.pass ] with
  | None -> Format.printf "rewrite failed@."
  | Some g' ->
      Format.printf "nodes: %d -> %d@." before (List.length (Graph_ir.nodes g'));
      matches Reuse_permute.pattern g');
  [%expect {|
    nodes: 34 -> 34
    no match |}]

let%expect_test "reuse_permute: no alternate layout to reuse is declined" =
  matches Reuse_permute.pattern
    (Graph_fixtures.reuse_permute_missing_alternate ());
  [%expect {| no match |}]

let%expect_test "reuse_permute: an existing consumer that is not the inverse" =
  matches Reuse_permute.pattern
    (Graph_fixtures.reuse_permute_wrong_alternate ());
  [%expect {| no match |}]

let%expect_test
    "reuse_permute: an incompatible candidate does not hide a later one" =
  run
    (Graph_fixtures.reuse_permute_backtrack_candidate ())
    [ Reuse_permute.pass ];
  [%expect
    {|
    before:
      graph
      inputs: [t0 f32 [H=2 W=3 C=4] ->[n4], t1 f32 [H=3 W=4 C=2] ->[n0, n2, n5]]
      nodes:
        n0: [t2 f16 [H=2 W=3 C=4] ->[n1]] = permute x=t1 perm=[H<-C, W<-H, C<-W]
        n1: [] = discard x=t2 <-n0
        n2: [t3 f32 [H=2 W=3 C=4] ->[n3]] = permute x=t1 perm=[H<-C, W<-H, C<-W]
        n3: [] = discard x=t3 <-n2
        n4: [t4 f32 [H=3 W=4 C=2] ->[n5]] = permute x=t0 perm=[H<-W, W<-C, C<-H]
        n5: [t5 f32 [H=3 W=4 C=2]] = add a=t4 <-n4 b=t1
      outputs: [t5 f32 [H=3 W=4 C=2] <-n5]
    after:
      graph
      inputs: [t0 f32 [H=2 W=3 C=4] ->[n6], t1 f32 [H=3 W=4 C=2] ->[n0, n2]]
      nodes:
        n0: [t2 f16 [H=2 W=3 C=4] ->[n1]] = permute x=t1 perm=[H<-C, W<-H, C<-W]
        n2: [t3 f32 [H=2 W=3 C=4] ->[n3, n6]] =
          permute x=t1 perm=[H<-C, W<-H, C<-W]
        n1: [] = discard x=t2 <-n0
        n3: [] = discard x=t3 <-n2
        n6: [t6 f32 [H=2 W=3 C=4] ->[n7]] = add a=t0 b=t3 <-n2
        n7: [t5 f32 [H=3 W=4 C=2]] = permute x=t6 <-n6 perm=[H<-W, W<-C, C<-H]
      outputs: [t5 f32 [H=3 W=4 C=2] <-n7]
    map:
      values:
        {t4} -> {} identical
        {} -> {t6} identical
      nodes:
        {n4} -> {n7}
        {n5} -> {n6}
      provenance:
        none |}]

let%expect_test
    "reuse_permute: a self-inverse permutation cannot resolve both roles at \
     once" =
  (* [Found in review, P1]: without disjointness tracking, [pb]'s node was
     both the unwrap target (removed) and the reuse source (its output kept
     alive), corrupting the graph — `Rewrite.apply` failed with "operand t1
     has no definition". There is no other node either operand could use
     instead, so the correct outcome is no match at all. *)
  matches Reuse_permute.pattern (Graph_fixtures.reuse_permute_self_inverse ());
  [%expect {| no match |}]

let%expect_test
    "reuse_permute: the self-inverse case does not corrupt the graph" =
  run (Graph_fixtures.reuse_permute_self_inverse ()) [ Reuse_permute.pass ];
  [%expect
    {|
    before:
      graph
      inputs: [t0 f32 [H=2 W=2 C=4] ->[n0, n1]]
      nodes:
        n0: [t1 f32 [H=2 W=2 C=4] ->[n1]] = permute x=t0 perm=[H<-W, W<-H]
        n1: [t2 f32 [H=2 W=2 C=4]] = add a=t1 <-n0 b=t0
      outputs: [t2 f32 [H=2 W=2 C=4] <-n1]
    after:
      graph
      inputs: [t0 f32 [H=2 W=2 C=4] ->[n0, n1]]
      nodes:
        n0: [t1 f32 [H=2 W=2 C=4] ->[n1]] = permute x=t0 perm=[H<-W, W<-H]
        n1: [t2 f32 [H=2 W=2 C=4]] = add a=t1 <-n0 b=t0
      outputs: [t2 f32 [H=2 W=2 C=4] <-n1]
    map:
      values:
        identity
      nodes:
        identity
      provenance:
        none |}]

let%expect_test "reuse_permute: sub keeps operand order (reuse first)" =
  run (Graph_fixtures.reuse_permute_sub_order ()) [ Reuse_permute.pass ];
  [%expect
    {|
    before:
      graph
      inputs: [t0 f32 [H=2 W=3 C=4] ->[n2], t1 f32 [H=3 W=4 C=2] ->[n0, n3]]
      nodes:
        n0: [t2 f32 [H=2 W=3 C=4] ->[n1]] = permute x=t1 perm=[H<-C, W<-H, C<-W]
        n1: [] = discard x=t2 <-n0
        n2: [t3 f32 [H=3 W=4 C=2] ->[n3]] = permute x=t0 perm=[H<-W, W<-C, C<-H]
        n3: [t4 f32 [H=3 W=4 C=2]] = sub a=t1 b=t3 <-n2
      outputs: [t4 f32 [H=3 W=4 C=2] <-n3]
    after:
      graph
      inputs: [t0 f32 [H=2 W=3 C=4] ->[n4], t1 f32 [H=3 W=4 C=2] ->[n0]]
      nodes:
        n0: [t2 f32 [H=2 W=3 C=4] ->[n1, n4]] =
          permute x=t1 perm=[H<-C, W<-H, C<-W]
        n1: [] = discard x=t2 <-n0
        n4: [t5 f32 [H=2 W=3 C=4] ->[n5]] = sub a=t2 <-n0 b=t0
        n5: [t4 f32 [H=3 W=4 C=2]] = permute x=t5 <-n4 perm=[H<-W, W<-C, C<-H]
      outputs: [t4 f32 [H=3 W=4 C=2] <-n5]
    map:
      values:
        {t3} -> {} identical
        {} -> {t5} identical
      nodes:
        {n2} -> {n5}
        {n3} -> {n4}
      provenance:
        none |}]

let%expect_test "reuse_permute: numeric equivalence, sub (non-commutative)" =
  (* The load-bearing check: getting the rebuilt op's operand order wrong would
     silently negate the result rather than fail to type-check. *)
  let g = Graph_fixtures.reuse_permute_sub_order () in
  let x_pre =
    Tensor.materialize (Graph_fixtures.nhwc ~h:2 ~w:3 ~c:4) (fun c ->
        float_of_int
          ((Dim.to_int (Vec6.get c Axis.H) * 12)
          + (Dim.to_int (Vec6.get c Axis.W) * 4)
          + Dim.to_int (Vec6.get c Axis.C)))
  in
  let a =
    Tensor.materialize (Graph_fixtures.nhwc ~h:3 ~w:4 ~c:2) (fun c ->
        float_of_int
          (100
          + (Dim.to_int (Vec6.get c Axis.H) * 8)
          + (Dim.to_int (Vec6.get c Axis.W) * 2)
          + Dim.to_int (Vec6.get c Axis.C)))
  in
  let same =
    match
      (output_tensor g [ x_pre; a ], rewritten g [ Reuse_permute.pass ])
    with
    | Some before, Some g' -> (
        match output_tensor g' [ x_pre; a ] with
        | Some after -> same_tensor before after
        | None -> false)
    | _ -> false
  in
  Format.printf "same: %b@." same;
  [%expect {| same: true |}]

let%expect_test "reuse_permute: div keeps operand order (reuse first)" =
  run (Graph_fixtures.reuse_permute_div_order ()) [ Reuse_permute.pass ];
  [%expect
    {|
    before:
      graph
      inputs: [t0 f32 [H=2 W=3 C=4] ->[n2], t1 f32 [H=3 W=4 C=2] ->[n0, n3]]
      nodes:
        n0: [t2 f32 [H=2 W=3 C=4] ->[n1]] = permute x=t1 perm=[H<-C, W<-H, C<-W]
        n1: [] = discard x=t2 <-n0
        n2: [t3 f32 [H=3 W=4 C=2] ->[n3]] = permute x=t0 perm=[H<-W, W<-C, C<-H]
        n3: [t4 f32 [H=3 W=4 C=2]] = div a=t1 b=t3 <-n2
      outputs: [t4 f32 [H=3 W=4 C=2] <-n3]
    after:
      graph
      inputs: [t0 f32 [H=2 W=3 C=4] ->[n4], t1 f32 [H=3 W=4 C=2] ->[n0]]
      nodes:
        n0: [t2 f32 [H=2 W=3 C=4] ->[n1, n4]] =
          permute x=t1 perm=[H<-C, W<-H, C<-W]
        n1: [] = discard x=t2 <-n0
        n4: [t5 f32 [H=2 W=3 C=4] ->[n5]] = div a=t2 <-n0 b=t0
        n5: [t4 f32 [H=3 W=4 C=2]] = permute x=t5 <-n4 perm=[H<-W, W<-C, C<-H]
      outputs: [t4 f32 [H=3 W=4 C=2] <-n5]
    map:
      values:
        {t3} -> {} identical
        {} -> {t5} identical
      nodes:
        {n2} -> {n5}
        {n3} -> {n4}
      provenance:
        none |}]

let%expect_test "reuse_permute: numeric equivalence, div (non-commutative)" =
  (* Same load-bearing concern as the sub test above, for the other
     non-commutative op. [x_pre] is offset away from zero so the permuted
     denominator never divides by zero. *)
  let g = Graph_fixtures.reuse_permute_div_order () in
  let x_pre =
    Tensor.materialize (Graph_fixtures.nhwc ~h:2 ~w:3 ~c:4) (fun c ->
        float_of_int
          (1
          + (Dim.to_int (Vec6.get c Axis.H) * 12)
          + (Dim.to_int (Vec6.get c Axis.W) * 4)
          + Dim.to_int (Vec6.get c Axis.C)))
  in
  let a =
    Tensor.materialize (Graph_fixtures.nhwc ~h:3 ~w:4 ~c:2) (fun c ->
        float_of_int
          (100
          + (Dim.to_int (Vec6.get c Axis.H) * 8)
          + (Dim.to_int (Vec6.get c Axis.W) * 2)
          + Dim.to_int (Vec6.get c Axis.C)))
  in
  let same =
    match
      (output_tensor g [ x_pre; a ], rewritten g [ Reuse_permute.pass ])
    with
    | Some before, Some g' -> (
        match output_tensor g' [ x_pre; a ] with
        | Some after -> same_tensor before after
        | None -> false)
    | _ -> false
  in
  Format.printf "same: %b@." same;
  [%expect {| same: true |}]
