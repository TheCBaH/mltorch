(* Transporting a permute through a keepdim=true Mean. Split from
   permute_passes_test.ml. *)

open Permute_pass_fixtures

(* ---- transporting a permute through a keepdim=true Mean ------------------- *)

let%expect_test "sink_permute_mean: no downstream inverse" =
  run (Graph_fixtures.sink_permute_mean_basic ()) [ Sink_permute_mean.pass ];
  [%expect
    {|
    before:
      graph
      inputs: [t0 f32 [H=2 W=3 C=4] ->[n0]]
      nodes:
        n0: [t1 f32 [H=3 W=4 C=2] ->[n1]] = permute x=t0 perm=[H<-W, W<-C, C<-H]
        n1: [t2 f32 [H=3 W=1 C=1]] =
          mean x=t1 <-n0 params={dims=[W, C]; keepdim=true}
      outputs: [t2 f32 [H=3 W=1 C=1] <-n1]
    after:
      graph
      inputs: [t0 f32 [H=2 W=3 C=4] ->[n2]]
      nodes:
        n2: [t3 f32 [W=3 C=1] ->[n3]] =
          mean x=t0 params={dims=[C, H]; keepdim=true}
        n3: [t2 f32 [H=3 W=1 C=1]] = permute x=t3 <-n2 perm=[H<-W, W<-C, C<-H]
      outputs: [t2 f32 [H=3 W=1 C=1] <-n3]
    map:
      values:
        {t1} -> {} identical
        {} -> {t3} identical
      nodes:
        {n0} -> {n3}
        {n1} -> {n2}
      provenance:
        none |}]

let%expect_test "sink_permute_mean: the motivating cycle collapses" =
  run
    (Graph_fixtures.sink_permute_mean_cycle ())
    [
      Pass.fixpoint Sink_permute_mean.pass;
      Pass.fixpoint Chain_permute.pass;
      Pass.fixpoint Trim_permute.pass;
    ];
  [%expect
    {|
    before:
      graph
      inputs: [t0 f32 [H=2 W=3 C=4] ->[n0]]
      nodes:
        n0: [t1 f32 [H=3 W=4 C=2] ->[n1]] = permute x=t0 perm=[H<-W, W<-C, C<-H]
        n1: [t2 f32 [H=3 W=1 C=1] ->[n2]] =
          mean x=t1 <-n0 params={dims=[W, C]; keepdim=true}
        n2: [t3 f32 [W=3 C=1] ->[n3]] = permute x=t2 <-n1 perm=[H<-C, W<-H, C<-W]
        n3: [t4 f32 [W=3 C=1]] = relu x=t3 <-n2
      outputs: [t4 f32 [W=3 C=1] <-n3]
    after:
      graph
      inputs: [t0 f32 [H=2 W=3 C=4] ->[n4]]
      nodes:
        n4: [t5 f32 [W=3 C=1] ->[n3]] =
          mean x=t0 params={dims=[C, H]; keepdim=true}
        n3: [t4 f32 [W=3 C=1]] = relu x=t5 <-n4
      outputs: [t4 f32 [W=3 C=1] <-n3]
    map:
      values:
        {t1} -> {} identical
        {t2} -> {} identical
        {t3} -> {t5} identical
      nodes:
        {n0, n2} -> {}
        {n1} -> {n4}
      provenance:
        none |}]

let%expect_test "sink_permute_mean: a shared input permutation is declined" =
  matches Sink_permute_mean.pattern (Graph_fixtures.sink_permute_mean_shared ());
  [%expect {| no match |}]

let%expect_test "sink_permute_mean: keepdim=false is declined" =
  matches Sink_permute_mean.pattern
    (Graph_fixtures.sink_permute_mean_not_keepdim ());
  [%expect {| no match |}]

let%expect_test "sink_permute_mean: numeric equivalence, dimension order kept" =
  (* [dims=[W,C]] maps through [rotate_hwc] to [dims=[C,H]] — not [[H,C]] — so
     this pins that [map_dims] preserves list order, not just set membership.
     Bit-identical output confirms the reduction visits axes in the same
     order before and after transport. *)
  let g = Graph_fixtures.sink_permute_mean_basic () in
  let x =
    Tensor.materialize (Graph_fixtures.nhwc ~h:2 ~w:3 ~c:4) (fun c ->
        float_of_int
          ((Dim.to_int (Vec6.get c Axis.H) * 12)
          + (Dim.to_int (Vec6.get c Axis.W) * 4)
          + Dim.to_int (Vec6.get c Axis.C)))
  in
  Format.printf "before: %s@." (evaluated g [ x ]);
  (match rewritten g [ Sink_permute_mean.pass ] with
  | None -> Format.printf "rewrite failed@."
  | Some g' ->
      Format.printf "after:  %s@." (evaluated g' [ x ]);
      let same =
        match (output_tensor g [ x ], output_tensor g' [ x ]) with
        | Some before, Some after -> same_tensor before after
        | _ -> false
      in
      Format.printf "same:   %b@." same);
  [%expect
    {|
    before: tensor f32 [H=3 W=1 C=1] {7.5, 11.5, 15.5}
    after:  tensor f32 [H=3 W=1 C=1] {7.5, 11.5, 15.5}
    same:   true |}]
