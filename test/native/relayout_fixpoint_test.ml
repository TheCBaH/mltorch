(* Why the relayout pass family needs an OUTER fixed point: one application
   of the sequence is not enough, a second sinks what the first exposed.
   Split from permute_passes_test.ml. *)

open Permute_pass_fixtures

(* ---- why the relayout family needs an OUTER fixed point ------------------ *)

(* The same list `bin/native_graph.ml`'s `relayout_pass` wraps in one more
   [Pass.fixpoint]: each pass already fixpoints on its own, but the group as a
   whole does not, because a later stage's rewrite can expose a match for an
   EARLIER one. *)
let relayout_sequence =
  [
    Pass.fixpoint Chain_permute.pass;
    Pass.fixpoint Trim_permute.pass;
    Pass.fixpoint Sink_permute.pass;
    Pass.fixpoint Reuse_permute.pass;
    Pass.fixpoint Sink_permute.pass;
    Pass.fixpoint Bypass_permute.pass;
    Pass.fixpoint Chain_permute.pass;
    Pass.fixpoint Trim_permute.pass;
  ]

let%expect_test "relayout: one application of the sequence is not enough" =
  (* [P]'s output is shared between an inverse [Q] (bypassable) and a plain
     [Relu] — a UNARY consumer, so [Reuse_permute] can never help here, that
     pass is binary-only. Both [Sink_permute] stages run BEFORE [Bypass_permute]
     in the list, so the sweep that removes [Q] and makes [y] interior comes
     too late for [P] to be sunk through the [Relu] in the SAME application:
     it is still sitting there afterwards. *)
  run (Graph_fixtures.bypass_unlocks_sink ()) relayout_sequence;
  [%expect
    {|
    before:
      graph
      inputs: [t0 f32 [H=2 W=3 C=4] ->[n0]]
      nodes:
        n0: [t1 f32 [H=3 W=4 C=2] ->[n1, n3]] =
          permute x=t0 perm=[H<-W, W<-C, C<-H]
        n1: [t2 f32 [H=2 W=3 C=4] ->[n2]] =
          permute x=t1 <-n0 perm=[H<-C, W<-H, C<-W]
        n2: [] = discard x=t2 <-n1
        n3: [t3 f32 [H=3 W=4 C=2]] = relu x=t1 <-n0
      outputs: [t3 f32 [H=3 W=4 C=2] <-n3]
    after:
      graph
      inputs: [t0 f32 [H=2 W=3 C=4] ->[n0, n2]]
      nodes:
        n0: [t1 f32 [H=3 W=4 C=2] ->[n3]] = permute x=t0 perm=[H<-W, W<-C, C<-H]
        n2: [] = discard x=t0
        n3: [t3 f32 [H=3 W=4 C=2]] = relu x=t1 <-n0
      outputs: [t3 f32 [H=3 W=4 C=2] <-n3]
    map:
      values:
        {t0, t2} -> {t0} identical
      nodes:
        {n1} -> {}
      provenance:
        none |}]

let%expect_test "relayout: a SECOND application sinks what the first exposed" =
  (* Wrapping the very same list in one more [Pass.fixpoint] is what
     `relayout_pass` does. The permute that survived the test above now has
     only one consumer, so this second pass through the sequence sinks it
     through the [Relu], leaving no permute at all. *)
  run
    (Graph_fixtures.bypass_unlocks_sink ())
    [ Pass.fixpoint (Pass.sequence ~name:"relayout" relayout_sequence) ];
  [%expect
    {|
    before:
      graph
      inputs: [t0 f32 [H=2 W=3 C=4] ->[n0]]
      nodes:
        n0: [t1 f32 [H=3 W=4 C=2] ->[n1, n3]] =
          permute x=t0 perm=[H<-W, W<-C, C<-H]
        n1: [t2 f32 [H=2 W=3 C=4] ->[n2]] =
          permute x=t1 <-n0 perm=[H<-C, W<-H, C<-W]
        n2: [] = discard x=t2 <-n1
        n3: [t3 f32 [H=3 W=4 C=2]] = relu x=t1 <-n0
      outputs: [t3 f32 [H=3 W=4 C=2] <-n3]
    after:
      graph
      inputs: [t0 f32 [H=2 W=3 C=4] ->[n2, n4]]
      nodes:
        n4: [t4 f32 [H=2 W=3 C=4] ->[n5]] = relu x=t0
        n2: [] = discard x=t0
        n5: [t3 f32 [H=3 W=4 C=2]] = permute x=t4 <-n4 perm=[H<-W, W<-C, C<-H]
      outputs: [t3 f32 [H=3 W=4 C=2] <-n5]
    map:
      values:
        {t0, t2} -> {t0} identical
        {t1} -> {} identical
        {} -> {t4} identical
      nodes:
        {n0} -> {n5}
        {n1} -> {}
        {n3} -> {n4}
      provenance:
        none |}]

let%expect_test "reshape_to_permute feeds the permute passes" =
  (* The point of the conversion: a reshape is opaque to fusion, a permute is
     not. Here the relabelling cancels against the permute above it, and the
     pair disappears. *)
  let g =
    Graph_builder.build ~name:"reshape_then_permute"
      ~outputs:(fun o -> [ o ])
      Graph_builder.(
        let* x = input ~shape:(Graph_fixtures.s 1 1 1 1 1 6) () in
        let* r =
          reshape { Reshape.Reshape.shape = Graph_fixtures.s 1 1 1 6 1 1 } x
        in
        let* p =
          permute Axis.[ (N, N); (T, T); (D, D); (H, W); (W, C); (C, H) ] r
        in
        relu p)
    |> Err.or_raise ~pp_error:Graph_builder.pp_error
  in
  run g
    [
      Reshape_to_permute.pass;
      Pass.fixpoint Chain_permute.pass;
      Trim_permute.pass;
    ];
  [%expect
    {|
    before:
      graph
      inputs: [t0 f32 [C=6] ->[n0]]
      nodes:
        n0: [t1 f32 [H=6 W=1 C=1] ->[n1]] =
          reshape x=t0 params={shape=[H=6 W=1 C=1]}
        n1: [t2 f32 [C=6] ->[n2]] = permute x=t1 <-n0 perm=[H<-W, W<-C, C<-H]
        n2: [t3 f32 [C=6]] = relu x=t2 <-n1
      outputs: [t3 f32 [C=6] <-n2]
    after:
      graph
      inputs: [t0 f32 [C=6] ->[n2]]
      nodes:
        n2: [t3 f32 [C=6]] = relu x=t0
      outputs: [t3 f32 [C=6] <-n2]
    map:
      values:
        {t0, t2} -> {t0} identical
        {t1} -> {} identical
      nodes:
        {n0, n1} -> {}
      provenance:
        none |}]
