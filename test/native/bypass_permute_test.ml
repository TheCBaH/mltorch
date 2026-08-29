(* Removing individual inverse consumers of a permute, independent of
   whether the permute itself can be dropped. Split from
   permute_passes_test.ml. *)

open Permute_pass_fixtures

(* ---- removing individual inverse consumers ------------------------------- *)

let%expect_test "bypass_permute: one P, one inverse Q, nothing else" =
  run (Graph_fixtures.bypass_permute_pair ()) [ Bypass_permute.pass ];
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

let%expect_test "bypass_permute: one P, several inverse Q consumers" =
  run (Graph_fixtures.bypass_permute_fanout ()) [ Bypass_permute.pass ];
  [%expect
    {|
    before:
      graph
      inputs: [t0 f32 [H=2 W=3 C=4] ->[n0]]
      nodes:
        n0: [t1 f32 [H=3 W=4 C=2] ->[n1, n2]] =
          permute x=t0 perm=[H<-W, W<-C, C<-H]
        n1: [t2 f32 [H=2 W=3 C=4] ->[n3]] =
          permute x=t1 <-n0 perm=[H<-C, W<-H, C<-W]
        n2: [t3 f32 [H=2 W=3 C=4] ->[n4]] =
          permute x=t1 <-n0 perm=[H<-C, W<-H, C<-W]
        n3: [t4 f32 [H=2 W=3 C=4]] = relu x=t2 <-n1
        n4: [t5 f32 [H=2 W=3 C=4]] = relu x=t3 <-n2
      outputs: [t4 f32 [H=2 W=3 C=4] <-n3, t5 f32 [H=2 W=3 C=4] <-n4]
    after:
      graph
      inputs: [t0 f32 [H=2 W=3 C=4] ->[n3, n4]]
      nodes:
        n3: [t4 f32 [H=2 W=3 C=4]] = relu x=t0
        n4: [t5 f32 [H=2 W=3 C=4]] = relu x=t0
      outputs: [t4 f32 [H=2 W=3 C=4] <-n3, t5 f32 [H=2 W=3 C=4] <-n4]
    map:
      values:
        {t0, t2, t3} -> {t0} identical
        {t1} -> {} identical
      nodes:
        {n0} -> {}
        {n1} -> {}
        {n2} -> {}
      provenance:
        none |}]

let%expect_test "bypass_permute: a shared P output stays, its inverse Q goes" =
  run (Graph_fixtures.bypass_permute_shared ()) [ Bypass_permute.pass ];
  [%expect
    {|
    before:
      graph
      inputs: [t0 f32 [H=2 W=3 C=4] ->[n0]]
      nodes:
        n0: [t1 f32 [H=3 W=4 C=2] ->[n1, n2]] =
          permute x=t0 perm=[H<-W, W<-C, C<-H]
        n1: [t2 f32 [H=2 W=3 C=4] ->[n3]] =
          permute x=t1 <-n0 perm=[H<-C, W<-H, C<-W]
        n2: [t3 f32 [H=3 W=4 C=2]] = relu x=t1 <-n0
        n3: [t4 f32 [H=2 W=3 C=4]] = relu x=t2 <-n1
      outputs: [t4 f32 [H=2 W=3 C=4] <-n3, t3 f32 [H=3 W=4 C=2] <-n2]
    after:
      graph
      inputs: [t0 f32 [H=2 W=3 C=4] ->[n0, n3]]
      nodes:
        n0: [t1 f32 [H=3 W=4 C=2] ->[n2]] = permute x=t0 perm=[H<-W, W<-C, C<-H]
        n3: [t4 f32 [H=2 W=3 C=4]] = relu x=t0
        n2: [t3 f32 [H=3 W=4 C=2]] = relu x=t1 <-n0
      outputs: [t4 f32 [H=2 W=3 C=4] <-n3, t3 f32 [H=3 W=4 C=2] <-n2]
    map:
      values:
        {t0, t2} -> {t0} identical
      nodes:
        {n1} -> {}
      provenance:
        none |}]

let%expect_test
    "bypass_permute: incompatible and compatible inverse consumers together" =
  (* Filtering, not an all-or-nothing guard: the compatible [q_ok] is
     bypassed while the incompatible [q_bad] is left as an ordinary consumer
     — which is exactly what keeps [P] alive, the same reason
     [bypass_permute_shared] keeps it for an unrelated [Relu]. *)
  run
    (Graph_fixtures.bypass_permute_mixed_compatibility ())
    [ Bypass_permute.pass ];
  [%expect
    {|
    before:
      graph
      inputs: [t0 f32 [H=2 W=3 C=4] ->[n0]]
      nodes:
        n0: [t1 f32 [H=3 W=4 C=2] ->[n1, n2]] =
          permute x=t0 perm=[H<-W, W<-C, C<-H]
        n1: [t2 f32 [H=2 W=3 C=4] ->[n3]] =
          permute x=t1 <-n0 perm=[H<-C, W<-H, C<-W]
        n2: [t3 f16 [H=2 W=3 C=4] ->[n4]] =
          permute x=t1 <-n0 perm=[H<-C, W<-H, C<-W]
        n3: [t4 f32 [H=2 W=3 C=4]] = relu x=t2 <-n1
        n4: [t5 f32 [H=2 W=3 C=4]] = relu x=t3 <-n2
      outputs: [t4 f32 [H=2 W=3 C=4] <-n3, t5 f32 [H=2 W=3 C=4] <-n4]
    after:
      graph
      inputs: [t0 f32 [H=2 W=3 C=4] ->[n0, n3]]
      nodes:
        n0: [t1 f32 [H=3 W=4 C=2] ->[n2]] = permute x=t0 perm=[H<-W, W<-C, C<-H]
        n3: [t4 f32 [H=2 W=3 C=4]] = relu x=t0
        n2: [t3 f16 [H=2 W=3 C=4] ->[n4]] =
          permute x=t1 <-n0 perm=[H<-C, W<-H, C<-W]
        n4: [t5 f32 [H=2 W=3 C=4]] = relu x=t3 <-n2
      outputs: [t4 f32 [H=2 W=3 C=4] <-n3, t5 f32 [H=2 W=3 C=4] <-n4]
    map:
      values:
        {t0, t2} -> {t0} identical
      nodes:
        {n1} -> {}
      provenance:
        none |}]

let%expect_test "bypass_permute: P's output is itself a graph output" =
  run (Graph_fixtures.bypass_permute_output ()) [ Bypass_permute.pass ];
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
      outputs: [t1 f32 [H=3 W=4 C=2] ->[n1] <-n0, t3 f32 [H=2 W=3 C=4] <-n2]
    after:
      graph
      inputs: [t0 f32 [H=2 W=3 C=4] ->[n0, n2]]
      nodes:
        n0: [t1 f32 [H=3 W=4 C=2]] = permute x=t0 perm=[H<-W, W<-C, C<-H]
        n2: [t3 f32 [H=2 W=3 C=4]] = relu x=t0
      outputs: [t1 f32 [H=3 W=4 C=2] <-n0, t3 f32 [H=2 W=3 C=4] <-n2]
    map:
      values:
        {t0, t2} -> {t0} identical
      nodes:
        {n1} -> {}
      provenance:
        none |}]
