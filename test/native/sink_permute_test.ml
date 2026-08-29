(* Sinking a permute through an elementwise op (relu, add, and every other
   accepted op), including the barriers (pad/slice/layer_norm/sdpa) and the
   combination with chain/trim. Split from permute_passes_test.ml. *)

open Permute_pass_fixtures

(* ---- sinking a permute through an elementwise op -------------------------- *)

let%expect_test "sink_permute: through relu, one sweep is enough" =
  run (Graph_fixtures.sink_permute_unary ()) [ Sink_permute.pass ];
  [%expect
    {|
    before:
      graph
      inputs: [t0 f32 [H=2 W=3 C=4] ->[n0]]
      nodes:
        n0: [t1 f32 [H=3 W=4 C=2] ->[n1]] = permute x=t0 perm=[H<-W, W<-C, C<-H]
        n1: [t2 f32 [H=3 W=4 C=2] ->[n2]] = relu x=t1 <-n0
        n2: [t3 f32 [H=2 W=3 C=4]] = permute x=t2 <-n1 perm=[H<-C, W<-H, C<-W]
      outputs: [t3 f32 [H=2 W=3 C=4] <-n2]
    after:
      graph
      inputs: [t0 f32 [H=2 W=3 C=4] ->[n3]]
      nodes:
        n3: [t4 f32 [H=2 W=3 C=4] ->[n4]] = relu x=t0
        n4: [t2 f32 [H=3 W=4 C=2] ->[n2]] =
          permute x=t4 <-n3 perm=[H<-W, W<-C, C<-H]
        n2: [t3 f32 [H=2 W=3 C=4]] = permute x=t2 <-n4 perm=[H<-C, W<-H, C<-W]
      outputs: [t3 f32 [H=2 W=3 C=4] <-n2]
    map:
      values:
        {t1} -> {} identical
        {} -> {t4} identical
      nodes:
        {n0} -> {n4}
        {n1} -> {n3}
      provenance:
        none |}]

let%expect_test "sink_permute: through add, one sweep only reaches the add" =
  (* [relu] hasn't sunk yet after a single application — its operand is still
     produced by [Add], not yet a [Permute] — so this pins that crossing BOTH
     [add] and [relu] genuinely needs the second sweep [Pass.fixpoint] below
     supplies, not that one sweep happens to already suffice. *)
  run (Graph_fixtures.sink_permute_binary ()) [ Sink_permute.pass ];
  [%expect
    {|
    before:
      graph
      inputs: [t0 f32 [H=2 W=3 C=4] ->[n0], t1 f32 [H=2 W=3 C=4] ->[n1]]
      nodes:
        n0: [t2 f32 [H=3 W=4 C=2] ->[n2]] = permute x=t0 perm=[H<-W, W<-C, C<-H]
        n1: [t3 f32 [H=3 W=4 C=2] ->[n2]] = permute x=t1 perm=[H<-W, W<-C, C<-H]
        n2: [t4 f32 [H=3 W=4 C=2] ->[n3]] = add a=t2 <-n0 b=t3 <-n1
        n3: [t5 f32 [H=3 W=4 C=2] ->[n4]] = relu x=t4 <-n2
        n4: [t6 f32 [H=2 W=3 C=4]] = permute x=t5 <-n3 perm=[H<-C, W<-H, C<-W]
      outputs: [t6 f32 [H=2 W=3 C=4] <-n4]
    after:
      graph
      inputs: [t0 f32 [H=2 W=3 C=4] ->[n5], t1 f32 [H=2 W=3 C=4] ->[n5]]
      nodes:
        n5: [t7 f32 [H=2 W=3 C=4] ->[n6]] = add a=t0 b=t1
        n6: [t4 f32 [H=3 W=4 C=2] ->[n3]] =
          permute x=t7 <-n5 perm=[H<-W, W<-C, C<-H]
        n3: [t5 f32 [H=3 W=4 C=2] ->[n4]] = relu x=t4 <-n6
        n4: [t6 f32 [H=2 W=3 C=4]] = permute x=t5 <-n3 perm=[H<-C, W<-H, C<-W]
      outputs: [t6 f32 [H=2 W=3 C=4] <-n4]
    map:
      values:
        {t2} -> {} identical
        {t3} -> {} identical
        {} -> {t7} identical
      nodes:
        {n0, n1} -> {n6}
        {n2} -> {n5}
      provenance:
        none |}]

let%expect_test "sink_permute: perm mismatch is declined outright" =
  matches Sink_permute.pattern (Graph_fixtures.sink_permute_mismatch ());
  [%expect {| no match |}]

let%expect_test "sink_permute: a shared operand is not interior" =
  matches Sink_permute.pattern (Graph_fixtures.sink_permute_shared ());
  [%expect {| no match |}]

let%expect_test "sink_permute: an operand that is also a graph output" =
  matches Sink_permute.pattern (Graph_fixtures.sink_permute_output ());
  [%expect {| no match |}]

(* Pad is NOT on the allowlist and must not be: it names axes, so sinking a
   permute past it would leave the pad entries attached to the axes they had
   BEFORE the reorder. The allowlist match has a default arm, so this exclusion
   is silent in the source -- it is a fixture or it is nothing. *)
let%expect_test "sink_permute: pad is a barrier" =
  matches Sink_permute.pattern (Graph_fixtures.sink_permute_pad ());
  [%expect {| no match |}]

(* Slice is excluded for the same reason and by the same silent default arm: it
   names an axis, so sinking a permute past it would leave the bounds attached
   to whichever axis landed where the named one used to be. Checked at the
   PATTERN rather than at the output, because a pass that sank it anyway would
   still produce a graph that validates -- the wrong axis is a different answer,
   not a malformed one. *)
let%expect_test "sink_permute: slice is a barrier" =
  matches Sink_permute.pattern (Graph_fixtures.sink_permute_slice ());
  [%expect {| no match |}]

(* Layer_norm is excluded for the same reason and by the same silent default
   arm, with one extra consequence the other two do not have: its affine
   operands carry the normalized_shape, so sinking would also read weight and
   bias on axes they were never sized for. *)
let%expect_test "sink_permute: layer_norm is a barrier" =
  matches Sink_permute.pattern (Graph_fixtures.sink_permute_layer_norm ());
  [%expect {| no match |}]

(* Sdpa is excluded for the same reason: it names D/H/W/C as batch/heads/
   sequence/head-dim, so sinking a permute past it would read those axes on
   data sized for different ones (op8-impl.md commit 1 step 6). *)
let%expect_test "sink_permute: sdpa is a barrier" =
  matches Sink_permute.pattern (Graph_fixtures.sink_permute_sdpa ());
  [%expect {| no match |}]

let%expect_test "sink_permute: every accepted op sinks" =
  run ~show_before:false
    (Graph_fixtures.sink_permute_allowlist ())
    [ Pass.fixpoint Sink_permute.pass ];
  [%expect
    {|
    after:
      graph
      inputs:
        [t0 f32 [H=2 W=3 C=4] ->[n57, n59, n61, n63, n65, n67, n69, n71, n73,
                                 n75, n77, n79, n81, n83, n85, n87, n89, n91],
         t1 f32 [H=2 W=3 C=4] ->[n85, n87, n89, n91]]
      nodes:
        n57: [t59 f32 [H=2 W=3 C=4] ->[n97]] = relu x=t0
        n59: [t60 f32 [H=2 W=3 C=4] ->[n97]] = sqrt x=t0
        n61: [t61 f32 [H=2 W=3 C=4] ->[n99]] = clone x=t0
        n63: [t62 f32 [H=2 W=3 C=4] ->[n99]] = add_scalar x=t0 scalar=3
        n65: [t63 f32 [H=2 W=3 C=4] ->[n101]] = div_scalar x=t0 scalar=6
        n67: [t64 f32 [H=2 W=3 C=4] ->[n101]] = clamp x=t0 params={min=0; max=6}
        n69: [t65 f32 [H=2 W=3 C=4] ->[n115]] =
          hardtanh x=t0 params={min_val=0; max_val=6}
        n71: [t66 f32 [H=2 W=3 C=4] ->[n103]] = silu x=t0
        n73: [t67 f32 [H=2 W=3 C=4] ->[n105]] = sigmoid x=t0
        n75: [t68 f32 [H=2 W=3 C=4] ->[n105]] = gelu x=t0 approximate=none
        n77: [t69 f32 [H=2 W=3 C=4] ->[n113]] = mul_scalar x=t0 scalar=2
        n79: [t70 f32 [H=2 W=3 C=4] ->[n125]] = pow x=t0 scalar=2
        n81: [t71 f32 [H=2 W=3 C=4] ->[n103]] = hardsigmoid x=t0
        n83: [t72 f32 [H=2 W=3 C=4] ->[n111]] = hardswish x=t0
        n85: [t73 f32 [H=2 W=3 C=4] ->[n93]] = add a=t0 b=t1
        n87: [t74 f32 [H=2 W=3 C=4] ->[n93]] = sub a=t0 b=t1
        n89: [t75 f32 [H=2 W=3 C=4] ->[n95]] = mul a=t0 b=t1
        n91: [t76 f32 [H=2 W=3 C=4] ->[n95]] = div a=t0 b=t1
        n97: [t79 f32 [H=2 W=3 C=4] ->[n117]] = add a=t59 <-n57 b=t60 <-n59
        n99: [t80 f32 [H=2 W=3 C=4] ->[n109]] = add a=t61 <-n61 b=t62 <-n63
        n101: [t81 f32 [H=2 W=3 C=4] ->[n109]] = add a=t63 <-n65 b=t64 <-n67
        n103: [t82 f32 [H=2 W=3 C=4] ->[n111]] = add a=t66 <-n71 b=t71 <-n81
        n105: [t83 f32 [H=2 W=3 C=4] ->[n113]] = add a=t67 <-n73 b=t68 <-n75
        n93: [t77 f32 [H=2 W=3 C=4] ->[n107]] = add a=t73 <-n85 b=t74 <-n87
        n95: [t78 f32 [H=2 W=3 C=4] ->[n107]] = add a=t75 <-n89 b=t76 <-n91
        n113: [t87 f32 [H=2 W=3 C=4] ->[n123]] = add a=t83 <-n105 b=t69 <-n77
        n111: [t86 f32 [H=2 W=3 C=4] ->[n121]] = add a=t82 <-n103 b=t72 <-n83
        n109: [t85 f32 [H=2 W=3 C=4] ->[n115]] = add a=t80 <-n99 b=t81 <-n101
        n107: [t84 f32 [H=2 W=3 C=4] ->[n117]] = add a=t77 <-n93 b=t78 <-n95
        n115: [t88 f32 [H=2 W=3 C=4] ->[n119]] = add a=t85 <-n109 b=t65 <-n69
        n117: [t89 f32 [H=2 W=3 C=4] ->[n119]] = add a=t84 <-n107 b=t79 <-n97
        n119: [t90 f32 [H=2 W=3 C=4] ->[n121]] = add a=t89 <-n117 b=t88 <-n115
        n121: [t91 f32 [H=2 W=3 C=4] ->[n123]] = add a=t90 <-n119 b=t86 <-n111
        n123: [t92 f32 [H=2 W=3 C=4] ->[n125]] = add a=t91 <-n121 b=t87 <-n113
        n125: [t93 f32 [H=2 W=3 C=4] ->[n126]] = add a=t92 <-n123 b=t70 <-n79
        n126: [t58 f32 [H=3 W=4 C=2]] =
          permute x=t93 <-n125 perm=[H<-W, W<-C, C<-H]
      outputs: [t58 f32 [H=3 W=4 C=2] <-n126]
    map:
      values:
        {t2} -> {} identical
        {t3} -> {} identical
        {t4} -> {} identical
        {t5} -> {} identical
        {t6} -> {} identical
        {t7} -> {} identical
        {t8} -> {} identical
        {t9} -> {} identical
        {t10} -> {} identical
        {t11} -> {} identical
        {t12} -> {} identical
        {t13} -> {} identical
        {t14} -> {} identical
        {t15} -> {} identical
        {t16} -> {} identical
        {t17} -> {} identical
        {t18} -> {} identical
        {t19} -> {} identical
        {t20} -> {} identical
        {t21} -> {} identical
        {t22} -> {} identical
        {t23} -> {} identical
        {t24} -> {} identical
        {t25} -> {} identical
        {t26} -> {} identical
        {t27} -> {} identical
        {t28} -> {} identical
        {t29} -> {} identical
        {t30} -> {} identical
        {t31} -> {} identical
        {t32} -> {} identical
        {t33} -> {} identical
        {t34} -> {} identical
        {t35} -> {} identical
        {t36} -> {} identical
        {t37} -> {} identical
        {t38} -> {} identical
        {t39} -> {} identical
        {t40} -> {} identical
        {t41} -> {} identical
        {t42} -> {} identical
        {t43} -> {} identical
        {t44} -> {} identical
        {t45} -> {} identical
        {t46} -> {} identical
        {t47} -> {} identical
        {t48} -> {} identical
        {t49} -> {} identical
        {t50} -> {} identical
        {t51} -> {} identical
        {t52} -> {} identical
        {t53} -> {} identical
        {t54} -> {} identical
        {t55} -> {} identical
        {t56} -> {} identical
        {t57} -> {} identical
        {} -> {t59} identical
        {} -> {t60} identical
        {} -> {t61} identical
        {} -> {t62} identical
        {} -> {t63} identical
        {} -> {t64} identical
        {} -> {t65} identical
        {} -> {t66} identical
        {} -> {t67} identical
        {} -> {t68} identical
        {} -> {t69} identical
        {} -> {t70} identical
        {} -> {t71} identical
        {} -> {t72} identical
        {} -> {t73} identical
        {} -> {t74} identical
        {} -> {t75} identical
        {} -> {t76} identical
        {} -> {t77} identical
        {} -> {t78} identical
        {} -> {t79} identical
        {} -> {t80} identical
        {} -> {t81} identical
        {} -> {t82} identical
        {} -> {t83} identical
        {} -> {t84} identical
        {} -> {t85} identical
        {} -> {t86} identical
        {} -> {t87} identical
        {} -> {t88} identical
        {} -> {t89} identical
        {} -> {t90} identical
        {} -> {t91} identical
        {} -> {t92} identical
        {} -> {t93} identical
      nodes:
        {n0, n2, n4, n6, n8, n10, n12, n14, n16, n18, n20, n22, n24, n26, n28,
         n29, n31, n32, n34, n35, n37, n38} -> {n126}
        {n1} -> {n57}
        {n3} -> {n59}
        {n5} -> {n61}
        {n7} -> {n63}
        {n9} -> {n65}
        {n11} -> {n67}
        {n13} -> {n69}
        {n15} -> {n71}
        {n17} -> {n73}
        {n19} -> {n75}
        {n21} -> {n77}
        {n23} -> {n79}
        {n25} -> {n81}
        {n27} -> {n83}
        {n30} -> {n85}
        {n33} -> {n87}
        {n36} -> {n89}
        {n39} -> {n91}
        {n40} -> {n93}
        {n41} -> {n95}
        {n42} -> {n97}
        {n43} -> {n107}
        {n44} -> {n99}
        {n45} -> {n101}
        {n46} -> {n109}
        {n47} -> {n115}
        {n48} -> {n117}
        {n49} -> {n103}
        {n50} -> {n111}
        {n51} -> {n119}
        {n52} -> {n105}
        {n53} -> {n113}
        {n54} -> {n121}
        {n55} -> {n123}
        {n56} -> {n125}
      provenance:
        none |}]

let%expect_test "sink_permute then chain/trim: unary case fully cancels" =
  run
    (Graph_fixtures.sink_permute_unary ())
    [ Pass.fixpoint Sink_permute.pass; Chain_permute.pass; Trim_permute.pass ];
  [%expect
    {|
    before:
      graph
      inputs: [t0 f32 [H=2 W=3 C=4] ->[n0]]
      nodes:
        n0: [t1 f32 [H=3 W=4 C=2] ->[n1]] = permute x=t0 perm=[H<-W, W<-C, C<-H]
        n1: [t2 f32 [H=3 W=4 C=2] ->[n2]] = relu x=t1 <-n0
        n2: [t3 f32 [H=2 W=3 C=4]] = permute x=t2 <-n1 perm=[H<-C, W<-H, C<-W]
      outputs: [t3 f32 [H=2 W=3 C=4] <-n2]
    after:
      graph
      inputs: [t0 f32 [H=2 W=3 C=4] ->[n3]]
      nodes:
        n3: [t4 f32 [H=2 W=3 C=4]] = relu x=t0
      outputs: [t4 f32 [H=2 W=3 C=4] <-n3]
    map:
      values:
        {t1} -> {} identical
        {t2} -> {} identical
        {t3} -> {t4} identical
      nodes:
        {n0, n2} -> {}
        {n1} -> {n3}
      provenance:
        none |}]

let%expect_test "sink_permute then chain/trim: a non-cancelling pair fuses" =
  (* [rotate_hwc] and [swap_hw] are not inverses, so this pins the other half
     of the end-to-end interaction: sinking exposes the adjacency, but here
     [Chain_permute] fuses it into one composite permute — [Trim_permute]
     leaves that alone since it isn't the identity — rather than the pair
     cancelling outright as in the unary case above. Three nodes become two,
     and the final output id is preserved either way. *)
  run
    (Graph_fixtures.sink_permute_fuse ())
    [ Pass.fixpoint Sink_permute.pass; Chain_permute.pass; Trim_permute.pass ];
  [%expect
    {|
    before:
      graph
      inputs: [t0 f32 [H=2 W=3 C=4] ->[n0]]
      nodes:
        n0: [t1 f32 [H=3 W=4 C=2] ->[n1]] = permute x=t0 perm=[H<-W, W<-C, C<-H]
        n1: [t2 f32 [H=3 W=4 C=2] ->[n2]] = relu x=t1 <-n0
        n2: [t3 f32 [H=4 W=3 C=2]] = permute x=t2 <-n1 perm=[H<-W, W<-H]
      outputs: [t3 f32 [H=4 W=3 C=2] <-n2]
    after:
      graph
      inputs: [t0 f32 [H=2 W=3 C=4] ->[n3]]
      nodes:
        n3: [t4 f32 [H=2 W=3 C=4] ->[n5]] = relu x=t0
        n5: [t3 f32 [H=4 W=3 C=2]] = permute x=t4 <-n3 perm=[H<-C, C<-H]
      outputs: [t3 f32 [H=4 W=3 C=2] <-n5]
    map:
      values:
        {t1} -> {} identical
        {t2} -> {} identical
        {} -> {t4} identical
      nodes:
        {n0, n2} -> {n5}
        {n1} -> {n3}
      provenance:
        none |}]

let%expect_test "sink_permute then chain/trim: binary case fully cancels" =
  (* The real ResNet residual-add shape: needs [Pass.fixpoint Sink_permute.pass]
     (add, then relu) before chain/trim have two adjacent permutes to collapse. *)
  run ~show_before:false
    (Graph_fixtures.sink_permute_binary ())
    [ Pass.fixpoint Sink_permute.pass; Chain_permute.pass; Trim_permute.pass ];
  [%expect
    {|
    after:
      graph
      inputs: [t0 f32 [H=2 W=3 C=4] ->[n5], t1 f32 [H=2 W=3 C=4] ->[n5]]
      nodes:
        n5: [t7 f32 [H=2 W=3 C=4] ->[n7]] = add a=t0 b=t1
        n7: [t8 f32 [H=2 W=3 C=4]] = relu x=t7 <-n5
      outputs: [t8 f32 [H=2 W=3 C=4] <-n7]
    map:
      values:
        {t2} -> {} identical
        {t3} -> {} identical
        {t4} -> {} identical
        {t5} -> {} identical
        {t6} -> {t8} identical
        {} -> {t7} identical
      nodes:
        {n0, n1, n4} -> {}
        {n2} -> {n5}
        {n3} -> {n7}
      provenance:
        none |}]

let%expect_test "sink_permute: numeric equivalence under broadcasting" =
  (* The commutation argument's load-bearing case: [b]'s pre-permute shape
     broadcasts against [a]'s, so after the shared perm their extent-1 axes
     land in different physical slots. Only the numbers confirm the rewrite
     still broadcasts correctly, the same reason reshape_to_permute gets a
     numeric test above. *)
  let g = Graph_fixtures.sink_permute_broadcast () in
  let a =
    Tensor.materialize (Graph_fixtures.nhwc ~h:2 ~w:3 ~c:4) (fun c ->
        float_of_int
          ((Dim.to_int (Vec6.get c Axis.H) * 12)
          + (Dim.to_int (Vec6.get c Axis.W) * 4)
          + Dim.to_int (Vec6.get c Axis.C)))
  in
  let b =
    Tensor.materialize (Graph_fixtures.nhwc ~h:1 ~w:1 ~c:4) (fun c ->
        float_of_int (Dim.to_int (Vec6.get c Axis.C)))
  in
  Format.printf "before: %s@." (evaluated g [ a; b ]);
  (match rewritten g [ Pass.fixpoint Sink_permute.pass ] with
  | None -> Format.printf "rewrite failed@."
  | Some g' ->
      Format.printf "after:  %s@." (evaluated g' [ a; b ]);
      (* [evaluated] prints only [Tensor.pp]'s first 8 of this fixture's 24
         elements — the real check is every coordinate, via [output_tensor]/
         [same_tensor] on the packed results directly. *)
      let same =
        match (output_tensor g [ a; b ], output_tensor g' [ a; b ]) with
        | Some before, Some after -> same_tensor before after
        | _ -> false
      in
      Format.printf "same:   %b@." same);
  [%expect
    {|
    before: tensor f32 [H=2 W=3 C=4] {0, 2, 4, 6, 4, 6, 8, 10, ...}
    after:  tensor f32 [H=2 W=3 C=4] {0, 2, 4, 6, 4, 6, 8, 10, ...}
    same:   true |}]
