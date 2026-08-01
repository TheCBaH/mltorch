(* The first real transformations: dropping cancelling permutes, fusing adjacent
   ones, and recognising a reshape that is only relabelling axes.
   See .ai/native_transform_design.md §12.

   Each test prints the source graph, the result, and the mapping — the mapping
   being the part a numerical checker will consume, so it is pinned as carefully
   as the graph. *)

let run ?(show_before = true) g passes =
  match Rewrite.origin g with
  | Error e -> Format.printf "origin: %a@." Rewrite.pp_error e.Core.Error.kind
  | Ok (Rewrite.Origin state) -> (
      if show_before then
        Format.printf "@[<v 2>before:@,%a@]@." Graph_ir.pp (Rewrite.graph state);
      match Pass.run_all state passes with
      | Error e -> Format.printf "%a@." Pass.pp_error e.Core.Error.kind
      | Ok (Rewrite.Step (final, map)) ->
          Format.printf "@[<v 2>after:@,%a@]@." Graph_ir.pp
            (Rewrite.graph final);
          Format.printf "@[<v 2>map:@,%a@]@." Graph_map.pp map)

(* Quiet form, for the numeric checks where the graphs are a means and the
   values are the point. *)
let rewritten g passes =
  match Rewrite.origin g with
  | Error _ -> None
  | Ok (Rewrite.Origin state) -> (
      match Pass.run_all state passes with
      | Error _ -> None
      | Ok (Rewrite.Step (final, _)) -> Some (Rewrite.graph final))

(* [inputs] pairs positionally with [g.Graph_ir.Graph.inputs], so a fixture
   with N declared inputs needs N tensors here — [reshape_to_permute]'s numeric
   test passes one, [sink_permute_broadcast]'s passes two independently
   shaped ones. *)
let evaluated g inputs =
  match
    Eval_direct.run g ~inputs:(List.combine g.Graph_ir.Graph.inputs inputs)
  with
  | Error e -> Format.asprintf "%a" Eval_direct.pp_error e.Core.Error.kind
  | Ok env -> (
      match g.Graph_ir.Graph.outputs with
      | [ out ] -> Format.asprintf "%a" Tensor.pp (Tensor_id.Map.find out env)
      | _ -> "expected exactly one output")

(* The single-output tensor itself, for a comparison that isn't limited to
   [Tensor.pp]'s abbreviated first-8-elements printout. *)
let output_tensor g inputs =
  match
    Eval_direct.run g ~inputs:(List.combine g.Graph_ir.Graph.inputs inputs)
  with
  | Error _ -> None
  | Ok env -> (
      match g.Graph_ir.Graph.outputs with
      | [ out ] -> Some (Tensor_id.Map.find out env)
      | _ -> None)

(* Every coordinate, not just [Tensor.pp]'s truncated preview. *)
let same_tensor (Tensor.Tensor a) (Tensor.Tensor b) =
  Stdlib.( = ) a.Tensor.shape b.Tensor.shape
  &&
  let same = ref true in
  Vec6.iter a.Tensor.shape (fun c ->
      if
        not
          (Float.equal
             (Tensor.read (Tensor.Tensor a) c)
             (Tensor.read (Tensor.Tensor b) c))
      then same := false);
  !same

(* What a pass matches, independent of what it then builds. *)
let matches pattern g =
  match Graph_view.of_graph g with
  | Error e -> Format.printf "view: %a@." Graph_view.pp_error e.Core.Error.kind
  | Ok view ->
      let found = Pattern.scan pattern view in
      if found = [] then Format.printf "no match@."
      else
        List.iter
          (fun (_, region) ->
            Format.printf "@[<v 2>matched %a:@,%a@]@." Region.pp region
              Graph_ir.pp
              (Region.extract view region))
          found

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

let%expect_test "sink_permute: every accepted op sinks" =
  run ~show_before:false
    (Graph_fixtures.sink_permute_allowlist ())
    [ Pass.fixpoint Sink_permute.pass ];
  [%expect
    {|
    after:
      graph
      inputs:
        [t0 f32 [H=2 W=3 C=4] ->[n36, n38, n40, n42, n44, n46, n48, n50, n52,
                                 n54, n56],
         t1 f32 [H=2 W=3 C=4] ->[n50, n52, n54, n56]]
      nodes:
        n36: [t38 f32 [H=2 W=3 C=4] ->[n62]] = relu x=t0
        n38: [t39 f32 [H=2 W=3 C=4] ->[n62]] = sqrt x=t0
        n40: [t40 f32 [H=2 W=3 C=4] ->[n64]] = clone x=t0
        n42: [t41 f32 [H=2 W=3 C=4] ->[n64]] = add_scalar x=t0 scalar=3
        n44: [t42 f32 [H=2 W=3 C=4] ->[n66]] = div_scalar x=t0 scalar=6
        n46: [t43 f32 [H=2 W=3 C=4] ->[n66]] = clamp x=t0 params={min=0; max=6}
        n48: [t44 f32 [H=2 W=3 C=4] ->[n72]] =
          hardtanh x=t0 params={min_val=0; max_val=6}
        n50: [t45 f32 [H=2 W=3 C=4] ->[n58]] = add a=t0 b=t1
        n52: [t46 f32 [H=2 W=3 C=4] ->[n58]] = sub a=t0 b=t1
        n54: [t47 f32 [H=2 W=3 C=4] ->[n60]] = mul a=t0 b=t1
        n56: [t48 f32 [H=2 W=3 C=4] ->[n60]] = div a=t0 b=t1
        n62: [t51 f32 [H=2 W=3 C=4] ->[n74]] = add a=t38 <-n36 b=t39 <-n38
        n64: [t52 f32 [H=2 W=3 C=4] ->[n70]] = add a=t40 <-n40 b=t41 <-n42
        n66: [t53 f32 [H=2 W=3 C=4] ->[n70]] = add a=t42 <-n44 b=t43 <-n46
        n58: [t49 f32 [H=2 W=3 C=4] ->[n68]] = add a=t45 <-n50 b=t46 <-n52
        n60: [t50 f32 [H=2 W=3 C=4] ->[n68]] = add a=t47 <-n54 b=t48 <-n56
        n70: [t55 f32 [H=2 W=3 C=4] ->[n72]] = add a=t52 <-n64 b=t53 <-n66
        n68: [t54 f32 [H=2 W=3 C=4] ->[n74]] = add a=t49 <-n58 b=t50 <-n60
        n72: [t56 f32 [H=2 W=3 C=4] ->[n76]] = add a=t55 <-n70 b=t44 <-n48
        n74: [t57 f32 [H=2 W=3 C=4] ->[n76]] = add a=t54 <-n68 b=t51 <-n62
        n76: [t58 f32 [H=2 W=3 C=4] ->[n77]] = add a=t57 <-n74 b=t56 <-n72
        n77: [t37 f32 [H=3 W=4 C=2]] =
          permute x=t58 <-n76 perm=[H<-W, W<-C, C<-H]
      outputs: [t37 f32 [H=3 W=4 C=2] <-n77]
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
        {} -> {t38} identical
        {} -> {t39} identical
        {} -> {t40} identical
        {} -> {t41} identical
        {} -> {t42} identical
        {} -> {t43} identical
        {} -> {t44} identical
        {} -> {t45} identical
        {} -> {t46} identical
        {} -> {t47} identical
        {} -> {t48} identical
        {} -> {t49} identical
        {} -> {t50} identical
        {} -> {t51} identical
        {} -> {t52} identical
        {} -> {t53} identical
        {} -> {t54} identical
        {} -> {t55} identical
        {} -> {t56} identical
        {} -> {t57} identical
        {} -> {t58} identical
      nodes:
        {n0, n2, n4, n6, n8, n10, n12, n14, n15, n17, n18, n20, n21, n23, n24} -> {n77}
        {n1} -> {n36}
        {n3} -> {n38}
        {n5} -> {n40}
        {n7} -> {n42}
        {n9} -> {n44}
        {n11} -> {n46}
        {n13} -> {n48}
        {n16} -> {n50}
        {n19} -> {n52}
        {n22} -> {n54}
        {n25} -> {n56}
        {n26} -> {n58}
        {n27} -> {n60}
        {n28} -> {n62}
        {n29} -> {n68}
        {n30} -> {n64}
        {n31} -> {n66}
        {n32} -> {n70}
        {n33} -> {n72}
        {n34} -> {n74}
        {n35} -> {n76}
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

(* ---- reshape as relabelling ---------------------------------------------- *)

let%expect_test "reshape_to_permute: a pure relabelling" =
  (* C=6 becomes H=6 with nothing else non-unit, so the row-major order is
     untouched and the reshape is the permutation H<-C. *)
  run (Graph_fixtures.reshape_relabel ()) [ Reshape_to_permute.pass ];
  [%expect
    {|
    before:
      graph
      inputs: [t0 f32 [C=6] ->[n0]]
      nodes:
        n0: [t1 f32 [H=6 W=1 C=1] ->[n1]] =
          reshape x=t0 params={shape=[H=6 W=1 C=1]}
        n1: [t2 f32 [H=6 W=1 C=1]] = relu x=t1 <-n0
      outputs: [t2 f32 [H=6 W=1 C=1] <-n1]
    after:
      graph
      inputs: [t0 f32 [C=6] ->[n2]]
      nodes:
        n2: [t1 f32 [H=6 W=1 C=1] ->[n1]] = permute x=t0 perm=[H<-C, W<-H, C<-W]
        n1: [t2 f32 [H=6 W=1 C=1]] = relu x=t1 <-n2
      outputs: [t2 f32 [H=6 W=1 C=1] <-n1]
    map:
      values:
        identity
      nodes:
        {n0} -> {n2}
      provenance:
        none |}]

let%expect_test "reshape_to_permute: a genuine flatten is left alone" =
  (* H=2,W=3 collapsing onto C=6 mixes extents; no permutation of the six axes
     computes it. *)
  matches Reshape_to_permute.pattern (Graph_fixtures.reshape_flatten ());
  [%expect {| no match |}]

let%expect_test "reshape_to_permute: the permute computes the same tensor" =
  (* The load-bearing check for this pass. Its legality argument is about
     row-major offsets, not a syntactic identity, so only the numbers confirm
     it — and this case moves TWO non-unit axes ([W=2 C=3] becoming [H=2 W=3]),
     where a plausible-but-wrong perm would still produce the right shape. *)
  let g =
    Graph_builder.build ~name:"relabel2"
      ~outputs:(fun o -> [ o ])
      Graph_builder.(
        let* x = input ~shape:(Graph_fixtures.s 1 1 1 1 2 3) () in
        reshape { Reshape.Reshape.shape = Graph_fixtures.s 1 1 1 2 3 1 } x)
    |> Core.or_raise Graph_builder.pp_error
  in
  let input =
    Tensor.materialize (Graph_fixtures.s 1 1 1 1 2 3) (fun c ->
        float_of_int
          ((Dim.to_int (Vec6.get c Axis.W) * 3) + Dim.to_int (Vec6.get c Axis.C)))
  in
  let before = evaluated g [ input ] in
  Format.printf "reshape:  %s@." before;
  (match rewritten g [ Reshape_to_permute.pass ] with
  | None -> Format.printf "rewrite failed@."
  | Some g' ->
      Format.printf "@[<v 2>as permute:@,%a@]@." Graph_ir.pp g';
      let after = evaluated g' [ input ] in
      Format.printf "permute:  %s@." after;
      Format.printf "same:     %b@." (String.equal before after));
  [%expect
    {|
    reshape:  tensor f32 [H=2 W=3 C=1] {0, 1, 2, 3, 4, 5}
    as permute:
      graph
      inputs: [t0 f32 [W=2 C=3] ->[n1]]
      nodes:
        n1: [t1 f32 [H=2 W=3 C=1]] = permute x=t0 perm=[H<-W, W<-C, C<-H]
      outputs: [t1 f32 [H=2 W=3 C=1] <-n1]
    permute:  tensor f32 [H=2 W=3 C=1] {0, 1, 2, 3, 4, 5}
    same:     true |}]

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
    |> Core.or_raise Graph_builder.pp_error
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
