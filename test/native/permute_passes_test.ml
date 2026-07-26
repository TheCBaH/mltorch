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

let evaluated g input =
  match
    Eval_direct.run g ~inputs:(List.combine g.Graph_ir.Graph.inputs [ input ])
  with
  | Error e -> Format.asprintf "%a" Eval_direct.pp_error e.Core.Error.kind
  | Ok env -> (
      match g.Graph_ir.Graph.outputs with
      | [ out ] -> Format.asprintf "%a" Tensor.pp (Tensor_id.Map.find out env)
      | _ -> "expected exactly one output")

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
      inputs: [t0 f32 [H=2 W=3 C=4]]
      nodes:
        n0: [t1 f32 [H=2 W=3 C=4]] = permute x=t0 perm=[]
        n1: [t2 f32 [H=2 W=3 C=4]] = relu x=t1
      outputs: [t2 f32 [H=2 W=3 C=4]]
    after:
      graph
      inputs: [t0 f32 [H=2 W=3 C=4]]
      nodes:
        n1: [t2 f32 [H=2 W=3 C=4]] = relu x=t0
      outputs: [t2 f32 [H=2 W=3 C=4]]
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
      inputs: [t0 f32 [H=2 W=3 C=4]]
      nodes:
        n0: [t1 f32 [H=3 W=4 C=2]] = permute x=t0 perm=[H<-W, W<-C, C<-H]
        n1: [t2 f32 [H=2 W=3 C=4]] = permute x=t1 perm=[H<-C, W<-H, C<-W]
        n2: [t3 f32 [H=2 W=3 C=4]] = relu x=t2
      outputs: [t3 f32 [H=2 W=3 C=4]]
    after:
      graph
      inputs: [t0 f32 [H=2 W=3 C=4]]
      nodes:
        n2: [t3 f32 [H=2 W=3 C=4]] = relu x=t0
      outputs: [t3 f32 [H=2 W=3 C=4]]
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
      inputs: [t0 f32 [H=2 W=3 C=4]]
      nodes:
        n0: [t1 f32 [H=3 W=4 C=2]] = permute x=t0 perm=[H<-W, W<-C, C<-H]
        n1: [t2 f32 [H=2 W=3 C=4]] = permute x=t1 perm=[H<-C, W<-H, C<-W]
      outputs: [t2 f32 [H=2 W=3 C=4]] |}]

let%expect_test "trim_permute: the run cancelling is looked for at every edge" =
  run (Graph_fixtures.permute_partial_cancel ()) [ Trim_permute.pass ];
  [%expect
    {|
    before:
      graph
      inputs: [t0 f32 [H=2 W=3 C=4]]
      nodes:
        n0: [t1 f32 [H=3 W=4 C=2]] = permute x=t0 perm=[H<-W, W<-C, C<-H]
        n1: [t2 f32 [H=2 W=3 C=4]] = permute x=t1 perm=[H<-C, W<-H, C<-W]
        n2: [t3 f32 [H=3 W=2 C=4]] = permute x=t2 perm=[H<-W, W<-H]
        n3: [t4 f32 [H=3 W=2 C=4]] = relu x=t3
      outputs: [t4 f32 [H=3 W=2 C=4]]
    after:
      graph
      inputs: [t0 f32 [H=2 W=3 C=4]]
      nodes:
        n2: [t3 f32 [H=3 W=2 C=4]] = permute x=t0 perm=[H<-W, W<-H]
        n3: [t4 f32 [H=3 W=2 C=4]] = relu x=t3
      outputs: [t4 f32 [H=3 W=2 C=4]]
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
      inputs: [t0 f32 [H=2 W=3 C=4]]
      nodes:
        n1: [t2 f32 [H=2 W=3 C=4]] = permute x=t0 perm=[]
        n2: [t3 f32 [H=2 W=3 C=4]] = permute x=t2 perm=[]
        n3: [t4 f32 [H=2 W=3 C=4]] = relu x=t3
      outputs: [t4 f32 [H=2 W=3 C=4]]
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
      inputs: [t0 f32 [H=2 W=3 C=4]]
      nodes:
        n3: [t4 f32 [H=2 W=3 C=4]] = relu x=t0
      outputs: [t4 f32 [H=2 W=3 C=4]]
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
      inputs: [t0 f32 [H=2 W=3 C=4]]
      nodes:
        n0: [t1 f32 [H=3 W=2 C=4]] = permute x=t0 perm=[H<-W, W<-H]
        n1: [t2 f32 [H=3 W=4 C=2]] = permute x=t1 perm=[W<-C, C<-W]
        n2: [t3 f32 [H=3 W=4 C=2]] = relu x=t2
      outputs: [t3 f32 [H=3 W=4 C=2]]
    after:
      graph
      inputs: [t0 f32 [H=2 W=3 C=4]]
      nodes:
        n3: [t2 f32 [H=3 W=4 C=2]] = permute x=t0 perm=[H<-W, W<-C, C<-H]
        n2: [t3 f32 [H=3 W=4 C=2]] = relu x=t2
      outputs: [t3 f32 [H=3 W=4 C=2]]
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
      inputs: [t0 f32 [H=2 W=3 C=4]]
      nodes:
        n3: [t4 f32 [H=2 W=3 C=4]] = relu x=t0
      outputs: [t4 f32 [H=2 W=3 C=4]]
    map:
      values:
        {t0, t3} -> {t0} identical
        {t1} -> {} identical
        {t2} -> {} identical
      nodes:
        {n0, n1, n2} -> {}
      provenance:
        none |}]

(* ---- reshape as relabelling ---------------------------------------------- *)

let%expect_test "reshape_to_permute: a pure relabelling" =
  (* C=6 becomes H=6 with nothing else non-unit, so the row-major order is
     untouched and the reshape is the permutation H<-C. *)
  run (Graph_fixtures.reshape_relabel ()) [ Reshape_to_permute.pass ];
  [%expect
    {|
    before:
      graph
      inputs: [t0 f32 [C=6]]
      nodes:
        n0: [t1 f32 [H=6 W=1 C=1]] = reshape x=t0 params={shape=[H=6 W=1 C=1]}
        n1: [t2 f32 [H=6 W=1 C=1]] = relu x=t1
      outputs: [t2 f32 [H=6 W=1 C=1]]
    after:
      graph
      inputs: [t0 f32 [C=6]]
      nodes:
        n2: [t1 f32 [H=6 W=1 C=1]] = permute x=t0 perm=[H<-C, W<-H, C<-W]
        n1: [t2 f32 [H=6 W=1 C=1]] = relu x=t1
      outputs: [t2 f32 [H=6 W=1 C=1]]
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
    match
      Graph_builder.build ~name:"relabel2"
        ~outputs:(fun o -> [ o ])
        Graph_builder.(
          let* x = input ~shape:(Graph_fixtures.s 1 1 1 1 2 3) () in
          reshape { Reshape.Reshape.shape = Graph_fixtures.s 1 1 1 2 3 1 } x)
    with
    | Ok g -> g
    | Error e ->
        invalid_arg
          (Format.asprintf "%a" Graph_builder.pp_error e.Core.Error.kind)
  in
  let input =
    Tensor.materialize (Graph_fixtures.s 1 1 1 1 2 3) (fun c ->
        float_of_int
          ((Dim.to_int (Vec6.get c Axis.W) * 3) + Dim.to_int (Vec6.get c Axis.C)))
  in
  let before = evaluated g input in
  Format.printf "reshape:  %s@." before;
  (match rewritten g [ Reshape_to_permute.pass ] with
  | None -> Format.printf "rewrite failed@."
  | Some g' ->
      Format.printf "@[<v 2>as permute:@,%a@]@." Graph_ir.pp g';
      let after = evaluated g' input in
      Format.printf "permute:  %s@." after;
      Format.printf "same:     %b@." (String.equal before after));
  [%expect
    {|
    reshape:  tensor f32 [H=2 W=3 C=1] {0, 1, 2, 3, 4, 5}
    as permute:
      graph
      inputs: [t0 f32 [W=2 C=3]]
      nodes:
        n1: [t1 f32 [H=2 W=3 C=1]] = permute x=t0 perm=[H<-W, W<-C, C<-H]
      outputs: [t1 f32 [H=2 W=3 C=1]]
    permute:  tensor f32 [H=2 W=3 C=1] {0, 1, 2, 3, 4, 5}
    same:     true |}]

let%expect_test "reshape_to_permute feeds the permute passes" =
  (* The point of the conversion: a reshape is opaque to fusion, a permute is
     not. Here the relabelling cancels against the permute above it, and the
     pair disappears. *)
  let g =
    match
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
    with
    | Ok g -> g
    | Error e ->
        invalid_arg
          (Format.asprintf "%a" Graph_builder.pp_error e.Core.Error.kind)
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
      inputs: [t0 f32 [C=6]]
      nodes:
        n0: [t1 f32 [H=6 W=1 C=1]] = reshape x=t0 params={shape=[H=6 W=1 C=1]}
        n1: [t2 f32 [C=6]] = permute x=t1 perm=[H<-W, W<-C, C<-H]
        n2: [t3 f32 [C=6]] = relu x=t2
      outputs: [t3 f32 [C=6]]
    after:
      graph
      inputs: [t0 f32 [C=6]]
      nodes:
        n2: [t3 f32 [C=6]] = relu x=t0
      outputs: [t3 f32 [C=6]]
    map:
      values:
        {t0, t2} -> {t0} identical
        {t1} -> {} identical
      nodes:
        {n0, n1} -> {}
      provenance:
        none |}]
