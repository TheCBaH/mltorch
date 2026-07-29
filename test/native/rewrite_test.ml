(* Exercises the rewriter: the four transformation kinds from hand-written
   recipes, every guard [apply] enforces, and the mapping it produces.
   See .ai/native_transform_design.md §5-§8. *)

open Graph_ir

let n_ n = Node_id.of_int n
let t_ n = Tensor_id.of_int n
let shape ~h ~w ~c = Vec6.shape ~n:1 ~t:1 ~d:1 ~h ~w ~c
let fail_with e = Format.printf "%a@." Rewrite.pp_error e.Core.Error.kind

(* The step's new state carries an existential version, so everything a test
   wants to see has to be consumed inside the branch that unpacks it. *)
let rewrite ?constants ?(show_before = false) g builder =
  match Rewrite.origin ?constants g with
  | Error e -> fail_with e
  | Ok (Rewrite.Origin state) -> (
      if show_before then
        Format.printf "@[<v 2>before:@,%a@]@." Graph_ir.pp (Rewrite.graph state);
      match Rewrite.plan state (Rewrite.allocator state) builder with
      | Error e -> fail_with e
      | Ok (recipe, _) -> (
          Format.printf "@[<v 2>recipe:@,%a@]@." Rewrite.pp_recipe recipe;
          match Rewrite.apply state recipe with
          | Error e -> fail_with e
          | Ok (Rewrite.Step (next, map)) ->
              Format.printf "@[<v 2>after:@,%a@]@." Graph_ir.pp
                (Rewrite.graph next);
              Format.printf "@[<v 2>map:@,%a@]@." Graph_map.pp map))

(* For the guards: the recipe itself is noise, only the verdict matters. *)
let rejected g builder =
  match Rewrite.origin g with
  | Error e -> fail_with e
  | Ok (Rewrite.Origin state) -> (
      match Rewrite.plan state (Rewrite.allocator state) builder with
      | Error e -> fail_with e
      | Ok (recipe, _) -> (
          match Rewrite.apply state recipe with
          | Error e -> fail_with e
          | Ok (Rewrite.Step _) -> Format.printf "accepted@."))

(* ---- trivial trimming ---------------------------------------------------- *)

let%expect_test "trim: removing a no-op permute ties its output to its input" =
  rewrite ~show_before:true
    (Graph_fixtures.permute_noop ())
    (Recipe.trim ~remove:[ n_ 0 ] ~tie:[ (t_ 1, t_ 0) ]);
  [%expect
    {|
    before:
      graph
      inputs: [t0 f32 [H=2 W=3 C=4] ->[n0]]
      nodes:
        n0: [t1 f32 [H=2 W=3 C=4] ->[n1]] = permute x=t0 perm=[]
        n1: [t2 f32 [H=2 W=3 C=4]] = relu x=t1 <-n0
      outputs: [t2 f32 [H=2 W=3 C=4] <-n1]
    recipe:
      remove: [n0]
      subst:
        t1 := t0
      claims:
        t1 -> t0 identical
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

let%expect_test "trim: a chain of two permutes collapses into one cluster" =
  (* Two independently matched no-ops give t2 := t1 and t1 := t0; normalisation
     has to resolve t2 all the way to t0, and the mapping has to close over the
     surviving source so the cluster is {t0,t1,t2} rather than {t1,t2}.

     IDENTITY permutes, not a cancelling pair: [permute_sequence]'s intermediate
     is a real rearrangement with a rotated shape, and tying it to the input is
     a claim [Graph_map.create] now rejects outright — corresponding shapes must
     agree. That rejection is correct, and it is not what this test is about. *)
  rewrite
    (Graph_fixtures.permute_identity_chain ())
    Recipe.(
      let* () = trim ~remove:[ n_ 0 ] ~tie:[ (t_ 1, t_ 0) ] in
      trim ~remove:[ n_ 1 ] ~tie:[ (t_ 2, t_ 1) ]);
  [%expect
    {|
    recipe:
      remove: [n0]
      subst:
        t1 := t0
      claims:
        t1 -> t0 identical
      remove: [n1]
      subst:
        t2 := t1
      claims:
        t2 -> t1 identical
    after:
      graph
      inputs: [t0 f32 [H=2 W=3 C=4] ->[n2]]
      nodes:
        n2: [t3 f32 [H=2 W=3 C=4] ->[n3]] = permute x=t0 perm=[]
        n3: [t4 f32 [H=2 W=3 C=4]] = relu x=t3 <-n2
      outputs: [t4 f32 [H=2 W=3 C=4] <-n3]
    map:
      values:
        {t0, t1, t2} -> {t0} identical
      nodes:
        {n0} -> {}
        {n1} -> {}
      provenance:
        none |}]

(* ---- decomposition and fusion -------------------------------------------- *)

let%expect_test "decompose: one node becomes two, the output id is preserved" =
  rewrite
    (Graph_fixtures.permute_noop ())
    Recipe.(
      (* relu -> relu(relu), an arbitrary 1:2 split keeping the final edge. *)
      let* mid = fresh (shape ~h:2 ~w:3 ~c:4) in
      let* out = fresh (shape ~h:2 ~w:3 ~c:4) in
      replace
        ~remove:[ n_ 1 ]
        ~insert:
          [
            { op = Relu { x = t_ 1 }; outputs = [ mid ]; from = [ n_ 1 ] };
            { op = Relu { x = mid }; outputs = [ out ]; from = [ n_ 1 ] };
          ]
        ~subst:[ (out, t_ 2) ]
        ());
  [%expect
    {|
    recipe:
      remove: [n1]
      insert:
        +0: [t3] = relu x=t1 from=[n1]
        +1: [t4] = relu x=t3 from=[n1]
      subst:
        t4 := t2
      claims:
        t2 -> t2 identical
    after:
      graph
      inputs: [t0 f32 [H=2 W=3 C=4] ->[n0]]
      nodes:
        n0: [t1 f32 [H=2 W=3 C=4] ->[n2]] = permute x=t0 perm=[]
        n2: [t3 f32 [H=2 W=3 C=4] ->[n3]] = relu x=t1 <-n0
        n3: [t2 f32 [H=2 W=3 C=4]] = relu x=t3 <-n2
      outputs: [t2 f32 [H=2 W=3 C=4] <-n3]
    map:
      values:
        {} -> {t3} identical
      nodes:
        {n1} -> {n2, n3}
      provenance:
        none |}]

let%expect_test "fuse: two nodes become one" =
  rewrite
    (Graph_fixtures.permute_sequence ())
    Recipe.(
      let* out = fresh (shape ~h:2 ~w:3 ~c:4) in
      replace
        ~remove:[ n_ 0; n_ 1 ]
        ~insert:
          [
            {
              op = Permute { perm = Graph_fixtures.identity_perm; x = t_ 0 };
              outputs = [ out ];
              from = [ n_ 0; n_ 1 ];
            };
          ]
        ~subst:[ (out, t_ 2) ]
        ());
  [%expect
    {|
    recipe:
      remove: [n0, n1]
      insert:
        +0: [t4] = permute x=t0 perm=[] from=[n0, n1]
      subst:
        t4 := t2
      claims:
        t2 -> t2 identical
    after:
      graph
      inputs: [t0 f32 [H=2 W=3 C=4] ->[n3]]
      nodes:
        n3: [t2 f32 [H=2 W=3 C=4] ->[n2]] = permute x=t0 perm=[]
        n2: [t3 f32 [H=2 W=3 C=4]] = relu x=t2 <-n3
      outputs: [t3 f32 [H=2 W=3 C=4] <-n2]
    map:
      values:
        {t1} -> {} identical
      nodes:
        {n0, n1} -> {n3}
      provenance:
        none |}]

(* ---- placement ----------------------------------------------------------- *)

let%expect_test "placement: a rewrite spanning two groups lands in their parent"
    =
  rewrite ~show_before:true
    (Graph_fixtures.grouped ())
    Recipe.(
      let* out = fresh (shape ~h:2 ~w:3 ~c:4) in
      replace
        ~remove:[ n_ 0; n_ 1 ]
        ~insert:
          [
            {
              op = Permute { perm = Graph_fixtures.identity_perm; x = t_ 0 };
              outputs = [ out ];
              from = [ n_ 0; n_ 1 ];
            };
          ]
        ~subst:[ (out, t_ 2) ]
        ());
  [%expect
    {|
    before:
      graph
      inputs: [t0 f32 [H=2 W=3 C=4] ->[n0]]
      nodes:
        group g1 first:
          n0: [t1 f32 [H=3 W=4 C=2] ->[n1]] =
            permute x=t0 perm=[H<-W, W<-C, C<-H]
        group g2 second:
          n1: [t2 f32 [H=2 W=3 C=4] ->[n2]] =
            permute x=t1 <-n0 perm=[H<-C, W<-H, C<-W]
        n2: [t3 f32 [H=2 W=3 C=4]] = relu x=t2 <-n1
      outputs: [t3 f32 [H=2 W=3 C=4] <-n2]
    recipe:
      remove: [n0, n1]
      insert:
        +0: [t4] = permute x=t0 perm=[] from=[n0, n1]
      subst:
        t4 := t2
      claims:
        t2 -> t2 identical
    after:
      graph
      inputs: [t0 f32 [H=2 W=3 C=4] ->[n3]]
      nodes:
        n3: [t2 f32 [H=2 W=3 C=4] ->[n2]] = permute x=t0 perm=[]
        n2: [t3 f32 [H=2 W=3 C=4]] = relu x=t2 <-n3
      outputs: [t3 f32 [H=2 W=3 C=4] <-n2]
    map:
      values:
        {t1} -> {} identical
      nodes:
        {n0, n1} -> {n3}
      provenance:
        none |}]

let%expect_test "placement: a replacement inside one group stays there" =
  rewrite
    (Graph_fixtures.grouped ())
    Recipe.(
      let* out = fresh (shape ~h:3 ~w:4 ~c:2) in
      replace
        ~remove:[ n_ 0 ]
        ~insert:
          [
            {
              op = Permute { perm = Graph_fixtures.rotate_hwc; x = t_ 0 };
              outputs = [ out ];
              from = [ n_ 0 ];
            };
          ]
        ~subst:[ (out, t_ 1) ]
        ());
  [%expect
    {|
    recipe:
      remove: [n0]
      insert:
        +0: [t4] = permute x=t0 perm=[H<-W, W<-C, C<-H] from=[n0]
      subst:
        t4 := t1
      claims:
        t1 -> t1 identical
    after:
      graph
      inputs: [t0 f32 [H=2 W=3 C=4] ->[n3]]
      nodes:
        group g1 first:
          n3: [t1 f32 [H=3 W=4 C=2] ->[n1]] =
            permute x=t0 perm=[H<-W, W<-C, C<-H]
        group g2 second:
          n1: [t2 f32 [H=2 W=3 C=4] ->[n2]] =
            permute x=t1 <-n3 perm=[H<-C, W<-H, C<-W]
        n2: [t3 f32 [H=2 W=3 C=4]] = relu x=t2 <-n1
      outputs: [t3 f32 [H=2 W=3 C=4] <-n2]
    map:
      values:
        identity
      nodes:
        {n0} -> {n3}
      provenance:
        none |}]

let%expect_test "placement: New_group wraps the insertions in a fresh child" =
  rewrite
    (Graph_fixtures.permute_sequence ())
    Recipe.(
      let* out = fresh (shape ~h:2 ~w:3 ~c:4) in
      replace
        ~remove:[ n_ 0; n_ 1 ]
        ~insert:
          [
            {
              op = Permute { perm = Graph_fixtures.identity_perm; x = t_ 0 };
              outputs = [ out ];
              from = [ n_ 0; n_ 1 ];
            };
          ]
        ~subst:[ (out, t_ 2) ]
        ~placement:(New_group (Some "fused")) ());
  [%expect
    {|
    recipe:
      remove: [n0, n1]
      insert:
        +0: [t4] = permute x=t0 perm=[] from=[n0, n1]
      subst:
        t4 := t2
      claims:
        t2 -> t2 identical
    after:
      graph
      inputs: [t0 f32 [H=2 W=3 C=4] ->[n3]]
      nodes:
        group g1 fused:
          n3: [t2 f32 [H=2 W=3 C=4] ->[n2]] = permute x=t0 perm=[]
        n2: [t3 f32 [H=2 W=3 C=4]] = relu x=t2 <-n3
      outputs: [t3 f32 [H=2 W=3 C=4] <-n2]
    map:
      values:
        {t1} -> {} identical
      nodes:
        {n0, n1} -> {n3}
      provenance:
        none |}]

(* ---- the guards ---------------------------------------------------------- *)

let raw r = Recipe.emit { Recipe.empty_replacement with Recipe.remove = r }

let%expect_test "guard: a substitution whose source survives needs a claim" =
  rejected
    (Graph_fixtures.permute_noop ())
    (Recipe.emit
       {
         Recipe.empty_replacement with
         remove = Node_id.Set.singleton (n_ 0);
         subst = Tensor_id.Map.singleton (t_ 1) (t_ 0);
       });
  [%expect {| t1 is substituted away without a value claim |}]

let%expect_test "guard: a preserved id whose definition changed needs a claim" =
  (* Swapping relu for a permute behind the same output id. The substitution
     check cannot see this — the replacement names no substitution at all — so
     the redefinition check is what catches it. *)
  rejected
    (Graph_fixtures.permute_noop ())
    (Recipe.emit
       {
         Recipe.empty_replacement with
         remove = Node_id.Set.singleton (n_ 1);
         insert =
           [
             {
               Recipe.op =
                 Permute { perm = Graph_fixtures.identity_perm; x = t_ 1 };
               outputs = [ t_ 2 ];
               from = [ n_ 1 ];
             };
           ];
       });
  [%expect {| t2 is kept but redefined without a value claim |}]

let%expect_test "guard: keeping an id whose signature changed is rejected" =
  (* The replacement is internally consistent — a rotate really does produce
     [H=3 W=4 C=2] — so the ONLY thing wrong is that it takes over t2, whose
     tensor was [H=2 W=3 C=4]. An id may be kept only for the exact same tensor,
     and no claim excuses otherwise. *)
  rejected
    (Graph_fixtures.permute_noop ())
    Recipe.(
      let* out = fresh (shape ~h:3 ~w:4 ~c:2) in
      replace
        ~remove:[ n_ 1 ]
        ~insert:
          [
            {
              op = Permute { perm = Graph_fixtures.rotate_hwc; x = t_ 1 };
              outputs = [ out ];
              from = [ n_ 1 ];
            };
          ]
        ~subst:[ (out, t_ 2) ]
        ~claims:[ (t_ 2, t_ 2, Correspondence.Identical) ]
        ());
  [%expect {| t2 is kept but its signature changed; that needs a new id |}]

let%expect_test "guard: an output signature contradicting its op is rejected" =
  (* Here the recipe declares no signature, so t2 keeps the old one while the
     node under it now produces a different shape. Arity alone would not notice
     and the graph would only fail at evaluation. *)
  rejected
    (Graph_fixtures.permute_noop ())
    Recipe.(
      replace
        ~remove:[ n_ 1 ]
        ~insert:
          [
            {
              op = Permute { perm = Graph_fixtures.rotate_hwc; x = t_ 1 };
              outputs = [ t_ 2 ];
              from = [ n_ 1 ];
            };
          ]
        ~claims:[ (t_ 2, t_ 2, Correspondence.Identical) ]
        ());
  [%expect {| t2's signature is not the shape its op produces |}]

let%expect_test "guard: a bound payload must match the signature it declares" =
  (* A pass that computes a parameter itself — batch-norm folding is the first —
     binds data the framework has no other way to check. Without this the
     mismatch surfaces only at evaluation, a long way from the recipe that
     caused it. The edge here is declared [C=3] and given a [C=4] tensor. *)
  rejected
    (Graph_fixtures.permute_noop ())
    Recipe.(
      let* id = fresh (shape ~h:1 ~w:1 ~c:3) in
      emit
        {
          empty_replacement with
          constants =
            [ (id, Tensor.materialize (shape ~h:1 ~w:1 ~c:4) (fun _ -> 1.)) ];
        });
  [%expect {| payload for t3 does not match its signature |}]

let%expect_test "guard: two replacements claiming the same node is rejected" =
  rejected
    (Graph_fixtures.permute_sequence ())
    Recipe.(
      let* () = trim ~remove:[ n_ 0 ] ~tie:[ (t_ 1, t_ 0) ] in
      trim ~remove:[ n_ 0 ] ~tie:[ (t_ 1, t_ 0) ]);
  [%expect {| two replacements both claim node n0 |}]

let%expect_test "guard: an unknown node is rejected" =
  rejected
    (Graph_fixtures.permute_noop ())
    (raw (Node_id.Set.singleton (n_ 42)));
  [%expect {| unknown node n42 |}]

let%expect_test "guard: a substitution cycle is rejected" =
  rejected
    (Graph_fixtures.permute_sequence ())
    Recipe.(
      let* () = trim ~remove:[ n_ 0 ] ~tie:[ (t_ 1, t_ 2) ] in
      trim ~remove:[ n_ 1 ] ~tie:[ (t_ 2, t_ 1) ]);
  [%expect {| substitution of t2 is cyclic |}]

let%expect_test "guard: a recipe planned past the state's watermark is rejected"
    =
  (* Applying a recipe to a LATER version is a compile error — [apply] demands
     the state's version and [Step] binds a fresh one — so the runtime check
     covers what types cannot: a second recipe planned from the first's advanced
     allocator, then applied to the original state. Its fresh ids assume the
     first recipe already ran. *)
  (match Rewrite.origin (Graph_fixtures.permute_sequence ()) with
  | Error e -> fail_with e
  | Ok (Rewrite.Origin state) -> (
      (* Both recipes must actually allocate: an allocation-free one leaves the
         supply untouched, so its watermarks legitimately still match. *)
      let relu_over ~out_shape ~node ~operand ~onto =
        Recipe.(
          let* out = fresh out_shape in
          replace ~remove:[ node ]
            ~insert:
              [
                {
                  op = Relu { x = operand };
                  outputs = [ out ];
                  from = [ node ];
                };
              ]
            ~subst:[ (out, onto) ]
            ())
      in
      match
        Rewrite.plan state (Rewrite.allocator state)
          (relu_over ~out_shape:(shape ~h:2 ~w:3 ~c:4) ~node:(n_ 2)
             ~operand:(t_ 2) ~onto:(t_ 3))
      with
      | Error e -> fail_with e
      | Ok (_, advanced) -> (
          match
            Rewrite.plan state advanced
              (relu_over ~out_shape:(shape ~h:2 ~w:3 ~c:4) ~node:(n_ 1)
                 ~operand:(t_ 1) ~onto:(t_ 2))
          with
          | Error e -> fail_with e
          | Ok (recipe, _) -> (
              match Rewrite.apply state recipe with
              | Error e -> fail_with e
              | Ok (Rewrite.Step _) -> Format.printf "accepted@."))));
  [%expect {| the recipe was planned against a different state |}]

let%expect_test "guard: merging recipes from a branched allocator is rejected" =
  (* Allocators are immutable, so planning twice from the same one gives two
     recipes with the same start watermark and overlapping fresh ids. *)
  (match Rewrite.origin (Graph_fixtures.permute_sequence ()) with
  | Error e -> fail_with e
  | Ok (Rewrite.Origin state) -> (
      let alloc = Rewrite.allocator state in
      let plan builder = Rewrite.plan state alloc builder in
      let relu_over ~node ~operand ~onto =
        Recipe.(
          let* out = fresh (shape ~h:2 ~w:3 ~c:4) in
          replace ~remove:[ node ]
            ~insert:
              [
                {
                  op = Relu { x = operand };
                  outputs = [ out ];
                  from = [ node ];
                };
              ]
            ~subst:[ (out, onto) ]
            ())
      in
      match
        ( plan (relu_over ~node:(n_ 2) ~operand:(t_ 2) ~onto:(t_ 3)),
          plan (relu_over ~node:(n_ 1) ~operand:(t_ 1) ~onto:(t_ 2)) )
      with
      | Ok (a, _), Ok (b, _) -> (
          match Rewrite.merge a b with
          | Error e -> fail_with e
          | Ok _ -> Format.printf "accepted@.")
      | Error e, _ | _, Error e -> fail_with e));
  [%expect {| recipes were planned from a branched allocator |}]
