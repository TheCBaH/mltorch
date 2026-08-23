(* Rewriting laws, over generated well-formed expressions.

   Results are pinned as a verdict per property rather than per case, in the
   style of native_walk_test.ml: a failure shows up as a golden diff naming the
   property and the seed that broke it, which is reproducible from the printed
   verdict alone. *)

open Expr

let cases = 400
let depth = 4

(* A pure environment: a source and coordinate map to a value, so the evaluator
   is exercised with no tensor, storage or graph type in sight -- which is the
   point of [Env.load] being a callback. *)
let env =
  {
    Eval.Env.load =
      (fun s c ->
        let mix =
          Coord.foldi (fun a acc i -> (acc * 7) + Axis.to_int a + i) 1 c
        in
        Err.return (float_of_int ((Source.to_int s * 13) + (mix mod 11)) /. 4.));
  }

let origin = Coord.of_fn (fun a -> Axis.to_int a mod 3)
let ev ?(output = origin) e = Eval.value env ~output e

(* Bits, not [=]: two NaNs must not silently agree, and a law that held only up
   to float equality would be a weaker claim than the one being made. *)
let same_value a b =
  match (a, b) with
  | Ok x, Ok y -> Core.Float_bits.equal_exact x y
  | Error _, Error _ -> true
  | _ -> false

let over_samples f =
  let st = Gen.create ~seed:20260803L in
  let rec go i failures =
    if i >= cases then failures
    else
      let e = Gen.expr st ~depth in
      go (i + 1) (if f e then failures else i :: failures)
  in
  go 0 []

let report name f =
  match over_samples f with
  | [] -> Fmt.pr "%-34s %d cases ok@." name cases
  | fs ->
      Fmt.pr "%-34s FAILED on case(s) %a@." name
        (Fmt.list ~sep:(Fmt.any ",") Fmt.int)
        (List.rev fs)

let%expect_test "generated samples are well-formed to begin with" =
  (* Without this the other properties could pass against malformed inputs, and
     a failure would be unattributable: an Expr defect and a generator defect
     would look the same. *)
  report "Check.value succeeds" (fun e ->
      match Check.value e with Ok () -> true | Error _ -> false);
  report "no free reducers" (fun e ->
      Reduce_var.Set.is_empty (Fold.free_reducers e));
  [%expect
    {|
    Check.value succeeds               400 cases ok
    no free reducers                   400 cases ok
    |}]

let%expect_test "freshening preserves interpretation" =
  report "eval (freshen e) = eval e" (fun e ->
      let f = Builder.run (Rewrite.freshen e) in
      same_value (ev f) (ev e));
  (* And structurally too: freshening is exactly a change of bound names, so an
     alpha-aware comparison must not be able to see it. *)
  report "freshen e = e up to alpha" (fun e ->
      Value.equal (Builder.run (Rewrite.freshen e)) e);
  report "and hashes agree" (fun e ->
      Value.hash (Builder.run (Rewrite.freshen e)) = Value.hash e);
  [%expect
    {|
    eval (freshen e) = eval e          400 cases ok
    freshen e = e up to alpha          400 cases ok
    and hashes agree                   400 cases ok
    |}]

let%expect_test "alpha-normalisation is idempotent and canonical" =
  report "normalize . normalize = normalize" (fun e ->
      let n = Rewrite.alpha_normalize e in
      Value.equal (Rewrite.alpha_normalize n) n
      && String.equal
           (Core.Pretty.to_string Pp.value (Rewrite.alpha_normalize n))
           (Core.Pretty.to_string Pp.value n));
  report "normalize preserves interpretation" (fun e ->
      same_value (ev (Rewrite.alpha_normalize e)) (ev e));
  [%expect
    {|
    normalize . normalize = normalize  400 cases ok
    normalize preserves interpretation 400 cases ok
    |}]

let%expect_test "substitution commutes with evaluation" =
  (* eval (substitute_output s e) o  =  eval e (eval_indices s o).
     The law that makes stage grounding sound: substituting a coordinate
     expression then evaluating must equal evaluating the coordinates first. *)
  let s =
    Coord.of_fn (fun a ->
        Index.clamp_low
          (Index.add
             (Index.of_position (Index.output a))
             (Index.const (Axis.to_int a))))
  in
  report "subst then eval = eval at image" (fun e ->
      let image =
        Coord.map
          (fun i ->
            match Eval.index ~output:origin ~reducers:(fun _ -> None) i with
            | Ok v -> v
            | Error _ -> 0)
          s
      in
      same_value
        (ev ~output:origin (Rewrite.substitute_output s e))
        (ev ~output:image e));
  [%expect {| subst then eval = eval at image    400 cases ok |}]

let%expect_test "source mapping changes only sources" =
  let bump s = Source.create (Source.to_int s + 10) in
  report "sources are remapped" (fun e ->
      let m = Rewrite.map_sources bump e in
      Source.Set.equal (Fold.sources m)
        (Source.Set.of_list
           (List.map bump (Source.Set.elements (Fold.sources e)))));
  report "shape is untouched" (fun e ->
      let m = Rewrite.map_sources bump e in
      Fold.size m = Fold.size e
      && Fold.depth m = Fold.depth e
      && Value.equal
           (Rewrite.map_sources bump (Rewrite.map_sources bump e))
           (Rewrite.map_sources (fun s -> bump (bump s)) e));
  [%expect
    {|
    sources are remapped               400 cases ok
    shape is untouched                 400 cases ok
    |}]

let%expect_test "compare = 0 implies equal hashes" =
  report "compare/hash agree" (fun e ->
      let n = Rewrite.alpha_normalize e in
      (not (Value.equal e n)) || Value.hash e = Value.hash n);
  [%expect {| compare/hash agree                 400 cases ok |}]

let%expect_test "freshening the inserted fragment repairs shadowing" =
  (* Two well-formed fragments from overlapping supplies, one placed inside the
     other. Neither references the other, so this shadows rather than captures:
     evaluation is unaffected, and the damage is structural. That is still a
     defect -- alpha-normalisation, comparison and every binder-counting fold
     stop meaning anything once one variable binds twice on a path. *)
  let a, b = Gen.colliding_pair () in
  (* Composition is itself a Builder computation, on ONE supply. Calling
     [Builder.run] partway through would restart at [initial] and mint a
     colliding outer binder too -- the same mistake, one level up. *)
  let compose ~freshen_first =
    let open Builder.Syntax in
    Builder.run
      (let* a' =
         if freshen_first then Rewrite.freshen a else Builder.return a
       in
       let* b' =
         if freshen_first then Rewrite.freshen b else Builder.return b
       in
       Builder.reduction ~kind:Reduction.Sum ~lo:Index.zero ~hi:(Index.const 2)
         (fun _ -> Builder.return (Value.add a' b')))
  in
  let unfreshened = compose ~freshen_first:false in
  let freshened = compose ~freshen_first:true in
  let verdict =
    Core.Pretty.err_result ~ok:(Fmt.any "ok") ~error:Check.pp_error
  in
  Fmt.pr "unfreshened: %a@." verdict (Check.value unfreshened);
  Fmt.pr "freshened:   %a@." verdict (Check.value freshened);
  [%expect
    {|
    unfreshened: reducer #0 is bound twice on one path
    freshened:   ok
    |}];
  (* The measurable symptom: three binder positions carrying one identity
     between them. The PRINTER does not show it -- its naming environment is
     scoped, so it gives the three distinct display names and renders the
     shadowing exactly as it resolves -- which is why the count is what this
     asserts on. *)
  let names e = List.length (Fold.binders e) in
  let distinct e =
    Reduce_var.Set.cardinal (Reduce_var.Set.of_list (Fold.binders e))
  in
  Fmt.pr "unfreshened binders %d, distinct %d@." (names unfreshened)
    (distinct unfreshened);
  Fmt.pr "freshened   binders %d, distinct %d@." (names freshened)
    (distinct freshened);
  [%expect
    {|
    unfreshened binders 3, distinct 1
    freshened   binders 3, distinct 3
    |}]

let%expect_test "capture that already happened cannot be freshened away" =
  (* The reason the rule is 'freshen the fragment BEFORE composing', not 'freshen
     the result'. Here the fragment binds a variable whose ordinal collides with
     the destination's AND references the destination's variable -- so at the
     moment it is built, the two references are already indistinguishable.

     Freshening afterwards renames the binder and every reference to it, which
     necessarily drags the outer reference along: there is no record of which
     binder it was supposed to mean. Both spellings below read the inner
     binder, and they evaluate identically -- the outer variable is simply
     unreachable. The two even PRINT identically, since the naming environment
     is scoped and resolves each reference the way evaluation does: freshening
     afterwards is not a partial repair, it is a no-op on the denotation. *)
  let build ~freshen_after =
    Builder.run
      (Builder.reduction ~kind:Reduction.Sum ~lo:Index.zero ~hi:(Index.const 2)
         (fun x ->
           let frag = Gen.capturing_fragment x in
           if freshen_after then Rewrite.freshen frag else Builder.return frag))
  in
  let before = build ~freshen_after:false in
  let after = build ~freshen_after:true in
  Fmt.pr "as built:  %a@." Pp.value before;
  Fmt.pr "freshened: %a@." Pp.value after;
  [%expect
    {|
    as built:  sum(r1=0..2: sum(r2=0..3: (value_of_index(r2) + value_of_index(r2))))
    freshened: sum(r1=0..2: sum(r2=0..3: (value_of_index(r2) + value_of_index(r2))))
    |}];
  Fmt.pr "still reads the inner binder: %b@."
    (same_value (ev before) (ev after));
  [%expect {| still reads the inner binder: true |}];
  (* The fix is not to create the collision: draw the fragment's variable from
     the destination's own supply, so the two are distinct from the start. *)
  let sound =
    Builder.run
      (Builder.reduction ~kind:Reduction.Sum ~lo:Index.zero ~hi:(Index.const 2)
         (fun x ->
           Builder.reduction ~kind:Reduction.Sum ~lo:Index.zero
             ~hi:(Index.const 3) (fun own ->
               Builder.return
                 (Value.add
                    (Value.value_of_index (Index.of_position x))
                    (Value.value_of_index (Index.of_position own))))))
  in
  Fmt.pr "one supply: %a@." Pp.value sound;
  [%expect
    {| one supply: sum(r1=0..2: sum(r2=0..3: (value_of_index(r1) + value_of_index(r2)))) |}];
  Fmt.pr "and it differs from the captured form: %b@."
    (not (same_value (ev sound) (ev before)));
  [%expect {| and it differs from the captured form: true |}]

let%expect_test "freshening must not capture a FREE reducer" =
  (* A fragment with a free reference -- to a binder that lives in whatever will
     receive it -- and one binder of its own. Freshening draws replacements from
     the supply, so unless it skips identities already free in the term, a
     replacement can land ON the free one and capture it.

     [alpha_normalize] makes this maximally likely: it always starts from
     [initial], so any free ordinal near zero is in the line of fire. *)
  let free, st = Builder.run_from Builder.initial Builder.fresh_reduce in
  let _, st = Builder.run_from st Builder.fresh_reduce in
  let frag =
    fst
      (Builder.run_from st
         (Builder.reduction ~kind:Reduction.Sum ~lo:Index.zero
            ~hi:(Index.const 2) (fun own ->
              Builder.return
                (Value.add
                   (Value.value_of_index
                      (Index.of_position (Index.reduce free)))
                   (Value.value_of_index (Index.of_position own))))))
  in
  let verdict =
    Core.Pretty.err_result ~ok:(Fmt.any "ok") ~error:Check.pp_error
  in
  Fmt.pr "before: %a@." verdict (Check.value frag);
  [%expect {| before: free reducer #0 |}];
  (* The free variable must still be free afterwards, and still be the SAME
     variable: freshening renames bound names, and a free reference is not one. *)
  let n = Rewrite.alpha_normalize frag in
  Fmt.pr "after:  %a@." verdict (Check.value n);
  [%expect {| after:  free reducer #0 |}];
  Fmt.pr "free set preserved: %b@."
    (Reduce_var.Set.equal (Fold.free_reducers frag) (Fold.free_reducers n));
  [%expect {| free set preserved: true |}];
  (* Which is what the numbering contract has to say: binders take the lowest
     ordinals NOT free here, so #0 is skipped and the single binder lands on #1
     rather than the 0, 1, ... a closed expression would get. *)
  Fmt.pr "binders after normalisation: %a@."
    (Fmt.list ~sep:(Fmt.any ",") Reduce_var.pp)
    (Fold.binders n);
  [%expect {| binders after normalisation: #1 |}];
  (* And the denotation is unchanged. Both still fail the same way -- the free
     reducer is unbound at top level -- which is itself the point: capture would
     have made the renamed one succeed with a different answer. *)
  Fmt.pr "same value: %b@." (same_value (ev frag) (ev n));
  [%expect {| same value: true |}]

(* ---- substitute_loads ------------------------------------------------------ *)

let%expect_test "substitute_loads replaces a load with a subtree" =
  let src = Source.create 3 in
  let coord = Coord.of_fn (fun a -> Index.output a) in
  let e = Value.add (Value.load src coord) (Value.const 1.0) in
  let replaced =
    Builder.run
      (Rewrite.substitute_loads
         (fun s _ ->
           if Source.equal s src then Some (Builder.return (Value.const 7.0))
           else None)
         e)
  in
  Fmt.pr "before: %a@.after:  %a@." Pp.value e Pp.value replaced;
  (* [None] keeps the load, so an unmatched source is untouched. *)
  let untouched = Builder.run (Rewrite.substitute_loads (fun _ _ -> None) e) in
  Fmt.pr "unmatched preserved: %b@." (Value.equal e untouched);
  [%expect
    {|
    before: (t3[N,T,D,H,W,C] + 1)
    after:  (7 + 1)
    unmatched preserved: true |}]

let%expect_test "substitute_loads leaves an intrinsic descriptor intact" =
  (* A max-pool holds a source and geometry, not a load node — there is nothing
     inside it one scalar subtree could stand in for. The callback must never
     see it, and the descriptor must come through unchanged. *)
  let src = Source.create 5 in
  let e =
    Value.intrinsic
      (Err.or_raise ~pp_error:Intrinsic.pp_error
         (Intrinsic.max_pool ~source:src ~in_h:4 ~in_w:4 ~kernel_h:2 ~kernel_w:2
            ~stride_h:2 ~stride_w:2 ~pad_h:0 ~pad_w:0
            ~out:(Coord.of_fn (fun a -> Index.output a))
            ~result:Intrinsic.Max_pool.Value))
  in
  let seen = ref [] in
  let out =
    Builder.run
      (Rewrite.substitute_loads
         (fun s _ ->
           seen := s :: !seen;
           Some (Builder.return (Value.const 0.0)))
         e)
  in
  Fmt.pr "callback saw %d sources@.unchanged: %b@." (List.length !seen)
    (Value.equal e out);
  (* The dependency is still THERE — it is simply not substitutable, which is
     exactly why [sources] and [loads] answer different questions. *)
  Fmt.pr "sources %d, loads %d, intrinsic_sources %d@."
    (Source.Set.cardinal (Fold.sources e))
    (List.length (Fold.loads e))
    (List.length (Fold.intrinsic_sources e));
  [%expect
    {|
    callback saw 0 sources
    unchanged: true
    sources 1, loads 0, intrinsic_sources 1 |}]

let%expect_test "substitute_loads mints in the destination's namespace" =
  (* The composition rule the whole design rests on. Insert a reduction-bearing
     fragment into a reduction-bearing destination, both built from [initial] so
     both mint ordinal 0.

     Threading ONE state through the freshening and the substitution keeps the
     binders apart; freshening each fragment from [initial] instead reintroduces
     the collision at the moment of composing, which is the mistake this API
     exists to make hard. *)
  let src = Source.create 1 in
  let coord = Coord.of_fn (fun a -> Index.output a) in
  let reduction leaf =
    Builder.run
      (Builder.reduction ~kind:Reduction.Sum ~lo:Index.zero ~hi:(Index.const 2)
         (fun r ->
           Builder.return
             (Value.add leaf (Value.value_of_index (Index.of_position r)))))
  in
  let destination = reduction (Value.load src coord) in
  let fragment = reduction (Value.const 1.0) in
  let compose ~shared =
    let open Builder.Syntax in
    Builder.run
      (let* d = Rewrite.freshen destination in
       Rewrite.substitute_loads
         (fun _ _ ->
           Some
             (if shared then Rewrite.freshen fragment
              else Builder.return (Builder.run (Rewrite.freshen fragment))))
         d)
  in
  let verdict =
    Core.Pretty.err_result ~ok:(Fmt.any "ok") ~error:Check.pp_error
  in
  Fmt.pr "threaded state:  %a@." verdict (Check.value (compose ~shared:true));
  Fmt.pr "fragment re-run from initial: %a@." verdict
    (Check.value (compose ~shared:false));
  [%expect
    {|
    threaded state:  ok
    fragment re-run from initial: reducer #0 is bound twice on one path |}]

let%expect_test "substitute_loads does not re-traverse what it inserted" =
  (* Stated in the .mli, and depended on by the one-edge fusion rule: a chain
     cannot be collapsed by a single pass, so a caller must either iterate or
     restrict itself to non-overlapping edges. *)
  let a = Source.create 1 and b = Source.create 2 in
  let coord = Coord.of_fn (fun a -> Index.output a) in
  let e = Value.load a coord in
  let out =
    Builder.run
      (Rewrite.substitute_loads
         (fun s _ ->
           if Source.equal s a then Some (Builder.return (Value.load b coord))
           else if Source.equal s b then
             Some (Builder.return (Value.const 99.0))
           else None)
         e)
  in
  Fmt.pr "%a@." Pp.value out;
  [%expect {| t2[N,T,D,H,W,C] |}]
