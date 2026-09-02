(* Split out of what was graph_symbolic_test.ml see graph_symbolic_fixtures.ml. *)

open Graph_symbolic_fixtures

let%expect_test "Symbolic lowering: stages are well-scoped, and reuse ordinals"
    =
  let result =
    let open Err.Syntax in
    let* g =
      lift_build
        Graph_builder.(
          build ~name:"two_reducers" ~outputs:(fun (_, _, _, y) -> [ y ])
          @@
          let* x = input ~shape:(s 1 1 1 2 3 3) ~name:"x_nchw" () in
          let* w = input ~shape:(s 1 1 1 2 2 2) ~name:"w" () in
          let* b = input ~shape:(s1c 1) ~name:"b" () in
          let* xh = permute ~name:"x_nhwc" p_to_nhwc x in
          (* TWO reduction-bearing stages, which is what makes the ordinal
             question observable at all. *)
          let* c1 = conv2d ~name:"c1" conv_params ~x:xh ~weight:w ~bias:b () in
          let* c2 = conv2d ~name:"c2" conv_params ~x:xh ~weight:w ~bias:b () in
          let* y = add ~name:"out" c1 c2 in
          return (x, w, b, y))
    in
    let prog = Eval_symbolic.run g in
    (* Every stage is closed and singly-bound on each path. This is the property
       that matters; it is checked per stage, not across the program. *)
    let checked =
      List.for_all
        (fun (st : Stage_program.Stage.t) ->
          match
            Region_program.pixel_expression st.Stage_program.Stage.computation
          with
          | None -> false
          | Some body -> (
              match Expr.Check.value body with
              | Ok () -> true
              | Error _ -> false))
        prog.Stage_program.stages
    in
    Format.printf "every stage well-scoped: %b@." checked;
    (* Stages DELIBERATELY share ordinals: each runs from [Builder.initial], and
       a reducer identity means nothing outside the expression that binds it
       (see Expr.Reduce_var). Asserting disjointness here would invent a
       graph-level contract the library refuses to make, and would let a future
       fusion pass skip freshening and still pass. *)
    let binders =
      List.concat_map
        (fun (st : Stage_program.Stage.t) ->
          Region_program.pixel_expression st.Stage_program.Stage.computation
          |> Option.value ~default:(Expr.Value.const 0.)
          |> Expr.Fold.binders)
        prog.Stage_program.stages
    in
    Format.printf "reducers bound across stages: %d, distinct identities: %d@."
      (List.length binders)
      (Expr.Reduce_var.Set.cardinal (Expr.Reduce_var.Set.of_list binders));
    (* Lowering is repeatable: a second run yields the same expressions, not
       merely equivalent ones shifted by whatever ran earlier. *)
    let again = Eval_symbolic.run g in
    let same =
      List.for_all2
        (fun (a : Stage_program.Stage.t) (b : Stage_program.Stage.t) ->
          match
            ( Region_program.pixel_expression a.Stage_program.Stage.computation,
              Region_program.pixel_expression b.Stage_program.Stage.computation
            )
          with
          | Some a, Some b -> Expr.Value.equal a b
          | None, None -> true
          | Some _, None | None, Some _ -> false)
        prog.Stage_program.stages again.Stage_program.stages
    in
    Format.printf "second lowering equal: %b@." same;
    Err.return ()
  in
  ignore (result : (unit, [> error ]) Err.t);
  [%expect
    {|
    every stage well-scoped: true
    reducers bound across stages: 6, distinct identities: 3
    second lowering equal: true |}]

(* One stage per output ordinal, and the count is the input's extent at the
   selected axis -- the symbolic program's SIZE varies with the operand
   signature, which no other op does. Each stage's index expression differs only
   in the constant folded onto the selected axis. *)
