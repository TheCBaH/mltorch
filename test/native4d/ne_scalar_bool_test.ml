(* [Eval_direct4]'s [Ne_scalar] dispatch, the Native4D twin of
   [Eval_direct]'s own arm ([ne_scalar_bool_test.ml] under test/native).
   Landed the full new-op surface (`domain.ml`, `lower_engine.ml`, `op.ml`,
   `graph_shape4.ml`, `output_transfer4.ml`, `eval_op4.ml`, `builder.ml`,
   `eval_direct4.ml`), the same eight-file shape [Eq_scalar]'s own Native4D
   landing used (see [eq_scalar_bool_test.ml]). *)

open Native4d

let shape5 = Shape4.of_ints ~n:1 ~h:1 ~w:1 ~c:5

let%expect_test "direct4: ne_scalar writes genuine Bool storage" =
  let g =
    Builder.build
      ~outputs:(fun o -> [ o ])
      (let open Builder in
       let* x = input ~shape:shape5 () in
       ne_scalar 2. x)
    |> Err.or_raise ~pp_error:Builder.pp_error
  in
  let x =
    Tensor.materialize (Shape4.to_vec6 shape5) (fun c ->
        [| 1.; 2.; 3.; Float.nan; Float.neg_infinity |].(Dim.to_int
                                                           (Vec6.get c Axis.C)))
  in
  let env =
    Eval_direct4.run g ~inputs:(List.combine g.Graph.Graph.inputs [ x ])
    |> Err.or_raise ~pp_error:Eval_direct4.pp_error
  in
  let out = Tensor_id.Map.find (List.hd g.Graph.Graph.outputs) env in
  Fmt.pr "%a@." Tensor.pp out;
  (* IEEE inequality, NaN unequal to everything (so [ne] is true), matching
     Native's own "Direct: ne_scalar" fixture's values exactly -- the
     pointwise negation of [eq_scalar_bool_test.ml]'s own {0, 1, 0, 0, 0}. *)
  [%expect {| tensor bool [C=5] {1, 0, 1, 1, 1} |}]
