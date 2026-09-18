(* [Eval_direct4]'s [Gt_scalar] dispatch, the Native4D twin of
   [Eval_direct]'s own P6.3 arm. Unlike [To_copy(Bool)]/[Bitwise_not],
   [Gt_scalar] did not exist as a Native4D op at all before this session --
   [Domain.check_node] explicitly routed it to the `unsupported ()` bucket,
   and [Op.t] had no variant for it. This landed the full new-op surface
   (`domain.ml`, `lower_engine.ml`, `op.ml`, `graph_shape4.ml`,
   `output_transfer4.ml`, `eval_op4.ml`, `builder.ml`, `eval_direct4.ml`),
   the Native4D mirror of Native's own "Full new-op surface, traced via the
   compiler" entry. *)

open Native4d

let shape5 = Shape4.of_ints ~n:1 ~h:1 ~w:1 ~c:5

let%expect_test "direct4: gt_scalar writes genuine Bool storage" =
  let g =
    Builder.build
      ~outputs:(fun o -> [ o ])
      (let open Builder in
       let* x = input ~shape:shape5 () in
       gt_scalar 2. x)
    |> Err.or_raise ~pp_error:Builder.pp_error
  in
  let x =
    Tensor.materialize (Shape4.to_vec6 shape5) (fun c ->
        [| 1.; 2.; 3.; Float.neg_infinity; -1. |].(Dim.to_int
                                                     (Vec6.get c Axis.C)))
  in
  let env =
    Eval_direct4.run g ~inputs:(List.combine g.Graph.Graph.inputs [ x ])
    |> Err.or_raise ~pp_error:Eval_direct4.pp_error
  in
  let out = Tensor_id.Map.find (List.hd g.Graph.Graph.outputs) env in
  Fmt.pr "%a@." Tensor.pp out;
  (* Strict ordering (2. > 2. is false, not the nearest boundary), matching
     Native's own "Direct: gt_scalar" fixture's values exactly. *)
  [%expect {| tensor bool [C=5] {0, 0, 1, 0, 0} |}]
