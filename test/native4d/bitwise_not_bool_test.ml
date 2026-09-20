(* [Eval_direct4]'s [Bitwise_not] dispatch, the Native4D twin of
   [Eval_direct]'s own arm. Same "already F32-consistent, no live
   mismatch" starting state as [to_copy_bool_test.ml]'s own entry --
   [Builder.bitwise_not] kept [op1]'s F32 default and [eval_direct4.ml] had
   no arm at all before. Chains with [to_copy] the same way
   Native's own `bool_acceptance_test.ml` does, mirroring EdgeNeXt's actual
   Float -> Bool cast -> Bitwise_not mask pattern end to end on Native4D. *)

open Native4d

let shape4 = Shape4.of_ints ~n:1 ~h:1 ~w:1 ~c:4

let%expect_test
    "direct4: To_copy(Bool) -> Bitwise_not writes genuine Bool storage \
     end-to-end" =
  let g =
    Builder.build
      ~outputs:(fun o -> [ o ])
      (let open Builder in
       let* x = input ~shape:shape4 () in
       let* mask = to_copy Pointwise.To_copy.Bool x in
       bitwise_not mask)
    |> Err.or_raise ~pp_error:Builder.pp_error
  in
  let x =
    Tensor.materialize (Shape4.to_vec6 shape4) (fun c ->
        [| 0.; 3.; -2.; 0. |].(Dim.to_int (Vec6.get c Axis.C)))
  in
  let env =
    Eval_direct4.run g ~inputs:(List.combine g.Graph.Graph.inputs [ x ])
    |> Err.or_raise ~pp_error:Eval_direct4.pp_error
  in
  let out = Tensor_id.Map.find (List.hd g.Graph.Graph.outputs) env in
  Fmt.pr "%a@." Tensor.pp out;
  (* x = {0, 3, -2, 0} -> mask (genuine Bool) {false, true, true, false} ->
     bitwise_not (genuine Bool) {true, false, false, true}, matching
     Native's own `bool_acceptance_test.ml`. *)
  [%expect {| tensor bool [C=4] {1, 0, 0, 1} |}]
