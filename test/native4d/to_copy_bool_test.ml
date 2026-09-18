(* [Eval_direct4]'s [To_copy(Bool)] dispatch, the Native4D twin of
   [Eval_direct]'s own P6.3 arm (`test/native/bool_acceptance_test.ml`).
   Native4D had no Bool storage dispatch at all before this session --
   [Builder.to_copy]'s [Bool] case kept [op1]'s F32 default, matching what
   [Eval_direct4]'s generic fallback actually computed, so there was no
   declared/actual mismatch (unlike a bug); both now declare/write genuine
   [Payload.Bool] storage together. *)

open Native4d

let shape5 = Shape4.of_ints ~n:1 ~h:1 ~w:1 ~c:5

let%expect_test
    "direct4: To_copy(Bool) on an F32 operand writes genuine Bool storage" =
  let g =
    Builder.build
      ~outputs:(fun o -> [ o ])
      (let open Builder in
       let* x = input ~shape:shape5 () in
       to_copy Pointwise.To_copy.Bool x)
    |> Err.or_raise ~pp_error:Builder.pp_error
  in
  let x =
    Tensor.materialize (Shape4.to_vec6 shape5) (fun c ->
        [| 0.; 3.; -2.; 0.; 1e-3 |].(Dim.to_int (Vec6.get c Axis.C)))
  in
  let env =
    Eval_direct4.run g ~inputs:(List.combine g.Graph.Graph.inputs [ x ])
    |> Err.or_raise ~pp_error:Eval_direct4.pp_error
  in
  let out = Tensor_id.Map.find (List.hd g.Graph.Graph.outputs) env in
  Fmt.pr "%a@." Tensor.pp out;
  [%expect {| tensor bool [C=5] {0, 1, 1, 0, 1} |}]

let%expect_test "direct4: To_copy(Bool) rejects every other source format" =
  let g fmt =
    Builder.build
      ~outputs:(fun o -> [ o ])
      (let open Builder in
       let* x = input ~shape:shape5 ~fmt () in
       to_copy Pointwise.To_copy.Bool x)
    |> Err.or_raise ~pp_error:Builder.pp_error
  in
  let run fmt =
    let g = g fmt in
    let x = Tensor.materialize_i64 (Shape4.to_vec6 shape5) (fun _ -> 0L) in
    Eval_direct4.run g ~inputs:(List.combine g.Graph.Graph.inputs [ x ])
  in
  Fmt.pr "%a@."
    (Core.Pretty.err_result
       ~ok:(fun fmt _ -> Fmt.string fmt "ok")
       ~error:Eval_direct4.pp_error)
    (run Payload.(Fmt I64));
  [%expect {| to_copy: Bool target has no exact Bool output for a i64 source |}]
