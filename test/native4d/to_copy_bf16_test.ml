(* [Eval_direct4]'s [To_copy] to [Bfloat16]: BF16 storage holding
   [Expr.Value.bf16_value] of each value, a tie to even, a value past a tie
   and a payload NaN, as Native's own arm (bf16_result_test.ml). *)

open Native4d

let shape4 = Shape4.of_ints ~n:1 ~h:1 ~w:1 ~c:3

let%expect_test "direct4: To_copy(Bfloat16) stores the bfloat16 rounding" =
  let g =
    Builder.build
      ~outputs:(fun o -> [ o ])
      (let open Builder in
       let* x = input ~shape:shape4 () in
       to_copy Pointwise.To_copy.Bfloat16 x)
    |> Err.or_raise ~pp_error:Builder.pp_error
  in
  let x =
    Tensor.materialize (Shape4.to_vec6 shape4) (fun c ->
        Int32.float_of_bits
          [| 0x3F808000l; 0x3F808001l; 0x7FFFFFFFl |].(Dim.to_int
                                                         (Vec6.get c Axis.C)))
  in
  let env =
    Eval_direct4.run g ~inputs:(List.combine g.Graph.Graph.inputs [ x ])
    |> Err.or_raise ~pp_error:Eval_direct4.pp_error
  in
  let (Tensor.Tensor t as out) =
    Tensor_id.Map.find (List.hd g.Graph.Graph.outputs) env
  in
  Fmt.pr "%s" (Payload.fmt_name t.Tensor.payload.Payload.fmt);
  for c = 0 to 2 do
    Fmt.pr " %08lx"
      (Int32.bits_of_float
         (Tensor.read out (Vec6.coord ~n:0 ~t:0 ~d:0 ~h:0 ~w:0 ~c)))
  done;
  [%expect {| bf16 3f800000 3f810000 7fc00000 |}]
