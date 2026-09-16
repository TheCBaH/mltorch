(* [Eval_direct4]'s dtype-preserving [To_copy] dispatch -- the Native4D twin
   of [Eval_direct]'s own P5.3 arms (`test/native/to_copy_i64_test.ml`/
   `to_copy_long_i64_test.ml`). Native4D had no I64 dispatch at all for
   either [To_copy] target before this session. *)

open Native4d

let shape3 = Shape4.of_ints ~n:1 ~h:1 ~w:1 ~c:3

let read t c =
  Err.or_raise
    ~pp_error:(fun fmt (`Wrong_format (Payload.Fmt f)) ->
      Fmt.pf fmt "wrong format %a" Payload.pp_fmt f)
    (Tensor.read_i64_at6 t (function
      | Axis.C -> c
      | Axis.N | Axis.T | Axis.D | Axis.H | Axis.W -> 0))

let%expect_test
    "direct4: To_copy(Float) reads an I64 operand via an explicit cast" =
  let g =
    Builder.build
      ~outputs:(fun o -> [ o ])
      (let open Builder in
       let* x = input ~shape:shape3 ~fmt:Payload.(Fmt I64) () in
       to_copy Pointwise.To_copy.Float x)
    |> Err.or_raise ~pp_error:Builder.pp_error
  in
  let x =
    Tensor.materialize_i64 (Shape4.to_vec6 shape3) (fun c ->
        Int64.of_int (1 + Dim.to_int (Vec6.get c Axis.C)))
  in
  let env =
    Eval_direct4.run g ~inputs:(List.combine g.Graph.Graph.inputs [ x ])
    |> Err.or_raise ~pp_error:Eval_direct4.pp_error
  in
  let out = Tensor_id.Map.find (List.hd g.Graph.Graph.outputs) env in
  Fmt.pr "%a@." Tensor.pp out;
  [%expect {| tensor f32 [C=3] {1, 2, 3} |}]

let build_to_copy_long x =
  let g =
    Builder.build
      ~outputs:(fun o -> [ o ])
      (let open Builder in
       let* x_ref =
         input ~shape:(Shape4.of_ints ~n:1 ~h:1 ~w:1 ~c:(Array.length x)) ()
       in
       to_copy Pointwise.To_copy.Long x_ref)
    |> Err.or_raise ~pp_error:Builder.pp_error
  in
  let x_t =
    Tensor.materialize
      (Shape4.to_vec6 (Shape4.of_ints ~n:1 ~h:1 ~w:1 ~c:(Array.length x)))
      (fun c -> x.(Dim.to_int (Vec6.get c Axis.C)))
  in
  let env =
    Eval_direct4.run g ~inputs:(List.combine g.Graph.Graph.inputs [ x_t ])
    |> Err.or_raise ~pp_error:Eval_direct4.pp_error
  in
  Tensor_id.Map.find (List.hd g.Graph.Graph.outputs) env

let%expect_test "direct4: To_copy(Long) truncates an F32 operand toward zero" =
  let out = build_to_copy_long [| 3.7; -3.7; 1000000. |] in
  Fmt.pr "%Ld,%Ld,%Ld@." (read out 0) (read out 1) (read out 2);
  [%expect {| 3,-3,1000000 |}]

let%expect_test "direct4: To_copy(Long) on an I64 operand is an identity" =
  let g =
    Builder.build
      ~outputs:(fun o -> [ o ])
      (let open Builder in
       let* x = input ~shape:shape3 ~fmt:Payload.(Fmt I64) () in
       to_copy Pointwise.To_copy.Long x)
    |> Err.or_raise ~pp_error:Builder.pp_error
  in
  let x =
    Tensor.materialize_i64 (Shape4.to_vec6 shape3) (fun c ->
        Int64.add 9007199254740993L
          (Int64.of_int (Dim.to_int (Vec6.get c Axis.C))))
  in
  let env =
    Eval_direct4.run g ~inputs:(List.combine g.Graph.Graph.inputs [ x ])
    |> Err.or_raise ~pp_error:Eval_direct4.pp_error
  in
  let out = Tensor_id.Map.find (List.hd g.Graph.Graph.outputs) env in
  Fmt.pr "%Ld,%Ld,%Ld@." (read out 0) (read out 1) (read out 2);
  [%expect {| 9007199254740993,9007199254740994,9007199254740995 |}]

(* [Err.Exn.E], the same convention Native's own fixture proves. *)
let catch f =
  try
    ignore (f ());
    "no exception"
  with Err.Exn.E e -> Format.asprintf "raised: %a" Err.Exn.pp_kind e

let%expect_test
    "direct4: To_copy(Long) rejects NaN, infinities and out-of-range magnitudes"
    =
  Fmt.pr "%s@." (catch (fun () -> build_to_copy_long [| Float.nan |]));
  Fmt.pr "%s@."
    (catch (fun () -> build_to_copy_long [| 9223372036854775808. |]));
  [%expect
    {|
    raised: Float-to-I64 cast of NaN
    raised: Float-to-I64 cast of 0x1p+63, outside [-2^63, 2^63)
    |}]
