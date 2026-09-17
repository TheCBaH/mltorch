(* [Eval_direct4]'s dtype-preserving tensor-tensor Add/Sub/Mul and explicit
   int64-input [Mul_scalar] promotion -- the Native4D twin of [Eval_direct]'s
   own P5.3/P5.4 arms (`test/native/pointwise_binary_i64_test.ml`/
   `mul_scalar_i64_test.ml`). Native4D had no I64 dispatch at all for these
   four ops before this session (confirmed by grep); its default arms
   delegate to Native's shared [Pointwise.{Add,Sub,Mul,Mul_scalar}.Compute(S)
   .pixel], which round-trips every format through [Payload.get_float] --
   exact for F32 but silently lossy above 2^53 for I64 operands. *)

open Native4d

let shape6 = Shape4.of_ints ~n:1 ~h:1 ~w:1 ~c:6

let read t c =
  Err.or_raise
    ~pp_error:(fun fmt (`Wrong_format (Payload.Fmt f)) ->
      Fmt.pf fmt "wrong format %a" Payload.pp_fmt f)
    (Tensor.read_i64_at6 t (function
      | Axis.C -> c
      | Axis.N | Axis.T | Axis.D | Axis.H | Axis.W -> 0))

let big6 =
  Tensor.materialize_i64 (Shape4.to_vec6 shape6) (fun c ->
      Int64.add 9_007_199_254_740_993L
        (Int64.of_int (Dim.to_int (Vec6.get c Axis.C))))

let small6 vals =
  Tensor.materialize_i64 (Shape4.to_vec6 shape6) (fun c ->
      List.nth vals (Dim.to_int (Vec6.get c Axis.C)))

let run_binary ~op_of a b =
  let g =
    Builder.build
      ~outputs:(fun o -> [ o ])
      (let open Builder in
       let* x = input ~shape:shape6 ~fmt:Payload.(Fmt I64) () in
       let* y = input ~shape:shape6 ~fmt:Payload.(Fmt I64) () in
       op_of x y)
    |> Err.or_raise ~pp_error:Builder.pp_error
  in
  let env =
    Eval_direct4.run g ~inputs:(List.combine g.Graph.Graph.inputs [ a; b ])
    |> Err.or_raise ~pp_error:Eval_direct4.pp_error
  in
  Tensor_id.Map.find (List.hd g.Graph.Graph.outputs) env

let pp6 out =
  Fmt.pr "%Ld,%Ld,%Ld,%Ld,%Ld,%Ld@." (read out 0) (read out 1) (read out 2)
    (read out 3) (read out 4) (read out 5)

let%expect_test "direct4: I64 add stays exact past 2^53" =
  pp6 (run_binary ~op_of:Builder.add big6 (small6 [ 1L; 2L; 3L; 4L; 5L; 6L ]));
  [%expect
    {| 9007199254740994,9007199254740996,9007199254740998,9007199254741000,9007199254741002,9007199254741004 |}]

let%expect_test "direct4: I64 sub stays exact past 2^53" =
  pp6 (run_binary ~op_of:Builder.sub big6 (small6 [ 1L; 2L; 3L; 4L; 5L; 6L ]));
  [%expect
    {| 9007199254740992,9007199254740992,9007199254740992,9007199254740992,9007199254740992,9007199254740992 |}]

let%expect_test "direct4: I64 mul stays exact past 2^53" =
  let shape1 = Shape4.of_ints ~n:1 ~h:1 ~w:1 ~c:1 in
  let a = Tensor.materialize_i64 (Shape4.to_vec6 shape1) (fun _ -> 100_000_003L)
  and b =
    Tensor.materialize_i64 (Shape4.to_vec6 shape1) (fun _ -> 100_000_003L)
  in
  let g =
    Builder.build
      ~outputs:(fun o -> [ o ])
      (let open Builder in
       let* x = input ~shape:shape1 ~fmt:Payload.(Fmt I64) () in
       let* y = input ~shape:shape1 ~fmt:Payload.(Fmt I64) () in
       mul x y)
    |> Err.or_raise ~pp_error:Builder.pp_error
  in
  let env =
    Eval_direct4.run g ~inputs:(List.combine g.Graph.Graph.inputs [ a; b ])
    |> Err.or_raise ~pp_error:Eval_direct4.pp_error
  in
  let out = Tensor_id.Map.find (List.hd g.Graph.Graph.outputs) env in
  (* 100000003 * 100000003 = 10000000600000009, past 2^53 and odd -- a real
     double multiply of the same two (exactly representable) float operands
     would round this product to the nearest even value. *)
  Fmt.pr "%Ld@." (read out 0);
  [%expect {| 10000000600000009 |}]

(* P5.4's own mixed-dtype rejection, the Native4D twin of the Native fixture:
   a mismatched I64/F32 pair fails at checked admission rather than silently
   computing through the default float path. *)
let%expect_test "direct4: mixed I64/F32 add/sub/mul are rejected" =
  let run op_of =
    let g =
      Builder.build
        ~outputs:(fun o -> [ o ])
        (let open Builder in
         let* x = input ~shape:shape6 ~fmt:Payload.(Fmt I64) () in
         let* y = input ~shape:shape6 () in
         op_of x y)
      |> Err.or_raise ~pp_error:Builder.pp_error
    in
    let x = small6 [ 1L; 2L; 3L; 0L; 0L; 0L ] in
    let y = Tensor.materialize (Shape4.to_vec6 shape6) (fun _ -> 1.0) in
    Eval_direct4.run g ~inputs:(List.combine g.Graph.Graph.inputs [ x; y ])
  in
  let pp fmt = function
    | Ok (_ : Tensor.packed Tensor_id.Map.t) -> Fmt.string fmt "ok"
    | Error e -> Fmt.pf fmt "%a" Eval_direct4.pp_error (Err.Error.kind e)
  in
  Fmt.pr "%a@." pp (run Builder.add);
  Fmt.pr "%a@." pp (run Builder.sub);
  Fmt.pr "%a@." pp (run Builder.mul);
  [%expect
    {|
    add: unsupported mixed dtype, a=i64 b=f32
    sub: unsupported mixed dtype, a=i64 b=f32
    mul: unsupported mixed dtype, a=i64 b=f32
    |}]

(* Explicit int64-input promotion for [Mul_scalar]: the output stays F32 by
   design, so only the read changes, not the write-back -- same architecture-
   only shape as Native's own `mul_scalar_i64_test.ml`. *)
(* The Native4D twin of `test/native/pointwise_binary_i64_test.ml`'s own
   Bool-arithmetic fixture: arithmetic on Bool stays rejected here too,
   checked BEFORE the I64 mixed-dtype guard so a Bool/I64 pair reports the
   Bool reason specifically. *)
let%expect_test "direct4: arithmetic on a Bool operand is rejected" =
  let run ~y_fmt op_of =
    let g =
      Builder.build
        ~outputs:(fun o -> [ o ])
        (let open Builder in
         let* x = input ~shape:shape6 ~fmt:Payload.(Fmt Bool) () in
         let* y = input ~shape:shape6 ~fmt:y_fmt () in
         op_of x y)
      |> Err.or_raise ~pp_error:Builder.pp_error
    in
    let x = Tensor.materialize_bool (Shape4.to_vec6 shape6) (fun _ -> true) in
    let y =
      match y_fmt with
      | Payload.Fmt Payload.I64 -> small6 [ 1L; 1L; 1L; 1L; 1L; 1L ]
      | _ -> Tensor.materialize (Shape4.to_vec6 shape6) (fun _ -> 1.0)
    in
    Eval_direct4.run g ~inputs:(List.combine g.Graph.Graph.inputs [ x; y ])
  in
  let pp fmt = function
    | Ok (_ : Tensor.packed Tensor_id.Map.t) -> Fmt.string fmt "ok"
    | Error e -> Fmt.pf fmt "%a" Eval_direct4.pp_error (Err.Error.kind e)
  in
  let run_all ~y_fmt =
    Fmt.pr "%a@." pp (run ~y_fmt Builder.add);
    Fmt.pr "%a@." pp (run ~y_fmt Builder.sub);
    Fmt.pr "%a@." pp (run ~y_fmt Builder.mul)
  in
  run_all ~y_fmt:Payload.(Fmt F32);
  run_all ~y_fmt:Payload.(Fmt I64);
  [%expect
    {|
    add: arithmetic on a Bool operand is not supported, a=bool b=f32
    sub: arithmetic on a Bool operand is not supported, a=bool b=f32
    mul: arithmetic on a Bool operand is not supported, a=bool b=f32
    add: arithmetic on a Bool operand is not supported, a=bool b=i64
    sub: arithmetic on a Bool operand is not supported, a=bool b=i64
    mul: arithmetic on a Bool operand is not supported, a=bool b=i64
    |}]

let%expect_test "direct4: Mul_scalar reads an I64 operand via an explicit cast"
    =
  let shape3 = Shape4.of_ints ~n:1 ~h:1 ~w:1 ~c:3 in
  let g =
    Builder.build
      ~outputs:(fun o -> [ o ])
      (let open Builder in
       let* x = input ~shape:shape3 ~fmt:Payload.(Fmt I64) () in
       mul_scalar 2.5 x)
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
  [%expect {| tensor f32 [C=3] {2.5, 5, 7.5} |}]

(* The Native4D twin of `test/native/mul_scalar_i64_test.ml`'s own
   Bool-rejection fixture: [Mul_scalar]'s existing per-format admission
   point (extended above for I64) also needs a Bool arm, mirroring the
   two-operand Add/Sub/Mul fixture above. *)
let%expect_test "direct4: Mul_scalar rejects a Bool operand" =
  let g =
    Builder.build
      ~outputs:(fun o -> [ o ])
      (let open Builder in
       let* x = input ~shape:shape6 ~fmt:Payload.(Fmt Bool) () in
       mul_scalar 2.5 x)
    |> Err.or_raise ~pp_error:Builder.pp_error
  in
  let x = Tensor.materialize_bool (Shape4.to_vec6 shape6) (fun _ -> true) in
  let pp fmt = function
    | Ok (_ : Tensor.packed Tensor_id.Map.t) -> Fmt.string fmt "ok"
    | Error e -> Fmt.pf fmt "%a" Eval_direct4.pp_error (Err.Error.kind e)
  in
  Fmt.pr "%a@." pp
    (Eval_direct4.run g ~inputs:(List.combine g.Graph.Graph.inputs [ x ]));
  [%expect
    {| mul_scalar: arithmetic on a Bool operand is not supported, x=bool |}]
