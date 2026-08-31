(* No-input factories exercise the graph evaluator's zero-operand path and
   must retain their declared storage dtype. *)

let build m =
  Graph_builder.build ~name:"zeros" ~outputs:(fun y -> [ y ]) m
  |> Err.or_raise ~pp_error:Graph_builder.pp_error

let%expect_test "direct: zeros preserves F64 dtype and shape" =
  let g =
    build
      (Graph_builder.zeros
         {
           Factory.Zeros.shape = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:2 ~w:3 ~c:4;
           fmt = Payload.Fmt Payload.F64;
         })
  in
  let env =
    Eval_direct.run g ~inputs:[] |> Err.or_raise ~pp_error:Eval_direct.pp_error
  in
  Format.printf "%a@." Tensor.pp
    (Tensor_id.Map.find (List.hd g.Graph_ir.Graph.outputs) env);
  [%expect {| tensor f64 [H=2 W=3 C=4] {0, 0, 0, 0, 0, 0, 0, 0, ...} |}]

let%expect_test "symbolic F32 zeros reaches the kernel adapter" =
  let g =
    build
      (Graph_builder.zeros
         {
           Factory.Zeros.shape = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:1 ~w:2 ~c:3;
           fmt = Payload.Fmt Payload.F32;
         })
  in
  (match Kernel_adapt.of_stage_program (Eval_symbolic.run g) with
  | Ok _ -> print_endline "kernel accepted"
  | Error e -> Format.printf "%a@." Kernel_adapt.pp_error (Err.Error.kind e));
  [%expect {| kernel accepted |}]

let%expect_test "direct: arange preserves Long indices exactly" =
  let g =
    build
      (Graph_builder.arange
         {
           Factory.Arange.start = 2.;
           stop = 7.;
           step = 2.;
           fmt = Payload.Fmt Payload.I64;
         })
  in
  let env =
    Eval_direct.run g ~inputs:[] |> Err.or_raise ~pp_error:Eval_direct.pp_error
  in
  Format.printf "%a@." Tensor.pp
    (Tensor_id.Map.find (List.hd g.Graph_ir.Graph.outputs) env);
  [%expect {| tensor i64 [C=3] {2, 4, 6} |}]

let%expect_test "symbolic F32 arange reaches the kernel adapter" =
  let g =
    build
      (Graph_builder.arange
         {
           Factory.Arange.start = 0.5;
           stop = 3.;
           step = 1.;
           fmt = Payload.Fmt Payload.F32;
         })
  in
  (match Kernel_adapt.of_stage_program (Eval_symbolic.run g) with
  | Ok _ -> print_endline "kernel accepted"
  | Error e -> Format.printf "%a@." Kernel_adapt.pp_error (Err.Error.kind e));
  [%expect {| kernel accepted |}]

(* [n <> m] (2x3), so a transposed row/column comparison would visibly
   misplace the diagonal ones. *)
let%expect_test "direct: eye preserves F64 dtype, diagonal on w=c" =
  let g =
    build
      (Graph_builder.eye
         {
           Factory.Eye.shape = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:1 ~w:2 ~c:3;
           fmt = Payload.Fmt Payload.F64;
         })
  in
  let env =
    Eval_direct.run g ~inputs:[] |> Err.or_raise ~pp_error:Eval_direct.pp_error
  in
  Format.printf "%a@." Tensor.pp
    (Tensor_id.Map.find (List.hd g.Graph_ir.Graph.outputs) env);
  [%expect {| tensor f64 [W=2 C=3] {1, 0, 0, 0, 1, 0} |}]

let%expect_test "symbolic F32 eye reaches the kernel adapter" =
  let g =
    build
      (Graph_builder.eye
         {
           Factory.Eye.shape = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:1 ~w:3 ~c:3;
           fmt = Payload.Fmt Payload.F32;
         })
  in
  (match Kernel_adapt.of_stage_program (Eval_symbolic.run g) with
  | Ok _ -> print_endline "kernel accepted"
  | Error e -> Format.printf "%a@." Kernel_adapt.pp_error (Err.Error.kind e));
  [%expect {| kernel accepted |}]
