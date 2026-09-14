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
           exact = None;
         })
  in
  let env =
    Eval_direct.run g ~inputs:[] |> Err.or_raise ~pp_error:Eval_direct.pp_error
  in
  Format.printf "%a@." Tensor.pp
    (Tensor_id.Map.find (List.hd g.Graph_ir.Graph.outputs) env);
  [%expect {| tensor i64 [C=3] {2, 4, 6} |}]

(* Above 2^53, [start]/[step] as plain [float]s can no longer represent every
   integer exactly -- [params.exact] is what lets [Eval_direct]'s Arange arm
   skip the lossy [Int64.of_float (Factory.Arange.value params i)] path
   entirely and generate the real int64 values directly. See the
   implementation tracker's D02/D09 notes. *)
let%expect_test "direct: arange with exact params stays exact past 2^53" =
  let g =
    build
      (Graph_builder.arange
         {
           Factory.Arange.start = 9_007_199_254_740_993.;
           stop = 9_007_199_254_740_996.;
           step = 1.;
           fmt = Payload.Fmt Payload.I64;
           exact =
             Some
               {
                 Factory.Arange.Exact.start = 9_007_199_254_740_993L;
                 stop = 9_007_199_254_740_996L;
                 step = 1L;
               };
         })
  in
  let env =
    Eval_direct.run g ~inputs:[] |> Err.or_raise ~pp_error:Eval_direct.pp_error
  in
  Format.printf "%a@." Tensor.pp
    (Tensor_id.Map.find (List.hd g.Graph_ir.Graph.outputs) env);
  [%expect
    {| tensor i64 [C=4] {9007199254740993, 9007199254740994, 9007199254740995, 9007199254740996} |}]

(* [start] near [Int64.max_int] with a [step] that pushes a later generated
   element past it: [checked_add] must catch this rather than let
   [Int64.add] wrap silently to a large negative value. *)
let%expect_test "direct: arange with exact params reports int64 overflow" =
  let start = Int64.sub Int64.max_int 5L in
  let g =
    build
      (Graph_builder.arange
         {
           (* The float bounds are placeholders for [length]'s element COUNT
              only (2 elements) -- at [start]'s real magnitude (~2^63), a
              float can no longer even distinguish [start] from [start+15],
              so the count must come from small, unrelated floats rather
              than a real projection of [exact]'s own bounds. Harmless: only
              [exact] drives the values [value_i64_exact] actually
              generates, which is what this test exercises. *)
           Factory.Arange.start = 0.;
           stop = 20.;
           step = 10.;
           fmt = Payload.Fmt Payload.I64;
           exact =
             Some
               {
                 Factory.Arange.Exact.start;
                 stop = Int64.add start 15L;
                 step = 10L;
               };
         })
  in
  Format.printf "%a@."
    (Core.Pretty.err_result ~ok:Tensor.pp ~error:Eval_direct.pp_error)
    (Result.map
       (fun env -> Tensor_id.Map.find (List.hd g.Graph_ir.Graph.outputs) env)
       (Eval_direct.run g ~inputs:[]));
  [%expect
    {| arange: exact int64 generation overflows at start=9223372036854775802 step=10 i=1 |}]

let%expect_test "symbolic F32 arange reaches the kernel adapter" =
  let g =
    build
      (Graph_builder.arange
         {
           Factory.Arange.start = 0.5;
           stop = 3.;
           step = 1.;
           fmt = Payload.Fmt Payload.F32;
           exact = None;
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
