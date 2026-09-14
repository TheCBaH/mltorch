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
    {| tensor i64 [C=3] {9007199254740993, 9007199254740994, 9007199254740995} |}]

(* [start] near [Int64.max_int] with a [stop] built by an unchecked
   [Int64.add] that itself wraps past [max_int] into a large negative value:
   once [length] routes through [length_exact] (the D02 count fix), this is
   caught as a [Count_overflow] shape error at graph-construction time --
   before [Eval_direct] ever runs -- rather than surviving to a later
   value-generation-time overflow the way it did when [length] still read
   only the float placeholder bounds and ignored [exact.stop] entirely. *)
let%expect_test
    "direct: arange with an inconsistent exact stop reports count overflow at \
     construction" =
  let start = Int64.sub Int64.max_int 5L in
  Format.printf "%a@."
    (Core.Pretty.err_result
       ~ok:(fun ppf _ -> Format.pp_print_string ppf "built")
       ~error:Graph_builder.pp_error)
    (Graph_builder.build ~name:"arange"
       ~outputs:(fun y -> [ y ])
       (Graph_builder.arange
          {
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
          }));
  [%expect
    {| arange(9.22337e+18, -9.22337e+18, 10): exact element-count computation overflows int64 |}]

(* [value_i64_exact] called directly, bypassing [length_exact]: proves the
   generation-time [checked_add] overflow guard the D02 note names is still
   real, for any caller that reaches an individual index without first
   deriving it from a self-consistent [length_exact] count (e.g. a
   hand-built [Kernel.Value_i64.t]) -- [length_exact]'s own success on a
   *consistent* range makes every in-range index provably non-overflowing
   (see the implementation tracker), so this guard is now defense in depth,
   not reachable through the ordinary graph-building path above. *)
let%expect_test
    "value_i64_exact: reports int64 overflow at the generating index" =
  let start = Int64.sub Int64.max_int 5L in
  let e = { Factory.Arange.Exact.start; stop = start; step = 10L } in
  Format.printf "%a@."
    (Core.Pretty.err_result
       ~ok:(fun ppf v -> Format.fprintf ppf "%Ld" v)
       ~error:Eval_direct.pp_error)
    (Factory.Arange.value_i64_exact e 1);
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
