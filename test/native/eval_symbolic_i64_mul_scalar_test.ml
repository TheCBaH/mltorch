(* The Symbolic/Kernel twin of `graph_direct_pointwise_test.ml`'s own I64
   Mul_scalar Direct-route fixture. Unlike Reshape/Permute/Add/Sub/Mul,
   [Mul_scalar]'s output stays the ordinary FLOAT carrier by design (ATen
   promotes an integer tensor times a float scalar to a float result), so
   this exercises the "[Eval_symbolic]'s new arm produces an ordinary
   [Stage.t], not a [Stage_i64.t]" half of that arm's own comment -- the
   Arange operand is exact I64 (a [Stage_i64.t]), but [Mul_scalar]'s own
   output is an ordinary float [Stage.t] consuming it. *)

open Graph_ir

let arange_params =
  {
    Factory.Arange.start = 9.007199254740993e15;
    stop = 9.007199254740999e15;
    step = 1.;
    fmt = Payload.(Fmt I64);
    exact =
      Some
        {
          Factory.Arange.Exact.start = 9_007_199_254_740_993L;
          stop = 9_007_199_254_740_999L;
          step = 1L;
        };
  }

let build =
  Graph_builder.(
    build ~name:"i64_arange_mul_scalar" ~outputs:(fun r -> [ r ])
    @@
    let* a = arange arange_params in
    mul_scalar ~name:"out" 2. a)
  |> Err.or_raise ~pp_error:Graph_builder.pp_error

let%expect_test
    "Symbolic -> Kernel: I64 Arange -> Mul_scalar produces an ordinary float \
     Stage, not a Stage_i64" =
  let stage_program = Eval_symbolic.run build in
  Format.printf "stages: %d, stages_i64: %d@."
    (List.length stage_program.Stage_program.stages)
    (List.length stage_program.Stage_program.stages_i64);
  let out_id = List.hd build.Graph.outputs in
  let kernel =
    Kernel_adapt.of_stage_program stage_program
    |> Err.or_raise ~pp_error:Kernel_adapt.pp_error
  in
  let result =
    Kernel_eval.run kernel ~bind:(fun _ -> None)
    |> Err.or_raise ~pp_error:Kernel_eval.pp_error
  in
  (match Tensor_id.Map.find_opt out_id result with
  | Some t -> Format.printf "%a@." Tensor.pp t
  | None -> print_endline "MISSING");
  [%expect
    {|
    stages: 1, stages_i64: 1
    tensor f32 [C=6] {1.80144e+16, 1.80144e+16, 1.80144e+16, 1.80144e+16, 1.80144e+16, 1.80144e+16}
    |}]
