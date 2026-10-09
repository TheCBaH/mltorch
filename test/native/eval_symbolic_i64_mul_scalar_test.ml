(* The Symbolic/Kernel twin of `graph_direct_pointwise_test.ml`'s own I64
   Mul_scalar Direct-route fixture. Unlike Reshape/Permute/Add/Sub/Mul,
   [Mul_scalar]'s output stays the ordinary FLOAT carrier by design (ATen
   promotes an integer tensor times a float scalar to a float result), so
   this exercises the "[Eval_symbolic]'s new arm produces an ordinary
   [Stage.t], not a [Stage_i64.t]" half of that arm's own comment -- the
   Arange operand is exact I64 (a [Stage_i64.t]), but [Mul_scalar]'s own
   output is an ordinary float [Stage.t] consuming it. The scalar is 2.5: a
   whole-number scalar keeps the int64 dtype (ATen's integer-scalar overload)
   and takes the exact path below instead. *)

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
    mul_scalar ~name:"out" 2.5 a)
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
    tensor f32 [C=6] {2.2518e+16, 2.2518e+16, 2.2518e+16, 2.2518e+16, 2.2518e+16, 2.2518e+16}
    |}]

let build_whole =
  Graph_builder.(
    build ~name:"i64_arange_mul_whole" ~outputs:(fun r -> [ r ])
    @@
    let* a = arange arange_params in
    mul_scalar ~name:"out" 2. a)
  |> Err.or_raise ~pp_error:Graph_builder.pp_error

(* 2^53 + 1 times 2: exact only if no float is involved. *)
let%expect_test
    "Symbolic -> Kernel: I64 Arange -> Mul_scalar by a whole number stays an \
     exact Stage_i64" =
  let stage_program = Eval_symbolic.run build_whole in
  Format.printf "stages: %d, stages_i64: %d@."
    (List.length stage_program.Stage_program.stages)
    (List.length stage_program.Stage_program.stages_i64);
  let out_id = List.hd build_whole.Graph.outputs in
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
    stages: 0, stages_i64: 2
    tensor i64 [C=6] {18014398509481986, 18014398509481988, 18014398509481990, 18014398509481992, 18014398509481994, 18014398509481996} |}]
