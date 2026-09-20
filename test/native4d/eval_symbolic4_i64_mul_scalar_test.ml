(* The Native4D twin of `test/native/eval_symbolic_i64_mul_scalar_test.ml` --
   see its own header for the full rationale. [Kernel_adapt]/[Kernel_eval]
   are the SAME (Native, not Native4D-specific) modules used there. *)

open Native4d

let arange_params =
  {
    Ops4.Arange4.start = 9.007199254740993e15;
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

let g =
  Builder.build
    ~outputs:(fun y -> [ y ])
    Builder.(
      let* a = arange4 arange_params in
      mul_scalar 2. a)
  |> Err.or_raise ~pp_error:Builder.pp_error

let%expect_test
    "Symbolic4 -> Kernel: I64 Arange4 -> Mul_scalar produces an ordinary float \
     Stage, not a Stage_i64" =
  let stage_program = Eval_symbolic4.run g in
  Format.printf "stages: %d, stages_i64: %d@."
    (List.length stage_program.Stage_program.stages)
    (List.length stage_program.Stage_program.stages_i64);
  let out_id = List.hd g.Graph.Graph.outputs in
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
