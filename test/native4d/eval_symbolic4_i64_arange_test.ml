(* The Native4D twin of `test/native/eval_symbolic_i64_arange_test.ml` -- see
   its own header for the full rationale. [Kernel_adapt]/[Kernel_eval] are the
   SAME (Native, not Native4D-specific) modules used there, confirmed by
   `compute_test.ml`'s own existing "authored Regions carry into the Kernel
   unchanged" fixture. *)

open Native4d

let build m =
  Builder.build ~outputs:(fun y -> [ y ]) m
  |> Err.or_raise ~pp_error:Builder.pp_error

let exact =
  {
    Factory.Arange.Exact.start = 9_007_199_254_740_993L;
    stop = 9_007_199_254_740_996L;
    step = 1L;
  }

let params =
  {
    Ops4.Arange4.start = 9.007199254740992e15;
    stop = 9.007199254740996e15;
    step = 1.;
    fmt = Payload.(Fmt I64);
    exact = Some exact;
  }

let%expect_test
    "Eval_symbolic4: exact I64 Arange4 builds a Stage_i64, not a Stage" =
  let g = build (Builder.arange4 params) in
  let stage_program = Eval_symbolic4.run g in
  Format.printf "stages: %d, stages_i64: %d@."
    (List.length stage_program.Stage_program.stages)
    (List.length stage_program.Stage_program.stages_i64);
  [%expect {| stages: 0, stages_i64: 1 |}]

let%expect_test
    "Symbolic4 -> Kernel: exact I64 Arange4 survives Kernel_adapt/Kernel_eval \
     past 2^53" =
  let g = build (Builder.arange4 params) in
  let stage_program =
    { (Eval_symbolic4.run g) with Stage_program.outputs = [] }
  in
  let arange_id = List.hd g.Graph.Graph.outputs in
  let kernel =
    Kernel_adapt.of_stage_program stage_program
    |> Err.or_raise ~pp_error:Kernel_adapt.pp_error
  in
  let result =
    Kernel_eval.run kernel ~bind:(fun _ -> None)
    |> Err.or_raise ~pp_error:Kernel_eval.pp_error
  in
  (match Tensor_id.Map.find_opt arange_id result with
  | Some t -> Format.printf "%a@." Tensor.pp t
  | None -> print_endline "MISSING");
  [%expect
    {| tensor i64 [C=3] {9007199254740993, 9007199254740994, 9007199254740995} |}]
