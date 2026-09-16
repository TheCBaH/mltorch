(* The Native4D twin of `test/native/eval_symbolic_i64_reshape_test.ml` -- see
   its own header for the full rationale. [Kernel_adapt]/[Kernel_eval] are the
   SAME (Native, not Native4D-specific) modules used there, confirmed by
   `eval_symbolic4_i64_arange_test.ml`'s own existing precedent. *)

open Native4d

let build m =
  Builder.build ~outputs:(fun y -> [ y ]) m
  |> Err.or_raise ~pp_error:Builder.pp_error

(* Six distinct consecutive values past 2^53 (9007199254740993..998), the
   same threshold every other exactness fixture in this tracker's slice
   uses. *)
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
  build
    Builder.(
      let* a = arange4 arange_params in
      reshape4 (Shape4.of_ints ~n:1 ~h:2 ~w:3 ~c:1) a)

let%expect_test
    "Eval_symbolic4: I64 Arange4 -> Reshape4 builds two chained Stage_i64s" =
  let stage_program = Eval_symbolic4.run g in
  Format.printf "stages: %d, stages_i64: %d@."
    (List.length stage_program.Stage_program.stages)
    (List.length stage_program.Stage_program.stages_i64);
  [%expect {| stages: 0, stages_i64: 2 |}]

(* Same [~outputs:[]] sidestep [eval_symbolic4_i64_arange_test.ml]'s own
   second test already documents -- see that file's comment. *)
let%expect_test
    "Symbolic4 -> Kernel: I64 Arange4 -> Reshape4 survives \
     Kernel_adapt/Kernel_eval past 2^53" =
  let stage_program =
    { (Eval_symbolic4.run g) with Stage_program.outputs = [] }
  in
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
    {| tensor i64 [H=2 W=3 C=1] {9007199254740993, 9007199254740994, 9007199254740995, 9007199254740996, 9007199254740997, 9007199254740998} |}]
