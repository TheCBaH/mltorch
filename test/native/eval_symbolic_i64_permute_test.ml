(* The Symbolic/Kernel twin of `permute_i64_test.ml`'s own Direct-route
   fixture, the same shape as `eval_symbolic_i64_reshape_test.ml` -- see its
   own header for the full rationale. A real `Graph_builder` graph chaining
   an exact-I64 `Arange` into a `Permute` (swap W<->C), through
   `Eval_symbolic.run`'s new `Permute` dispatch. *)

open Graph_ir

let s n t d h w c = Vec6.shape ~n ~t ~d ~h ~w ~c

let swap_wc =
  [
    (Axis.N, Axis.N);
    (Axis.T, Axis.T);
    (Axis.D, Axis.D);
    (Axis.H, Axis.H);
    (Axis.W, Axis.C);
    (Axis.C, Axis.W);
  ]

(* Six distinct consecutive values past 2^53 (9007199254740993..998), the
   same threshold every other exactness fixture uses. *)
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
    build ~name:"i64_arange_permute" ~outputs:(fun r -> [ r ])
    @@
    let* a = arange arange_params in
    permute ~name:"out" swap_wc a)
  |> Err.or_raise ~pp_error:Graph_builder.pp_error

let%expect_test
    "Eval_symbolic: I64 Arange -> Permute builds two chained Stage_i64s" =
  let stage_program = Eval_symbolic.run build in
  Format.printf "stages: %d, stages_i64: %d@."
    (List.length stage_program.Stage_program.stages)
    (List.length stage_program.Stage_program.stages_i64);
  [%expect {| stages: 0, stages_i64: 2 |}]

(* Same [~outputs:[]] sidestep [eval_symbolic_i64_reshape_test.ml]'s own
   second test already documents -- see that file's comment. *)
let%expect_test
    "Symbolic -> Kernel: I64 Arange -> Permute survives \
     Kernel_adapt/Kernel_eval past 2^53" =
  let stage_program =
    { (Eval_symbolic.run build) with Stage_program.outputs = [] }
  in
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
    {| tensor i64 [W=6 C=1] {9007199254740993, 9007199254740994, 9007199254740995, 9007199254740996, 9007199254740997, 9007199254740998} |}]
