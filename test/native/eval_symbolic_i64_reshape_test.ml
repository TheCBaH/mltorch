(* The Symbolic/Kernel twin of [reshape_i64_test.ml]'s own Direct-route
   fixture: a real [Graph_builder] graph chaining an exact-I64 [Arange] into
   a non-flattening [Reshape], through [Eval_symbolic.run]'s new [Reshape]
   dispatch (see its own doc comment) -- the first real CONSUMER of another
   [Stage_i64.t] this tracker's Symbolic/Kernel route has, closing the
   P4.1/P5.1 entry's own "still open" item: "any int64-to-int64 ... CONSUMER
   reachable through Symbolic". [Kernel.Value_i64.t]'s existing
   forward-reference machinery ([check_values_i64_order]/
   [materialize_values_i64]) resolves the Reshape's [I64_load] of the
   Arange's own [values_i64] entry with no further change -- this fixture is
   the proof, not merely an assertion that it should work. *)

open Graph_ir

let s n t d h w c = Vec6.shape ~n ~t ~d ~h ~w ~c

(* Six distinct consecutive values past 2^53 (9007199254740993..998): a float
   round trip would collapse the first two, the same threshold every other
   exactness fixture in this tracker's slice uses. *)
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
    build ~name:"i64_arange_reshape" ~outputs:(fun r -> [ r ])
    @@
    let* a = arange arange_params in
    reshape ~name:"out" { Reshape.Reshape.shape = s 1 1 1 2 3 1 } a)
  |> Err.or_raise ~pp_error:Graph_builder.pp_error

let%expect_test
    "Eval_symbolic: I64 Arange -> Reshape builds two chained Stage_i64s" =
  let stage_program = Eval_symbolic.run build in
  Format.printf "stages: %d, stages_i64: %d@."
    (List.length stage_program.Stage_program.stages)
    (List.length stage_program.Stage_program.stages_i64);
  Format.printf "%a@." Stage_program.pp stage_program;
  [%expect
    {|
    stages: 0, stages_i64: 2
    inputs:
    t0 = (9007199254740993 + (1 * i64_of_index(C)))
    t1 = t0[floor_div(3*2*N+T+D+H+W+C,6)+-1*floor_div(3*2*N+T+D+H+W+C,6),floor_div(3*2*N+T+D+H+W+C,6)+-1*floor_div(3*2*N+T+D+H+W+C,6),floor_div(3*2*N+T+D+H+W+C,6)+-1*floor_div(3*2*N+T+D+H+W+C,6),floor_div(3*2*N+T+D+H+W+C,6)+-1*floor_div(3*2*N+T+D+H+W+C,6),floor_div(3*2*N+T+D+H+W+C,6)+-1*floor_div(3*2*N+T+D+H+W+C,6),3*2*N+T+D+H+W+C+-6*floor_div(3*2*N+T+D+H+W+C,6)]
    outputs: t1
    |}]

(* Same [~outputs:[]] sidestep [eval_symbolic_i64_arange_test.ml]'s own
   second test already documents: [Kernel_adapt.analyse]'s [required] call
   validates [Stage_program.outputs] against [stages] (float) and the
   boundary table alone, unconditionally, so an int64-only graph output
   raises [`Unknown_program_output] regardless of [~outputs] -- not attempted
   here either, since recognising an int64-only output is separately scoped
   (see that file's own comment). *)
let%expect_test
    "Symbolic -> Kernel: I64 Arange -> Reshape survives \
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
    {| tensor i64 [H=2 W=3 C=1] {9007199254740993, 9007199254740994, 9007199254740995, 9007199254740996, 9007199254740997, 9007199254740998} |}]
