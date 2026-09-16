(* The Symbolic/Kernel twin of [i64_acceptance_test.ml]'s own Direct-route
   Arange fixtures: [Eval_symbolic.run]'s new exact-Arange dispatch (see its
   own doc comment) building a real [Stage_program.Stage_i64.t] from a real
   [Graph_builder] graph, then that value surviving [Kernel_adapt.
   of_stage_program] -> [Kernel_eval.run] with its exact past-2^53 value
   intact -- the D09/D10 "narrowest real slice" this session's own tracker
   entry scoped out: a standalone int64 Arange with no int64-to-int64 or
   int64-to-float CONSUMER (Reshape/Permute/etc's own Symbolic dispatch is a
   separate, still-open gap; see the tracker), so this only proves the
   PRODUCER side of the Stage_program/Kernel representation gap is closed. *)

open Graph_ir

let build params =
  Graph_builder.(
    build ~name:"i64_arange" ~outputs:(fun r -> [ r ]) @@ arange params)
  |> Err.or_raise ~pp_error:Graph_builder.pp_error

(* Past 2^53: three distinct consecutive values that a float round trip would
   corrupt (9007199254740993/94/95 do not all survive an IEEE double), the
   same threshold every other exactness fixture in this tracker's slice
   uses. *)
let exact =
  {
    Factory.Arange.Exact.start = 9_007_199_254_740_993L;
    stop = 9_007_199_254_740_996L;
    step = 1L;
  }

let params =
  {
    Factory.Arange.start = 9.007199254740992e15;
    stop = 9.007199254740996e15;
    step = 1.;
    fmt = Payload.(Fmt I64);
    exact = Some exact;
  }

let%expect_test
    "Eval_symbolic: exact I64 Arange builds a Stage_i64, not a Stage" =
  let g = build params in
  let stage_program = Eval_symbolic.run g in
  Format.printf "stages: %d, stages_i64: %d@."
    (List.length stage_program.Stage_program.stages)
    (List.length stage_program.Stage_program.stages_i64);
  Format.printf "%a@." Stage_program.pp stage_program;
  [%expect
    {|
    stages: 0, stages_i64: 1
    inputs:
    t0 = (9007199254740993 + (1 * i64_of_index(C)))
    outputs: t0
    |}]

(* [of_stage_program]'s own [analyse] validates [Stage_program.outputs]
   unconditionally against [stages] (float) and the boundary table alone,
   BEFORE the [~outputs] argument is even consulted -- so an int64-only
   output raises [`Unknown_program_output] no matter what [~outputs] is
   passed (confirmed by trying [~outputs:[]] first and getting exactly that).
   Recognising an int64-only graph output is real, separately-scoped work
   this fixture does NOT attempt (see this session's own tracker entry) --
   sidestepped here by overriding the record's own [outputs] field to [],
   which [Eval_symbolic.run] cannot do itself (it always mirrors the source
   graph's real output list). [Kernel_eval.run]'s own returned map still
   contains every [values_i64] entry unconditionally regardless of whether it
   is a declared [Kernel.Output.t] -- confirmed by reading [Kernel_eval.run]'s
   own [Tensor_id.Map.union] with [materialize_values_i64]'s result. *)
let%expect_test
    "Symbolic -> Kernel: exact I64 Arange survives Kernel_adapt/Kernel_eval \
     past 2^53" =
  let g = build params in
  let stage_program =
    { (Eval_symbolic.run g) with Stage_program.outputs = [] }
  in
  let arange_id = List.hd g.Graph.outputs in
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
