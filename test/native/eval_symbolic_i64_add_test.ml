(* The Symbolic/Kernel twin of `graph_direct_pointwise_test.ml`'s own I64
   Add/Sub/Mul Direct-route fixtures, the same shape as
   `eval_symbolic_i64_reshape_test.ml` -- see its own header for the full
   rationale. Two real exact-I64 `Arange`s (both producing a `Stage_i64.t`)
   feed a real `Add`/`Sub`/`Mul`, through `Eval_symbolic.run`'s new
   tensor-tensor dispatch. *)

open Graph_ir

(* [a]: six distinct consecutive values past 2^53
   (9007199254740993..998); [b]: small values 1..6, so [a]'s own exactness is
   what is being tested (b never approaches 2^53). *)
let a_params =
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

let b_params =
  {
    Factory.Arange.start = 1.;
    stop = 7.;
    step = 1.;
    fmt = Payload.(Fmt I64);
    exact = Some { Factory.Arange.Exact.start = 1L; stop = 7L; step = 1L };
  }

let build op_name op =
  Graph_builder.(
    build ~name:("i64_arange_" ^ op_name) ~outputs:(fun r -> [ r ])
    @@
    let* a = arange a_params in
    let* b = arange b_params in
    op a b)
  |> Err.or_raise ~pp_error:Graph_builder.pp_error

let check op_name op =
  let build = build op_name op in
  let stage_program = Eval_symbolic.run build in
  Format.printf "%s stages: %d, stages_i64: %d@." op_name
    (List.length stage_program.Stage_program.stages)
    (List.length stage_program.Stage_program.stages_i64);
  let stage_program = { stage_program with Stage_program.outputs = [] } in
  let out_id = List.hd build.Graph.outputs in
  let kernel =
    Kernel_adapt.of_stage_program stage_program
    |> Err.or_raise ~pp_error:Kernel_adapt.pp_error
  in
  let result =
    Kernel_eval.run kernel ~bind:(fun _ -> None)
    |> Err.or_raise ~pp_error:Kernel_eval.pp_error
  in
  match Tensor_id.Map.find_opt out_id result with
  | Some t -> Format.printf "%s = %a@." op_name Tensor.pp t
  | None -> print_endline "MISSING"

let%expect_test
    "Symbolic -> Kernel: I64 Arange -> Add/Sub/Mul builds three Stage_i64s and \
     survives Kernel_adapt/Kernel_eval past 2^53" =
  check "add" Graph_builder.add;
  check "sub" Graph_builder.sub;
  check "mul" Graph_builder.mul;
  [%expect
    {|
    add stages: 0, stages_i64: 3
    add = tensor i64 [C=6] {9007199254740994, 9007199254740996, 9007199254740998, 9007199254741000, 9007199254741002, 9007199254741004}
    sub stages: 0, stages_i64: 3
    sub = tensor i64 [C=6] {9007199254740992, 9007199254740992, 9007199254740992, 9007199254740992, 9007199254740992, 9007199254740992}
    mul stages: 0, stages_i64: 3
    mul = tensor i64 [C=6] {9007199254740993, 18014398509481988, 27021597764222985, 36028797018963984, 45035996273704985, 54043195528445988}
    |}]
