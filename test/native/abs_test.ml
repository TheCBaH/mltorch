(* [Pointwise.Abs]: dtype-preserving absolute value.

   Floats: ATen's rule, with the sign of zero and NaN observable -- [|-0.|] is
   [+0.], infinities negate, a NaN stays a NaN. Printing the float32 BIT PATTERN
   rather than the value is what makes [-0.] versus [+0.] visible at all.

   Int64: exact, in wrapping arithmetic, including past 2^53 where a float
   round-trip would lose bits, and [abs min_int = min_int] as in ATen. The same
   graph also runs through the Symbolic -> Kernel route, which must agree. *)

open Graph_ir
open Graph_direct_fixtures

let bits f = Printf.sprintf "%08lx" (Int32.bits_of_float f)

let%expect_test "Direct: abs on the float32 edge cases" =
  let fixture =
    [|
      -1.5; -0.; 0.; 0.5; infinity; neg_infinity; nan; 1e-45; -1e-45; -3.4e38;
    |]
  in
  let x_shape = s1c (Array.length fixture) in
  let x = Tensor.materialize x_shape (fun c -> fixture.(chan c)) in
  let module A = Pointwise.Abs.Compute (Direct) in
  let out =
    Schedule.evaluate
      (Err.or_raise ~pp_error:Shape_error.pp
         (Pointwise.Abs.output_shape x_shape))
      (A.pixel x)
  in
  Array.iteri
    (fun i v ->
      let got = Tensor.read out (Vec6.coord ~n:0 ~t:0 ~d:0 ~h:0 ~w:0 ~c:i) in
      Format.printf "%-12g bits %s -> %-12g bits %s@." v (bits v) got (bits got))
    fixture;
  [%expect
    {|
    -1.5         bits bfc00000 -> 1.5          bits 3fc00000
    -0           bits 80000000 -> 0            bits 00000000
    0            bits 00000000 -> 0            bits 00000000
    0.5          bits 3f000000 -> 0.5          bits 3f000000
    inf          bits 7f800000 -> inf          bits 7f800000
    -inf         bits ff800000 -> inf          bits 7f800000
    nan          bits 7fc00000 -> nan          bits 7fc00000
    1e-45        bits 00000001 -> 1.4013e-45   bits 00000001
    -1e-45       bits 80000001 -> 1.4013e-45   bits 00000001
    -3.4e+38     bits ff7fc99e -> 3.4e+38      bits 7f7fc99e |}]

(* A graph input of the given format, carrying [values]. *)
let run_abs_graph ~fmt ~materialize values =
  let n = Array.length values in
  let result =
    let open Err.Syntax in
    let* g =
      lift_build
        Graph_builder.(
          build ~name:"abs" ~outputs:(fun r -> [ r ])
          @@
          let* x = input ~fmt ~shape:(s1c n) ~name:"x" () in
          abs ~name:"out" x)
    in
    let x = materialize (s1c n) (fun c -> values.(chan c)) in
    let* env =
      lift_eval (Eval_direct.run g ~inputs:(List.combine g.Graph.inputs [ x ]))
    in
    tensor_of_name g env "out"
  in
  Format.printf "%a@." (pp_result (pp_named_tensor "out")) result

let%expect_test "Direct graph: abs keeps float32 and int64 apart" =
  run_abs_graph
    ~fmt:Payload.(Fmt F32)
    ~materialize:Tensor.materialize [| -2.5; 0.; 3. |];
  run_abs_graph
    ~fmt:Payload.(Fmt I64)
    ~materialize:Tensor.materialize_i64 [| -3L; 0L; 7L |];
  [%expect
    {|
    out = tensor f32 [C=3] {2.5, 0, 3}
    out = tensor i64 [C=3] {3, 0, 7} |}]

let%expect_test
    "Direct graph: int64 abs is exact past 2^53 and wraps at min_int" =
  run_abs_graph
    ~fmt:Payload.(Fmt I64)
    ~materialize:Tensor.materialize_i64
    [|
      -9_007_199_254_740_993L;
      9_007_199_254_740_993L;
      -9_223_372_036_854_775_807L;
      Int64.min_int;
      Int64.max_int;
    |];
  [%expect
    {| out = tensor i64 [C=5] {9007199254740993, 9007199254740993, 9223372036854775807, -9223372036854775808, 9223372036854775807} |}]

(* The same int64 abs through the staged route: an [I64_load], a signed
   comparison and a [Select] over int64, evaluated by the Kernel interpreter. *)
let%expect_test "Symbolic -> Kernel: int64 abs agrees with Direct" =
  let params start stop =
    {
      Factory.Arange.start = Int64.to_float start;
      stop = Int64.to_float stop;
      step = 1.;
      fmt = Payload.(Fmt I64);
      exact = Some { Factory.Arange.Exact.start; stop; step = 1L };
    }
  in
  let build =
    Graph_builder.(
      build ~name:"abs_i64" ~outputs:(fun r -> [ r ])
      @@
      let* a =
        arange (params (-9_007_199_254_740_995L) (-9_007_199_254_740_991L))
      in
      abs a)
    |> Err.or_raise ~pp_error:Graph_builder.pp_error
  in
  let stage_program = Eval_symbolic.run build in
  Format.printf "stages: %d, stages_i64: %d@."
    (List.length stage_program.Stage_program.stages)
    (List.length stage_program.Stage_program.stages_i64);
  let out_id = List.hd build.Graph_ir.Graph.outputs in
  let kernel =
    Kernel_adapt.of_stage_program
      { stage_program with Stage_program.outputs = [] }
    |> Err.or_raise ~pp_error:Kernel_adapt.pp_error
  in
  let result =
    Kernel_eval.run kernel ~bind:(fun _ -> None)
    |> Err.or_raise ~pp_error:Kernel_eval.pp_error
  in
  (match Graph_ir.Tensor_id.Map.find_opt out_id result with
  | Some t -> Format.printf "kernel: %a@." Tensor.pp t
  | None -> print_endline "kernel: MISSING");
  let direct =
    Eval_direct.run build ~inputs:[]
    |> Err.or_raise ~pp_error:Eval_direct.pp_error
  in
  (match Graph_ir.Tensor_id.Map.find_opt out_id direct with
  | Some t -> Format.printf "direct: %a@." Tensor.pp t
  | None -> print_endline "direct: MISSING");
  [%expect
    {|
    stages: 0, stages_i64: 2
    kernel: tensor i64 [C=4] {9007199254740995, 9007199254740994, 9007199254740993, 9007199254740992}
    direct: tensor i64 [C=4] {9007199254740995, 9007199254740994, 9007199254740993, 9007199254740992} |}]
