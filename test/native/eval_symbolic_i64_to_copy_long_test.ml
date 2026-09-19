(* Symbolic -> Kernel for [To_copy(Long)] on an F32 operand: mvitv2's
   [add.Tensor -> _to_copy(Long)] shape, an int64 stage reading a COMPUTED float
   stage (the tracker's D10), then a float [To_copy] reading the int64 stage.
   Each result is compared with [Eval_direct] cell for cell, and the expected
   integers are worked out by hand: truncation toward zero after the F32 add,
   and 1e18 -- which F32 can only hold as 999999984306749440 -- surviving exactly
   past 2^53 in the int64 storage. *)

let shape = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:1 ~w:1 ~c:4
let chan c = Dim.to_int (Vec6.get c Axis.C)
let input cells = Tensor.materialize shape (fun c -> cells.(chan c))

let build name m =
  Graph_builder.(build ~name ~outputs:(fun r -> [ r ]) m)
  |> Err.or_raise ~pp_error:Graph_builder.pp_error

let ints t =
  List.init 4 (fun c ->
      match Tensor.read_i64_at6 t (function Axis.C -> c | _ -> 0) with
      | Ok v -> Int64.to_string v
      | Error _ -> "not i64")
  |> String.concat ","

let floats t =
  List.init 4 (fun c -> Tensor.read_at_raw t (function Axis.C -> c | _ -> 0))
  |> List.map (Printf.sprintf "%g")
  |> String.concat ","

let run_kernel g a =
  let program = Eval_symbolic.run g in
  Fmt.pr "stages: %d, stages_i64: %d@."
    (List.length program.Stage_program.stages)
    (List.length program.Stage_program.stages_i64);
  (* An int64 stage cannot be a graph output of a stage program, so, as in the
     sibling int64 fixtures, results are read from the kernel's result map. *)
  let program = { program with Stage_program.outputs = [] } in
  let kernel =
    Kernel_adapt.of_stage_program program
    |> Err.or_raise ~pp_error:Kernel_adapt.pp_error
  in
  Kernel_eval.run kernel ~bind:(fun _ -> Some a)
  |> Err.or_raise ~pp_error:Kernel_eval.pp_error

let run_direct g a =
  Eval_direct.run g ~inputs:(List.combine g.Graph_ir.Graph.inputs [ a ])
  |> Err.or_raise ~pp_error:Eval_direct.pp_error

(* a -> t1 = a + 52. -> t2 = long t1 -> t3 = float t2 *)
let%expect_test "an int64 stage reads a computed float stage, a float reads it"
    =
  let g =
    build "to_long_chain"
      Graph_builder.(
        let* a = input ~shape ~name:"a" () in
        let* s = add_scalar 52. a in
        let* l = to_copy Pointwise.To_copy.Long s in
        to_copy Pointwise.To_copy.Float l)
  in
  let a = input [| -1.9; -0.5; 2.4; 3.9 |] in
  let kernel = run_kernel g a and direct = run_direct g a in
  List.iter
    (fun id ->
      let k = Tensor_id.Map.find (Tensor_id.of_int id) kernel
      and d = Tensor_id.Map.find (Tensor_id.of_int id) direct in
      Fmt.pr "t%d kernel = direct: %b@." id (Tensor.equal_bits k d))
    [ 2; 3 ];
  Fmt.pr "t2 = %s@.t3 = %s@."
    (ints (Tensor_id.Map.find (Tensor_id.of_int 2) kernel))
    (floats (Tensor_id.Map.find (Tensor_id.of_int 3) kernel));
  [%expect
    {|
    stages: 2, stages_i64: 1
    t2 kernel = direct: true
    t3 kernel = direct: true
    t2 = 50,51,54,55
    t3 = 50,51,54,55 |}]

let%expect_test "a value F32 holds only past 2^53 stays exact in int64" =
  let g =
    build "to_long_big"
      Graph_builder.(
        let* a = input ~shape ~name:"a" () in
        to_copy Pointwise.To_copy.Long a)
  in
  let a = input [| 1e18; -1e18; 0.99; 9007199254740993. |] in
  let kernel = run_kernel g a and direct = run_direct g a in
  let k = Tensor_id.Map.find (Tensor_id.of_int 1) kernel
  and d = Tensor_id.Map.find (Tensor_id.of_int 1) direct in
  Fmt.pr "kernel = direct: %b@.t1 = %s@." (Tensor.equal_bits k d) (ints k);
  [%expect
    {|
    stages: 0, stages_i64: 1
    kernel = direct: true
    t1 = 999999984306749440,-999999984306749440,0,9007199254740992 |}]

let%expect_test "a NaN or out-of-range operand is a structured error" =
  let g =
    build "to_long_bad"
      Graph_builder.(
        let* a = input ~shape ~name:"a" () in
        to_copy Pointwise.To_copy.Long a)
  in
  let a = input [| 1.; Float.nan; 2.; 3. |] in
  (match run_kernel g a with
  | _ -> print_endline "no error"
  | exception Err.Exn.E _ -> print_endline "structured error");
  [%expect {|
    stages: 0, stages_i64: 1
    structured error |}]
