(* The Native4D twin of `test/native/eval_symbolic_i64_to_copy_long_test.ml` --
   see its header for the rationale. An int64 stage ([To_copy(Long)]) reads a
   computed float stage, and a float [To_copy] reads it back; the Kernel result
   is compared with [Eval_direct4] and the integers are worked out by hand. *)

open Native4d

let shape = Shape4.of_ints ~n:1 ~h:1 ~w:1 ~c:4

let g =
  Builder.build
    ~outputs:(fun y -> [ y ])
    Builder.(
      let* a = input ~shape () in
      let* s = add_scalar (Json_util.f32_to_f32 52.) a in
      let* l = to_copy Pointwise.To_copy.Long s in
      to_copy Pointwise.To_copy.Float l)
  |> Err.or_raise ~pp_error:Builder.pp_error

let a =
  let cells = [| -1.9; -0.5; 2.4; 3.9 |] in
  Tensor.materialize (Shape4.to_vec6 shape) (fun c ->
      cells.(Dim.to_int (Vec6.get c Axis.C)))

let ints t =
  List.init 4 (fun c ->
      match Tensor.read_i64_at6 t (function Axis.C -> c | _ -> 0) with
      | Ok v -> Int64.to_string v
      | Error _ -> "not i64")
  |> String.concat ","

let%expect_test "Symbolic4 -> Kernel: an int64 stage reads a computed float" =
  let program = Eval_symbolic4.run g in
  Format.printf "stages: %d, stages_i64: %d@."
    (List.length program.Stage_program.stages)
    (List.length program.Stage_program.stages_i64);
  let program = { program with Stage_program.outputs = [] } in
  let kernel =
    Kernel_adapt.of_stage_program program
    |> Err.or_raise ~pp_error:Kernel_adapt.pp_error
  in
  let result =
    Kernel_eval.run kernel ~bind:(fun _ -> Some a)
    |> Err.or_raise ~pp_error:Kernel_eval.pp_error
  in
  let direct =
    Eval_direct4.run g ~constants:[]
      ~inputs:(List.combine g.Graph.Graph.inputs [ a ])
    |> Err.or_raise ~pp_error:Eval_direct4.pp_error
  in
  let id = Tensor_id.of_int in
  Format.printf "t2 = %s@." (ints (Tensor_id.Map.find (id 2) result));
  List.iter
    (fun i ->
      Format.printf "t%d kernel = direct: %b@." i
        (Tensor.equal_bits
           (Tensor_id.Map.find (id i) result)
           (Tensor_id.Map.find (id i) direct)))
    [ 2; 3 ];
  [%expect
    {|
    stages: 2, stages_i64: 1
    t2 = 50,51,54,55
    t2 kernel = direct: true
    t3 kernel = direct: true |}]
