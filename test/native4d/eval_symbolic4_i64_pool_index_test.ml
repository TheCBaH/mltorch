(* The Native4D twin of `test/native/eval_symbolic_i64_pool_index_test.ml`: the
   same 4x4 input (5*i mod 11) through relu -> max_pool2d_with_indices, so the
   index stage reads a computed float stage. Hand-computed answer: values
   9,10,10,9 at flat input positions 4,2,13,15, the index an exact int64. *)

open Native4d

let shape = Shape4.of_ints ~n:1 ~h:4 ~w:4 ~c:1

let params : Pool.MaxPool2dWithIndices.params =
  {
    ceil_mode = false;
    kernel = { h = Dim.extent 2; w = Dim.extent 2 };
    stride = { h = Op_config.Pos.of_int 2; w = Op_config.Pos.of_int 2 };
    pad = { h = Op_config.Nonneg.of_int 0; w = Op_config.Nonneg.of_int 0 };
  }

let g =
  Builder.build ~outputs:Fun.id
    Builder.(
      let* x = input ~shape () in
      let* r = relu x in
      max_pool2d_with_indices params r)
  |> Err.or_raise ~pp_error:Builder.pp_error

let x =
  Tensor.materialize (Shape4.to_vec6 shape) (fun c ->
      let i =
        (Dim.to_int (Vec6.get c Axis.H) * 4) + Dim.to_int (Vec6.get c Axis.W)
      in
      float_of_int (5 * i mod 11))

let show name t = Fmt.pr "%s = %a@." name Tensor.pp t

let%expect_test
    "Native4D: the index output is I64 on Direct and the Kernel route" =
  let value_id, index_id =
    match g.Graph.Graph.outputs with [ v; i ] -> (v, i) | _ -> assert false
  in
  let direct =
    Eval_direct4.run g ~constants:[]
      ~inputs:(List.combine g.Graph.Graph.inputs [ x ])
    |> Err.or_raise ~pp_error:Eval_direct4.pp_error
  in
  show "direct value" (Tensor_id.Map.find value_id direct);
  show "direct index" (Tensor_id.Map.find index_id direct);
  let program = Eval_symbolic4.run g in
  Fmt.pr "stages: %d, stages_i64: %d@."
    (List.length program.Stage_program.stages)
    (List.length program.Stage_program.stages_i64);
  let kernel =
    Kernel_adapt.of_stage_program { program with Stage_program.outputs = [] }
    |> Err.or_raise ~pp_error:Kernel_adapt.pp_error
  in
  let result =
    Kernel_eval.run kernel ~bind:(fun _ -> Some x)
    |> Err.or_raise ~pp_error:Kernel_eval.pp_error
  in
  show "kernel value" (Tensor_id.Map.find value_id result);
  show "kernel index" (Tensor_id.Map.find index_id result);
  Fmt.pr "kernel = direct: %b@."
    (Tensor.equal_bits
       (Tensor_id.Map.find index_id result)
       (Tensor_id.Map.find index_id direct));
  [%expect
    {|
    direct value = tensor f32 [H=2 W=2 C=1] {9, 10, 10, 9}
    direct index = tensor i64 [H=2 W=2 C=1] {4, 2, 13, 15}
    stages: 2, stages_i64: 1
    kernel value = tensor f32 [H=2 W=2 C=1] {9, 10, 10, 9}
    kernel index = tensor i64 [H=2 W=2 C=1] {4, 2, 13, 15}
    kernel = direct: true |}]

(* A dead index is neither computed nor allocated on Direct: absent from the
   result map, while the live one (control) is present. *)
let%expect_test "Native4D Direct does not allocate a dead index" =
  let run ~live_index =
    let g =
      Builder.build
        ~outputs:(fun ids ->
          match ids with
          | [ v; i ] -> if live_index then [ v; i ] else [ v ]
          | l -> l)
        Builder.(
          let* x = input ~shape () in
          max_pool2d_with_indices params x)
      |> Err.or_raise ~pp_error:Builder.pp_error
    in
    let env =
      Eval_direct4.run g ~constants:[]
        ~inputs:(List.combine g.Graph.Graph.inputs [ x ])
      |> Err.or_raise ~pp_error:Eval_direct4.pp_error
    in
    (* t1 is the value edge and t2 the index edge, in allocation order. *)
    Fmt.pr "live index=%b: value present %b, index present %b@." live_index
      (Tensor_id.Map.mem (Tensor_id.of_int 1) env)
      (Tensor_id.Map.mem (Tensor_id.of_int 2) env)
  in
  run ~live_index:true;
  run ~live_index:false;
  [%expect
    {|
    live index=true: value present true, index present true
    live index=false: value present true, index present false |}]
