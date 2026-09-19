(* Live max-pool indices are exact int64 on every route (V13). relu -> max_pool2d
   with both outputs live: the index stage reads a COMPUTED float stage, the
   D10 shape a real network has. The 4x4 input is (5*i mod 11) for flat position
   i, worked out by hand:

     h0:  0  5 10  4
     h1:  9  3  8  2
     h2:  7  1  6  0
     h3:  5 10  4  9

   2x2/stride-2 windows give values 9,10,10,9 at flat input positions
   ih*4 + iw = 4, 2, 13, 15. *)

let shape = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:4 ~w:4 ~c:1

let params : Pool.MaxPool2dWithIndices.params =
  {
    ceil_mode = false;
    kernel = { h = Dim.extent 2; w = Dim.extent 2 };
    stride = { h = Op_config.Pos.of_int 2; w = Op_config.Pos.of_int 2 };
    pad = { h = Op_config.Nonneg.of_int 0; w = Op_config.Nonneg.of_int 0 };
  }

let g =
  Graph_builder.build ~name:"pool_index"
    ~outputs:(fun (v, i) -> [ v; i ])
    Graph_builder.(
      let* x = input ~shape ~name:"x" () in
      let* r = relu x in
      max_pool2d_with_indices params r)
  |> Err.or_raise ~pp_error:Graph_builder.pp_error

let x =
  Tensor.materialize shape (fun c ->
      let i =
        (Dim.to_int (Vec6.get c Axis.H) * 4) + Dim.to_int (Vec6.get c Axis.W)
      in
      float_of_int (5 * i mod 11))

let show name t = Fmt.pr "%s = %a@." name Tensor.pp t

let%expect_test
    "the index output is I64 on the graph, Direct and the Kernel route" =
  let value_id, index_id =
    match g.Graph_ir.Graph.outputs with [ v; i ] -> (v, i) | _ -> assert false
  in
  let sig_fmt id =
    match (Tensor_id.Map.find id g.Graph_ir.Graph.tensors).Tensor_sig.fmt with
    | Payload.Fmt f -> Payload.fmt_name f
  in
  Fmt.pr "declared: value %s, index %s@." (sig_fmt value_id) (sig_fmt index_id);
  let direct =
    Eval_direct.run g ~inputs:(List.combine g.Graph_ir.Graph.inputs [ x ])
    |> Err.or_raise ~pp_error:Eval_direct.pp_error
  in
  show "direct value" (Tensor_id.Map.find value_id direct);
  show "direct index" (Tensor_id.Map.find index_id direct);
  let program = Eval_symbolic.run g in
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
       (Tensor_id.Map.find index_id direct)
    && Tensor.equal_bits
         (Tensor_id.Map.find value_id result)
         (Tensor_id.Map.find value_id direct));
  [%expect
    {|
    declared: value f32, index i64
    direct value = tensor f32 [H=2 W=2 C=1] {9, 10, 10, 9}
    direct index = tensor i64 [H=2 W=2 C=1] {4, 2, 13, 15}
    stages: 2, stages_i64: 1
    kernel value = tensor f32 [H=2 W=2 C=1] {9, 10, 10, 9}
    kernel index = tensor i64 [H=2 W=2 C=1] {4, 2, 13, 15}
    kernel = direct: true |}]

(* [Max_dim]'s index is I64 too. A row [5, 3, 5, 1] ties at positions 0 and 2;
   the incumbent (first) index wins, so the answer is value 5, index 0. *)
let%expect_test "max_dim's index output is I64 on Direct and the Kernel route" =
  let row = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:1 ~w:4 ~c:1 in
  let g =
    Graph_builder.build ~name:"max_dim_index"
      ~outputs:(fun (v, i) -> [ v; i ])
      Graph_builder.(
        let* x = input ~shape:row ~name:"x" () in
        let* r = relu x in
        max_dim { Reduce.MaxDim.axis = Axis.W; keepdim = false } r)
    |> Err.or_raise ~pp_error:Graph_builder.pp_error
  in
  let x =
    Tensor.materialize row (fun c ->
        [| 5.; 3.; 5.; 1. |].(Dim.to_int (Vec6.get c Axis.W)))
  in
  let value_id, index_id =
    match g.Graph_ir.Graph.outputs with [ v; i ] -> (v, i) | _ -> assert false
  in
  let direct =
    Eval_direct.run g ~inputs:(List.combine g.Graph_ir.Graph.inputs [ x ])
    |> Err.or_raise ~pp_error:Eval_direct.pp_error
  in
  show "direct value" (Tensor_id.Map.find value_id direct);
  show "direct index" (Tensor_id.Map.find index_id direct);
  let kernel =
    Kernel_adapt.of_stage_program
      { (Eval_symbolic.run g) with Stage_program.outputs = [] }
    |> Err.or_raise ~pp_error:Kernel_adapt.pp_error
  in
  let result =
    Kernel_eval.run kernel ~bind:(fun _ -> Some x)
    |> Err.or_raise ~pp_error:Kernel_eval.pp_error
  in
  show "kernel index" (Tensor_id.Map.find index_id result);
  [%expect
    {|
    direct value = tensor f32 [C=1] {5}
    direct index = tensor i64 [C=1] {0}
    kernel index = tensor i64 [C=1] {0} |}]
