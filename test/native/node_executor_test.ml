(* T2.4's mutation, kept as a permanent regression test, the [Node_executor]
   twin of [region_compute_test.ml]'s own "region_executor: default is
   inert, an explicit override is honored" (T1.2 there): [?node_executor]
   omitted must still reach [Eval_direct_compute.compute] (proving the
   default is inert), and an EXPLICIT override must actually be the executor
   that runs (proving the parameter is really wired through [eval_node] to
   the [compute] call site, not dropped somewhere in the chain). [Relu] is
   the subject: a single-output, non-Region-authored op that reaches the
   default float-pixel arm of [Eval_direct_compute.compute]. *)

let%expect_test
    "node_executor: default is inert, an explicit override is honored" =
  let shape = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:1 ~w:1 ~c:4 in
  let x =
    Tensor.materialize shape (fun coord ->
        -2. +. float_of_int (Dim.to_int (Vec6.get coord Axis.C)))
  in
  let g =
    Err.or_raise ~pp_error:Graph_builder.pp_error
      Graph_builder.(
        build ~name:"node_executor_relu" ~outputs:(fun r -> [ r ])
        @@
        let* xi = input ~shape ~name:"x" () in
        relu xi)
  in
  let inputs = List.combine g.Graph_ir.Graph.inputs [ x ] in
  let output = List.hd g.Graph_ir.Graph.outputs in
  let output_shape =
    (Tensor_id.Map.find output g.Graph_ir.Graph.tensors).Tensor_sig.shape
  in
  let wrong_value = 12345. in
  let wrong_node_executor : Node_executor.t =
    {
      run =
        (fun _ _ ~output:_ ~out_shape ~operands:_ ~direct:_ ->
          Err.return (Tensor.materialize out_shape (fun _ -> wrong_value)));
    }
  in
  let run ?node_executor () =
    let env =
      Err.or_raise ~pp_error:Eval_direct.pp_error
        (Eval_direct.run ?node_executor ~inputs g)
    in
    Tensor_id.Map.find output env
  in
  let omitted = run () in
  let explicit_default = run ~node_executor:Node_executor.default () in
  let overridden = run ~node_executor:wrong_node_executor () in
  let default_inert = Tensor.equal_bits omitted explicit_default in
  let override_differs = not (Tensor.equal_bits omitted overridden) in
  let override_took_effect = ref true in
  Vec6.iter output_shape (fun coord ->
      if Tensor.read overridden coord <> wrong_value then
        override_took_effect := false);
  let override_took_effect = !override_took_effect in
  Fmt.pr "default_inert=%b override_differs=%b override_took_effect=%b@."
    default_inert override_differs override_took_effect;
  [%expect
    {| default_inert=true override_differs=true override_took_effect=true |}]
