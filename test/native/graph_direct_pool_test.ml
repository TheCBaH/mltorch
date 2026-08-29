(* Split out of what was graph_test.ml see graph_direct_fixtures.ml. *)

open Graph_ir
open Graph_direct_fixtures

let mp_params =
  {
    Pool.MaxPool2d.ceil_mode = false;
    kernel = Op_config.Hw.{ h = Dim.extent 2; w = Dim.extent 2 };
    stride =
      Op_config.Hw.{ h = Op_config.Pos.of_int 2; w = Op_config.Pos.of_int 2 };
    pad =
      Op_config.Hw.
        { h = Op_config.Nonneg.of_int 0; w = Op_config.Nonneg.of_int 0 };
  }

let%expect_test "Direct graph: max_pool2d_with_indices (two outputs)" =
  let result =
    let open Err.Syntax in
    let* g =
      lift_build
        Graph_builder.(
          build ~name:"mp" ~outputs:(fun (v, i) -> [ v; i ])
          @@
          let* x = input ~shape:(s 1 1 1 4 4 1) ~name:"x" () in
          max_pool2d_with_indices ~name:"vals" mp_params x)
    in
    let x =
      Tensor.materialize (s 1 1 1 4 4 1) (fun c ->
          float_of_int
            ((Dim.to_int (Vec6.get c Axis.H) * 4)
            + Dim.to_int (Vec6.get c Axis.W)))
    in
    let* env =
      lift_eval (Eval_direct.run g ~inputs:(List.combine g.Graph.inputs [ x ]))
    in
    match g.Graph.outputs with
    | [ vid; iid ] ->
        Err.return (Tensor_id.Map.find vid env, Tensor_id.Map.find iid env)
    | _ -> Err.fail (`Missing_named_tensor "two outputs")
  in
  Format.printf "%a@."
    (pp_result (pp_named_tensor_pair "values" "indices"))
    result;
  [%expect
    {|
    values = tensor f32 [H=2 W=2 C=1] {5, 7, 13, 15}
    indices = tensor f32 [H=2 W=2 C=1] {5, 7, 13, 15} |}]

(* Reshape reinterprets the same flat buffer under a new shape (contiguous). *)
