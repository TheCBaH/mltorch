(* Split out of what was graph_symbolic_test.ml see graph_symbolic_fixtures.ml. *)

open Graph_ir
open Graph_symbolic_fixtures

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

let%expect_test "Symbolic graph: max_pool2d_with_indices ground matches Direct"
    =
  let result =
    let open Err.Syntax in
    let* g =
      lift_build
        Graph_builder.(
          build ~name:"mp" ~outputs:(fun (v, i) -> [ v; i ])
          @@
          let* x = input ~shape:(s 1 1 1 4 4 1) ~name:"x" () in
          max_pool2d_with_indices mp_params x)
    in
    let prog = Eval_symbolic.run g in
    Format.printf "%a@." Stage_program.pp prog;
    let x =
      Tensor.materialize (s 1 1 1 4 4 1) (fun c ->
          float_of_int
            ((Dim.to_int (Vec6.get c Axis.H) * 4)
            + Dim.to_int (Vec6.get c Axis.W)))
    in
    let inputs = List.combine g.Graph.inputs [ x ] in
    let bind id = List.assoc id inputs in
    let grounded = Stage_program.ground prog ~bind in
    let* direct = lift_eval (Eval_direct.run g ~inputs) in
    Err.return
      (List.map
         (fun oid ->
           let d = Tensor_id.Map.find oid direct in
           let gr = Tensor_id.Map.find oid grounded in
           (d, Tensor.equal_bits d gr))
         g.Graph.outputs)
  in
  (match result with
  | Ok outs ->
      List.iter
        (fun (t, m) ->
          Format.printf "%a  ground matches direct: %b@." Tensor.pp t m)
        outs
  | Error e -> Format.printf "%a@." pp_error (Err.Error.kind e));
  [%expect
    {|
    inputs: t0
    t1 = max_pool2d_value(t0; k=2x2 s=2x2 p=0x0; out=[N,T,D,H,W,C])
    t2 = max_pool2d_index(t0; k=2x2 s=2x2 p=0x0; out=[N,T,D,H,W,C])
    outputs: t1, t2
    tensor f32 [H=2 W=2 C=1] {5, 7, 13, 15}  ground matches direct: true
    tensor f32 [H=2 W=2 C=1] {5, 7, 13, 15}  ground matches direct: true |}]

(* Reshape symbolically: the index expression carries div/mod over the flat
   offset (the value_of_index-free delinearize path); ground must match Direct. *)
