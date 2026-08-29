(* Split out of what was graph_test.ml see graph_direct_fixtures.ml. *)

open Graph_ir
open Graph_direct_fixtures

let%expect_test "Direct graph: concat of two inputs" =
  let result =
    let open Err.Syntax in
    let* g =
      lift_build
        Graph_builder.(
          build ~name:"concat" ~outputs:(fun r -> [ r ])
          @@
          let* a = input ~shape:(s1c 2) ~name:"a" () in
          let* b = input ~shape:(s1c 3) ~name:"b" () in
          concat ~name:"out" { Concat.Concat.axis = Axis.C } [ a; b ])
    in
    let a = Tensor.materialize (s1c 2) (fun c -> float_of_int (chan c)) in
    let b = Tensor.materialize (s1c 3) (fun c -> float_of_int (10 + chan c)) in
    let* env =
      lift_eval
        (Eval_direct.run g ~inputs:(List.combine g.Graph.inputs [ a; b ]))
    in
    tensor_of_name g env "out"
  in
  Format.printf "%a@." (pp_result (pp_named_tensor "out")) result;
  [%expect {| out = tensor f32 [C=5] {0, 1, 10, 11, 12} |}]

(* Stack inserts a new axis per operand, unlike Concat above which joins an
   existing one -- the fact worth pinning here is that a single [Stack] node
   appears, not N [Reshape]s plus a [Concat]. *)
let%expect_test "Direct graph: stack of two inputs" =
  let result =
    let open Err.Syntax in
    let* g =
      lift_build
        Graph_builder.(
          build ~name:"stack" ~outputs:(fun r -> [ r ])
          @@
          let* a = input ~shape:(s1c 3) ~name:"a" () in
          let* b = input ~shape:(s1c 3) ~name:"b" () in
          stack ~name:"out" { Concat.Stack.axis = Axis.W } [ a; b ])
    in
    Format.printf "%a@." Graph_ir.pp g;
    let a = Tensor.materialize (s1c 3) (fun c -> float_of_int (chan c)) in
    let b = Tensor.materialize (s1c 3) (fun c -> float_of_int (10 + chan c)) in
    let* env =
      lift_eval
        (Eval_direct.run g ~inputs:(List.combine g.Graph.inputs [ a; b ]))
    in
    tensor_of_name g env "out"
  in
  Format.printf "%a@." (pp_result (pp_named_tensor "out")) result;
  [%expect
    {|
    graph
    inputs: [t0 f32 [C=3] ->[n0], t1 f32 [C=3] ->[n0]]
    nodes:
      n0: [t2 f32 [W=2 C=3]] = stack xs=[t0, t1] params={axis=W}
    outputs: [t2 f32 [W=2 C=3] <-n0]
    out = tensor f32 [W=2 C=3] {0, 1, 2, 10, 11, 12} |}]

let%expect_test "Direct graph: unbind is a variable-arity node" =
  let result =
    let open Err.Syntax in
    let* g =
      lift_build
        Graph_builder.(
          build ~name:"ub" ~outputs:Fun.id
          @@
          let* x = input ~shape:(s 1 1 1 2 3 1) ~name:"x" () in
          unbind ~name:"slice" { Split.Unbind.axis = Axis.W } x)
    in
    let x =
      Tensor.materialize (s 1 1 1 2 3 1) (fun c ->
          float_of_int
            ((Dim.to_int (Vec6.get c Axis.H) * 10)
            + Dim.to_int (Vec6.get c Axis.W)))
    in
    let* env =
      lift_eval (Eval_direct.run g ~inputs:(List.combine g.Graph.inputs [ x ]))
    in
    Err.return
      (List.map (fun oid -> Tensor_id.Map.find oid env) g.Graph.outputs)
  in
  (match result with
  | Ok outs ->
      List.iteri (fun i t -> Format.printf "out%d = %a@." i Tensor.pp t) outs
  | Error e -> Format.printf "%a@." pp_error (Err.Error.kind e));
  [%expect
    {|
    out0 = tensor f32 [W=2 C=1] {0, 10}
    out1 = tensor f32 [W=2 C=1] {1, 11}
    out2 = tensor f32 [W=2 C=1] {2, 12} |}]

(* Same variable-arity path as unbind above, except the arity comes from
   [params.sizes]'s length rather than the input's own extent, and every
   output KEEPS the split axis instead of dropping it. *)
let%expect_test "Direct graph: split_with_sizes divides one axis into windows" =
  let result =
    let open Err.Syntax in
    let* g =
      lift_build
        Graph_builder.(
          build ~name:"sws" ~outputs:Fun.id
          @@
          let* x = input ~shape:(s 1 1 1 2 5 1) ~name:"x" () in
          split_with_sizes ~name:"pieces"
            { Split.Split_with_sizes.axis = Axis.W; sizes = [ 2; 3 ] }
            x)
    in
    let x =
      Tensor.materialize (s 1 1 1 2 5 1) (fun c ->
          float_of_int
            ((Dim.to_int (Vec6.get c Axis.H) * 10)
            + Dim.to_int (Vec6.get c Axis.W)))
    in
    let* env =
      lift_eval (Eval_direct.run g ~inputs:(List.combine g.Graph.inputs [ x ]))
    in
    Err.return
      (List.map (fun oid -> Tensor_id.Map.find oid env) g.Graph.outputs)
  in
  (match result with
  | Ok outs ->
      List.iteri (fun i t -> Format.printf "out%d = %a@." i Tensor.pp t) outs
  | Error e -> Format.printf "%a@." pp_error (Err.Error.kind e));
  [%expect
    {|
    out0 = tensor f32 [H=2 W=2 C=1] {0, 1, 10, 11}
    out1 = tensor f32 [H=2 W=3 C=1] {2, 3, 4, 12, 13, 14} |}]

(* Through the builder with BOTH affine operands absent, so this exercises
   [Eval_op]'s fill path (identity weight=1, bias=0) end to end -- the same
   [H=2 W=1 C=4] two-group fixture [compute_test.ml]'s hand-computed test
   uses, so the two can be cross-checked against each other. *)
