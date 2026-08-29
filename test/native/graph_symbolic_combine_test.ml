(* Split out of what was graph_symbolic_test.ml see graph_symbolic_fixtures.ml. *)

open Graph_ir
open Graph_symbolic_fixtures

let%expect_test "Symbolic graph: concat stage DAG + ground matches Direct" =
  let result =
    let open Err.Syntax in
    let* g =
      lift_build
        Graph_builder.(
          build ~name:"cat" ~outputs:(fun r -> [ r ])
          @@
          let* a = input ~shape:(s1c 2) ~name:"a" () in
          let* b = input ~shape:(s1c 3) ~name:"b" () in
          concat ~name:"out" { Concat.Concat.axis = Axis.C } [ a; b ])
    in
    let prog = Eval_symbolic.run g in
    Format.printf "%a@." Stage_program.pp prog;
    let a = Tensor.materialize (s1c 2) (fun c -> float_of_int (chan c)) in
    let b = Tensor.materialize (s1c 3) (fun c -> float_of_int (10 + chan c)) in
    let inputs = List.combine g.Graph.inputs [ a; b ] in
    let bind id = List.assoc id inputs in
    let grounded = Stage_program.ground prog ~bind in
    let* direct = lift_eval (Eval_direct.run g ~inputs) in
    compare_output g grounded direct
  in
  [%expect
    {|
    inputs: t0, t1
    t2 = select((C = max(0,min(1,C))), t0[N,T,D,H,W,max(0,min(1,C))], t1[N,T,D,H,W,max(0,min(2,C+-2))])
    outputs: t2 |}];
  Format.printf "%a@." (pp_result (pp_ground_result "ground")) result;
  [%expect
    {|
    ground = tensor f32 [C=5] {0, 1, 10, 11, 12}
    ground matches direct: true |}]

(* The non-outermost-dim case, same as the Direct compute_test.ml regression:
   [dim] relabels which axis carries each operand's real extent, so the
   staged term is the evidence that [Stack] reads its OWN axis
   ([source_coord]'s remap), not [out]'s coordinate unchanged the way a naive
   [Concat] pass-through would. *)
let%expect_test
    "Symbolic graph: stack at a non-outermost dim, ground matches Direct" =
  let result =
    let open Err.Syntax in
    let* g =
      lift_build
        Graph_builder.(
          build ~name:"stack" ~outputs:(fun r -> [ r ])
          @@
          let* a = input ~shape:(s 1 1 1 1 3 4) ~name:"a" () in
          let* b = input ~shape:(s 1 1 1 1 3 4) ~name:"b" () in
          stack ~name:"out" { Concat.Stack.axis = Axis.W } [ a; b ])
    in
    let prog = Eval_symbolic.run g in
    Format.printf "%a@." Stage_program.pp prog;
    let a =
      Tensor.materialize (s 1 1 1 1 3 4) (fun c ->
          float_of_int
            ((Dim.to_int (Vec6.get c Axis.W) * 10)
            + Dim.to_int (Vec6.get c Axis.C)))
    in
    let b =
      Tensor.materialize (s 1 1 1 1 3 4) (fun c ->
          float_of_int
            (100
            + (Dim.to_int (Vec6.get c Axis.W) * 10)
            + Dim.to_int (Vec6.get c Axis.C)))
    in
    let inputs = List.combine g.Graph.inputs [ a; b ] in
    let bind id = List.assoc id inputs in
    let grounded = Stage_program.ground prog ~bind in
    let* direct = lift_eval (Eval_direct.run g ~inputs) in
    compare_output g grounded direct
  in
  [%expect
    {|
    inputs: t0, t1
    t2 = select((W = 0), t0[0,N,T,D,H,C], t1[0,N,T,D,H,C])
    outputs: t2 |}];
  Format.printf "%a@." (pp_result (pp_ground_result "ground")) result;
  [%expect
    {|
    ground = tensor f32 [H=3 W=2 C=4] {0, 1, 2, 3, 100, 101, 102, 103, ...}
    ground matches direct: true |}]

let%expect_test "Symbolic graph: unbind emits one stage per slice" =
  let result =
    let open Err.Syntax in
    let* g =
      lift_build
        Graph_builder.(
          build ~name:"ub" ~outputs:Fun.id
          @@
          let* x = input ~shape:(s 1 1 1 2 3 1) ~name:"x" () in
          unbind { Split.Unbind.axis = Axis.W } x)
    in
    let prog = Eval_symbolic.run g in
    Format.printf "%a@." Stage_program.pp prog;
    let x =
      Tensor.materialize (s 1 1 1 2 3 1) (fun c ->
          float_of_int
            ((Dim.to_int (Vec6.get c Axis.H) * 10)
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
    t1 = t0[T,D,H,W,max(0,0),C]
    t2 = t0[T,D,H,W,max(0,1),C]
    t3 = t0[T,D,H,W,max(0,2),C]
    outputs: t1, t2, t3
    tensor f32 [W=2 C=1] {0, 10}  ground matches direct: true
    tensor f32 [W=2 C=1] {1, 11}  ground matches direct: true
    tensor f32 [W=2 C=1] {2, 12}  ground matches direct: true |}]

(* Same shape as the [unbind] test above, but the axis is KEPT and the pieces
   are different widths, so the stage count comes from [sizes]'s length and
   each stage's offset is a distinct constant rather than the ordinal
   itself. *)
let%expect_test "Symbolic graph: split_with_sizes emits one stage per piece" =
  let result =
    let open Err.Syntax in
    let* g =
      lift_build
        Graph_builder.(
          build ~name:"sws" ~outputs:Fun.id
          @@
          let* x = input ~shape:(s 1 1 1 2 5 1) ~name:"x" () in
          split_with_sizes
            { Split.Split_with_sizes.axis = Axis.W; sizes = [ 2; 3 ] }
            x)
    in
    let prog = Eval_symbolic.run g in
    Format.printf "%a@." Stage_program.pp prog;
    let x =
      Tensor.materialize (s 1 1 1 2 5 1) (fun c ->
          float_of_int
            ((Dim.to_int (Vec6.get c Axis.H) * 10)
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
    t1 = t0[N,T,D,H,max(0,W),C]
    t2 = t0[N,T,D,H,max(0,2+W),C]
    outputs: t1, t2
    tensor f32 [H=2 W=2 C=1] {0, 1, 10, 11}  ground matches direct: true
    tensor f32 [H=2 W=3 C=1] {2, 3, 4, 12, 13, 14}  ground matches direct: true |}]

(* Group norm symbolically: the one op whose reduction bounds a NESTED [S.sum]
   with a per-pixel, non-zero lower bound ([group * channels_per_group],
   [Symbolic]'s [Expr.Index] arithmetic derived from [floor_div_pos] and
   [scale] rather than a compile-time constant like every other reduction's
   [lo]) -- the [H=2 W=1 C=4] two-group fixture [compute_test.ml]'s
   hand-computed test uses, so a staging bug that only shows up once the
   bound is genuinely dynamic has a fixture to appear in. *)
