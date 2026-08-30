(* Split out of what was graph_test.ml see graph_direct_fixtures.ml. *)

open Graph_ir
open Graph_direct_fixtures

let%expect_test "Direct graph: batch_norm per-channel over C" =
  let result =
    let open Err.Syntax in
    let vec2 a0 a1 =
      Tensor.materialize (s1c 2) (fun c -> [| a0; a1 |].(chan c))
    in
    let* g =
      lift_build
        Graph_builder.(
          build ~name:"bn" ~outputs:(fun y -> [ y ])
          @@
          let* x = input ~shape:(s 1 1 1 2 1 2) ~name:"x" () in
          let* w = input ~shape:(s1c 2) ~name:"w" () in
          let* b = input ~shape:(s1c 2) ~name:"b" () in
          let* rm = input ~shape:(s1c 2) ~name:"rm" () in
          let* rv = input ~shape:(s1c 2) ~name:"rv" () in
          batch_norm ~name:"y"
            { Norm.BatchNorm.channel = Axis.C; eps = 0. }
            ~x ~weight:w ~bias:b ~running_mean:rm ~running_var:rv ())
    in
    let x =
      Tensor.materialize (s 1 1 1 2 1 2) (fun c ->
          [| [| 1.; 3. |]; [| 5.; 7. |] |].(chan c).(Dim.to_int
                                                       (Vec6.get c Axis.H)))
    in
    let inputs =
      List.combine g.Graph.inputs
        [ x; vec2 2. 10.; vec2 1. (-1.); vec2 1. 5.; vec2 4. 4. ]
    in
    let* env = lift_eval (Eval_direct.run g ~inputs) in
    tensor_of_name g env "y"
  in
  Format.printf "%a@." (pp_result (pp_named_tensor "y")) result;
  [%expect {| y = tensor f32 [H=2 W=1 C=2] {1, -1, 3, 9} |}]

let%expect_test "Direct graph: batch_norm_no_stats preserves all three outputs"
    =
  let vec2 a0 a1 =
    Tensor.materialize (s1c 2) (fun c -> [| a0; a1 |].(chan c))
  in
  let g =
    Err.or_raise ~pp_error:Graph_builder.pp_error
      Graph_builder.(
        build ~name:"bn_no_stats" ~outputs:Fun.id
        @@
        let* x = input ~shape:(s 1 1 1 2 1 2) () in
        let* w = input ~shape:(s1c 2) () in
        let* b = input ~shape:(s1c 2) () in
        batch_norm_no_stats
          { Norm.BatchNormNoStats.channel = Axis.C; eps = 0. }
          ~x ~weight:w ~bias:b ())
  in
  let x =
    Tensor.materialize (s 1 1 1 2 1 2) (fun c ->
        [| [| 1.; 3. |]; [| 5.; 7. |] |].(chan c).(Dim.to_int
                                                     (Vec6.get c Axis.H)))
  in
  let env =
    Eval_direct.run g
      ~inputs:(List.combine g.Graph.inputs [ x; vec2 2. 10.; vec2 1. (-1.) ])
    |> Err.or_raise ~pp_error:Eval_direct.pp_error
  in
  List.iter
    (fun id -> Format.printf "%a@." Tensor.pp (Tensor_id.Map.find id env))
    g.Graph.outputs;
  [%expect
    {|
    tensor f32 [H=2 W=1 C=2] {-1, -11, 3, 9}
    tensor f32 [C=2] {2, 6}
    tensor f32 [C=2] {1, 1} |}]

(* max_pool2d_with_indices produces TWO outputs — exercising the multi-output
   eval loop. value(h,w)=h*4+w; each 2x2/stride-2 window's max is its
   bottom-right corner, and the argmax index is that corner's flat position. *)
let%expect_test "Direct graph: group_norm splits C into groups, absent affine" =
  let result =
    let open Err.Syntax in
    let* g =
      lift_build
        Graph_builder.(
          build ~name:"gn" ~outputs:(fun r -> [ r ])
          @@
          let* x = input ~shape:(s 1 1 1 2 1 4) ~name:"x" () in
          group_norm ~name:"out"
            {
              Norm.GroupNorm.channel = Axis.C;
              groups = Op_config.Pos.of_int 2;
              eps = 0.;
            }
            ~x ())
    in
    let x =
      Tensor.materialize (s 1 1 1 2 1 4) (fun c ->
          float_of_int ((Dim.to_int (Vec6.get c Axis.H) * 10) + chan c))
    in
    let* env =
      lift_eval (Eval_direct.run g ~inputs:(List.combine g.Graph.inputs [ x ]))
    in
    tensor_of_name g env "out"
  in
  Format.printf "%a@." (pp_result (pp_named_tensor "out")) result;
  [%expect
    {|
    out = tensor f32 [H=2 W=1 C=4] {-1.09454, -0.895533, -1.09454, -0.895533, 0.895533, 1.09454, 0.895533, 1.09454} |}]

(* Pad through the builder: the fill is narrowed to f32 there, and the shape
   rule sees the input edge, so a graph whose pad empties an axis cannot be
   built at all. *)
