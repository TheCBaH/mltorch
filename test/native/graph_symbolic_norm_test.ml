(* Split out of what was graph_symbolic_test.ml see graph_symbolic_fixtures.ml. *)

open Graph_ir
open Graph_symbolic_fixtures

let%expect_test "Symbolic graph: batch_norm ground matches Direct" =
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
    let prog = Eval_symbolic.run g in
    Format.printf "%a@." Stage_program.pp prog;
    let x =
      Tensor.materialize (s 1 1 1 2 1 2) (fun c ->
          [| [| 1.; 3. |]; [| 5.; 7. |] |].(chan c).(Dim.to_int
                                                       (Vec6.get c Axis.H)))
    in
    let inputs =
      List.combine g.Graph.inputs
        [ x; vec2 2. 10.; vec2 1. (-1.); vec2 1. 5.; vec2 4. 4. ]
    in
    let bind id = List.assoc id inputs in
    let grounded = Stage_program.ground prog ~bind in
    let* direct = lift_eval (Eval_direct.run g ~inputs) in
    compare_output g grounded direct
  in
  [%expect
    {|
    inputs: t0, t1, t2, t3, t4
    t5 = ((((t0[N,T,D,H,W,C] - t3[0,0,0,0,0,C]) * (1 / sqrt((t4[0,0,0,0,0,C] + 0)))) * t1[0,0,0,0,0,C]) + t2[0,0,0,0,0,C])
    outputs: t5 |}];
  Format.printf "%a@." (pp_result (pp_ground_result "ground")) result;
  [%expect
    {|
    ground = tensor f32 [H=2 W=1 C=2] {1, -1, 3, 9}
    ground matches direct: true |}]

(* max_pool2d_with_indices has two outputs and the index output uses the
   value_of_index bridge + an argmax reduction; its stage DAG is large, so we
   assert Direct==Symbolic on both outputs rather than pin the program text. *)
let%expect_test "Symbolic graph: group_norm ground matches Direct" =
  let result =
    let open Err.Syntax in
    let* g =
      lift_build
        Graph_builder.(
          build ~name:"gn" ~outputs:(fun y -> [ y ])
          @@
          let* x = input ~shape:(s 1 1 1 2 1 4) ~name:"x" () in
          group_norm ~name:"y"
            {
              Norm.GroupNorm.channel = Axis.C;
              groups = Op_config.Pos.of_int 2;
              eps = 0.;
            }
            ~x ())
    in
    let prog = Eval_symbolic.run g in
    Format.printf "%a@." Stage_program.pp prog;
    let x =
      Tensor.materialize (s 1 1 1 2 1 4) (fun c ->
          float_of_int
            ((Dim.to_int (Vec6.get c Axis.H) * 10)
            + Dim.to_int (Vec6.get c Axis.C)))
    in
    let inputs = List.combine g.Graph.inputs [ x ] in
    let bind id = List.assoc id inputs in
    let grounded = Stage_program.ground prog ~bind in
    let* direct = lift_eval (Eval_direct.run g ~inputs) in
    compare_output g grounded direct
  in
  Format.printf "%a@." (pp_result (pp_ground_result "ground")) result;
  [%expect
    {|
    inputs: t0
    t1 = ((((t0[N,T,D,H,W,C] - (sum(r1=max(0,2*floor_div(C,2))..max(0,2*floor_div(C,2))+2: sum(r2=0..1: sum(r3=0..1: sum(r4=0..2: sum(r5=0..1: t0[N,r2,r3,r4,r5,r1]))))) / 4)) * (1 / sqrt(((sum(r6=max(0,2*floor_div(C,2))..max(0,2*floor_div(C,2))+2: sum(r7=0..1: sum(r8=0..1: sum(r9=0..2: sum(r10=0..1: ((t0[N,r7,r8,r9,r10,r6] - (sum(r11=max(0,2*floor_div(C,2))..max(0,2*floor_div(C,2))+2: sum(r12=0..1: sum(r13=0..1: sum(r14=0..2: sum(r15=0..1: t0[N,r12,r13,r14,r15,r11]))))) / 4)) * (t0[N,r7,r8,r9,r10,r6] - (sum(r16=max(0,2*floor_div(C,2))..max(0,2*floor_div(C,2))+2: sum(r17=0..1: sum(r18=0..1: sum(r19=0..2: sum(r20=0..1: t0[N,r17,r18,r19,r20,r16]))))) / 4)))))))) / 4) + 0)))) * t2[0,0,0,0,0,C]) + t3[0,0,0,0,0,C])
    outputs: t1
    ground = tensor f32 [H=2 W=1 C=4] {-1.09454, -0.895533, -1.09454, -0.895533, 0.895533, 1.09454, 0.895533, 1.09454}
    ground matches direct: true |}]

(* Pad staged symbolically. Reflect is the interesting mode: its coordinate map
   is the only one in the engine built from [index_max], so this is what proves
   [Symbolic]'s [Expr.Index.max] and [Direct]'s [Stdlib.max] agree. Constant
   mode additionally stages a [select] on an [index_eq] over the summed
   displacement, which is the region test. *)
