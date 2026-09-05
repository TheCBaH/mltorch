(* Split out of what was graph_symbolic_test.ml see graph_symbolic_fixtures.ml. *)

open Graph_ir
open Graph_symbolic_fixtures

let%expect_test "Symbolic graph: pad reflect ground matches Direct" =
  let result =
    let open Err.Syntax in
    let* g =
      lift_build
        Graph_builder.(
          build ~name:"pad" ~outputs:(fun r -> [ r ])
          @@
          let* x = input ~shape:(s 1 1 1 2 3 1) ~name:"x" () in
          pad ~name:"out"
            {
              Pad.Pad.pads = [ (Axis.W, { Pad.Pad.before = 1; after = 1 }) ];
              mode = Pad.Pad.Reflect;
            }
            x)
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
    let grounded =
      Err.or_raise ~pp_error:Stage_program.pp_error
        (Stage_program.ground prog ~bind)
    in
    let* direct = lift_eval (Eval_direct.run g ~inputs) in
    compare_output g grounded direct
  in
  [%expect
    {|
    inputs: t0
    t1 = t0[N,T,D,H,max(0,min(2,2+-1*max(2+-1*max(W+-1,-1*W+-1),-1*2+-1*max(W+-1,-1*W+-1)))),C]
    outputs: t1 |}];
  Format.printf "%a@." (pp_result (pp_ground_result "ground")) result;
  [%expect
    {|
    ground = tensor f32 [H=2 W=5 C=1] {1, 0, 1, 2, 1, 11, 10, 11, ...}
    ground matches direct: true |}]

let%expect_test "Symbolic graph: pad constant with a crop ground matches Direct"
    =
  let result =
    let open Err.Syntax in
    let* g =
      lift_build
        Graph_builder.(
          build ~name:"pad" ~outputs:(fun r -> [ r ])
          @@
          let* x = input ~shape:(s 1 1 1 3 3 1) ~name:"x" () in
          pad ~name:"out"
            {
              Pad.Pad.pads =
                [
                  (Axis.H, { Pad.Pad.before = 1; after = 0 });
                  (Axis.W, { Pad.Pad.before = -1; after = 0 });
                ];
              mode = Pad.Pad.Constant 0.5;
            }
            x)
    in
    let prog = Eval_symbolic.run g in
    Format.printf "%a@." Stage_program.pp prog;
    let x =
      Tensor.materialize (s 1 1 1 3 3 1) (fun c ->
          float_of_int
            ((Dim.to_int (Vec6.get c Axis.H) * 10)
            + Dim.to_int (Vec6.get c Axis.W)))
    in
    let inputs = List.combine g.Graph.inputs [ x ] in
    let bind id = List.assoc id inputs in
    let grounded =
      Err.or_raise ~pp_error:Stage_program.pp_error
        (Stage_program.ground prog ~bind)
    in
    let* direct = lift_eval (Eval_direct.run g ~inputs) in
    compare_output g grounded direct
  in
  [%expect
    {|
    inputs: t0
    t1 = select((max(max(0,min(2,H+-1))+-1*H+-1,-1*max(0,min(2,H+-1))+-1*H+-1)+max(max(0,min(2,W+1))+-1*W+1,-1*max(0,min(2,W+1))+-1*W+1) = 0), t0[N,T,D,max(0,min(2,H+-1)),max(0,min(2,W+1)),C], 0.5)
    outputs: t1 |}];
  Format.printf "%a@." (pp_result (pp_ground_result "ground")) result;
  [%expect
    {|
    ground = tensor f32 [H=4 W=2 C=1] {0.5, 0.5, 1, 2, 11, 12, 21, 22}
    ground matches direct: true |}]

(* The staged term is the evidence here: [slice] must appear as an affine
   INDEX map on one axis and nothing else -- no [select], no bound, no arithmetic
   on the value. A [max(0,...)] wrapper is [clamp_low], the sound
   delta-to-position conversion; it is exact for these bounds and would be a
   defect only if it were [assume_index]. *)
let%expect_test "Symbolic graph: slice ground matches Direct" =
  let result =
    let open Err.Syntax in
    let* g =
      lift_build
        Graph_builder.(
          build ~name:"slice" ~outputs:(fun r -> [ r ])
          @@
          let* x = input ~shape:(s 1 1 1 2 5 1) ~name:"x" () in
          slice ~name:"out"
            {
              Split.Slice.axis = Axis.W;
              start = 1;
              stop = 5;
              step = Op_config.Pos.of_int 2;
            }
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
    let grounded =
      Err.or_raise ~pp_error:Stage_program.pp_error
        (Stage_program.ground prog ~bind)
    in
    let* direct = lift_eval (Eval_direct.run g ~inputs) in
    compare_output g grounded direct
  in
  [%expect
    {|
    inputs: t0
    t1 = t0[N,T,D,H,max(0,1+2*W),C]
    outputs: t1 |}];
  Format.printf "%a@." (pp_result (pp_ground_result "ground")) result;
  [%expect
    {|
    ground = tensor f32 [H=2 W=2 C=1] {1, 3, 11, 13}
    ground matches direct: true |}]

(* Select reuses [Slice]'s staged term over a one-wide window and folds the
   dropped axis away, so the stage should show the same affine index shape
   [Slice]'s does -- a fixed offset, no [select]/bound/arithmetic on the
   value -- confirming the delegation carries into the Symbolic path too, not
   just Direct. *)
let%expect_test "Symbolic graph: select ground matches Direct" =
  let result =
    let open Err.Syntax in
    let* g =
      lift_build
        Graph_builder.(
          build ~name:"select" ~outputs:(fun r -> [ r ])
          @@
          let* x = input ~shape:(s 1 1 1 2 5 1) ~name:"x" () in
          select ~name:"out" { Split.Select.axis = Axis.W; index = 3 } x)
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
    let grounded =
      Err.or_raise ~pp_error:Stage_program.pp_error
        (Stage_program.ground prog ~bind)
    in
    let* direct = lift_eval (Eval_direct.run g ~inputs) in
    compare_output g grounded direct
  in
  [%expect
    {|
    inputs: t0
    t1 = t0[T,D,H,W,max(0,3+0),C]
    outputs: t1 |}];
  Format.printf "%a@." (pp_result (pp_ground_result "ground")) result;
  [%expect
    {|
    ground = tensor f32 [W=2 C=1] {3, 13}
    ground matches direct: true |}]

(* Select_scatter stages an [S.index_eq]/[S.select] branch on [self]'s OWN
   coordinate against a constant [index] -- the first op in this file whose
   staged term shows a genuine equality-guarded select rather than an affine
   index alone -- confirming the branch grounds identically to Direct's. *)
let%expect_test "Symbolic graph: select_scatter ground matches Direct" =
  let result =
    let open Err.Syntax in
    let* g =
      lift_build
        Graph_builder.(
          build ~name:"select_scatter" ~outputs:(fun r -> [ r ])
          @@
          let* self = input ~shape:(s 1 1 1 2 5 1) ~name:"self" () in
          let* src = input ~shape:(s 1 1 1 1 5 1) ~name:"src" () in
          select_scatter ~name:"out"
            { Split.Select_scatter.axis = Axis.H; index = 1 }
            ~self ~src)
    in
    let prog = Eval_symbolic.run g in
    Format.printf "%a@." Stage_program.pp prog;
    let self =
      Tensor.materialize (s 1 1 1 2 5 1) (fun c ->
          float_of_int
            ((Dim.to_int (Vec6.get c Axis.H) * 10)
            + Dim.to_int (Vec6.get c Axis.W)))
    in
    let src =
      Tensor.materialize (s 1 1 1 1 5 1) (fun c ->
          float_of_int (900 + Dim.to_int (Vec6.get c Axis.W)))
    in
    let inputs = List.combine g.Graph.inputs [ self; src ] in
    let bind id = List.assoc id inputs in
    let grounded =
      Err.or_raise ~pp_error:Stage_program.pp_error
        (Stage_program.ground prog ~bind)
    in
    let* direct = lift_eval (Eval_direct.run g ~inputs) in
    compare_output g grounded direct
  in
  [%expect
    {|
    inputs: t0, t1
    t2 = select((H = 1), t1[0,N,T,D,W,C], t0[N,T,D,H,W,C])
    outputs: t2 |}];
  Format.printf "%a@." (pp_result (pp_ground_result "ground")) result;
  [%expect
    {|
    ground = tensor f32 [H=2 W=5 C=1] {0, 1, 2, 3, 4, 900, 901, 902, ...}
    ground matches direct: true |}]
