(* Split out of what was graph_symbolic_test.ml see graph_symbolic_fixtures.ml. *)

open Graph_ir
open Graph_symbolic_fixtures

let%expect_test "Symbolic graph: add -> relu stage DAG + ground matches Direct"
    =
  let result =
    let open Err.Syntax in
    let* g =
      lift_build
        Graph_builder.(
          build ~name:"seq" ~outputs:(fun r -> [ r ])
          @@
          let* a = input ~shape:(s1c 3) ~name:"a" () in
          let* b = input ~shape:(s1c 3) ~name:"b" () in
          let* t = add ~name:"sum" a b in
          relu ~name:"out" t)
    in
    let prog = Eval_symbolic.run g in
    Format.printf "%a@." Stage_program.pp prog;
    let a = Tensor.materialize (s1c 3) (fun c -> [| 1.; -5.; 2. |].(chan c)) in
    let b = Tensor.materialize (s1c 3) (fun c -> [| -3.; 1.; 2. |].(chan c)) in
    let inputs = List.combine g.Graph.inputs [ a; b ] in
    let bind id = List.assoc id inputs in
    let grounded = Stage_program.ground prog ~bind in
    let* direct = lift_eval (Eval_direct.run g ~inputs) in
    compare_output g grounded direct
  in
  [%expect
    {|
    inputs: t0, t1
    t2 = (t0[0,0,0,0,0,C] + t1[0,0,0,0,0,C])
    t3 = select((t2[N,T,D,H,W,C] < 0), 0, t2[N,T,D,H,W,C])
    outputs: t3 |}];
  Format.printf "%a@." (pp_result (pp_ground_result "ground")) result;
  [%expect
    {|
    ground = tensor f32 [C=3] {0, 0, 4}
    ground matches direct: true |}]

let%expect_test "Symbolic graph: mul stage DAG + ground matches Direct" =
  let result =
    let open Err.Syntax in
    let* g =
      lift_build
        Graph_builder.(
          build ~name:"prod" ~outputs:(fun r -> [ r ])
          @@
          let* a = input ~shape:(s1c 3) ~name:"a" () in
          let* b = input ~shape:(s1c 3) ~name:"b" () in
          mul ~name:"out" a b)
    in
    let prog = Eval_symbolic.run g in
    Format.printf "%a@." Stage_program.pp prog;
    let a = Tensor.materialize (s1c 3) (fun c -> [| 1.; -5.; 2. |].(chan c)) in
    let b = Tensor.materialize (s1c 3) (fun c -> [| -3.; 1.; 2. |].(chan c)) in
    let inputs = List.combine g.Graph.inputs [ a; b ] in
    let bind id = List.assoc id inputs in
    let grounded = Stage_program.ground prog ~bind in
    let* direct = lift_eval (Eval_direct.run g ~inputs) in
    compare_output g grounded direct
  in
  [%expect
    {|
    inputs: t0, t1
    t2 = (t0[0,0,0,0,0,C] * t1[0,0,0,0,0,C])
    outputs: t2 |}];
  Format.printf "%a@." (pp_result (pp_ground_result "ground")) result;
  [%expect
    {|
    ground = tensor f32 [C=3] {-3, -5, 4}
    ground matches direct: true |}]

(* MobileNet-v3's hardsigmoid, through the same shared [Eval_op] functor. The
   stage DAG is what shows the scalars really are [const] leaves in the symbolic
   expression rather than loads of a materialised operand, and that clamp lowers
   to nested selects. *)
let%expect_test "Symbolic graph: mul_scalar stage DAG + ground matches Direct" =
  let result =
    let open Err.Syntax in
    let* g =
      lift_build
        Graph_builder.(
          build ~name:"mul_scalar" ~outputs:(fun r -> [ r ])
          @@
          let* a = input ~shape:(s1c 3) ~name:"a" () in
          mul_scalar ~name:"out" 3. a)
    in
    let prog = Eval_symbolic.run g in
    Format.printf "%a@." Stage_program.pp prog;
    let a = Tensor.materialize (s1c 3) (fun c -> float_of_int (chan c) +. 1.) in
    let inputs = List.combine g.Graph.inputs [ a ] in
    let bind id = List.assoc id inputs in
    let grounded = Stage_program.ground prog ~bind in
    let* direct = lift_eval (Eval_direct.run g ~inputs) in
    compare_output g grounded direct
  in
  [%expect {|
    inputs: t0
    t1 = (t0[N,T,D,H,W,C] * 3)
    outputs: t1 |}];
  Format.printf "%a@." (pp_result (pp_ground_result "ground")) result;
  [%expect
    {|
    ground = tensor f32 [C=3] {3, 6, 9}
    ground matches direct: true |}]

let%expect_test "Symbolic graph: rsub_scalar stage DAG + ground matches Direct"
    =
  let result =
    let open Err.Syntax in
    let* g =
      lift_build
        Graph_builder.(
          build ~name:"rsub_scalar" ~outputs:(fun r -> [ r ])
          @@
          let* a = input ~shape:(s1c 3) ~name:"a" () in
          rsub_scalar ~name:"out"
            { Pointwise.Rsub_scalar.other = 1.; alpha = 1. }
            a)
    in
    let prog = Eval_symbolic.run g in
    Format.printf "%a@." Stage_program.pp prog;
    let a = Tensor.materialize (s1c 3) (fun c -> float_of_int (chan c)) in
    let inputs = List.combine g.Graph.inputs [ a ] in
    let bind id = List.assoc id inputs in
    let grounded = Stage_program.ground prog ~bind in
    let* direct = lift_eval (Eval_direct.run g ~inputs) in
    compare_output g grounded direct
  in
  [%expect
    {|
    inputs: t0
    t1 = (1 - (1 * t0[N,T,D,H,W,C]))
    outputs: t1 |}];
  Format.printf "%a@." (pp_result (pp_ground_result "ground")) result;
  [%expect
    {|
    ground = tensor f32 [C=3] {1, 0, -1}
    ground matches direct: true |}]

(* [a]'s H axis is 1; [size]'s is 2 -- so the staged read must show [H] pinned
   to the constant 0 ([broadcast_coord]'s substitution) even though the OTHER
   axes read the live loop coordinate, which is what proves the broadcast is
   really happening rather than merely printing trivially for a size-1 axis
   (as every other axis here does anyway, since [s1c] itself is all-1 outside
   C). *)
let%expect_test "Symbolic graph: expand stage DAG + ground matches Direct" =
  let result =
    let open Err.Syntax in
    let* g =
      lift_build
        Graph_builder.(
          build ~name:"expand" ~outputs:(fun r -> [ r ])
          @@
          let* a = input ~shape:(s1c 3) ~name:"a" () in
          expand ~name:"out"
            { Pointwise.Expand.size = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:2 ~w:1 ~c:3 }
            a)
    in
    let prog = Eval_symbolic.run g in
    Format.printf "%a@." Stage_program.pp prog;
    let a = Tensor.materialize (s1c 3) (fun c -> float_of_int (chan c)) in
    let inputs = List.combine g.Graph.inputs [ a ] in
    let bind id = List.assoc id inputs in
    let grounded = Stage_program.ground prog ~bind in
    let* direct = lift_eval (Eval_direct.run g ~inputs) in
    compare_output g grounded direct
  in
  [%expect {|
    inputs: t0
    t1 = t0[0,0,0,0,0,C]
    outputs: t1 |}];
  Format.printf "%a@." (pp_result (pp_ground_result "ground")) result;
  [%expect
    {|
    ground = tensor f32 [H=2 W=1 C=3] {0, 1, 2, 0, 1, 2}
    ground matches direct: true |}]

(* The retained Group 5 [Hardsigmoid] op (one node), not the decomposed
   add_scalar/clamp/div_scalar graph the test above builds by hand: it reuses
   [Clamp.apply] on the value [x + 3], not a load, so the staged form must show
   that same nested-select clamp, applied to an addition rather than a bare
   load. *)
let%expect_test "Symbolic graph: sub/div/sqrt stage DAG + ground matches Direct"
    =
  let result =
    let open Err.Syntax in
    let* g =
      lift_build
        Graph_builder.(
          build ~name:"scale" ~outputs:(fun r -> [ r ])
          @@
          let* a = input ~shape:(s1c 3) ~name:"a" () in
          let* b = input ~shape:(s1c 3) ~name:"b" () in
          let* d = sub ~name:"diff" a b in
          let* r = sqrt ~name:"root" b in
          div ~name:"out" d r)
    in
    let prog = Eval_symbolic.run g in
    Format.printf "%a@." Stage_program.pp prog;
    let a = Tensor.materialize (s1c 3) (fun c -> [| 5.; 1.; 8. |].(chan c)) in
    let b = Tensor.materialize (s1c 3) (fun c -> [| 4.; 1.; 16. |].(chan c)) in
    let inputs = List.combine g.Graph.inputs [ a; b ] in
    let bind id = List.assoc id inputs in
    let grounded = Stage_program.ground prog ~bind in
    let* direct = lift_eval (Eval_direct.run g ~inputs) in
    compare_output g grounded direct
  in
  [%expect
    {|
    inputs: t0, t1
    t2 = (t0[0,0,0,0,0,C] - t1[0,0,0,0,0,C])
    t3 = sqrt(t1[N,T,D,H,W,C])
    t4 = (t2[0,0,0,0,0,C] / t3[0,0,0,0,0,C])
    outputs: t4 |}];
  Format.printf "%a@." (pp_result (pp_ground_result "ground")) result;
  [%expect
    {|
    ground = tensor f32 [C=3] {0.5, 0, -2}
    ground matches direct: true |}]

(* A [Discard] node emits NO stage (its producer "dead" still does), so the
   stage DAG lists sum + dead but nothing for the sink; grounding the kept
   output still matches Direct. *)
