(* Split out of what was graph_symbolic_test.ml see graph_symbolic_fixtures.ml. *)

open Graph_ir
open Graph_symbolic_fixtures

let%expect_test "Symbolic graph: hardsigmoid stage DAG + ground matches Direct"
    =
  let result =
    let open Err.Syntax in
    let* g =
      lift_build
        Graph_builder.(
          build ~name:"hardsigmoid" ~outputs:(fun r -> [ r ])
          @@
          let* a = input ~shape:(s1c 5) ~name:"a" () in
          let* s = add_scalar 3. a in
          let* c = clamp { Pointwise.Clamp.min = Some 0.; max = Some 6. } s in
          div_scalar ~name:"out" 6. c)
    in
    let prog = Eval_symbolic.run g in
    Format.printf "%a@." Stage_program.pp prog;
    let a =
      Tensor.materialize (s1c 5) (fun c -> [| -4.; -3.; 0.; 3.; 4. |].(chan c))
    in
    let inputs = List.combine g.Graph.inputs [ a ] in
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
    t1 = (t0[N,T,D,H,W,C] + 3)
    t2 = select((6 < select((t1[N,T,D,H,W,C] < 0), 0, t1[N,T,D,H,W,C])), 6, select((t1[N,T,D,H,W,C] < 0), 0, t1[N,T,D,H,W,C]))
    t3 = (t2[N,T,D,H,W,C] / 6)
    outputs: t3 |}];
  Format.printf "%a@." (pp_result (pp_ground_result "ground")) result;
  [%expect
    {|
    ground = tensor f32 [C=5] {0, 0, 0.5, 1, 1}
    ground matches direct: true |}]

(* Hardtanh delegates to clamp's [Compute], so the symbolic form must be the
   same nested selects with both bounds present; clone must stage as a bare
   load. Grounding checks both against Direct. *)
let%expect_test "Symbolic graph: hardtanh/clone ground matches Direct" =
  let result =
    let open Err.Syntax in
    let* g =
      lift_build
        Graph_builder.(
          build ~name:"hardtanh_clone" ~outputs:(fun r -> [ r ])
          @@
          let* a = input ~shape:(s1c 4) ~name:"a" () in
          let* h =
            hardtanh { Pointwise.Hardtanh.min_val = 0.; max_val = 6. } a
          in
          clone ~name:"out" h)
    in
    let prog = Eval_symbolic.run g in
    Format.printf "%a@." Stage_program.pp prog;
    let a =
      Tensor.materialize (s1c 4) (fun c -> [| -3.; 0.; 2.5; 7. |].(chan c))
    in
    let inputs = List.combine g.Graph.inputs [ a ] in
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
    t1 = select((6 < select((t0[N,T,D,H,W,C] < 0), 0, t0[N,T,D,H,W,C])), 6, select((t0[N,T,D,H,W,C] < 0), 0, t0[N,T,D,H,W,C]))
    t2 = t1[N,T,D,H,W,C]
    outputs: t2 |}];
  Format.printf "%a@." (pp_result (pp_ground_result "ground")) result;
  [%expect
    {|
    ground = tensor f32 [C=4] {0, 0, 2.5, 6}
    ground matches direct: true |}]

(* The first [S.exp] consumer to travel Symbolic -> Expr -> Stage_program
   (op5-impl F4). *)
let%expect_test "Symbolic graph: silu stage DAG + ground matches Direct" =
  let result =
    let open Err.Syntax in
    let* g =
      lift_build
        Graph_builder.(
          build ~name:"silu" ~outputs:(fun r -> [ r ])
          @@
          let* a = input ~shape:(s1c 4) ~name:"a" () in
          silu ~name:"out" a)
    in
    let prog = Eval_symbolic.run g in
    Format.printf "%a@." Stage_program.pp prog;
    let a =
      Tensor.materialize (s1c 4) (fun c -> [| -6.; -0.5; 0.5; 6. |].(chan c))
    in
    let inputs = List.combine g.Graph.inputs [ a ] in
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
    t1 = (t0[N,T,D,H,W,C] / (1 + exp((0 - t0[N,T,D,H,W,C]))))
    outputs: t1 |}];
  Format.printf "%a@." (pp_result (pp_ground_result "ground")) result;
  [%expect
    {|
    ground = tensor f32 [C=4] {-0.0148357, -0.18877, 0.31123, 5.98516}
    ground matches direct: true |}]

let%expect_test "Symbolic graph: sigmoid stage DAG + ground matches Direct" =
  let result =
    let open Err.Syntax in
    let* g =
      lift_build
        Graph_builder.(
          build ~name:"sigmoid" ~outputs:(fun r -> [ r ])
          @@
          let* a = input ~shape:(s1c 4) ~name:"a" () in
          sigmoid ~name:"out" a)
    in
    let prog = Eval_symbolic.run g in
    Format.printf "%a@." Stage_program.pp prog;
    let a =
      Tensor.materialize (s1c 4) (fun c -> [| -6.; -0.5; 0.5; 6. |].(chan c))
    in
    let inputs = List.combine g.Graph.inputs [ a ] in
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
    t1 = (1 / (1 + exp((0 - t0[N,T,D,H,W,C]))))
    outputs: t1 |}];
  Format.printf "%a@." (pp_result (pp_ground_result "ground")) result;
  [%expect
    {|
    ground = tensor f32 [C=4] {0.00247262, 0.377541, 0.622459, 0.997527}
    ground matches direct: true |}]

(* The first [S.erf] consumer to travel Symbolic -> Expr -> Stage_program;
   proves the emitted expression tree actually contains an [erf] node, not
   just that the numeric result matches. *)
let%expect_test "Symbolic graph: gelu stage DAG + ground matches Direct" =
  let result =
    let open Err.Syntax in
    let* g =
      lift_build
        Graph_builder.(
          build ~name:"gelu" ~outputs:(fun r -> [ r ])
          @@
          let* a = input ~shape:(s1c 4) ~name:"a" () in
          gelu ~name:"out" Pointwise.Gelu.Exact a)
    in
    let prog = Eval_symbolic.run g in
    Format.printf "%a@." Stage_program.pp prog;
    let a =
      Tensor.materialize (s1c 4) (fun c -> [| -6.; -0.5; 0.5; 6. |].(chan c))
    in
    let inputs = List.combine g.Graph.inputs [ a ] in
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
    t1 = ((0.5 * t0[N,T,D,H,W,C]) * (1 + erf((t0[N,T,D,H,W,C] / sqrt(2)))))
    outputs: t1 |}];
  Format.printf "%a@." (pp_result (pp_ground_result "ground")) result;
  [%expect
    {|
    ground = tensor f32 [C=4] {-5.94073e-09, -0.154269, 0.345731, 6}
    ground matches direct: true |}]

let%expect_test
    "Symbolic graph: Hardsigmoid op stage DAG + ground matches Direct" =
  let result =
    let open Err.Syntax in
    let* g =
      lift_build
        Graph_builder.(
          build ~name:"hardsigmoid" ~outputs:(fun r -> [ r ])
          @@
          let* a = input ~shape:(s1c 4) ~name:"a" () in
          hardsigmoid ~name:"out" a)
    in
    let prog = Eval_symbolic.run g in
    Format.printf "%a@." Stage_program.pp prog;
    let a =
      Tensor.materialize (s1c 4) (fun c -> [| -6.; -0.5; 0.5; 6. |].(chan c))
    in
    let inputs = List.combine g.Graph.inputs [ a ] in
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
    t1 = (select((6 < select(((t0[N,T,D,H,W,C] + 3) < 0), 0, (t0[N,T,D,H,W,C] + 3))), 6, select(((t0[N,T,D,H,W,C] + 3) < 0), 0, (t0[N,T,D,H,W,C] + 3))) / 6)
    outputs: t1 |}];
  Format.printf "%a@." (pp_result (pp_ground_result "ground")) result;
  [%expect
    {|
    ground = tensor f32 [C=4] {0, 0.416667, 0.583333, 1}
    ground matches direct: true |}]

(* Hardswish: multiply precedes divide in the staged expression too, matching
   the pinned association (op5-impl F7). *)
let%expect_test "Symbolic graph: hardswish stage DAG + ground matches Direct" =
  let result =
    let open Err.Syntax in
    let* g =
      lift_build
        Graph_builder.(
          build ~name:"hardswish" ~outputs:(fun r -> [ r ])
          @@
          let* a = input ~shape:(s1c 4) ~name:"a" () in
          hardswish ~name:"out" a)
    in
    let prog = Eval_symbolic.run g in
    Format.printf "%a@." Stage_program.pp prog;
    let a =
      Tensor.materialize (s1c 4) (fun c -> [| -6.; -0.5; 0.5; 6. |].(chan c))
    in
    let inputs = List.combine g.Graph.inputs [ a ] in
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
    t1 = ((t0[N,T,D,H,W,C] * select((6 < select(((t0[N,T,D,H,W,C] + 3) < 0), 0, (t0[N,T,D,H,W,C] + 3))), 6, select(((t0[N,T,D,H,W,C] + 3) < 0), 0, (t0[N,T,D,H,W,C] + 3)))) / 6)
    outputs: t1 |}];
  Format.printf "%a@." (pp_result (pp_ground_result "ground")) result;
  [%expect
    {|
    ground = tensor f32 [C=4] {-0, -0.208333, 0.291667, 6}
    ground matches direct: true |}]

(* The three ops batch-norm folding is written in terms of, through the shared
   [Eval_op] functor: one graph exercising sub, div and sqrt at once, so the
   stage DAG shows how they compose and grounding checks all three against
   Direct. *)
