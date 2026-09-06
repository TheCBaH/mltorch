(* Split out of what was graph_symbolic_test.ml see graph_symbolic_fixtures.ml. *)

open Graph_ir
open Graph_symbolic_fixtures

let%expect_test "Symbolic graph: Discard emits no stage; ground matches Direct"
    =
  let result =
    let open Err.Syntax in
    let* g =
      lift_build
        Graph_builder.(
          build ~name:"discard" ~outputs:(fun (out, _) -> [ out ])
          @@
          let* a = input ~shape:(s1c 3) ~name:"a" () in
          let* b = input ~shape:(s1c 3) ~name:"b" () in
          let* out = add ~name:"out" a b in
          let* dead = mul ~name:"dead" a b in
          let* () = discard dead in
          return (out, dead))
    in
    let prog = Eval_symbolic.run g in
    Format.printf "%a@." Stage_program.pp prog;
    let a = Tensor.materialize (s1c 3) (fun c -> [| 1.; -5.; 2. |].(chan c)) in
    let b = Tensor.materialize (s1c 3) (fun c -> [| -3.; 1.; 2. |].(chan c)) in
    let inputs = List.combine g.Graph.inputs [ a; b ] in
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
    t2 = (t0[0,0,0,0,0,C] + t1[0,0,0,0,0,C])
    t3 = (t0[0,0,0,0,0,C] * t1[0,0,0,0,0,C])
    outputs: t2 |}];
  Format.printf "%a@." (pp_result (pp_ground_result "ground")) result;
  [%expect
    {|
    ground = tensor f32 [C=3] {-2, -4, 4}
    ground matches direct: true |}]

(* Batch norm symbolically: the per-channel affine over running stats, grounded
   against Direct. *)
let%expect_test "Symbolic graph: reshape ground matches Direct" =
  let result =
    let open Err.Syntax in
    let* g =
      lift_build
        Graph_builder.(
          build ~name:"reshape" ~outputs:(fun r -> [ r ])
          @@
          let* x = input ~shape:(s 1 1 1 2 3 1) ~name:"x" () in
          reshape ~name:"out" { Reshape.Reshape.shape = s 1 1 1 1 1 6 } x)
    in
    let prog = Eval_symbolic.run g in
    Format.printf "%a@." Stage_program.pp prog;
    let x =
      Tensor.materialize (s 1 1 1 2 3 1) (fun c ->
          float_of_int
            ((Dim.to_int (Vec6.get c Axis.H) * 3)
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
    t1 = t0[floor_div(6*N+T+D+H+W+C,6)+-1*floor_div(6*N+T+D+H+W+C,6),floor_div(6*N+T+D+H+W+C,6)+-1*floor_div(6*N+T+D+H+W+C,6),floor_div(6*N+T+D+H+W+C,6)+-1*floor_div(6*N+T+D+H+W+C,6),floor_div(6*N+T+D+H+W+C,3)+-2*floor_div(floor_div(6*N+T+D+H+W+C,3),2),6*N+T+D+H+W+C+-3*floor_div(6*N+T+D+H+W+C,3),6*N+T+D+H+W+C+-1*6*N+T+D+H+W+C]
    outputs: t1 |}];
  Format.printf "%a@." (pp_result (pp_ground_result "ground")) result;
  [%expect
    {|
    ground = tensor f32 [C=6] {0, 1, 2, 3, 4, 5}
    ground matches direct: true |}]

let%expect_test "Symbolic graph: repeat ground matches Direct" =
  let result =
    let open Err.Syntax in
    let* g =
      lift_build
        Graph_builder.(
          build ~name:"repeat" ~outputs:(fun r -> [ r ])
          @@
          let* x = input ~shape:(s1c 2) ~name:"x" () in
          repeat ~name:"out" { Repeat.Repeat.repeats = s1c 3 } x)
    in
    let prog = Eval_symbolic.run g in
    Format.printf "%a@." Stage_program.pp prog;
    let x = Tensor.materialize (s1c 2) (fun c -> float_of_int (chan c)) in
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
    t1 = t0[N+-1*N,T+-1*T,D+-1*D,H+-1*H,W+-1*W,C+-2*floor_div(C,2)]
    outputs: t1 |}];
  Format.printf "%a@." (pp_result (pp_ground_result "ground")) result;
  [%expect
    {|
    ground = tensor f32 [C=6] {0, 1, 0, 1, 0, 1}
    ground matches direct: true |}]

let%expect_test "Symbolic graph: repeat_interleave ground matches Direct" =
  let result =
    let open Err.Syntax in
    let* g =
      lift_build
        Graph_builder.(
          build ~name:"repeat_interleave" ~outputs:(fun r -> [ r ])
          @@
          let* x = input ~shape:(s1c 2) ~name:"x" () in
          repeat_interleave ~name:"out"
            {
              Repeat.RepeatInterleave.axis = Axis.C;
              repeats = Op_config.Pos.of_int 3;
            }
            x)
    in
    let prog = Eval_symbolic.run g in
    Format.printf "%a@." Stage_program.pp prog;
    let x = Tensor.materialize (s1c 2) (fun c -> float_of_int (chan c)) in
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
    t1 = t0[N,T,D,H,W,floor_div(C,3)]
    outputs: t1 |}];
  Format.printf "%a@." (pp_result (pp_ground_result "ground")) result;
  [%expect
    {|
    ground = tensor f32 [C=6] {0, 0, 0, 1, 1, 1}
    ground matches direct: true |}]

let%expect_test "Symbolic graph: unfold ground matches Direct" =
  let result =
    let open Err.Syntax in
    let* g =
      lift_build
        Graph_builder.(
          build ~name:"unfold" ~outputs:(fun r -> [ r ])
          @@
          let* x = input ~shape:(s1c 8) ~name:"x" () in
          unfold ~name:"out"
            {
              Unfold.Unfold.axis = Axis.W;
              size = Dim.extent 3;
              step = Op_config.Pos.of_int 2;
            }
            x)
    in
    let prog = Eval_symbolic.run g in
    Format.printf "%a@." Stage_program.pp prog;
    let x = Tensor.materialize (s1c 8) (fun c -> float_of_int (chan c)) in
    let inputs = List.combine g.Graph.inputs [ x ] in
    let bind id = List.assoc id inputs in
    let grounded =
      Err.or_raise ~pp_error:Stage_program.pp_error
        (Stage_program.ground prog ~bind)
    in
    let* direct = lift_eval (Eval_direct.run g ~inputs) in
    compare_output g grounded direct
  in
  [%expect {|
    inputs: t0
    t1 = t0[0,N,T,D,H,2*W+C]
    outputs: t1 |}];
  Format.printf "%a@." (pp_result (pp_ground_result "ground")) result;
  [%expect
    {|
    ground = tensor f32 [W=3 C=3] {0, 1, 2, 2, 3, 4, 4, 5, ...}
    ground matches direct: true |}]

(* Conv decomposition symbolically: the conv stage loads the permute stage's
   signature, and the final permute loads the conv stage's — i.e. symbolic
   execution extends through the whole graph by tensor signature. *)
