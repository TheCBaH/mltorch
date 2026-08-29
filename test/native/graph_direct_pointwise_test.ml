(* Split out of what was graph_test.ml see graph_direct_fixtures.ml. *)

open Graph_ir
open Graph_direct_fixtures

let%expect_test "Direct graph: add of two inputs" =
  let result =
    let open Err.Syntax in
    let* g =
      lift_build
        Graph_builder.(
          build ~name:"add" ~outputs:(fun r -> [ r ])
          @@
          let* a = input ~shape:(s1c 3) ~name:"a" () in
          let* b = input ~shape:(s1c 3) ~name:"b" () in
          add ~name:"out" a b)
    in
    let a = Tensor.materialize (s1c 3) (fun c -> float_of_int (chan c)) in
    let b = Tensor.materialize (s1c 3) (fun _ -> 10.) in
    let* env =
      lift_eval
        (Eval_direct.run g ~inputs:(List.combine g.Graph.inputs [ a; b ]))
    in
    tensor_of_name g env "out"
  in
  Format.printf "%a@." (pp_result (pp_named_tensor "out")) result;
  [%expect {| out = tensor f32 [C=3] {10, 11, 12} |}]

let%expect_test "Direct graph: div of two inputs" =
  let result =
    let open Err.Syntax in
    let* g =
      lift_build
        Graph_builder.(
          build ~name:"div" ~outputs:(fun r -> [ r ])
          @@
          let* a = input ~shape:(s1c 3) ~name:"a" () in
          let* b = input ~shape:(s1c 3) ~name:"b" () in
          div ~name:"out" a b)
    in
    let a = Tensor.materialize (s1c 3) (fun c -> float_of_int (chan c) +. 1.) in
    let b = Tensor.materialize (s1c 3) (fun _ -> 4.) in
    let* env =
      lift_eval
        (Eval_direct.run g ~inputs:(List.combine g.Graph.inputs [ a; b ]))
    in
    tensor_of_name g env "out"
  in
  Format.printf "%a@." (pp_result (pp_named_tensor "out")) result;
  [%expect {| out = tensor f32 [C=3] {0.25, 0.5, 0.75} |}]

let%expect_test "Direct graph: mul of two inputs" =
  let result =
    let open Err.Syntax in
    let* g =
      lift_build
        Graph_builder.(
          build ~name:"mul" ~outputs:(fun r -> [ r ])
          @@
          let* a = input ~shape:(s1c 3) ~name:"a" () in
          let* b = input ~shape:(s1c 3) ~name:"b" () in
          mul ~name:"out" a b)
    in
    let a = Tensor.materialize (s1c 3) (fun c -> float_of_int (chan c)) in
    let b = Tensor.materialize (s1c 3) (fun _ -> 10.) in
    let* env =
      lift_eval
        (Eval_direct.run g ~inputs:(List.combine g.Graph.inputs [ a; b ]))
    in
    tensor_of_name g env "out"
  in
  Format.printf "%a@." (pp_result (pp_named_tensor "out")) result;
  [%expect {| out = tensor f32 [C=3] {0, 10, 20} |}]

let%expect_test "Direct graph: sqrt of an input" =
  let result =
    let open Err.Syntax in
    let* g =
      lift_build
        Graph_builder.(
          build ~name:"sqrt" ~outputs:(fun r -> [ r ])
          @@
          let* a = input ~shape:(s1c 4) ~name:"a" () in
          sqrt ~name:"out" a)
    in
    let a =
      Tensor.materialize (s1c 4) (fun c -> [| 0.; 1.; 4.; 2.25 |].(chan c))
    in
    let* env =
      lift_eval (Eval_direct.run g ~inputs:(List.combine g.Graph.inputs [ a ]))
    in
    tensor_of_name g env "out"
  in
  Format.printf "%a@." (pp_result (pp_named_tensor "out")) result;
  [%expect {| out = tensor f32 [C=4] {0, 1, 2, 1.5} |}]

let%expect_test "Direct graph: sub of two inputs" =
  let result =
    let open Err.Syntax in
    let* g =
      lift_build
        Graph_builder.(
          build ~name:"sub" ~outputs:(fun r -> [ r ])
          @@
          let* a = input ~shape:(s1c 3) ~name:"a" () in
          let* b = input ~shape:(s1c 3) ~name:"b" () in
          sub ~name:"out" a b)
    in
    let a = Tensor.materialize (s1c 3) (fun c -> float_of_int (chan c)) in
    let b = Tensor.materialize (s1c 3) (fun _ -> 10.) in
    let* env =
      lift_eval
        (Eval_direct.run g ~inputs:(List.combine g.Graph.inputs [ a; b ]))
    in
    tensor_of_name g env "out"
  in
  Format.printf "%a@." (pp_result (pp_named_tensor "out")) result;
  [%expect {| out = tensor f32 [C=3] {-10, -9, -8} |}]

(* MobileNet-v3's hardsigmoid, as the exporter serialises it: the +3 and /6 are
   compile-time scalars in Tensor slots, so they lower to the scalar-parameter
   ops rather than to edges that would need binding. *)
let%expect_test "Direct graph: mul_scalar" =
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
    let a = Tensor.materialize (s1c 3) (fun c -> float_of_int (chan c) +. 1.) in
    let* env =
      lift_eval (Eval_direct.run g ~inputs:(List.combine g.Graph.inputs [ a ]))
    in
    tensor_of_name g env "out"
  in
  Format.printf "%a@." (pp_result (pp_named_tensor "out")) result;
  [%expect {| out = tensor f32 [C=3] {3, 6, 9} |}]

let%expect_test "Direct graph: clamp with no bounds fails to build" =
  let result =
    let open Err.Syntax in
    let+ g =
      lift_build
        Graph_builder.(
          build ~name:"clamp" ~outputs:(fun r -> [ r ])
          @@
          let* a = input ~shape:(s1c 3) ~name:"a" () in
          clamp { Pointwise.Clamp.min = None; max = None } a)
    in
    Format.asprintf "%a" Graph_ir.pp g
  in
  Format.printf "%a@." (pp_result Format.pp_print_string) result;
  [%expect {| clamp: at least one of 'min' or 'max' must be given |}]

(* A [Discard] sink consumes a dead edge and produces no output. The graph still
   evaluates, the discarded producer ("dead") is still materialised in the env
   (so tools can inspect it), and the node prints with an empty output list. *)
