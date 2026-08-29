(* Split out of what was graph_test.ml see graph_direct_fixtures.ml. *)

open Graph_ir
open Graph_direct_fixtures

let%expect_test "Direct graph: hardsigmoid chain (add_scalar/clamp/div_scalar)"
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
          let* lo = clamp { Pointwise.Clamp.min = Some 0.; max = None } s in
          let* hi = clamp { Pointwise.Clamp.min = None; max = Some 6. } lo in
          div_scalar ~name:"out" 6. hi)
    in
    let a =
      Tensor.materialize (s1c 5) (fun c -> [| -4.; -3.; 0.; 3.; 4. |].(chan c))
    in
    let* env =
      lift_eval (Eval_direct.run g ~inputs:(List.combine g.Graph.inputs [ a ]))
    in
    tensor_of_name g env "out"
  in
  Format.printf "%a@." (pp_result (pp_named_tensor "out")) result;
  [%expect {| out = tensor f32 [C=5] {0, 0, 0.5, 1, 1} |}]

let%expect_test "Direct graph: hardtanh and clone" =
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
    let a =
      Tensor.materialize (s1c 4) (fun c -> [| -3.; 0.; 2.5; 7. |].(chan c))
    in
    let* env =
      lift_eval (Eval_direct.run g ~inputs:(List.combine g.Graph.inputs [ a ]))
    in
    tensor_of_name g env "out"
  in
  Format.printf "%a@." (pp_result (pp_named_tensor "out")) result;
  [%expect {| out = tensor f32 [C=4] {0, 0, 2.5, 6} |}]

let%expect_test "Direct graph: silu" =
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
    let a =
      Tensor.materialize (s1c 4) (fun c -> [| -6.; -0.5; 0.5; 6. |].(chan c))
    in
    let* env =
      lift_eval (Eval_direct.run g ~inputs:(List.combine g.Graph.inputs [ a ]))
    in
    tensor_of_name g env "out"
  in
  Format.printf "%a@." (pp_result (pp_named_tensor "out")) result;
  [%expect
    {| out = tensor f32 [C=4] {-0.0148357, -0.18877, 0.31123, 5.98516} |}]

let%expect_test "Direct graph: sigmoid" =
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
    let a =
      Tensor.materialize (s1c 4) (fun c -> [| -6.; -0.5; 0.5; 6. |].(chan c))
    in
    let* env =
      lift_eval (Eval_direct.run g ~inputs:(List.combine g.Graph.inputs [ a ]))
    in
    tensor_of_name g env "out"
  in
  Format.printf "%a@." (pp_result (pp_named_tensor "out")) result;
  [%expect
    {| out = tensor f32 [C=4] {0.00247262, 0.377541, 0.622459, 0.997527} |}]

let%expect_test "Direct graph: gelu" =
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
    let a =
      Tensor.materialize (s1c 4) (fun c -> [| -6.; -0.5; 0.5; 6. |].(chan c))
    in
    let* env =
      lift_eval (Eval_direct.run g ~inputs:(List.combine g.Graph.inputs [ a ]))
    in
    tensor_of_name g env "out"
  in
  Format.printf "%a@." (pp_result (pp_named_tensor "out")) result;
  [%expect {| out = tensor f32 [C=4] {-5.94073e-09, -0.154269, 0.345731, 6} |}]

let%expect_test "Direct graph: hardsigmoid" =
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
    let a =
      Tensor.materialize (s1c 4) (fun c -> [| -6.; -0.5; 0.5; 6. |].(chan c))
    in
    let* env =
      lift_eval (Eval_direct.run g ~inputs:(List.combine g.Graph.inputs [ a ]))
    in
    tensor_of_name g env "out"
  in
  Format.printf "%a@." (pp_result (pp_named_tensor "out")) result;
  [%expect {| out = tensor f32 [C=4] {0, 0.416667, 0.583333, 1} |}]

let%expect_test "Direct graph: hardswish" =
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
    let a =
      Tensor.materialize (s1c 4) (fun c -> [| -6.; -0.5; 0.5; 6. |].(chan c))
    in
    let* env =
      lift_eval (Eval_direct.run g ~inputs:(List.combine g.Graph.inputs [ a ]))
    in
    tensor_of_name g env "out"
  in
  Format.printf "%a@." (pp_result (pp_named_tensor "out")) result;
  [%expect {| out = tensor f32 [C=4] {-0, -0.208333, 0.291667, 6} |}]

(* The both-absent clamp is rejected where the node is built, so the error
   surfaces from the builder rather than from evaluation. *)
